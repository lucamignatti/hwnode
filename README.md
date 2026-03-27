# HW-NODE: Weight-Shared Hammerstein-Wiener Neural ODE

`hwnode` is a PyTorch research codebase for studying whether a structured, weight-shared Hammerstein-Wiener latent-flow block can serve as a parameter-efficient alternative to standard feedforward backbones.

The main idea is simple: instead of allocating a fresh parameter set at every depth step, HW-NODE repeatedly applies the same latent dynamical operator. That does not make depth free in compute, but it does decouple virtual depth from parameter growth. The research question is whether this structure preserves enough expressivity to stay competitive under tight parameter budgets.

This repository currently contains:
- a core shared HW-NODE block in [`hwnode/model.py`](/Users/lucamignatti/Projects/hwnode/hwnode/model.py)
- PPO training and sweep code for Gymnasium control tasks in [`hwnode/run.py`](/Users/lucamignatti/Projects/hwnode/hwnode/run.py) and [`hwnode/train.py`](/Users/lucamignatti/Projects/hwnode/hwnode/train.py)
- an MLP baseline in [`hwnode/baseline.py`](/Users/lucamignatti/Projects/hwnode/hwnode/baseline.py)
- experimental Taylor and Chebyshev variants in [`experiments/taylor_vs_chebyshev.py`](/Users/lucamignatti/Projects/hwnode/experiments/taylor_vs_chebyshev.py)

## Theory

HW-NODE is motivated by classical Hammerstein-Wiener structure: a static input nonlinearity, a linear dynamical core, and a static output nonlinearity. In this repository, that structure is used inside a repeated latent-flow block.

Let `h_l ∈ R^d` be the working representation at virtual depth `l`, and let `z_l ∈ R^n` be a lower-dimensional latent state with `n << d`. The idealized update is

```text
h_0 = x
z_l(0) = phi(W_in h_l)
z_l(Δt) = exp(A Δt) z_l(0)
u_l = psi(W_out z_l(Δt))
h_{l+1} = h_l + u_l
```

where the same parameters `(W_in, A, Δt, W_out)` are reused for `l = 0, ..., T-1`.

This gives three linked design choices:

1. Hammerstein compression.
   `W_in` maps the model state into a lower-dimensional latent state, after which `phi` introduces the input-side nonlinearity.

2. Linear latent dynamics.
   The latent state evolves under `dz/dt = A z`, whose exact solution over one step is `exp(A Δt) z`.

3. Wiener expansion.
   `W_out` projects the latent state back to model space, and `psi` provides the output-side nonlinearity before the residual update.

The point of the architecture is not to imitate an arbitrary deep MLP exactly. It imposes a strong inductive bias: repeated nonlinear projections through a shared latent dynamical system. That bias trades raw flexibility for parameter efficiency and structural regularity.

### Implemented Approximation

The code in [`hwnode/model.py`](/Users/lucamignatti/Projects/hwnode/hwnode/model.py) implements this idea with a few concrete approximations:

- The matrix exponential is approximated by a truncated Taylor polynomial

```text
P_K(M) = sum_{k=0}^K M^k / k!,  where M = A_hat Δt
```

- `A_hat` is formed by dividing `A` by a one-step power-iteration estimate of its top singular value.
- The default input nonlinearity is `LeakyReLU`.
- The default output nonlinearity is `LeakyReLU` followed by squaring when `activation="relu_squared"`.
- `SharedHWNODE` keeps one shared parameter set across virtual layers, so parameter count does not increase with `num_virtual_layers`, although compute does.

For PPO, [`HWNodeNetwork`](/Users/lucamignatti/Projects/hwnode/hwnode/model.py) uses the shared block in residual mode, wraps it with an input embedding and output normalization, and compares it against a residual MLP baseline with a matched interface.

### Experimental Variants

The `experiments/` directory extends the core idea rather than replacing it:

- [`experiments/taylor_vs_chebyshev.py`](/Users/lucamignatti/Projects/hwnode/experiments/taylor_vs_chebyshev.py) adds a learned Taylor-basis variant and a Chebyshev-basis variant for the latent flow operator.
- [`experiments/eval_suite.py`](/Users/lucamignatti/Projects/hwnode/experiments/eval_suite.py) runs mixed comparisons among baseline MLPs, the shared HW-NODE backbone, and those experimental variants.

These variants test a second research question: whether the choice of polynomial basis for the latent operator matters once the model is constrained to a small shared dynamical core.

## Results

### Reinforcement Learning (PPO)

The RL experiments compare HW-NODE against MLP baselines on `LunarLander-v3`. The figures below are the reported results from the project’s evaluation runs and correspond to the experimental setup used in the original README.

| Architecture | Parameters | Mean Final Reward ± Std |
|:--|--:|--:|
| `hwnode-standard` | 6,343 | 215.4 ± 13.3 |
| `mlp-narrow` | 9,573 | 240.6 ± 9.1 |
| `hwnode-scaled` | 21,895 | 228.5 ± 17.1 |
| `chebyshev-scaled` | 61,295 | 230.4 ± 28.8 |
| `hwnode-xl` | 80,647 | 204.2 ± 32.7 |
| `mlp-large` | 136,581 | 228.8 ± 9.8 |
| `hwnode-xxl` | 308,743 | 46.5 ± 129.2 |
| `hwnode-xxxl` | 1,207,303 | -114.4 ± 29.7 |

### Language Modeling / Parameter Golf

The table below is retained from the project’s earlier research notes because it is part of the empirical motivation for the architecture. The current checkout does not yet contain the full training and evaluation pipeline for these runs, so these should be read as reported historical results rather than a fully reproduced benchmark in this repo.

Metric: `final_int6_roundtrip_exact` (lower is better).

| Run ID | Architecture | Configuration | Wallclock | Params | Val BPB (fp32) | Final Metric (int6) | Artifact Size |
|:--|:--|:--|--:|--:|--:|--:|--:|
| `mlp_mlp1` | MLP | `MULT=1.0` | 600s | 18.3M | 2.695 | 3.634 | 3.6MB |
| `hwnode_s384_o2_v2` | HW-NODE | `s384/o2/v2` | 600s | 18.5M | 3.135 | 3.535 | 4.0MB |
| `mlp_baseline` | MLP | `MULT=3.0` | 600s | 29.9M | 3.243 | 3.603 | 4.1MB |
| `hwnode_s384_o2_v6` | HW-NODE | `s384/o2/v6` | 600s | 18.5M | 3.079 | 3.581 | 4.1MB |
| `hwnode_s384_o2_v4` | HW-NODE | `s384/o2/v4` | 600s | 18.5M | 2.790 | 3.744 | 4.1MB |
| `hwnode_s512_o2_v8` | HW-NODE | `s512/o2/v8` | 600s | 21.2M | 3.310 | 3.666 | 4.6MB |

## Analysis

### What The RL Table Supports

The PPO results support a modest but real claim: HW-NODE can reach strong `LunarLander-v3` performance with a small parameter count. The `hwnode-standard` result exceeds 200 mean reward with 6.3k parameters, which is enough to show that the shared latent-flow block is not merely a theoretical construction.

At the same time, this table does not support a claim that HW-NODE is strictly better than tuned MLP baselines on this task. The best mean reward in the table belongs to `mlp-narrow`, not to an HW-NODE variant. The careful interpretation is therefore:

- HW-NODE is competitive at low parameter count.
- On this benchmark and budget, the best reported MLP still performs better in absolute return.
- The current table is evidence of viability, not dominance.

The non-monotonic relationship between parameter count and reward also matters. `mlp-large` does not outperform `mlp-narrow`, and `chebyshev-scaled` does not clearly outperform `hwnode-scaled` despite using far more parameters. This is definitively corroborated by the extreme-scaling tests (`hwnode-xl`, `hwnode-xxl`, and `hwnode-xxxl`), which show that pushing the architecture past 80K parameters on this task results in catastrophic optimization variance and complete policy collapse within the 500K step budget. That means this setup is not a clean scaling-law study. It is better read as a parameter-efficiency comparison under a fixed training budget rather than as evidence that "larger is worse" universally.

### What The Language-Model Table Supports

The Parameter Golf table supports a similarly narrow conclusion. Under the reported `final_int6_roundtrip_exact` metric, the best listed run is `hwnode_s384_o2_v2` at 3.535. That is better than both reported MLP baselines in the table, including one with substantially more parameters. So the data does support the claim that at least one HW-NODE configuration was competitive, and slightly superior, on that compressed evaluation metric.

What it does not support is a sweeping conclusion that “virtual depth wins” or that deeper recurrence is intrinsically superior. In fact, the reported results show a mismatch between online fp32 validation and final int6 performance:

- `hwnode_s384_o2_v4` has the best fp32 validation BPB in the table.
- That same run has a worse final int6 score than the shallower `v2` and `v6` variants.

The scientifically cautious interpretation is that, in these reported runs, the best fp32 configuration was not the most compression-robust configuration. That is consistent with some interaction between architecture choice and quantized deployment behavior, but the table alone does not identify the mechanism. Claims about why this happens need direct ablation evidence, not just outcome differences.

### Overall Interpretation

Taken together, the current evidence supports three claims:

1. HW-NODE is a viable structured alternative to standard feedforward backbones.
2. Its strongest case, in the results currently reported here, is parameter efficiency rather than outright task dominance.
3. The architecture appears sensitive to configuration details, especially when moving from training-time proxy metrics to deployment-oriented compressed metrics.

What the current evidence does not yet establish is a general scaling advantage over MLPs, a universal benefit from deeper virtual recurrence, or a definitive mechanism for the quantized-language-model behavior. Those are natural next research questions, but they are not conclusions.

## Reproducing The Code In This Repo

Install:

```bash
pip install -e .[dev]
```

Run tests:

```bash
pytest -q
```

Run a single PPO experiment:

```bash
python -m hwnode.run --env CartPole-v1 --agent hwnode --no-wandb
```

Run the MLP baseline:

```bash
python -m hwnode.run --env CartPole-v1 --agent mlp --no-wandb
```

Run the sweep definitions used by the repo:

```bash
python -m hwnode.run --sweep --no-wandb
python -m hwnode.run --sweep-v2 --no-wandb
PYTHONPATH=. python experiments/eval_suite.py --env LunarLander-v3 --num-seeds 3 --no-wandb
```

Run the polynomial-basis comparison:

```bash
PYTHONPATH=. python experiments/taylor_vs_chebyshev.py --env LunarLander-v3 --no-wandb
```

## Notes

- The PPO code in this repository assumes flat vector observations.
- `--device auto` selects `cuda` if available and otherwise falls back to `cpu`.
- The core package is reproducible from this checkout; the historical language-model table is documented here for continuity, but not yet fully reproduced by code in this repo.
