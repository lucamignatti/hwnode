# HW-NODE: Hammerstein-Wiener Neural ODE

A PyTorch implementation of the **HW-NODE** (Hammerstein-Wiener Neural ODE) block — a parameter-efficient alternative to standard feedforward layers — tested on reinforcement learning benchmarks via PPO.

## Architecture

```
x ∈ ℝᵈ → [Hammerstein] → z₀ ∈ ℝⁿ → [ODE Core] → z₁ ∈ ℝⁿ → [Wiener] → y ∈ ℝᵈ
           (d→n, σ)        exp(A·Δt)·z₀            (n→d, σ)
```

- **Hammerstein**: Linear compression `d→n` + ReLU² nonlinearity  
- **ODE Core**: `z₁ = exp(A·Δt)·z₀` via truncated Taylor series, with `A` spectrally normalized  
- **Wiener**: ReLU² + linear expansion `n→d`

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Quick training (no wandb)
python -m hwnode.run --env CartPole-v1 --agent hwnode --total-timesteps 50000 --no-wandb

# Compare against MLP baseline
python -m hwnode.run --env CartPole-v1 --agent mlp --total-timesteps 50000 --no-wandb

# Full experiment sweep (with wandb)
python -m hwnode.run --sweep --wandb-project hwnode-research
```

## CLI Options

```
--env ENV              Gymnasium environment (default: CartPole-v1)
--agent {hwnode,mlp}   Backbone type (default: hwnode)
--hidden-dim N         Working dimension (default: 64)
--state-dim N          ODE latent dimension (default: 16)
--num-blocks N         Number of blocks/layers (default: 2)
--order N              Taylor series order (default: 4)
--total-timesteps N    Total training steps (default: 500000)
--seed N               Random seed (default: 42)
--sweep                Run full experiment matrix
--no-wandb             Disable wandb logging
```

## Experiment Design

### Sweep v1: Standard Scale (60 runs)

| Config | Backbone | Hidden | State | Params |
|---|---|---|---|---|
| hwnode-small | HW-NODE | 64 | 16 | ~11K |
| hwnode-medium | HW-NODE | 64 | 32 | ~23K |
| mlp-matched-width | MLP | 64 | — | ~35K |
| mlp-narrow | MLP | 32 | — | ~9K |

Tested on: `CartPole-v1`, `Acrobot-v1`, `LunarLander-v3` (5 seeds each, 500K steps)

### Sweep v2: Extreme Compression + Continuous Control (80 runs)

**Tiny discrete** (CartPole, Acrobot):

| Config | Backbone | Hidden | State | Blocks | Params |
|---|---|---|---|---|---|
| hwnode-tiny | HW-NODE | 16 | 4 | 2 | ~1K |
| hwnode-micro | HW-NODE | 16 | 4 | 1 | ~669 |
| mlp-tiny | MLP | 16 | — | 2 | ~2.6K |
| mlp-micro | MLP | 8 | — | 2 | ~779 |

**Continuous control** (Pendulum-v1, MountainCarContinuous-v0):

| Config | Backbone | Hidden | State | Params |
|---|---|---|---|---|
| hwnode-small | HW-NODE | 64 | 16 | ~11K |
| hwnode-tiny | HW-NODE | 32 | 8 | ~3K |
| mlp-matched-width | MLP | 64 | — | ~35K |
| mlp-narrow | MLP | 32 | — | ~9K |

## Results

### Sweep v1: Standard Scale Results

**Mean +/- std across 5 seeds, 500K steps:**

| Config | Params | CartPole-v1 | Acrobot-v1 | LunarLander-v3 |
|---|---|---|---|---|
| hwnode-small | 11K | 492.4 +/- 12.9 | -82.9 +/- 1.9 | 208.5 +/- 45.5 |
| hwnode-medium | 23K | 489.6 +/- 4.5 | -82.3 +/- 2.9 | 242.3 +/- 18.8 |
| mlp-matched-width | 35K | 491.7 +/- 7.2 | -79.5 +/- 4.8 | 233.0 +/- 53.7 |
| mlp-narrow | 9K | 492.4 +/- 7.5 | -79.3 +/- 2.2 | 249.9 +/- 15.0 |

**Observations:**
- **CartPole:** All four configs are equivalent (~490-492). Task is too easy to differentiate.
- **Acrobot:** All configs within a ~3-point band. MLP variants are marginally better (-79.3/-79.5 vs -82.3/-82.9) but the differences are small.
- **LunarLander:** The most informative environment. `mlp-narrow` (9K params, 249.9 +/- 15.0) is the **best performer** — higher mean and lower variance than all others. `hwnode-small` (11K params, 208.5 +/- 45.5) has the **worst mean and highest variance** on this task. `hwnode-medium` (23K, 242.3) approaches `mlp-narrow` (9K, 249.9) but uses 2.5x more parameters to do so.

**The "3x compression" claim is misleading.** HW-NODE at 11K matches the 35K MLP *of the same hidden width*, but a narrower MLP (9K) achieves equal or better performance with even fewer parameters. The bottleneck compression removes over-parameterization, but so does simply using a smaller MLP.

### Sweep v2: Extreme Compression Results

**Mean across 5 seeds, 500K steps:**

| Config | Params | CartPole-v1 | Acrobot-v1 |
|---|---|---|---|
| hwnode-micro | 669 | 494.2 | -83.9 |
| mlp-micro | 779 | 495.6 | -80.9 |
| hwnode-tiny | 1,063 | 487.2 | -81.5 |
| mlp-tiny | 2,579 | 491.3 | -82.3 |

All configs solve CartPole (>474). `mlp-micro` (779 params) slightly outperforms `hwnode-micro` (669 params) on both tasks. The 669-param HW-NODE is notable as the smallest network tested that still solves CartPole (including perfect 500.0 runs), but it does not outperform the MLP baseline.

### Continuous Control

- **Pendulum-v1:** All architectures struggled (~-730 to -875 mean reward). `mlp-narrow` (9K params, -732.3) was the best performer. Performance was dominated by PPO hyperparameter sensitivity (high variance across seeds), not architectural differences.
- **MountainCarContinuous-v0:** No learning for any architecture (reward = -0.0). This environment's sparse reward is incompatible with PPO's Gaussian exploration. Not a valid benchmark for architecture comparison.
- **BipedalWalker-v3:** At medium scale (~25K params, 1M steps), `taylor-learned` scored -140.7 and `cheb-ortho-param` scored -135.4. Neither learned to walk (solved = ~300). Insufficient training budget for this task.

### Polynomial Basis Comparison: Taylor vs Chebyshev

Tested at ~1.2K params on LunarLander-v3 (3 seeds, 500K steps) via `experiments/taylor_vs_chebyshev.py`:

| Variant | Basis | Coefficients | A Constraint | Mean +/- Std |
|---|---|---|---|---|
| taylor-fixed | Monomials | Fixed 1/k! | Spectral norm | 71.3 +/- 38.5 |
| **taylor-learned** | **Monomials** | **Learnable w_k** | **Spectral norm** | **131.3 +/- 33.3** |
| chebyshev | Chebyshev T_k | Learnable w_k | Spectral norm | 56.4 +/- 116.0 |
| cheb-ortho (init) | Chebyshev T_k | Learnable w_k | Spectral norm + ortho init | 114.7 +/- 81.1 |
| cheb-ortho-param | Chebyshev T_k | Learnable w_k | Orthogonal parametrization | 62.7 +/- 56.2 |

`taylor-learned` is the clear winner: highest mean, lowest variance. Chebyshev produced the single best run (seed 0: 208.6) but also the worst (seed 1: -72.9), indicating an unstable optimization landscape. Orthogonal init reduced Chebyshev's variance (sigma 116 to 81) but didn't fix it. Orthogonal parametrization during training was worse than orthogonal init alone, likely because it reduces A's degrees of freedom from 16 to 6 at state_dim=4.

## Honest Assessment: HW-NODE vs MLP

### What the Data Actually Shows
1. **On easy tasks (CartPole, Acrobot), all architectures are equivalent.** These benchmarks cannot differentiate any reasonable architecture. Even 669 parameters suffices.
2. **On the hardest discrete task (LunarLander), MLP is slightly but consistently better.** `mlp-narrow` (9K params) outperforms `hwnode-small` (11K params) by ~41 points on LunarLander (249.9 vs 208.5), with much lower variance (15.0 vs 45.5). HW-NODE needs 2.5x more parameters (`hwnode-medium`, 23K) to approach `mlp-narrow`'s performance.
3. **HW-NODE compresses relative to same-width MLP, but so does a narrower MLP.** The 35K -> 11K compression sounds impressive until you note that a 9K MLP does equally well. HW-NODE's bottleneck removes redundancy, but the redundancy wasn't necessary in the first place.
4. **On continuous control, neither architecture succeeded sufficiently to compare.** Pendulum showed no architectural signal; BipedalWalker was under-trained.

### What We Learned
5. **Learnable term weights are the single best improvement** to the polynomial core (+84% mean reward at 1.2K params on LunarLander, 131.3 vs 71.3).
6. **Chebyshev basis is high-variance.** It produced the best single run (208.6) and the worst (-72.9) across all variants. The optimization landscape is too sensitive at small scale.
7. **Spectral normalization is doing real work.** Wandb logs show the optimizer consistently pushing ||A||_2 to the constraint boundary of 1.0.
8. **These benchmarks are insufficient** to make strong claims about HW-NODE vs MLP. The tasks are either too easy (CartPole/Acrobot), too sensitive to PPO hyperparameters (Pendulum), or require more training budget (BipedalWalker).

### Open Questions
- Does HW-NODE provide benefits on **high-dimensional continuous control** (Humanoid, 376-dim obs) where the bottleneck compression acts as regularization?
- Does it improve **sample efficiency** even if asymptotic performance is matched?
- Does the architecture scale differently when used as an **FFN replacement in transformers** (CHODE)?

## Future Research & Optimizations

### Performance Optimization
* **Matrix Polynomial Caching:** In on-policy RL algorithms like PPO, network weights don't change during the rollout phase. Compute the polynomial matrix expansion once at the start of the rollout and cache it for the duration of data collection.

### Architectural Exploration
* **Terminology Shift:** The core is better conceptualized as a *deeply coupled polynomial expansion* rather than a strict Taylor Series of a linear system, since the non-linearities break the ODE interpretation.
* **Parametrized Term Weighting:** Learn free scalars w_k for each term. **Tested: best single improvement (+84% reward at 1.2K params).**
* **Dynamic Series Depth (T):** Map observations to a dynamic order cutoff, spending fewer FLOPs on easy states.
* **Residual Series:** Make the polynomial accumulation strictly residual at each step.
* **Alternative Activations:** Test `SiLU` vs `ReLU^2` in the Hammerstein and Wiener mappings.
* **Chebyshev Polynomial Basis:** Orthogonal basis with minimax-optimal approximation on [-1, 1]. **Tested: highest ceiling (reward 208) but catastrophic variance at tiny scale. Needs investigation at larger state_dim.**
* **Orthogonal A Parametrization:** Constrain A to the orthogonal manifold (all singular values = 1). **Tested: over-constraining at small n, but theoretically sound at larger scale where DOF ratio improves.**

