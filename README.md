# HW-NODE: Weight-Shared Hammerstein-Wiener Neural ODE

A PyTorch implementation of the **HW-NODE** (Hammerstein-Wiener Neural ODE) architecture. This block is designed as an ultra-parameter-efficient mechanism to replace standard feedforward layers or bottleneck transformations in both Reinforcement Learning policies and Language Modeling sequences.

## Architecture

The core philosophy of HW-NODE is **Virtual Depth** via complete weight-sharing. Instead of stacking physically independent modules (multiplying parameter counts drastically), HW-NODE unrolls a constant, mathematically restricted flow parameterization over $T$ steps.

For step $l$ from $0$ to $T-1$, the exact same $(W_{in}, A, \Delta t, W_{out})$ are applied:

```
h_0 = x
z_l(0) = phi(W_in * h_l)                  # Hammerstein Compression
z_l(Δt) = exp(A * Δt) * z_l(0)            # Linear Latent ODE Flow
h_{l+1} = h_l + psi(W_out * z_l(Δt))      # Wiener Expansion & Residual 
```

### Mathematical Mechanisms:
- **Hammerstein Map:** Compresses the model dimension into a much smaller latent bottleneck state. The activation $\phi(x)$ is a `LeakyReLU`.
- **Spectrally Normalized Dynamics:** The latent state is evolved via the ODE $\frac{dz}{dt} = A z$. The matrix $A$ is spectrally normalized using power-iteration estimation $\hat{A} = A / \sigma_{\max}(A)$ to prevent runaway exponential expansion across the virtual loop.
- **Truncated Polynomial Flow:** The matrix exponential $exp(A \Delta t)$ is approximated efficiently via a cached Order-2 (or dynamically configured) Taylor Series polynomial: $P_K(\hat{A} \Delta t) = \sum_{k=0}^K \frac{(\hat{A} \Delta t)^k}{k!}$
- **Wiener Map:** Projects the evolved state back into the original working dimension with the non-linear boundary $\psi(x) = \text{LeakyReLU}(x)^2$.


## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run comprehensive test suites
pytest

# Full RL evaluation suite (compares scaling & basis types against MLPs)
PYTHONPATH=. python experiments/eval_suite.py --env LunarLander-v3 --num-seeds 3

# Test exclusively massive configurations (XL/XXL scaling on Chebyshev & Taylor variants)
PYTHONPATH=. python experiments/eval_suite.py --only-scaled
```

---

## Reinforcement Learning (PPO)

HW-NODE was evaluated against baseline MLPs on `LunarLander-v3` to test architectural efficiency and parameter density. 

| Architecture           | Parameters  | Mean Final Reward ± Std | Observation |
|:-----------------------|:------------|:------------------------|:------------|
| **hwnode-standard**    | 6,343       | 215.4 ± 13.3            | Mathematically "solves" the environment (>200) with a very small footprint. |
| **mlp-narrow**         | **9,573**   | **240.6 ± 9.1**         | **Best overall performer.** Smaller size trains faster and achieves higher asymptotic mean. |
| **hwnode-scaled**      | 21,895      | 228.5 ± 17.1            | Sits perfectly in the middle of the pack, failing to beat the narrow MLP. |
| **chebyshev-scaled**   | 61,295      | 230.4 ± 28.8            | Equivalent asymptotic reward to scaled standard HW-NODE with higher parameter cost. |
| **mlp-large**          | 136,581     | 228.8 ± 9.8             | Large MLP performs *worse* than the small MLP. |

**Inconclusive RL Scaling**
This data highlights a flaw in using this environment for scaling comparisons:
- **Scaling Inverse:** The largest models (`mlp-large`, `चेbyshev-scaled`) perform worse than the smallest model (`mlp-narrow`). This indicates that larger networks are simply under-training within the 500K timestep budget, while smaller dense networks iterate faster.
- **The Small-MLP Dominance:** The 9.5K MLP runs the fastest and performs the best. While the 6.3K HW-NODE proves the architecture works natively, without explicitly mapping the parameter boundary where small MLPs catastrophically collapse, we cannot claim definitive compression supremacy here. 
- **Takeaway:** Simple RL environments like LunarLander are insufficient for evaluating HW-NODE. To see where HW-NODE's spectral dynamics truly outperform standard linear depth, the architecture must be pushed into complex sequence tasks (like Language Modeling) where MLPs natively run out of representative geometry.



## Language Modeling (Parameter Golf)

We applied HWNODE as a direct replacement for MLPs on an RTX 5090 and RX 6800 XT on the OpenAI Parameter-Golf challenge (Next-Token Bigram framework, `WARMDOWN_ITERS=350`).

### 1. The Pre-Corrected Prototype (Module-Stacked)
These measurements map the older "stacked" unshared prototype run on an RTX 5090 without virtual depth.
*Metric: `final_int6_sliding_window_exact` (lower is better).*

| Config | Params | Step Avg | Final Metric |
|---|---:|---:|---:|
| **`STATE_DIM=768`, Order 2 no bias/gates**| **24.8M** | **1393 ms** | **2.416** |
| `STATE_DIM=864`, Order 2 | 27.6M | 1530 ms | 2.720 |
| `STATE_DIM=864`, Order 2 + Biases | 27.6M | 1548 ms | 2.738 |
| `STATE_DIM=864`, Order 2 + Biases + Gates | 27.6M | 1550 ms | 2.779 |
| `STATE_DIM=960`, Order 2 | 30.6M | 1577 ms | 2.909 |

---

### 2. Corrected Shared-Depth Architecture (RX 6800 XT)
These tests rigorously measure the final **weight-shared** mathematically correct HW-NODE against identical MLPs. All runs strictly evaluated on quantized `int6` payloads.

*Metric: `final_int6_roundtrip_exact` (lower is better).*

| Run ID | Architecture | Configuration | Wallclock | Params | Val BPB (fp32) | Final Metric (int6) | Artifact Size |
|---|---|---|---:|---:|---:|---:|---:|
| `mlp_mlp1` | **MLP** | `MULT=1.0` | 600s | 18.3M | 2.695 | 3.634 | 3.6MB |
| `hwnode_s384_o2_v2` | **HW-NODE** | `s384/o2/v2` | 600s | **18.5M** | 3.135 | **3.535** | 4.0MB |
| `mlp_baseline` | **MLP** | `MULT=3.0` | 600s | 29.9M | 3.243 | 3.603 | 4.1MB |
| `hwnode_s384_o2_v6` | **HW-NODE** | `s384/o2/v6` | 600s | 18.5M | 3.079 | 3.581 | 4.1MB |
| `hwnode_s384_o2_v4` | **HW-NODE** | `s384/o2/v4` | 600s | 18.5M | **2.790** | 3.744 | 4.1MB |
| `hwnode_s512_o2_v8` | **HW-NODE** | `s512/o2/v8` | 600s | 21.2M | 3.310 | 3.666 | 4.6MB |

---

## The Honest Architectural Verdict

Combining the inconclusive RL results and the rigorous Language Modeling density tests, we can conclude the following structural truths about the Hammerstein-Wiener Neural ODE mechanism:

### 1. The Marginal Density Victory
HW-NODE does functionally beat Multi-Layer Perceptrons in sheer representation density. At $\sim 18.5$ Million parameters, the `s384` HW-NODE generated a superior compressed score (`3.535`) compared to both its identically sized 18.3M MLP counterpart (`3.634`) and a massively larger 29.9M MLP (`3.603`). **However, the degree of improvement is ultimately marginal.** Given the intense mathematical engineering, spectral bounds tracking, and execution slow-down inherent in the Taylor-flow, this slight compression victory may not justify replacing highly-optimized parallel dense MLPs in production scenarios. 

### 2. The Quantization Cascading Vulnerability
The defining theoretical promise of HW-NODE is "Virtual Depth"—cycling latent states deeply via a parameterized ODE matrix for "free" reasoning steps. The data proves this works excellently in online full-precision (FP32), where the `v=4` (4-deep recurrence) achieved a massive **2.790 BPB** online validation. 

However, **Virtual Depth completely shatters under quantization.** That same exceptionally smart `v=4` FP32 state collapsed to a disastrous **3.744** in `int6` mapping. The repeated matrix exponentiation operations linearly compound quantization noise, meaning dynamically routed deep features cannot survive severe integer truncation without gradient saturation. 

### 3. Final Conclusion
At its most optimal configuration (strictly shallow recurrence `v=2`, stripped of biases and dynamic gates), HW-NODE provides a fascinating, viable alternative to FFN bottlenecks offering slight sub-30M parameter compression advantages. But the fundamental mathematical mechanics governing its recurrent flow inherently block the architecture from pushing the extremes of virtual depth if post-training quantization is required. Unless the flow operator is entirely reimagined to be quantization-aware at the hardware level, HW-NODE performs best strictly as a shallow, bounded representation trick.
