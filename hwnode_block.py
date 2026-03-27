"""
HW-NODE: Hammerstein-Wiener Neural ODE Block
=============================================

A parameter-efficient drop-in replacement for standard feedforward layers.

Architecture (single block):
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  x ∈ ℝᵈ ─► LayerNorm ─► W_in(d→n) ─► σ ─► exp(A·Δt)·z ─► σ ─► W_out(n→d) ─► y ∈ ℝᵈ │
│             ╰── Stage 1: Hammerstein ──╯   ╰─ Stage 2 ─╯   ╰── Stage 3: Wiener ──╯    │
│                  (compress)                 (dynamics)        (expand)                  │
└─────────────────────────────────────────────────────────────────────────┘

Stage 1 — Hammerstein: Projects d-dim input to n-dim bottleneck + nonlinearity.
Stage 2 — ODE Core:   Applies P(A) = Σₖ (A·Δt)ᵏ/k! to the n-dim latent state.
                       A is spectrally normalized (‖A‖₂ ≤ 1) for convergence.
Stage 3 — Wiener:     Nonlinearity + projects n-dim back to d-dim output.

Parameter count per block:  2·n·d + n² + n + 3·d + 1
Compression ratio vs MLP:   ≈ 2n/d + (n/d)²    (e.g. 0.56 at n=16, d=64)

Usage:
    # Single block (replaces one hidden layer)
    block = HWNodeBlock(input_dim=64, state_dim=16)
    y = block(x)  # x: (batch, 64) -> y: (batch, 64)

    # Stacked backbone (replaces full MLP backbone)
    net = HWNodeNetwork(obs_dim=8, hidden_dim=64, state_dim=16, num_blocks=2)
    h = net(x)  # x: (batch, 8) -> h: (batch, 64)
"""

import math
import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P


def _relu_squared(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x).square()


ACTIVATIONS = {
    "relu_squared": _relu_squared,
    "relu": torch.relu,
    "gelu": nn.functional.gelu,
    "silu": nn.functional.silu,
}


class HWNodeBlock(nn.Module):
    """Single Hammerstein-Wiener Neural ODE block.

    Drop-in replacement for a feedforward layer. Input and output have the
    same dimension (input_dim), so it can be used with residual connections.

    Args:
        input_dim:  Working dimension d.
        state_dim:  Bottleneck dimension n. Must be ≤ input_dim.
                    Smaller = fewer params, less expressive dynamics.
        order:      Polynomial truncation order K. Higher = more accurate
                    exp(A) approximation. K=4 gives <0.4% error for ‖A‖₂≤1.
        activation: Nonlinearity for Hammerstein/Wiener stages.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        order: int = 4,
        activation: str = "relu_squared",
    ):
        super().__init__()
        assert state_dim <= input_dim, f"state_dim ({state_dim}) must be ≤ input_dim ({input_dim})"
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.order = order
        self.act = ACTIVATIONS[activation]

        # Stage 1: Hammerstein (d → n)
        self.norm = nn.LayerNorm(input_dim)
        self.W_in = nn.Linear(input_dim, state_dim, bias=True)

        # Stage 2: ODE core (n → n), spectral norm ensures ‖A‖₂ ≤ 1
        self.A = nn.Linear(state_dim, state_dim, bias=False)
        P.spectral_norm(self.A)
        self.dt = nn.Parameter(torch.ones(1))

        # Stage 3: Wiener (n → d)
        self.W_out = nn.Linear(state_dim, input_dim, bias=True)

        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
        with torch.no_grad():
            nn.init.normal_(self.A.weight, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, input_dim) -> (batch, input_dim)"""
        z = self.act(self.W_in(self.norm(x)))           # Hammerstein: (batch, n)
        z = z @ self._poly(self.A.weight * self.dt).T   # ODE core:   (batch, n)
        return self.W_out(self.act(z))                   # Wiener:     (batch, d)

    def _poly(self, A: torch.Tensor) -> torch.Tensor:
        """exp(A) ≈ I + A + A²/2! + ... + Aᴷ/K!"""
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        result, Ak = I.clone(), I.clone()
        for k in range(1, self.order + 1):
            Ak = Ak @ A / k
            result = result + Ak
        return result

    def extra_repr(self) -> str:
        return f"{self.input_dim}→{self.state_dim}→{self.input_dim}, order={self.order}, params={sum(p.numel() for p in self.parameters()):,}"


class HWNodeNetwork(nn.Module):
    """Stacked HW-NODE blocks with residual connections.

    Full backbone network: embed → [block + residual] × N → LayerNorm.

    Args:
        obs_dim:    Input dimension (e.g. observation space).
        hidden_dim: Working dimension d for all blocks.
        state_dim:  Bottleneck dimension n within each block.
        num_blocks: Number of residual HW-NODE blocks.
        order:      Polynomial truncation order.
        activation: Nonlinearity name.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64, state_dim: int = 16,
                 num_blocks: int = 2, order: int = 4, activation: str = "relu_squared"):
        super().__init__()
        self.embed = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            HWNodeBlock(hidden_dim, state_dim, order, activation)
            for _ in range(num_blocks)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, obs_dim) -> (batch, hidden_dim)"""
        h = self.embed(x)
        for block in self.blocks:
            h = h + block(h)  # residual
        return self.norm_out(h)


if __name__ == "__main__":
    # Self-test: verify shapes, param counts, and gradient flow
    print("HW-NODE Self-Test")
    print("=" * 60)

    for d, n, blocks in [(64, 16, 2), (16, 4, 1), (128, 32, 3)]:
        net = HWNodeNetwork(obs_dim=8, hidden_dim=d, state_dim=n, num_blocks=blocks)
        x = torch.randn(4, 8)
        y = net(x)
        loss = y.sum()
        loss.backward()

        params = sum(p.numel() for p in net.parameters())
        grad_ok = all(p.grad is not None and p.grad.abs().sum() > 0
                      for p in net.parameters() if p.requires_grad)

        print(f"  d={d:>3}, n={n:>2}, blocks={blocks} | "
              f"params={params:>6,} | out={tuple(y.shape)} | grads={'✓' if grad_ok else '✗'}")

    print()
    print("Block detail:")
    b = HWNodeBlock(64, 16)
    print(f"  {b}")
    print()
    print("Network detail:")
    net = HWNodeNetwork(8, 64, 16, 2)
    print(f"  {net}")
