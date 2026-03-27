"""HW-NODE: Hammerstein-Wiener Neural ODE block (Mathematical implementation).

Architecture
============

    h_0 = x
    z_l(0) = phi(W_in h_l)
    z_l(Δt) = exp(A Δt) z_l(0)
    h_{l+1} = psi(W_out z_l(Δt))

for l = 0, 1, ..., T-1, where the SAME parameters
(W_in, A, Δt, W_out) are reused at every virtual depth step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def _relu_squared(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x).square()

_ACTIVATIONS: dict[str, callable] = {
    "relu_squared": _relu_squared,
    "relu": torch.relu,
    "gelu": nn.functional.gelu,
    "silu": nn.functional.silu,
}

class SharedHWNODE(nn.Module):
    r"""
    Weight-shared Hammerstein-Wiener Neural ODE.
    
    This module implements the mathematical idea behind HWNODE:
        h_0 = x
        z_l(0) = phi(W_in h_l)
        z_l(Δt) = exp(A Δt) z_l(0)
        h_{l+1} = psi(W_out z_l(Δt))

    for l = 0, 1, ..., T-1, where the SAME parameters
    (W_in, A, Δt, W_out) are reused at every virtual depth step.
    """

    def __init__(
        self,
        model_dim: int | None = None,
        state_dim: int | None = None,
        num_virtual_layers: int = 2,
        taylor_order: int = 2,
        negative_slope: float = 0.5,
        square_output: bool = True,
        residual: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        # Backwards compatibility arg patching
        if model_dim is None:
            model_dim = kwargs.get("input_dim")
        if "input_dim" in kwargs and model_dim != kwargs["input_dim"]:
             model_dim = kwargs["input_dim"]
        if "order" in kwargs:
             taylor_order = kwargs["order"]
        if "activation" in kwargs:
             square_output = (kwargs["activation"] == "relu_squared")

        # positional fallback for legacy initializers like super().__init__(input_dim, state_dim, order, activation)
        if len(args) >= 1:
             taylor_order = args[0]
        if len(args) >= 2:
             square_output = (args[1] == "relu_squared")

        if isinstance(taylor_order, str): # if arg mixup happened
             square_output = (taylor_order == "relu_squared")
             taylor_order = kwargs.get("order", 2)
        if isinstance(num_virtual_layers, str):
             num_virtual_layers = 2
             
        self.model_dim = model_dim
        self.state_dim = state_dim
        self.num_virtual_layers = num_virtual_layers
        self.taylor_order = taylor_order
        self.negative_slope = negative_slope
        self.square_output = square_output
        self.residual = residual

        # For backwards compatibility with older wrapper code naming:
        self.input_dim = model_dim
        self.order = taylor_order

        # Shared Hammerstein map: model space -> latent ODE state space.
        self.W_in = nn.Linear(model_dim, state_dim, bias=False)

        # Shared latent linear dynamics for dz/dt = A z.
        self.A = nn.Parameter(torch.empty(state_dim, state_dim))
        nn.init.normal_(self.A, std=0.1)

        # Shared learnable step size for the latent flow.
        self.dt = nn.Parameter(torch.ones(1))

        # Shared Wiener map: latent state space -> model space.
        self.W_out = nn.Linear(state_dim, model_dim, bias=False)

        # Buffers for one-step power iteration used in spectral normalization.
        self.register_buffer("_power_u", F.normalize(torch.randn(state_dim), dim=0), persistent=False)
        self.register_buffer("_power_v", F.normalize(torch.randn(state_dim), dim=0), persistent=False)

        # Cache exp(A Δt) during eval for speed.
        self.register_buffer("_cached_flow", None, persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._cached_flow = None
        return self

    def _hammerstein_nonlinearity(self, x: Tensor) -> Tensor:
        """Input-side nonlinearity phi."""
        return F.leaky_relu(x, negative_slope=self.negative_slope)

    def _wiener_nonlinearity(self, x: Tensor) -> Tensor:
        """Output-side nonlinearity psi."""
        y = F.leaky_relu(x, negative_slope=self.negative_slope)
        if self.square_output:
            y = y.square()
        return y

    def _spectrally_normalized_A(self) -> Tensor:
        r"""
        Return a spectrally normalized copy of A.
        We estimate sigma_max(A) with one power-iteration update and rescale.
        """
        A = self.A

        with torch.no_grad():
            u = self._power_u
            v = F.normalize(A.T @ u, dim=0)
            u = F.normalize(A @ v, dim=0)
            self._power_u.copy_(u)
            self._power_v.copy_(v)

        u = self._power_u.detach().clone()
        v = self._power_v.detach().clone()
        sigma = (u @ A @ v).clamp(min=1e-8)
        return A / sigma

    def _matrix_exp_approx(self, M: Tensor) -> Tensor:
        """Old compat interface for Taylor matrix approximation"""
        device, dtype = M.device, M.dtype
        I = torch.eye(self.state_dim, device=device, dtype=dtype)
        flow = I.clone()
        current_term = I.clone()

        for k in range(1, self.taylor_order + 1):
            current_term = current_term @ M / k
            flow = flow + current_term
        return flow

    def _flow_matrix(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        r"""Approximate exp(A Δt) with a truncated Taylor series."""
        use_cache = not self.training and not torch.is_grad_enabled()
        if use_cache and self._cached_flow is not None:
            return self._cached_flow

        A_hat = self._spectrally_normalized_A()
        M = (A_hat * self.dt).to(device=device, dtype=dtype)
        
        flow = self._matrix_exp_approx(M)

        if use_cache:
            self._cached_flow = flow

        return flow

    def _one_shared_hwnode_step(self, h: Tensor, flow: Tensor) -> Tensor:
        r"""Apply ONE shared HWNODE step."""
        z0 = self._hammerstein_nonlinearity(self.W_in(h))
        zt = z0 @ flow.T
        y = self.W_out(zt)
        y = self._wiener_nonlinearity(y)
        return y

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Unroll the SAME HWNODE step across virtual depth.
        """
        h = x
        flow = self._flow_matrix(device=x.device, dtype=x.dtype)

        for _ in range(self.num_virtual_layers):
            update = self._one_shared_hwnode_step(h, flow)
            if self.residual:
                h = h + update
            else:
                h = update

        return h


# Backwards compatibility alias
HWNodeBlock = SharedHWNODE


class HWNodeNetwork(nn.Module):
    """Network wrapper mapping RL configuration to SharedHWNODE."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 64,
        state_dim: int = 16,
        num_blocks: int = 2,
        order: int = 4,
        activation: str = "relu_squared",
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(obs_dim, hidden_dim)
        
        # Directly leverage the single shared node with virtual depth matching `num_blocks`
        self.shared_node = SharedHWNODE(
            model_dim=hidden_dim,
            state_dim=state_dim,
            num_virtual_layers=num_blocks,
            taylor_order=order,
            square_output=(activation == "relu_squared"),
            residual=True
        )
        
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.shared_node(h)
        h = self.norm_out(h)
        return h
