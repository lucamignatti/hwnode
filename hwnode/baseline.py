"""MLP baseline backbone — same interface as HWNodeNetwork."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPNetwork(nn.Module):
    """Standard MLP backbone with residual connections.

    Drop-in replacement for HWNodeNetwork: same constructor signature
    (ignoring HW-NODE-specific args) and same forward interface.

    Parameters
    ----------
    obs_dim : int
        Observation space dimension.
    hidden_dim : int
        Hidden layer width.
    num_blocks : int
        Number of hidden layers.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 64,
        num_blocks: int = 2,
        **kwargs,  # absorb hwnode-specific args (state_dim, order, etc.)
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(obs_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_blocks):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))

        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, obs_dim)

        Returns
        -------
        h : (batch, hidden_dim)
        """
        h = self.embed(x)
        for layer in self.layers:
            h = h + layer(h)  # residual
        h = self.norm_out(h)
        return h
