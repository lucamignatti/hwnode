"""PPO agent with configurable HW-NODE or MLP backbone.

Supports both discrete (Categorical) and continuous (Normal) action spaces.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from hwnode.model import HWNodeNetwork
from hwnode.baseline import MLPNetwork
from hwnode.config import ModelConfig


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Separate actor (policy) and critic (value) networks.

    Both share the same backbone architecture but have independent weights.
    Supports both discrete and continuous action spaces.
    """

    def __init__(
        self, obs_dim: int, act_dim: int, cfg: ModelConfig, continuous: bool = False
    ) -> None:
        super().__init__()
        self.continuous = continuous
        self.act_dim = act_dim
        Backbone = HWNodeNetwork if cfg.backbone == "hwnode" else MLPNetwork

        backbone_kwargs = dict(
            obs_dim=obs_dim,
            hidden_dim=cfg.hidden_dim,
            num_blocks=cfg.num_blocks,
        )
        if cfg.backbone == "hwnode":
            if cfg.virtual_depth > 0:
                # New semantics: num_blocks = physical, virtual_depth = internal
                backbone_kwargs.update(
                    state_dim=cfg.state_dim,
                    order=cfg.order,
                    activation=cfg.activation,
                    num_blocks=cfg.num_blocks,
                    virtual_depth=cfg.virtual_depth,
                )
            else:
                # Backwards compat: 1 physical block, num_blocks as virtual depth
                backbone_kwargs.update(
                    state_dim=cfg.state_dim,
                    order=cfg.order,
                    activation=cfg.activation,
                    num_blocks=1,
                    virtual_depth=cfg.num_blocks,
                )

        self.actor_backbone = Backbone(**backbone_kwargs)
        self.critic_backbone = Backbone(**backbone_kwargs)

        if continuous:
            # Continuous: output mean, learn a state-independent log_std
            self.policy_mean = nn.Linear(self.actor_backbone.output_dim, act_dim)
            self.policy_log_std = nn.Parameter(torch.zeros(act_dim))
            nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
            nn.init.zeros_(self.policy_mean.bias)
        else:
            # Discrete: output logits
            self.policy_head = nn.Linear(self.actor_backbone.output_dim, act_dim)
            nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
            nn.init.zeros_(self.policy_head.bias)

        self.value_head = nn.Linear(self.critic_backbone.output_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _get_dist(self, obs: torch.Tensor) -> Union[Categorical, Normal]:
        """Get action distribution from observations."""
        features = self.actor_backbone(obs)
        if self.continuous:
            mean = self.policy_mean(features)
            std = self.policy_log_std.exp().expand_as(mean)
            return Normal(mean, std)
        else:
            logits = self.policy_head(features)
            return Categorical(logits=logits)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[Union[Categorical, Normal], torch.Tensor]:
        """Return action distribution and state value."""
        dist = self._get_dist(obs)
        value = self.value_head(self.critic_backbone(obs)).squeeze(-1)
        return dist, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.critic_backbone(obs)).squeeze(-1)

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probs, values, and entropy for given obs/actions."""
        dist, value = self.forward(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        # For continuous: sum log_prob and entropy across action dims
        if self.continuous:
            log_probs = log_probs.sum(-1)
            entropy = entropy.sum(-1)
        return log_probs, value, entropy

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """Fixed-size buffer for PPO rollouts with GAE computation.

    Supports both discrete (scalar) and continuous (vector) actions.
    """

    def __init__(
        self,
        rollout_steps: int,
        obs_dim: int,
        device: torch.device,
        act_dim: int = 1,
        continuous: bool = False,
    ):
        self.rollout_steps = rollout_steps
        self.device = device
        self.continuous = continuous
        self.obs = torch.zeros(rollout_steps, obs_dim, device=device)

        if continuous:
            self.actions = torch.zeros(rollout_steps, act_dim, device=device)
        else:
            self.actions = torch.zeros(rollout_steps, dtype=torch.long, device=device)

        self.rewards = torch.zeros(rollout_steps, device=device)
        self.dones = torch.zeros(rollout_steps, device=device)
        self.log_probs = torch.zeros(rollout_steps, device=device)
        self.values = torch.zeros(rollout_steps, device=device)
        self.advantages = torch.zeros(rollout_steps, device=device)
        self.returns = torch.zeros(rollout_steps, device=device)
        self.ptr = 0

    def store(
        self,
        obs: np.ndarray,
        action,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        self.obs[self.ptr] = torch.as_tensor(obs, device=self.device)
        if self.continuous:
            self.actions[self.ptr] = torch.as_tensor(action, device=self.device)
        else:
            self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute GAE advantages and discounted returns."""
        last_gae = 0.0
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t].item()
            else:
                next_value = self.values[t + 1].item()
                next_non_terminal = 1.0 - self.dones[t].item()

            delta = (
                self.rewards[t].item()
                + gamma * next_value * next_non_terminal
                - self.values[t].item()
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """Yield random minibatches of indices."""
        indices = np.random.permutation(self.rollout_steps)
        for start in range(0, self.rollout_steps, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            yield (
                self.obs[batch_idx],
                self.actions[batch_idx],
                self.log_probs[batch_idx],
                self.returns[batch_idx],
                self.advantages[batch_idx],
            )

    def reset(self) -> None:
        self.ptr = 0
