"""Head-to-head: Taylor vs Chebyshev polynomial basis in HW-NODE.

Standalone script that creates a Chebyshev variant of HWNodeBlock,
then races it against the original Taylor variant on LunarLander-v3
with tiny parameter budgets where the polynomial basis quality matters.

Usage:
    PYTHONPATH=. python experiments/taylor_vs_chebyshev.py --no-wandb
    PYTHONPATH=. python experiments/taylor_vs_chebyshev.py --wandb-project hwnode-research
"""

from __future__ import annotations

import argparse
import copy
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as P

from hwnode.model import HWNodeBlock, HWNodeNetwork, _ACTIVATIONS
from hwnode.agent import ActorCritic, RolloutBuffer
from hwnode.baseline import MLPNetwork
from hwnode.config import ModelConfig


# ---------------------------------------------------------------------------
# Chebyshev HW-NODE block
# ---------------------------------------------------------------------------

class ChebyshevHWNodeBlock(nn.Module):
    """HW-NODE block using Chebyshev polynomial basis instead of Taylor.

    The core computes: z₁ = P(A) · z₀ where P(A) = Σ wₖ Tₖ(A)
    with Chebyshev polynomials Tₖ computed via the three-term recurrence:
        T₀(A) = I
        T₁(A) = A
        Tₖ₊₁(A) = 2A·Tₖ(A) - Tₖ₋₁(A)

    The coefficients wₖ are learnable (not fixed to exp() Taylor coefficients),
    making this a learned polynomial operator in an orthogonal basis.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        order: int = 4,
        activation: str = "relu_squared",
        a_init: str = "normal",
        a_constraint: str = "spectral_norm",
    ) -> None:
        super().__init__()
        assert state_dim <= input_dim
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.order = order
        self.activation_fn = _ACTIVATIONS[activation]
        self.a_init = a_init
        self.a_constraint = a_constraint

        # Stage 1: Hammerstein
        self.norm_in = nn.LayerNorm(input_dim)
        self.W_in = nn.Linear(input_dim, state_dim, bias=True)

        # Stage 2: ODE core — constrained A + learnable Chebyshev coeffs
        self.A = nn.Linear(state_dim, state_dim, bias=False)
        if a_constraint == "orthogonal":
            # Keeps A on the orthogonal manifold: all singular values = 1
            # Strictly stronger than spectral_norm (Lipschitz = 1 exactly)
            P.orthogonal(self.A)
        else:
            # Default: spectral norm (largest singular value <= 1)
            P.spectral_norm(self.A)

        # Learnable weights for each Chebyshev term
        init_weights = [1.0] + [1.0 / max(1, k) for k in range(1, order + 1)]
        self.cheb_weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))

        # Stage 3: Wiener
        self.W_out = nn.Linear(state_dim, input_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
        # For orthogonal parametrization, the parametrization handles A.
        # For spectral_norm, optionally use orthogonal init.
        if self.a_constraint != "orthogonal":
            with torch.no_grad():
                if self.a_init == "orthogonal":
                    nn.init.orthogonal_(self.A.weight)
                else:
                    nn.init.normal_(self.A.weight, std=0.1)

    def _chebyshev_poly(self, A: torch.Tensor) -> torch.Tensor:
        """Compute P(A) = Σ wₖ Tₖ(A) via Chebyshev three-term recurrence.

        Since A is spectrally normalized (||A||₂ ≤ 1), eigenvalues lie in [-1, 1],
        which is exactly the domain where Chebyshev polynomials are orthogonal
        and provide minimax-optimal approximation.
        """
        n = A.shape[0]
        I = torch.eye(n, device=A.device, dtype=A.dtype)

        # T₀ = I, T₁ = A
        T_prev = I       # T₀
        T_curr = A        # T₁
        result = self.cheb_weights[0] * T_prev  # w₀·T₀

        if self.order >= 1:
            result = result + self.cheb_weights[1] * T_curr  # w₁·T₁

        for k in range(2, self.order + 1):
            T_next = 2 * A @ T_curr - T_prev  # Tₖ₊₁ = 2A·Tₖ - Tₖ₋₁
            result = result + self.cheb_weights[k] * T_next
            T_prev = T_curr
            T_curr = T_next

        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stage 1: Hammerstein
        z = self.W_in(self.norm_in(x))
        z = self.activation_fn(z)

        # Stage 2: Chebyshev polynomial operator
        A = self.A.weight
        P_A = self._chebyshev_poly(A)
        z = z @ P_A.T

        # Stage 3: Wiener
        z = self.activation_fn(z)
        y = self.W_out(z)
        return y


class ChebyshevHWNodeNetwork(nn.Module):
    """Stacked Chebyshev HW-NODE blocks with residual connections."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 64,
        state_dim: int = 16,
        num_blocks: int = 2,
        order: int = 4,
        activation: str = "relu_squared",
        a_init: str = "normal",
        a_constraint: str = "spectral_norm",
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ChebyshevHWNodeBlock(hidden_dim, state_dim, order, activation,
                                 a_init=a_init, a_constraint=a_constraint)
            for _ in range(num_blocks)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.norm_out(h)
        return h


# ---------------------------------------------------------------------------
# Also add a Taylor variant with learnable weights for fair comparison
# ---------------------------------------------------------------------------

class LearnedTaylorHWNodeBlock(nn.Module):
    """Taylor basis with learnable term weights (fair comparison to Chebyshev).

    P(A) = Σ wₖ · Aᵏ  (note: monomial basis, NOT orthogonal)
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        order: int = 4,
        activation: str = "relu_squared",
    ) -> None:
        super().__init__()
        assert state_dim <= input_dim
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.order = order
        self.activation_fn = _ACTIVATIONS[activation]

        self.norm_in = nn.LayerNorm(input_dim)
        self.W_in = nn.Linear(input_dim, state_dim, bias=True)

        self.A = nn.Linear(state_dim, state_dim, bias=False)
        P.spectral_norm(self.A)
        # Learnable weights initialized to Taylor coefficients of exp(x)
        import math as _math
        init_weights = [1.0 / _math.factorial(k) for k in range(order + 1)]
        self.taylor_weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))

        self.W_out = nn.Linear(state_dim, input_dim, bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.xavier_uniform_(self.W_out.weight)
        with torch.no_grad():
            nn.init.normal_(self.A.weight, std=0.1)

    def _learned_taylor_poly(self, A: torch.Tensor) -> torch.Tensor:
        n = A.shape[0]
        I = torch.eye(n, device=A.device, dtype=A.dtype)
        result = self.taylor_weights[0] * I
        A_power = I
        for k in range(1, self.order + 1):
            A_power = A_power @ A
            result = result + self.taylor_weights[k] * A_power
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.W_in(self.norm_in(x))
        z = self.activation_fn(z)
        A = self.A.weight
        P_A = self._learned_taylor_poly(A)
        z = z @ P_A.T
        z = self.activation_fn(z)
        y = self.W_out(z)
        return y


class LearnedTaylorHWNodeNetwork(nn.Module):
    """Stacked learned-Taylor HW-NODE blocks."""

    def __init__(self, obs_dim, hidden_dim=64, state_dim=16, num_blocks=2,
                 order=4, activation="relu_squared"):
        super().__init__()
        self.embed = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            LearnedTaylorHWNodeBlock(hidden_dim, state_dim, order, activation)
            for _ in range(num_blocks)
        ])
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.norm_out(h)
        return h


# ---------------------------------------------------------------------------
# Patched ActorCritic that accepts a custom backbone class
# ---------------------------------------------------------------------------

class FlexActorCritic(nn.Module):
    """ActorCritic that accepts an arbitrary backbone class."""

    def __init__(self, obs_dim, act_dim, BackboneClass, continuous=False, **backbone_kwargs):
        super().__init__()
        self.continuous = continuous
        self.act_dim = act_dim

        # obs_dim is passed separately but backbones need it
        backbone_kwargs["obs_dim"] = obs_dim
        self.actor_backbone = BackboneClass(**backbone_kwargs)
        self.critic_backbone = BackboneClass(**backbone_kwargs)

        if continuous:
            from torch.distributions import Normal
            self.policy_mean = nn.Linear(self.actor_backbone.output_dim, act_dim)
            self.policy_log_std = nn.Parameter(torch.zeros(act_dim))
            nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
            nn.init.zeros_(self.policy_mean.bias)
        else:
            from torch.distributions import Categorical
            self.policy_head = nn.Linear(self.actor_backbone.output_dim, act_dim)
            nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
            nn.init.zeros_(self.policy_head.bias)

        self.value_head = nn.Linear(self.critic_backbone.output_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _get_dist(self, obs):
        features = self.actor_backbone(obs)
        if self.continuous:
            from torch.distributions import Normal
            mean = self.policy_mean(features)
            std = self.policy_log_std.exp().expand_as(mean)
            return Normal(mean, std)
        else:
            from torch.distributions import Categorical
            return Categorical(logits=self.policy_head(features))

    def forward(self, obs):
        dist = self._get_dist(obs)
        value = self.value_head(self.critic_backbone(obs)).squeeze(-1)
        return dist, value

    def get_value(self, obs):
        return self.value_head(self.critic_backbone(obs)).squeeze(-1)

    def evaluate_actions(self, obs, actions):
        dist, value = self.forward(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        if self.continuous:
            log_probs = log_probs.sum(-1)
            entropy = entropy.sum(-1)
        return log_probs, value, entropy

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Minimal PPO training loop (self-contained)
# ---------------------------------------------------------------------------

def train_agent(
    env_id: str,
    model: FlexActorCritic,
    seed: int,
    total_timesteps: int = 500_000,
    rollout_steps: int = 2048,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    entropy_coeff: float = 0.01,
    value_coeff: float = 0.5,
    max_grad_norm: float = 0.5,
    num_epochs: int = 4,
    batch_size: int = 64,
    wandb_run=None,
    label: str = "",
    env_kwargs: dict = None,
) -> dict:
    """Train a PPO agent and return metrics."""
    device = next(model.parameters()).device
    env = gym.make(env_id, **(env_kwargs or {}))
    continuous = isinstance(env.action_space, gym.spaces.Box)
    act_dim = env.action_space.shape[0] if continuous else env.action_space.n

    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = RolloutBuffer(
        rollout_steps, env.observation_space.shape[0], device,
        act_dim=act_dim if continuous else 1, continuous=continuous,
    )

    total_params = model.param_count()
    print(f"[{label}] Params: {total_params:,} | Env: {env_id} | Seed: {seed}")

    obs, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    global_step = 0
    num_updates = 0
    recent_rewards = []
    start_time = time.time()
    total_updates = total_timesteps // rollout_steps

    while global_step < total_timesteps:
        buffer.reset()
        model.eval()

        for step in range(rollout_steps):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dist, value = model(obs_t)
                action = dist.sample()
                if continuous:
                    log_prob = dist.log_prob(action).sum(-1)
                else:
                    log_prob = dist.log_prob(action)

            if continuous:
                action_np = action.squeeze(0).cpu().numpy()
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
                env_action = action_np
            else:
                env_action = action.item()

            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            if continuous:
                buffer.store(obs, action_np, float(reward), done, log_prob.item(), value.item())
            else:
                buffer.store(obs, action.item(), float(reward), done, log_prob.item(), value.item())

            obs = next_obs
            episode_reward += reward
            episode_length += 1
            global_step += 1

            if done:
                recent_rewards.append(episode_reward)
                episode_count += 1
                if wandb_run:
                    wandb_run.log({
                        f"{label}/episode_reward": episode_reward,
                        f"{label}/episode_length": episode_length,
                        "global_step": global_step,
                    }, step=global_step)
                obs, _ = env.reset()
                episode_reward = 0.0
                episode_length = 0

        # GAE
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = model.get_value(obs_t).item()
        buffer.compute_gae(last_value, gamma, gae_lambda)

        # PPO update
        model.train()
        for epoch in range(num_epochs):
            for batch in buffer.get_batches(batch_size):
                b_obs, b_actions, b_old_lp, b_returns, b_adv = batch
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
                new_lp, new_v, ent = model.evaluate_actions(b_obs, b_actions)
                ratio = torch.exp(new_lp - b_old_lp)
                s1 = ratio * b_adv
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_adv
                p_loss = -torch.min(s1, s2).mean()
                v_loss = nn.functional.mse_loss(new_v, b_returns)
                loss = p_loss + value_coeff * v_loss - entropy_coeff * ent.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        num_updates += 1
        mean_reward = np.mean(recent_rewards[-20:]) if recent_rewards else 0.0
        sps = int(global_step / (time.time() - start_time))

        if num_updates % 5 == 0:
            print(f"  [{label}] update {num_updates:>3}/{total_updates} | "
                  f"step={global_step:>7} | reward={mean_reward:>8.1f} | SPS={sps}")

        if wandb_run:
            log_dict = {
                f"{label}/mean_reward_20ep": mean_reward,
                f"{label}/sps": sps,
                "global_step": global_step,
            }
            # Log Chebyshev/Taylor learned weights
            for module in model.modules():
                if hasattr(module, "cheb_weights"):
                    for i, w in enumerate(module.cheb_weights.data):
                        log_dict[f"{label}/cheb_w{i}"] = w.item()
                    break
                elif hasattr(module, "taylor_weights"):
                    for i, w in enumerate(module.taylor_weights.data):
                        log_dict[f"{label}/taylor_w{i}"] = w.item()
                    break
            wandb_run.log(log_dict, step=global_step)

    env.close()
    final = np.mean(recent_rewards[-100:]) if recent_rewards else 0.0
    print(f"  [{label}] DONE — final reward (100ep): {final:.1f}")
    return {"final_mean_reward": final, "param_count": total_params, "label": label, "seed": seed}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Taylor vs Chebyshev HW-NODE")
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="hwnode-research")
    parser.add_argument("--variant", type=str, default="all",
                        choices=["all", "taylor-fixed", "taylor-learned", "chebyshev",
                                 "cheb-ortho", "cheb-ortho-param"])
    parser.add_argument("--continuous", action="store_true",
                        help="Force continuous action space (e.g. LunarLander continuous mode)")
    parser.add_argument("--scale", type=str, default=None,
                        choices=["tiny", "small", "medium"],
                        help="Preset scale: tiny=h16/n4, small=h64/n16, medium=h64/n32")
    args = parser.parse_args()

    # Apply scale presets
    if args.scale == "tiny":
        args.hidden_dim, args.state_dim = 16, 4
    elif args.scale == "small":
        args.hidden_dim, args.state_dim = 64, 16
    elif args.scale == "medium":
        args.hidden_dim, args.state_dim = 64, 32

    # Create env with optional kwargs (e.g. continuous=True for LunarLander)
    env_kwargs = {}
    if args.continuous:
        env_kwargs["continuous"] = True

    env_tmp = gym.make(args.env, **env_kwargs)
    backbone_kwargs = dict(
        obs_dim=env_tmp.observation_space.shape[0],
        hidden_dim=args.hidden_dim,
        state_dim=args.state_dim,
        num_blocks=args.num_blocks,
        order=args.order,
    )
    continuous = isinstance(env_tmp.action_space, gym.spaces.Box)
    act_dim = env_tmp.action_space.shape[0] if continuous else env_tmp.action_space.n
    env_tmp.close()

    # Factories for variant-specific kwargs
    def _make_cheb_ortho_init(**kwargs):
        return ChebyshevHWNodeNetwork(**kwargs, a_init="orthogonal")

    def _make_cheb_ortho_param(**kwargs):
        """Chebyshev with orthogonal parametrization: A stays on the orthogonal
        manifold throughout training. All singular values = 1 at all times."""
        return ChebyshevHWNodeNetwork(**kwargs, a_constraint="orthogonal")

    # Five variants to compare:
    variants = [
        ("taylor-fixed",     HWNodeNetwork,              "Taylor (fixed 1/k! coefficients)"),
        ("taylor-learned",   LearnedTaylorHWNodeNetwork,  "Taylor (learned wₖ coefficients)"),
        ("chebyshev",        ChebyshevHWNodeNetwork,      "Chebyshev (learned wₖ, normal A init)"),
        ("cheb-ortho",       _make_cheb_ortho_init,       "Chebyshev (learned wₖ, orthogonal A init)"),
        ("cheb-ortho-param", _make_cheb_ortho_param,      "Chebyshev (learned wₖ, orthogonal A parametrization)"),
    ]

    if args.variant != "all":
        variants = [v for v in variants if v[0] == args.variant]

    all_results = []

    for seed in range(args.num_seeds):
        for tag, BackboneClass, desc in variants:
            label = f"{tag}-s{seed}"
            print(f"\n{'='*60}")
            print(f" {desc} | seed={seed}")
            print(f"{'='*60}")

            wandb_run = None
            if not args.no_wandb:
                import wandb
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=f"{tag}-{args.env}-s{seed}",
                    group=f"basis-comparison-{args.env}",
                    config={
                        "env": args.env, "seed": seed, "variant": tag,
                        "hidden_dim": args.hidden_dim, "state_dim": args.state_dim,
                        "num_blocks": args.num_blocks, "order": args.order,
                    },
                    reinit=True,
                )

            model = FlexActorCritic(
                backbone_kwargs["obs_dim"], act_dim, BackboneClass,
                continuous=continuous,
                **{k: v for k, v in backbone_kwargs.items() if k != "obs_dim"},
            )

            result = train_agent(
                args.env, model, seed,
                total_timesteps=args.total_timesteps,
                wandb_run=wandb_run,
                label=tag,
                env_kwargs=env_kwargs,
            )
            result["variant"] = tag
            all_results.append(result)

            if wandb_run:
                import wandb
                wandb.finish()

    # Summary
    print(f"\n{'='*60}")
    print(" COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"{'Variant':>20} | {'Seed':>4} | {'Reward':>8} | {'Params':>8}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['variant']:>20} | {r['seed']:>4} | {r['final_mean_reward']:>8.1f} | {r['param_count']:>8,}")

    # Averages
    print("-" * 55)
    for tag, _, desc in variants:
        rewards = [r["final_mean_reward"] for r in all_results if r["variant"] == tag]
        params = [r["param_count"] for r in all_results if r["variant"] == tag][0]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        print(f"{tag:>20} | {'AVG':>4} | {mean_r:>5.1f}±{std_r:<4.1f} | {params:>8,}")


if __name__ == "__main__":
    main()
