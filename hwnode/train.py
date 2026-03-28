"""Training loop: environment interaction → PPO updates → wandb logging.

Supports both discrete and continuous action spaces.
"""

from __future__ import annotations

import time
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from hwnode.agent import ActorCritic, RolloutBuffer
from hwnode.config import ExperimentConfig


def _get_spectral_norms(model: ActorCritic) -> list[float]:
    """Extract spectral norms of A matrices from HW-NODE blocks."""
    norms = []
    for module in model.modules():
        if hasattr(module, "A") and hasattr(module.A, "weight"):
            with torch.no_grad():
                norms.append(torch.linalg.norm(module.A.weight, ord=2).item())
    return norms


def train(cfg: ExperimentConfig, wandb_run=None) -> dict:
    """Run a single training experiment.

    Parameters
    ----------
    cfg : ExperimentConfig
        Full experiment configuration.
    wandb_run : optional
        An initialized wandb run (None if --no-wandb).

    Returns
    -------
    dict with final metrics.
    """
    # ---- Setup ----
    device = torch.device(cfg.device)
    env = gym.make(cfg.env_id)

    # Seed everything
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    obs_dim = env.observation_space.shape[0]

    # Detect action space type
    continuous = isinstance(env.action_space, gym.spaces.Box)
    if continuous:
        act_dim = env.action_space.shape[0]
        act_low_np = env.action_space.low
        act_high_np = env.action_space.high
    else:
        act_dim = env.action_space.n

    model = ActorCritic(obs_dim, act_dim, cfg.model, continuous=continuous).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ppo.lr)
    buffer = RolloutBuffer(
        cfg.ppo.rollout_steps, obs_dim, device,
        act_dim=act_dim if continuous else 1,
        continuous=continuous,
    )

    total_params = model.param_count()
    action_type = "continuous" if continuous else "discrete"
    print(f"[train] Model: {cfg.model.backbone} | Params: {total_params:,} | Actions: {action_type}({act_dim})")
    print(f"[train] Env: {cfg.env_id} | Seed: {cfg.seed}")
    print(f"[train] Device: {device}")
    if wandb_run:
        wandb_run.config.update({"model/param_count": total_params, "action_type": action_type})

    # ---- Training loop ----
    obs, _ = env.reset(seed=cfg.seed)
    episode_reward = 0.0
    episode_length = 0
    episode_count = 0
    global_step = 0
    num_updates = 0
    recent_rewards: list[float] = []

    start_time = time.time()
    total_updates = cfg.total_timesteps // cfg.ppo.rollout_steps

    while global_step < cfg.total_timesteps:
        # ---- Collect rollout ----
        buffer.reset()
        model.eval()

        for step in range(cfg.ppo.rollout_steps):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                dist, value = model(obs_t)
                action = dist.sample()

                if continuous:
                    log_prob = dist.log_prob(action).sum(-1)
                else:
                    log_prob = dist.log_prob(action)

            # Convert action for env.step
            if continuous:
                action_np = action.squeeze(0).cpu().numpy()
                # Clip to action bounds
                action_np = np.clip(action_np, act_low_np, act_high_np)
                env_action = action_np
            else:
                env_action = action.item()

            next_obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            if continuous:
                buffer.store(obs, action_np, reward, done, log_prob.item(), value.item())
            else:
                buffer.store(obs, action.item(), reward, done, log_prob.item(), value.item())

            obs = next_obs
            episode_reward += reward
            episode_length += 1
            global_step += 1

            if done:
                recent_rewards.append(episode_reward)
                episode_count += 1

                if wandb_run:
                    wandb_run.log({
                        "episode/reward": episode_reward,
                        "episode/length": episode_length,
                        "episode/count": episode_count,
                        "global_step": global_step,
                    }, step=global_step)

                obs, _ = env.reset()
                episode_reward = 0.0
                episode_length = 0

        # ---- Compute advantages ----
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = model.get_value(obs_t).item()
        buffer.compute_gae(last_value, cfg.ppo.gamma, cfg.ppo.gae_lambda)

        # ---- PPO update ----
        model.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_batches = 0

        for epoch in range(cfg.ppo.num_epochs):
            for batch in buffer.get_batches(cfg.ppo.batch_size):
                b_obs, b_actions, b_old_logprobs, b_returns, b_advantages = batch

                # Normalize advantages
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

                new_log_probs, new_values, entropy = model.evaluate_actions(b_obs, b_actions)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - b_old_logprobs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - cfg.ppo.clip_eps, 1 + cfg.ppo.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_values, b_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + cfg.ppo.value_coeff * value_loss
                    + cfg.ppo.entropy_coeff * entropy_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.ppo.max_grad_norm)
                optimizer.step()

                # Approximate KL for monitoring
                with torch.no_grad():
                    approx_kl = (b_old_logprobs - new_log_probs).mean().item()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                n_batches += 1

        num_updates += 1

        # ---- Logging ----
        if num_updates % cfg.log_interval == 0:
            avg_policy_loss = total_policy_loss / max(n_batches, 1)
            avg_value_loss = total_value_loss / max(n_batches, 1)
            avg_entropy = total_entropy / max(n_batches, 1)
            avg_kl = total_approx_kl / max(n_batches, 1)
            sps = int(global_step / (time.time() - start_time))
            mean_reward = np.mean(recent_rewards[-20:]) if recent_rewards else 0.0

            print(
                f"[update {num_updates:>4}/{total_updates}] "
                f"step={global_step:>7} | "
                f"reward={mean_reward:>8.1f} | "
                f"ploss={avg_policy_loss:>7.4f} | "
                f"vloss={avg_value_loss:>7.4f} | "
                f"ent={avg_entropy:>6.3f} | "
                f"SPS={sps}"
            )

            if wandb_run:
                log_dict = {
                    "train/policy_loss": avg_policy_loss,
                    "train/value_loss": avg_value_loss,
                    "train/entropy": avg_entropy,
                    "train/approx_kl": avg_kl,
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "train/sps": sps,
                    "train/mean_reward_20ep": mean_reward,
                    "global_step": global_step,
                }

                # Log spectral norms for HW-NODE
                spec_norms = _get_spectral_norms(model)
                for i, sn in enumerate(spec_norms):
                    log_dict[f"model/spectral_norm_A_{i}"] = sn

                wandb_run.log(log_dict, step=global_step)

    env.close()

    final_metrics = {
        "final_mean_reward": np.mean(recent_rewards[-100:]) if recent_rewards else 0.0,
        "total_episodes": episode_count,
        "total_steps": global_step,
        "param_count": total_params,
    }
    print(f"\n[done] Final mean reward (last 100 eps): {final_metrics['final_mean_reward']:.1f}")
    return final_metrics
