"""Compare HW-NODE variants against MLP at matched parameter budget on Pendulum-v1.

Fixed: ~9k params, 180s wall clock per run, 5 seeds.

Usage:
    PYTHONPATH=. python experiments/pendulum_comparison.py --no-wandb
    PYTHONPATH=. python experiments/pendulum_comparison.py --max-seconds 300 --num-seeds 10 --no-wandb
"""

import argparse
import numpy as np
import torch
import gymnasium as gym

from hwnode.model import HWNodeNetwork
from hwnode.baseline import MLPNetwork
from experiments.taylor_vs_chebyshev import FlexActorCritic, train_agent


def main():
    parser = argparse.ArgumentParser(description="HW-NODE vs MLP at matched params")
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    env_id = args.env

    env_tmp = gym.make(env_id)
    obs_dim = env_tmp.observation_space.shape[0]
    continuous = isinstance(env_tmp.action_space, gym.spaces.Box)
    act_dim = env_tmp.action_space.shape[0] if continuous else env_tmp.action_space.n
    env_tmp.close()

    # All configs designed to hit ~9k total params (actor + critic)
    configs = [
        ("mlp-baseline", MLPNetwork, dict(hidden_dim=32, num_blocks=2)),
        (
            "hwnode-shallow",
            HWNodeNetwork,
            dict(hidden_dim=32, state_dim=24, num_blocks=1, order=4),
        ),
        (
            "hwnode-medium",
            HWNodeNetwork,
            dict(hidden_dim=30, state_dim=22, num_blocks=3, order=4),
        ),
        (
            "hwnode-deep",
            HWNodeNetwork,
            dict(hidden_dim=28, state_dim=20, num_blocks=5, order=4),
        ),
        (
            "hwnode-high-order",
            HWNodeNetwork,
            dict(hidden_dim=32, state_dim=16, num_blocks=2, order=8),
        ),
    ]

    # Verify param counts
    print(f"\nConfig verification:")
    for name, Backbone, kwargs in configs:
        model = FlexActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            BackboneClass=Backbone,
            continuous=continuous,
            **kwargs,
        )
        p = sum(x.numel() for x in model.parameters() if x.requires_grad)
        print(f"  {name:>22}  {p:>6,} params  {kwargs}")

    print(f"\n{'=' * 60}")
    print(f" Pendulum-v1 comparison")
    print(f" Budget: {args.max_seconds}s wall clock | Seeds: {args.num_seeds}")
    print(f"{'=' * 60}\n")

    all_results = []

    for name, Backbone, kwargs in configs:
        rewards = []
        for seed in range(args.num_seeds):
            model = FlexActorCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                BackboneClass=Backbone,
                continuous=continuous,
                **kwargs,
            )

            res = train_agent(
                env_id=env_id,
                model=model,
                seed=seed,
                total_timesteps=10_000_000,
                max_wallclock_seconds=args.max_seconds,
                wandb_run=None,
                label=f"{name}-s{seed}",
            )
            rewards.append(res["final_mean_reward"])

        params = sum(x.numel() for x in model.parameters() if x.requires_grad)
        mu, sd = np.mean(rewards), np.std(rewards)
        all_results.append({"name": name, "params": params, "mean": mu, "std": sd})
        print(f"  {name:>22}  {params:>6,}  {mu:>7.1f} ± {sd:<5.1f}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f" RESULTS")
    print(f"{'=' * 60}")
    print(f"{'Config':>22} | {'Params':>8} | {'Reward':>14}")
    print("-" * 52)
    for r in all_results:
        print(
            f"{r['name']:>22} | {r['params']:>8,} | {r['mean']:>7.1f} ± {r['std']:<5.1f}"
        )


if __name__ == "__main__":
    main()
