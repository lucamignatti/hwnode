"""1:1 comparison: HW-NODE vs MLP on Pendulum-v1.

Same hdim, same state_dim, same num_blocks. HW-NODE uses weight sharing
across virtual layers so it has fewer params — that's the point.

Usage:
    PYTHONPATH=. python experiments/pendulum_comparison.py --no-wandb
"""

import argparse
import numpy as np
import gymnasium as gym

from hwnode.model import HWNodeNetwork
from hwnode.baseline import MLPNetwork
from experiments.taylor_vs_chebyshev import FlexActorCritic, train_agent


def main():
    parser = argparse.ArgumentParser(description="HW-NODE vs MLP 1:1 comparison")
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

    HDIM = 32
    SDIM = 32  # same as hdim for 1:1 fair comparison

    # num_blocks = virtual depth for HW-NODE, physical layers for MLP.
    depths = [1, 2, 4, 8]

    configs = []
    for d in depths:
        configs.append((f"mlp-d{d}", MLPNetwork, dict(hidden_dim=HDIM, num_blocks=d)))
        configs.append(
            (
                f"hwnode-d{d}",
                HWNodeNetwork,
                dict(hidden_dim=HDIM, state_dim=SDIM, num_blocks=d, order=4),
            )
        )

    # Verify param counts
    print(f"\nhdim={HDIM}, state_dim={SDIM}")
    print(f"\n{'Config':>18} | {'Params':>8} | {'depth':>5}")
    print("-" * 40)
    for name, Backbone, kwargs in configs:
        model = FlexActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            BackboneClass=Backbone,
            continuous=continuous,
            **kwargs,
        )
        p = sum(x.numel() for x in model.parameters() if x.requires_grad)
        print(f"{name:>18} | {p:>8,} | {kwargs['num_blocks']:>5}")

    print(f"\n{'=' * 60}")
    print(f" Pendulum-v1 | Budget: {args.max_seconds}s | Seeds: {args.num_seeds}")
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
        all_results.append(
            {
                "name": name,
                "params": params,
                "depth": kwargs["num_blocks"],
                "mean": mu,
                "std": sd,
            }
        )
        print(f"  {name:>18}  {params:>6,} params  {mu:>7.1f} ± {sd:<5.1f}")

    # Summary grouped by depth
    print(f"\n{'=' * 60}")
    print(f" RESULTS")
    print(f"{'=' * 60}")
    for d in depths:
        mlp = next(r for r in all_results if r["name"] == f"mlp-d{d}")
        hw = next(r for r in all_results if r["name"] == f"hwnode-d{d}")
        delta = hw["mean"] - mlp["mean"]
        sign = "+" if delta >= 0 else ""
        print(f"\n  depth={d}:")
        print(
            f"    mlp:    {mlp['params']:>6,} params  {mlp['mean']:>7.1f} ± {mlp['std']:<5.1f}"
        )
        print(
            f"    hwnode: {hw['params']:>6,} params  {hw['mean']:>7.1f} ± {hw['std']:<5.1f}  ({sign}{delta:.1f} vs MLP)"
        )


if __name__ == "__main__":
    main()
