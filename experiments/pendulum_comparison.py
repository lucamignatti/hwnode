"""1:1 comparison: HW-NODE vs MLP on Pendulum-v1.

MLP: hdim=32, num_blocks=2 → ~9,155 params.
HW-NODE: hdim=32, state_dim=32, num_blocks=3 → ~9,670 params.
Vary virtual_depth internally for HW-NODE variants.

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

    HDIM = 18
    SDIM = 18

    configs = [
        ("mlp", MLPNetwork, dict(hidden_dim=HDIM, num_blocks=2)),
    ]

    for vdepth in [2, 4, 6, 12]:
        for order in [2, 3]:
            configs.append(
                (
                    f"hwnode-v{vdepth}-o{order}",
                    HWNodeNetwork,
                    dict(
                        hidden_dim=HDIM,
                        state_dim=SDIM,
                        num_blocks=2,
                        order=order,
                        virtual_depth=vdepth,
                    ),
                )
            )

    # Verify param counts
    print(f"\n{'Config':>20} | {'Params':>8}")
    print("-" * 35)
    for name, Backbone, kwargs in configs:
        model = FlexActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            BackboneClass=Backbone,
            continuous=continuous,
            **kwargs,
        )
        p = sum(x.numel() for x in model.parameters() if x.requires_grad)
        print(f"{name:>20} | {p:>8,}")

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
        all_results.append({"name": name, "params": params, "mean": mu, "std": sd})
        print(f"  {name:>20}  {params:>6,} params  {mu:>7.1f} ± {sd:<5.1f}")

    # Summary
    mlp_r = next(r for r in all_results if r["name"] == "mlp")
    print(f"\n{'=' * 60}")
    print(f" RESULTS")
    print(f"{'=' * 60}")
    print(
        f"  {'mlp':>20}  {mlp_r['params']:>6,} params  {mlp_r['mean']:>7.1f} ± {mlp_r['std']:<5.1f}"
    )
    print(f"{'-' * 60}")
    for r in sorted(all_results, key=lambda x: x["mean"], reverse=True):
        if r["name"] == "mlp":
            continue
        delta = r["mean"] - mlp_r["mean"]
        sign = "+" if delta >= 0 else ""
        print(
            f"  {r['name']:>20}  {r['params']:>6,} params  {r['mean']:>7.1f} ± {r['std']:<5.1f}  ({sign}{delta:.1f})"
        )


if __name__ == "__main__":
    main()
