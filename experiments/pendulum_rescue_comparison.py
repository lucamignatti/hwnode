"""Focused Pendulum rescue sweep for HW-NODE.

This suite compares the strongest baseline against a few concrete rescue
hypotheses that came out of debugging:

1. Keep the original HW-NODE shape as the control.
2. Remove the squared Wiener output ("nosquare"), which helped in short runs.
3. Spend the parameter budget on width instead of recurrence ("shallowwide"),
   which is much faster at nearly the same parameter count.

Usage:
    PYTHONPATH=. python experiments/pendulum_rescue_comparison.py --no-wandb
"""

import argparse
import numpy as np
import gymnasium as gym

from hwnode.baseline import MLPNetwork
from hwnode.model import HWNodeNetwork
from experiments.taylor_vs_chebyshev import FlexActorCritic, train_agent


def main():
    parser = argparse.ArgumentParser(description="Pendulum rescue comparison")
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    env_id = args.env

    env_tmp = gym.make(env_id)
    obs_dim = env_tmp.observation_space.shape[0]
    continuous = isinstance(env_tmp.action_space, gym.spaces.Box)
    act_dim = env_tmp.action_space.shape[0] if continuous else env_tmp.action_space.n
    env_tmp.close()

    configs = [
        ("mlp", MLPNetwork, dict(hidden_dim=32, num_blocks=2)),
        (
            "hwnode-v2-o2",
            HWNodeNetwork,
            dict(
                hidden_dim=22,
                state_dim=22,
                num_blocks=3,
                order=2,
                virtual_depth=2,
                activation="relu_squared",
            ),
        ),
        (
            "hwnode-v2-o2-nosquare",
            HWNodeNetwork,
            dict(
                hidden_dim=22,
                state_dim=22,
                num_blocks=3,
                order=2,
                virtual_depth=2,
                activation="relu",
            ),
        ),
        (
            "hwnode-shallowwide-o1-v1",
            HWNodeNetwork,
            dict(
                hidden_dim=22,
                state_dim=48,
                num_blocks=1,
                order=1,
                virtual_depth=1,
                activation="relu",
            ),
        ),
    ]

    print(f"\n{'Config':>28} | {'Params':>8}")
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
        print(f"{name:>28} | {p:>8,}")

    print(f"\n{'=' * 60}")
    print(f" {env_id} | Budget: {args.max_seconds}s | Seeds: {args.num_seeds}")
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
        print(f"  {name:>28}  {params:>6,} params  {mu:>7.1f} ± {sd:<5.1f}")

    mlp_r = next(r for r in all_results if r["name"] == "mlp")
    print(f"\n{'=' * 60}")
    print(" RESULTS")
    print(f"{'=' * 60}")
    print(
        f"  {'mlp':>28}  {mlp_r['params']:>6,} params  {mlp_r['mean']:>7.1f} ± {mlp_r['std']:<5.1f}"
    )
    print(f"{'-' * 60}")
    for r in sorted(all_results, key=lambda x: x["mean"], reverse=True):
        if r["name"] == "mlp":
            continue
        delta = r["mean"] - mlp_r["mean"]
        sign = "+" if delta >= 0 else ""
        print(
            f"  {r['name']:>28}  {r['params']:>6,} params  {r['mean']:>7.1f} ± {r['std']:<5.1f}  ({sign}{delta:.1f})"
        )


if __name__ == "__main__":
    main()
