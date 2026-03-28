"""Compare HW-NODE variants against MLP at matched hdim on Pendulum-v1.

All configs use hdim=32. HW-NODE naturally has ~half the params due to
weight sharing. Tests whether the inductive bias compensates.

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
    parser = argparse.ArgumentParser(description="HW-NODE vs MLP at matched hdim")
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

    # All hdim=32. HW-NODE variants explore virtual depth × bottleneck × Taylor order.
    configs = [
        ("mlp-baseline", MLPNetwork, dict(hidden_dim=32, num_blocks=2)),
        (
            "hwnode-wide-shallow",
            HWNodeNetwork,
            dict(hidden_dim=32, state_dim=32, num_blocks=1, order=4),
        ),
        (
            "hwnode-wide-deep",
            HWNodeNetwork,
            dict(hidden_dim=32, state_dim=32, num_blocks=6, order=4),
        ),
        (
            "hwnode-mid-shallow",
            HWNodeNetwork,
            dict(hidden_dim=32, state_dim=24, num_blocks=1, order=4),
        ),
        (
            "hwnode-mid-deep",
            HWNodeNetwork,
            dict(hidden_dim=32, state_dim=24, num_blocks=6, order=4),
        ),
        (
            "hwnode-tight-deep",
            HWNodeNetwork,
            dict(hidden_dim=32, state_dim=16, num_blocks=8, order=4),
        ),
        (
            "hwnode-high-order",
            HWNodeNetwork,
            dict(hidden_dim=32, state_dim=16, num_blocks=2, order=8),
        ),
    ]

    # Verify param counts
    print(
        f"\n{'Config':>22} | {'Params':>8} | {'sdim':>4} | {'depth':>5} | {'order':>5}"
    )
    print("-" * 55)
    for name, Backbone, kwargs in configs:
        model = FlexActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            BackboneClass=Backbone,
            continuous=continuous,
            **kwargs,
        )
        p = sum(x.numel() for x in model.parameters() if x.requires_grad)
        print(
            f"{name:>22} | {p:>8,} | {kwargs.get('state_dim', '—'):>4} | "
            f"{kwargs['num_blocks']:>5} | {kwargs.get('order', '—'):>5}"
        )

    print(f"\n{'=' * 60}")
    print(f" Pendulum-v1 | Budget: {args.max_seconds}s | Seeds: {args.num_seeds}")
    print(f"{'=' * 60}\n")

    all_results = []

    for name, Backbone, kwargs in configs:
        rewards = []
        # Compute params once (same for all seeds)
        _ref = FlexActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            BackboneClass=Backbone,
            continuous=continuous,
            **kwargs,
        )
        params = sum(x.numel() for x in _ref.parameters() if x.requires_grad)
        del _ref

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
        mu, sd = np.mean(rewards), np.std(rewards)
        all_results.append({"name": name, "params": params, "mean": mu, "std": sd})
        print(f"  {name:>22}  {params:>6,} params  {mu:>7.1f} ± {sd:<5.1f}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f" RESULTS (sorted by reward)")
    print(f"{'=' * 60}")
    print(f"{'Config':>22} | {'Params':>8} | {'vs MLP':>8} | {'Reward':>14}")
    print("-" * 60)
    mlp_reward = next(r["mean"] for r in all_results if r["name"] == "mlp-baseline")
    for r in sorted(all_results, key=lambda x: x["mean"], reverse=True):
        delta = r["mean"] - mlp_reward
        sign = "+" if delta >= 0 else ""
        print(
            f"{r['name']:>22} | {r['params']:>8,} | {sign}{delta:>6.1f} | {r['mean']:>7.1f} ± {r['std']:<5.1f}"
        )


if __name__ == "__main__":
    main()
