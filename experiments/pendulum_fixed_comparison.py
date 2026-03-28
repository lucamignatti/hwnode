"""Fixed-compute, fixed-parameter Pendulum comparison with corrected HW-NODE semantics.

This uses the existing 600s / multi-seed test setup, but evaluates the
corrected compositional virtual-layer HW-NODE implementation instead of the
earlier buggy residual virtual-depth behavior.

Included configs:
    - `mlp`: reference baseline from the original experiment
    - `hwnode-fixed-v2-o2-sq`: corrected virtual-depth semantics, squared Wiener output
    - `hwnode-fixed-v2-o2-relu`: corrected virtual-depth semantics, non-squared Wiener output
    - `hwnode-fixed-shallowwide`: spends the same parameter budget on width instead of recurrence

Usage:
    PYTHONPATH=. python experiments/pendulum_fixed_comparison.py --no-wandb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from experiments.taylor_vs_chebyshev import FlexActorCritic, train_agent
from hwnode.baseline import MLPNetwork
from hwnode.model import HWNodeNetwork


def _build_configs():
    return [
        ("mlp", MLPNetwork, dict(hidden_dim=32, num_blocks=2)),
        (
            "hwnode-fixed-v2-o2-sq",
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
            "hwnode-fixed-v2-o2-relu",
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
            "hwnode-fixed-shallowwide",
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Corrected HW-NODE Pendulum comparison")
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    env_id = args.env

    env_tmp = gym.make(env_id)
    obs_dim = env_tmp.observation_space.shape[0]
    continuous = isinstance(env_tmp.action_space, gym.spaces.Box)
    act_dim = env_tmp.action_space.shape[0] if continuous else env_tmp.action_space.n
    env_tmp.close()

    configs = _build_configs()

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
        params = sum(x.numel() for x in model.parameters() if x.requires_grad)
        print(f"{name:>28} | {params:>8,}")

    print(f"\n{'=' * 60}")
    print(f" {env_id} | Budget: {args.max_seconds}s | Seeds: {args.num_seeds}")
    print(f"{'=' * 60}\n")

    all_results = []
    raw_runs = []

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
            raw_runs.append(
                {
                    "config": name,
                    "seed": seed,
                    "final_mean_reward": res["final_mean_reward"],
                    "param_count": res["param_count"],
                }
            )

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

    if args.json_out:
        payload = {
            "env": env_id,
            "max_seconds": args.max_seconds,
            "num_seeds": args.num_seeds,
            "summary": all_results,
            "runs": raw_runs,
        }
        out_path = Path(args.json_out).expanduser()
        _write_json(out_path, payload)
        print(f"\nSaved JSON results to {out_path}")


if __name__ == "__main__":
    main()
