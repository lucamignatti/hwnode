"""Find the MLP parameter budget that yields ~50% Pendulum-v1 performance.

Uses wall clock time as the compute budget so that different architectures
(including high virtual-depth HW-NODE configs) compete on equal footing.

Usage:
    PYTHONPATH=. python experiments/mlp_param_sweep.py --no-wandb
    PYTHONPATH=. python experiments/mlp_param_sweep.py --max-seconds 300 --num-seeds 3 --no-wandb
"""

import argparse
import itertools
import numpy as np
import gymnasium as gym

from hwnode.baseline import MLPNetwork
from experiments.taylor_vs_chebyshev import FlexActorCritic, train_agent


def main():
    parser = argparse.ArgumentParser(
        description="MLP param budget sweep on Pendulum-v1"
    )
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--min-params", type=int, default=1000)
    parser.add_argument("--max-params", type=int, default=15000)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    env_id = args.env
    target_reward = -8.1  # Pendulum range [-16.27, 0]; 50% ≈ -8.1

    # Detect env dimensions
    env_tmp = gym.make(env_id)
    obs_dim = env_tmp.observation_space.shape[0]
    continuous = isinstance(env_tmp.action_space, gym.spaces.Box)
    act_dim = env_tmp.action_space.shape[0] if continuous else env_tmp.action_space.n
    env_tmp.close()

    def mlp_params(hdim, num_blocks):
        """Estimate total param count for actor + critic MLP pair."""
        embed = obs_dim * hdim + hdim
        layer = 2 * (hdim * hdim + hdim)
        norm = 2 * hdim
        per_backbone = embed + num_blocks * layer + norm + norm
        return 2 * per_backbone + (hdim + 1) + (hdim + 1)

    # Build grid
    configs = []
    for hdim, nb in itertools.product(range(6, 41), [1, 2, 3, 4]):
        p = mlp_params(hdim, nb)
        if args.min_params <= p <= args.max_params:
            configs.append((hdim, nb, p))
    configs.sort(key=lambda x: x[2])

    if not configs:
        print("No configs in param range.")
        return

    print(f"\nMLP param sweep on {env_id}")
    print(
        f"Configs: {len(configs)} | Seeds: {args.num_seeds} | Budget: {args.max_seconds}s wall clock"
    )
    print(
        f"Param range: {args.min_params:,}–{args.max_params:,} | Target reward: {target_reward}\n"
    )

    results = []

    for hdim, nb, est in configs:
        for seed in range(args.num_seeds):
            label = f"mlp-h{hdim}-b{nb}"
            model = FlexActorCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                BackboneClass=MLPNetwork,
                continuous=continuous,
                hidden_dim=hdim,
                num_blocks=nb,
            )

            res = train_agent(
                env_id=env_id,
                model=model,
                seed=seed,
                total_timesteps=10_000_000,  # ceiling, wall clock stops it first
                max_wallclock_seconds=args.max_seconds,
                wandb_run=None,
                label=label,
            )

            results.append(
                {
                    "hdim": hdim,
                    "nb": nb,
                    "params": res["param_count"],
                    "seed": seed,
                    "reward": res["final_mean_reward"],
                }
            )

    # Aggregate
    print(f"\n{'=' * 65}")
    print(f" RESULTS  ({env_id}, {args.max_seconds}s budget)")
    print(f"{'=' * 65}")
    print(f"{'hdim':>5} {'blocks':>6} {'params':>8} | {'reward':>14}")
    print("-" * 45)

    by_p = {}
    for r in results:
        by_p.setdefault(r["params"], []).append(r)

    sorted_ps = sorted(by_p)
    threshold_met = None

    for p in sorted_ps:
        runs = by_p[p]
        hdim = runs[0]["hdim"]
        nb = runs[0]["nb"]
        rews = [r["reward"] for r in runs]
        mu = np.mean(rews)
        sd = np.std(rews)

        marker = ""
        if threshold_met is None and mu >= target_reward:
            threshold_met = p
            marker = " <<< 50%"

        print(f"{hdim:>5} {nb:>6} {p:>8,} | {mu:>7.1f} ± {sd:<5.1f}{marker}")

    print()
    if threshold_met:
        r = by_p[threshold_met][0]
        mu = np.mean([x["reward"] for x in by_p[threshold_met]])
        print(
            f"Target budget: ~{threshold_met:,} params (hdim={r['hdim']}, blocks={r['nb']}, reward={mu:.1f})"
        )
    else:
        print("50% threshold not reached in this param range.")


if __name__ == "__main__":
    main()
