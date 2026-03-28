"""Find a stable MLP baseline at HW-NODE-matching param counts.

For hdim=state_dim=d with num_blocks=2, MLP and HW-NODE have the same
param count. This sweeps those matching dims with enough seeds to find
a stable config.

Usage:
    PYTHONPATH=. python experiments/mlp_param_sweep.py --no-wandb
"""

import argparse
import numpy as np
import gymnasium as gym

from hwnode.baseline import MLPNetwork
from experiments.taylor_vs_chebyshev import FlexActorCritic, train_agent


def main():
    parser = argparse.ArgumentParser(description="MLP baseline at matching dims")
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    env_id = args.env

    env_tmp = gym.make(env_id)
    obs_dim = env_tmp.observation_space.shape[0]
    continuous = isinstance(env_tmp.action_space, gym.spaces.Box)
    act_dim = env_tmp.action_space.shape[0] if continuous else env_tmp.action_space.n
    env_tmp.close()

    # hdim = state_dim = d values where MLP and HW-NODE match params.
    # Formula: MLP = 14d² + 20d + 2, HW-NODE = 12d² + 14d + 6
    # Also include d=32 (MLP reference from earlier stable run).
    hdim_grid = [16, 18, 20, 22, 24, 32]

    print(f"\nMatching-dim sweep on {env_id}")
    print(f"hdim grid: {hdim_grid} | num_blocks={args.num_blocks}")
    print(f"Seeds: {args.num_seeds} | Budget: {args.max_seconds}s/run\n")

    results = {}

    for hdim in hdim_grid:
        rewards = []
        params = 0
        for seed in range(args.num_seeds):
            model = FlexActorCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                BackboneClass=MLPNetwork,
                continuous=continuous,
                hidden_dim=hdim,
                num_blocks=args.num_blocks,
            )
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            res = train_agent(
                env_id=env_id,
                model=model,
                seed=seed,
                total_timesteps=10_000_000,
                max_wallclock_seconds=args.max_seconds,
                wandb_run=None,
                label=f"mlp-d{hdim}-s{seed}",
            )
            rewards.append(res["final_mean_reward"])

        mu = np.mean(rewards)
        sd = np.std(rewards)
        results[hdim] = {"params": params, "mean": mu, "std": sd}
        print(f"  d={hdim:>3}  params={params:>6,}  reward={mu:>7.1f} ± {sd:<5.1f}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f" RESULTS")
    print(f"{'=' * 60}")
    print(f"{'d':>4} | {'Params':>8} | {'Reward':>14} | {'Stable':>6}")
    print("-" * 42)
    for hdim in sorted(results):
        r = results[hdim]
        stable = "yes" if r["std"] < 100 else "no"
        print(
            f"{hdim:>4} | {r['params']:>8,} | {r['mean']:>7.1f} ± {r['std']:<5.1f} | {stable:>6}"
        )

    # Recommend the best stable config
    stable = [(h, r) for h, r in results.items() if r["std"] < 100]
    if stable:
        best = max(stable, key=lambda x: x[1]["mean"])
        print(
            f"\nBest stable config: d={best[0]}, params={best[1]['params']:,}, "
            f"reward={best[1]['mean']:.1f} ± {best[1]['std']:.1f}"
        )
    else:
        print("\nNo stable configs found (all std > 100).")


if __name__ == "__main__":
    main()
