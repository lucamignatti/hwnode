"""Find a stable MLP baseline at HW-NODE-matching param counts.

3 seeds per config. Early stopping: skip 3rd seed if first 2 have huge
variance. Stop searching if a config is stable and above -800.

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
    parser.add_argument("--max-seeds", type=int, default=3)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    env_id = args.env
    HALF_WAY = -800.0

    env_tmp = gym.make(env_id)
    obs_dim = env_tmp.observation_space.shape[0]
    continuous = isinstance(env_tmp.action_space, gym.spaces.Box)
    act_dim = env_tmp.action_space.shape[0] if continuous else env_tmp.action_space.n
    env_tmp.close()

    # Matching dims where MLP and HW-NODE have equal params (hdim=state_dim=d, 2 blocks).
    hdim_grid = [16, 18, 20, 22, 24, 32]

    print(f"\nMatching-dim sweep on {env_id}")
    print(
        f"hdim grid: {hdim_grid} | max seeds: {args.max_seeds} | budget: {args.max_seconds}s/run"
    )
    print(f"Stable = std < 100 | Early stop if stable + reward > {HALF_WAY}\n")

    results = {}

    for hdim in hdim_grid:
        rewards = []
        params = 0
        for seed in range(args.max_seeds):
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

            # After 2 seeds: check if variance is hopeless
            if len(rewards) == 2 and abs(rewards[0] - rewards[1]) > 500:
                print(
                    f"  d={hdim:>3} s{seed}: {rewards[-1]:>7.1f}  (high variance, skipping seed 3)"
                )
                break

        mu = np.mean(rewards)
        sd = np.std(rewards)
        stable = sd < 100
        results[hdim] = {"params": params, "mean": mu, "std": sd, "seeds": len(rewards)}
        tag = "stable" if stable else "unstable"
        print(
            f"  d={hdim:>3}  params={params:>6,}  {mu:>7.1f} ± {sd:<5.1f}  [{tag}, {len(rewards)} seeds]"
        )

        # Early stop: found a stable config above halfway
        if stable and mu >= HALF_WAY:
            print(f"\n  Found stable winner at d={hdim}. Stopping sweep.")
            break

    # Summary
    print(f"\n{'=' * 60}")
    print(f" RESULTS")
    print(f"{'=' * 60}")
    for hdim in sorted(results):
        r = results[hdim]
        tag = "yes" if r["std"] < 100 else "no"
        print(
            f"  d={hdim:>3}  {r['params']:>6,} params  {r['mean']:>7.1f} ± {r['std']:<5.1f}  stable={tag}"
        )

    stable = [(h, r) for h, r in results.items() if r["std"] < 100]
    if stable:
        best = max(stable, key=lambda x: x[1]["mean"])
        print(
            f"\nBest: d={best[0]}, {best[1]['params']:,} params, {best[1]['mean']:.1f} ± {best[1]['std']:.1f}"
        )
    else:
        print("\nNo stable configs found.")


if __name__ == "__main__":
    main()
