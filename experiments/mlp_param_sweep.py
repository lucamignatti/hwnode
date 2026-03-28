"""Find the MLP param budget yielding ~50% Pendulum reward.

Sweeps a coarse grid of hdim with 3 seeds each, then reports
the budget where mean reward crosses the halfway threshold.

Usage:
    PYTHONPATH=. python experiments/mlp_param_sweep.py --no-wandb
"""

import argparse
import numpy as np
import gymnasium as gym

from hwnode.baseline import MLPNetwork
from experiments.taylor_vs_chebyshev import FlexActorCritic, train_agent


def main():
    parser = argparse.ArgumentParser(description="Coarse MLP param budget sweep")
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    env_id = args.env
    target_reward = -800.0  # Pendulum episode range [-1600, 0]; halfway ≈ -800

    env_tmp = gym.make(env_id)
    obs_dim = env_tmp.observation_space.shape[0]
    continuous = isinstance(env_tmp.action_space, gym.spaces.Box)
    act_dim = env_tmp.action_space.shape[0] if continuous else env_tmp.action_space.n
    env_tmp.close()

    # Coarse grid — fill in between hdim=23 (too low) and hdim=35 (at target)
    hdim_grid = [20, 24, 28, 32, 36]

    print(f"\nCoarse sweep on {env_id}")
    print(f"hdim grid: {hdim_grid} | num_blocks={args.num_blocks}")
    print(f"Seeds: {args.num_seeds} | Budget: {args.max_seconds}s/run")
    print(f"Target reward: {target_reward}\n")

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
                label=f"mlp-h{hdim}-s{seed}",
            )
            rewards.append(res["final_mean_reward"])

        mu = np.mean(rewards)
        sd = np.std(rewards)
        results[hdim] = {"params": params, "mean": mu, "std": sd}
        marker = " <<<" if mu >= target_reward else ""
        print(
            f"  hdim={hdim:>3}  params={params:>6,}  reward={mu:>7.1f} ± {sd:<5.1f}{marker}"
        )

    # Summary
    print(f"\n{'=' * 55}")
    print(f" RESULTS")
    print(f"{'=' * 55}")
    for hdim, r in results.items():
        marker = " <<<" if r["mean"] >= target_reward else ""
        print(
            f"  hdim={hdim:>3}  params={r['params']:>6,}  {r['mean']:>7.1f} ± {r['std']:<5.1f}{marker}"
        )

    # Find threshold
    threshold = None
    for hdim in sorted(results):
        if results[hdim]["mean"] >= target_reward:
            threshold = hdim
            break

    print()
    if threshold:
        r = results[threshold]
        print(
            f"Target budget: ~{r['params']:,} params (hdim={threshold}, reward={r['mean']:.1f})"
        )
    else:
        print("Threshold not reached. Increase hdim range.")


if __name__ == "__main__":
    main()
