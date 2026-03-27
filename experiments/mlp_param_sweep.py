"""Binary search for the MLP param budget yielding ~50% Pendulum reward.

Narrows hdim (num_blocks=2) with 1 seed each, then validates with 2 more.

Usage:
    PYTHONPATH=. python experiments/mlp_param_sweep.py --no-wandb
"""

import argparse
import numpy as np
import gymnasium as gym

from hwnode.baseline import MLPNetwork
from experiments.taylor_vs_chebyshev import FlexActorCritic, train_agent


def main():
    parser = argparse.ArgumentParser(description="Binary search MLP param budget")
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-seeds", type=int, default=2)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    env_id = args.env
    target_reward = -8.1  # Pendulum: [-16.27, 0], 50% ≈ -8.1

    env_tmp = gym.make(env_id)
    obs_dim = env_tmp.observation_space.shape[0]
    continuous = isinstance(env_tmp.action_space, gym.spaces.Box)
    act_dim = env_tmp.action_space.shape[0] if continuous else env_tmp.action_space.n
    env_tmp.close()

    def run_one(hdim, seed):
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
            label=f"mlp-h{hdim}",
        )
        return res["final_mean_reward"], params

    # --- Binary search phase ---
    lo, hi = 6, 40
    search_log = []

    print(f"\nBinary search: hdim [{lo}, {hi}], num_blocks={args.num_blocks}")
    print(f"Budget: {args.max_seconds}s/run | Target reward: {target_reward}")
    print(f"Env: {env_id}\n")

    while hi - lo > 1:
        mid = (lo + hi) // 2
        reward, params = run_one(mid, args.seed)
        search_log.append((mid, reward, params))
        marker = " above" if reward >= target_reward else " below"
        print(f"  hdim={mid:>3}  params={params:>6,}  reward={reward:>7.1f}{marker}")

        if reward >= target_reward:
            hi = mid
        else:
            lo = mid

    # lo is below, hi is at/above. Pick hi as the threshold config.
    print(f"\nSearch done: threshold between hdim={lo} and hdim={hi}")

    # --- Validation phase ---
    print(f"\nValidating hdim={hi} with {args.validation_seeds} more seeds...")
    rewards = []
    params = 0
    for i in range(args.validation_seeds):
        seed = args.seed + 1 + i  # skip the search seed
        r, p = run_one(hi, seed)
        rewards.append(r)
        params = p
        print(f"  seed {seed}: reward={r:>7.1f}")

    mu = np.mean(rewards)
    sd = np.std(rewards)
    print(f"\n{'=' * 50}")
    print(f" RESULT")
    print(f"{'=' * 50}")
    print(f"  hdim={hi}, num_blocks={args.num_blocks}, params={params:,}")
    print(f"  Reward: {mu:.1f} ± {sd:.1f}")
    print(f"  Target: {target_reward}")
    print(f"  Budget: {args.max_seconds}s wall clock per run")


if __name__ == "__main__":
    main()
