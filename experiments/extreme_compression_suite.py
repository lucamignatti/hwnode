"""Extreme Compression Evaluation Suite.

Specifically forces both MLPs and HW-NODEs to compete on complex geometries (BipedalWalker-v3) 
with an exact fixed parameter budget (~5.9k). Tests whether virtual depth scaling and 
Taylor polynomial degree optimizations natively generate intelligence gradients 
when standard physical parameters are completely starved.
"""

import argparse
import numpy as np

from hwnode.model import HWNodeNetwork
from hwnode.baseline import MLPNetwork
from experiments.taylor_vs_chebyshev import FlexActorCritic, train_agent
import gymnasium as gym

def main():
    parser = argparse.ArgumentParser(description="Strict Parameter Floor Compression Suite")
    # Switch to BipedalWalker-v3 to avoid LunarLander ceilings
    parser.add_argument("--env", type=str, default="BipedalWalker-v3")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=1_500_000)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    
    env_id = args.env
    total_timesteps = args.total_timesteps
    num_seeds = args.num_seeds
    use_wandb = not args.no_wandb

    env_tmp = gym.make(env_id)
    continuous = isinstance(env_tmp.action_space, gym.spaces.Box)
    act_dim = env_tmp.action_space.shape[0] if continuous else env_tmp.action_space.n
    obs_dim = env_tmp.observation_space.shape[0]
    env_tmp.close()

    master_results = []

    print(f"\n{'='*80}")
    print(f" STARTING EXTREME COMPRESSION SUITE ON: {env_id} | Seeds: {num_seeds}")
    print(f" Strict Parameter Constraint: ~5.9k Params")
    print(f" Budget allowed: {total_timesteps:,} steps")
    print(f"{'='*80}\n")

    # Name, BackboneClass, hidden_dim, state_dim, num_blocks (for MLP it's actual layers, for HW-NODE it's virtual layers), order
    variants = [
        # 1. Parity MLPs (Accounting for Actor + Critic dual-backbones + continuous heads)
        # ~5,926 Params (1 residual block containing 2 linear layer expansions, hdim=31)
        ("mlp-wide-shallow", MLPNetwork, 31, 0, 1, 0),     
        # ~5,914 Params (3 residual blocks containing 6 linear layer expansions, hdim=19)
        ("mlp-narrow-deep", MLPNetwork, 19, 0, 3, 0),      
        
        # 2. Parity HW-NODEs (All are identically 5,899 Params physically!)
        ("hwnode-o2-v2", HWNodeNetwork, 40, 18, 2, 2),
        ("hwnode-o2-v4", HWNodeNetwork, 40, 18, 4, 2),
        ("hwnode-o2-v6", HWNodeNetwork, 40, 18, 6, 2),
        ("hwnode-o2-v8", HWNodeNetwork, 40, 18, 8, 2),
        ("hwnode-o3-v8", HWNodeNetwork, 40, 18, 8, 3),    # Rescuing loose bounds
        ("hwnode-o4-v12", HWNodeNetwork, 40, 18, 12, 4),  # High precision, massive depth
        ("hwnode-o4-v16", HWNodeNetwork, 40, 18, 16, 4),  # Absolute limit test
    ]

    for seed in range(num_seeds):
        for name, BackboneClass, hdim, sdim, num_blocks, order in variants:
            print(f"\n--- Running {name} (seed {seed}) ---")
            
            wandb_run = None
            if use_wandb:
                import wandb
                wandb_run = wandb.init(
                    project="hwnode-extreme-compression",
                    name=f"{name}-{env_id}-s{seed}",
                    config={
                        "env": env_id, "seed": seed, "variant": name,
                        "hidden_dim": hdim, "state_dim": sdim,
                        "num_blocks_or_virtual": num_blocks, "order_or_layers": order,
                    },
                    reinit=True,
                )

            model = FlexActorCritic(
                obs_dim=obs_dim,
                act_dim=act_dim,
                BackboneClass=BackboneClass,
                continuous=continuous,
                hidden_dim=hdim,
                state_dim=sdim,
                num_blocks=num_blocks,
                order=order,
            )

            res = train_agent(
                env_id=env_id,
                model=model,
                seed=seed,
                total_timesteps=total_timesteps,
                wandb_run=wandb_run,
                label=name,
                env_kwargs={}  # Continuous strictly inferred by FlexActor Critic natively
            )
            
            master_results.append({
                "config_name": name,
                "env_id": env_id,
                "seed": seed,
                "final_mean_reward": res["final_mean_reward"],
                "param_count": res["param_count"]
            })

            if wandb_run:
                import wandb
                wandb.finish()

    # Reporting
    print(f"\n{'='*80}")
    print(f" EXTREME COMPRESSION EVALUATION ({env_id})")
    print(f"{'='*80}")
    print(f"{'Config Name':>20} | {'Params':>10} | {'Seed':>5} | {'Final Reward':>12}")
    print("-" * 60)
    for r in master_results:
        print(f"{r['config_name']:>20} | {r['param_count']:>10,} | {r['seed']:>5} | {r['final_mean_reward']:>12.1f}")
    
    print("-" * 60)
    print(f"{'AVERAGES':>20} | {'Params':>10} | {'Seeds':>5} | {'Mean ± Std':>15}")
    print("-" * 60)
    
    grouped = {}
    for r in master_results:
        grouped.setdefault(r["config_name"], []).append(r)
        
    for name, runs in grouped.items():
        rewards = [r["final_mean_reward"] for r in runs]
        params = runs[0]["param_count"]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        print(f"{name:>20} | {params:>10,} | {len(rewards):>5} | {mean_r:>8.1f} ± {std_r:<5.1f}")

if __name__ == "__main__":
    main()
