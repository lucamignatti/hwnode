"""Master Evaluation Suite.

Runs all baseline MLP configs, standard HW-NODE configs, and the newly 
scaled up Taylor/Chebyshev representation variants on the same environments
to provide an apples-to-apples comparison of performance vs. parameter count.
"""

import argparse
import itertools
import numpy as np

# Import original runner functions
from hwnode.run import build_config, run_single

# Import the new experimental architectures and train loop
from experiments.taylor_vs_chebyshev import (
    train_agent,
    FlexActorCritic,
    HWNodeNetwork,
    LearnedTaylorHWNodeNetwork,
    ChebyshevHWNodeNetwork
)
import gymnasium as gym

def _make_cheb_ortho_init(**kwargs):
    return ChebyshevHWNodeNetwork(**kwargs, a_init="orthogonal")

def _make_cheb_ortho_param(**kwargs):
    return ChebyshevHWNodeNetwork(**kwargs, a_constraint="orthogonal")

def main():
    parser = argparse.ArgumentParser(description="Full apples-to-apples evaluation suite")
    parser.add_argument("--env", type=str, default="LunarLander-v3")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--only-scaled", action="store_true", help="Run only the massive experimental scales")
    args = parser.parse_args()
    
    env_id = args.env
    total_timesteps = args.total_timesteps
    num_seeds = args.num_seeds
    use_wandb = not args.no_wandb

    # Detect if continuous
    env_tmp = gym.make(env_id)
    continuous = isinstance(env_tmp.action_space, gym.spaces.Box)
    act_dim = env_tmp.action_space.shape[0] if continuous else env_tmp.action_space.n
    obs_dim = env_tmp.observation_space.shape[0]
    env_tmp.close()

    master_results = []

    print(f"\n{'='*80}")
    print(f" STARTING FULL EVALUATION SUITE ON: {env_id} | Seeds: {num_seeds}")
    print(f"{'='*80}\n")

    # 1. Baseline MLPs via hwnode.train (standard run.py mechanism)
    baseline_configs = [
        ("mlp-narrow", "mlp", 32, 0, 2),
        ("mlp-medium", "mlp", 64, 0, 2),
        ("mlp-large",  "mlp", 128, 0, 2),
    ]

    if not args.only_scaled:
        for seed in range(num_seeds):
            for name, backbone, hdim, sdim, num_blocks in baseline_configs:
                # Fake parsing args for build_config
                class FakeArgs:
                    pass
                args_mock = FakeArgs()
                args_mock.env = env_id
                args_mock.agent = backbone
                args_mock.hidden_dim = hdim
                args_mock.state_dim = sdim
                args_mock.num_blocks = num_blocks
                args_mock.order = 4
                args_mock.activation = "relu_squared"
                args_mock.lr = 3e-4
                args_mock.total_timesteps = total_timesteps
                args_mock.rollout_steps = 2048
                args_mock.num_epochs = 4
                args_mock.batch_size = 64
                args_mock.seed = seed
                args_mock.num_seeds = 1
                args_mock.wandb_project = "hwnode-full-eval"
                args_mock.no_wandb = args.no_wandb
                args_mock.device = "auto"
                    
                cfg = build_config(args_mock, seed_override=seed)
                metrics = run_single(cfg)
                metrics["config_name"] = name
                metrics["env_id"] = env_id
                metrics["seed"] = seed
                master_results.append(metrics)

    # 2. Scaled-up HW-NODE variants via experiments.taylor_vs_chebyshev mechanism
    if not args.only_scaled:
        variants = [
            ("hwnode-standard-fixed", HWNodeNetwork, 64, 16, 2, 4),
            ("taylor-learned", LearnedTaylorHWNodeNetwork, 64, 16, 2, 4),
            ("chebyshev-learned", ChebyshevHWNodeNetwork, 64, 16, 2, 4),
            ("cheb-ortho-init", _make_cheb_ortho_init, 64, 16, 2, 4),
            ("cheb-ortho-param", _make_cheb_ortho_param, 64, 16, 2, 4),
            ("hwnode-scaled", HWNodeNetwork, 128, 32, 3, 6),
            ("chebyshev-scaled", ChebyshevHWNodeNetwork, 128, 32, 3, 6),
        ]
    else:
        # Massive scaled architectures to thoroughly push the weight-tying limits against Chebyshev
        variants = [
            ("hwnode-xl", HWNodeNetwork, 256, 64, 4, 8),
            ("chebyshev-xl", ChebyshevHWNodeNetwork, 256, 64, 4, 8),
            ("hwnode-xxl", HWNodeNetwork, 512, 128, 4, 8),
            ("chebyshev-xxl", ChebyshevHWNodeNetwork, 512, 128, 4, 8),
        ]

    for seed in range(num_seeds):
        for name, BackboneClass, hdim, sdim, num_blocks, order in variants:
            print(f"\n--- Running {name} (seed {seed}) ---")
            
            wandb_run = None
            if use_wandb:
                import wandb
                wandb_run = wandb.init(
                    project="hwnode-full-eval",
                    name=f"{name}-{env_id}-s{seed}",
                    config={
                        "env": env_id, "seed": seed, "variant": name,
                        "hidden_dim": hdim, "state_dim": sdim,
                        "num_blocks": num_blocks, "order": order,
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
                env_kwargs={"continuous": True} if continuous and "LunarLander" in env_id else {}
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

    # Summarize Apples-to-Apples
    print(f"\n{'='*80}")
    print(f" APPLES-TO-APPLES EVALUATION SUMMARY ({env_id})")
    print(f"{'='*80}")
    print(f"{'Config Name':>25} | {'Params':>10} | {'Seed':>5} | {'Final Reward':>12}")
    print("-" * 65)
    for r in master_results:
        print(f"{r['config_name']:>25} | {r['param_count']:>10,} | {r['seed']:>5} | {r['final_mean_reward']:>12.1f}")
    
    print("-" * 65)
    print(f"{'AVERAGES':>25} | {'Params':>10} | {'Seeds':>5} | {'Mean ± Std':>15}")
    print("-" * 65)
    
    # Group by name
    grouped = {}
    for r in master_results:
        grouped.setdefault(r["config_name"], []).append(r)
        
    for name, runs in grouped.items():
        rewards = [r["final_mean_reward"] for r in runs]
        params = runs[0]["param_count"]
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        print(f"{name:>25} | {params:>10,} | {len(rewards):>5} | {mean_r:>8.1f} ± {std_r:<5.1f}")


if __name__ == "__main__":
    main()
