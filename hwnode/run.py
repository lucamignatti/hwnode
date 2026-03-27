"""CLI entry point for HW-NODE experiments."""

from __future__ import annotations

import argparse
import itertools
import sys

from hwnode.config import ExperimentConfig, ModelConfig, PPOConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HW-NODE: Hammerstein-Wiener Neural ODE RL Experiments"
    )

    # Mode
    p.add_argument("--sweep", action="store_true", help="Run original experiment matrix")
    p.add_argument("--sweep-v2", action="store_true", help="Run v2: tiny params + continuous control")

    # Environment
    p.add_argument("--env", type=str, default="CartPole-v1")

    # Model
    p.add_argument("--agent", type=str, default="hwnode", choices=["hwnode", "mlp"])
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--state-dim", type=int, default=16)
    p.add_argument("--num-blocks", type=int, default=2)
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--activation", type=str, default="relu_squared")

    # PPO
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--num-epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)

    # Experiment
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-seeds", type=int, default=5)
    p.add_argument("--wandb-project", type=str, default="hwnode-research")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--device", type=str, default="auto")

    return p.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # MPS lacks aten::vdot needed by spectral_norm — default to CPU.
        # Use --device mps with PYTORCH_ENABLE_MPS_FALLBACK=1 to force MPS.
        return "cpu"
    return requested


def build_config(args: argparse.Namespace, seed_override: int | None = None) -> ExperimentConfig:
    """Build ExperimentConfig from CLI args."""
    return ExperimentConfig(
        env_id=args.env,
        total_timesteps=args.total_timesteps,
        seed=seed_override if seed_override is not None else args.seed,
        num_seeds=args.num_seeds,
        wandb_project=args.wandb_project,
        use_wandb=not args.no_wandb,
        device=_resolve_device(args.device),
        model=ModelConfig(
            backbone=args.agent,
            hidden_dim=args.hidden_dim,
            state_dim=args.state_dim,
            num_blocks=args.num_blocks,
            order=args.order,
            activation=args.activation,
        ),
        ppo=PPOConfig(
            lr=args.lr,
            rollout_steps=args.rollout_steps,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
        ),
    )


def _init_wandb(cfg: ExperimentConfig, run_name: str):
    """Initialize wandb run."""
    import wandb

    return wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        config={
            "env_id": cfg.env_id,
            "seed": cfg.seed,
            "total_timesteps": cfg.total_timesteps,
            "backbone": cfg.model.backbone,
            "hidden_dim": cfg.model.hidden_dim,
            "state_dim": cfg.model.state_dim,
            "num_blocks": cfg.model.num_blocks,
            "order": cfg.model.order,
            "activation": cfg.model.activation,
            "lr": cfg.ppo.lr,
            "rollout_steps": cfg.ppo.rollout_steps,
            "batch_size": cfg.ppo.batch_size,
            "clip_eps": cfg.ppo.clip_eps,
            "gamma": cfg.ppo.gamma,
            "gae_lambda": cfg.ppo.gae_lambda,
        },
        reinit=True,
    )


def run_single(cfg: ExperimentConfig) -> dict:
    """Execute a single training run."""
    from hwnode.train import train

    run_name = f"{cfg.model.backbone}-{cfg.env_id}-s{cfg.seed}"
    print(f"\n{'='*60}")
    print(f" RUN: {run_name}")
    print(f"{'='*60}")

    wandb_run = None
    if cfg.use_wandb:
        wandb_run = _init_wandb(cfg, run_name)

    try:
        metrics = train(cfg, wandb_run)
    finally:
        if wandb_run:
            import wandb
            wandb.finish()

    return metrics


def _print_summary(results: list[dict]) -> None:
    """Print sweep results summary."""
    print(f"\n{'='*60}")
    print(" SWEEP COMPLETE")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['config_name']:>20} | {r['env_id']:>25} | "
              f"seed={r['seed']} | reward={r['final_mean_reward']:>8.1f} | "
              f"params={r['param_count']:,}")


def _run_sweep_matrix(args, envs, configs) -> list[dict]:
    """Run a sweep over envs × configs × seeds."""
    results = []
    for env_id in envs:
        for name, backbone, hdim, sdim, num_blocks in configs:
            for seed in range(args.num_seeds):
                args.env = env_id
                args.agent = backbone
                args.hidden_dim = hdim
                args.state_dim = sdim
                args.num_blocks = num_blocks

                cfg = build_config(args, seed_override=seed)
                metrics = run_single(cfg)
                metrics["config_name"] = name
                metrics["env_id"] = env_id
                metrics["seed"] = seed
                results.append(metrics)
    return results


def run_sweep(args: argparse.Namespace) -> None:
    """Execute the original experiment matrix (v1)."""
    envs = ["CartPole-v1", "Acrobot-v1", "LunarLander-v3"]
    configs = [
        # (name, backbone, hidden_dim, state_dim, num_blocks)
        ("hwnode-small", "hwnode", 64, 16, 2),
        ("hwnode-medium", "hwnode", 64, 32, 2),
        ("mlp-matched-width", "mlp", 64, 0, 2),
        ("mlp-narrow", "mlp", 32, 0, 2),
    ]
    results = _run_sweep_matrix(args, envs, configs)
    _print_summary(results)


def run_sweep_v2(args: argparse.Namespace) -> None:
    """Execute v2 experiment matrix: tiny params + continuous control.

    Two sub-experiments:
      A) Tiny discrete: CartPole & Acrobot with very small networks
         where param budget actually matters.
      B) Continuous control: Pendulum-v1 and MountainCarContinuous-v0
         to test continuous action support.
    """
    results = []

    # ---- A) Tiny discrete ----
    # At hidden_dim=16 with state_dim=4, each HW-NODE block has:
    #   W_in: 16*4+4=68, A: 4*4=16, W_out: 4*16+16=80, norms~=32 → ~196/block
    # vs MLP at hidden_dim=8 with same block count:
    #   each layer: LN(8)=16, Linear(8→8)+bias=72, ReLU, Linear(8→8)+bias=72 → ~160/layer
    tiny_discrete_envs = ["CartPole-v1", "Acrobot-v1"]
    tiny_configs = [
        # (name, backbone, hidden_dim, state_dim, num_blocks)
        ("hwnode-tiny",    "hwnode", 16, 4, 2),
        ("hwnode-micro",   "hwnode", 16, 4, 1),  # single block
        ("mlp-tiny",       "mlp",   16, 0, 2),
        ("mlp-micro",      "mlp",    8, 0, 2),   # param-matched to hwnode-tiny
    ]
    print("\n" + "=" * 60)
    print(" SWEEP V2 — Part A: Tiny Discrete")
    print("=" * 60)
    results += _run_sweep_matrix(args, tiny_discrete_envs, tiny_configs)

    # ---- B) Continuous control ----
    # Pendulum-v1: obs=3, act=1 (torque), reward ∈ [-16.27, 0]
    # MountainCarContinuous-v0: obs=2, act=1 (force), reward based on reaching goal
    continuous_envs = ["Pendulum-v1", "MountainCarContinuous-v0"]
    continuous_configs = [
        ("hwnode-small",       "hwnode", 64, 16, 2),
        ("hwnode-tiny",        "hwnode", 32,  8, 2),
        ("mlp-matched-width",  "mlp",   64,  0, 2),
        ("mlp-narrow",         "mlp",   32,  0, 2),
    ]
    print("\n" + "=" * 60)
    print(" SWEEP V2 — Part B: Continuous Control")
    print("=" * 60)
    results += _run_sweep_matrix(args, continuous_envs, continuous_configs)

    _print_summary(results)


def main() -> None:
    args = parse_args()

    if args.sweep:
        run_sweep(args)
    elif args.sweep_v2:
        run_sweep_v2(args)
    else:
        cfg = build_config(args)
        run_single(cfg)


if __name__ == "__main__":
    main()
