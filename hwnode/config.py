"""Configuration dataclasses for HW-NODE experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """HW-NODE or MLP model hyperparameters."""

    backbone: str = "hwnode"          # "hwnode" or "mlp"
    hidden_dim: int = 64             # working / hidden dimension
    state_dim: int = 16              # ODE latent dimension (hwnode only)
    num_blocks: int = 2              # number of HW-NODE blocks or MLP layers
    order: int = 4                   # Taylor series order (hwnode only)
    activation: str = "relu_squared" # pointwise nonlinearity


@dataclass
class PPOConfig:
    """PPO training hyperparameters."""

    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    num_epochs: int = 4              # PPO update epochs per rollout
    batch_size: int = 64             # minibatch size
    rollout_steps: int = 2048        # steps per rollout


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    env_id: str = "CartPole-v1"
    total_timesteps: int = 500_000
    seed: int = 42
    num_seeds: int = 5               # for sweep mode
    wandb_project: str = "hwnode-research"
    use_wandb: bool = True
    log_interval: int = 1            # log every N updates
    save_interval: int = 50          # checkpoint every N updates
    device: str = "cpu"              # auto-detected at runtime

    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
