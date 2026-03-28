#!/usr/bin/env python3
"""Check parameter counts for MLP and HW-NODE configurations."""

import sys

sys.path.insert(0, ".")

import torch
import gymnasium as gym
from hwnode.baseline import MLPNetwork
from hwnode.model import HWNodeNetwork
from experiments.taylor_vs_chebyshev import FlexActorCritic


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    env_id = "Pendulum-v1"
    env_tmp = gym.make(env_id)
    obs_dim = env_tmp.observation_space.shape[0]
    continuous = isinstance(env_tmp.action_space, gym.spaces.Box)
    act_dim = env_tmp.action_space.shape[0] if continuous else env_tmp.action_space.n
    env_tmp.close()

    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}, Continuous: {continuous}")
    print("=" * 60)

    # Test MLP with various hidden_dim and num_blocks=2
    print("MLP (num_blocks=2):")
    for hdim in [16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40]:
        model = FlexActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            BackboneClass=MLPNetwork,
            continuous=continuous,
            hidden_dim=hdim,
            num_blocks=2,
        )
        params = count_params(model)
        print(f"  hdim={hdim:2d} -> {params:5,} params")

    print("\nHW-NODE (state_dim=32, num_blocks=3, order=2):")
    for hdim in [16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40]:
        model = FlexActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            BackboneClass=HWNodeNetwork,
            continuous=continuous,
            hidden_dim=hdim,
            state_dim=hdim,  # state_dim = hidden_dim
            num_blocks=3,
            order=2,
        )
        params = count_params(model)
        print(f"  hdim={hdim:2d} -> {params:5,} params")

    print("\nHW-NODE (state_dim=32, num_blocks=3, order=3):")
    for hdim in [16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40]:
        model = FlexActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            BackboneClass=HWNodeNetwork,
            continuous=continuous,
            hidden_dim=hdim,
            state_dim=hdim,  # state_dim = hidden_dim
            num_blocks=3,
            order=3,
        )
        params = count_params(model)
        print(f"  hdim={hdim:2d} -> {params:5,} params")

    # Now find the closest to 9,155 for MLP with num_blocks=2
    target = 9155
    print(f"\nTarget params: {target}")
    print("MLP closest to target (num_blocks=2):")
    best_diff = float("inf")
    best_hdim = None
    for hdim in [16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40]:
        model = FlexActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            BackboneClass=MLPNetwork,
            continuous=continuous,
            hidden_dim=hdim,
            num_blocks=2,
        )
        params = count_params(model)
        diff = abs(params - target)
        if diff < best_diff:
            best_diff = diff
            best_hdim = hdim
        print(f"  hdim={hdim:2d} -> {params:5,} params (diff: {diff:4d})")
    print(
        f"Best MLP: hdim={best_hdim} -> {count_params(FlexActorCritic(obs_dim, act_dim, MLPNetwork, continuous, hidden_dim=best_hdim, num_blocks=2)):,} params"
    )

    # Now find the closest to 9,155 for HW-NODE with num_blocks=3
    print("\nHW-NODE closest to target (num_blocks=3, order=2):")
    best_diff = float("inf")
    best_hdim = None
    for hdim in [16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40]:
        model = FlexActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            BackboneClass=HWNodeNetwork,
            continuous=continuous,
            hidden_dim=hdim,
            state_dim=hdim,
            num_blocks=3,
            order=2,
        )
        params = count_params(model)
        diff = abs(params - target)
        if diff < best_diff:
            best_diff = diff
            best_hdim = hdim
        print(f"  hdim={hdim:2d} -> {params:5,} params (diff: {diff:4d})")
    print(
        f"Best HW-NODE (order=2): hdim={best_hdim} -> {count_params(FlexActorCritic(obs_dim, act_dim, HWNodeNetwork, continuous, hidden_dim=best_hdim, state_dim=best_hdim, num_blocks=3, order=2)):,} params"
    )

    print("\nHW-NODE closest to target (num_blocks=3, order=3):")
    best_diff = float("inf")
    best_hdim = None
    for hdim in [16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40]:
        model = FlexActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            BackboneClass=HWNodeNetwork,
            continuous=continuous,
            hidden_dim=hdim,
            state_dim=hdim,
            num_blocks=3,
            order=3,
        )
        params = count_params(model)
        diff = abs(params - target)
        if diff < best_diff:
            best_diff = diff
            best_hdim = hdim
        print(f"  hdim={hdim:2d} -> {params:5,} params (diff: {diff:4d})")
    print(
        f"Best HW-NODE (order=3): hdim={best_hdim} -> {count_params(FlexActorCritic(obs_dim, act_dim, HWNodeNetwork, continuous, hidden_dim=best_hdim, state_dim=best_hdim, num_blocks=3, order=3)):,} params"
    )


if __name__ == "__main__":
    main()
