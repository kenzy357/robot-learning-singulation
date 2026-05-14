# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Target-aware additive rewards for the eval2 pick-and-place task.

Each episode one of the two cubes (``block`` or ``block_b``) is the target,
indicated by ``env.target_idx`` (set by ``randomize_cube_colors_and_target``).
All rewards dispatch to that cube via ``_target_block_pos``.

Reward terms (same shape as the eval1 design, just target-aware):
    reach            dense EE → target, always on
    lift             binary 1/0 when target is above ``minimal_height``
    goal_xy_distance dense target xy → bowl xy, gated on lifted
    success          binary 1/0 when target sits inside the bowl region
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _target_block_pos(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """World-frame position (N, 3) of the target cube selected per env.

    Falls back to ``block`` if the target buffer hasn't been initialized yet
    (e.g. very first step before any reset event has run).
    """
    a: RigidObject = env.scene["block"]
    b: RigidObject = env.scene["block_b"]
    if not hasattr(env, "target_idx"):
        return a.data.root_pos_w
    is_a = (env.target_idx == 0).unsqueeze(-1)
    return torch.where(is_a, a.data.root_pos_w, b.data.root_pos_w)


def target_ee_distance_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """``1 - tanh(|ee - target| / std)`` — dense reach signal."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    target_pos_w = _target_block_pos(env)
    distance = torch.norm(
        target_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1
    )
    return 1.0 - torch.tanh(distance / std)


def target_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
) -> torch.Tensor:
    """``1.0`` while the target is above ``minimal_height``."""
    target_pos_w = _target_block_pos(env)
    return torch.where(target_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def target_to_goal_xy_distance_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    """``(target.z > h) * (1 - tanh(|target_xy - bowl_xy| / std))``."""
    bowl: RigidObject = env.scene[bowl_cfg.name]
    target_pos_w = _target_block_pos(env)
    xy_dist = torch.norm(
        target_pos_w[:, :2] - bowl.data.root_pos_w[:, :2], dim=1
    )
    lifted = (target_pos_w[:, 2] > minimal_height).float()
    return lifted * (1.0 - torch.tanh(xy_dist / std))


def target_in_bowl(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.04,
    z_max_above_bowl: float = 0.05,
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    """``1.0`` while the target sits inside the bowl xy footprint and z window."""
    bowl: RigidObject = env.scene[bowl_cfg.name]
    target_pos_w = _target_block_pos(env)
    bowl_pos = bowl.data.root_pos_w
    xy_dist = torch.norm(target_pos_w[:, :2] - bowl_pos[:, :2], dim=1)
    dz = target_pos_w[:, 2] - bowl_pos[:, 2]
    inside_xy = xy_dist < xy_threshold
    inside_z = (dz > -0.01) & (dz < z_max_above_bowl)
    return (inside_xy & inside_z).float()
