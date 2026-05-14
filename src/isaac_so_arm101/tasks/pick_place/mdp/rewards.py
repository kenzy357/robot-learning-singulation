# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Additive rewards for the pick-and-place task.

Modeled after the upstream Isaac Lab Lift task:
    - reach            dense EE → block, always on
    - lift             binary 1/0 when block is above ``minimal_height``
    - goal_xy_distance dense block xy → bowl xy, gated on lifted
    - success          binary 1/0 when block sits inside the bowl region

No stage gating, no transition bonuses — just a smooth additive sum that
PPO can climb directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def block_ee_distance_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """``1 - tanh(|ee - block| / std)`` — dense reach signal, always on."""
    block: RigidObject = env.scene[block_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    distance = torch.norm(
        block.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1
    )
    return 1.0 - torch.tanh(distance / std)


def block_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """``1.0`` while the block is above ``minimal_height``, else ``0.0``."""
    block: RigidObject = env.scene[block_cfg.name]
    return torch.where(
        block.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0
    )


def block_height_shaped(
    env: "ManagerBasedRLEnv",
    max_height: float,
    rest_height: float = 0.02,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """Dense lift signal in ``[0, 1]``: ``clamp(block.z - rest_height, 0, max_height) / max_height``.

    Provides gradient for any upward motion of the block, so the policy can
    discover lifting before a hard threshold is crossed.
    """
    block: RigidObject = env.scene[block_cfg.name]
    height_above_rest = (block.data.root_pos_w[:, 2] - rest_height).clamp(
        min=0.0, max=max_height
    )
    return height_above_rest / max_height


def block_to_goal_xy_distance_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    """``(block.z > h) * (1 - tanh(|block_xy - bowl_xy| / std))`` — gated on lifted."""
    block: RigidObject = env.scene[block_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    xy_dist = torch.norm(
        block.data.root_pos_w[:, :2] - bowl.data.root_pos_w[:, :2], dim=1
    )
    lifted = (block.data.root_pos_w[:, 2] > minimal_height).float()
    return lifted * (1.0 - torch.tanh(xy_dist / std))


def block_above_goal_distance_tanh(
    env: "ManagerBasedRLEnv",
    std: float,
    height_above_goal: float = 0.03,
    minimal_height: float = 0.04,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    """Dense reward to drive the *block* to a point ``height_above_goal`` (m) above
    the bowl center. Gated on the block being lifted so it only activates after pickup."""
    block: RigidObject = env.scene[block_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]

    target_pos = bowl.data.root_pos_w.clone()
    target_pos[:, 2] = target_pos[:, 2] + height_above_goal
    distance = torch.norm(block.data.root_pos_w - target_pos, dim=1)

    lifted = (block.data.root_pos_w[:, 2] > minimal_height).float()
    return lifted * (1.0 - torch.tanh(distance / std))


def gripper_open_at_drop(
    env: "ManagerBasedRLEnv",
    radius: float = 0.02,
    height_above_goal: float = 0.03,
    open_value: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    """Dense-in-gripper, hard-gated-on-position: ``(|block - drop_target| < radius) * openness``.

    * Gate: block must be within ``radius`` of ``bowl + height_above_goal``.
    * ``openness = clamp(gripper_pos / open_value, 0, 1)`` — 0 fully closed, 1 fully
      open. Provides a smooth gradient toward opening *only* once the block is at
      the drop pose, so the policy isn't pushed to open during transport.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    block: RigidObject = env.scene[block_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]

    target_pos = bowl.data.root_pos_w.clone()
    target_pos[:, 2] = target_pos[:, 2] + height_above_goal
    at_drop = (torch.norm(block.data.root_pos_w - target_pos, dim=1) < radius).float()

    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids[0]]
    # Only the upper band of the gripper range counts as "open" — while gripping
    # the cube the joint settles partially closed, so a linear ramp would pay
    # out for holding. Ramp from 0.8*open_value → open_value.
    open_threshold = 0.8 * open_value
    openness = (
        (gripper_pos - open_threshold) / (open_value - open_threshold)
    ).clamp(0.0, 1.0)

    return at_drop * openness


def block_in_bowl(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.04,
    z_max_above_bowl: float = 0.05,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    """``1.0`` while the block sits inside the bowl xy footprint and z window."""
    block: RigidObject = env.scene[block_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    block_pos = block.data.root_pos_w
    bowl_pos = bowl.data.root_pos_w
    xy_dist = torch.norm(block_pos[:, :2] - bowl_pos[:, :2], dim=1)
    dz = block_pos[:, 2] - bowl_pos[:, 2]
    inside_xy = xy_dist < xy_threshold
    inside_z = (dz > -0.01) & (dz < z_max_above_bowl)
    return (inside_xy & inside_z).float()
