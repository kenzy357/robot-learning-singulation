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


def reach(
    env: "ManagerBasedRLEnv",
    std: float,
    cylinder_radius: float,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    
) -> torch.Tensor:
    """``1 - tanh(|ee - block| / std)`` — dense reach signal, always on."""
    block: RigidObject = env.scene[block_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    distance = torch.norm(
        block.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1
    )

    bowl: RigidObject = env.scene[bowl_cfg.name]
    xy_dist = torch.norm(
        block.data.root_pos_w[:, :2] - bowl.data.root_pos_w[:, :2], dim=1
    )

    in_cylinder = (xy_dist <= cylinder_radius).float() 
    return (1.0 - torch.tanh(distance / std)) * (1-in_cylinder)


def lift(
    env: "ManagerBasedRLEnv",
    max_height: float,
    rest_height: float = 0.02,
    cylinder_radius: float = 0.04, 
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    """Dense lift signal in ``[0, 1]``: ``clamp(block.z - rest_height, 0, max_height) / max_height``.

    Provides gradient for any upward motion of the block, so the policy can
    discover lifting before a hard threshold is crossed.
    """
    block: RigidObject = env.scene[block_cfg.name]
    height_above_rest = (block.data.root_pos_w[:, 2] - rest_height).clamp(
        min=-0.5, max=max_height
    )

    bowl: RigidObject = env.scene[bowl_cfg.name]
    xy_dist = torch.norm(
        block.data.root_pos_w[:, :2] - bowl.data.root_pos_w[:, :2], dim=1
    )

    in_cylinder = (xy_dist <= cylinder_radius).float() 

    return (height_above_rest / max_height) * (1-in_cylinder)


def go_above_goal(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    cylinder_radius:float,
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
    in_cylinder = xy_dist <= cylinder_radius 


    gate = ((lifted > 0) | (in_cylinder > 0)).float()

    
    return gate * (1.0 - torch.tanh(xy_dist / std)) 


def drop(
    env: "ManagerBasedRLEnv",
    std: float,
    z_cylinder: float = 0.01,
    cylinder_radius: float = 0.04,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    """Dense reward to drive the *block* to a point ``height_above_goal`` (m) above
    the bowl center. Gated on the block being lifted so it only activates after pickup."""
    block: RigidObject = env.scene[block_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]

    target_pos = bowl.data.root_pos_w.clone()
    target_pos[:, 2] = z_cylinder
    distance = torch.norm(block.data.root_pos_w - target_pos, dim=1)

    xy_dist = torch.norm(
        block.data.root_pos_w[:, :2] - bowl.data.root_pos_w[:, :2], dim=1
    )

    in_cylinder = (xy_dist <= cylinder_radius).float() 

    gate = ((z_cylinder > 0) | (in_cylinder > 0)).float()

    return in_cylinder * (1.0 - torch.tanh(distance / std))


def success(
    env: ManagerBasedRLEnv,
    xy_threshold: float = 0.04,
    z_cylinder: float = 0.05,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    block: RigidObject = env.scene[block_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    block_pos = block.data.root_pos_w
    bowl_pos = bowl.data.root_pos_w


    xy_distance = torch.norm(block_pos[:, :2] - bowl_pos[:, :2], dim=1)
    inside_xy = xy_distance < xy_threshold


    dz = block_pos[:, 2]
    inside_z = (dz > -0.01) & (dz < z_cylinder)

    return inside_xy & inside_z

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
