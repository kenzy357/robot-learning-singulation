# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

from .rewards import (
    item_grasped,
    robot_touching_bin,
    robot_touching_item,
    robot_touching_table,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

def goal_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
) -> torch.Tensor:
    """The position of the goal (bowl) in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]
    goal_pos_w = goal.data.root_pos_w[:, :3]
    goal_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], goal_pos_w
    )
    return goal_pos_b


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """The position of the end-effector (ee_frame target) in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], ee_pos_w
    )
    return ee_pos_b


# ---------------------------------------------------------------------------
# Privileged observations — teacher-only, ground-truth state the vision
# student must instead infer from the wrist camera. Used by the ``privileged``
# ObsGroup (see pick_place_env_cfg.py).
# ---------------------------------------------------------------------------
def cube_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """``(N, 3)`` cube position in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        cube.data.root_pos_w[:, :3],
    )
    return cube_pos_b


def cube_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """``(N, 4)`` cube orientation (quaternion, wxyz) in the robot root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    _, cube_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        cube.data.root_pos_w[:, :3],
        cube.data.root_quat_w,
    )
    return cube_quat_b


def cube_lin_vel(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """``(N, 3)`` cube linear velocity in the world frame."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return cube.data.root_lin_vel_w


def gripper_openness(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_joint_name: str = "gripper",
) -> torch.Tensor:
    """``(N, 1)`` gripper opening normalized to ``[0, 1]`` (0 closed, 1 open)."""
    robot: Articulation = env.scene[robot_cfg.name]
    idx = robot.data.joint_names.index(gripper_joint_name)
    pos = robot.data.joint_pos[:, idx]
    lo = robot.data.soft_joint_pos_limits[:, idx, 0]
    hi = robot.data.soft_joint_pos_limits[:, idx, 1]
    return ((pos - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0).unsqueeze(-1)


def privileged_contact_states(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``(N, 4)`` contact flags: ``[is_grasped, touching_item, touching_table,
    touching_bin]``. These come from PhysX filtered contact forces — privileged
    because the vision student has no direct contact sensing.
    """
    return torch.stack(
        [
            item_grasped(env),
            robot_touching_item(env),
            robot_touching_table(env),
            robot_touching_bin(env),
        ],
        dim=-1,
    )
