# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the Squint pick-and-place task (Isaac Lab port).

This is a faithful port of ``Place.compute_dense_reward`` from the upstream
ManiSkill3/SAPIEN env (``squint/envs/place.py``). The upstream reward is a
*staged* reward: later stages OVERRIDE the value of earlier stages rather than
adding to them. Isaac Lab's reward manager sums terms additively, so the whole
staged computation is implemented as a single term (``place_dense_reward``,
weight 1.0) instead of being split into additive sub-terms.

Stage ladder (matches upstream exactly):

    base            reward = 2 * (1 - tanh(5 * d_ee_cube))           # reach
    grasped         reward = 3 + place_reward                        # override
    cube over bowl  reward = 4 + place_reward + dropped              # override
                             + gripper_openness + robot_static
    success         reward = 9                                       # override
    always          reward -= 1 * (cube not lifted)

where ``place_reward = place_reward_final + place_reward_z`` (see below).

Differences from upstream, all forced by Isaac Lab not exposing SAPIEN's
contact queries (the SO-ARM101 USD is imported with contact sensors disabled):

  * ``robot_touching_item`` -> proxied by ``is_grasped`` (holding == touching).
  * ``robot_touching_bin``  -> not observable; the ``-3 * touching_bin``
    penalty is dropped. Re-add it as a contact-sensor term if you enable
    contact sensors on the gripper bodies.
  * ``is_item_grasped``     -> proxied by ``cube_grasped`` (gripper closed +
    cube within ``grasp_diff_threshold`` of the EE frame).

The CAPS-style action-rate penalty that upstream folds into this function is
kept as a separate additive term: use the built-in ``mdp.action_rate_l2``
with weight ``-action_smooth_coef`` in the env cfg.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cube_grasped(
    env: "ManagerBasedRLEnv",
    diff_threshold: float = 0.02,
    grasp_threshold: float = 0.26,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    gripper_joint_name: str = "gripper",
) -> torch.Tensor:
    """``1.0`` when the cube is grasped: it is within ``diff_threshold`` metres
    of the EE frame AND the gripper joint is closed past ``grasp_threshold``.

    Proxy for SAPIEN's contact-based ``agent.is_grasping`` (Isaac Lab port).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    pos_diff = torch.norm(cube.data.root_pos_w - ee_pos_w, dim=-1)

    gripper_idx = robot.data.joint_names.index(gripper_joint_name)
    gripper_pos = robot.data.joint_pos[:, gripper_idx]

    grasped = (pos_diff < diff_threshold) & (gripper_pos < grasp_threshold)
    return grasped.float()


# ---------------------------------------------------------------------------
# Main staged dense reward — faithful port of Place.compute_dense_reward
# ---------------------------------------------------------------------------
def place_dense_reward(
    env: "ManagerBasedRLEnv",
    cube_half_size: float = 0.01,
    bowl_radius: float = 0.05,
    rim_height: float = 0.025,
    grasp_diff_threshold: float = 0.02,
    grasp_threshold: float = 0.26,
    gripper_joint_name: str = "gripper",
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Staged dense reward for placing the cube into the bowl.

    Args:
        cube_half_size:  Cube half-extent (m). Cube is ``2*cube_half_size`` per side.
        bowl_radius:     Bowl footprint radius (m) — the "cube is over the bowl" gate.
        rim_height:      Bowl wall height (m). The placement target is the bowl
                         rim, so the cube can drop straight down into the bowl.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[:, 0, :]      # (N, 3)
    item_pos = cube.data.root_pos_w                   # (N, 3)
    bowl_pos = bowl.data.root_pos_w                   # (N, 3)

    # --- reaching reward: 2 * (1 - tanh(5 d_ee_cube)) ----------------------
    tcp_to_item_dist = torch.norm(ee_pos - item_pos, dim=-1)
    reward = 2.0 * (1.0 - torch.tanh(5.0 * tcp_to_item_dist))

    # --- placement target: bowl centre xy, bowl-rim z ----------------------
    goal_xyz = bowl_pos.clone()
    goal_xyz[:, 2] = bowl_pos[:, 2] + rim_height + cube_half_size

    # overall distance reward
    item_to_goal_dist = torch.norm(goal_xyz - item_pos, dim=-1)
    place_reward_final = 1.0 - torch.tanh(5.0 * item_to_goal_dist)

    # xy / z split with far/close logic (encourages lift-then-lower)
    item_to_goal_dist_xy = torch.norm(goal_xyz[:, :2] - item_pos[:, :2], dim=-1)
    # Far: aim above the bowl rim so the policy lifts before traversing.
    item_to_goal_dist_z_far = torch.abs(
        (goal_xyz[:, 2] + rim_height + 0.03) - item_pos[:, 2]
    )
    # Close: aim at the final placement height.
    item_to_goal_dist_z_close = torch.abs(goal_xyz[:, 2] - item_pos[:, 2])
    item_close_to_goal = item_to_goal_dist_xy <= bowl_radius
    item_to_goal_dist_z = torch.where(
        item_close_to_goal, item_to_goal_dist_z_close, item_to_goal_dist_z_far
    )
    place_reward_z = 1.0 - torch.tanh(10.0 * item_to_goal_dist_z)
    place_reward = place_reward_final + place_reward_z

    # --- gripper openness in [0, 1] ----------------------------------------
    gripper_idx = robot.data.joint_names.index(gripper_joint_name)
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    g_lo = robot.data.soft_joint_pos_limits[:, gripper_idx, 0]
    g_hi = robot.data.soft_joint_pos_limits[:, gripper_idx, 1]
    gripper_openness = ((gripper_pos - g_lo) / (g_hi - g_lo + 1e-8)).clamp(0.0, 1.0)

    # --- stage flags -------------------------------------------------------
    is_grasped = cube_grasped(
        env,
        diff_threshold=grasp_diff_threshold,
        grasp_threshold=grasp_threshold,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        cube_cfg=cube_cfg,
        gripper_joint_name=gripper_joint_name,
    ).bool()
    is_item_above_bin = item_to_goal_dist_xy <= bowl_radius
    item_lifted = item_pos[:, 2] >= (cube_half_size + 1e-3)

    # robot-static reward (arm joints only — exclude the gripper joint)
    arm_idx = [i for i in range(robot.data.joint_vel.shape[-1]) if i != gripper_idx]
    robot_v = torch.norm(robot.data.joint_vel[:, arm_idx], dim=-1)
    static_robot_reward = 1.0 - torch.tanh(robot_v * 10.0)

    # ~robot_touching_item is not observable -> proxy with ~is_grasped.
    is_item_dropped = (~is_grasped).float()

    # --- staged overrides (later stages replace earlier values) ------------
    grasped_reward = 3.0 + place_reward
    reward = torch.where(is_grasped, grasped_reward, reward)

    above_bin_reward = (
        4.0 + place_reward + is_item_dropped + gripper_openness + static_robot_reward
    )
    reward = torch.where(is_item_above_bin, above_bin_reward, reward)

    # success -> flat 9 (see terminations.place_success for the same predicate)
    is_item_static = torch.norm(cube.data.root_lin_vel_w, dim=-1) <= 2e-2
    is_robot_static = robot_v <= 2e-2
    success = is_item_above_bin & is_item_static & is_robot_static & (~is_grasped)
    reward = torch.where(success, torch.full_like(reward, 9.0), reward)

    # --- always-on penalty: encourage picking the cube up fast -------------
    reward = reward - 1.0 * (~item_lifted).float()

    # NOTE: upstream also subtracts 3 * robot_touching_bin here. That contact
    # query is unavailable in this Isaac Lab port (see module docstring).
    return reward


def place_normalized_dense_reward(
    env: "ManagerBasedRLEnv", **kwargs
) -> torch.Tensor:
    """``place_dense_reward / 9`` — matches ``compute_normalized_dense_reward``."""
    return place_dense_reward(env, **kwargs) / 9.0


# ---------------------------------------------------------------------------
# Contact penalty — robot touching the bowl
# ---------------------------------------------------------------------------
def robot_touches_bowl(
    env: "ManagerBasedRLEnv",
    sensor_names: tuple[str, ...] = ("gripper_contact", "jaw_contact"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """``1.0`` while any gripper body is in contact with any bowl part, else ``0.0``.

    Restores the upstream ``- 3 * robot_touching_bin`` penalty that the first
    Isaac Lab port had to drop (use this term with a negative weight, e.g.
    ``-3.0``). Unlike SAPIEN's contact query, this reads PhysX *filtered*
    contact forces from one or more ``ContactSensor``s — each sensor watches a
    single gripper body and is filtered against the bowl floor + wall prims, so
    only robot↔bowl contacts register (cube↔bowl and bowl↔table do not).

    Args:
        sensor_names: ContactSensor scene-entity names to read. One sensor per
            gripper body (PhysX filtered contacts are one-body-to-many).
        force_threshold: Contact-force norm (N) above which a touch is counted.
    """
    touching = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in sensor_names:
        sensor = env.scene.sensors[name]
        force_matrix = sensor.data.force_matrix_w  # (N, B, M, 3) or None
        if force_matrix is None:
            continue
        # norm over xyz -> (N, B, M); max over (bodies, filtered bowl parts).
        max_force = torch.norm(force_matrix, dim=-1).amax(dim=(1, 2))
        touching |= max_force > force_threshold
    return touching.float()
