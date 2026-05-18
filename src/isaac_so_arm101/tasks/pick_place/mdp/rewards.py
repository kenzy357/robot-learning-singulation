# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the Squint pick-and-place task (Isaac Lab port).

Faithful port of ``Place.compute_dense_reward`` / ``Place.evaluate`` from the
upstream ManiSkill3/SAPIEN env (``squint/envs/place.py``).

This version reads real PhysX *filtered* contact forces instead of geometric
proxies, so the contact-dependent logic matches upstream:

  * ``robot_touching_item``  -> gripper/jaw ContactSensors filtered vs the cube
  * ``robot_touching_bin``   -> gripper/jaw ContactSensors filtered vs the bowl
  * ``robot_touching_table`` -> gripper/jaw ContactSensors filtered vs the table
  * ``is_item_grasped``      -> BOTH gripper bodies in contact with the cube,
                                a proxy for SAPIEN's two-finger ``is_grasping``

The upstream reward is *staged*: later stages OVERRIDE earlier values rather
than adding. The staged ``torch.where`` ladder produces four mutually
exclusive regions (exactly one active per env), so it is decomposed here into
four additive ``place_stage_*`` terms plus three penalties. Their SUM equals
the original single staged reward bit-for-bit — the split exists only so each
part is logged separately in wandb (``Episode_Reward/<term>``).

Staged ladder (matches upstream):

    base             reward = 2 * (1 - tanh(5 d_ee_cube))
    grasped          reward = 3 + place_reward
    cube above bowl  reward = 4 + place_reward + (~touching_item)
                              + gripper_openness + robot_static
    success          reward = 9
    always (after)   reward -= 6 * touching_table
                     reward -= 3 * touching_bin
                     reward -= 1 * (~item_lifted)

where ``place_reward = (1 - tanh(5 d_goal)) + (1 - tanh(10 d_goal_z))``.

``_place_components`` computes every piece once; the term functions below are
thin wrappers that select one piece each. The CAPS-style action-rate penalty
is NOT part of upstream's dense reward; keep it as a separate term in the cfg.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Default ContactSensor scene-entity names (see ObjectTableSceneCfg).
_ITEM_SENSORS = ("gripper_item_contact", "jaw_item_contact")
_BOWL_SENSORS = ("gripper_bowl_contact", "jaw_bowl_contact")
_TABLE_SENSORS = ("gripper_table_contact", "jaw_table_contact")


# ---------------------------------------------------------------------------
# Contact helpers — read PhysX filtered contact forces
# ---------------------------------------------------------------------------
def _body_touching(
    env: "ManagerBasedRLEnv", sensor_name: str, force_threshold: float
) -> torch.Tensor:
    """``(N,)`` bool: does this single-body ``ContactSensor`` report a filtered
    contact-force norm above ``force_threshold`` (N) against any filter prim?

    ``force_matrix_w`` has shape ``(N, B, M, 3)`` — ``B`` bodies matched by the
    sensor ``prim_path`` (1 here), ``M`` filtered prims. We take the norm over
    xyz and the max over both ``B`` and ``M`` -> a per-env scalar.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    force_matrix = sensor.data.force_matrix_w  # (N, B, M, 3) or None
    if force_matrix is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.norm(force_matrix, dim=-1).amax(dim=(1, 2)) > force_threshold


def robot_touching_item(
    env: "ManagerBasedRLEnv",
    item_sensor_names: tuple[str, ...] = _ITEM_SENSORS,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """``1.0`` if EITHER gripper body is in contact with the cube.

    Isaac Lab port of SAPIEN's ``agent.is_touching(item)``.
    """
    touching = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in item_sensor_names:
        touching |= _body_touching(env, name, force_threshold)
    return touching.float()


def robot_touching_bin(
    env: "ManagerBasedRLEnv",
    bin_sensor_names: tuple[str, ...] = _BOWL_SENSORS,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """``1.0`` if EITHER gripper body is in contact with any bowl part.

    Isaac Lab port of SAPIEN's ``agent.is_touching(bin)``. Use with weight
    ``-3.0`` to restore upstream's ``- 3 * robot_touching_bin`` penalty.
    """
    touching = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in bin_sensor_names:
        touching |= _body_touching(env, name, force_threshold)
    return touching.float()


def robot_touching_table(
    env: "ManagerBasedRLEnv",
    table_sensor_names: tuple[str, ...] = _TABLE_SENSORS,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """``1.0`` if EITHER gripper body is in contact with the table.

    Isaac Lab port of SAPIEN's ``agent.is_touching(table)``. Use with weight
    ``-6.0`` to restore upstream's ``- 6 * robot_touching_table`` penalty.
    """
    touching = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in table_sensor_names:
        touching |= _body_touching(env, name, force_threshold)
    return touching.float()


def item_grasped(
    env: "ManagerBasedRLEnv",
    item_sensor_names: tuple[str, ...] = _ITEM_SENSORS,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """``1.0`` when BOTH gripper bodies are in contact with the cube.

    Proxy for SAPIEN's two-finger ``agent.is_grasping(item)``: the original
    checks that both fingers contact the object (with a force-direction test
    that PhysX filtered contacts cannot reproduce). Requiring both bodies to
    register a contact is the closest observable equivalent.
    """
    grasped = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for name in item_sensor_names:
        grasped &= _body_touching(env, name, force_threshold)
    return grasped.float()


def not_lifted(
    env: "ManagerBasedRLEnv",
    cube_half_size: float = 0.01,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """``1.0`` while the cube has NOT been lifted off the table.

    Upstream's "encourage picking the item up fast" penalty — use with weight
    ``-1.0``. ``item_lifted`` is ``cube.z >= cube_half_size + 1e-3``.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    item_lifted = cube.data.root_pos_w[:, 2] >= (cube_half_size + 1e-3)
    return (~item_lifted).float()


def cube_displaced_on_table(
    env: "ManagerBasedRLEnv",
    cube_half_size: float = 0.01,
    max_radius: float = 0.04,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """``1.0`` while the cube sits on the table yet has been shoved more than
    ``max_radius`` (metres) horizontally from its spawn position.

    Penalizes sliding/dragging the cube across the table instead of lifting it
    straight up. ``on_table`` reuses the ``not_lifted`` predicate
    (``z < cube_half_size + 1e-3``); once the cube is lifted this is ``0.0`` —
    only table-bound displacement is penalized.

    The spawn position comes from ``env.block_spawn_pos_w``, snapshotted each
    reset by the ``record_block_spawn`` event — so it stays correct even if the
    block reset is domain-randomized. Falls back to ``default_root_state`` if
    that event is not registered (correct only without randomization).
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    on_table = cube.data.root_pos_w[:, 2] < (cube_half_size + 1e-3)
    spawn_pos = getattr(env, "block_spawn_pos_w", None)
    if spawn_pos is None:
        spawn_pos = cube.data.default_root_state[:, :3] + env.scene.env_origins
    disp = torch.linalg.vector_norm(
        cube.data.root_pos_w[:, :2] - spawn_pos[:, :2], dim=-1
    )
    return (on_table & (disp > max_radius)).float()


# ---------------------------------------------------------------------------
# Success predicate — shared by the success termination and the reward
# ---------------------------------------------------------------------------
def place_success(
    env: "ManagerBasedRLEnv",
    bowl_radius: float = 0.05,
    robot_static_threshold: float = 2e-2,
    force_threshold: float = 1.0,
    item_sensor_names: tuple[str, ...] = _ITEM_SENSORS,
    bowl_sensor_names: tuple[str, ...] = _BOWL_SENSORS,
    gripper_joint_name: str = "gripper",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
) -> torch.Tensor:
    """Squint ``Place`` success predicate (``evaluate``'s ``success``):

        success = is_item_above_bin & ~robot_touching_item
                  & is_robot_static & ~robot_touching_bin

    Used as BOTH the success termination and the success stage of the staged
    reward — mirroring how upstream reuses ``info["success"]``.

    ``is_item_above_bin`` is the circular analog of upstream's rectangular
    ``inside_x & inside_y``: the cube xy is within ``bowl_radius`` of the bowl
    centre. ``is_robot_static`` checks the arm joints (gripper excluded).
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    xy_dist = torch.norm(
        cube.data.root_pos_w[:, :2] - bowl.data.root_pos_w[:, :2], dim=-1
    )
    is_item_above_bin = xy_dist <= bowl_radius

    touching_item = robot_touching_item(env, item_sensor_names, force_threshold).bool()
    touching_bin = robot_touching_bin(env, bowl_sensor_names, force_threshold).bool()

    gripper_idx = robot.data.joint_names.index(gripper_joint_name)
    arm_idx = [i for i in range(robot.data.joint_vel.shape[-1]) if i != gripper_idx]
    robot_v = torch.norm(robot.data.joint_vel[:, arm_idx], dim=-1)
    is_robot_static = robot_v <= robot_static_threshold

    return is_item_above_bin & (~touching_item) & is_robot_static & (~touching_bin)


# ---------------------------------------------------------------------------
# Staged dense reward — faithful port of Place.compute_dense_reward
# ---------------------------------------------------------------------------
def _place_components(
    env: "ManagerBasedRLEnv",
    cube_half_size: float = 0.01,
    bowl_radius: float = 0.05,
    rim_height: float = 0.025,
    force_threshold: float = 1.0,
    robot_static_threshold: float = 2e-2,
    gripper_joint_name: str = "gripper",
    item_sensor_names: tuple[str, ...] = _ITEM_SENSORS,
    bowl_sensor_names: tuple[str, ...] = _BOWL_SENSORS,
    table_sensor_names: tuple[str, ...] = _TABLE_SENSORS,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Compute every piece of the staged Place reward once.

    Returns a dict whose four ``stage_*`` entries are mutually exclusive
    (exactly one is nonzero per env): ``stage_base + stage_grasp +
    stage_above + stage_success`` reproduces the staged ``torch.where``
    ladder, and subtracting the three penalty entries reproduces upstream's
    ``compute_dense_reward`` bit-for-bit.
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
    reach = 2.0 * (1.0 - torch.tanh(5.0 * tcp_to_item_dist))

    # --- placement target: bowl centre xy, bowl-rim z ----------------------
    goal_xyz = bowl_pos.clone()
    goal_xyz[:, 2] = bowl_pos[:, 2] + rim_height + cube_half_size

    item_to_goal_dist = torch.norm(goal_xyz - item_pos, dim=-1)
    place_reward_final = 1.0 - torch.tanh(5.0 * item_to_goal_dist)

    # xy / z split with far/close logic (encourages lift-then-lower)
    item_to_goal_dist_xy = torch.norm(goal_xyz[:, :2] - item_pos[:, :2], dim=-1)
    item_to_goal_dist_z_far = torch.abs(
        (goal_xyz[:, 2] + rim_height + 0.03) - item_pos[:, 2]
    )
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

    # --- stage flags (contact-based) ---------------------------------------
    is_grasped = item_grasped(env, item_sensor_names, force_threshold).bool()
    is_item_above_bin = item_to_goal_dist_xy <= bowl_radius
    item_lifted = item_pos[:, 2] >= (cube_half_size + 1e-3)

    touching_item = robot_touching_item(env, item_sensor_names, force_threshold)
    touching_bin = robot_touching_bin(env, bowl_sensor_names, force_threshold)
    touching_table = robot_touching_table(env, table_sensor_names, force_threshold)
    is_item_dropped = 1.0 - touching_item  # release bonus: (~touching_item)

    arm_idx = [i for i in range(robot.data.joint_vel.shape[-1]) if i != gripper_idx]
    robot_v = torch.norm(robot.data.joint_vel[:, arm_idx], dim=-1)
    static_robot_reward = 1.0 - torch.tanh(robot_v * 10.0)

    success = place_success(
        env,
        bowl_radius=bowl_radius,
        robot_static_threshold=robot_static_threshold,
        force_threshold=force_threshold,
        item_sensor_names=item_sensor_names,
        bowl_sensor_names=bowl_sensor_names,
        gripper_joint_name=gripper_joint_name,
        robot_cfg=robot_cfg,
        cube_cfg=cube_cfg,
        bowl_cfg=bowl_cfg,
    )

    # --- staged values -----------------------------------------------------
    grasped_value = 3.0 + place_reward
    above_value = (
        4.0 + place_reward + is_item_dropped + gripper_openness + static_robot_reward
    )

    # mutually exclusive region masks (matches the torch.where ladder:
    # base -> grasped -> above_bin -> success, later stages override earlier)
    region_success = success
    region_above = is_item_above_bin & (~success)
    region_grasp = is_grasped & (~is_item_above_bin) & (~success)
    region_base = (~is_grasped) & (~is_item_above_bin) & (~success)

    zeros = torch.zeros_like(reach)
    return {
        "stage_base": torch.where(region_base, reach, zeros),
        "stage_grasp": torch.where(region_grasp, grasped_value, zeros),
        "stage_above": torch.where(region_above, above_value, zeros),
        "stage_success": torch.where(region_success, torch.full_like(reach, 9.0), zeros),
        "touching_table": touching_table,
        "touching_bin": touching_bin,
        "not_lifted": (~item_lifted).float(),
    }


def place_stage_reach(
    env: "ManagerBasedRLEnv",
    cube_half_size: float = 0.01,
    bowl_radius: float = 0.05,
    rim_height: float = 0.025,
) -> torch.Tensor:
    """Base stage: ``2*(1 - tanh(5 d_ee_cube))``, active only before grasp."""
    return _place_components(
        env, cube_half_size=cube_half_size, bowl_radius=bowl_radius, rim_height=rim_height
    )["stage_base"]


def place_stage_grasp(
    env: "ManagerBasedRLEnv",
    cube_half_size: float = 0.01,
    bowl_radius: float = 0.05,
    rim_height: float = 0.025,
) -> torch.Tensor:
    """Grasped stage: ``3 + place_reward``, active while grasped & not over bowl."""
    return _place_components(
        env, cube_half_size=cube_half_size, bowl_radius=bowl_radius, rim_height=rim_height
    )["stage_grasp"]


def place_stage_above_bin(
    env: "ManagerBasedRLEnv",
    cube_half_size: float = 0.01,
    bowl_radius: float = 0.05,
    rim_height: float = 0.025,
) -> torch.Tensor:
    """Above-bowl stage: ``4 + place_reward + (~touch_item) + openness + static``."""
    return _place_components(
        env, cube_half_size=cube_half_size, bowl_radius=bowl_radius, rim_height=rim_height
    )["stage_above"]


def place_stage_success(
    env: "ManagerBasedRLEnv",
    cube_half_size: float = 0.01,
    bowl_radius: float = 0.05,
    rim_height: float = 0.025,
) -> torch.Tensor:
    """Success stage: flat ``9`` (same predicate as the success termination)."""
    return _place_components(
        env, cube_half_size=cube_half_size, bowl_radius=bowl_radius, rim_height=rim_height
    )["stage_success"]


def place_dense_reward(env: "ManagerBasedRLEnv", **kwargs) -> torch.Tensor:
    """Full staged dense reward — the sum of the four stage terms minus the
    three penalties. Equals upstream's ``compute_dense_reward``.

    The env cfg uses the split ``place_stage_*`` terms (so each logs to wandb
    separately); this single-call form is kept for reference / eval scripts.
    """
    c = _place_components(env, **kwargs)
    return (
        c["stage_base"]
        + c["stage_grasp"]
        + c["stage_above"]
        + c["stage_success"]
        - 6.0 * c["touching_table"]
        - 3.0 * c["touching_bin"]
        - c["not_lifted"]
    )


def place_normalized_dense_reward(
    env: "ManagerBasedRLEnv", **kwargs
) -> torch.Tensor:
    """``place_dense_reward / 9`` — matches ``compute_normalized_dense_reward``."""
    return place_dense_reward(env, **kwargs) / 9.0
