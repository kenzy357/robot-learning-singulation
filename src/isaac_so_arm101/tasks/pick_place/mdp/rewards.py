# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-conditional dense rewards for the pick-and-place task.

Each step, the per-env current stage is computed from observed state:
    0  default            EE far from block
    1  approach           EE close to block but not grasped
    2  grasped            gripper closed on block, still on table
    3  lifted             grasped + block above table
    4  above bowl         grasped + block over bowl xy
    5  released in bowl   block in bowl AND gripper not gripping it

Each reward term is **active only in its own stage** (zero elsewhere) and dense
within its stage — so the policy gets a smooth per-step gradient that points
toward the next stage:

    reach     stage 0  → pulls EE toward block
    grasp     stage 1  → rewards gripper closing while near block
    lift      stage 2  → rewards block height
    transport stage 3  → pulls block xy toward bowl
    place     stage 4  → rewards lowering block into bowl
    success   stage 5  → one-shot terminal bonus

``reset_episode_stage`` zeroes the per-env stage buffer at reset — wire it as
an EventTerm(mode="reset"). The buffer is only used by
``stage_regression_penalty``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Gripper joint open/close range (matches BinaryJointPositionActionCfg
# in joint_pos_env_cfg.py: open=0.5, close=0.0).
GRIPPER_OPEN = 0.5
GRIPPER_CLOSED = 0.0


# ---------------------------------------------------------------------------
# Stage buffer (only used by the regression penalty)
# ---------------------------------------------------------------------------
def _ensure_stage_buffer(env: "ManagerBasedRLEnv") -> None:
    if not hasattr(env, "episode_max_stage"):
        env.episode_max_stage = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )


def reset_episode_stage(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """Zero the per-env max-stage buffer at episode reset."""
    _ensure_stage_buffer(env)
    env.episode_max_stage[env_ids] = 0


# ---------------------------------------------------------------------------
# Stage detection (single source of truth, called per term per step)
# ---------------------------------------------------------------------------
def _compute_current_stage(
    env: "ManagerBasedRLEnv",
    gripper_closed_threshold: float = 0.15,
    ee_to_block_threshold: float = 0.04,
    approach_threshold: float = 0.05,
    lift_threshold: float = 0.05,
    above_bowl_xy_threshold: float = 0.06,
    in_bowl_xy_threshold: float = 0.04,
    in_bowl_z_max: float = 0.05,
) -> torch.Tensor:
    robot = env.scene["robot"]
    block: RigidObject = env.scene["block"]
    bowl: RigidObject = env.scene["bowl_floor"]
    ee_frame: FrameTransformer = env.scene["ee_frame"]

    block_pos_w = block.data.root_pos_w
    bowl_pos_w = bowl.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]

    gripper_idx = robot.data.joint_names.index("gripper")
    is_closed = robot.data.joint_pos[:, gripper_idx] < gripper_closed_threshold
    ee_dist = torch.norm(ee_pos_w - block_pos_w, dim=1)
    grasped = is_closed & (ee_dist < ee_to_block_threshold)

    xy_dist_bowl = torch.norm(block_pos_w[:, :2] - bowl_pos_w[:, :2], dim=1)
    dz = block_pos_w[:, 2] - bowl_pos_w[:, 2]
    in_bowl = (xy_dist_bowl < in_bowl_xy_threshold) & (dz > -0.01) & (dz < in_bowl_z_max)
    lifted = block_pos_w[:, 2] > lift_threshold

    s1 = (ee_dist < approach_threshold).long() * 1
    s2 = grasped.long() * 2
    s3 = (grasped & lifted).long() * 3
    s4 = (grasped & lifted & (xy_dist_bowl < above_bowl_xy_threshold)).long() * 4
    s5 = (in_bowl & ~grasped).long() * 5

    return torch.stack([s1, s2, s3, s4, s5], dim=-1).max(dim=-1).values


# ---------------------------------------------------------------------------
# Phase-conditional reward terms (each dense within its stage gate)
# ---------------------------------------------------------------------------
def reach_phase_reward(
    env: "ManagerBasedRLEnv",
    std: float = 0.05,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Active stage 0 only: ``1 - tanh(|ee-block|/std)``, else 0."""
    block: RigidObject = env.scene[block_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    distance = torch.norm(
        block.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1
    )
    signal = 1.0 - torch.tanh(distance / std)
    stage = _compute_current_stage(env)
    return signal * (stage == 0).float()


def grasp_phase_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Active stage 1 only: rewards gripper closing.

    Signal is 0 when the gripper is fully open and 1 when fully closed, scaled
    linearly by the gripper joint position. The PD-controlled gripper takes a
    few steps to travel from open → closed, so PPO sees a smooth gradient that
    pushes it to commit to the close action.
    """
    robot = env.scene["robot"]
    gripper_idx = robot.data.joint_names.index("gripper")
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    closed_signal = 1.0 - ((gripper_pos - GRIPPER_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED)).clamp(0.0, 1.0)
    stage = _compute_current_stage(env)
    return closed_signal * (stage == 1).float()


def lift_phase_reward(
    env: "ManagerBasedRLEnv",
    max_height: float = 0.15,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
) -> torch.Tensor:
    """Active stage 2 only: normalized block z height in [0, 1]."""
    block: RigidObject = env.scene[block_cfg.name]
    height = block.data.root_pos_w[:, 2].clamp(min=0.0, max=max_height) / max_height
    stage = _compute_current_stage(env)
    return height * (stage == 2).float()


def transport_phase_reward(
    env: "ManagerBasedRLEnv",
    std: float = 0.1,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    """Active stage 3 only: ``1 - tanh(block_xy_to_bowl/std)``."""
    block: RigidObject = env.scene[block_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    xy_dist = torch.norm(
        block.data.root_pos_w[:, :2] - bowl.data.root_pos_w[:, :2], dim=1
    )
    signal = 1.0 - torch.tanh(xy_dist / std)
    stage = _compute_current_stage(env)
    return signal * (stage == 3).float()


def place_phase_reward(
    env: "ManagerBasedRLEnv",
    std: float = 0.05,
    block_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    bowl_cfg: SceneEntityCfg = SceneEntityCfg("bowl_floor"),
) -> torch.Tensor:
    """Active stage 4 only: rewards block descending toward bowl floor.

    The only way out of stage 4 is to drop the block in the bowl (→ stage 5).
    """
    block: RigidObject = env.scene[block_cfg.name]
    bowl: RigidObject = env.scene[bowl_cfg.name]
    dz = (block.data.root_pos_w[:, 2] - bowl.data.root_pos_w[:, 2]).clamp(min=0.0)
    signal = 1.0 - torch.tanh(dz / std)
    stage = _compute_current_stage(env)
    return signal * (stage == 4).float()


def success_phase_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Active stage 5 only: ``1.0``. Pair with a large weight (e.g. 200)."""
    stage = _compute_current_stage(env)
    return (stage == 5).float()


# ---------------------------------------------------------------------------
# Sparse transition signals (use the stage buffer)
#
# ``stage_progress_reward`` is the single owner of the buffer update. If you
# also wire ``stage_regression_penalty``, the buffer is still maintained
# correctly regardless of which term Isaac Lab evaluates first. If you wire
# ONLY ``stage_regression_penalty`` without stage_progress_reward, the buffer
# will never update and the penalty will never fire.
# ---------------------------------------------------------------------------
def stage_progress_reward(
    env: "ManagerBasedRLEnv",
    transition_bonuses: tuple[float, ...] = (0.0, 2.0, 5.0, 15.0, 30.0, 100.0),
) -> torch.Tensor:
    """One-shot bonus the step the policy enters a strictly higher stage.

    Compensates for the dip in per-step dense reward that can occur right at a
    stage boundary (e.g. grasp's peak ~1.0 → lift's initial ~0.05 when the
    block has only just been picked up). Pair with ``weight=1.0`` — the bonus
    tuple is already in absolute reward magnitudes.
    """
    _ensure_stage_buffer(env)
    current = _compute_current_stage(env)
    prev_max = env.episode_max_stage.clone()
    transitioned = current > prev_max
    env.episode_max_stage = torch.maximum(prev_max, current)
    bonuses = torch.tensor(transition_bonuses, device=env.device, dtype=torch.float32)
    return bonuses[current] * transitioned.float()


def stage_regression_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """1.0 per step while current stage is below the episode max. Use NEGATIVE weight.

    Read-only: relies on ``stage_progress_reward`` to maintain
    ``env.episode_max_stage``. Wire both together.
    """
    _ensure_stage_buffer(env)
    current = _compute_current_stage(env)
    return (current < env.episode_max_stage).float()
