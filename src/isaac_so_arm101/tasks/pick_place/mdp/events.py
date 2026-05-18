# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_bowl_position_uniform(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_names: list[str],
) -> None:
    """Sample one (dx, dy) offset per env and apply it to every bowl part.

    The bowl is built from a floor cylinder plus 8 wall segments arranged
    around it. To randomize the bowl position we must shift all 9 parts by
    the *same* offset so the rim stays attached to the floor.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    n = env_ids.numel()
    dx_lo, dx_hi = pose_range.get("x", (0.0, 0.0))
    dy_lo, dy_hi = pose_range.get("y", (0.0, 0.0))
    dx = torch.empty(n, device=env.device).uniform_(dx_lo, dx_hi)
    dy = torch.empty(n, device=env.device).uniform_(dy_lo, dy_hi)

    for name in asset_names:
        asset: RigidObject = env.scene[name]
        default_state = asset.data.default_root_state[env_ids].clone()
        default_state[:, 0] += dx
        default_state[:, 1] += dy
        default_state[:, :3] += env.scene.env_origins[env_ids]
        asset.write_root_pose_to_sim(default_state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(default_state[:, 7:], env_ids=env_ids)


def reset_block_and_bowl_uniform(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    block_pose_range: dict[str, tuple[float, float]],
    bowl_pose_range: dict[str, tuple[float, float]],
    min_distance: float,
    bowl_asset_names: list[str],
    block_asset_name: str = "block",
    max_resample: int = 20,
) -> None:
    """Reset block + bowl with rejection sampling so they never overlap.

    For each env, sample a block (dx, dy) and a bowl (dx, dy). If the
    resulting xy distance is below ``min_distance``, resample just those
    envs. After ``max_resample`` tries any remaining bad envs are clamped
    by snapping the block away from the bowl.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    n = env_ids.numel()
    device = env.device

    bx_lo, bx_hi = block_pose_range.get("x", (0.0, 0.0))
    by_lo, by_hi = block_pose_range.get("y", (0.0, 0.0))
    wx_lo, wx_hi = bowl_pose_range.get("x", (0.0, 0.0))
    wy_lo, wy_hi = bowl_pose_range.get("y", (0.0, 0.0))

    block_asset: RigidObject = env.scene[block_asset_name]
    bowl_floor: RigidObject = env.scene[bowl_asset_names[0]]

    block_default = block_asset.data.default_root_state[env_ids].clone()
    bowl_floor_default = bowl_floor.data.default_root_state[env_ids].clone()

    def _sample(n_, lo, hi):
        return torch.empty(n_, device=device).uniform_(lo, hi)

    block_dx = _sample(n, bx_lo, bx_hi)
    block_dy = _sample(n, by_lo, by_hi)
    bowl_dx = _sample(n, wx_lo, wx_hi)
    bowl_dy = _sample(n, wy_lo, wy_hi)

    for _ in range(max_resample):
        block_x = block_default[:, 0] + block_dx
        block_y = block_default[:, 1] + block_dy
        bowl_x = bowl_floor_default[:, 0] + bowl_dx
        bowl_y = bowl_floor_default[:, 1] + bowl_dy
        dist = torch.sqrt((block_x - bowl_x) ** 2 + (block_y - bowl_y) ** 2)
        bad = dist < min_distance
        if not bool(bad.any()):
            break
        n_bad = int(bad.sum())
        block_dx[bad] = _sample(n_bad, bx_lo, bx_hi)
        block_dy[bad] = _sample(n_bad, by_lo, by_hi)
        bowl_dx[bad] = _sample(n_bad, wx_lo, wx_hi)
        bowl_dy[bad] = _sample(n_bad, wy_lo, wy_hi)

    block_state = block_default.clone()
    block_state[:, 0] += block_dx
    block_state[:, 1] += block_dy
    block_state[:, :3] += env.scene.env_origins[env_ids]
    block_asset.write_root_pose_to_sim(block_state[:, :7], env_ids=env_ids)
    block_asset.write_root_velocity_to_sim(block_state[:, 7:], env_ids=env_ids)

    for name in bowl_asset_names:
        asset: RigidObject = env.scene[name]
        state = asset.data.default_root_state[env_ids].clone()
        state[:, 0] += bowl_dx
        state[:, 1] += bowl_dy
        state[:, :3] += env.scene.env_origins[env_ids]
        asset.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(state[:, 7:], env_ids=env_ids)


def record_block_spawn_pose(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    asset_name: str = "block",
) -> None:
    """Snapshot the block's world position at episode reset into
    ``env.block_spawn_pos_w`` (an ``(num_envs, 3)`` tensor, created on first
    call).

    Register as a ``mode="reset"`` event *after* any block-randomization event:
    it reads ``root_pos_w``, so it captures the actual per-env spawn — unlike
    ``default_root_state``, which is the static configured pose and ignores
    randomization. Rewards/terminations that need the spawn read
    ``env.block_spawn_pos_w``.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    block: RigidObject = env.scene[asset_name]
    if not hasattr(env, "block_spawn_pos_w"):
        env.block_spawn_pos_w = torch.zeros(
            env.scene.num_envs, 3, device=env.device
        )
    env.block_spawn_pos_w[env_ids] = block.data.root_pos_w[env_ids].clone()
