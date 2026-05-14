# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reset events for the eval2 task.

* ``reset_paired_blocks_uniform`` — places two cubes adjacent (touching faces),
  sampling the pair's center position and orientation of adjacency.
* ``reset_bowl_assembly_uniform`` — samples one xy offset and applies it to
  every asset in ``asset_names`` so the bowl floor + 8 walls move as a unit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sim.utils import enable_extension

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


def reset_paired_blocks_uniform(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    cube_size: float = 0.02,
    epsilon: float = 0.001,
    block_a_cfg: SceneEntityCfg = SceneEntityCfg("block"),
    block_b_cfg: SceneEntityCfg = SceneEntityCfg("block_b"),
) -> None:
    """Place ``block_a`` at a random delta from its default pose; place
    ``block_b`` flush against it on a randomly chosen horizontal face."""

    n = len(env_ids)
    device = env.device

    block_a: RigidObject = env.scene[block_a_cfg.name]
    block_b: RigidObject = env.scene[block_b_cfg.name]

    x_range = pose_range.get("x", (0.0, 0.0))
    y_range = pose_range.get("y", (0.0, 0.0))
    z_range = pose_range.get("z", (0.0, 0.0))

    dx = torch.empty(n, device=device).uniform_(*x_range)
    dy = torch.empty(n, device=device).uniform_(*y_range)
    dz = torch.empty(n, device=device).uniform_(*z_range)

    a_default = block_a.data.default_root_state[env_ids].clone()
    a_pos = a_default[:, :3] + torch.stack([dx, dy, dz], dim=-1) + env.scene.env_origins[env_ids]
    a_quat = a_default[:, 3:7]

    # adjacency direction in {+x, -x, +y, -y}
    direction = torch.randint(0, 4, (n,), device=device)
    spacing = cube_size + epsilon
    zeros = torch.zeros_like(dx)
    offset_x = torch.where(direction == 0, spacing, torch.where(direction == 1, -spacing, zeros))
    offset_y = torch.where(direction == 2, spacing, torch.where(direction == 3, -spacing, zeros))

    b_pos = a_pos.clone()
    b_pos[:, 0] += offset_x
    b_pos[:, 1] += offset_y
    b_quat = a_quat.clone()

    block_a.write_root_pose_to_sim(torch.cat([a_pos, a_quat], dim=-1), env_ids=env_ids)
    block_b.write_root_pose_to_sim(torch.cat([b_pos, b_quat], dim=-1), env_ids=env_ids)

    zero_vel = torch.zeros(n, 6, device=device)
    block_a.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
    block_b.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)


def reset_bowl_assembly_uniform(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    asset_names: list[str],
) -> None:
    """Sample one xyz delta per env and apply it to every named asset.

    Keeps multi-prim assemblies (e.g. bowl_floor + 8 walls) rigidly together
    while randomizing the assembly's position.
    """
    n = len(env_ids)
    device = env.device

    x_range = pose_range.get("x", (0.0, 0.0))
    y_range = pose_range.get("y", (0.0, 0.0))
    z_range = pose_range.get("z", (0.0, 0.0))

    dx = torch.empty(n, device=device).uniform_(*x_range)
    dy = torch.empty(n, device=device).uniform_(*y_range)
    dz = torch.empty(n, device=device).uniform_(*z_range)
    delta = torch.stack([dx, dy, dz], dim=-1)
    env_origins = env.scene.env_origins[env_ids]
    zero_vel = torch.zeros(n, 6, device=device)

    for name in asset_names:
        asset: RigidObject = env.scene[name]
        default = asset.data.default_root_state[env_ids].clone()
        new_pos = default[:, :3] + delta + env_origins
        new_quat = default[:, 3:7]
        asset.write_root_pose_to_sim(torch.cat([new_pos, new_quat], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)


class randomize_cube_colors_and_target(ManagerTermBase):
    """Pick palette colors for ``block`` and ``block_b``, choose one as the target.

    Each reset:
        1. Per env, sample a palette index for ``block``  → ``color_a``.
        2. Per env, sample a palette index for ``block_b`` → ``color_b`` (≠ ``color_a``).
        3. Per env, sample ``target_idx ∈ {0, 1}`` selecting which cube is the goal.
        4. Apply the colors to the cube meshes (Replicator material override).
        5. Push ``env.target_idx`` and ``env.target_color`` for observations/rewards.

    Requires ``InteractiveSceneCfg.replicate_physics = False`` so each env has
    its own cube prim that can be re-materialed independently.

    Replicator's ``modify.attribute`` applies to *all* envs at once, so we keep
    long-lived buffers and only overwrite the indices for ``env_ids``.
    """

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        enable_extension("omni.replicator.core")
        import omni.replicator.core as rep
        try:
            from isaacsim.core.utils.stage import get_current_stage
        except ImportError:
            from omni.isaac.core.utils.stage import get_current_stage

        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "randomize_cube_colors_and_target needs replicate_physics=False."
            )

        block_a_cfg: SceneEntityCfg = cfg.params["block_a_cfg"]
        block_b_cfg: SceneEntityCfg = cfg.params["block_b_cfg"]
        asset_a = env.scene[block_a_cfg.name]
        asset_b = env.scene[block_b_cfg.name]

        stage = get_current_stage()
        prims_a = rep.functional.get.prims(path_pattern=asset_a.cfg.prim_path, stage=stage)
        prims_b = rep.functional.get.prims(path_pattern=asset_b.cfg.prim_path, stage=stage)
        for prim in list(prims_a) + list(prims_b):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)

        self.materials_a = rep.functional.create_batch.material(
            mdl="OmniPBR.mdl", bind_prims=prims_a, count=len(prims_a), project_uvw=True
        )
        self.materials_b = rep.functional.create_batch.material(
            mdl="OmniPBR.mdl", bind_prims=prims_b, count=len(prims_b), project_uvw=True
        )

        palette = cfg.params["palette"]
        self.palette = torch.tensor(palette, dtype=torch.float32, device=env.device)

        n = env.num_envs
        self.color_idx_a = torch.zeros(n, dtype=torch.long, device=env.device)
        self.color_idx_b = torch.ones(n, dtype=torch.long, device=env.device)
        self.target_idx_buf = torch.zeros(n, dtype=torch.long, device=env.device)
        # initial public buffers
        env.target_idx = self.target_idx_buf
        env.target_color = self.palette[self.color_idx_a].clone()

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        palette: list[tuple[float, float, float]],
        block_a_cfg: SceneEntityCfg = SceneEntityCfg("block"),
        block_b_cfg: SceneEntityCfg = SceneEntityCfg("block_b"),
    ):
        import omni.replicator.core as rep

        n_palette = len(self.palette)
        n_reset = len(env_ids)
        device = env.device

        new_a = torch.randint(0, n_palette, (n_reset,), device=device)
        offset = torch.randint(1, n_palette, (n_reset,), device=device)
        new_b = (new_a + offset) % n_palette
        new_target = torch.randint(0, 2, (n_reset,), device=device)

        self.color_idx_a[env_ids] = new_a
        self.color_idx_b[env_ids] = new_b
        self.target_idx_buf[env_ids] = new_target

        a_colors = self.palette[self.color_idx_a]
        b_colors = self.palette[self.color_idx_b]
        is_a = (self.target_idx_buf == 0).unsqueeze(-1)
        env.target_idx = self.target_idx_buf
        env.target_color = torch.where(is_a, a_colors, b_colors)

        rep.functional.modify.attribute(
            self.materials_a, "diffuse_color_constant", a_colors.cpu().numpy()
        )
        rep.functional.modify.attribute(
            self.materials_b, "diffuse_color_constant", b_colors.cpu().numpy()
        )
