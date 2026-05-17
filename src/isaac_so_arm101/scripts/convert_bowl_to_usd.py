# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert the real-scan bowl mesh (``bowl.obj``) into a USD asset.

The squint ManiSkill ``Place`` env loads the bowl directly from ``bowl.obj`` /
``bowl.ply``. Isaac Lab spawns assets from USD, so this one-shot script runs
Isaac Lab's :class:`MeshConverter` to produce ``bowl.usd`` next to the source
mesh (in ``tasks/pick_place/assets``). The bowl is concave, so the collider is
built with convex decomposition — that preserves the inner cavity the cube
must rest inside.

Run once::

    .venv/bin/python -m isaac_so_arm101.scripts.convert_bowl_to_usd
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert bowl.obj to a USD asset.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Force headless — this is a non-interactive conversion.
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

ASSETS_DIR = Path(__file__).resolve().parents[1] / "tasks" / "pick_place" / "assets"
SRC_MESH = ASSETS_DIR / "bowl.obj"


def main() -> None:
    if not SRC_MESH.exists():
        raise FileNotFoundError(f"Source mesh not found: {SRC_MESH}")

    cfg = MeshConverterCfg(
        asset_path=str(SRC_MESH),
        usd_dir=str(ASSETS_DIR),
        usd_file_name="bowl.usd",
        force_usd_conversion=True,
        make_instanceable=False,
        collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
        # Concave bowl — convex decomposition keeps the inner cavity so the
        # cube physically rests inside (a convex hull would fill the bowl in).
        mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(
            max_convex_hulls=32,
            shrink_wrap=True,
        ),
        mass_props=schemas_cfg.MassPropertiesCfg(mass=1.0),
        rigid_props=schemas_cfg.RigidBodyPropertiesCfg(
            rigid_body_enabled=True, disable_gravity=True
        ),
    )
    converter = MeshConverter(cfg)
    print(f"[convert_bowl_to_usd] wrote USD: {converter.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
