"""One-shot: copy per-vertex colors from the broken backups onto the repaired bowls.

The original real-scan bowl has per-vertex colors (mean RGB ~ 174,171,165 — a
warm beige). The voxel-remesh + decimation pipeline in ``repair_bowl.py``
generates new vertex positions, dropping the colors. This script samples each
repaired vertex's color from the nearest source vertex in the broken backup,
then rewrites bowl.obj/.ply in both project locations.

Run with:
    .venv/bin/python scripts/colorize_bowl.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

SQUINT_DIR = Path('/home/kenzy/singulation-project/squint/envs/meshes')
ISAAC_DIR  = Path('/home/kenzy/isaac_so_arm101/src/isaac_so_arm101/tasks/pick_place/assets')


def transfer_colors(target: trimesh.Trimesh, source: trimesh.Trimesh) -> None:
    src_visual = getattr(source, 'visual', None)
    if src_visual is None or src_visual.kind != 'vertex':
        raise RuntimeError('source mesh has no per-vertex colors')
    src_colors = np.asarray(src_visual.vertex_colors)
    tree = cKDTree(source.vertices)
    _, idx = tree.query(target.vertices, k=1)
    target.visual = trimesh.visual.color.ColorVisuals(
        mesh=target, vertex_colors=src_colors[idx]
    )


def main() -> None:
    # Source of truth for color is the broken OBJ backup (5411 colored verts).
    src_path = SQUINT_DIR / 'bowl.obj.broken.bak'
    source = trimesh.load(src_path, process=False, file_type='obj')
    print(f'source: {src_path.name}  v={len(source.vertices)}  '
          f'mean RGB={np.asarray(source.visual.vertex_colors)[:, :3].mean(axis=0).round(1)}')

    targets = [
        (SQUINT_DIR / 'bowl.obj', 'obj'),
        (SQUINT_DIR / 'bowl.ply', 'ply'),
        (ISAAC_DIR  / 'bowl.obj', 'obj'),
    ]
    for tgt_path, ext in targets:
        m = trimesh.load(tgt_path, process=False, file_type=ext)
        transfer_colors(m, source)
        m.export(tgt_path)
        vc = np.asarray(m.visual.vertex_colors)[:, :3]
        print(f'wrote {tgt_path}  v={len(m.vertices)}  '
              f'mean RGB={vc.mean(axis=0).round(1)}')

    # Invalidate the cached USD asset hash so convert_bowl_to_usd.py rebuilds.
    asset_hash = ISAAC_DIR / '.asset_hash'
    if asset_hash.exists():
        asset_hash.unlink()
        print(f'removed {asset_hash}  (USD will rebuild on next conversion)')


if __name__ == '__main__':
    main()
