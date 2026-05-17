"""Repair the broken bowl mesh.

The input bowl.obj has 12 disconnected bodies and 190 boundary loops, so it is
a thin shell that has been shattered into many fragments. The bowl is HOLLOW
(not solid), so we must not simply fill it.

Pipeline
--------
1. Voxelise the surface, then morphologically CLOSE (dilate -> erode). This
   bridges the gaps between fragments without filling the cavity.
2. Run marching cubes back to a triangle mesh and apply the voxel transform
   so the result is in metres again.
3. Decimate to a sim-friendly face count (fast_simplification quadric).
4. Patch any small holes the decimator opened with pymeshfix (joincomp=False,
   keep all components — at this stage the input is already a single body so
   pymeshfix is purely additive).
5. Validate, back up the originals, then write to both copies of bowl.obj
   (squint env + isaac assets) and the squint .ply.
6. Drop the cached USD asset hash so convert_bowl_to_usd.py rebuilds bowl.usd.

Run with the project's uv venv:
    .venv/bin/python scripts/repair_bowl.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pymeshfix
import trimesh
from scipy import ndimage

# --- locations -----------------------------------------------------------------
SQUINT_DIR = Path('/home/kenzy/singulation-project/squint/envs/meshes')
ISAAC_DIR  = Path('/home/kenzy/isaac_so_arm101/src/isaac_so_arm101/tasks/pick_place/assets')

SRC_OBJ = SQUINT_DIR / 'bowl.obj'   # both copies are identical (md5 verified)

# --- repair parameters ---------------------------------------------------------
VOXEL_PITCH      = 0.001    # 1 mm voxels (bowl is ~0.15 m wide)
VOXEL_CLOSE_ITER = 1        # 1 dilate + 1 erode bridges gaps up to 2 mm
TARGET_FACES     = 8000     # well below the 11985 input count, plenty for sim


# --- helpers -------------------------------------------------------------------
def stat(label: str, m: trimesh.Trimesh) -> None:
    vol = float(m.volume) if m.is_volume else float('nan')
    ext = m.bounds[1] - m.bounds[0]
    print(f'{label:<22s} v={len(m.vertices):6d} f={len(m.faces):6d} '
          f'bodies={m.body_count:2d} watertight={str(m.is_watertight):5s} '
          f'vol={vol:.3e} m^3 extents={np.round(ext, 4).tolist()}')


def voxel_close(mesh: trimesh.Trimesh, pitch: float, n_close: int) -> trimesh.Trimesh:
    """Surface voxelise -> morphological closing -> marching cubes (no fill)."""
    vg = mesh.voxelized(pitch=pitch)
    mat = ndimage.binary_dilation(vg.matrix, iterations=n_close)
    mat = ndimage.binary_erosion(mat,        iterations=n_close)
    vg2 = trimesh.voxel.VoxelGrid(mat, transform=vg.transform)
    rm = vg2.marching_cubes
    rm.apply_transform(vg.transform)   # marching_cubes returns voxel-index coords
    rm.merge_vertices()
    rm.remove_unreferenced_vertices()
    rm.fix_normals()
    return rm


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh
    out = mesh.simplify_quadric_decimation(face_count=target_faces, aggression=3)
    out.merge_vertices()
    out.remove_unreferenced_vertices()
    out.fix_normals()
    return out


def patch_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Patch small holes opened by decimation. Safe because input is single-body."""
    mf = pymeshfix.MeshFix(mesh.vertices.copy(), mesh.faces.copy())
    mf.repair(joincomp=False, remove_smallest_components=False)
    out = trimesh.Trimesh(vertices=mf.points, faces=mf.faces, process=True)
    out.merge_vertices()
    out.remove_unreferenced_vertices()
    out.fix_normals()
    return out


def transfer_vertex_colors(target: trimesh.Trimesh, source: trimesh.Trimesh) -> None:
    """Sample per-vertex colors from ``source`` onto ``target`` by nearest neighbour.

    Voxel remeshing creates new vertices, so colors must be resampled by
    position. The original real-scan bowl carries per-vertex RGBA, which the
    USD MeshConverter preserves as a ``displayColor`` primvar.
    """
    src_visual = getattr(source, 'visual', None)
    if src_visual is None or src_visual.kind != 'vertex':
        return
    src_colors = np.asarray(src_visual.vertex_colors)
    if src_colors is None or len(src_colors) != len(source.vertices):
        return
    from scipy.spatial import cKDTree
    tree = cKDTree(source.vertices)
    _, idx = tree.query(target.vertices, k=1)
    target.visual = trimesh.visual.color.ColorVisuals(
        mesh=target, vertex_colors=src_colors[idx]
    )

def backup(path: Path) -> None:
    if not path.exists():
        return
    bak = path.with_suffix(path.suffix + '.broken.bak')
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f'  backup -> {bak.name}')


# --- pipeline ------------------------------------------------------------------
def main() -> None:
    src = trimesh.load(SRC_OBJ, process=False)
    stat('INPUT', src)
    src_ext = src.bounds[1] - src.bounds[0]

    # Pick the colored "ground truth" we'll resample onto the repaired mesh.
    # Prefer the broken backup if the source has already been overwritten by a
    # previous run (in which case the source is colorless or coarse).
    color_src_path = SRC_OBJ.with_suffix(SRC_OBJ.suffix + '.broken.bak')
    if color_src_path.exists():
        color_src = trimesh.load(color_src_path, process=False, file_type='obj')
    else:
        color_src = src
    has_colors = (
        getattr(color_src, 'visual', None) is not None
        and color_src.visual.kind == 'vertex'
    )

    closed = voxel_close(src, VOXEL_PITCH, VOXEL_CLOSE_ITER)
    stat('after voxel-close', closed)

    decim = decimate(closed, TARGET_FACES)
    stat('after decimate', decim)

    final = patch_holes(decim)
    stat('FINAL', final)

    if has_colors:
        transfer_vertex_colors(final, color_src)
        print('  applied per-vertex colors from real-scan source')

    # Validate BEFORE touching disk so a failure doesn't trash the source files.
    if not final.is_watertight or final.body_count != 1:
        raise RuntimeError(
            f'repair did not converge: watertight={final.is_watertight} '
            f'bodies={final.body_count}'
        )
    fin_ext = final.bounds[1] - final.bounds[0]
    rel_err = float(np.linalg.norm(fin_ext - src_ext) / np.linalg.norm(src_ext))
    print(f'\nextent change vs input: {rel_err*100:.2f}%')
    if rel_err > 0.05:
        raise RuntimeError(f'repaired bowl differs from input by {rel_err*100:.1f}%')

    # Write to both project locations.
    targets = [
        SQUINT_DIR / 'bowl.obj',
        SQUINT_DIR / 'bowl.ply',
        ISAAC_DIR  / 'bowl.obj',
    ]
    for tgt in targets:
        backup(tgt)
        final.export(tgt)
        print(f'wrote {tgt}')

    # Invalidate the cached USD/asset hash so convert_bowl_to_usd.py rebuilds.
    asset_hash = ISAAC_DIR / '.asset_hash'
    if asset_hash.exists():
        asset_hash.unlink()
        print(f'removed {asset_hash}  (USD will rebuild on next launch)')


if __name__ == '__main__':
    main()
