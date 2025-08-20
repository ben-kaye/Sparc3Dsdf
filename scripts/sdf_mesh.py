from pathlib import Path

from sparc3d_sdf import (
    compute_sdf_on_grid,
    convert_dense_to_sparse,
    SparseCube,
    load_obj,
    save_obj,
)
from sparc3d_sdf.utils import Timer
from sparc3d_sdf.sparc3d import calculate_displacements
import einops

import math
import torch


def _flatten_dhw(tensor: torch.Tensor):
    return einops.rearrange(tensor, "d h w ... -> (d h w) ...")


def _unflatten_dhw(tensor: torch.Tensor, shape: tuple[int, int, int]):
    return einops.rearrange(
        tensor, "(d h w) ... -> d h w ...", d=shape[0], h=shape[1], w=shape[2]
    )


def main(object_path: str, N: int, output_path: str, eta: float = 0.0):
    object_path = Path(object_path)
    vertices, faces = load_obj(object_path)

    threshold = math.sqrt(3) / N

    with Timer(label="SDF computation took "):
        sdf, grid_xyz = compute_sdf_on_grid(
            vertices,
            faces,
            resolution=N,
            surface_threshold=threshold,
            initial_resolution=[64, 256],
        )

    # threshold of UDF for sparse extraction
    vertex_resolution = grid_xyz.shape[:3]
    cube_resolution = tuple(r - 1 for r in vertex_resolution)

    with Timer(label="Sparse conversion took "):
        sparse_indices, cube_idx, adj_idx = convert_dense_to_sparse(
            _flatten_dhw(sdf.cuda().abs() < threshold),
            cube_resolution,
        )

    sparse_indices, cube_idx, adj_idx = (
        t.cpu() for t in (sparse_indices, cube_idx, adj_idx)
    )

    grid_xyz = _flatten_dhw(grid_xyz)[sparse_indices]

    if eta:
        # do not use! waste of memory and time
        displacements = _flatten_dhw(
            calculate_displacements(sdf.abs(), eta, clip="tanh", device="cuda")
        )[sparse_indices]
        grid_xyz = grid_xyz.cuda() + displacements

    sdf = _flatten_dhw(sdf)[sparse_indices]

    with Timer(label="Marching cubes computation took "):
        # visualise the occupancy
        vertices, faces, L_dev = SparseCube()(
            voxelgrid_vertices=grid_xyz.cuda(),
            scalar_field=sdf.cuda(),
            cube_idx=cube_idx.cuda(),
            adj_idx=adj_idx.cuda(),
        )

    save_obj(output_path, vertices.cpu(), faces.cpu())


if __name__ == "__main__":
    N = 1024
    main(
        Path("assets/plane.obj"),
        N=N,
        output_path=Path(f"assets/plane_sdf{N}.obj"),
        eta=0.0,
    )
