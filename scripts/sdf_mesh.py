from pathlib import Path

from sparc3d_sdf import (
    compute_sdf_on_grid,
    convert_dense_to_sparse,
    SparseCube,
    load_obj,
    save_obj,
)
from sparc3d_sdf.utils import Timer
import einops

import math


def main(
    object_path: str, N: int, eta: float, truncate_distance: float, output_path: str
):
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
            einops.rearrange(sdf.cuda().abs() < threshold, "d h w -> (d h w)"),
            cube_resolution,
        )
        sparse_indices = sparse_indices.cpu()
        grid_xyz = einops.rearrange(grid_xyz, "d h w c -> (d h w) c")[sparse_indices]
        sdf = einops.rearrange(sdf, "d h w  -> (d h w)")[sparse_indices]

    with Timer(label="Marching cubes computation took "):
        # visualise the occupancy
        vertices, faces, L_dev = SparseCube()(
            voxelgrid_vertices=grid_xyz.cuda(),
            scalar_field=sdf.cuda(),
            cube_idx=cube_idx,
            adj_idx=adj_idx,
        )

    save_obj(output_path, vertices.cpu(), faces.cpu())


if __name__ == "__main__":
    N = 1024
    main(
        Path("assets/plane.obj"),
        N=N,
        eta=1e-3,
        truncate_distance=0.1,
        output_path=Path(f"assets/plane_sdf{N}.obj"),
    )
