from pathlib import Path

from sparc3d_sdf.obj import load_obj, Timer, save_obj
from sparc3d_sdf.sdf_fast import compute_sdf_on_grid
from sparc3d_sdf.sparc3d import (
    sdf_to_sparcubes,
    sparcubes_to_mesh_dense,
)

import torch


def main(
    object_path: str, N: int, eta: float, truncate_distance: float, output_path: str
):
    object_path = Path(object_path)
    vertices, faces = load_obj(object_path)

    with Timer(label="SDF computation in.."):
        sdf, grid_xyz = compute_sdf_on_grid(
            vertices,
            faces,
            resolution=N,
            surface_threshold=1 / N,
            initial_resolution=64,
        )

    # note eta and truncation distance are not specified in paper

    with Timer(label="SDF adjustment in.."):
        sparcubes = sdf_to_sparcubes(
            sdf, grid_xyz, truncate_distance, eta, clip_displacements=False
        )
    with Timer(label="SDF to mesh in.."):
        try:
            vertices_out, faces_out = sparcubes_to_mesh_dense(sparcubes, "cuda")
        except torch.cuda.OutOfMemoryError as e:
            print(f"CUDA out of memory, defaulting to CPU: {e}")
            vertices_out, faces_out = sparcubes_to_mesh_dense(sparcubes, "cpu")

    save_obj(output_path, vertices_out, faces_out)


if __name__ == "__main__":
    N = 1024
    main(
        Path("assets/plane.obj"),
        N=N,
        eta=1e-3,
        truncate_distance=0.1,
        output_path=Path(f"assets/plane_sdf{N}.obj"),
    )
