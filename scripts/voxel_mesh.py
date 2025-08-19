from pathlib import Path

from sparc3d_sdf.obj import load_obj, Timer, save_obj
from sparc3d_sdf.sdf_fast import compute_sdf_on_grid
from sparc3d_sdf.march_voxels import march_voxels

import math


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
            surface_threshold=math.sqrt(3) / N,
            initial_resolution=64,
        )

    with Timer(label="Marching voxels in.."):
        vertices_out, faces_out = (t.cpu() for t in march_voxels((sdf <= 0).cuda()))

    save_obj(output_path, vertices_out, faces_out)


if __name__ == "__main__":
    N = 1024
    main(
        Path("assets/plane.obj"),
        N=N,
        eta=1e-3,
        truncate_distance=0.1,
        output_path=Path(f"assets/plane_vox{N}.obj"),
    )
