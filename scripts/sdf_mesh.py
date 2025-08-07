from pathlib import Path

from sparc3d_sdf.obj import load_obj
from sparc3d_sdf.sdf_fast import compute_sdf_on_grid
from sparc3d_sdf.sparc3d import sdf_to_sparcubes


def main(object_path: str, N: int, eta: float, truncate_distance: float):
    object_path = Path(object_path)
    vertices, faces = load_obj(object_path)

    sdf, grid_xyz = compute_sdf_on_grid(
        vertices, faces, resolution=N, surface_threshold=1 / N
    )

    # note eta and truncation distance are not specified in paper

    sparcubes = sdf_to_sparcubes(sdf, grid_xyz, truncate_distance, eta)


if __name__ == "__main__":
    main(Path("assets/plane.obj"), N=128, eta=1e-2, truncate_distance=0.1)
