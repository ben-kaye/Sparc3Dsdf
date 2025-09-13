import math
from pathlib import Path
from typing import Literal


from sparc3d_sdf import (
    compute_sdf_on_grid,
    load_obj,
    save_obj,
)
from sparc3d_sdf.convert import sdf_to_mesh_dense, sdf_to_mesh_sparse, sdf_to_mesh_voxel
from sparc3d_sdf.sparc3d import calculate_displacements
from sparc3d_sdf.utils import Timer


def main(
    object_path: str,
    N: int,
    output_path: str,
    eta: float = 0.0,
    times: bool = False,
    mode: Literal["flexicubes", "sparse_flexicubes", "voxel"] = "sparse_flexicubes",
):
    object_path = Path(object_path)

    # note this triangulates non manifold faces
    # and rescales to [-0.5, 0.5]
    vertices, faces = load_obj(object_path)

    with Timer(label="SDF computation in..", print_time=times):
        sdf, grid = compute_sdf_on_grid(
            vertices,
            faces,
            resolution=N,
            surface_threshold=math.sqrt(3) / N,
            intermediate_resolutions=[64, 256],
        )

    if eta > 0:
        grid += calculate_displacements(sdf.abs(), eta, clip="tanh", device="cuda")

    with Timer(label="Mesh conversion in..", print_time=times):
        if mode == "sparse_flexicubes":
            vertices, faces = sdf_to_mesh_sparse(sdf, grid, times=times, device="cuda")
        elif mode == "flexicubes":
            vertices, faces = sdf_to_mesh_dense(sdf, times=times, device="cuda")
        elif mode == "voxel":
            vertices, faces = sdf_to_mesh_voxel(sdf, times=times, device="cuda")
        else:
            raise ValueError(f"Invalid mode: {mode}")

    save_obj(output_path, vertices.cpu(), faces.cpu())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--object_path", type=str, required=True)
    parser.add_argument(
        "--N", type=int, required=True, help="Cube resolution of the grid"
    )
    parser.add_argument("-o", "--output_path", type=str, required=True)
    parser.add_argument(
        "--eta",
        type=float,
        required=False,
        default=0.0,
        help="Step size for Sparc3D adjustment step",
    )
    parser.add_argument("--times", action="store_true", help="Print computation times")
    parser.add_argument(
        "--mode",
        type=Literal["flexicubes", "sparse_flexicubes", "voxel"],
        required=False,
        default="sparse_flexicubes",
        help="Mode for mesh extraction",
    )
    args, unparsed = parser.parse_known_args()

    object_path = Path(args.object_path)
    assert object_path.exists(), f"Object path {object_path} does not exist"

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    main(object_path, args.N, output_path, args.eta, args.times, args.mode)
