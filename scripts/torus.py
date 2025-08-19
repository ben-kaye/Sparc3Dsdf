from sparc3d_sdf.generics import torus_sdf, rotate_point
import sparc3d_sdf.sdf_fast as sdf_fast
import sparc3d_sdf.march_voxels as march_voxels
import torch
import time
import viser
from pathlib import Path


def main(N: int, R: float, r: float, output_path: Path):
    axis_angle = torch.tensor([0.5, 0.5, 0])
    axis_angle /= axis_angle.norm(keepdim=True)
    axis_angle *= torch.pi / 4

    # vertex grid with 'xy' indexing
    grid_xyz = sdf_fast.vertex_grid(N, indexing="ij")

    basis = rotate_point(grid_xyz, axis_angle)
    sdf = torus_sdf(basis, R, r)

    occupancy = sdf <= 0

    # visualise the occupancy

    vertices, faces = march_voxels.march_voxels(occupancy)

    # visualise with viser
    vertices_np = vertices.cpu().numpy()
    faces_np = faces.cpu().numpy()

    import trimesh

    trimesh.Trimesh(vertices_np, faces_np, process=False).export(output_path)

    server = viser.ViserServer()

    # Add the mesh to the scene with a unique name
    server.add_mesh(
        name="/torus",
        vertices=vertices_np,
        faces=faces_np,
        color=(255, 150, 50),  # Orange color
        wireframe=False,
    )

    print("\nVisualization server running. Check your browser.")
    print("Press Ctrl+C to exit.")

    # Keep the script alive to view the visualization
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting.")


if __name__ == "__main__":
    main(N=128, R=0.3, r=0.13, output_path=Path("torus.obj"))
