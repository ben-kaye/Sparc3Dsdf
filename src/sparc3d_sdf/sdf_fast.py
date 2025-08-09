import einops
import torch
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import index_vertices_by_faces
from scipy.ndimage import label
from typing import Generator, Literal
import sparc3d_sdf.obj as utils


def create_grid(N: int) -> torch.Tensor:
    """
    Create a grid of vertices of N^3 cubes packed in cube of side length 2.

    returns: (N+1, N+1, N+1, 3)
    """

    x, y, z = torch.meshgrid(
        torch.linspace(-1, 1, N + 1),
        torch.linspace(-1, 1, N + 1),
        torch.linspace(-1, 1, N + 1),
        indexing="xy",
    )

    queries = torch.stack((x, y, z), dim=-1)
    return queries


def unsigned_distance_field(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    queries: torch.Tensor,
    device: Literal["auto", "cpu"] | torch.device,
) -> torch.Tensor:
    """
    compute the UDF for a set of queries using kaolin's point_to_mesh_distance

    queries: (N, 3)
    vertices: (M, 3)
    faces: (F, 3)

    returns: (N, )

    warning: mesh but be manifold!
    """

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _verts = vertices[None].float().to(device)
    face_vertices = index_vertices_by_faces(_verts, faces.to(device).long())

    if "cuda" in device:
        try:
            square_dist, *_ = point_to_mesh_distance(
                queries[None].float().to(device), face_vertices
            )
        except torch.cuda.OutOfMemoryError as e:
            # try again with CPU
            print("UDF calculation: CUDA out of memory, defaulting to CPU")
            device = "cpu"

    if device == "cpu":
        square_dist, *_ = point_to_mesh_distance(
            queries.float().cpu(), face_vertices.cpu()
        )

    unsigned_distance = square_dist[0].sqrt().cpu()

    return unsigned_distance


def occupancy_label(udf: torch.Tensor, surface_threshold: float) -> torch.Tensor:
    """
    udf: (N,N,N)
    surface_threshold: float

    returns: (N,N,N)
    """

    N = udf.shape[0]
    assert N == udf.shape[1] == udf.shape[2], "udf must be a cube"
    assert surface_threshold >= 1 / N, "truncation_distance too samll"

    is_near_surface = udf <= surface_threshold
    return _flood_occupancy_grid(is_near_surface)


def _flood_occupancy_grid(
    near_surface_mask: torch.BoolTensor,
    inside_outside_labels: tuple[int, int] = (-1, 1),
) -> torch.Tensor:
    """
    Generates an occupancy grid using SciPy's one-shot connected components algorithm.
    This is the fastest method using standard libraries.


    """
    _inside, _outside = inside_outside_labels

    device = near_surface_mask.device
    resolution = near_surface_mask.shape

    # empty space is anywhere that is NOT the surface wall.
    empty_space = (~near_surface_mask).cpu().numpy()

    # propogate labels to all connected components
    labeled_array, num_features = label(empty_space)

    if num_features < 1:
        # if the grid is fully occupied, return a full grid of inside labels
        return torch.full(resolution, _inside, dtype=torch.int8, device=device)

    # choose a corner that is not in surface mask (eg 0, 0, 0)
    # TODO find a corner that is not in surface mask

    corner_index = (0, 0, 0)
    assert not near_surface_mask[corner_index], "corner is in surface mask"

    outside_id = labeled_array[corner_index]
    is_outside = torch.from_numpy(labeled_array == outside_id)

    # assign results
    occupancy_grid = torch.full(resolution, _inside, dtype=torch.int8)
    occupancy_grid[is_outside] = _outside

    # Move the final result back to the original device.
    return occupancy_grid.to(device)


def compute_sdf_on_grid(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    resolution: int,
    surface_threshold: float,
    print_times: bool = False,
    initial_resolution: int | None = None,
) -> torch.Tensor:
    """
    Compute the SDF on a grid of resolution^3 cubes packed in a cube of side length 2.

    IF initial_resolution is not None, use a staged UDF calculation
        - calculate a coarse UDF, then refine it at the higher resolution

    returns: SDF: (N+1, N+1, N+1), Grid xyz: (N+1, N+1, N+1, 3)
    """

    _debug_times = {}
    grid_xyz = create_grid(resolution)
    flat_grid = einops.rearrange(grid_xyz, "x y z c -> (x y z) c")

    # Fast calculate the UDF using kaolin
    with utils.Timer(label="Calculated UDF in", print_time=print_times) as t:
        if initial_resolution is None:
            udf = unsigned_distance_field(
                vertices,
                faces,
                flat_grid,
            )
        else:
            (
                sparse_udf,
                sparse_grid_xyz,
                sparse_grid_indices,
            ) = sparse_udf(vertices, faces, initial_resolution, resolution)
            udf = torch.zeros_like(grid_xyz[..., 0])
            torch.fill_(udf, 2 * torch.sqrt(torch.tensor(3.0)))
            udf[
                sparse_grid_indices[:, 0],
                sparse_grid_indices[:, 1],
                sparse_grid_indices[:, 2],
            ] = sparse_udf
            udf = einops.rearrange(udf, "n1 n2 n3 -> (n1 n2 n3)")

    _debug_times["udf"] = t.elapsed

    udf = einops.rearrange(
        udf,
        "(n1 n2 n3) -> n1 n2 n3",
        n1=resolution + 1,
        n2=resolution + 1,
        n3=resolution + 1,
    )

    # Generate the occupancy grid using SciPy
    # Flood fill occupancy grid from a corner
    with utils.Timer(label="Generated occupancy grid in", print_time=print_times) as t:
        sign = occupancy_label(udf, surface_threshold)

    _debug_times["occ"] = t.elapsed

    sdf = sign * udf

    return sdf, grid_xyz


def _staged_udf_vertices(
    active_vertices: torch.BoolTensor, initial_resolution: int, resolution: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    get the coordinates of the active vertices with spacing at <resolution>


    Ni is the initial resolution
    N is the final resolution
    active_vertices: (Ni + 1, Ni + 1, Ni + 1)

    returns SPARSE tensor of shape (N + 1, N + 1, N + 1, 3) where the active vertices are the vertices of the grid
    as values (P, 3) and indices (P, 3)
    """

    assert resolution % initial_resolution == 0, (
        "resolution must be a multiple of initial_resolution"
    )

    # naive dense
    dense_grid = create_grid(resolution)
    # upscale the active_vertices to the dense grid

    # TODO / FIXME verify this calculation, we may want to consider any cube with any active vertex instead of cubes with all active vertices
    active_cubes_mask = (
        active_vertices[:-1, :-1, :-1]
        & active_vertices[1:, :-1, :-1]
        & active_vertices[:-1, 1:, :-1]
        & active_vertices[:-1, :-1, 1:]
        & active_vertices[1:, 1:, :-1]
        & active_vertices[1:, :-1, 1:]
        & active_vertices[:-1, 1:, 1:]
        & active_vertices[1:, 1:, 1:]
    )

    scale_factor = resolution // initial_resolution

    # for each active cube, get the vertex indices, then
    def _upscale_cube_indices(
        active_cubes_mask: torch.Tensor, scale_factor: int
    ) -> Generator[torch.Tensor, None, None]:
        """
        upscale the cube indices by the scale factor
        """
        for cube_index_ijk in torch.nonzero(active_cubes_mask, as_tuple=False):
            # index is corner of cube,

            vertex_indices = torch.meshgrid(
                torch.arange(scale_factor),
                torch.arange(scale_factor),
                torch.arange(scale_factor),
                indexing="ij",
            )
            vertex_indices = einops.rearrange(
                torch.stack(vertex_indices, dim=-1), "x y z c -> (x y z) c"
            )

            vertex_indices = vertex_indices + scale_factor * cube_index_ijk

            yield vertex_indices

    vertex_indices = torch.cat(
        list(_upscale_cube_indices(active_cubes_mask, scale_factor))
    )

    return dense_grid[
        vertex_indices[:, 0], vertex_indices[:, 1], vertex_indices[:, 2]
    ], vertex_indices


def sparse_udf(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    initial_resolution: int,
    resolution: int,
    threshold_factor: float = 3,
):
    """
    Compute a sparse UDF at the specified resolution, via a 2 stage calculation:
    - densely compute a UDF at the initial resolution,
    - obtain a sparse voxel mask of active cubes with thresholding
    - refine the UDF at full resolution within the sparse voxel mask

    returns:
    - detailed_udf_values: (P, )
    - detailed_sparse_grid_xyz: (P, 3)
    - detailed_sparse_grid_indices: (P, 3)
    """

    grid_vertices = create_grid(initial_resolution)
    grid_flat = einops.rearrange(grid_vertices, "x y z c -> (x y z) c")
    initial_udf = unsigned_distance_field(vertices, faces, grid_flat)

    spacing = 2 / initial_resolution
    threshold = (
        threshold_factor * torch.sqrt(torch.tensor(3.0)) * spacing
    )  # max distance a cube can resolve * threshold factor

    active_vertices_flat = initial_udf <= threshold
    # evaluate within all the cubes that have an active vertex

    active_vertices = einops.rearrange(
        active_vertices_flat,
        "(n1 n2 n3) -> n1 n2 n3",
        n1=initial_resolution + 1,
        n2=initial_resolution + 1,
        n3=initial_resolution + 1,
    )

    # TODO could/should avoid recomputing the UDF for the active vertices
    # effect matters less at high scale factors..

    detailed_sparse_grid_xyz, detailed_sparse_grid_indices = _staged_udf_vertices(
        active_vertices, initial_resolution, resolution
    )

    detailed_udf_values = unsigned_distance_field(
        vertices, faces, detailed_sparse_grid_xyz
    )

    return detailed_udf_values, detailed_sparse_grid_xyz, detailed_sparse_grid_indices
