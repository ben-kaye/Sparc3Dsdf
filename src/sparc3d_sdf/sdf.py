"""
Dense SDF computation.

Uses algorithm described in Sparc3D <arXiv:2505.14521 [cs.CV]>
1. Calculate UDF on a grid.
2. Label the exterior with a flood-fill algorithm.
"""

import einops
import torch
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import index_vertices_by_faces
from scipy.ndimage import label
from typing import Literal
import sparc3d_sdf.utils as utils
import math


def vertex_grid(
    L: int,
    indexing: Literal["xy", "ij"] = "ij",
    limits: tuple[float, float] = (-0.5, 0.5),
) -> torch.Tensor:
    """
    Create a grid of vertices of L^3 cubes packed in cube of side length 1.0.

    returns: (L+1, L+1, L+1, 3)
    """

    if indexing == "xy":
        raise ValueError()

    x, y, z = torch.meshgrid(
        torch.linspace(*limits, L + 1),
        torch.linspace(*limits, L + 1),
        torch.linspace(*limits, L + 1),
        indexing=indexing,
    )

    queries = torch.stack((x, y, z), dim=-1)
    return queries


def unsigned_distance_field(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    queries: torch.Tensor,
    device: Literal["auto", "cpu"] | torch.device = "auto",
) -> torch.Tensor:
    """
    compute the UDF for a set of queries using kaolin's point_to_mesh_distance

    queries: (N, 3)
    vertices: (M, 3)
    faces: (F, 3)

    returns: (N, ) on CPU

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
        except torch.cuda.OutOfMemoryError:
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
    initial_resolution: int | list[int] | None = None,
) -> torch.Tensor:
    """
    Compute the SDF on a grid of resolution^3 cubes packed in a cube of side length 2.

    IF initial_resolution is not None, use a staged UDF calculation
        - calculate a coarse UDF, then refine it at the higher resolution, within the thre

    returns: SDF: (N+1, N+1, N+1), Grid xyz: (N+1, N+1, N+1, 3)
    """

    _debug_times = {}
    grid_xyz = vertex_grid(resolution)
    flat_grid = einops.rearrange(grid_xyz, "x y z c -> (x y z) c")

    # Fast calculate the UDF using kaolin
    with utils.Timer(label="Calculated UDF in", print_time=print_times) as t:
        if initial_resolution is None:
            udf = unsigned_distance_field(
                vertices,
                faces,
                flat_grid,
            )

            udf = einops.rearrange(
                udf,
                "(n1 n2 n3) -> n1 n2 n3",
                n1=resolution + 1,
                n2=resolution + 1,
                n3=resolution + 1,
            )

        else:
            if isinstance(initial_resolution, int):
                resolutions = [initial_resolution, resolution]
            else:
                resolutions = [*initial_resolution, resolution]
            udf = _iterative_udf(vertices, faces, resolutions)

    _debug_times["udf"] = t.elapsed

    # Generate the occupancy grid using SciPy
    # Flood fill occupancy grid from a corner
    with utils.Timer(label="Generated occupancy grid in", print_time=print_times) as t:
        sign = occupancy_label(udf, surface_threshold)

    _debug_times["occ"] = t.elapsed

    sdf = sign * udf

    return sdf, grid_xyz


def get_active_cubes_exclusive(active_vertices: torch.BoolTensor) -> torch.BoolTensor:
    """
    return mask where active cube has all positive vertices
    """
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

    return active_cubes_mask


def get_active_cubes_inclusive(active_vertices: torch.BoolTensor) -> torch.BoolTensor:
    """
    return mask where active cube has any positive vertices
    """
    active_cubes_mask = (
        active_vertices[:-1, :-1, :-1]
        | active_vertices[1:, :-1, :-1]
        | active_vertices[:-1, 1:, :-1]
        | active_vertices[:-1, :-1, 1:]
        | active_vertices[1:, 1:, :-1]
        | active_vertices[1:, :-1, 1:]
        | active_vertices[:-1, 1:, 1:]
        | active_vertices[1:, 1:, 1:]
    )

    return active_cubes_mask


def _compute_upscale_coordinates(
    active_vertices: torch.BoolTensor,
    initial_resolution: int,
    resolution: int,
    mode: Literal["inclusive", "exclusive"] = "inclusive",
) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    get the coordinates of the active vertices with spacing at <resolution>



    active_vertices: (Ni + 1, Ni + 1, Ni + 1)
    initial_resolution: int (Ni)
    resolution: int (N)
    mode: Literal["inclusive", "exclusive"] ->
        inclusive: active cube must have ANY positive vertex
        exclusive: active cube must have ALL positive vertices

    returns:
        original_coordinates: (P, 3) coordinates of the computed vertices in the initial resolution grid
        new_coordinates: (P, 3) coordinates of computed vertices in the next resolution grid
        next_coordinates: (P, 3) coordinates to compute the UDF at the next resolution
    """

    assert resolution % initial_resolution == 0, (
        "resolution must be a multiple of initial_resolution"
    )

    scale_factor = resolution // initial_resolution

    active_cubes_mask = (
        get_active_cubes_exclusive(active_vertices)
        if mode == "exclusive"
        else get_active_cubes_inclusive(active_vertices)
    )

    vertex_offsets = torch.meshgrid(
        torch.arange(scale_factor),
        torch.arange(scale_factor),
        torch.arange(scale_factor),
        indexing="ij",
    )

    # drop the first vertex as that is the already computed value
    vertex_offsets = einops.rearrange(
        torch.stack(vertex_offsets, dim=-1), "x y z c ->  1 (x y z) c"
    )[:, 1:]

    original_coordinates = torch.nonzero(active_cubes_mask, as_tuple=False)
    new_coordinates = scale_factor * original_coordinates

    next_coordinates = new_coordinates[:, None] + vertex_offsets
    next_coordinates = einops.rearrange(next_coordinates, "n l c -> (n l) c")

    return original_coordinates, new_coordinates, next_coordinates


def _iterative_udf(vertices, faces, resolutions: list[int]):
    """
    compute a dense UDF sparsely by iteratively refining the active UDF vertices at each resolution
    """

    initial_resolution, *resolutions = resolutions
    grid = einops.rearrange(vertex_grid(initial_resolution), "x y z c -> (x y z) c")
    udf = unsigned_distance_field(vertices, faces, grid)
    udf = einops.rearrange(
        udf,
        "(n1 n2 n3) -> n1 n2 n3",
        n1=initial_resolution + 1,
        n2=initial_resolution + 1,
        n3=initial_resolution + 1,
    )

    for next_resolution in resolutions:
        udf = _iterative_udf_k(
            vertices, faces, udf, initial_resolution, next_resolution
        )
        initial_resolution = next_resolution

    return udf


def _iterative_udf_k(
    vertices: torch.FloatTensor,
    faces: torch.LongTensor,
    udf: torch.FloatTensor,
    resolution: int,
    next_resolution: int,
) -> torch.FloatTensor:
    """
    udf: (N0+1, N0+1, N0+1)
    resolution: int (N0)
    next_resolution: int (N1)

    returns: (N1+1, N1+1, N1+1)

    """
    assert udf.dim() == 3

    # threshold is the diagonal length of a voxel
    threshold = math.sqrt(3) / resolution
    mask = udf <= threshold
    indices, indices_at_next, next_indices = _compute_upscale_coordinates(
        mask, resolution, next_resolution
    )
    next_udf = torch.full(
        (next_resolution + 1, next_resolution + 1, next_resolution + 1),
        fill_value=math.sqrt(3),
        device=udf.device,
    )
    next_udf[indices_at_next[:, 0], indices_at_next[:, 1], indices_at_next[:, 2]] = udf[
        indices[:, 0], indices[:, 1], indices[:, 2]
    ]
    next_grid = vertex_grid(next_resolution)[
        next_indices[:, 0], next_indices[:, 1], next_indices[:, 2]
    ]
    next_udf[next_indices[:, 0], next_indices[:, 1], next_indices[:, 2]] = (
        unsigned_distance_field(vertices, faces, next_grid)
    )

    return next_udf
