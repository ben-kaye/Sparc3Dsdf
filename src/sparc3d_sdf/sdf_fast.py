import einops
import torch
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import index_vertices_by_faces
from scipy.ndimage import label

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
    vertices: torch.Tensor, faces: torch.Tensor, queries: torch.Tensor
) -> torch.Tensor:
    """
    warning: mesh but be manifold!
    """

    _verts = vertices[None].float().cuda()
    face_vertices = index_vertices_by_faces(_verts, faces.cuda().long())
    square_dist, index, dist_type = point_to_mesh_distance(
        queries.cuda()[None].float(), face_vertices.cuda()
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


def create_sdf_grid(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    resolution: int,
    surface_threshold: float,
    print_times: bool = False,
) -> torch.Tensor:
    _debug_times = {}
    grid_xyz = create_grid(resolution)
    flat_grid = einops.rearrange(grid_xyz, "x y z c -> (x y z) c")

    # Fast calculate the UDF using kaolin
    with utils.Timer(label="Calculated UDF in", print_time=print_times) as t:
        udf = unsigned_distance_field(
            vertices,
            faces,
            flat_grid,
        )

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
