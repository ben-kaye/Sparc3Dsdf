import torch
from dataclasses import dataclass
import torch.nn.functional as F
from typing import Literal
import kaolin.ops.conversions as K


@dataclass
class Sparcube:
    grid: torch.Tensor
    displacement: torch.Tensor
    sdf: torch.Tensor
    active_mask: torch.Tensor  # this describes the active vertices in the grid


def get_active(sdf: torch.Tensor, truncate_distance: float):
    active_mask = sdf <= truncate_distance

    return active_mask


def spatial_gradient(udf: torch.Tensor, delta: float):
    """
    compute spatial gradient of udf using finite differences 2nd order approx

    udf: (N, N, N)
    delta: float (grid spacing)

    returns: (N, N, N, 3)
    """

    grads_tuple = torch.gradient(udf.float(), dim=(0, 1, 2), spacing=delta)
    gradient = torch.stack(grads_tuple, dim=-1)

    return gradient


# TODO calculate displacements or gradients only within the active mask
def get_displacements(
    udf: torch.Tensor,
    eta: float,
    clip: bool,
    clip_mode: Literal["abs", "norm"],
) -> torch.Tensor:
    """
    displace vertices in -Grad(udf) direction

    udf: (N, N, N)
    eta: float (step size)
    clip: bool (whether to clip displacements)
    clip_mode: Literal["abs", "norm"] (how to clip displacements)
    clip displacements to be less than 1/2 the grid spacing

    abs mode: clip displacements to be less than 1/2 the grid spacing
    norm mode: clip displacements to have norm less than 1/2 the grid spacing

    returns: (N, N, N, 3)
    """

    delta = 2 / udf.shape[0]
    udf_gradient = spatial_gradient(udf, delta)

    vertex_displacements = -eta * udf_gradient

    if clip:
        if clip_mode == "norm":
            clip_mask = (vertex_displacements.abs() > delta / 2).any(dim=-1)
            vertex_displacements[clip_mask] *= (
                delta / 2 / vertex_displacements[clip_mask].norm(keepdim=True)
            )
        elif clip_mode == "abs":
            vertex_displacements = torch.clamp(
                vertex_displacements, min=-delta / 2, max=delta / 2
            )
    return vertex_displacements


def sdf_to_sparcubes(
    sdf: torch.Tensor,
    grid_xyz: torch.Tensor,
    truncate_distance: float,
    eta: float,
    clip_displacements: bool = True,
    clip_mode: Literal["abs", "norm"] = "norm",
) -> Sparcube:
    """
    sdf: (N, N, N)
    grid_xyz: (N, N, N, 3)
    truncate_distance: float, value at which to truncate the sdf, note only one sided
    eta: float, step size of vertex displacement
    """

    assert truncate_distance > 0, "truncate_distance must be positive"

    active_mask = get_active(sdf, truncate_distance)
    vertex_displacements = get_displacements(
        sdf.abs(), eta, clip=clip_displacements, clip_mode=clip_mode
    )

    return Sparcube(grid_xyz, vertex_displacements, sdf, active_mask)
def generate_dense_cube_indices(res_d, res_h, res_w, device):
    d_coords = torch.arange(res_d - 1, device=device)
    h_coords = torch.arange(res_h - 1, device=device)
    w_coords = torch.arange(res_w - 1, device=device)
    grid_d, grid_h, grid_w = torch.meshgrid(d_coords, h_coords, w_coords, indexing="ij")
    origin_corner_idx = (grid_d * res_h * res_w + grid_h * res_w + grid_w).flatten()
    num_cubes = len(origin_corner_idx)
    cube_idx = torch.empty((num_cubes, 8), dtype=torch.long, device=device)
    cube_idx[:, 0] = origin_corner_idx
    cube_idx[:, 1] = origin_corner_idx + 1
    cube_idx[:, 2] = origin_corner_idx + res_w
    cube_idx[:, 3] = origin_corner_idx + res_w + 1
    cube_idx[:, 4] = origin_corner_idx + res_h * res_w
    cube_idx[:, 5] = origin_corner_idx + res_h * res_w + 1
    cube_idx[:, 6] = origin_corner_idx + res_h * res_w + res_w
    cube_idx[:, 7] = origin_corner_idx + res_h * res_w + res_w + 1
    return cube_idx


def sparcubes_to_mesh_dense(sparcube: Sparcube, device):
    # crop the sdf to the bounding box

    # get vertex positions cropped to the bounding box
    voxelgrid_vertices = (sparcube.grid + sparcube.displacement).reshape(-1, 3)
    # flatten the sdf
    scalar_field = sparcube.sdf.flatten()

    res_d, res_h, res_w = sparcube.sdf.shape

    # generate the cube indices
    dense_cube_idx = generate_dense_cube_indices(
        res_d,
        res_h,
        res_w,
        device=device,
    )

    # run FlexiCubes
    flexicubes = K.FlexiCubes(device=device)

    vertices, faces, _ = flexicubes(
        voxelgrid_vertices=voxelgrid_vertices.to(device),
        scalar_field=scalar_field.to(device),
        cube_idx=dense_cube_idx.to(device),
        resolution=[res_d - 1, res_h - 1, res_w - 1],
    )
    return vertices, faces
