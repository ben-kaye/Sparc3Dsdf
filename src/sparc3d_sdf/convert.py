import math
from typing import Literal

import einops
import torch
from kaolin.ops.conversions.flexicubes import FlexiCubes

from sparc3d_sdf import (
    SparseCube,
    convert_dense_to_sparse,
)
from sparc3d_sdf.sparc3d import calculate_displacements
from sparc3d_sdf.utils import Timer
from sparc3d_sdf.voxelize import march_voxels


def _flatten_dhw(tensor: torch.Tensor):
    return einops.rearrange(tensor, "d h w ... -> (d h w) ...")


def _unflatten_dhw(tensor: torch.Tensor, shape: tuple[int, int, int]):
    return einops.rearrange(
        tensor, "(d h w) ... -> d h w ...", d=shape[0], h=shape[1], w=shape[2]
    )


def sdf_to_mesh_sparse(
    sdf: torch.Tensor,
    grid: torch.Tensor,
    threshold: float | None = None,
    device: Literal["cuda", "cpu"] = "cuda",
):
    """
    Extract a mesh from a grid (D, H, W) based SDF.

    args:
        sdf: (D + 1, H + 1, W + 1) - SDF grid
        grid: (D + 1, H + 1, W+1, 3) - vertex grid positions, cartesian
        threshold: float - threshold for truncated SDF
        times: bool -print computation times
        device: str
    """

    # threshold of UDF for ssparse extraction
    vertex_resolution = grid.shape[:3]
    cube_resolution = tuple(r - 1 for r in vertex_resolution)
    if threshold is None:
        threshold = math.sqrt(3) / cube_resolution[0]

    sparse_indices, cube_idx, adj_idx = convert_dense_to_sparse(
        _flatten_dhw(sdf.to(device).abs() < threshold),
        cube_resolution,
    )

    sparse_indices, cube_idx, adj_idx = (
        t.cpu() for t in (sparse_indices, cube_idx, adj_idx)
    )

    grid_xyz = _flatten_dhw(grid)[sparse_indices]

    sdf = _flatten_dhw(sdf)[sparse_indices]

    vertices, faces, L_dev = SparseCube()(
        voxelgrid_vertices=grid_xyz.to(device),
        scalar_field=sdf.to(device),
        cube_idx=cube_idx.cuda(),
        adj_idx=adj_idx.to(device),
    )

    return vertices, faces


def sdf_to_mesh_dense(
    sdf: torch.Tensor,
    threshold: float | None = None,
    device: Literal["cuda", "cpu"] = "cuda",
):
    """
    Extract a mesh from a grid (D, H, W) based SDF.

    Using Flexicubes, note that this requires the SDF to be on the DHW grid in range [-0.5, 0.5]
    """

    vertex_resolution = sdf.shape
    cube_resolution = tuple(r - 1 for r in vertex_resolution)

    flexi = FlexiCubes(device=device)
    grid, cube_idx = flexi.construct_voxel_grid(cube_resolution)

    vertices, faces = flexi(
        voxelgrid_vertices=grid.to(device),
        scalar_field=sdf.to(device),
        cube_idx=cube_idx.to(device),
        resolution=cube_resolution,
    )

    return vertices, faces


def sdf_to_mesh_voxel(
    sdf: torch.Tensor,
    device: Literal["cuda", "cpu"] = "cuda",
):
    """
    Extract a mesh from a grid (D, H, W) based SDF.
    """

    vertices, faces = march_voxels((sdf <= 0).to(device))

    return vertices, faces


def apply_sparc3d_refinement(
    sdf: torch.Tensor,
    grid: torch.Tensor,
    eta: float,
    device: Literal["cuda", "cpu"] = "cuda",
):
    assert eta > 0.0

    displacements = _flatten_dhw(
        calculate_displacements(sdf.abs(), eta, clip="tanh", device=device)
    )
    grid = grid.to(device) + displacements

    return grid
