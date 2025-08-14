"""
implement a voxelized version of the marching cubes algorithm
memory efficiently!
"""

import torch
import torch.nn.functional as F
import einops
from typing import Literal


def _get_adjacency(matrix: torch.BoolTensor):
    """
    Checks the state of the 6 adjacent voxels for each voxel in a grid
    that uses standard 'ij' indexing (X, Y, Z memory layout).
    """
    assert matrix.dim() == 3

    matrix = F.pad(matrix, (1, 1, 1, 1, 1, 1), mode="constant", value=False)
    # This simplified implementation is great! It directly creates the (X, Y, Z, 6) tensor.
    adjacency = torch.stack(
        [
            matrix[:-2, 1:-1, 1:-1],  # x-1
            matrix[2:, 1:-1, 1:-1],  # x+1
            matrix[1:-1, :-2, 1:-1],  # y-1
            matrix[1:-1, 2:, 1:-1],  # y+1
            matrix[1:-1, 1:-1, :-2],  # z-1
            matrix[1:-1, 1:-1, 2:],  # z+1
        ],
        dim=-1,
    )
    return adjacency


def _face_mask(occupied_voxels: torch.BoolTensor):
    """
    compute face mask memory efficiently

    occupied_voxels: (X, Y, Z)
    returns: (N, 4)

    effectively implmenting:
    face_mask = occupied_voxels[..., None] & ~_get_adjacency(occupied_voxels)
    coordinates = torch.nonzero(face_mask, as_tuple=False)
    return coordinates
    """

    _forward = slice(2, None)
    _backward = slice(0, -2)
    _identity = slice(1, -1)
    _slices = [
        [
            _backward,
            _identity,
            _identity,
        ],
        [
            _forward,
            _identity,
            _identity,
        ],
        [
            _identity,
            _backward,
            _identity,
        ],
        [
            _identity,
            _forward,
            _identity,
        ],
        [
            _identity,
            _identity,
            _backward,
        ],
        [
            _identity,
            _identity,
            _forward,
        ],
    ]

    occupied_voxels = F.pad(
        occupied_voxels, (1, 1, 1, 1, 1, 1), mode="constant", value=False
    )

    result = []
    for k, _slice in enumerate(_slices):
        coordinates = torch.nonzero(
            occupied_voxels[_identity, _identity, _identity]
            & ~occupied_voxels[_slice[0], _slice[1], _slice[2]]
        )
        coordinates = F.pad(coordinates, (0, 1), mode="constant", value=k)
        result.append(coordinates)
    return torch.cat(result, dim=0)


def _get_vertices(grid: torch.BoolTensor):
    """
    Gathers the 8 corner vertices for each voxel in the grid.
    """
    # FIX: torch.stack expects a list or tuple of tensors.
    return torch.stack(
        [
            grid[:-1, :-1, :-1],  # v0 (x,y,z)
            grid[:-1, :-1, 1:],  # v1 (x,y,z+1)
            grid[:-1, 1:, :-1],  # v2 (x,y+1,z)
            grid[:-1, 1:, 1:],  # v3 (x,y+1,z+1)
            grid[1:, :-1, :-1],  # v4 (x+1,y,z)
            grid[1:, :-1, 1:],  # v5 (x+1,y,z+1)
            grid[1:, 1:, :-1],  # v6 (x+1,y+1,z)
            grid[1:, 1:, 1:],  # v7 (x+1,y+1,z+1)
        ],
        dim=-1,
    )


def _faces_triangles():
    """
    Generates constant triangle indices for the 6 faces of a cube.
    The winding order is corrected to produce consistent outward-facing normals.

    Vertex Order (from meshgrid 'ij'):
    v0:(0,0,0), v1:(0,0,1), v2:(0,1,0), v3:(0,1,1)
    v4:(1,0,0), v5:(1,0,1), v6:(1,1,0), v7:(1,1,1)
    """
    # CRITICAL FIX: The winding order is corrected for outward-facing normals.
    triangles = torch.tensor(
        [
            # -x face (v0, v1, v2, v3)
            [0, 1, 3, 0, 3, 2],
            # +x face (v4, v5, v6, v7)t
            [4, 6, 7, 4, 7, 5],
            # -y face (v0, v1, v4, v5)
            [0, 4, 5, 0, 5, 1],
            # +y face (v2, v3, v6, v7)
            [2, 3, 7, 2, 7, 6],
            # -z face (v0, v2, v4, v6)
            [0, 2, 6, 0, 6, 4],
            # +z face (v1, v3, v5, v7)
            [1, 5, 7, 1, 7, 3],
        ],
        dtype=torch.long,
    )
    return triangles


def _faces_vertices(face_coordinates: torch.LongTensor, spacing: float):
    if face_coordinates.shape[0] == 0:
        return torch.empty((0, 3), device=face_coordinates.device), torch.empty(
            (0, 3), dtype=torch.long, device=face_coordinates.device
        )
    device = face_coordinates.device
    voxel_coords = face_coordinates[:, :3]
    face_indices = face_coordinates[:, 3]

    _basis = torch.arange(2, device=device, dtype=torch.long)
    basis = torch.stack(torch.meshgrid(_basis, _basis, _basis, indexing="ij"), dim=-1)
    basis = einops.rearrange(basis, "x y z d -> (x y z) d")
    voxel_vertex_coords = voxel_coords[:, None, :] + basis[None, :, :]
    local_triangle_indices = _faces_triangles().to(device)[face_indices]
    voxel_selector = torch.arange(face_coordinates.shape[0], device=device)[:, None]
    triangle_vertex_coords_int = voxel_vertex_coords[
        voxel_selector, local_triangle_indices
    ]
    flat_vertex_coords_int = einops.rearrange(
        triangle_vertex_coords_int, "n (t v) d -> (n t v) d", t=2, v=3
    )
    unique_vertices_int, inverse_indices = torch.unique(
        flat_vertex_coords_int, dim=0, return_inverse=True
    )
    faces = einops.rearrange(inverse_indices, "(m v) -> m v", v=3)
    vertices = unique_vertices_int.float() * spacing
    return vertices, faces


def _edge_coordinates(
    vertex_sign: torch.BoolTensor, mode: Literal["dense", "efficient"]
):
    occupied_voxels = _get_vertices(vertex_sign).all(dim=-1)
    if mode == "efficient":
        coordinates = _face_mask(occupied_voxels)
        return coordinates

    if mode != "dense":
        raise ValueError(f"Invalid mode: {mode}")

    face_mask = occupied_voxels[..., None] & ~_get_adjacency(occupied_voxels)
    coordinates = torch.nonzero(face_mask, as_tuple=False)
    return coordinates


def march_voxels(
    vertex_sign: torch.BoolTensor, mode: Literal["dense", "efficient"] = "efficient"
):
    spacing = 2.0 / (vertex_sign.shape[0] - 1)
    coordinates = _edge_coordinates(vertex_sign, mode=mode)
    vertices, faces = _faces_vertices(coordinates, spacing)
    vertices -= 1.0
    return vertices, faces
