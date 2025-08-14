"""
Implements a highly memory-efficient, "greedy" voxel meshing algorithm.
This is distinct from classic Marching Cubes as it doesn't interpolate an isosurface.
Instead, it identifies contiguous blocks of "solid" voxels and generates a mesh
representing their external faces, resulting in a voxelized or "blocky" appearance.
"""

import torch
import torch.nn.functional as F
import einops
from typing import Literal


def _get_adjacency(matrix: torch.BoolTensor) -> torch.BoolTensor:
    """
    For each voxel, finds the state of its 6 cardinal neighbors (+/- x, y, z).
    This is the 'dense' method for finding neighboring voxels.

    Args:
        matrix: A boolean tensor of shape (X, Y, Z) representing occupied voxels.

    Returns:
        A boolean tensor of shape (X, Y, Z, 6) where the last dimension indicates
        the occupancy of the 6 neighbors.
    """
    assert matrix.dim() == 3

    # --- Pad the grid ---
    # Pad the matrix by 1 voxel on all sides. This simplifies neighbor access
    # by ensuring that even voxels on the original border have 6 neighbors to query.
    padded_matrix = F.pad(matrix, (1, 1, 1, 1, 1, 1), mode="constant", value=False)

    # --- Extract Neighbor Grids ---
    # By slicing the padded grid with different offsets, we can create 6 grids,
    # each representing the state of a specific neighbor for all original voxels.
    # For example, `padded_matrix[:-2, 1:-1, 1:-1]` corresponds to the `x-1` neighbors.
    adjacency = torch.stack(
        [
            padded_matrix[:-2, 1:-1, 1:-1],  # x-1
            padded_matrix[2:, 1:-1, 1:-1],  # x+1
            padded_matrix[1:-1, :-2, 1:-1],  # y-1
            padded_matrix[1:-1, 2:, 1:-1],  # y+1
            padded_matrix[1:-1, 1:-1, :-2],  # z-1
            padded_matrix[1:-1, 1:-1, 2:],  # z+1
        ],
        dim=-1,
    )
    return adjacency


def _face_mask(occupied_voxels: torch.BoolTensor) -> torch.LongTensor:
    """
    Computes the coordinates of all external faces in a memory-efficient way.
    An external face is one where a voxel is occupied but its neighbor in a
    given direction is not. This avoids creating a large intermediate (X,Y,Z,6) tensor.

    Args:
        occupied_voxels: A boolean tensor of shape (X, Y, Z) of occupied voxels.

    Returns:
        A long tensor of shape (N, 4) where N is the number of external faces.
        Each row is `(x, y, z, direction_index)`.
    """
    # --- Define Slicing Patterns ---
    # These slices are used to compare a voxel with its neighbor along each axis.
    # `_identity` selects the original voxel grid from the padded grid.
    # `_backward` and `_forward` select the neighboring grids.
    _forward = slice(2, None)
    _backward = slice(0, -2)
    _identity = slice(1, -1)

    # Pre-defined slice combinations for the 6 directions.
    _slices = [
        [_backward, _identity, _identity],
        [_forward, _identity, _identity],
        [_identity, _backward, _identity],
        [_identity, _forward, _identity],
        [_identity, _identity, _backward],
        [_identity, _identity, _forward],
    ]

    padded_voxels = F.pad(
        occupied_voxels, (1, 1, 1, 1, 1, 1), mode="constant", value=False
    )

    # --- Iteratively Find Surface Faces ---
    # For each of the 6 directions, we find the coordinates of all faces on the boundary.
    result = []
    for k, s in enumerate(_slices):
        # A face exists if the current voxel is occupied AND its neighbor is not.
        is_surface_face = (
            padded_voxels[_identity, _identity, _identity]
            & ~padded_voxels[s[0], s[1], s[2]]
        )

        # Get the (x, y, z) coordinates of these surface faces.
        coordinates = torch.nonzero(is_surface_face)

        # Append the direction index `k` to each coordinate, creating (x, y, z, k).
        coordinates_with_direction = F.pad(
            coordinates, (0, 1), mode="constant", value=k
        )
        result.append(coordinates_with_direction)

    if not result:
        return torch.empty((0, 4), dtype=torch.long, device=occupied_voxels.device)

    return torch.cat(result, dim=0)


def _get_vertices(grid: torch.BoolTensor) -> torch.BoolTensor:
    """
    From a grid of vertex states, gathers the 8 corner vertices for each cell.
    A grid of size (S, S, S) has (S-1, S-1, S-1) cells.

    Args:
        grid: A boolean tensor of shape (X, Y, Z) representing vertex states.

    Returns:
        A boolean tensor of shape (X-1, Y-1, Z-1, 8) where the last dimension
        holds the states of the 8 vertices for each cell.
    """
    return torch.stack(
        [
            grid[:-1, :-1, :-1],  # v0 (x, y, z)
            grid[:-1, :-1, 1:],  # v1 (x, y, z+1)
            grid[:-1, 1:, :-1],  # v2 (x, y+1, z)
            grid[:-1, 1:, 1:],  # v3 (x, y+1, z+1)
            grid[1:, :-1, :-1],  # v4 (x+1, y, z)
            grid[1:, :-1, 1:],  # v5 (x+1, y, z+1)
            grid[1:, 1:, :-1],  # v6 (x+1, y+1, z)
            grid[1:, 1:, 1:],  # v7 (x+1, y+1, z+1)
        ],
        dim=-1,
    )


def _faces_triangles() -> torch.LongTensor:
    """
    Provides a constant lookup table for cube face triangulation.
    Each of the 6 faces of a cube is made of two triangles (6 vertices).

    The winding order is defined counter-clockwise (using the right-hand rule)
    such that the normals of the faces point outwards from the cube.

    Vertex Order (matches 'ij' indexing from meshgrid):
    v0:(0,0,0), v1:(0,0,1), v2:(0,1,0), v3:(0,1,1)
    v4:(1,0,0), v5:(1,0,1), v6:(1,1,0), v7:(1,1,1)


    Returns:
        A long tensor of shape (6, 6) with local vertex indices for triangles.
    """
    triangles = torch.tensor(
        [
            # -x face (v0, v1, v2, v3)
            [0, 1, 3, 0, 3, 2],
            # +x face (v4, v5, v6, v7)
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


def _faces_vertices(
    face_coordinates: torch.LongTensor, spacing: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The core meshing function. Converts face coordinates into a vertex and face list.
    It generates all necessary vertices, then deduplicates them for an efficient mesh.

    Args:
        face_coordinates: (N, 4) tensor of (x, y, z, direction_index).
        spacing: The distance between vertices in the final mesh.

    Returns:
        A tuple of (vertices, faces):
        - vertices: (V, 3) float tensor of unique vertex positions.
        - faces: (F, 3) long tensor of triangle indices.
    """
    if face_coordinates.shape[0] == 0:
        return torch.empty((0, 3), device=face_coordinates.device), torch.empty(
            (0, 3), dtype=torch.long, device=face_coordinates.device
        )

    device = face_coordinates.device
    voxel_coords = face_coordinates[:, :3]  # Voxel grid location of each face
    face_indices = face_coordinates[:, 3]  # Direction index (0-5) of each face

    # --- Generate Global Vertex Coordinates for each Face's Voxel ---
    # 1. Create a local basis for a cube's 8 vertices (0,0,0) to (1,1,1).
    _basis = torch.arange(2, device=device, dtype=torch.long)
    basis = torch.stack(torch.meshgrid(_basis, _basis, _basis, indexing="ij"), dim=-1)
    basis = einops.rearrange(basis, "x y z d -> (x y z) d")  # Shape: (8, 3)

    # 2. Use broadcasting to get the global integer coordinates of all 8 vertices
    # for every voxel that contains a surface face. Shape: (N, 8, 3).
    voxel_vertex_coords = voxel_coords[:, None, :] + basis[None, :, :]

    # --- Select Triangle Vertices and Deduplicate ---
    # 1. For each face, get the 6 local indices of the vertices that form its two triangles.
    local_triangle_indices = _faces_triangles().to(device)[
        face_indices
    ]  # Shape: (N, 6)

    # 2. Use advanced indexing to get the global coordinates for each vertex of each triangle.
    # This gathers the 6 required vertex coords from the 8 available for each voxel.
    voxel_selector = torch.arange(face_coordinates.shape[0], device=device)[:, None]
    triangle_vertex_coords_int = voxel_vertex_coords[
        voxel_selector, local_triangle_indices
    ]  # Shape: (N, 6, 3)

    # 3. Flatten the list of triangles to a simple list of vertices (3 per triangle).
    flat_vertex_coords_int = einops.rearrange(
        triangle_vertex_coords_int, "n (t v) d -> (n t v) d", t=2, v=3
    )

    # 4. The key step: find unique vertices and get the inverse indices to remap the faces.
    # This is the standard, efficient way to create an indexed mesh.
    unique_vertices_int, inverse_indices = torch.unique(
        flat_vertex_coords_int, dim=0, return_inverse=True
    )

    # --- Finalize Mesh ---
    # Reshape the inverse indices back into a face list (F, 3).
    faces = einops.rearrange(inverse_indices, "(m v) -> m v", v=3)
    # Scale integer vertex coordinates to their final float positions.
    vertices = unique_vertices_int.float() * spacing

    return vertices, faces


def _edge_coordinates(
    vertex_sign: torch.BoolTensor, mode: Literal["dense", "efficient"]
) -> torch.LongTensor:
    """
    Dispatcher function that finds the coordinates of all external faces.
    It first defines "occupied" voxels and then uses either a dense or
    memory-efficient method to find the boundaries.
    """
    # --- Define Occupied Voxels ---
    # A voxel is considered "occupied" or "solid" if ALL 8 of its corner
    # vertices are inside the surface. This is the core of the voxelization logic.
    occupied_voxels = _get_vertices(vertex_sign).all(dim=-1)

    # --- Find Surface Faces ---
    if mode == "efficient":
        # This mode avoids creating the large intermediate adjacency tensor.
        return _face_mask(occupied_voxels)

    if mode == "dense":
        # This mode is more straightforward but uses more memory. It computes the
        # full (X,Y,Z,6) adjacency mask and then finds where a voxel is
        # occupied but a neighbor is not.
        face_mask = occupied_voxels[..., None] & ~_get_adjacency(occupied_voxels)
        return torch.nonzero(face_mask, as_tuple=False)

    raise ValueError(f"Invalid mode: {mode}")


def march_voxels(
    vertex_sign: torch.BoolTensor, mode: Literal["dense", "efficient"] = "efficient"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Main entrypoint for the voxel meshing algorithm.

    Args:
        vertex_sign: A boolean grid of shape (S, S, S) where `True` indicates a
                     vertex is inside the desired volume.
        mode: 'efficient' uses less memory, 'dense' is more straightforward.

    Returns:
        A tuple of (vertices, faces) for the generated mesh, normalized to [-1, 1].
    """
    # --- Calculate Spacing and Find Surface Faces ---
    # The spacing is calculated to normalize the final mesh into a [-1, 1] cube.
    # A grid of size S has S-1 cells, so the total length is (S-1) * spacing.
    if vertex_sign.shape[0] <= 1:
        return torch.empty((0, 3)), torch.empty((0, 3), dtype=torch.long)
    spacing = 2.0 / (vertex_sign.shape[0] - 1)

    coordinates = _edge_coordinates(vertex_sign, mode=mode)

    # --- Generate and Center the Mesh ---
    vertices, faces = _faces_vertices(coordinates, spacing)
    # The vertices are currently in a [0, 2] range, so we shift to center at the origin.
    vertices -= 1.0

    return vertices, faces
