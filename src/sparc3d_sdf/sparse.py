"""
Extract sparse representation given an active vertex mask in a dense DHW grid.
"""

import torch
import math
import einops

from sparc3d_sdf.grid_primitives import VERTEX_OFFSETS, ADJACENCY_OFFSETS


def _vertex_to_cube(**kwargs) -> torch.LongTensor:
    """
    index offsets for a cube
    (8, 3)
    """
    return torch.tensor(VERTEX_OFFSETS, dtype=torch.long, **kwargs)


def _adjacency(**kwargs) -> torch.LongTensor:
    """
    index offsets for adjacent cubes
    (6, 3)
    """
    return torch.tensor(ADJACENCY_OFFSETS, dtype=torch.long, **kwargs)


def _cube_index_to_linear(index: torch.LongTensor, DHW: tuple[int, int, int]):
    """
    Converts (..., 3) coordinates to linear indices for a grid of shape `DHW`.
    """
    d, h, w = DHW

    index = index.long()
    return index[..., 0] * (h * w) + index[..., 1] * w + index[..., 2]


def extract_sparse_field(active_mask: torch.BoolTensor):
    """
    Extracts active cubes, their vertex indices, and their adjacency in the DENSE grid.

    Args:
        sdf_mask: A boolean tensor for the VERTEX grid, shape (L+1, L+1, L+1).

    Returns:
        vertex_idx (N, 8): Dense linear indices for the 8 vertices of each active cube.
        adj_cubes (N, 6): Dense linear indices for the 6 neighbors of each active cube.
        cube_coords (N, 3): The (x,y,z) integer coordinates of the N active cubes.
        cube_grid_shape (tuple): The shape of the grid of cubes, e.g., (L, L, L).
    """
    device = active_mask.device
    vertex_grid_shape = active_mask.shape
    cube_grid_shape = tuple(s - 1 for s in vertex_grid_shape)
    cube_grid_shape_t = torch.tensor(cube_grid_shape, device=device, dtype=torch.long)

    # Find all active vertices
    active_vertex_coords = torch.nonzero(active_mask)

    # From active vertices, determine the set of active cubes
    potential_cubes = einops.rearrange(
        active_vertex_coords[:, None, :] - _vertex_to_cube().to(device)[None, :, :],
        "n1 n2 d -> (n1 n2) d",
    )

    # A cube is valid if its origin (x,y,z) is within the cube grid bounds
    valid_mask = (potential_cubes >= 0).all(dim=-1) & (
        potential_cubes < cube_grid_shape_t
    ).all(dim=-1)
    cube_coords = torch.unique(potential_cubes[valid_mask], dim=0)

    # --- Calculate vertex indices for each cube ---
    # (N, 1, 3) + (1, 8, 3) -> (N, 8, 3)
    vertex_coords = cube_coords[:, None, :] + _vertex_to_cube(device=device)[None, :, :]
    vertex_idx = _cube_index_to_linear(vertex_coords, vertex_grid_shape)

    # --- Calculate adjacency for each cube ---
    adj_coords = cube_coords[:, None, :] + _adjacency(device=device)[None, :, :]

    # Mask out neighbors that are outside the grid boundaries
    outside_mask = ((adj_coords < 0) | (adj_coords >= cube_grid_shape_t)).any(dim=-1)

    # Convert to dense linear indices.
    adj_coords_clamped = adj_coords.clone()
    adj_coords_clamped[outside_mask] = 0  # Clamp to a safe index to avoid errors

    adj_cubes = _cube_index_to_linear(adj_coords_clamped, cube_grid_shape)
    adj_cubes[outside_mask] = -1  # Re-apply the -1 for out-of-bounds neighbors

    return vertex_idx, adj_cubes, cube_coords, cube_grid_shape


def remap_sparse_field(
    cube_idx: torch.LongTensor,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Remaps a dense index tensor (like vertex indices) to a sparse representation.

    Args:
        cube_idx: A tensor of dense indices, e.g., (N, 8).

    Returns:
        unique_indices: The sorted list of unique dense indices that appeared.
        reverse_map: A tensor of the same shape as cube_idx, but with sparse indices.
    """
    assert cube_idx.dim() == 2, "cube_idx must be a 2D tensor"
    assert cube_idx.shape[1] == 8, "cube_idx must have 8 columns"

    unique_indices, reverse_map = torch.unique(cube_idx, return_inverse=True)
    return unique_indices, reverse_map.view_as(cube_idx)


def remap_adjacency(
    cube_coords: torch.Tensor,
    adj_dense_indices: torch.Tensor,
    cube_grid_shape: tuple[int, int, int],
) -> torch.Tensor:
    """
    Remaps dense linear adjacency indices to sparse indices using a memory-efficient
    sort-and-search method, avoiding large lookup tables.

    Args:
        cube_coords: The (x,y,z) coordinates of the N active cubes. Shape: (N, 3).
        adj_dense_indices: Dense linear indices of adjacent cubes. Shape: (N, 6).
        cube_grid_shape: The dimensions of the dense grid of cubes (L, L, L).

    Returns:
        A tensor of shape (N, 6) where values are sparse indices (0..N-1) or -1.
    """
    device = cube_coords.device
    num_active_cubes = cube_coords.shape[0]

    # 1. Get the dense linear indices of our *active* cubes.
    active_dense_indices = _cube_index_to_linear(cube_coords, cube_grid_shape)

    # 2. Create the "map" from dense indices to new sparse indices (0..N-1)
    # by sorting the dense indices.
    sorted_dense_indices, sort_permutation = torch.sort(active_dense_indices)

    # The values of our map are the new sparse indices (0..N-1), permuted to
    # align with the sorted dense indices.
    map_values = torch.empty_like(sort_permutation)
    map_values[sort_permutation] = torch.arange(num_active_cubes, device=device)

    # 3. Flatten the adjacency tensor and use binary search (searchsorted) to find
    # the locations of our neighbors in the sorted list of active dense indices.
    flat_adj_dense = adj_dense_indices.flatten()
    locations = torch.searchsorted(sorted_dense_indices, flat_adj_dense)

    # 4. Verify the matches and perform the remapping.
    remapped_flat = torch.full_like(flat_adj_dense, -1)

    # Clamp locations to prevent index out of bounds on the next line.
    locations.clamp_(max=num_active_cubes - 1)

    # Check which of the found locations correspond to an actual match.
    retrieved_dense_indices = sorted_dense_indices[locations]
    where_matches = retrieved_dense_indices == flat_adj_dense

    # For the matches, get the corresponding sparse index from our map.
    matched_indices = locations[where_matches]
    remapped_flat[where_matches] = map_values[matched_indices]

    # 5. Reshape back to the original (N, 6) adjacency shape.
    return remapped_flat.view_as(adj_dense_indices)


def convert_dense_to_sparse(
    active_mask: torch.BoolTensor, cube_resolution: tuple[int, int, int]
):
    """
    Example orchestration function to convert a dense field to a sparse representation.

    Args:
        active_mask (M,): Dense boolean mask of active vertices
        resolution (D,H,W): The resolution of the CUBE grid.
    """

    # Reshape the mask into a 3D grid for extraction
    vertex_grid_shape = tuple(r + 1 for r in cube_resolution)

    assert active_mask.dim() == 1, "active_mask must be a 1D tensor"
    assert active_mask.shape[0] == math.prod(vertex_grid_shape), (
        "must be same length as the number of vertices in the cube grid"
    )

    sdf_mask_grid = active_mask.view(vertex_grid_shape)

    # 1. Extract sparse cubes and their DENSE indices
    cube_vtx_idx_dense, adj_idx_dense, cube_coords, cube_grid_shape = (
        extract_sparse_field(sdf_mask_grid)
    )

    # 2. Remap the cube vertex indices to a sparse set of vertices
    sparse_vtx_indices, cube_vtx_idx_sparse = remap_sparse_field(cube_vtx_idx_dense)

    # 3. Remap the cube adjacency indices to sparse cube indices
    adj_idx_sparse = remap_adjacency(cube_coords, adj_idx_dense, cube_grid_shape)

    return sparse_vtx_indices, cube_vtx_idx_sparse, adj_idx_sparse
