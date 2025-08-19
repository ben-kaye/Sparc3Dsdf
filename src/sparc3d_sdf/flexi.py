import math

import einops
import kaolin.utils.testing as testing
import torch
from kaolin.ops.conversions.flexicubes import FlexiCubes
from typing import Callable
from sparc3d_sdf.grid_primitives import ADJACENCY_OFFSETS, VERTEX_OFFSETS


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


def _offset_to_adjacency_index(offsets: torch.LongTensor) -> torch.LongTensor:
    """
    Calculates the adjacency index (0-5) for a batch of 3D offset vectors.
    This is the vectorized inverse of the `_adjacency()` function.

    Args:
        offsets (torch.Tensor): A tensor of shape (N, 3) containing valid
                                adjacency offset vectors (one non-zero element which is +/-1).

    Returns:
        torch.Tensor: A tensor of shape (N,) with the corresponding indices (0-5).
    """
    # Find the axis of the non-zero element (0 for x, 1 for y, 2 for z)
    axis = torch.argmax(offsets.abs(), dim=1)

    # Get the sign of the offset (-1 or +1) from the corresponding axis
    sign = torch.gather(offsets, 1, axis.unsqueeze(1)).squeeze(1)

    return 2 * axis + (sign + 1) // 2


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
    assert active_mask.shape[0] == math.prod(cube_resolution), (
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


class SparseCube(FlexiCubes):
    """
    Memory efficient version of Flexicubes.
    Avoid instantiating a full voxel grid
    """

    @torch.no_grad()
    def _get_case_id(
        self,
        occ_fx8: torch.BoolTensor,
        surf_cubes: torch.BoolTensor,
        adj_idx: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Obtains the ID of topology cases based on cell corner occupancy.

        args:
            occ_fx8: (N, 8)
            surf_cubes: (N,)
            adj_idx: (N, 6) (-1 for out-of-bounds)

        returns:
            case_ids: (P,)
        """
        all_case_ids = (
            occ_fx8 * self.cube_corners_idx.to(self.device).unsqueeze(0)
        ).sum(-1)
        all_problem_config = self.check_table.to(self.device)[all_case_ids]

        # 2. Identify which cubes need checking: they must be on the surface AND have a problematic configuration.
        is_problematic = all_problem_config[..., 0] == 1
        to_check_mask = surf_cubes & is_problematic

        # If no problematic surface cubes exist, we are done. Return the case IDs for surface cubes.
        if not to_check_mask.any():
            return all_case_ids[surf_cubes]

        indices_to_check = torch.nonzero(to_check_mask).squeeze(1)

        # 3. For these problematic cubes, get their specific configuration and adjacency info.
        problem_config_to_check = all_problem_config[indices_to_check]
        adjacency_to_check = adj_idx[indices_to_check]

        # 4. Determine the direction of the ambiguous face for each problematic cube.
        # This is a 3D offset vector stored in the problem config table.
        direction_offsets = problem_config_to_check[..., 1:4]

        # 5. Convert this 3D offset vector into an adjacency index (0-5) using our helper.
        # This tells us which of the 6 neighbors we need to inspect.
        lookup_indices = _offset_to_adjacency_index(direction_offsets)

        # 6. Using the adjacency indices, get the sparse index of the neighbor to check for each cube.
        neighbor_sparse_indices = torch.gather(
            adjacency_to_check, 1, lookup_indices.unsqueeze(1)
        ).squeeze(1)

        # 7. Filter out cubes whose critical neighbor is out-of-bounds (indicated by -1).
        in_bounds_mask = neighbor_sparse_indices != -1
        if not in_bounds_mask.any():
            return all_case_ids[surf_cubes]  # No valid neighbors to check

        cubes_to_invert_indices = indices_to_check[in_bounds_mask]
        neighbor_indices_to_check = neighbor_sparse_indices[in_bounds_mask]
        config_of_cubes_to_invert = problem_config_to_check[in_bounds_mask]

        # 8. Check if the neighbors are ALSO problematic by looking up their config.
        neighbor_is_problematic = all_problem_config[neighbor_indices_to_check, 0] == 1

        if not neighbor_is_problematic.any():
            return all_case_ids[surf_cubes]

        # 9. Final filtering: select only the cubes whose neighbors were also problematic.
        final_indices_to_invert = cubes_to_invert_indices[neighbor_is_problematic]
        final_config = config_of_cubes_to_invert[neighbor_is_problematic]

        # 10. Update the case IDs for these cubes using the "inverted case ID" from the table.
        inverted_case_ids = final_config[..., -1]
        all_case_ids[final_indices_to_invert] = inverted_case_ids

        return all_case_ids[surf_cubes]

    def __call__(
        self,
        voxelgrid_vertices: torch.FloatTensor,
        scalar_field: torch.FloatTensor,
        cube_idx: torch.LongTensor,
        adj_idx: torch.LongTensor,
        qef_reg_scale=1e-3,
        weight_scale=0.99,
        beta=None,
        alpha=None,
        gamma_f=None,
        training: bool = False,
        output_tetmesh: bool = False,
        grad_func: Callable = None,
        voxelgrid_features: torch.Tensor = None,
    ):
        r"""
        Extract Isosurface from a sparse SDF field.

        voxelgrid_vertices: (N, 3)
        scalar_field: (N,)
        cube_idx: (N, 8)
        adj_idx: (N, 6)

        indices must correspond to the voxelgrid_vertices and scalar_field
        """
        assert torch.is_tensor(voxelgrid_vertices) and testing.check_tensor(
            voxelgrid_vertices, (None, 3), throw=False
        ), "'voxelgrid_vertices' should be a tensor of shape (num_vertices, 3)"
        num_vertices = voxelgrid_vertices.shape[0]
        assert torch.is_tensor(scalar_field) and testing.check_tensor(
            scalar_field, (num_vertices,), throw=False
        ), "'scalar_field' should be a tensor of shape (num_vertices,)"
        assert torch.is_tensor(cube_idx) and testing.check_tensor(
            cube_idx, (None, 8), throw=False
        ), "'cube_idx' should be a tensor of shape (num_cubes, 8)"
        num_cubes = cube_idx.shape[0]
        assert beta is None or (
            torch.is_tensor(beta)
            and testing.check_tensor(beta, (num_cubes, 12), throw=False)
        ), "'beta' should be a tensor of shape (num_cubes, 12)"
        assert alpha is None or (
            torch.is_tensor(alpha)
            and testing.check_tensor(alpha, (num_cubes, 8), throw=False)
        ), "'alpha' should be a tensor of shape (num_cubes, 8)"
        assert gamma_f is None or (
            torch.is_tensor(gamma_f)
            and testing.check_tensor(gamma_f, (num_cubes,), throw=False)
        ), "'gamma_f' should be a tensor of shape (num_cubes,)"
        assert voxelgrid_features is None or (
            torch.is_tensor(voxelgrid_features)
            and testing.check_tensor(
                voxelgrid_features, (num_vertices, None), throw=False
            )
        ), "'voxelgrid_features' should be a tensor of shape (num_cubes, num_features)"
        assert voxelgrid_features is None or not (
            output_tetmesh or grad_func is not None
        ), "'voxelgrid_features' is not supported with 'output_tetmesh' or 'grad_func'"

        surf_cubes, occ_fx8 = self._identify_surf_cubes(scalar_field, cube_idx)
        if surf_cubes.sum() == 0:
            if voxelgrid_features is None:
                return (
                    torch.zeros((0, 3), device=self.device),
                    torch.zeros((0, 4), dtype=torch.long, device=self.device)
                    if output_tetmesh
                    else torch.zeros((0, 3), dtype=torch.long, device=self.device),
                    torch.zeros((0), device=self.device),
                )
            else:
                return (
                    torch.zeros((0, 3), device=self.device),
                    torch.zeros((0, 3), dtype=torch.long, device=self.device),
                    torch.zeros((0), device=self.device),
                    torch.zeros((0, voxelgrid_features.shape[-1]), device=self.device),
                )
        beta, alpha, gamma_f = self._normalize_weights(
            beta, alpha, gamma_f, surf_cubes, weight_scale
        )

        case_ids = self._get_case_id(occ_fx8, surf_cubes, adj_idx)

        surf_edges, idx_map, edge_counts, surf_edges_mask = self._identify_surf_edges(
            scalar_field, cube_idx, surf_cubes
        )

        vd, L_dev, vd_gamma, vd_idx_map, vd_features = self._compute_vd(
            voxelgrid_vertices,
            cube_idx[surf_cubes],
            surf_edges,
            scalar_field,
            case_ids,
            beta,
            alpha,
            gamma_f,
            idx_map,
            grad_func,
            qef_reg_scale,
            voxelgrid_features,
        )
        vertices, faces, s_edges, edge_indices, vertices_features = self._triangulate(
            scalar_field,
            surf_edges,
            vd,
            vd_gamma,
            edge_counts,
            idx_map,
            vd_idx_map,
            surf_edges_mask,
            training,
            grad_func,
            vd_features,
        )
        if output_tetmesh:
            vertices, tets = self._tetrahedralize(
                voxelgrid_vertices,
                scalar_field,
                cube_idx,
                vertices,
                faces,
                surf_edges,
                s_edges,
                vd_idx_map,
                case_ids,
                edge_indices,
                surf_cubes,
                training,
            )
            return vertices, tets, L_dev
        elif voxelgrid_features is None:
            return vertices, faces, L_dev
        else:
            return vertices, faces, L_dev, vertices_features
