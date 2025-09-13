import kaolin.utils.testing as testing
import torch
from kaolin.ops.conversions.flexicubes import FlexiCubes
from typing import Callable


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


# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

        # This is a rewrite of the Flexicubes _get_case_id function
        # but uses a sparse adjacency index instead of a dense adjacency index
        # memory efficient but computationally slower
        # also not as clean an implementation..

        all_case_ids = (
            occ_fx8 * self.cube_corners_idx.to(self.device).unsqueeze(0)
        ).sum(-1)
        all_problem_config = self.check_table.to(self.device)[all_case_ids]

        is_problematic = all_problem_config[..., 0] == 1
        to_check_mask = surf_cubes & is_problematic

        if not to_check_mask.any():
            return all_case_ids[surf_cubes]

        indices_to_check = torch.nonzero(to_check_mask).squeeze(1)

        problem_config_to_check = all_problem_config[indices_to_check]
        adjacency_to_check = adj_idx[indices_to_check]

        direction_offsets = problem_config_to_check[..., 1:4]

        lookup_indices = _offset_to_adjacency_index(direction_offsets)

        neighbor_sparse_indices = torch.gather(
            adjacency_to_check, 1, lookup_indices.unsqueeze(1)
        ).squeeze(1)

        in_bounds_mask = neighbor_sparse_indices != -1
        if not in_bounds_mask.any():
            return all_case_ids[surf_cubes]  # No valid neighbors to check

        cubes_to_invert_indices = indices_to_check[in_bounds_mask]
        neighbor_indices_to_check = neighbor_sparse_indices[in_bounds_mask]
        config_of_cubes_to_invert = problem_config_to_check[in_bounds_mask]

        neighbor_is_problematic = all_problem_config[neighbor_indices_to_check, 0] == 1

        if not neighbor_is_problematic.any():
            return all_case_ids[surf_cubes]

        final_indices_to_invert = cubes_to_invert_indices[neighbor_is_problematic]
        final_config = config_of_cubes_to_invert[neighbor_is_problematic]

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
