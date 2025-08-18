import kaolin.utils.testing as testing
import torch
import math

from kaolin.ops.conversions.flexicubes.flexicubes import (
    check_table,
    dmc_table,
    num_vd_table,
    tet_table,
)
from kaolin.ops.conversions.flexicubes.flexicubes import Flexicubes
import einops

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


def _vertex_to_cube(*, dim=-1):
    _prim = torch.arange(2, dtype=torch.long)
    offsets = torch.stack(torch.meshgrid(_prim, _prim, _prim, indexing="ij"), dim=dim)
    return offsets


def _cube_index_to_linear(idx: torch.LongTensor, res: tuple[int, int, int]):
    # FIXME validate
    d, h, w = res
    return idx[..., 0] * (d * h) + idx[..., 1] * h + idx[..., 2]


def extract_sparse_field(scalar_field: torch.FloatTensor):
    """
    returns:
        vertex_index (N, 8)
    """

    # threshold SDF to diagonal max distance one cell can permit
    # for all coords construct a list of possible cube indices that generated this vertex set
    # then torch.unique them
    # coords : N, 3
    # finally construct a the sdf_n8 which is the N cubes with the signs
    _res = 2 * math.sqrt(3)
    sparse_mask = scalar_field.abs() < _res
    coords = torch.nonzero(sparse_mask)

    cube_indices = einops.rearrange(
        coords[:, None] - _vertex_to_cube(dim=0)[None], "n1 n2 d -> (n1 n2) d"
    )
    cube_indices = torch.unique(cube_indices)

    # (N, 1, 3) + (1, 8, 3) -> (N 8 3)
    vertex_idx = cube_indices[:, None, :] + _vertex_to_cube(dim=1)[None, :, :]
    vertex_idx = _cube_index_to_linear(vertex_idx, scalar_field.shape[:2])

    return vertex_idx


def remap_sparse_field(dense_field, cube_idx):
    """
    remap the cube_idx to point only to the N,3 sparse indices
    """
    unique_indices, reverse_map = torch.unique(cube_idx[..., 0], return_inverse=True)
    sparse_field = dense_field[unique_indices]
    cube_idx = cube_idx[reverse_map]

    return sparse_field, cube_idx

class SparseCube(Flexicubes):
    """
    Memory efficient version of Flexicubes.
    Avoid instantiating a full voxel grid
    """

    dilate_factor: float = 2.0

    @torch.no_grad()
    def _identify_surf_edges(self, scalar_field, cube_idx, surf_cubes):
        """
        Identifies grid edges that intersect with the underlying surface by checking for opposite signs. As each edge
        can be shared by multiple cubes, this function also assigns a unique index to each surface-intersecting edge
        and marks the cube edges with this index.
        """
        occ_n = scalar_field < 0
        all_edges = cube_idx[surf_cubes][:, self.cube_edges].reshape(-1, 2)
        unique_edges, _idx_map, counts = torch.unique(
            all_edges, dim=0, return_inverse=True, return_counts=True
        )

        unique_edges = unique_edges.long()
        mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1

        surf_edges_mask = mask_edges[_idx_map]
        counts = counts[_idx_map]

        mapping = (
            torch.ones(
                (unique_edges.shape[0]), dtype=torch.long, device=cube_idx.device
            )
            * -1
        )
        mapping[mask_edges] = torch.arange(mask_edges.sum(), device=cube_idx.device)
        # Shaped as [number of cubes x 12 edges per cube]. This is later used to map a cube edge to the unique index
        # for a surface-intersecting edge. Non-surface-intersecting edges are marked with -1.
        idx_map = mapping[_idx_map]
        surf_edges = unique_edges[mask_edges]
        return surf_edges, idx_map, counts, surf_edges_mask

    def __call__(
        self,
        scalar_field: torch.FloatTensor,
        qef_reg_scale=1e-3,
        weight_scale=0.99,
        beta=None,
        alpha=None,
        gamma_f=None,
        training=False,
        output_tetmesh=False,
        grad_func=None,
        voxelgrid_features=None,
    ):
        """
        scalar_field: (L+1, L+1, L+1)

        """
