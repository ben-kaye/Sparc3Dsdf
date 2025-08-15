import kaolin.utils.testing as testing
import torch


from kaolin.ops.conversions.flexicubes.flexicubes import (
    check_table,
    dmc_table,
    num_vd_table,
    tet_table,
)
from kaolin.ops.conversions.flexicubes.flexicubes import Flexicubes


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


class SparseCube(Flexicubes):
    """
    Memory efficient version of Flexicubes.
    Avoid instantiating a full voxel grid
    """

    def _identify_surf_cubes(self, scalar_field, cube_idx):
        pass

    def __call__(
        self,
        voxelgrid_vertices,
        scalar_field,
        cube_idx,
        resolution,
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
        surf_cubes, occ_fx8 = self._identify_surf_cubes(scalar_field, cube_idx)
