"""
sparc3d_sdf: A library for efficient SDF generation and sparse mesh extraction.
"""

from sparc3d_sdf.sdf import compute_sdf_on_grid, vertex_grid
from sparc3d_sdf.isosurface import SparseCube
from sparc3d_sdf.sparse import convert_dense_to_sparse
from sparc3d_sdf.utils import load_obj, save_obj

__all__ = [
    "compute_sdf_on_grid",
    "vertex_grid",
    "convert_dense_to_sparse",
    "SparseCube",
    "load_obj",
    "save_obj",
]
