"""
Generic shape sign distance functions.
"""

import torch
from pytorch3d.transforms import so3_exp_map


def torus_sdf(p: torch.Tensor, R: float, r: float) -> torch.Tensor:
    """
    Torus SDF
    """

    sdf_ = (
        ((p[..., 0] ** 2 + p[..., 1] ** 2) ** (1 / 2) - R) ** 2 + p[..., 2] ** 2
    ) ** (1 / 2) - r

    return sdf_


def cube_sdf(p: torch.Tensor, L: float) -> torch.Tensor:
    """
    Cube SDF
    """
    sdf_ = p.abs().max(dim=-1).values - L
    return sdf_


def sphere_sdf(p: torch.Tensor, R: float) -> torch.Tensor:
    """
    Sphere SDF
    """
    sdf_ = p.norm(dim=-1) - R
    return sdf_


def rotate_point(p: torch.Tensor, axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Rotate a point by an axis angle.
    """

    if axis_angle.dim() == 1:
        axis_angle = axis_angle[None]
    rot = so3_exp_map(axis_angle)[0]
    assert rot.dim() == 2
    return p @ rot.T
