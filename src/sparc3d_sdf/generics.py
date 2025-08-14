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


def rotate_point(p: torch.Tensor, axis_angle: torch.Tensor) -> torch.Tensor:
    if axis_angle.dim() == 1:
        axis_angle = axis_angle[None]
    rot = so3_exp_map(axis_angle)[0]
    assert rot.dim() == 2
    return p @ rot.T
