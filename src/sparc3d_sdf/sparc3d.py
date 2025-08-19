"""
Sparc3D utils
"""

import torch
from typing import Literal


def spatial_gradient(udf: torch.Tensor, delta: float):
    """
    compute spatial gradient of udf using finite differences 2nd order approx

    udf: (N, N, N)
    delta: float (grid spacing)

    returns: (N, N, N, 3)
    """

    grads_tuple = torch.gradient(udf.float(), dim=(0, 1, 2), spacing=delta)
    gradient = torch.stack(grads_tuple, dim=-1)

    return gradient


# TODO calculate displacements or gradients only within the active mask
def get_displacements(
    udf: torch.Tensor,
    eta: float,
    clip: bool,
    clip_mode: Literal["abs", "norm"],
) -> torch.Tensor:
    """
    displace vertices in -Grad(udf) direction

    udf: (N, N, N)
    eta: float (step size)
    clip: bool (whether to clip displacements)
    clip_mode: Literal["abs", "norm"] (how to clip displacements)
    clip displacements to be less than 1/2 the grid spacing

    abs mode: clip displacements to be less than 1/2 the grid spacing
    norm mode: clip displacements to have norm less than 1/2 the grid spacing

    returns: (N, N, N, 3)
    """

    delta = 2 / udf.shape[0]
    udf_gradient = spatial_gradient(udf, delta)

    vertex_displacements = -eta * udf_gradient

    if clip:
        if clip_mode == "norm":
            clip_mask = (vertex_displacements.abs() > delta / 2).any(dim=-1)
            vertex_displacements[clip_mask] *= (
                delta / 2 / vertex_displacements[clip_mask].norm(keepdim=True)
            )
        elif clip_mode == "abs":
            vertex_displacements = torch.clamp(
                vertex_displacements, min=-delta / 2, max=delta / 2
            )
    return vertex_displacements
