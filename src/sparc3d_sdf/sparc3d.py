"""
Sparc3D utils
"""

from typing import Literal

import torch


def calculate_displacements(
    udf: torch.Tensor,
    eta: float,
    clip: Literal["abs", "norm", "tanh", None],
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> torch.Tensor:
    """
    displace vertices in -Grad(udf) direction

    udf: (N, N, N)
    eta: float (step size)

    clip: Literal["abs", "norm"] (how to clip displacements)
    clip displacements to be less than 1/2 the grid spacing

    abs mode: clip displacements to be less than 1/2 the grid spacing
    norm mode: clip displacements to have norm less than 1/2 the grid spacing

    returns: (N, N, N, 3)
    """

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    delta = 1 / (udf.shape[0] - 1)

    # there was an attempt here to minimise gradients as this has a cursed memory cost
    udf = torch.stack(
        torch.gradient(udf.to(device), dim=(2, 1, 0), spacing=delta), dim=-1
    )

    vertex_displacements = -eta * udf

    if clip:
        if clip == "norm":
            clip_mask = (vertex_displacements.abs() > delta / 2).any(dim=-1)
            vertex_displacements[clip_mask] *= (
                delta / 2 / vertex_displacements[clip_mask].norm(keepdim=True)
            )
        elif clip == "abs":
            vertex_displacements = torch.clamp(
                vertex_displacements, min=-delta / 2, max=delta / 2
            )
        elif clip == "tanh":
            vertex_displacements = torch.tanh(vertex_displacements) * delta / 2
    return vertex_displacements
