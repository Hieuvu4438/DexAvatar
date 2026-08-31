"""Stable differentiable SO(3) logarithm used at the SMPL-X boundary."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor


def _vee(matrix: Tensor) -> Tensor:
    return torch.stack((matrix[..., 2, 1], matrix[..., 0, 2], matrix[..., 1, 0]), dim=-1)


def log_map(rotation: Tensor, eps: float = 1e-7) -> Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must end in [3,3]")
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1) * 0.5).clamp(-1 + eps, 1 - eps)
    theta = torch.acos(cosine)
    sine = torch.sin(theta)
    raw = 0.5 * _vee(rotation - rotation.transpose(-1, -2))
    vector = raw * (theta / sine.clamp_min(eps))[..., None]
    small = theta < 1e-4
    vector = torch.where(small[..., None], raw * (1 + theta[..., None].square() / 6), vector)
    near_pi = theta > torch.pi - 1e-3
    if bool(near_pi.any()):
        diagonal = rotation.diagonal(dim1=-2, dim2=-1)
        axis_absolute = torch.sqrt(((diagonal + 1) * 0.5).clamp_min(0))
        signs = torch.stack(
            (
                torch.ones_like(axis_absolute[..., 0]),
                torch.sign(rotation[..., 0, 1] + rotation[..., 1, 0] + eps),
                torch.sign(rotation[..., 0, 2] + rotation[..., 2, 0] + eps),
            ),
            dim=-1,
        )
        axis = functional.normalize(axis_absolute * signs, dim=-1, eps=eps)
        vector = torch.where(near_pi[..., None], axis * theta[..., None], vector)
    return vector
