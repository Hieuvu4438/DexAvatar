from __future__ import annotations

import torch

from ...geometry.so3 import skew


def stable_exp_map(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """SO(3) exponential with finite gradients at an exactly zero tangent."""
    if vector.shape[-1] != 3:
        raise ValueError("stable_exp_map expects [...,3]")
    theta_squared = (vector * vector).sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta_squared.clamp_min(eps))
    small = theta_squared < 1e-8
    a = torch.where(
        small,
        1 - theta_squared / 6 + theta_squared.square() / 120,
        torch.sin(theta) / theta,
    )[..., None]
    b = torch.where(
        small,
        0.5 - theta_squared / 24 + theta_squared.square() / 720,
        (1 - torch.cos(theta)) / theta_squared.clamp_min(eps),
    )[..., None]
    tangent = skew(vector)
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device).expand(tangent.shape)
    return identity + a * tangent + b * (tangent @ tangent)

