from __future__ import annotations

import torch


def robust_unit_bones(
    joints: torch.Tensor,
    edges: tuple[tuple[int, int], ...],
    valid: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parent = torch.as_tensor([edge[0] for edge in edges], device=joints.device)
    child = torch.as_tensor([edge[1] for edge in edges], device=joints.device)
    bone = joints[..., child, :] - joints[..., parent, :]
    length = torch.linalg.vector_norm(bone, dim=-1)
    usable = valid[..., child] & valid[..., parent] & (length > 1e-4)
    unit = bone / length.clamp_min(eps)[..., None]
    return unit, length, usable


def bone_direction_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    valid: torch.Tensor,
    delta: float = 0.05,
) -> torch.Tensor:
    residual = predicted - observed
    absolute = residual.abs()
    huber = torch.where(
        absolute <= delta,
        0.5 * residual.square() / delta,
        absolute - 0.5 * delta,
    ).sum(dim=-1)
    weight = valid.to(huber.dtype)
    return (huber * weight).sum() / weight.sum().clamp_min(1.0)

