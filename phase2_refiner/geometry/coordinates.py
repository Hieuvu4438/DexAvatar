"""Explicit homogeneous-coordinate transforms used by Phase 2 caches."""

from __future__ import annotations

import torch


def validate_transform(transform: torch.Tensor, atol: float = 1e-4) -> None:
    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"Expected (...,4,4), got {tuple(transform.shape)}")
    if not torch.isfinite(transform).all():
        raise ValueError("Transform contains NaN or Inf")
    bottom = transform[..., 3, :]
    expected = torch.zeros_like(bottom)
    expected[..., 3] = 1.0
    if not torch.allclose(bottom, expected, atol=atol):
        raise ValueError("Invalid homogeneous-transform bottom row")
    rotation = transform[..., :3, :3]
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    if not torch.allclose(
        rotation @ rotation.transpose(-1, -2), identity.expand_as(rotation), atol=atol
    ):
        raise ValueError("Transform rotation is not orthonormal")


def invert_transform(transform: torch.Tensor) -> torch.Tensor:
    validate_transform(transform)
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    inverse = torch.zeros_like(transform)
    inverse[..., :3, :3] = rotation.transpose(-1, -2)
    inverse[..., :3, 3] = -(
        rotation.transpose(-1, -2) @ translation[..., None]
    ).squeeze(-1)
    inverse[..., 3, 3] = 1.0
    return inverse


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    if points.shape[-1] != 3:
        raise ValueError(f"Expected (...,N,3), got {tuple(points.shape)}")
    validate_transform(transform)
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return points @ rotation.transpose(-1, -2) + translation[..., None, :]


def compose_transforms(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compose transforms so the returned transform applies second then first."""
    validate_transform(first)
    validate_transform(second)
    result = first @ second
    validate_transform(result)
    return result
