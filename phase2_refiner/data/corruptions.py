"""Synthetic burst failures for controlled Phase 2 training."""

from __future__ import annotations

import math

import torch

from phase2_refiner.geometry.rotations import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)


def _random_rotation_vectors(
    shape: tuple[int, ...], max_degrees: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    direction = torch.randn(shape, device=device, dtype=dtype)
    direction = direction / torch.linalg.vector_norm(
        direction, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    magnitude = torch.rand(shape[:-1] + (1,), device=device, dtype=dtype)
    return direction * magnitude * math.radians(max_degrees)


def apply_burst_corruption(
    features: torch.Tensor,
    initial_matrix: torch.Tensor,
    frame_valid: torch.Tensor,
    probability: float = 0.65,
    min_duration: int = 2,
    max_duration: int = 16,
    max_rotation_degrees: float = 35.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask one part for a contiguous interval and perturb its initializer rotations."""
    features = features.clone()
    initial_matrix = initial_matrix.clone()
    corruption_mask = torch.zeros(
        features.shape[:3], dtype=torch.bool, device=features.device
    )
    batch = features.shape[0]
    groups = ((12, 21), (21, 36), (36, 51))
    for batch_idx in range(batch):
        if torch.rand((), device=features.device) > probability:
            continue
        valid_length = int(frame_valid[batch_idx].sum().item())
        if valid_length < min_duration:
            continue
        duration_limit = min(max_duration, valid_length)
        duration = int(
            torch.randint(min_duration, duration_limit + 1, (), device=features.device)
        )
        start = int(
            torch.randint(0, valid_length - duration + 1, (), device=features.device)
        )
        group_idx = int(torch.randint(0, len(groups), (), device=features.device))
        joint_start, joint_end = groups[group_idx]
        corruption_mask[batch_idx, start : start + duration, joint_start:joint_end] = (
            True
        )
        current = initial_matrix[
            batch_idx, start : start + duration, joint_start:joint_end
        ]
        residual = _random_rotation_vectors(
            current.shape[:-2] + (3,),
            max_rotation_degrees,
            current.device,
            current.dtype,
        )
        initial_matrix[batch_idx, start : start + duration, joint_start:joint_end] = (
            axis_angle_to_matrix(residual) @ current
        )
        features[batch_idx, start : start + duration, joint_start:joint_end, 18] = 0.0
        features[batch_idx, start : start + duration, joint_start:joint_end, 19] = 0.0
        features[batch_idx, start : start + duration, joint_start:joint_end, 20] = 1.0
        features[batch_idx, start : start + duration, joint_start:joint_end, 26:28] = (
            0.0
        )

    rot6d = matrix_to_rotation_6d(initial_matrix)
    features[..., :6] = rot6d
    features[..., 6:12] = 0.0
    features[..., 12:18] = 0.0
    features[:, 1:, :, 6:12] = rot6d[:, 1:] - rot6d[:, :-1]
    features[:, 2:, :, 12:18] = features[:, 2:, :, 6:12] - features[:, 1:-1, :, 6:12]
    return features, initial_matrix, corruption_mask
