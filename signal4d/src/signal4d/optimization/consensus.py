from __future__ import annotations

import torch

from ..geometry.so3 import exp_map, log_map
from .window import Window, hann_weights


def weighted_karcher_mean(
    rotations: torch.Tensor, weights: torch.Tensor, tolerance: float = 1e-7, max_steps: int = 32
) -> torch.Tensor:
    if rotations.shape[-2:] != (3, 3) or rotations.shape[0] != weights.shape[0]:
        raise ValueError("rotations [K,...,3,3] and weights [K,...] are required")
    mean = rotations[torch.argmax(weights.reshape(weights.shape[0], -1).mean(-1))].clone()
    expanded = weights
    while expanded.ndim < rotations.ndim - 2:
        expanded = expanded.unsqueeze(-1)
    for _ in range(max_steps):
        delta = (
            expanded[..., None] * log_map(mean.unsqueeze(0).transpose(-1, -2) @ rotations)
        ).sum(0)
        delta = delta / expanded.sum(0).clamp_min(1e-12)[..., None]
        mean = mean @ exp_map(delta)
        if torch.linalg.vector_norm(delta, dim=-1).max() < tolerance:
            break
    return mean


def merge_trajectories(
    windows: list[Window],
    joints: list[torch.Tensor],
    uncertainties: list[torch.Tensor],
    total_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (len(windows) == len(joints) == len(uncertainties)):
        raise ValueError("window result lengths differ")
    joint_count = joints[0].shape[1]
    merged = joints[0].new_zeros((total_frames, joint_count, 3))
    denominator = joints[0].new_zeros((total_frames, joint_count, 1))
    for window, trajectory, uncertainty in zip(windows, joints, uncertainties, strict=True):
        taper = hann_weights(window.length, trajectory.device, trajectory.dtype)[:, None]
        weight = taper / uncertainty.clamp_min(1e-6)
        merged[window.start : window.end] += trajectory * weight[..., None]
        denominator[window.start : window.end] += weight[..., None]
    if (denominator == 0).any():
        raise RuntimeError("window merge left uncovered frames")
    merged = merged / denominator
    return merged, denominator.squeeze(-1).reciprocal()
