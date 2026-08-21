from __future__ import annotations

import torch


def project_points(points: torch.Tensor, camera_k: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if points.shape[-1] != 3 or camera_k.shape[-2:] != (3, 3):
        raise ValueError("points must be [...,N,3] and camera_k [...,3,3]")
    homogeneous = points @ camera_k.transpose(-1, -2)
    depth = homogeneous[..., 2:3]
    safe_depth = torch.where(depth.abs() < eps, torch.full_like(depth, eps), depth)
    return homogeneous[..., :2] / safe_depth


def normalized_reprojection_residual(
    projected: torch.Tensor, observed: torch.Tensor, image_size_hw: torch.Tensor
) -> torch.Tensor:
    diagonal = torch.linalg.vector_norm(image_size_hw.to(projected), dim=-1).clamp_min(1)
    while diagonal.ndim < projected.ndim:
        diagonal = diagonal.unsqueeze(-1)
    return (projected - observed) / diagonal
