from __future__ import annotations

import torch


def _validate_points(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape or prediction.shape[-1] != 3:
        raise ValueError(
            f"point shapes must match [...,N,3], got {prediction.shape}, {target.shape}"
        )
    if prediction.shape[-2] < 3:
        raise ValueError("at least three points are required")


def translation_align(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_points(prediction, target)
    return prediction + (target.mean(dim=-2, keepdim=True) - prediction.mean(dim=-2, keepdim=True))


def procrustes_align(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Similarity-align prediction to target using a proper rotation."""
    _validate_points(prediction, target)
    pred_mean = prediction.mean(dim=-2, keepdim=True)
    target_mean = target.mean(dim=-2, keepdim=True)
    pred_centered = prediction - pred_mean
    target_centered = target - target_mean
    covariance = pred_centered.transpose(-1, -2) @ target_centered
    u, singular, vh = torch.linalg.svd(covariance)
    correction = torch.ones_like(singular)
    correction[..., -1] = torch.sign(torch.det(u @ vh)).clamp(min=-1, max=1)
    rotation = u @ torch.diag_embed(correction) @ vh
    denominator = (pred_centered * pred_centered).sum(dim=(-2, -1)).clamp_min(eps)
    scale = (singular * correction).sum(-1) / denominator
    aligned = scale[..., None, None] * (pred_centered @ rotation)
    return aligned + target_mean


def mean_point_error_mm(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _validate_points(prediction, target)
    return torch.linalg.vector_norm(prediction - target, dim=-1).mean(dim=-1) * 1000.0
