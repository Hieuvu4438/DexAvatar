from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_heatmaps(heatmaps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    heatmaps = heatmaps.clamp_min(0)
    mass = heatmaps.sum(dim=(-2, -1), keepdim=True)
    valid = mass.squeeze(-1).squeeze(-1) > 0
    normalized = heatmaps / mass.clamp_min(1e-12)
    return normalized, valid


def sample_heatmap_nll(
    heatmaps: torch.Tensor,
    xy_pixels: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    batch, joints, height, width = heatmaps.shape
    x = 2.0 * xy_pixels[..., 0] / max(width - 1, 1) - 1.0
    y = 2.0 * xy_pixels[..., 1] / max(height - 1, 1) - 1.0
    grid = torch.stack((x, y), dim=-1).reshape(batch * joints, 1, 1, 2)
    probability = heatmaps.reshape(batch * joints, 1, height, width)
    sampled = F.grid_sample(
        probability,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch, joints)
    nll = -torch.log(sampled.clamp_min(eps))
    weight = valid.to(nll.dtype)
    return (nll * weight).sum() / weight.sum().clamp_min(1.0)


def entropy_confidence(
    heatmaps: torch.Tensor,
    visibility: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized, valid = normalize_heatmaps(heatmaps)
    entropy = -(normalized * torch.log(normalized.clamp_min(1e-12))).sum(dim=(-2, -1))
    max_entropy = torch.log(torch.as_tensor(
        heatmaps.shape[-2] * heatmaps.shape[-1],
        dtype=heatmaps.dtype,
        device=heatmaps.device,
    ))
    confidence = visibility * (1.0 - entropy / max_entropy).clamp(0.0, 1.0)
    return confidence * valid.to(confidence.dtype), entropy

