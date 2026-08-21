from __future__ import annotations

import torch

from ..geometry.alignment import mean_point_error_mm, procrustes_align, translation_align


def tr_v2v_mm(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mean_point_error_mm(translation_align(prediction, target), target)


def pa_mpvpe_mm(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mean_point_error_mm(procrustes_align(prediction, target), target)


def region_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    regions: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    metrics: dict[str, torch.Tensor] = {}
    for name, indices in regions.items():
        metrics[f"tr_v2v_{name}_mm"] = tr_v2v_mm(
            prediction[..., indices, :], target[..., indices, :]
        )
        metrics[f"pa_mpvpe_{name}_mm"] = pa_mpvpe_mm(
            prediction[..., indices, :], target[..., indices, :]
        )
    return metrics
