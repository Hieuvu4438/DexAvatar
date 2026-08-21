from __future__ import annotations

import torch


def coverage(error: torch.Tensor, radius: torch.Tensor) -> float:
    if error.shape != radius.shape:
        raise ValueError("error and radius must match")
    return float((error <= radius).float().mean())


def risk_coverage_curve(error: torch.Tensor, risk: torch.Tensor) -> dict[str, torch.Tensor | float]:
    error = error.flatten()
    risk = risk.flatten()
    if error.shape != risk.shape or error.numel() == 0:
        raise ValueError("error and risk must be non-empty and match")
    order = torch.argsort(risk)
    sorted_error = error[order]
    cumulative = torch.cumsum(sorted_error, dim=0) / torch.arange(
        1, error.numel() + 1, device=error.device
    )
    coverage_values = (
        torch.arange(1, error.numel() + 1, device=error.device, dtype=error.dtype) / error.numel()
    )
    aurc = float(torch.trapezoid(cumulative, coverage_values))
    return {"coverage": coverage_values, "selective_risk": cumulative, "aurc": aurc}


def spearman_risk_error(error: torch.Tensor, risk: torch.Tensor) -> float:
    error_rank = torch.argsort(torch.argsort(error.flatten())).float()
    risk_rank = torch.argsort(torch.argsort(risk.flatten())).float()
    error_rank -= error_rank.mean()
    risk_rank -= risk_rank.mean()
    denominator = torch.sqrt((error_rank.square().sum()) * (risk_rank.square().sum())).clamp_min(
        1e-12
    )
    return float((error_rank * risk_rank).sum() / denominator)
