from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class FactorResult:
    loss: torch.Tensor
    valid_count: int
    per_frame: torch.Tensor
    residual_quantiles: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, torch.Tensor | float] = field(default_factory=dict)


@dataclass(frozen=True)
class FactorContext:
    fps: float
    uncertainty: torch.Tensor
    change_probability: torch.Tensor
    contact_candidates: object | None = None


def pseudo_huber(residual: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return delta * delta * (torch.sqrt(1 + (residual / delta).square()) - 1)


def summarize(residual: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, float]:
    values = residual.detach().abs()
    if valid is not None:
        values = values[valid]
    if values.numel() == 0:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
    quantiles = torch.quantile(values.float(), torch.tensor([0.5, 0.9, 0.99], device=values.device))
    return {"p50": float(quantiles[0]), "p90": float(quantiles[1]), "p99": float(quantiles[2])}
