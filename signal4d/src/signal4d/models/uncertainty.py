from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


class UncertaintyCalibrator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 32,
        sigma_min: float = 0.002,
        sigma_max: float = 0.2,
    ) -> None:
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 7),
        )

    def forward(self, features: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.network(features)
        sigma_xyz = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw[..., :3])
        sigma_rot = 0.01 + (1.5 - 0.01) * torch.sigmoid(raw[..., 3:6])
        risk_logit = raw[..., 6]
        sigma_xyz = torch.where(
            valid[..., None], sigma_xyz, torch.full_like(sigma_xyz, self.sigma_max)
        )
        sigma_rot = torch.where(valid[..., None], sigma_rot, torch.full_like(sigma_rot, 1.5))
        risk_logit = torch.where(valid, risk_logit, torch.full_like(risk_logit, 12.0))
        return {"sigma_xyz": sigma_xyz, "sigma_rot": sigma_rot, "risk_logit": risk_logit}


def student_t_nll(
    residual: torch.Tensor, sigma: torch.Tensor, degrees_of_freedom: float = 3.0
) -> torch.Tensor:
    sigma = sigma.clamp_min(1e-8)
    normalized = residual / sigma
    return torch.log(sigma) + 0.5 * (degrees_of_freedom + 1) * torch.log1p(
        normalized.square() / degrees_of_freedom
    )


def heuristic_uncertainty(
    features: torch.Tensor, valid: torch.Tensor, sigma_min: float, sigma_max: float
) -> dict[str, torch.Tensor]:
    score = (
        torch.sigmoid(features[..., :4].mean(dim=-1))
        if features.shape[-1] >= 4
        else torch.sigmoid(features.mean(dim=-1))
    )
    sigma = sigma_min + (sigma_max - sigma_min) * score
    sigma = torch.where(valid, sigma, torch.full_like(sigma, sigma_max))
    return {
        "sigma_xyz": sigma[..., None].expand(*sigma.shape, 3),
        "sigma_rot": (sigma / sigma_max * 1.5).clamp_min(0.01)[..., None].expand(*sigma.shape, 3),
        "risk_logit": torch.logit(score.clamp(1e-5, 1 - 1e-5)),
    }


@dataclass(frozen=True)
class GroupCalibration:
    scales: dict[str, float]
    nominal_coverage: float

    @classmethod
    def fit(
        cls,
        residual: torch.Tensor,
        sigma: torch.Tensor,
        group_ids: list[str],
        nominal_coverage: float = 0.9,
    ) -> GroupCalibration:
        if residual.shape != sigma.shape or residual.numel() != len(group_ids):
            raise ValueError("residual, sigma and group_ids must describe the same samples")
        scores = residual.abs() / sigma.clamp_min(1e-12)
        scales: dict[str, float] = {}
        for group in sorted(set(group_ids)):
            mask = torch.tensor([value == group for value in group_ids], device=scores.device)
            group_scores = scores[mask]
            if not group_scores.numel():
                continue
            rank = min(
                group_scores.numel() - 1,
                int(torch.ceil(torch.tensor((group_scores.numel() + 1) * nominal_coverage)).item())
                - 1,
            )
            scales[group] = float(group_scores.sort().values[max(rank, 0)].item())
        return cls(scales=scales, nominal_coverage=nominal_coverage)

    def write(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"scales": self.scales, "nominal_coverage": self.nominal_coverage},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
