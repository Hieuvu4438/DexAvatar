"""Candidate validity, robust normalization, selection, and disagreement."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cusp_sl.geometry import geodesic_distance


def robust_standardize(value: torch.Tensor, median: torch.Tensor, mad: torch.Tensor) -> torch.Tensor:
    return (value - median) / mad.clamp_min(1e-6)


def huber(value: torch.Tensor, delta: float) -> torch.Tensor:
    absolute = value.abs()
    return torch.where(absolute <= delta, 0.5 * value.square(), delta * (absolute - 0.5 * delta))


@dataclass
class EnergyStatistics:
    median: torch.Tensor
    mad: torch.Tensor

    @classmethod
    def fit(cls, terms: torch.Tensor) -> "EnergyStatistics":
        median = terms.median(dim=0).values
        mad = (terms - median).abs().median(dim=0).values
        return cls(median=median, mad=mad.clamp_min(1e-6))


def candidate_energy(
    terms: torch.Tensor, statistics: EnergyStatistics, weights: torch.Tensor
) -> torch.Tensor:
    if terms.shape[-1] != 4 or weights.shape != (4,):
        raise ValueError("Expected terms [...,4] and weights [4]")
    standardized = robust_standardize(terms, statistics.median, statistics.mad)
    signed_weights = weights.clone()
    signed_weights[3] = -signed_weights[3]
    return (standardized * signed_weights).sum(dim=-1)


def select_candidates(
    rotations: torch.Tensor, energy: torch.Tensor, valid: torch.Tensor,
    temperature: float,
) -> dict[str, torch.Tensor]:
    """Select from rotations [B,K,T,J,3,3], always honoring validity."""
    if rotations.ndim != 6 or energy.shape != valid.shape:
        raise ValueError("Invalid candidate shapes")
    safe = energy.masked_fill(~valid, torch.inf)
    if (~valid).all(dim=1).any():
        raise ValueError("At least one valid candidate is required per sequence")
    index = safe.argmin(dim=1)
    gather = index[:, None, None, None, None, None].expand(
        -1, 1, rotations.shape[2], rotations.shape[3], 3, 3
    )
    selected = rotations.gather(1, gather).squeeze(1)
    logits = (-safe / temperature).masked_fill(~valid, -torch.inf)
    probability = logits.softmax(dim=1)
    distance = geodesic_distance(rotations, selected[:, None]).square()
    disagreement = (probability[:, :, None, None] * distance).sum(dim=1)
    margin = safe.topk(min(2, safe.shape[1]), largest=False).values
    margin = margin[:, 1] - margin[:, 0] if margin.shape[1] > 1 else torch.full_like(margin[:, 0], torch.inf)
    return {
        "index": index,
        "rotation": selected,
        "weights": probability,
        "disagreement": disagreement,
        "energy_margin": margin,
    }
