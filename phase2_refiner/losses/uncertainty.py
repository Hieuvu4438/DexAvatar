"""Heteroscedastic likelihood helpers."""

from __future__ import annotations

import torch

from phase2_refiner.losses.geometry import masked_mean


def heteroscedastic_nll(
    error: torch.Tensor, log_variance: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    log_variance = log_variance.clamp(-8.0, 6.0)
    nll = 0.5 * (error.square() * torch.exp(-log_variance) + log_variance)
    return masked_mean(nll, valid)


def regional_worst_decile_ranking_loss(
    error: torch.Tensor,
    log_variance: torch.Tensor,
    valid: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Rank the worst-decile error above ordinary error in every region."""
    regions = (slice(0, 21), slice(21, 36), slice(36, 51))
    losses = []
    for indices in regions:
        selected_error = error[..., indices][valid[..., indices]]
        selected_variance = log_variance[..., indices][valid[..., indices]]
        count = selected_error.numel()
        if count < 20:
            continue
        high_count = max(1, int(round(count * 0.10)))
        order = torch.argsort(selected_error.detach())
        high = selected_variance[order[-high_count:]].mean()
        ordinary = selected_variance[order[:-high_count]].mean()
        losses.append(torch.nn.functional.softplus(margin - (high - ordinary)))
    if not losses:
        return error.new_zeros(())
    return torch.stack(losses).mean()
