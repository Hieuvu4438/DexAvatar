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
