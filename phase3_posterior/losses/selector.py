"""Listwise selector supervision; targets are used only during training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def listwise_ranking_loss(
    scores: torch.Tensor, target_errors: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    if scores.shape != target_errors.shape:
        raise ValueError("scores and target_errors must have matching (B,K) shapes")
    target_distribution = torch.softmax(-target_errors / temperature, dim=-1)
    return -(target_distribution * F.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
