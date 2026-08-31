"""Channel-balanced denoising objective."""

from __future__ import annotations

import torch
from torch import Tensor


def denoising_loss(
    predicted: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    channel_weights: Tensor,
    supervision_mask: Tensor | None = None,
) -> Tensor:
    if predicted.shape != target.shape:
        raise ValueError("prediction/target shape mismatch")
    if channel_weights.shape != predicted.shape[-1:]:
        raise ValueError("channel weight mismatch")
    squared = (predicted - target).square() * channel_weights.to(predicted)
    active = valid_mask[..., None].expand_as(squared)
    if supervision_mask is not None:
        if supervision_mask.shape != predicted.shape or supervision_mask.dtype != torch.bool:
            raise ValueError("supervision mask must be bool [B,T,D]")
        active = active & supervision_mask
    if not bool(active.any()):
        return predicted.sum() * 0
    return squared[active].mean()
