from __future__ import annotations

import torch
from torch import Tensor


def geman_mcclure(residual: Tensor, scale: float | Tensor, eps: float = 1e-12) -> Tensor:
    squared = residual.square()
    scale_squared = torch.as_tensor(scale, dtype=residual.dtype, device=residual.device).square()
    return scale_squared * squared / (scale_squared + squared + eps)


def charbonnier(residual: Tensor, epsilon: float = 1e-6) -> Tensor:
    return torch.sqrt(residual.square() + epsilon * epsilon)


def masked_mean(values: Tensor, valid: Tensor | None = None, eps: float = 1e-8) -> Tensor:
    if valid is None:
        return values.mean() if values.numel() else values.sum()
    weights = valid.to(dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(eps)

