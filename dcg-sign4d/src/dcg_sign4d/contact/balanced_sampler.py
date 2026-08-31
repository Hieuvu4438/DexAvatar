"""HACO-style edge/class balancing adapted to dynamic contact events."""

from __future__ import annotations

import torch
from torch import Tensor


def effective_number_weights(counts: Tensor, beta: float = 0.999) -> Tensor:
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0,1)")
    counts = counts.float()
    weights = torch.zeros_like(counts)
    observed = counts > 0
    weights[observed] = (1 - beta) / (1 - beta ** counts[observed])
    if bool(observed.any()):
        weights[observed] /= weights[observed].mean()
    return weights


def balanced_sample_weights(labels: Tensor, counts: Tensor, beta: float = 0.999) -> Tensor:
    if labels.dtype != torch.long:
        raise ValueError("labels must be long")
    class_weights = effective_number_weights(counts, beta)
    return class_weights[labels]


def balanced_window_sample_weights(
    labels: Tensor,
    edge_valid: Tensor,
    frame_valid: Tensor,
    uncertain: Tensor,
    *,
    beta: float = 0.999,
) -> Tensor:
    """HACO-style edge/event rarity weights for sampling trajectory windows."""

    if labels.ndim != 3 or labels.dtype != torch.long:
        raise ValueError("labels must be long [B,T,E]")
    batch, time, edges = labels.shape
    if edge_valid.shape != (batch, edges) or frame_valid.shape != (batch, time):
        raise ValueError("edge/frame masks do not match labels")
    if uncertain.shape != labels.shape or uncertain.dtype != torch.bool:
        raise ValueError("uncertain mask must be bool [B,T,E]")
    active = edge_valid[:, None, :] & frame_valid[:, :, None] & ~uncertain
    edge_ids = torch.arange(edges, device=labels.device)[None, None, :].expand_as(labels)
    joint_class = edge_ids * 4 + labels
    counts = torch.bincount(joint_class[active], minlength=edges * 4)
    rarity = effective_number_weights(counts, beta)
    window_weights = labels.new_zeros(batch, dtype=torch.float)
    for index in range(batch):
        selected = rarity[joint_class[index][active[index]]]
        if selected.numel():
            window_weights[index] = 0.5 * (selected.mean() + selected.max())
    if not bool((window_weights > 0).any()):
        raise ValueError("no certain valid contact labels are available for sampling")
    minimum = window_weights[window_weights > 0].min()
    return torch.where(window_weights > 0, window_weights, minimum)
