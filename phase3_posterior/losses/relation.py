"""Sparse relation supervision losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def focal_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    gamma: float = 2.0,
    positive_alpha: float | None = None,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    probability = torch.sigmoid(logits)
    correct = torch.where(target, probability, 1.0 - probability)
    loss = loss * (1.0 - correct).pow(gamma)
    mask = valid.to(loss.dtype)
    if positive_alpha is not None:
        alpha = torch.where(target, positive_alpha, 1.0 - positive_alpha)
        loss = loss * alpha
    if weight is not None:
        loss = loss * weight
        mask = mask * weight
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def relation_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    positive_alpha: float | None = None,
    contact_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    contact = focal_bce(
        outputs["contact_logits"],
        batch["contact_target"],
        batch["contact_valid"],
        positive_alpha=positive_alpha,
        weight=contact_weight,
    )
    persistence = focal_bce(
        outputs["persistence_logits"],
        batch["persistence_target"],
        batch["contact_valid"],
        positive_alpha=positive_alpha,
        weight=contact_weight,
    )
    valid = batch["edge_valid"]
    depth = (
        F.cross_entropy(
            outputs["depth_logits"][valid],
            batch["depth_target"][valid],
            reduction="mean",
        )
        if valid.any()
        else outputs["depth_logits"].sum() * 0.0
    )
    distance = outputs["contact_logits"].sum() * 0.0
    if "distance" in outputs:
        distance_valid = batch.get("distance_valid", valid)
        if distance_valid.any():
            difference = torch.nn.functional.smooth_l1_loss(
                outputs["distance"],
                batch["target_edge_features"][..., 3],
                beta=0.01,
                reduction="none",
            )
            mask = distance_valid.to(difference.dtype)
            distance = (difference * mask).sum() / mask.sum().clamp_min(1.0)
    return {
        "contact": contact,
        "persistence": persistence,
        "depth": depth,
        "distance": distance,
    }


def stratified_sign_contact_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    sign_hand_body: torch.Tensor,
    *,
    gamma: float = 2.0,
    hard_negative_ratio: int = 8,
) -> torch.Tensor:
    """Balance rare sign positives against only the hardest sign negatives."""
    if hard_negative_ratio < 1:
        raise ValueError("hard_negative_ratio must be positive")
    positive = sign_hand_body & target
    negative = sign_hand_body & ~target
    raw = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target.float(), reduction="none"
    )
    probability = torch.sigmoid(logits)
    correct = torch.where(target, probability, 1.0 - probability)
    raw = raw * (1.0 - correct).pow(gamma)
    if not positive.any():
        return raw[negative].mean() if negative.any() else logits.sum() * 0.0
    positive_loss = raw[positive].mean()
    negative_values = raw[negative]
    if not negative_values.numel():
        return positive_loss
    keep = min(negative_values.numel(), int(positive.sum()) * hard_negative_ratio)
    hard_negative_loss = negative_values.topk(keep, sorted=False).values.mean()
    return 0.5 * (positive_loss + hard_negative_loss)


def conditional_persistence_loss(
    logits: torch.Tensor,
    persistence_target: torch.Tensor,
    contact_target: torch.Tensor,
    contact_valid: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Predict persistence only where a contact actually exists."""
    valid = contact_valid & contact_target
    return focal_bce(
        logits,
        persistence_target,
        valid,
        gamma=2.0,
        positive_alpha=0.5,
        weight=weight,
    )
