"""Sparse relation supervision losses."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def focal_bce(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, gamma: float = 2.0
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction="none")
    probability = torch.sigmoid(logits)
    correct = torch.where(target, probability, 1.0 - probability)
    loss = loss * (1.0 - correct).pow(gamma)
    mask = valid.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def relation_losses(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    contact = focal_bce(
        outputs["contact_logits"], batch["contact_target"], batch["contact_valid"]
    )
    persistence = focal_bce(
        outputs["persistence_logits"],
        batch["persistence_target"],
        batch["contact_valid"],
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
    return {"contact": contact, "persistence": persistence, "depth": depth}
