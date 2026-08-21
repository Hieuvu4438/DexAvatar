from __future__ import annotations

import torch
import torch.nn.functional as functional

from ..models.contact_proposer import ContactCandidates
from .base import FactorResult, pseudo_huber, summarize


def contact_factor(
    joints: torch.Tensor,
    contact_logits: torch.Tensor,
    candidates: ContactCandidates,
    change_probability: torch.Tensor,
) -> FactorResult:
    switches = torch.sigmoid(contact_logits)
    distances = torch.stack(
        [
            torch.linalg.vector_norm(joints[:, edge.joint_a] - joints[:, edge.joint_b], dim=-1)
            for edge in candidates.edges
        ],
        dim=-1,
    )
    targets = joints.new_tensor([edge.target_distance_m for edge in candidates.edges])
    attraction = switches * pseudo_huber((distances - targets) / 0.01, 2.0)
    evidence = functional.binary_cross_entropy_with_logits(
        contact_logits, candidates.probability, reduction="none"
    )
    persistence_weight = (1 - change_probability[1:, None]).clamp_min(0.05)
    persistence = (switches[1:] - switches[:-1]).abs() * persistence_weight
    total = attraction.mean() + evidence.mean() + 0.1 * persistence.mean()
    per_frame = attraction + evidence
    if persistence.numel():
        per_frame[1:] += 0.1 * persistence
    return FactorResult(
        total,
        total.numel(),
        per_frame.mean(-1),
        summarize(distances),
        {"mean_switch": switches.mean().detach()},
    )
