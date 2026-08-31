from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContactProposal:
    region_a: str
    region_b: str
    confidence: float
    target_distance_m: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence={self.confidence}")
        if not 0.0 < self.target_distance_m <= 0.03:
            raise ValueError(f"target distance={self.target_distance_m}")
        if self.region_a == self.region_b:
            raise ValueError("contact regions must differ")


def symmetric_contact_distance(a, b):
    import torch

    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[-1] != 3 or b.shape[-1] != 3:
        raise ValueError(f"contact point shapes {a.shape}/{b.shape}")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError("non-finite contact points")
    distances = torch.cdist(a, b)
    return 0.5 * (
        distances.min(dim=2).values.mean()
        + distances.min(dim=1).values.mean()
    )


def contact_attraction(a, b, target_distance: float = 0.003):
    import torch

    distance = symmetric_contact_distance(a, b)
    return torch.nn.functional.smooth_l1_loss(
        distance, distance.new_tensor(float(target_distance))
    )


def gated_contact_loss(vertices, proposals, regions, confidence_threshold: float = 0.70):
    total = vertices.new_zeros(())
    normalizer = 0.0
    active = 0
    for raw in proposals:
        proposal = raw if isinstance(raw, ContactProposal) else ContactProposal(**raw)
        if proposal.confidence < confidence_threshold:
            continue
        try:
            ids_a, ids_b = regions[proposal.region_a], regions[proposal.region_b]
        except KeyError as error:
            raise KeyError(f"unknown contact region {error.args[0]}") from error
        term = contact_attraction(
            vertices[:, ids_a], vertices[:, ids_b], proposal.target_distance_m
        )
        total = total + proposal.confidence * term
        normalizer += proposal.confidence
        active += 1
    return total / max(normalizer, 1.0), active


def penetration_barrier(signed_distances, margin_m: float = 0.0):
    """Differentiable loss for distances from a collision-search backend.

    Negative signed distances are inside the opposing surface. Collision pair
    discovery stays outside this function (and may run under ``no_grad``), while
    the signed point-to-triangle distances supplied here retain gradients.
    """
    import torch

    if not torch.isfinite(signed_distances).all():
        raise FloatingPointError("non-finite signed distances")
    return torch.square(torch.relu(float(margin_m) - signed_distances)).mean()
