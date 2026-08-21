from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ContactEdgeSpec:
    edge_id: str
    joint_a: int
    joint_b: int
    target_distance_m: float
    enter_threshold_m: float
    exit_threshold_m: float
    allow_tangential_slide: bool = True

    def __post_init__(self) -> None:
        if self.enter_threshold_m >= self.exit_threshold_m:
            raise ValueError("contact edge enter threshold must be below exit threshold")


@dataclass
class ContactCandidates:
    probability: torch.Tensor
    distance: torch.Tensor
    valid: torch.Tensor
    edges: tuple[ContactEdgeSpec, ...]


def propose_contacts(
    joints: torch.Tensor,
    edges: tuple[ContactEdgeSpec, ...],
    uncertainty: torch.Tensor | None = None,
    proposal_radius_m: float = 0.08,
) -> ContactCandidates:
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError("joints must have shape [T,J,3]")
    distances = []
    probabilities = []
    for edge in edges:
        distance = torch.linalg.vector_norm(
            joints[:, edge.joint_a] - joints[:, edge.joint_b], dim=-1
        )
        probability = torch.sigmoid(
            (edge.enter_threshold_m - distance) / max(edge.enter_threshold_m * 0.2, 1e-4)
        )
        if uncertainty is not None:
            local_uq = 0.5 * (uncertainty[:, edge.joint_a] + uncertainty[:, edge.joint_b])
            # Uncertainty attenuates ambiguous proximity but cannot veto very strong
            # geometric evidence on its own; hard decisions still require hysteresis.
            attenuation = 0.7 + 0.3 * torch.exp(-local_uq / proposal_radius_m)
            probability = probability * attenuation
        distances.append(distance)
        probabilities.append(probability)
    distance_tensor = (
        torch.stack(distances, dim=-1) if distances else joints.new_zeros((joints.shape[0], 0))
    )
    probability_tensor = (
        torch.stack(probabilities, dim=-1)
        if probabilities
        else joints.new_zeros((joints.shape[0], 0))
    )
    return ContactCandidates(
        probability=probability_tensor,
        distance=distance_tensor,
        valid=distance_tensor <= proposal_radius_m,
        edges=edges,
    )


def decode_hysteresis(
    probability: torch.Tensor,
    distance: torch.Tensor,
    enter_probability: float,
    exit_probability: float,
    enter_distance_m: float,
    exit_distance_m: float,
) -> torch.Tensor:
    if probability.shape != distance.shape or probability.ndim != 2:
        raise ValueError("probability and distance must match [T,C]")
    result = torch.zeros_like(probability, dtype=torch.bool)
    active = torch.zeros(probability.shape[1], dtype=torch.bool, device=probability.device)
    for index in range(probability.shape[0]):
        entering = (
            (~active)
            & (probability[index] >= enter_probability)
            & (distance[index] <= enter_distance_m)
        )
        exiting = active & (
            (probability[index] < exit_probability) | (distance[index] > exit_distance_m)
        )
        active = (active | entering) & ~exiting
        result[index] = active
    return result
