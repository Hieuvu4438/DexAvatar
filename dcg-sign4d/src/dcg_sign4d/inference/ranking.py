"""Validation-fitted, ground-truth-free hypothesis ranking contract."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.diffusion.state_codec import TrajectoryState
from dcg_sign4d.geometry.contact_geometry import GeometryOutput
from dcg_sign4d.observations.schema import ObservationBatch


@dataclass(frozen=True)
class RankingWeights:
    observation: float
    contact: float
    event: float
    motion: float
    fitted_split: str = "validation"
    uses_ground_truth: bool = False

    def __post_init__(self) -> None:
        if self.uses_ground_truth:
            raise ValueError("deployment ranking cannot use ground truth")
        if self.fitted_split != "validation":
            raise ValueError("ranking weights must be fit/frozen on validation")


class HypothesisRanker:
    def __init__(
        self,
        weights: RankingWeights,
        *,
        observation_score: Callable[
            [TrajectoryState, ObservationBatch, ContactGraphBatch], Tensor | float
        ]
        | None = None,
        allow_missing_observation_score: bool = False,
        allow_missing_penetration: bool = False,
    ) -> None:
        self.weights = weights
        self.observation_score = observation_score
        self.allow_missing_observation_score = allow_missing_observation_score
        self.allow_missing_penetration = allow_missing_penetration

    def terms(
        self,
        trajectory: TrajectoryState,
        graph: ContactGraphBatch,
        observations: ObservationBatch,
        geometry: GeometryOutput,
    ) -> dict[str, float]:
        if not geometry.penetration_available and not self.allow_missing_penetration:
            raise RuntimeError(
                "hypothesis ranking requires signed penetration geometry; missing penetration "
                "is permitted only for explicitly development-only runs"
            )
        if self.observation_score is None:
            if not self.allow_missing_observation_score:
                raise RuntimeError(
                    "hypothesis ranking requires an audited observation residual; detector "
                    "reliability alone is not a hypothesis-dependent observation score"
                )
            observation = 0.0
        else:
            observation_value = self.observation_score(trajectory, observations, graph)
            observation = float(
                observation_value.detach()
                if isinstance(observation_value, Tensor)
                else observation_value
            )
            if not math.isfinite(observation):
                raise FloatingPointError("observation score is NaN/Inf")
        contact_probability = graph.event_probability[..., 1:3].sum(-1)
        active = graph.edge_valid[:, None, :] & trajectory.valid_mask[:, :, None]
        certain = active & ~graph.uncertain_mask
        if bool(certain.any()):
            contact_cost = (contact_probability * geometry.distance)[certain].mean()
            contact_cost = contact_cost + geometry.penetration_depth[certain].mean()
            contact_cost = contact_cost + geometry.penetration_area[certain].mean()
            hold = (graph.event_state == 2) & certain
            if bool(hold.any()):
                contact_cost = contact_cost + geometry.relative_speed[hold].mean()
        else:
            contact_cost = geometry.distance.sum() * 0
        selected = graph.event_probability.gather(-1, graph.event_state[..., None]).squeeze(-1)
        event = (
            float(selected[certain].clamp_min(1e-12).log().mean().detach())
            if bool(certain.any())
            else 0.0
        )
        velocity = trajectory.root_velocity
        acceleration = velocity[:, 1:] - velocity[:, :-1]
        valid_pair = trajectory.valid_mask[:, 1:] & trajectory.valid_mask[:, :-1]
        motion = -float(acceleration[valid_pair].square().mean()) if bool(valid_pair.any()) else 0.0
        return {
            "observation": observation,
            "contact": -float(contact_cost.detach()),
            "event": event,
            "motion": motion,
        }

    def score(self, terms: dict[str, float]) -> float:
        return sum(getattr(self.weights, name) * value for name, value in terms.items())
