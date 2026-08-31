"""Reliability-aware soft part-mask guidance."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn.functional as functional
from torch import Tensor

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.diffusion.state_codec import TrajectoryState
from dcg_sign4d.observations.schema import ObservationBatch


class SilhouetteGuidance:
    name = "silhouette"

    def __init__(self, render_part_masks: Callable[[TrajectoryState], Tensor]) -> None:
        self.render_part_masks = render_part_masks

    def loss(
        self,
        clean_state: TrajectoryState,
        observations: ObservationBatch,
        graph: ContactGraphBatch,
    ) -> Tensor:
        del graph
        predicted = self.render_part_masks(clean_state)
        if observations.part_masks is None:
            return predicted.sum() * 0
        if predicted.shape != observations.part_masks.shape:
            raise ValueError("rendered and observed part masks must have equal shape")
        reliability = observations.mask_reliability * observations.frame_valid[:, :, None]
        active = reliability > 0
        if not bool(active.any()):
            return predicted.sum() * 0
        target = observations.part_masks.to(predicted).clamp(0, 1)
        probability = predicted.clamp(1e-6, 1 - 1e-6)
        bce = functional.binary_cross_entropy(probability, target, reduction="none").mean((-1, -2))
        intersection = (probability * target).sum((-1, -2))
        denominator = probability.sum((-1, -2)) + target.sum((-1, -2))
        dice = 1 - (2 * intersection + 1) / (denominator + 1)
        return ((bce + dice) * reliability)[active].mean()
