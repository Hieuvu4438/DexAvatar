"""Relative-depth ordering guidance without an absolute monocular scale."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn.functional as functional
from torch import Tensor

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.diffusion.state_codec import TrajectoryState
from dcg_sign4d.observations.schema import ObservationBatch


class RelativeDepthGuidance:
    name = "depth"

    def __init__(
        self,
        predict_depth_difference: Callable[[TrajectoryState], Tensor],
        temperature_m: float = 0.02,
    ) -> None:
        if temperature_m <= 0:
            raise ValueError("depth temperature must be positive")
        self.predict_depth_difference = predict_depth_difference
        self.temperature_m = temperature_m

    def loss(
        self,
        clean_state: TrajectoryState,
        observations: ObservationBatch,
        graph: ContactGraphBatch,
    ) -> Tensor:
        del graph
        predicted = self.predict_depth_difference(clean_state)
        if observations.depth_order is None:
            return predicted.sum() * 0
        if predicted.shape != observations.depth_order.shape:
            raise ValueError("predicted depth differences must match observed ordering")
        reliability = observations.depth_reliability * observations.frame_valid[:, :, None]
        active = (reliability > 0) & (observations.depth_order != 0)
        if not bool(active.any()):
            return predicted.sum() * 0
        signed_margin = observations.depth_order.to(predicted) * predicted / self.temperature_m
        return (functional.softplus(-signed_margin) * reliability)[active].mean()
