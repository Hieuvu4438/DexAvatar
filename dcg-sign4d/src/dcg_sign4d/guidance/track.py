"""Temporal 2D displacement guidance for tracked points."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as functional
from torch import Tensor

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.diffusion.state_codec import TrajectoryState
from dcg_sign4d.observations.schema import ObservationBatch


class TrackGuidance:
    name = "track"

    def __init__(self, project_tracks: Callable[[TrajectoryState], Tensor]) -> None:
        self.project_tracks = project_tracks

    def loss(
        self,
        clean_state: TrajectoryState,
        observations: ObservationBatch,
        graph: ContactGraphBatch,
    ) -> Tensor:
        del graph
        predicted = self.project_tracks(clean_state)
        if observations.tracks_2d is None:
            return predicted.sum() * 0
        if predicted.shape != observations.tracks_2d.shape:
            raise ValueError("projected and observed tracks must have equal shape")
        observed_delta = observations.tracks_2d[:, 1:] - observations.tracks_2d[:, :-1]
        predicted_delta = predicted[:, 1:] - predicted[:, :-1]
        reliability = torch.sqrt(
            observations.track_reliability[:, 1:].clamp_min(0)
            * observations.track_reliability[:, :-1].clamp_min(0)
        )
        valid_frames = observations.frame_valid[:, 1:] & observations.frame_valid[:, :-1]
        active = (reliability > 0) & valid_frames[:, :, None]
        if not bool(active.any()):
            return predicted.sum() * 0
        residual = torch.linalg.vector_norm(predicted_delta - observed_delta, dim=-1)
        loss = functional.huber_loss(residual, torch.zeros_like(residual), reduction="none")
        return (loss * reliability)[active].mean()
