"""Reliability-scaled robust 2D keypoint likelihood."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as functional
from torch import Tensor

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.diffusion.state_codec import TrajectoryState
from dcg_sign4d.observations.schema import ObservationBatch


class KeypointGuidance:
    name = "keypoint"

    def __init__(
        self,
        project_joints: Callable[[TrajectoryState], Tensor],
        sigma_min: float = 1.0,
        sigma_occ: float = 20.0,
    ) -> None:
        self.project_joints = project_joints
        self.sigma_min = sigma_min
        self.sigma_occ = sigma_occ

    def loss(
        self,
        clean_state: TrajectoryState,
        observations: ObservationBatch,
        graph: ContactGraphBatch,
    ) -> Tensor:
        del graph
        projected = self.project_joints(clean_state)
        valid = observations.keypoint_valid & observations.frame_valid[:, :, None]
        if not bool(valid.any()):
            return projected.sum() * 0
        scale = self.sigma_min + (1 - observations.keypoint_reliability) * self.sigma_occ
        residual = (
            torch.linalg.vector_norm(projected - observations.keypoints_2d.nan_to_num(), dim=-1)
            / scale
        )
        return functional.huber_loss(residual[valid], torch.zeros_like(residual[valid]))
