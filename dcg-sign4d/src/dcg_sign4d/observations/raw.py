"""Raw detector outputs kept separate from calibrated observation likelihoods."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RawKeypointBatch:
    frame_ids: Tensor
    timestamps_sec: Tensor
    keypoints_2d: Tensor
    raw_score: Tensor
    keypoint_available: Tensor
    frame_available: Tensor

    def validate(self) -> RawKeypointBatch:
        if self.frame_ids.ndim != 1 or self.frame_ids.dtype != torch.long:
            raise ValueError("frame_ids must be long [T]")
        time = self.frame_ids.shape[0]
        if self.timestamps_sec.shape != (time,):
            raise ValueError("timestamps_sec must be [T]")
        if self.keypoints_2d.ndim != 3 or self.keypoints_2d.shape[0] != time:
            raise ValueError("raw keypoints must be [T,J,2]")
        if self.keypoints_2d.shape[-1] != 2:
            raise ValueError("raw keypoints must end in xy")
        expected = self.keypoints_2d.shape[:-1]
        if self.raw_score.shape != expected or self.keypoint_available.shape != expected:
            raise ValueError("raw score/availability shape mismatch")
        if self.frame_available.shape != (time,):
            raise ValueError("frame availability must be [T]")
        if self.keypoint_available.dtype != torch.bool or self.frame_available.dtype != torch.bool:
            raise ValueError("raw availability masks must be boolean")
        if not torch.isfinite(self.keypoints_2d[self.keypoint_available]).all():
            raise ValueError("available raw keypoints contain NaN/Inf")
        if not torch.isfinite(self.raw_score[self.keypoint_available]).all():
            raise ValueError("available raw scores contain NaN/Inf")
        # Detector scores are deliberately not constrained to [0,1]. Existing
        # Sapiens outputs exceed one, so only a fitted calibrator may turn them
        # into probabilities/reliabilities.
        return self
