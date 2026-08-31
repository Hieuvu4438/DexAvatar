"""Calibrated cue tensors with explicit validity and missingness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def _check_reliability(name: str, value: Tensor | None) -> None:
    if value is None:
        return
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    if bool(((value < 0) | (value > 1)).any()):
        raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class ObservationBatch:
    keypoints_2d: Tensor
    keypoint_reliability: Tensor
    keypoint_valid: Tensor
    frame_valid: Tensor
    part_masks: Tensor | None = None
    mask_reliability: Tensor | None = None
    tracks_2d: Tensor | None = None
    track_reliability: Tensor | None = None
    depth_order: Tensor | None = None
    depth_reliability: Tensor | None = None
    metadata: tuple[dict[str, Any], ...] = ()

    def validate(self) -> ObservationBatch:
        if self.keypoints_2d.ndim != 4 or self.keypoints_2d.shape[-1] != 2:
            raise ValueError("keypoints_2d must have shape [B,T,J,2]")
        expected = self.keypoints_2d.shape[:-1]
        if self.keypoint_reliability.shape != expected:
            raise ValueError("keypoint_reliability shape mismatch")
        if self.keypoint_valid.shape != expected or self.keypoint_valid.dtype != torch.bool:
            raise ValueError("keypoint_valid must be bool [B,T,J]")
        if self.frame_valid.shape != expected[:2] or self.frame_valid.dtype != torch.bool:
            raise ValueError("frame_valid must be bool [B,T]")
        if not torch.isfinite(self.keypoints_2d[self.keypoint_valid]).all():
            raise ValueError("valid keypoints contain NaN/Inf")
        _check_reliability("keypoint_reliability", self.keypoint_reliability)
        _check_reliability("mask_reliability", self.mask_reliability)
        _check_reliability("track_reliability", self.track_reliability)
        _check_reliability("depth_reliability", self.depth_reliability)
        for cue, reliability in (
            (self.part_masks, self.mask_reliability),
            (self.tracks_2d, self.track_reliability),
            (self.depth_order, self.depth_reliability),
        ):
            if (cue is None) != (reliability is None):
                raise ValueError("optional cue and reliability must be present together")
        batch, time = expected[:2]
        if self.metadata and len(self.metadata) != batch:
            raise ValueError("observation metadata must contain one record per batch item")
        for item in self.metadata:
            frame_ids = item.get("frame_ids")
            timestamps = item.get("timestamps_sec")
            if (frame_ids is None) != (timestamps is None):
                raise ValueError("frame IDs and timestamps must be recorded together")
            if frame_ids is not None:
                if len(frame_ids) != time or len(timestamps) != time:
                    raise ValueError("frame metadata length must match observation time")
                if len(set(frame_ids)) != len(frame_ids) or list(frame_ids) != sorted(frame_ids):
                    raise ValueError("frame IDs must be strictly increasing")
                timestamp_tensor = torch.as_tensor(timestamps, dtype=torch.float64)
                if not torch.isfinite(timestamp_tensor).all() or bool(
                    (timestamp_tensor[1:] <= timestamp_tensor[:-1]).any()
                ):
                    raise ValueError("timestamps must be finite and strictly increasing")
        if self.part_masks is not None:
            if self.part_masks.ndim != 5 or self.part_masks.shape[:2] != (batch, time):
                raise ValueError("part_masks must be [B,T,P,H,W]")
            if self.mask_reliability.shape != self.part_masks.shape[:3]:
                raise ValueError("mask_reliability must be [B,T,P]")
        if self.tracks_2d is not None:
            if self.tracks_2d.ndim != 4 or self.tracks_2d.shape[:2] != (batch, time):
                raise ValueError("tracks_2d must be [B,T,N,2]")
            if self.tracks_2d.shape[-1] != 2:
                raise ValueError("tracks_2d must end in xy")
            if self.track_reliability.shape != self.tracks_2d.shape[:-1]:
                raise ValueError("track_reliability must be [B,T,N]")
        if self.depth_order is not None:
            if self.depth_order.ndim != 3 or self.depth_order.shape[:2] != (batch, time):
                raise ValueError("depth_order must be [B,T,D]")
            if self.depth_reliability.shape != self.depth_order.shape:
                raise ValueError("depth_reliability must match depth_order")
            if bool((self.depth_order.abs() > 1).any()):
                raise ValueError("depth_order values must be in [-1,1]")
        return self
