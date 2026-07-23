"""Padded whole-sequence dataset backed by Phase 2 NPZ caches."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.geometry.rotations import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)


def features_from_clip(
    clip, window: slice | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    window = window or slice(0, len(clip.frame_names))
    init_aa = torch.from_numpy(clip.init_axis_angle[window]).float()
    init_matrix = axis_angle_to_matrix(init_aa)
    rot6d = matrix_to_rotation_6d(init_matrix)
    velocity = torch.zeros_like(rot6d)
    acceleration = torch.zeros_like(rot6d)
    if len(rot6d) > 1:
        velocity[1:] = rot6d[1:] - rot6d[:-1]
    if len(rot6d) > 2:
        acceleration[2:] = velocity[2:] - velocity[1:-1]
    observations = torch.from_numpy(clip.observation_features[window]).float()
    keypoints = torch.from_numpy(clip.keypoints_2d[window]).float()
    return torch.cat(
        (rot6d, velocity, acceleration, observations, keypoints), dim=-1
    ), init_matrix


def collate_sequences(items: list[dict]) -> dict:
    tensor_keys = (
        "features",
        "initial_matrix",
        "target_matrix",
        "joint_valid",
        "frame_valid",
        "refine_mask",
        "length",
    )
    batch = {key: torch.stack([item[key] for item in items]) for key in tensor_keys}
    batch["clip_id"] = [item["clip_id"] for item in items]
    batch["frame_names"] = [item["frame_names"] for item in items]
    return batch


class SequenceCacheDataset(Dataset):
    def __init__(
        self,
        cache_glob: str,
        max_frames: int = 64,
        training: bool = False,
        identity_target: bool = False,
        seed: int = 42,
    ) -> None:
        self.paths = [Path(path) for path in sorted(glob.glob(cache_glob))]
        if not self.paths:
            raise ValueError(f"No cache files match: {cache_glob}")
        self.max_frames = max_frames
        self.training = training
        self.identity_target = identity_target
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def _window(self, length: int) -> slice:
        if length <= self.max_frames:
            return slice(0, length)
        if self.training:
            start = int(self.rng.integers(0, length - self.max_frames + 1))
        else:
            start = (length - self.max_frames) // 2
        return slice(start, start + self.max_frames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | list[str]]:
        clip = load_cache_clip(self.paths[index])
        window = self._window(len(clip.frame_names))
        features, init_matrix = features_from_clip(clip, window)
        init_aa = torch.from_numpy(clip.init_axis_angle[window]).float()

        target_np = clip.target_axis_angle
        if target_np is None:
            if not self.identity_target:
                raise ValueError(
                    f"{self.paths[index]} has no target. Use identity_target only for smoke tests."
                )
            target_aa = init_aa.clone()
        else:
            target_aa = torch.from_numpy(target_np[window]).float()
        target_matrix = axis_angle_to_matrix(target_aa)

        length = len(init_aa)
        pad = self.max_frames - length
        if pad > 0:
            features = torch.nn.functional.pad(features, (0, 0, 0, 0, 0, pad))
            init_matrix = torch.nn.functional.pad(
                init_matrix, (0, 0, 0, 0, 0, 0, 0, pad)
            )
            target_matrix = torch.nn.functional.pad(
                target_matrix, (0, 0, 0, 0, 0, 0, 0, pad)
            )
            joint_valid = torch.nn.functional.pad(
                torch.from_numpy(clip.keypoint_valid[window]), (0, 0, 0, pad)
            )
        else:
            joint_valid = torch.from_numpy(clip.keypoint_valid[window])
        frame_valid = torch.zeros(self.max_frames, dtype=torch.bool)
        frame_valid[:length] = True
        return {
            "clip_id": clip.clip_id,
            "frame_names": clip.frame_names[window].astype(str).tolist(),
            "features": features,
            "initial_matrix": init_matrix,
            "target_matrix": target_matrix,
            "joint_valid": joint_valid.bool(),
            "frame_valid": frame_valid,
            "refine_mask": torch.from_numpy(clip.refine_mask).bool(),
            "length": torch.tensor(length, dtype=torch.long),
        }
