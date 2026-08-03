"""SO(3) state conversion and normalization for the 51-joint posterior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from phase2_refiner.geometry.rotations import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


NUM_JOINTS = 51
ROTATION_DIM = 6
BODY_JOINTS = slice(0, 21)
LEFT_HAND_JOINTS = slice(21, 36)
RIGHT_HAND_JOINTS = slice(36, 51)


@dataclass(frozen=True)
class RotationNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def __post_init__(self) -> None:
        if self.mean.shape != (NUM_JOINTS, ROTATION_DIM):
            raise ValueError(f"mean must be (51,6), got {tuple(self.mean.shape)}")
        if self.std.shape != (NUM_JOINTS, ROTATION_DIM):
            raise ValueError(f"std must be (51,6), got {tuple(self.std.shape)}")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("normalization contains NaN or Inf")
        if torch.any(self.std <= 0):
            raise ValueError("normalization std must be positive")

    @classmethod
    def identity(cls) -> "RotationNormalizer":
        return cls(
            torch.zeros(NUM_JOINTS, ROTATION_DIM), torch.ones(NUM_JOINTS, ROTATION_DIM)
        )

    @classmethod
    def load(cls, path: str | Path) -> "RotationNormalizer":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                torch.from_numpy(data["mean"]).float(),
                torch.from_numpy(data["std"]).float(),
            )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            np.savez_compressed(
                handle, mean=self.mean.cpu().numpy(), std=self.std.cpu().numpy()
            )

    def to(self, device: torch.device | str) -> "RotationNormalizer":
        return RotationNormalizer(self.mean.to(device), self.std.to(device))

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.std

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std + self.mean


def matrices_to_state(
    matrix: torch.Tensor, normalizer: RotationNormalizer | None = None
) -> torch.Tensor:
    if matrix.shape[-3:] != (NUM_JOINTS, 3, 3):
        raise ValueError(f"Expected (...,51,3,3), got {tuple(matrix.shape)}")
    value = matrix_to_rotation_6d(matrix)
    return value if normalizer is None else normalizer.normalize(value)


def state_to_matrices(
    state: torch.Tensor, normalizer: RotationNormalizer | None = None
) -> torch.Tensor:
    if state.shape[-2:] != (NUM_JOINTS, ROTATION_DIM):
        raise ValueError(f"Expected (...,51,6), got {tuple(state.shape)}")
    value = state if normalizer is None else normalizer.denormalize(state)
    return rotation_6d_to_matrix(value.float())


def region_ids(device: torch.device | str | None = None) -> torch.Tensor:
    values = torch.zeros(NUM_JOINTS, dtype=torch.long, device=device)
    values[LEFT_HAND_JOINTS] = 1
    values[RIGHT_HAND_JOINTS] = 2
    return values
