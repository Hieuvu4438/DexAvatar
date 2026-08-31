"""Clean-room adapter for the published DPoser-X whole-body normalizers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from dcg_sign4d.utils.hashing import file_sha256


@dataclass(frozen=True)
class ZScoreStats:
    mean: Tensor
    std: Tensor

    def validate(self, dimension: int) -> None:
        if self.mean.shape != (dimension,) or self.std.shape != (dimension,):
            raise ValueError(f"normalizer statistics must have shape [{dimension}]")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("normalizer statistics must be finite")
        if bool((self.std <= 0).any()):
            raise ValueError("normalizer standard deviations must be positive")


class DPoserXWholeBodyNormalizer:
    """DPoser-X order: body, mirrored left hand, right hand, jaw, expression."""

    PARTS: tuple[tuple[str, int], ...] = (
        ("body_pose", 63),
        ("left_hand_pose", 45),
        ("right_hand_pose", 45),
        ("jaw_pose", 3),
        ("expression", 100),
    )
    TOTAL_DIMENSION = 256
    RELATIVE_PATHS = {
        "body_pose": "data/body_data/body_normalizer/axis_normalize2.pt",
        "left_hand_pose": "data/hand_data/hand_normalizer/axis_normalize2.pt",
        "right_hand_pose": "data/hand_data/hand_normalizer/axis_normalize2.pt",
        "jaw_pose": "data/face_data/jaw_normalizer/axis_normalize2.pt",
        "expression": "data/face_data/expression_normalizer/axis_normalize2.pt",
    }

    def __init__(self, statistics: dict[str, ZScoreStats]) -> None:
        expected = {name for name, _ in self.PARTS}
        if set(statistics) != expected:
            raise ValueError(f"normalizer parts must be exactly {sorted(expected)}")
        for name, dimension in self.PARTS:
            statistics[name].validate(dimension)
        self.statistics = statistics

    @classmethod
    def from_runtime_root(
        cls,
        root: str | Path,
        *,
        expected_hashes: dict[str, str],
        device: torch.device,
    ) -> DPoserXWholeBodyNormalizer:
        root = Path(root)
        statistics: dict[str, ZScoreStats] = {}
        for name, relative in cls.RELATIVE_PATHS.items():
            path = root / relative
            if file_sha256(path) != expected_hashes.get(relative):
                raise ValueError(f"DPoser-X normalizer hash mismatch: {path}")
            values = torch.load(path, map_location=device, weights_only=True)
            statistics[name] = ZScoreStats(
                mean=values["mean_poses"].to(device=device),
                std=values["std_poses"].to(device=device),
            )
        return cls(statistics)

    @staticmethod
    def flip_left_hand(axis_angle: Tensor) -> Tensor:
        if axis_angle.shape[-1] != 45:
            raise ValueError("left hand axis-angle vector must have 45 values")
        result = axis_angle.clone()
        result[..., 1::3] *= -1
        result[..., 2::3] *= -1
        return result

    def normalize_parts(self, parts: dict[str, Tensor]) -> Tensor:
        vectors: list[Tensor] = []
        batch_size: int | None = None
        for name, dimension in self.PARTS:
            value = parts[name]
            if value.ndim != 2 or value.shape[1] != dimension:
                raise ValueError(f"{name} must have shape [B,{dimension}]")
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError("all DPoser-X parts must have the same batch size")
            if name == "left_hand_pose":
                value = self.flip_left_hand(value)
            stats = self.statistics[name]
            vectors.append((value - stats.mean) / stats.std)
        return torch.cat(vectors, dim=-1)

    def denormalize_parts(self, vector: Tensor) -> dict[str, Tensor]:
        if vector.ndim != 2 or vector.shape[1] != self.TOTAL_DIMENSION:
            raise ValueError("normalized DPoser-X vector must have shape [B,256]")
        result: dict[str, Tensor] = {}
        start = 0
        for name, dimension in self.PARTS:
            stats = self.statistics[name]
            value = vector[:, start : start + dimension] * stats.std + stats.mean
            if name == "left_hand_pose":
                value = self.flip_left_hand(value)
            result[name] = value
            start += dimension
        return result
