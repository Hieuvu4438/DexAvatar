"""Train-only residual normalization for rectified-flow coordinates."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch


class ResidualNormalizer:
    """Affine normalization with one 3-vector statistic per refined joint."""

    def __init__(
        self,
        mean: torch.Tensor | np.ndarray | None = None,
        std: torch.Tensor | np.ndarray | None = None,
        *,
        source: str | Path | None = None,
    ) -> None:
        mean = torch.zeros(51, 3) if mean is None else torch.as_tensor(mean).float()
        std = torch.ones(51, 3) if std is None else torch.as_tensor(std).float()
        if mean.shape != (51, 3) or std.shape != (51, 3):
            raise ValueError("Residual statistics must both have shape [51,3]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Residual statistics must be finite")
        if (std <= 0).any():
            raise ValueError("Residual standard deviations must be positive")
        self.mean = mean
        self.std = std
        self.source = Path(source).resolve() if source is not None else None

    @classmethod
    def from_path(cls, path: str | Path | None) -> "ResidualNormalizer":
        if path is None:
            return cls()
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Residual statistics not found: {source}")
        with np.load(source, allow_pickle=False) as payload:
            return cls(payload["mean"], payload["std"], source=source)

    @property
    def sha256(self) -> str | None:
        if self.source is None:
            return None
        return hashlib.sha256(self.source.read_bytes()).hexdigest()

    def _parameters(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.mean.to(device=value.device, dtype=value.dtype),
            self.std.to(device=value.device, dtype=value.dtype),
        )

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        mean, std = self._parameters(value)
        return (value - mean) / std

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        mean, std = self._parameters(value)
        return value * std + mean


__all__ = ["ResidualNormalizer"]
