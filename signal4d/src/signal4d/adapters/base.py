from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch


@dataclass(frozen=True)
class ClipContext:
    clip_id: str
    frame_ids: tuple[int, ...]
    image_paths: tuple[Path, ...]
    fps: float
    camera_k: torch.Tensor


@dataclass
class AdapterOutput:
    source_name: str
    tensors: dict[str, torch.Tensor]
    masks: dict[str, torch.Tensor]
    metadata: dict[str, object]
    diagnostics: dict[str, object]

    def validate_length(self, expected_t: int) -> None:
        for group in (self.tensors, self.masks):
            for name, value in group.items():
                if value.shape[0] != expected_t:
                    raise ValueError(
                        f"adapter {self.source_name} changed T for {name}: "
                        f"{value.shape[0]} != {expected_t}"
                    )


class EstimatorAdapter(Protocol):
    source_name: str

    def validate_assets(self) -> None: ...

    def infer(self, clip: ClipContext) -> AdapterOutput: ...

    def canonicalize(self, raw: AdapterOutput) -> AdapterOutput: ...
