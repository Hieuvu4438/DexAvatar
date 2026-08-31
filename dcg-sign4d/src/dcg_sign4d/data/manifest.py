"""Strict video-manifest contract and frame-map validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ManifestItem(BaseModel):
    """One immutable input clip under the method-freeze schema."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clip_id: str = Field(min_length=1)
    video_path: Path
    fps_native: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    signer_id: str = Field(min_length=1)
    split: Literal["train", "calibration", "validation", "test", "gold_validation"]
    camera_id: str = Field(min_length=1)
    dataset_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    license_id: str = Field(min_length=1)
    fps_effective: float | None = Field(default=None, gt=0)
    frame_mapping: tuple[int, ...] | None = None

    @field_validator("clip_id", "dataset_version", "license_id")
    @classmethod
    def reject_placeholders(cls, value: str) -> str:
        if value.strip().upper() in {"AUTHOR_REQUIRED", "UNKNOWN", "TODO"}:
            raise ValueError("scientific/provenance placeholder is not valid manifest data")
        return value

    @model_validator(mode="after")
    def validate_mapping(self) -> ManifestItem:
        mapping = self.frame_mapping
        if mapping is None:
            if self.fps_effective is not None and self.fps_effective != self.fps_native:
                raise ValueError("resampled clips require an exact frame_mapping")
            return self
        if not mapping:
            raise ValueError("frame_mapping cannot be empty")
        if tuple(sorted(set(mapping))) != mapping:
            raise ValueError("frame_mapping must be strictly increasing and unique")
        if mapping[0] < 0 or mapping[-1] >= self.frame_count:
            raise ValueError("frame_mapping contains a native index outside the clip")
        if self.fps_effective is None:
            raise ValueError("frame_mapping requires fps_effective")
        return self

    @property
    def effective_frame_count(self) -> int:
        return len(self.frame_mapping) if self.frame_mapping is not None else self.frame_count

    def timestamp_sec(self, frame_idx: int) -> float:
        if not 0 <= frame_idx < self.effective_frame_count:
            raise IndexError(frame_idx)
        if self.frame_mapping is None:
            return frame_idx / self.fps_native
        return self.frame_mapping[frame_idx] / self.fps_native


def load_manifest(path: str | Path, *, require_existing_video: bool = False) -> list[ManifestItem]:
    """Load JSONL without silently accepting duplicate clips or missing videos."""

    manifest_path = Path(path)
    items: list[ManifestItem] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                item = ManifestItem.model_validate(raw)
            except Exception as exc:
                raise ValueError(f"{manifest_path}:{line_no}: {exc}") from exc
            if item.clip_id in seen:
                raise ValueError(f"{manifest_path}:{line_no}: duplicate clip_id {item.clip_id!r}")
            if require_existing_video and not item.video_path.is_file():
                raise FileNotFoundError(item.video_path)
            seen.add(item.clip_id)
            items.append(item)
    if not items:
        raise ValueError(f"empty manifest: {manifest_path}")
    return items
