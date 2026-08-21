from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ..utils.hashing import sha256_json

Split = Literal["train", "calibration", "development", "test"]


class ClipManifest(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset: str
    clip_id: str
    signer_id: str = "unknown"
    split: Split
    fps: float = Field(gt=0)
    frame_ids: list[int] = Field(min_length=1)
    image_relpaths: list[str] = Field(min_length=1)
    frame_start: int | None = None
    frame_end_exclusive: int | None = None
    is_contiguous: bool = True
    gt_relpath: str | None = None
    sign_type: str = "unknown"
    language: str = "unknown"
    allowed_for_calibration: bool = False
    allowed_for_hparam_selection: bool = False
    allowed_for_final_reporting: bool = False

    @model_validator(mode="after")
    def validate_frames(self) -> ClipManifest:
        if len(self.frame_ids) != len(self.image_relpaths):
            raise ValueError("frame_ids and image_relpaths must have the same length")
        if len(set(self.frame_ids)) != len(self.frame_ids):
            raise ValueError("frame_ids must be unique")
        if self.frame_ids != sorted(self.frame_ids):
            raise ValueError("frame_ids must be sorted")
        if self.is_contiguous:
            start = self.frame_ids[0] if self.frame_start is None else self.frame_start
            end = (
                self.frame_ids[-1] + 1
                if self.frame_end_exclusive is None
                else self.frame_end_exclusive
            )
            if self.frame_ids != list(range(start, end)):
                raise ValueError("contiguous manifest must use [start, end_exclusive) without gaps")
        if self.split == "test" and (
            self.allowed_for_calibration or self.allowed_for_hparam_selection
        ):
            raise ValueError("test clips cannot allow calibration or hyperparameter selection")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


def load_manifest(path: str | Path) -> list[ClipManifest]:
    rows: list[ClipManifest] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(ClipManifest.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid manifest row {line_number} in {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    clip_ids = [row.clip_id for row in rows]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("clip_id values must be unique")
    return rows


def write_manifest(rows: list[ClipManifest], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n")


def build_sgnify_manifest(
    frames_root: str | Path,
    segment_path: str | Path,
    split: Split = "development",
    fps: float = 15.0,
) -> list[ClipManifest]:
    frames_root = Path(frames_root)
    segments = json.loads(Path(segment_path).read_text(encoding="utf-8"))
    rows: list[ClipManifest] = []
    for clip_id in sorted(segments):
        image_paths = sorted(
            (frames_root / clip_id).glob("low_*.png"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        if not image_paths:
            raise FileNotFoundError(f"No frames for {clip_id} under {frames_root}")
        available = {int(path.stem.split("_")[-1]): path for path in image_paths}
        start, end_inclusive = map(int, segments[clip_id])
        selected = [
            frame_id for frame_id in sorted(available) if start <= frame_id <= end_inclusive
        ]
        if not selected:
            raise ValueError(f"No selected central frames for {clip_id}")
        relpaths = [
            str(available[frame_id].relative_to(frames_root.parent)) for frame_id in selected
        ]
        contiguous = selected == list(range(selected[0], selected[-1] + 1))
        rows.append(
            ClipManifest(
                dataset="sgnify",
                clip_id=clip_id,
                split=split,
                fps=fps,
                frame_ids=selected,
                image_relpaths=relpaths,
                frame_start=selected[0] if contiguous else None,
                frame_end_exclusive=selected[-1] + 1 if contiguous else None,
                is_contiguous=contiguous,
                gt_relpath=f"data/smplx_gt/{clip_id}",
                allowed_for_calibration=split == "calibration",
                allowed_for_hparam_selection=split == "development",
                allowed_for_final_reporting=split == "test",
            )
        )
    return rows
