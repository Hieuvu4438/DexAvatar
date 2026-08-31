from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


SCHEMA_VERSION = "signpk-manifest-v1"


@dataclass(frozen=True)
class FrameRecord:
    sequence_index: int
    video_frame_id: int
    gt_frame_id: int
    prediction_frame_id: int
    timestamp_sec: float
    rgb_path: Path
    gt_obj_path: Path | None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["rgb_path"] = str(self.rgb_path)
        data["gt_obj_path"] = None if self.gt_obj_path is None else str(self.gt_obj_path)
        return data

    @classmethod
    def from_dict(cls, value: Mapping) -> "FrameRecord":
        data = dict(value)
        data["rgb_path"] = Path(data["rgb_path"])
        data["gt_obj_path"] = None if data.get("gt_obj_path") is None else Path(data["gt_obj_path"])
        return cls(**data)


@dataclass(frozen=True)
class SignManifest:
    sign_name: str
    segment_start: int
    segment_end: int
    handedness_class: str
    dominant_hand: str
    sampling_policy: str
    boundary_padding: str
    records: tuple[FrameRecord, ...]
    schema_version: str = SCHEMA_VERSION

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(record.video_frame_id for record in self.records)

    @property
    def gt_ids(self) -> tuple[int, ...]:
        return tuple(record.gt_frame_id for record in self.records)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["records"] = [record.to_dict() for record in self.records]
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, validate_paths: bool = False) -> "SignManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["records"] = tuple(FrameRecord.from_dict(item) for item in data["records"])
        manifest = cls(**data)
        manifest.validate(require_gt=validate_paths, check_paths=validate_paths)
        return manifest

    def validate(self, require_gt: bool = True, check_paths: bool = True) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema {self.schema_version!r}")
        if self.boundary_padding not in {"reflect", "replicate"}:
            raise ValueError(f"invalid boundary padding {self.boundary_padding!r}")
        if not self.records:
            raise ValueError(f"manifest {self.sign_name} contains no frames")
        sequence = [r.sequence_index for r in self.records]
        if sequence != list(range(len(self.records))):
            raise ValueError("sequence_index must be dense and zero-based")
        for field in ("video_frame_id", "gt_frame_id", "prediction_frame_id"):
            values = [getattr(r, field) for r in self.records]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {field}")
            if values != sorted(values):
                raise ValueError(f"{field} must be strictly monotonic")
        timestamps = [r.timestamp_sec for r in self.records]
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("timestamps must be strictly monotonic")
        if check_paths:
            missing_rgb = [str(r.rgb_path) for r in self.records if not r.rgb_path.is_file()]
            if missing_rgb:
                raise FileNotFoundError(f"missing RGB frames: {missing_rgb[:3]}")
            missing_gt = [str(r.gt_obj_path) for r in self.records if r.gt_obj_path and not r.gt_obj_path.is_file()]
            if missing_gt:
                raise FileNotFoundError(f"missing GT meshes: {missing_gt[:3]}")
            if require_gt and any(r.gt_obj_path is None for r in self.records):
                raise FileNotFoundError("manifest has records without GT paths")


def load_sign_classes(path: str | Path) -> dict[str, str]:
    classes: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if tokens:
            classes[tokens[0]] = tokens[1] if len(tokens) > 1 else "unknown"
    return classes


def _extract_id(path: Path, pattern: re.Pattern[str]) -> int:
    match = pattern.search(path.stem)
    if match is None:
        raise ValueError(f"cannot extract frame ID from {path}")
    return int(match.group(1))


def build_manifest(
    sign_name: str,
    segment: tuple[int, int],
    frames_root: str | Path,
    gt_root: str | Path,
    handedness_class: str,
    *,
    image_glob: str = "low_*.png",
    image_id_regex: str = r"(?:low_)?(\d+)",
    gt_id_multiplier: int = 2,
    fps: float = 25.0,
    boundary_padding: str = "reflect",
    strict_gt: bool = True,
) -> SignManifest:
    frames_dir = Path(frames_root) / sign_name
    gt_dir = Path(gt_root) / sign_name
    pattern = re.compile(image_id_regex)
    start, end = (int(segment[0]), int(segment[1]))
    candidates = sorted(
        ((_extract_id(path, pattern), path.resolve()) for path in frames_dir.glob(image_glob)),
        key=lambda pair: pair[0],
    )
    selected = [(frame_id, path) for frame_id, path in candidates if start <= frame_id <= end]
    if not selected:
        raise FileNotFoundError(f"no central frames for {sign_name} in {frames_dir}")
    records: list[FrameRecord] = []
    for sequence_index, (video_id, rgb_path) in enumerate(selected):
        gt_id = video_id * gt_id_multiplier
        matches = sorted(gt_dir.glob(f"{gt_id:05d}.obj")) or sorted(gt_dir.glob(f"{gt_id}.obj"))
        gt_path = matches[0].resolve() if matches else None
        if strict_gt and gt_path is None:
            raise FileNotFoundError(f"GT {gt_id} for {sign_name}/{video_id} is missing")
        records.append(
            FrameRecord(
                sequence_index=sequence_index,
                video_frame_id=video_id,
                gt_frame_id=gt_id,
                prediction_frame_id=gt_id,
                timestamp_sec=video_id / fps,
                rgb_path=rgb_path,
                gt_obj_path=gt_path,
            )
        )
    dominant = "right" if handedness_class == "0" else "unknown"
    manifest = SignManifest(
        sign_name=sign_name,
        segment_start=start,
        segment_end=end,
        handedness_class=handedness_class,
        dominant_hand=dominant,
        sampling_policy=f"video_to_gt_x{gt_id_multiplier}",
        boundary_padding=boundary_padding,
        records=tuple(records),
    )
    manifest.validate(require_gt=strict_gt, check_paths=True)
    return manifest


def build_all_manifests(
    segments: Mapping[str, Iterable[int]],
    signs: Mapping[str, str],
    **kwargs,
) -> list[SignManifest]:
    unknown = sorted(set(signs) - set(segments))
    if unknown:
        raise KeyError(f"signs missing from segment file: {unknown}")
    return [
        build_manifest(name, tuple(segments[name]), handedness_class=class_name, **kwargs)
        for name, class_name in sorted(signs.items())
    ]

