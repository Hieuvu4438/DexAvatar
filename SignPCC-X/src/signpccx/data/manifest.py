from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Iterable

from signpccx.io import atomic_write_json, atomic_write_text, sha256_file


_DIGITS = re.compile(r"\d+")


def first_int(path: Path) -> int:
    match = _DIGITS.search(path.stem)
    if match is None:
        raise ValueError(f"Filename has no integer: {path}")
    return int(match.group())


@dataclass(frozen=True)
class FrameRecord:
    sign: str
    sign_class: str
    source_path: str
    source_frame_id: int
    sequence_index: int
    evaluator_index: int
    gt_frame_id: int | None


def read_sign_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        tokens = raw.split()
        if tokens:
            if len(tokens) < 2:
                raise ValueError(f"Expected '<sign> <class>': {raw!r}")
            result[tokens[0]] = tokens[1]
    return dict(sorted(result.items()))


def image_paths(folder: Path) -> list[Path]:
    paths = [path for path in folder.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    return sorted(paths, key=first_int)


def evaluator_gt_ids(gt_sign_dir: Path, segment: tuple[int, int]) -> list[int]:
    available = {int(path.stem) for path in gt_sign_dir.glob("*.obj")}
    start, end = int(segment[0]) * 2, int(segment[1]) * 2
    return [frame for frame in range(start, end + 1) if frame in available]


def build_sign_manifest(
    sign: str,
    sign_class: str,
    image_dir: Path,
    gt_sign_dir: Path | None,
    segment: tuple[int, int],
) -> list[FrameRecord]:
    start, end = map(int, segment)
    selected = [path for path in image_paths(image_dir) if start <= first_int(path) <= end]
    if not selected:
        raise RuntimeError(f"No central images for {sign}: {segment}")
    gt_ids = None if gt_sign_dir is None else evaluator_gt_ids(gt_sign_dir, segment)
    if gt_ids is not None and len(gt_ids) != len(selected):
        raise RuntimeError(
            f"{sign}: input central frames={len(selected)}, evaluator GT frames={len(gt_ids)}; "
            "resolve sampling/rate instead of truncating"
        )
    return [
        FrameRecord(
            sign=sign,
            sign_class=sign_class,
            source_path=str(path.resolve()),
            source_frame_id=first_int(path),
            sequence_index=index,
            evaluator_index=index,
            gt_frame_id=None if gt_ids is None else gt_ids[index],
        )
        for index, path in enumerate(selected)
    ]


def write_jsonl(records: Iterable[FrameRecord], path: Path) -> None:
    content = "\n".join(json.dumps(asdict(record), sort_keys=True) for record in records) + "\n"
    atomic_write_text(path, content)


def read_jsonl(path: Path) -> list[FrameRecord]:
    return [FrameRecord(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def prepare_manifests(
    image_root: Path,
    gt_root: Path,
    signs_file: Path,
    segments_file: Path,
    output: Path,
    expected_signs: int | None = None,
    expected_frames: int | None = None,
) -> dict[str, object]:
    signs = read_sign_file(signs_file)
    segments = json.loads(segments_file.read_text(encoding="utf-8"))
    if set(signs) != set(segments):
        raise RuntimeError(f"sign/segment mismatch: {sorted(set(signs) ^ set(segments))}")
    summaries = []
    total = 0
    for sign, sign_class in signs.items():
        records = build_sign_manifest(sign, sign_class, image_root / sign, gt_root / sign, tuple(segments[sign]))
        manifest_path = output / f"{sign}.jsonl"
        write_jsonl(records, manifest_path)
        total += len(records)
        summaries.append({"sign": sign, "class": sign_class, "frames": len(records), "sha256": sha256_file(manifest_path)})
    if expected_signs is not None and len(summaries) != expected_signs:
        raise RuntimeError(f"sign count {len(summaries)} != {expected_signs}")
    if expected_frames is not None and total != expected_frames:
        raise RuntimeError(f"frame count {total} != {expected_frames}")
    summary = {"schema_version": "signpccx.manifest-summary.v1", "signs": len(summaries), "frames": total, "items": summaries}
    atomic_write_json(output / "summary.json", summary)
    return summary

