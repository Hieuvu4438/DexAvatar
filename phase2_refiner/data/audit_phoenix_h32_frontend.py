"""Validate complete, target-independent H32 payloads before cache building."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from phase2_refiner.provenance import sha256_file


SCHEMA = "signal4d-phoenix-h32-frontend-audit-v1"


def _array(payload: dict[str, Any], name: str) -> np.ndarray:
    if name not in payload:
        raise ValueError(f"H32 payload lacks {name}")
    value = payload[name]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def validate_payload(
    path: Path,
    clip: dict[str, Any],
    *,
    now: float,
    minimum_age_seconds: float,
) -> dict[str, Any]:
    """Validate one stable H32 artifact without opening any target field."""
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.stat()
    age = now - before.st_mtime
    if age < minimum_age_seconds:
        raise ValueError(f"H32 payload is not yet stable ({age:.3f}s): {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"H32 payload changed while being audited: {path}")
    if not isinstance(payload, dict):
        raise ValueError(f"H32 payload must be a mapping: {path}")
    indices = _array(payload, "total_valid_index").astype(np.int64).reshape(-1)
    n = len(indices)
    if n == 0 or len(set(indices.tolist())) != n:
        raise ValueError(f"Invalid H32 frame indices: {path}")
    frame_count = int(clip["source_contract"]["frame_count"])
    if int(indices.min()) < 0 or int(indices.max()) >= frame_count:
        raise ValueError(f"H32 frame index outside source video: {path}")
    expected_shapes = {
        "smplx": (n, 182),
        "unsmooth_smplx": (n, 169),
        "pred_2d": (n, 106, 2),
        "bb2img_trans": (n, 2, 3),
    }
    for field, expected in expected_shapes.items():
        value = _array(payload, field)
        if value.shape != expected:
            raise ValueError(
                f"H32 {field} shape {value.shape}, expected {expected}: {path}"
            )
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite H32 {field}: {path}")
    if int(payload.get("width", 0)) <= 0 or int(payload.get("height", 0)) <= 0:
        raise ValueError(f"Invalid H32 image size: {path}")
    return {
        "source_video_frames": frame_count,
        "h32_retained_frames": n,
        "sha256": sha256_file(path),
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
    }


def audit(
    selection_root: Path,
    h32_root: Path,
    splits: tuple[str, ...],
    minimum_age_seconds: float = 2.0,
) -> dict[str, Any]:
    selection_root = selection_root.resolve()
    h32_root = h32_root.resolve()
    now = time.time()
    seen = set()
    combined = hashlib.sha256()
    reports = {}
    for split in splits:
        selection_path = selection_root / split / "selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        clips = selection["clips"]
        retained_frames = source_frames = 0
        for clip in clips:
            name = str(clip["source_clip"])
            if name in seen:
                raise ValueError(f"H32 clip occurs in multiple audited splits: {name}")
            seen.add(name)
            path = h32_root / f"{name}.pkl"
            item = validate_payload(
                path,
                clip,
                now=now,
                minimum_age_seconds=minimum_age_seconds,
            )
            digest = item["sha256"]
            combined.update(name.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
            retained_frames += int(item["h32_retained_frames"])
            source_frames += int(item["source_video_frames"])
        reports[split] = {
            "clips": len(clips),
            "source_video_frames": source_frames,
            "h32_retained_frames": retained_frames,
            "h32_frame_coverage": retained_frames / max(source_frames, 1),
            "selection": str(selection_path),
            "selection_sha256": sha256_file(selection_path),
        }
    return {
        "schema": SCHEMA,
        "target_fields_opened": False,
        "splits": reports,
        "clips": len(seen),
        "h32_content_set_sha256": combined.hexdigest(),
        "minimum_age_seconds": minimum_age_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--h32-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--minimum-age-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit(
        args.selection_root,
        args.h32_root,
        tuple(args.splits),
        args.minimum_age_seconds,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
