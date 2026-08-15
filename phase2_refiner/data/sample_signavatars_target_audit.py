"""Freeze a deterministic, source-disjoint 100-clip SignAvatars audit sample."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.provenance import sha256_file


DIMENSIONS = ("signer", "hand_activity", "hand_size", "truncation", "motion")


def _side_motion(clip, joint_slice: slice) -> float:
    if len(clip.frame_names) < 2:
        return 0.0
    valid = clip.track_valid[1:, joint_slice] & clip.track_valid[:-1, joint_slice]
    displacement = np.linalg.norm(
        clip.keypoints_2d[1:, joint_slice] - clip.keypoints_2d[:-1, joint_slice],
        axis=-1,
    )
    return float(np.median(displacement[valid])) if valid.any() else 0.0


def _hand_size(clip) -> float:
    sizes = []
    for frame in range(len(clip.frame_names)):
        for joint_slice in (slice(21, 36), slice(36, 51)):
            valid = clip.track_valid[frame, joint_slice]
            points = clip.keypoints_2d[frame, joint_slice][valid]
            if len(points) >= 2:
                sizes.append(float(np.linalg.norm(points.max(0) - points.min(0))))
    return float(np.median(sizes)) if sizes else 0.0


def _record(path: Path, signer_map: dict[str, str]) -> dict[str, Any]:
    clip = load_cache_clip(path)
    metadata = json.loads(clip.metadata_json)
    source_clip = str(metadata.get("source_clip", clip.clip_id))
    source_group = str(metadata.get("source_group", ""))
    if not source_group:
        raise ValueError(f"Audit candidate lacks source_group: {clip.clip_id}")
    signer = str(signer_map.get(source_clip, "")).strip()
    if not signer:
        raise ValueError(f"Signer map lacks {source_clip}")
    left_motion = _side_motion(clip, slice(21, 36))
    right_motion = _side_motion(clip, slice(36, 51))
    left_active = left_motion >= 0.002
    right_active = right_motion >= 0.002
    if left_active and right_active:
        activity = "both"
    elif left_active:
        activity = "left"
    elif right_active:
        activity = "right"
    else:
        activity = "static"
    hand_in_frame = clip.in_frame[:, 21:51]
    return {
        "cache_path": str(path.resolve()),
        "clip_id": clip.clip_id,
        "source_clip": source_clip,
        "source_group": source_group,
        "signer": signer,
        "hand_activity": activity,
        "hand_size_value": _hand_size(clip),
        "truncation_value": float(1.0 - hand_in_frame.mean()),
        "motion_value": max(left_motion, right_motion),
    }


def _assign_tertiles(records: list[dict[str, Any]], value: str, label: str) -> None:
    values = np.asarray([record[value] for record in records], dtype=np.float64)
    low, high = np.quantile(values, (1 / 3, 2 / 3))
    for record in records:
        current = record[value]
        record[label] = "low" if current <= low else "high" if current > high else "mid"


def select_stratified(
    records: list[dict[str, Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    groups = {record["source_group"] for record in records}
    if len(groups) < sample_size:
        raise ValueError(
            f"Need {sample_size} source-disjoint groups, found only {len(groups)}"
        )
    rng = np.random.default_rng(seed)
    tie_break = rng.permutation(len(records))
    remaining = list(range(len(records)))
    counts: Counter[tuple[str, str]] = Counter()
    used_groups: set[str] = set()
    selected = []
    while len(selected) < sample_size:
        eligible = [
            index
            for index in remaining
            if records[index]["source_group"] not in used_groups
        ]
        if not eligible:
            raise RuntimeError(
                "Stratified sampler exhausted source-disjoint candidates"
            )

        def score(index: int) -> tuple[float, int]:
            record = records[index]
            balance = sum(
                1.0 / (1.0 + counts[(dimension, str(record[dimension]))])
                for dimension in DIMENSIONS
            )
            return balance, -int(tie_break[index])

        chosen = max(eligible, key=score)
        record = records[chosen]
        selected.append(record)
        used_groups.add(record["source_group"])
        remaining.remove(chosen)
        for dimension in DIMENSIONS:
            counts[(dimension, str(record[dimension]))] += 1
    return selected


def build_sample(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"Append-only audit manifest exists: {args.output}")
    signer_map = json.loads(args.signer_map.read_text(encoding="utf-8"))
    if not isinstance(signer_map, dict):
        raise ValueError("Signer map must be a JSON object keyed by source clip")
    records = [
        _record(path, signer_map) for path in _manifest_paths(args.manifest.resolve())
    ]
    _assign_tertiles(records, "hand_size_value", "hand_size")
    _assign_tertiles(records, "truncation_value", "truncation")
    _assign_tertiles(records, "motion_value", "motion")
    selected = select_stratified(records, args.sample_size, args.seed)
    distributions = {
        dimension: dict(
            sorted(Counter(str(row[dimension]) for row in selected).items())
        )
        for dimension in DIMENSIONS
    }
    payload = {
        "schema": "signavatars-target-audit-sample-v1",
        "seed": args.seed,
        "sample_size": len(selected),
        "source_group_disjoint": len({row["source_group"] for row in selected})
        == len(selected),
        "stratified_by": list(DIMENSIONS),
        "distributions": distributions,
        "clips": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**payload, "sample_manifest_sha256": sha256_file(args.output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--signer-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build_sample(parse_args()), indent=2, sort_keys=True))
