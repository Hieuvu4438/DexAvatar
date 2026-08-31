"""Describe temporal spikes in released PHOENIX/SOKE pose targets.

This audit is deliberately descriptive.  A large local-rotation excursion can
be annotation jitter or a valid fast sign transition; without an independent
3D reference it must not be converted automatically into a supervision mask.
Only consecutive original video frames form a triplet, so missing fitted
frames are never treated as adjacent observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phase2_refiner.data.build_phoenix_soke_full_cache import _target_pose
from phase2_refiner.geometry.rotations import axis_angle_to_matrix, geodesic_distance
from phase2_refiner.provenance import sha256_file


SCHEMA = "signal4d-phoenix-soke-target-temporal-audit-v1"
REGIONS = {"body": (0, 21), "left_hand": (21, 36), "right_hand": (36, 51)}
PERCENTILES = (50.0, 90.0, 95.0, 99.0, 99.5, 99.9, 100.0)
THRESHOLDS_DEGREES = (10.0, 20.0, 30.0, 45.0, 60.0, 90.0)


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "percentiles_degrees": {}, "above_threshold": {}}
    percentiles = np.percentile(values, PERCENTILES)
    return {
        "count": int(len(values)),
        "percentiles_degrees": {
            f"p{value:g}": float(result)
            for value, result in zip(PERCENTILES, percentiles)
        },
        "above_threshold": {
            f"gt_{threshold:g}_degrees": {
                "count": int((values > threshold).sum()),
                "fraction": float((values > threshold).mean()),
            }
            for threshold in THRESHOLDS_DEGREES
        },
    }


def audit(selection_path: Path, sample_clips: int) -> dict[str, Any]:
    selection_path = selection_path.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    clips = selection.get("clips", [])
    if not isinstance(clips, list) or not clips:
        raise ValueError(f"Selection has no clips: {selection_path}")
    if sample_clips <= 0:
        raise ValueError("sample_clips must be positive")
    count = min(sample_clips, len(clips))
    indices = np.unique(np.linspace(0, len(clips) - 1, count, dtype=np.int64))
    excess_by_region: dict[str, list[np.ndarray]] = {name: [] for name in REGIONS}
    step_by_region: dict[str, list[np.ndarray]] = {name: [] for name in REGIONS}
    frames_read = 0
    consecutive_triplets = 0
    for index in indices:
        clip = clips[int(index)]
        frame_numbers = np.asarray(
            clip["target_frame_indices_one_based"], dtype=np.int64
        )
        paths = [
            Path(clip["target_dir"]) / f"images{int(frame):04d}.pkl"
            for frame in frame_numbers
        ]
        pose, valid = _target_pose(paths)
        frames_read += len(frame_numbers)
        if len(frame_numbers) < 3:
            continue
        matrix = axis_angle_to_matrix(torch.from_numpy(pose).float())
        previous = geodesic_distance(matrix[:-2], matrix[1:-1]).numpy()
        following = geodesic_distance(matrix[1:-1], matrix[2:]).numpy()
        across = geodesic_distance(matrix[:-2], matrix[2:]).numpy()
        consecutive = (
            (np.diff(frame_numbers[:-1]) == 1)
            & (np.diff(frame_numbers[1:]) == 1)
        )
        consecutive_triplets += int(consecutive.sum())
        triplet_valid = (
            valid[:-2] & valid[1:-1] & valid[2:] & consecutive[:, None]
        )
        # Triangle excess is near zero for motion along one local geodesic and
        # grows for an isolated out-and-back excursion at the middle frame.
        excess = np.maximum(0.0, 0.5 * (previous + following - across))
        maximum_step = np.maximum(previous, following)
        excess = np.rad2deg(excess)
        maximum_step = np.rad2deg(maximum_step)
        for name, (start, stop) in REGIONS.items():
            mask = triplet_valid[:, start:stop]
            excess_by_region[name].append(excess[:, start:stop][mask])
            step_by_region[name].append(maximum_step[:, start:stop][mask])
    regions = {}
    for name in REGIONS:
        excess = (
            np.concatenate(excess_by_region[name])
            if excess_by_region[name]
            else np.empty(0, dtype=np.float64)
        )
        step = (
            np.concatenate(step_by_region[name])
            if step_by_region[name]
            else np.empty(0, dtype=np.float64)
        )
        regions[name] = {
            "triangle_excess": _summary(excess),
            "maximum_adjacent_step": _summary(step),
        }
    return {
        "schema": SCHEMA,
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "official_split": selection.get("official_split"),
        "declared_clips": len(clips),
        "sampled_clips": len(indices),
        "sampling": "deterministic uniform indices over released loader order",
        "sample_indices": indices.tolist(),
        "frames_read": frames_read,
        "consecutive_frame_triplets": consecutive_triplets,
        "metric": (
            "0.5 * (d(R[t-1],R[t]) + d(R[t],R[t+1]) - "
            "d(R[t-1],R[t+1])) in degrees"
        ),
        "regions": regions,
        "automatic_quality_mask_recommended": False,
        "interpretation": (
            "Descriptive tail statistics cannot distinguish fitting jitter from "
            "valid rapid sign transitions without an independent 3D reference. "
            "Do not smooth or mask released targets from this audit alone."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--sample-clips", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit(args.selection, args.sample_clips)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
