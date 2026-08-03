"""Automatic pseudo-target quality bands without benchmark supervision."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from phase2_refiner.data.cache_schema import load_cache_clip
from phase3_posterior.provenance import atomic_json, sha256_file


def assess_clip(path: str | Path) -> dict:
    source = Path(path).resolve()
    clip = load_cache_clip(source)
    target = clip.target_axis_angle
    target_missing = target is None
    rotation_max = (
        0.0 if target_missing else float(np.linalg.norm(target, axis=-1).max())
    )
    shape_std = 0.0  # Phase 2 caches already enforce one shared beta vector.
    reliability = float(np.mean(clip.u0_reliability))
    reprojection = float(np.linalg.norm(clip.reprojection_residual_2d, axis=-1).mean())
    palm_jump = 0.0
    if len(clip.palm_normals) > 1:
        palm_jump = float(
            np.linalg.norm(np.diff(clip.palm_normals, axis=0), axis=-1).max()
        )
    catastrophic = bool(
        target_missing
        or rotation_max > 4.0 * np.pi
        or not np.isfinite(reprojection)
        or not np.isfinite(palm_jump)
    )
    score = reliability - min(1.0, reprojection) * 0.25 - min(1.0, palm_jump) * 0.25
    if catastrophic:
        band = "Q2"
    elif score >= 0.55:
        band = "Q0"
    elif score >= 0.25:
        band = "Q1"
    else:
        band = "Q2"
    return {
        "clip_id": clip.clip_id,
        "clip_path": str(source),
        "clip_sha256": sha256_file(source),
        "band": band,
        "catastrophic": catastrophic,
        "metrics": {
            "mean_u0_reliability": reliability,
            "mean_reprojection_residual": reprojection,
            "maximum_palm_jump": palm_jump,
            "maximum_target_rotation": rotation_max,
            "shape_std": shape_std,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output, assess_clip(args.clip))


if __name__ == "__main__":
    main()
