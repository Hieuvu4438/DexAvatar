"""Freeze a rank-calibrated external-only hand policy on How2Sign."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phase2_refiner.geometry.rotations import geodesic_distance
from phase2_refiner.provenance import sha256_file
from signal4d_external.hand_v2_core import (
    ALPHA_GRID,
    COVERAGE_GRID,
    HAND_REGIONS,
    SMOOTHING_HALF_WINDOW_SECONDS_GRID,
    exact_rank_selection,
    geodesic_blend,
    smooth_clips,
)


def _load_index(root: Path) -> dict[str, Any]:
    index_path = root / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("method") != "SIGNAL4D_EXTERNAL_HAND_V2_PREDICTION_CACHE":
        raise ValueError(f"Unexpected prediction cache: {index_path}")
    if int(payload.get("sgnify_training_or_selection_reads", -1)) != 0:
        raise ValueError("Prediction cache does not prove zero SGNify reads")
    return payload


def _load_region_clips(
    index: dict[str, Any], split: str, region: str
) -> list[dict[str, Any]]:
    start, end, group_index = HAND_REGIONS[region]
    clips = []
    for row in index["clips"]:
        if row["split"] != split:
            continue
        path = Path(row["output"])
        if sha256_file(path) != row["output_sha256"]:
            raise ValueError(f"Prediction-cache hash mismatch: {path}")
        with np.load(path) as item:
            initial = torch.from_numpy(item["initial_matrix"][:, start:end]).float()
            candidate = torch.from_numpy(item["candidate_matrix"][:, start:end]).float()
            target = torch.from_numpy(item["target_matrix"][:, start:end]).float()
            valid = np.asarray(item["target_valid"][:, start:end], dtype=bool)
            scores = np.asarray(
                item["benefit_probability"][:, group_index], dtype=np.float64
            )
            timestamps = np.asarray(item["timestamps"], dtype=np.float64)
            eligible = np.asarray(item[f"eligible_{region}"], dtype=bool)
        valid_tensor = torch.from_numpy(valid)
        denominator = valid_tensor.sum(dim=-1)
        valid_frame = denominator > 0
        baseline_joint = geodesic_distance(initial, target)
        baseline_frame = (
            baseline_joint * valid_tensor
        ).sum(dim=-1) / denominator.clamp_min(1)
        candidates = {}
        for alpha in ALPHA_GRID:
            blended = geodesic_blend(initial, candidate, alpha)
            error = geodesic_distance(blended, target)
            candidates[float(alpha)] = (
                (error * valid_tensor).sum(dim=-1) / denominator.clamp_min(1)
            ).numpy()
        clips.append(
            {
                "clip_id": row["clip_id"],
                "signer": row["signer"],
                "scores": scores,
                "timestamps": timestamps,
                "eligible": eligible,
                "valid": valid_frame.numpy(),
                "baseline": baseline_frame.numpy(),
                "candidates": candidates,
            }
        )
    if not clips:
        raise ValueError(f"No {split}/{region} clips in prediction cache")
    return clips


def _evaluate(
    clips: list[dict[str, Any]],
    alpha: float,
    coverage: float,
    half_window_seconds: float,
) -> dict[str, Any]:
    scores = smooth_clips(
        [clip["scores"] for clip in clips],
        [clip["timestamps"] for clip in clips],
        half_window_seconds,
        [clip["eligible"] for clip in clips],
    )
    selected = exact_rank_selection(
        scores, coverage, [clip["eligible"] for clip in clips]
    )
    baseline_clip = []
    hybrid_clip = []
    signer_baseline: dict[str, list[float]] = defaultdict(list)
    signer_hybrid: dict[str, list[float]] = defaultdict(list)
    valid_frames = 0
    selected_valid_frames = 0
    for clip, chosen in zip(clips, selected, strict=True):
        valid = clip["valid"] & clip["eligible"]
        if not valid.any():
            continue
        baseline = float(np.mean(clip["baseline"][valid]))
        hybrid_frame = np.where(
            chosen,
            clip["candidates"][float(alpha)],
            clip["baseline"],
        )
        hybrid = float(np.mean(hybrid_frame[valid]))
        baseline_clip.append(baseline)
        hybrid_clip.append(hybrid)
        signer_baseline[clip["signer"]].append(baseline)
        signer_hybrid[clip["signer"]].append(hybrid)
        valid_frames += int(valid.sum())
        selected_valid_frames += int((chosen & valid).sum())
    if not baseline_clip:
        raise ValueError("Policy has no externally supervised frames")
    baseline_deg = float(np.rad2deg(np.mean(baseline_clip)))
    hybrid_deg = float(np.rad2deg(np.mean(hybrid_clip)))
    signer_gain = {
        signer: float(
            np.rad2deg(
                np.mean(signer_baseline[signer]) - np.mean(signer_hybrid[signer])
            )
        )
        for signer in sorted(signer_baseline)
    }
    return {
        "alpha": float(alpha),
        "coverage": float(coverage),
        "smoothing_half_window_seconds": float(half_window_seconds),
        "clips": len(baseline_clip),
        "valid_frames": valid_frames,
        "selected_valid_frames": selected_valid_frames,
        "selection_fraction_valid": selected_valid_frames / valid_frames,
        "baseline_error_deg": baseline_deg,
        "hybrid_error_deg": hybrid_deg,
        "gain_deg": baseline_deg - hybrid_deg,
        "signer_gain_deg": signer_gain,
        "worst_signer_gain_deg": min(signer_gain.values()),
    }


def _select_on_validation(clips: list[dict[str, Any]]) -> tuple[dict[str, Any], list]:
    grid = [
        _evaluate(clips, alpha, coverage, half_window_seconds)
        for alpha in ALPHA_GRID
        for coverage in COVERAGE_GRID
        for half_window_seconds in SMOOTHING_HALF_WINDOW_SECONDS_GRID
    ]
    # Conservative tie break: less coverage, lower alpha and less smoothing.
    selected = max(
        grid,
        key=lambda row: (
            row["gain_deg"],
            -row["coverage"],
            -row["alpha"],
            -row["smoothing_half_window_seconds"],
        ),
    )
    return selected, grid


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.prediction_cache.resolve()
    index = _load_index(root)
    regions = {}
    passes = {}
    for region in HAND_REGIONS:
        validation_clips = _load_region_clips(index, "validation", region)
        selected, grid = _select_on_validation(validation_clips)
        calibration_clips = _load_region_clips(index, "calibration", region)
        calibration = _evaluate(
            calibration_clips,
            selected["alpha"],
            selected["coverage"],
            selected["smoothing_half_window_seconds"],
        )
        checks = {
            "positive_mean_gain": calibration["gain_deg"] > 0.0,
            "selection_fraction_at_least_0.10": calibration[
                "selection_fraction_valid"
            ]
            >= 0.10 - 1e-9,
            "worst_signer_gain_at_least_minus_0.25_deg": calibration[
                "worst_signer_gain_deg"
            ]
            >= -0.25,
        }
        passes[region] = all(checks.values())
        regions[region] = {
            "selected_on_validation": selected,
            "validation_grid": grid,
            "fixed_policy_on_calibration": calibration,
            "calibration_checks": checks,
            "decision": "PASS" if passes[region] else "FAIL",
        }
    result = {
        "schema_version": 1,
        "method": "SIGNAL4D_EXTERNAL_HAND_V2_RANK_CALIBRATION",
        "decision": "PASS" if all(passes.values()) else "FAIL",
        "decision_rule": (
            "Both hands must gain on signer-held-out How2Sign calibration, select "
            "at least 10% of valid frames, and regress no calibration signer by "
            "more than 0.25 degrees"
        ),
        "selection_protocol": (
            "alpha, global coverage, and time-domain score smoothing are selected "
            "on signer-disjoint How2Sign validation; calibration is gate-only"
        ),
        "target_inference_protocol": (
            "apply the frozen alpha and exact global coverage to ranks computed "
            "from unlabeled target benefit probabilities"
        ),
        "prediction_cache": str(root),
        "prediction_index_sha256": sha256_file(root / "index.json"),
        "config": index["config"],
        "config_sha256": index["config_sha256"],
        "checkpoint": index["checkpoint"],
        "checkpoint_sha256": index["checkpoint_sha256"],
        "manifests": index["manifests"],
        "regions": regions,
        "sgnify_training_or_selection_reads": 0,
        "unlabeled_target_covariates_used_at_inference": [
            "benefit_probability_rank"
        ],
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
