"""Evaluate the frozen Lane-L hard, clean, and safety subsets for G6."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.provenance import sha256_file


REGIONS = {
    "ubody": (0, 21, 0),
    "lhand": (21, 36, 1),
    "rhand": (36, 51, 2),
}
SUBSET_CONTRACT = {
    "version": "lane_l_observation_difficulty_v2_active_refine_mask",
    "population": "only joints enabled by the immutable cache refine_mask",
    "hard": (
        "mean U0 reliability <0.35 OR missing fraction >0.25 OR "
        "truncation >0.25 OR duplicate/disagreement flag"
    ),
    "clean": (
        "mean U0 reliability >=0.75 AND missing fraction <=0.05 AND "
        "truncation <0.10 AND no duplicate/disagreement flag"
    ),
}


def _relative_gain(rows: list[dict], region: str) -> tuple[float | None, int]:
    pairs = [
        (float(row[f"prediction_{region}"]), float(row[f"baseline_{region}"]))
        for row in rows
        if row[f"prediction_{region}"] != "" and row[f"baseline_{region}"] != ""
    ]
    if not pairs:
        return None, 0
    prediction = float(np.mean([pair[0] for pair in pairs]))
    baseline = float(np.mean([pair[1] for pair in pairs]))
    return (baseline - prediction) / baseline, len(pairs)


def evaluate_diagnostics(
    per_frame: Path, cache_root: Path, prediction_root: Path
) -> dict:
    with per_frame.open("r", encoding="utf-8") as handle:
        frame_rows = list(csv.DictReader(handle))
    by_id = {(row["sign"], row["frame"]): row for row in frame_rows}
    classifications: dict[tuple[str, str, str], tuple[bool, bool]] = {}
    fallback_count = group_frames = 0
    diagnostic_frames = 0
    t5_clips = 0
    t5_accepted_clips: Counter[str] = Counter()
    fallback_clips: Counter[str] = Counter()
    for cache_path in sorted((cache_root / "clips").glob("*.npz")):
        clip = load_cache_clip(cache_path)
        diagnostic_path = (
            prediction_root / clip.clip_id / "phase2_diagnostics" / "sequence.npz"
        )
        if not diagnostic_path.is_file():
            raise FileNotFoundError(diagnostic_path)
        with np.load(diagnostic_path, allow_pickle=False) as data:
            diagnostic_names = data["frame_names"].astype(str)
            fallback = data["fallback_mask"].astype(bool)
            t5_accepted = (
                data["t5_accepted_regions"].astype(bool)
                if "t5_accepted_regions" in data.files
                else None
            )
        if not np.array_equal(diagnostic_names, clip.frame_names.astype(str)):
            raise ValueError(f"Diagnostic/cache frame mismatch for {clip.clip_id}")
        fallback_count += int(fallback.sum())
        group_frames += int(fallback.size)
        diagnostic_frames += len(clip.frame_names)
        for region, (_, _, group_index) in REGIONS.items():
            fallback_clips[region] += int(fallback[:, group_index].any())
        if t5_accepted is not None:
            if t5_accepted.shape != (len(REGIONS),):
                raise ValueError(
                    f"Invalid T5 acceptance shape for {clip.clip_id}: "
                    f"{t5_accepted.shape}"
                )
            t5_clips += 1
            for accepted, region in zip(t5_accepted, REGIONS, strict=True):
                t5_accepted_clips[region] += int(accepted)
        observations = clip.observation_features
        for index, frame in enumerate(clip.frame_names.astype(str)):
            if (clip.clip_id, frame) not in by_id:
                raise ValueError(f"Evaluator row missing for {clip.clip_id}/{frame}")
            for region, (start, end, _) in REGIONS.items():
                active = clip.refine_mask[start:end]
                if not active.any():
                    raise ValueError(
                        f"No active refined joints for {clip.clip_id}/{region}"
                    )
                reliability = clip.u0_reliability[index, start:end][active]
                group = observations[index, start:end][active]
                mean_reliability = float(reliability.mean())
                missing_fraction = float(group[:, 2].mean())
                max_truncation = float(group[:, 4].max())
                flagged = bool((group[:, 6:8] > 0).any())
                hard = (
                    mean_reliability < 0.35
                    or missing_fraction > 0.25
                    or max_truncation > 0.25
                    or flagged
                )
                clean = (
                    mean_reliability >= 0.75
                    and missing_fraction <= 0.05
                    and max_truncation < 0.10
                    and not flagged
                )
                classifications[(clip.clip_id, frame, region)] = (hard, clean)
    if diagnostic_frames != len(frame_rows):
        raise ValueError(
            f"Coverage mismatch: {diagnostic_frames} diagnostics vs {len(frame_rows)} evaluator rows"
        )

    hard_gains = {}
    clean_regressions = {}
    counts = {"hard": {}, "clean": {}}
    for region in REGIONS:
        hard_rows = [
            row
            for row in frame_rows
            if classifications[(row["sign"], row["frame"], region)][0]
        ]
        clean_rows = [
            row
            for row in frame_rows
            if classifications[(row["sign"], row["frame"], region)][1]
        ]
        hard_gain, hard_count = _relative_gain(hard_rows, region)
        clean_gain, clean_count = _relative_gain(clean_rows, region)
        hard_gains[region] = hard_gain
        clean_regressions[region] = None if clean_gain is None else -clean_gain
        counts["hard"][region] = hard_count
        counts["clean"][region] = clean_count
    valid_hard = [gain for gain in hard_gains.values() if gain is not None]
    return {
        "subset_contract": SUBSET_CONTRACT,
        "per_frame": str(per_frame.resolve()),
        "per_frame_sha256": sha256_file(per_frame),
        "cache_manifest": str((cache_root / "manifest.json").resolve()),
        "cache_manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "frames": len(frame_rows),
        "counts": counts,
        "hard_subset_regional_relative_gain": hard_gains,
        "hard_subset_relative_gain": (
            float(np.mean(valid_hard)) if valid_hard else None
        ),
        "hard_subset_regions_present": [
            region for region, gain in hard_gains.items() if gain is not None
        ],
        "clean_regression_fraction": clean_regressions,
        "fallback_group_frames": fallback_count,
        "total_group_frames": group_frames,
        "group_frame_fallback_fraction": fallback_count / group_frames,
        "fallback_clip_count_by_region": {
            region: fallback_clips[region] for region in REGIONS
        },
        "t5_clips_attempted": t5_clips,
        "t5_accepted_clip_count": {
            region: t5_accepted_clips[region] for region in REGIONS
        },
        "t5_accepted_clip_fraction": {
            region: t5_accepted_clips[region] / max(t5_clips, 1)
            for region in REGIONS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-frame", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    report = evaluate_diagnostics(
        args.per_frame.resolve(), args.cache_root.resolve(), args.prediction_root.resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
