"""Fail-closed audit for exact-expert, independent-target T2 residual caches."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.geometry.rotations import axis_angle_to_matrix, geodesic_distance


INDEPENDENT_TARGET_TYPES = {"independent_gt", "independent_pseudo_target"}


def audit_manifest(
    manifest: Path, require_locked_initializer: bool = False
) -> tuple[dict, set[str]]:
    failure_counts: Counter[str] = Counter()
    failure_examples: dict[str, list[str]] = {}

    def fail(reason: str, clip_id: str) -> None:
        failure_counts[reason] += 1
        examples = failure_examples.setdefault(reason, [])
        if len(examples) < 5:
            examples.append(clip_id)
    clips = frames = nonzero_frames = 0
    initializer_experts: set[str] = set()
    target_providers: set[str] = set()
    source_groups: set[str] = set()
    for path in _manifest_paths(manifest):
        clip = load_cache_clip(path)
        clips += 1
        frames += len(clip.frame_names)
        metadata = json.loads(clip.metadata_json)
        target_type = str(metadata.get("target_type", ""))
        initializer = str(metadata.get("initializer_expert", ""))
        provider = str(metadata.get("target_provider", ""))
        source_group = str(metadata.get("source_group", ""))
        if target_type not in INDEPENDENT_TARGET_TYPES:
            fail("target_type_is_not_independent", clip.clip_id)
        if not initializer or not provider or initializer.lower() == provider.lower():
            fail("initializer_and_target_provider_not_distinct", clip.clip_id)
        if require_locked_initializer and not bool(
            metadata.get("initializer_matches_locked_lane_a1", False)
        ):
            fail("initializer_does_not_match_locked_lane_a1", clip.clip_id)
        if not source_group:
            fail("source_group_missing", clip.clip_id)
        if clip.target_axis_angle is None or clip.target_rotation_valid is None:
            fail("target_rotations_missing", clip.clip_id)
            continue
        valid = clip.target_rotation_valid
        raw_difference = np.max(
            np.abs(clip.init_axis_angle - clip.target_axis_angle), axis=-1
        )
        candidate_frames = ((raw_difference > 1e-7) & valid).any(axis=1)
        if candidate_frames.any():
            # The fast raw comparison removes the common identity-cache case;
            # SO(3) distance prevents alternate axis-angle encodings from being
            # misclassified as a real residual.
            delta = geodesic_distance(
                axis_angle_to_matrix(
                    np_to_torch(clip.init_axis_angle[candidate_frames])
                ),
                axis_angle_to_matrix(
                    np_to_torch(clip.target_axis_angle[candidate_frames])
                ),
            ).numpy()
            nonzero_frames += int(
                ((delta > 1e-6) & valid[candidate_frames]).any(axis=1).sum()
            )
        if initializer:
            initializer_experts.add(initializer)
        if provider:
            target_providers.add(provider)
        if source_group:
            source_groups.add(source_group)
    nonzero_fraction = nonzero_frames / frames if frames else 0.0
    if nonzero_fraction < 0.10:
        fail("measurable_real_residual_below_10pct", "aggregate")
    return (
        {
            "manifest": str(manifest.resolve()),
            "clips": clips,
            "frames": frames,
            "nonzero_residual_frames": nonzero_frames,
            "nonzero_residual_fraction": nonzero_fraction,
            "initializer_experts": sorted(initializer_experts),
            "target_providers": sorted(target_providers),
            "locked_initializer_required": require_locked_initializer,
            "source_groups": len(source_groups),
            "failure_counts": dict(sorted(failure_counts.items())),
            "failure_examples": failure_examples,
            "passed": not failure_counts,
        },
        source_groups,
    )


def np_to_torch(value: np.ndarray):
    import torch

    return torch.from_numpy(value).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument(
        "--require-locked-initializer",
        action="store_true",
        help="Require every initializer to match the exact frozen Lane-L A1 stack",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    train, train_groups = audit_manifest(
        args.train_manifest.resolve(), args.require_locked_initializer
    )
    validation, validation_groups = audit_manifest(
        args.val_manifest.resolve(), args.require_locked_initializer
    )
    calibration = calibration_groups = None
    if args.calibration_manifest:
        calibration, calibration_groups = audit_manifest(
            args.calibration_manifest.resolve(), args.require_locked_initializer
        )
    source_overlap = {
        "train_validation": sorted(train_groups & validation_groups),
        "train_calibration": (
            sorted(train_groups & calibration_groups)
            if calibration_groups is not None
            else []
        ),
        "validation_calibration": (
            sorted(validation_groups & calibration_groups)
            if calibration_groups is not None
            else []
        ),
    }
    report = {
        "train": train,
        "validation": validation,
        "calibration": calibration,
        "source_group_overlap": source_overlap,
        "split_disjoint_verified": not any(source_overlap.values()),
        "passed": train["passed"]
        and validation["passed"]
        and (calibration is None or calibration["passed"])
        and not any(source_overlap.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
