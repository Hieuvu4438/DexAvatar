"""Fail-closed data-lineage checks for the no-SGNify training lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.provenance import sha256_file


ALLOWED_DATASETS = {
    "How2Sign",
    "ARCTIC",
    "InterHand2.6M",
    "SignAvatars",
    "SOKE",
    "SOKE+SignAvatars",
}
FORBIDDEN_PATH_PARTS = {
    "sgnify",
    "smplx_gt",
    "evaluation_from_author",
    "signal4d_v5",
    "v6_final",
}


def _manifest_clip_paths(manifest_path: Path, payload: dict[str, Any]) -> list[Path]:
    entries = payload.get("clips")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Manifest has no clips: {manifest_path}")
    paths = []
    for entry in entries:
        path = Path(entry)
        if not path.is_absolute():
            path = manifest_path.parent / path
        paths.append(path.resolve())
    return paths


def _forbidden_path_hit(path: Path) -> str | None:
    lowered = {part.lower() for part in path.parts}
    for forbidden in FORBIDDEN_PATH_PARTS:
        if forbidden in lowered:
            return forbidden
    return None


def audit_manifest(path: str | Path, *, scan_clips: bool = True) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    dataset = payload.get("dataset")
    if dataset not in ALLOWED_DATASETS:
        raise ValueError(f"Training dataset is not allow-listed: {dataset!r}")
    if payload.get("sgnify_excluded") is not True:
        raise ValueError(f"Manifest must declare sgnify_excluded=true: {manifest_path}")
    hit = _forbidden_path_hit(manifest_path)
    if hit:
        raise ValueError(f"Forbidden training path component {hit!r}: {manifest_path}")
    clip_paths = _manifest_clip_paths(manifest_path, payload)
    missing = [clip for clip in clip_paths if not clip.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing training clip: {missing[0]}")

    frames = 0
    datasets: set[str] = set()
    targetless: list[str] = []
    official_splits: set[str] = set()
    if scan_clips:
        for clip_path in clip_paths:
            hit = _forbidden_path_hit(clip_path)
            if hit:
                raise ValueError(
                    f"Forbidden training path component {hit!r}: {clip_path}"
                )
            clip = load_cache_clip(clip_path)
            metadata = json.loads(clip.metadata_json)
            clip_dataset = metadata.get("dataset")
            if clip_dataset not in ALLOWED_DATASETS:
                raise ValueError(
                    f"Clip dataset is not allow-listed: {clip_dataset!r} ({clip_path})"
                )
            if int(metadata.get("sgnify_training_reads", 0)) != 0:
                raise ValueError(f"Clip reports SGNify training reads: {clip_path}")
            if clip.target_axis_angle is None:
                targetless.append(str(clip_path))
            official_split = metadata.get("official_split")
            if official_split is not None:
                official_splits.add(str(official_split))
            frames += len(clip.frame_names)
            datasets.add(str(clip_dataset))
    if targetless:
        raise ValueError(f"Training clip has no target: {targetless[0]}")
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "declared_dataset": dataset,
        "clip_datasets": sorted(datasets),
        "clips": len(clip_paths),
        "frames": frames,
        "sgnify_reads": 0,
        "official_splits": sorted(official_splits),
        "source_groups": sorted(payload.get("source_groups", [])),
    }


def audit_protocol(
    train_manifest: str | Path,
    validation_manifest: str | Path,
    calibration_manifest: str | Path | None = None,
    test_manifest: str | Path | None = None,
    *,
    scan_clips: bool = True,
    allow_official_test_source_overlap: bool = False,
) -> dict[str, Any]:
    reports = {
        "train": audit_manifest(train_manifest, scan_clips=scan_clips),
        "validation": audit_manifest(validation_manifest, scan_clips=scan_clips),
    }
    if calibration_manifest is not None:
        reports["calibration"] = audit_manifest(
            calibration_manifest, scan_clips=scan_clips
        )
    if test_manifest is not None:
        reports["test"] = audit_manifest(test_manifest, scan_clips=scan_clips)
        if scan_clips and reports["test"]["official_splits"] != ["test"]:
            raise ValueError(
                "The official-test overlap policy requires only official test clips"
            )
    group_sets = {
        name: set(report.pop("source_groups")) for name, report in reports.items()
    }
    overlaps = {}
    allowed_test_overlaps = {}
    names = list(group_sets)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = sorted(group_sets[first] & group_sets[second])
            overlaps[f"{first}__{second}"] = overlap
            if overlap:
                is_test_pair = "test" in {first, second}
                if allow_official_test_source_overlap and is_test_pair:
                    allowed_test_overlaps[f"{first}__{second}"] = overlap
                else:
                    raise ValueError(
                        f"Source-group leakage between {first} and {second}: "
                        f"{overlap[:3]}"
                    )
    decision = (
        "PASS_WITH_REPORTED_OFFICIAL_TEST_SOURCE_OVERLAP"
        if allowed_test_overlaps
        else "PASS"
    )
    return {
        "schema_version": 1,
        "decision": decision,
        "allowed_datasets": sorted(ALLOWED_DATASETS),
        "manifests": reports,
        "source_group_overlaps": overlaps,
        "allowed_official_test_source_overlaps": allowed_test_overlaps,
        "test_protocol": (
            "official-held-out-clips-not-signer-disjoint"
            if allowed_test_overlaps
            else "signer-disjoint"
        ),
        "sgnify_training_or_selection_reads": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--test", type=Path)
    parser.add_argument(
        "--allow-official-test-source-overlap",
        action="store_true",
        help=(
            "Permit and report source overlap involving an official test split; "
            "development splits remain strictly source-disjoint"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    report = audit_protocol(
        args.train,
        args.validation,
        args.calibration,
        args.test,
        scan_clips=not args.manifest_only,
        allow_official_test_source_overlap=args.allow_official_test_source_overlap,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
