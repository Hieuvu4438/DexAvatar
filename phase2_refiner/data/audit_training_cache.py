"""Audit Phase 2 cache integrity, leakage exclusions, and G2 readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from phase2_refiner.data.cache_schema import load_cache_clip


FORBIDDEN_TARGET_MARKERS = ("smplx_gt", "evaluation_from_author")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_paths(path: Path) -> list[Path]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("clips", payload)
    if not isinstance(entries, list):
        raise ValueError(f"Invalid manifest: {path}")
    return [
        (path.parent / entry).resolve()
        if not Path(entry).is_absolute()
        else Path(entry)
        for entry in entries
    ]


def _audit_split(
    manifest: Path, expected_split: str
) -> tuple[dict, set[str], set[str]]:
    paths = _manifest_paths(manifest)
    clip_ids: set[str] = set()
    frames = complete_frames = 0
    region_targets = {"body": 0, "left_hand": 0, "right_hand": 0}
    lengths = []
    leakage = []
    split_mismatch = []
    motion_domains: dict[str, dict[str, int]] = {}
    source_groups: set[str] = set()
    target_types: dict[str, int] = {}
    quality_screened = quality_failed = 0
    maximum_catastrophic_fraction = 0.0
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        clip = load_cache_clip(path)
        if clip.clip_id in clip_ids:
            raise ValueError(f"Duplicate clip_id in {manifest}: {clip.clip_id}")
        clip_ids.add(clip.clip_id)
        length = len(clip.frame_names)
        lengths.append(length)
        frames += length
        valid = clip.target_rotation_valid
        if clip.target_axis_angle is None or valid is None:
            raise ValueError(f"Training clip has no rotation target: {path}")
        region_targets["body"] += int(valid[:, :21].all(axis=1).sum())
        region_targets["left_hand"] += int(valid[:, 21:36].all(axis=1).sum())
        region_targets["right_hand"] += int(valid[:, 36:51].all(axis=1).sum())
        complete_frames += int(valid.all(axis=1).sum())
        metadata = json.loads(clip.metadata_json)
        motion_domain = str(metadata.get("motion_domain", "unspecified")).lower()
        domain = motion_domains.setdefault(motion_domain, {"clips": 0, "frames": 0})
        domain["clips"] += 1
        domain["frames"] += length
        source_group = metadata.get("source_group")
        if source_group is not None:
            source_groups.add(str(source_group))
        target_type = str(metadata.get("target_type", "unspecified"))
        target_types[target_type] = target_types.get(target_type, 0) + 1
        quality = metadata.get("quality")
        if isinstance(quality, dict):
            quality_screened += 1
            fraction = float(quality.get("catastrophic_frame_fraction", 1.0))
            maximum_catastrophic_fraction = max(maximum_catastrophic_fraction, fraction)
            quality_failed += int(not bool(quality.get("passed", False)))
        metadata_without_assertions = dict(metadata)
        metadata_without_assertions.pop("forbidden_training_sources", None)
        searchable = (
            " ".join(clip.source_paths.astype(str))
            + " "
            + json.dumps(metadata_without_assertions, sort_keys=True)
        )
        for marker in FORBIDDEN_TARGET_MARKERS:
            if marker.lower() in searchable.lower():
                leakage.append({"clip": clip.clip_id, "marker": marker})
        official = metadata.get("official_split")
        if official is not None and official != expected_split:
            split_mismatch.append(clip.clip_id)
    clips = len(paths)
    return (
        {
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "clips": clips,
            "frames": frames,
            "minimum_length": min(lengths) if lengths else 0,
            "maximum_length": max(lengths) if lengths else 0,
            "clips_at_least_16": sum(length >= 16 for length in lengths),
            "fraction_clips_at_least_16": (
                sum(length >= 16 for length in lengths) / clips if clips else 0.0
            ),
            "region_complete_target_frames": region_targets,
            "complete_body_and_both_hand_frames": complete_frames,
            "complete_body_and_both_hand_fraction": (
                complete_frames / frames if frames else 0.0
            ),
            "forbidden_source_hits": leakage,
            "official_split_mismatches": split_mismatch,
            "motion_domains": motion_domains,
            "source_groups": len(source_groups),
            "target_types": target_types,
            "target_quality_screen": {
                "clips_screened": quality_screened,
                "clips_failed": quality_failed,
                "failure_fraction": (
                    quality_failed / quality_screened if quality_screened else None
                ),
                "maximum_catastrophic_frame_fraction": (
                    maximum_catastrophic_fraction if quality_screened else None
                ),
            },
        },
        clip_ids,
        source_groups,
    )


def audit(train_manifest: Path, val_manifest: Path) -> dict:
    train, train_ids, train_groups = _audit_split(train_manifest, "train")
    val, val_ids, val_groups = _audit_split(val_manifest, "val")
    overlap = sorted(train_ids & val_ids)
    source_group_overlap = sorted(train_groups & val_groups)
    integrity_go = not (
        train["forbidden_source_hits"]
        or val["forbidden_source_hits"]
        or train["official_split_mismatches"]
        or val["official_split_mismatches"]
        or overlap
        or source_group_overlap
    )
    data_volume_go = train["clips"] >= 10000 or train["frames"] >= 250000
    sign_domains = {
        name: counts
        for name, counts in train["motion_domains"].items()
        if "sign" in name and not name.startswith("generic")
    }
    sign_clips = sum(counts["clips"] for counts in sign_domains.values())
    sign_frames = sum(counts["frames"] for counts in sign_domains.values())
    sign_domain_volume_go = sign_clips >= 10000 or sign_frames >= 250000
    length_go = train["fraction_clips_at_least_16"] >= 0.8
    completeness_go = train["complete_body_and_both_hand_fraction"] >= 0.7
    has_pseudo_targets = any("pseudo" in name.lower() for name in train["target_types"])
    quality = train["target_quality_screen"]
    target_quality_go = not has_pseudo_targets or (
        quality["clips_screened"] >= 100
        and quality["failure_fraction"] is not None
        and quality["failure_fraction"] < 0.10
    )
    main_training_go = (
        integrity_go
        and data_volume_go
        and sign_domain_volume_go
        and length_go
        and completeness_go
        and target_quality_go
    )
    return {
        "train": train,
        "validation": val,
        "train_validation_clip_overlap": overlap,
        "train_validation_source_group_overlap": source_group_overlap,
        "gates": {
            "integrity_and_leakage_exclusion": integrity_go,
            "data_volume": data_volume_go,
            "sign_domain_data_volume": sign_domain_volume_go,
            "length_distribution": length_go,
            "complete_body_and_hands": completeness_go,
            "pseudo_target_quality_screen": target_quality_go,
            "G2_main_training": main_training_go,
            "T1_hand_feasibility": integrity_go
            and train["region_complete_target_frames"]["left_hand"] > 0
            and train["region_complete_target_frames"]["right_hand"] > 0,
        },
        "sign_domain_training_volume": {
            "domains": sign_domains,
            "clips": sign_clips,
            "frames": sign_frames,
            "threshold": "at least 10,000 clips or 250,000 frames",
        },
        "decision": (
            "GO: full Phase 2 training"
            if main_training_go
            else "NO-GO: main Phase 2; T1 synthetic hand-feasibility only"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-main-go", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args.train_manifest.resolve(), args.val_manifest.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["gates"]["integrity_and_leakage_exclusion"]:
        raise SystemExit(2)
    if args.require_main_go and not report["gates"]["G2_main_training"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
