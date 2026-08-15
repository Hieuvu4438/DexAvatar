"""Create append-only Phase 2 manifests with signer- and source-disjoint splits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

from phase2_refiner.data.audit_training_cache import _manifest_paths
from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.provenance import sha256_file


SPLITS = ("train", "val", "calibration")


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def repartition(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Append-only signer split output exists: {output}")
    signer_map = _load_object(args.signer_map.resolve())
    assignment = _load_object(args.assignment.resolve())
    signer_to_split: dict[str, str] = {}
    for split in SPLITS:
        signers = assignment.get(split)
        if not isinstance(signers, list) or not signers:
            raise ValueError(f"Signer assignment lacks non-empty {split}")
        for signer in map(str, signers):
            if signer in signer_to_split:
                raise ValueError(f"Signer assigned to multiple splits: {signer}")
            signer_to_split[signer] = split

    records = []
    clip_ids = set()
    group_signers: dict[str, set[str]] = defaultdict(set)
    for manifest in args.manifest:
        for path in _manifest_paths(manifest.resolve()):
            clip = load_cache_clip(path)
            if clip.clip_id in clip_ids:
                raise ValueError(
                    f"Duplicate clip across input manifests: {clip.clip_id}"
                )
            clip_ids.add(clip.clip_id)
            metadata = json.loads(clip.metadata_json)
            source_clip = str(metadata.get("source_clip", clip.clip_id))
            source_group = str(metadata.get("source_group", ""))
            signer = str(signer_map.get(source_clip, "")).strip()
            if not source_group:
                raise ValueError(f"Clip lacks source_group: {clip.clip_id}")
            if not signer:
                raise ValueError(f"Signer map lacks {source_clip}")
            if signer not in signer_to_split:
                raise ValueError(f"Signer has no split assignment: {signer}")
            group_signers[source_group].add(signer)
            records.append(
                {
                    "path": path.resolve(),
                    "clip_id": clip.clip_id,
                    "source_group": source_group,
                    "source_clip": source_clip,
                    "signer": signer,
                    "frames": len(clip.frame_names),
                    "official_split": str(metadata.get("official_split", "unknown")),
                    "phase2_split": signer_to_split[signer],
                }
            )
    group_split_conflicts = {
        group: sorted({signer_to_split[signer] for signer in signers})
        for group, signers in group_signers.items()
        if len({signer_to_split[signer] for signer in signers}) != 1
    }
    if group_split_conflicts:
        raise ValueError(
            f"Source groups would cross signer-assigned splits: {group_split_conflicts}"
        )

    output.mkdir(parents=True)
    report_splits = {}
    groups_by_split = {}
    signers_by_split = {}
    for split in SPLITS:
        rows = sorted(
            (record for record in records if record["phase2_split"] == split),
            key=lambda record: record["clip_id"],
        )
        if not rows:
            raise ValueError(f"Repartition produced empty split: {split}")
        manifest = output / f"{split}.json"
        manifest.write_text(
            json.dumps({"clips": [str(row["path"]) for row in rows]}, indent=2) + "\n",
            encoding="utf-8",
        )
        groups = {row["source_group"] for row in rows}
        signers = {row["signer"] for row in rows}
        groups_by_split[split] = groups
        signers_by_split[split] = signers
        report_splits[split] = {
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "clips": len(rows),
            "frames": sum(row["frames"] for row in rows),
            "source_groups": len(groups),
            "signers": sorted(signers),
            "official_split_composition": dict(
                sorted(Counter(row["official_split"] for row in rows).items())
            ),
        }
    source_overlaps = {}
    signer_overlaps = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            key = f"{left}__{right}"
            source_overlaps[key] = sorted(
                groups_by_split[left] & groups_by_split[right]
            )
            signer_overlaps[key] = sorted(
                signers_by_split[left] & signers_by_split[right]
            )
    checks = {
        "source_group_disjoint": not any(source_overlaps.values()),
        "signer_disjoint": not any(signer_overlaps.values()),
        "all_input_clips_assigned_once": sum(
            split["clips"] for split in report_splits.values()
        )
        == len(records),
    }
    report = {
        "schema": "phase2r-signer-repartition-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "signer_map_sha256": sha256_file(args.signer_map),
        "assignment_sha256": sha256_file(args.assignment),
        "input_manifests": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.manifest
        ],
        "multi_identity_source_groups": {
            group: sorted(signers)
            for group, signers in sorted(group_signers.items())
            if len(signers) > 1
        },
        "splits": report_splits,
        "source_group_overlaps": source_overlaps,
        "signer_overlaps": signer_overlaps,
    }
    report_path = output / "repartition_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not report["passed"]:
        raise RuntimeError(f"Signer repartition failed: {checks}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--signer-map", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(repartition(parse_args()), indent=2, sort_keys=True))
