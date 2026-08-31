"""Prove that locked PHOENIX selections match SOKE's released loader splits."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

from phase2_refiner.data.prepare_phoenix_soke_full import (
    SPLITS,
    _source_group,
    _target_indices,
)
from phase2_refiner.provenance import sha256_file


SCHEMA = "signal4d-phoenix-soke-loader-split-parity-v2"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rb") as handle:
        rows = pickle.load(handle, encoding="latin1")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Invalid SOKE split annotations: {path}")
    return rows


def audit(selection_root: Path, soke_root: Path) -> dict[str, Any]:
    selection_root = selection_root.resolve()
    soke_root = soke_root.resolve()
    reports = {}
    split_clip_ids: dict[str, set[str]] = {}
    split_source_groups: dict[str, set[str]] = {}
    for split in SPLITS:
        annotation = soke_root / f"phoenix14t.{split}"
        selection_path = selection_root / split / "selection.json"
        rows = _load_rows(annotation)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        clips = selection.get("clips", [])
        if len(rows) != len(clips):
            raise ValueError(
                f"SOKE/selection row-count mismatch for {split}: "
                f"{len(rows)} != {len(clips)}"
            )
        fitted_frames = video_frames = 0
        clip_ids: set[str] = set()
        source_groups: set[str] = set()
        for index, (row, clip) in enumerate(zip(rows, clips)):
            expected_name = str(row.get("name", ""))
            if clip.get("official_name") != expected_name:
                raise ValueError(
                    f"SOKE order/name mismatch {split} row {index}: "
                    f"{expected_name!r} != {clip.get('official_name')!r}"
                )
            prefix, separator, source_clip = expected_name.partition("/")
            if not separator or prefix != split or clip.get("source_clip") != source_clip:
                raise ValueError(f"Invalid SOKE clip binding: {expected_name}")
            clip_id = str(clip.get("clip_id", source_clip))
            if clip_id in clip_ids:
                raise ValueError(f"Duplicate clip ID within {split}: {clip_id}")
            clip_ids.add(clip_id)
            source_group = _source_group(source_clip)
            declared_group = clip.get("source_group")
            if declared_group is not None and str(declared_group) != source_group:
                raise ValueError(
                    f"Invalid source group for {expected_name}: "
                    f"{declared_group!r} != {source_group!r}"
                )
            source_groups.add(source_group)
            for soke_field, selection_field in (
                ("gloss", "gloss"),
                ("text", "text"),
                ("signer", "signer_id"),
            ):
                if str(row.get(soke_field, "")) != str(clip.get(selection_field, "")):
                    raise ValueError(
                        f"SOKE {soke_field} mismatch for {expected_name}"
                    )
            if row.get("src") not in (None, "phoenix"):
                raise ValueError(f"Unexpected SOKE source for {expected_name}: {row.get('src')}")
            contract_frames = int(clip["source_contract"]["frame_count"])
            if int(row.get("num_frames", contract_frames)) != contract_frames:
                raise ValueError(f"SOKE/video frame-count mismatch for {expected_name}")
            target_indices = _target_indices(Path(clip["target_dir"]))
            declared_targets = [
                int(value) for value in clip["target_frame_indices_one_based"]
            ]
            if target_indices != declared_targets:
                raise ValueError(f"SOKE target-file coverage mismatch for {expected_name}")
            if [value - 1 for value in target_indices] != [
                int(value) for value in clip["frame_indices"]
            ]:
                raise ValueError(f"SOKE RGB/target index mismatch for {expected_name}")
            fitted_frames += len(target_indices)
            video_frames += contract_frames
        reports[split] = {
            "clips": len(clips),
            "soke_loader_pose_frames": fitted_frames,
            "source_video_frames": video_frames,
            "missing_fitted_frames": video_frames - fitted_frames,
            "selection": str(selection_path),
            "selection_sha256": sha256_file(selection_path),
            "soke_annotation": str(annotation),
            "soke_annotation_sha256": sha256_file(annotation),
            "exact_order_and_metadata_match": True,
            "exact_target_filename_coverage": True,
        }
        split_clip_ids[split] = clip_ids
        split_source_groups[split] = source_groups

    cross_split = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        clip_overlap = sorted(split_clip_ids[left] & split_clip_ids[right])
        source_group_overlap = sorted(
            split_source_groups[left] & split_source_groups[right]
        )
        if clip_overlap:
            raise ValueError(
                f"Exact PHOENIX clip leakage between {left}/{right}: "
                f"{clip_overlap[:3]}"
            )
        cross_split[f"{left}_{right}"] = {
            "clip_id_overlap_count": len(clip_overlap),
            "source_group_overlap_count": len(source_group_overlap),
            "source_group_overlap_examples": source_group_overlap[:10],
        }
    return {
        "schema": SCHEMA,
        "split_authority": "SOKE H2SMotionDatasetVQ phoenix14t.{train,dev,test}",
        "pose_loader": "sorted files under PHOENIX_ROOT/<split>/<clip>",
        "pose_payloads_opened": False,
        "exact_soke_loader_parity": True,
        "no_cross_split_clip_overlap": True,
        "source_group_disjoint": all(
            item["source_group_overlap_count"] == 0
            for item in cross_split.values()
        ),
        "source_group_note": (
            "SOKE's official split is authoritative. Different sentence clips "
            "from the same dated news broadcast may occur in multiple splits; "
            "these context overlaps are reported but are not moved across splits."
        ),
        "cross_split": cross_split,
        "splits": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--soke-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = audit(args.selection_root, args.soke_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
