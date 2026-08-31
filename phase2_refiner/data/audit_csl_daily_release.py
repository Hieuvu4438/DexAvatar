"""Audit CSL-Daily SOKE poses against metadata, RGB, and keypoint tracks.

The pose release contains millions of tiny files in a nested ZIP.  This audit
streams the central-directory listing through ``unzip -Z1`` and never expands
those files onto the filesystem.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import pickle
import subprocess
from typing import Any

from phase2_refiner.provenance import sha256_file


def _load_metadata(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Invalid or empty metadata: {path}")
    required = {"name", "signer", "gloss", "text", "num_frames"}
    for row in payload:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"Invalid metadata row: {path}")
    return payload


def _pose_counts(path: Path) -> tuple[dict[str, int], int]:
    process = subprocess.Popen(
        ["unzip", "-Z1", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("Failed to open unzip output streams")
    counts: dict[str, int] = defaultdict(int)
    files = 0
    for line in process.stdout:
        parts = line.rstrip("\n").split("/")
        if len(parts) == 3 and parts[0] == "csl-daily_pose" and parts[2].endswith(
            ".pkl"
        ):
            counts[parts[1]] += 1
            files += 1
    stderr = process.stderr.read()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"unzip -Z1 failed ({return_code}): {stderr[-1000:]}")
    return dict(counts), files


def _stem_set(root: Path, suffix: str) -> set[str]:
    return {path.stem for path in root.glob(f"*{suffix}") if path.is_file()}


def _difference(first: set[str], second: set[str]) -> dict[str, Any]:
    values = sorted(first - second)
    return {"count": len(values), "examples": values[:10]}


def audit(args: argparse.Namespace) -> dict[str, Any]:
    rows_by_split = {
        split: _load_metadata(args.metadata_root / f"csl_clean.{split}")
        for split in ("train", "val", "test")
    }
    names_by_split = {
        split: {str(row["name"]) for row in rows}
        for split, rows in rows_by_split.items()
    }
    for split, rows in rows_by_split.items():
        if len(names_by_split[split]) != len(rows):
            raise ValueError(f"Duplicate metadata clip in {split}")
    split_overlaps = {}
    splits = tuple(names_by_split)
    for index, first in enumerate(splits):
        for second in splits[index + 1 :]:
            overlap = sorted(names_by_split[first] & names_by_split[second])
            split_overlaps[f"{first}__{second}"] = overlap[:10]
            if overlap:
                raise ValueError(f"Metadata clip overlap: {first}/{second}")

    all_rows = [row for rows in rows_by_split.values() for row in rows]
    metadata_names = {str(row["name"]) for row in all_rows}
    expected_frames = {str(row["name"]): int(row["num_frames"]) for row in all_rows}
    pose_counts, pose_files = _pose_counts(args.pose_zip)
    pose_names = set(pose_counts)
    rgb_names = _stem_set(args.rgb_root, ".mp4")
    keypoint_names = _stem_set(args.keypoint_root, ".pkl")
    frame_mismatch = sorted(
        name
        for name in metadata_names & pose_names
        if pose_counts[name] != expected_frames[name]
    )

    comparisons = {
        "metadata_missing_pose": _difference(metadata_names, pose_names),
        "pose_missing_metadata": _difference(pose_names, metadata_names),
        "metadata_missing_rgb": _difference(metadata_names, rgb_names),
        "rgb_missing_metadata": _difference(rgb_names, metadata_names),
        "metadata_missing_keypoint": _difference(metadata_names, keypoint_names),
        "keypoint_missing_metadata": _difference(keypoint_names, metadata_names),
        "pose_frame_count_mismatch": {
            "count": len(frame_mismatch),
            "examples": [
                {
                    "clip": name,
                    "metadata": expected_frames[name],
                    "pose_files": pose_counts[name],
                }
                for name in frame_mismatch[:10]
            ],
        },
    }
    passed = not any(value["count"] for value in comparisons.values())
    report = {
        "schema_version": 1,
        "decision": "PASS" if passed else "FAIL",
        "metadata": {
            split: {
                "clips": len(rows),
                "signers": sorted({str(row["signer"]) for row in rows}),
                "nonempty_gloss": sum(bool(str(row["gloss"]).strip()) for row in rows),
                "nonempty_text": sum(bool(str(row["text"]).strip()) for row in rows),
            }
            for split, rows in rows_by_split.items()
        },
        "metadata_clips": len(metadata_names),
        "pose_sequences": len(pose_names),
        "pose_files": pose_files,
        "rgb_clips": len(rgb_names),
        "keypoint_tracks": len(keypoint_names),
        "split_clip_overlaps": split_overlaps,
        "signer_sets_overlap_by_official_design": True,
        "comparisons": comparisons,
        "artifacts": {
            "pose_zip": str(args.pose_zip.resolve()),
            "pose_zip_sha256": sha256_file(args.pose_zip),
            "rgb_root": str(args.rgb_root.resolve()),
            "keypoint_root": str(args.keypoint_root.resolve()),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--pose-zip", type=Path, required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--keypoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["decision"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
