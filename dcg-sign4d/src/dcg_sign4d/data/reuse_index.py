"""Immutable reference indexes for already-computed upstream artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from dcg_sign4d.utils.hashing import canonical_hash, file_sha256


def _manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty reuse manifest")
    ids = [row.get("clip_id") for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("reuse manifest has missing/duplicate clip IDs")
    return rows


def build_reuse_index(
    *,
    artifact_type: str,
    source_root: str | Path,
    source_report_name: str,
    expected_source_report_sha256: str,
    source_marker: str,
    per_clip_artifact_name: str,
    per_clip_hash_field: str,
    manifest_path: str | Path,
    output: str | Path,
    development_only: bool,
    required_split: str | None = None,
) -> dict[str, Any]:
    """Verify and index an immutable source without copying or recomputing tensors."""

    source = Path(source_root)
    report_path = source / source_report_name
    if not (source / source_marker).is_file():
        raise ValueError(f"source has no {source_marker} marker")
    if file_sha256(report_path) != expected_source_report_sha256:
        raise ValueError("source report hash mismatch")
    report = json.loads(report_path.read_text("utf-8"))
    if bool(report.get("development_only")) != development_only:
        raise ValueError("source/config development status mismatch")
    manifest_path = Path(manifest_path)
    rows = _manifest_rows(manifest_path)
    if required_split is not None and any(row.get("split") != required_split for row in rows):
        raise ValueError(f"manifest contains clips outside split {required_split!r}")
    report_rows = {row["clip_id"]: row for row in report.get("per_clip", [])}
    expected_ids = {row["clip_id"] for row in rows}
    if set(report_rows) != expected_ids:
        missing = sorted(expected_ids - set(report_rows))
        extra = sorted(set(report_rows) - expected_ids)
        raise ValueError(f"source/manifest clip mismatch; missing={missing}, extra={extra}")
    indexed = []
    for row in rows:
        clip_id = row["clip_id"]
        clip_root = source / clip_id
        metadata_path = clip_root / "metadata.json"
        artifact_path = clip_root / per_clip_artifact_name
        metadata = json.loads(metadata_path.read_text("utf-8"))
        if file_sha256(artifact_path) != metadata.get(per_clip_hash_field):
            raise ValueError(f"per-clip artifact hash mismatch: {clip_id}")
        frame_ids = metadata.get("frame_ids")
        if frame_ids is None:
            with np.load(artifact_path, allow_pickle=False) as arrays:
                if "frame_ids" not in arrays:
                    raise ValueError(f"{clip_id}: artifact contains no frame IDs")
                frame_ids = [int(value) for value in arrays["frame_ids"]]
        expected_frames = row.get("frame_ids")
        if expected_frames is not None and frame_ids != expected_frames:
            raise ValueError(f"{clip_id}: source/manifest frame mapping mismatch")
        if int(report_rows[clip_id]["frames"]) != len(frame_ids):
            raise ValueError(f"{clip_id}: report frame count mismatch")
        indexed.append(
            {
                "clip_id": clip_id,
                "frames": len(frame_ids),
                "frame_ids": frame_ids,
                "artifact": str(artifact_path.resolve()),
                "artifact_sha256": file_sha256(artifact_path),
                "metadata": str(metadata_path.resolve()),
                "metadata_sha256": file_sha256(metadata_path),
            }
        )
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"immutable reuse index exists: {destination}")
    destination.mkdir(parents=True)
    incomplete = destination / ".index_incomplete"
    incomplete.write_text("incomplete\n", "utf-8")
    payload = {
        "schema_version": "dcg_reused_artifact_index_v1",
        "artifact_type": artifact_type,
        "development_only": development_only,
        "scientific_status": (
            "DEVELOPMENT_ONLY_NOT_FINAL_EVIDENCE" if development_only else "FROZEN_SOURCE_REUSE"
        ),
        "source_root": str(source.resolve()),
        "source_report_sha256": expected_source_report_sha256,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "clips": len(indexed),
        "frames": sum(row["frames"] for row in indexed),
        "recomputed": False,
        "copied": False,
        "per_clip": indexed,
    }
    payload["index_identity_sha256"] = canonical_hash(payload)
    (destination / "index.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", "utf-8"
    )
    os.replace(incomplete, destination / "REUSE_INDEX_COMPLETE")
    return payload
