from __future__ import annotations

import json
from pathlib import Path
import stat

from signeft.io_utils import atomic_write_json, sha256_file


def _read_signs(path: Path) -> dict[str, str]:
    signs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) != 2:
            raise ValueError(f"invalid sign row: {line!r}")
        if tokens[0] in signs:
            raise ValueError(f"duplicate sign: {tokens[0]}")
        signs[tokens[0]] = tokens[1]
    return signs


def _read_reference_records(root: Path, sign: str) -> list[dict[str, object]]:
    path = root / f"{sign}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [int(row["source_frame_id"]) for row in rows]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise RuntimeError(f"reference frame IDs are duplicate/unsorted: {path}")
    return rows


def lock_protocol(
    signs_file: Path,
    segments_file: Path,
    evaluator: Path,
    reference_manifest_root: Path,
    output: Path,
    *,
    expected_evaluator_sha256: str,
    expected_signs: int,
    expected_frames: int,
    reported_paper_frames: int,
) -> dict[str, object]:
    if not evaluator.is_file():
        raise FileNotFoundError(evaluator)
    evaluator_sha = sha256_file(evaluator)
    if evaluator_sha != expected_evaluator_sha256:
        raise RuntimeError(f"official evaluator hash mismatch: {evaluator_sha}")
    mode = evaluator.stat().st_mode
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise PermissionError(f"official evaluator must be read-only: {evaluator}")
    signs = _read_signs(signs_file)
    segments = json.loads(segments_file.read_text(encoding="utf-8"))
    if set(signs) != set(segments):
        raise RuntimeError(f"sign/segment mismatch: {sorted(set(signs) ^ set(segments))}")
    if len(signs) != expected_signs:
        raise RuntimeError(f"sign count {len(signs)} != {expected_signs}")
    items = []
    record_ids: set[str] = set()
    total = 0
    for sign in sorted(signs):
        rows = _read_reference_records(reference_manifest_root, sign)
        frames = []
        for row in rows:
            frame_id = int(row["source_frame_id"])
            record_id = f"{sign}/{frame_id}"
            if record_id in record_ids:
                raise RuntimeError(f"duplicate protocol record: {record_id}")
            record_ids.add(record_id)
            frames.append({
                "source_frame_id": frame_id,
                "sequence_index": int(row["sequence_index"]),
                "gt_frame_id": None if row.get("gt_frame_id") is None else int(row["gt_frame_id"]),
            })
        total += len(frames)
        items.append({
            "sign_id": sign,
            "sign_class": signs[sign],
            "segment": [int(value) for value in segments[sign]],
            "frames": frames,
        })
    if total != expected_frames:
        raise RuntimeError(f"frame count {total} != {expected_frames}")
    report = {
        "schema_version": "signeft.protocol-lock.v1",
        "status": "ok",
        "evaluator": str(evaluator.resolve()),
        "evaluator_sha256": evaluator_sha,
        "evaluator_read_only": True,
        "signs_file": str(signs_file.resolve()),
        "signs_sha256": sha256_file(signs_file),
        "segments_file": str(segments_file.resolve()),
        "segments_sha256": sha256_file(segments_file),
        "reference_manifest_root": str(reference_manifest_root.resolve()),
        "sign_count": len(items),
        "frame_count": total,
        "reported_paper_frame_count": int(reported_paper_frames),
        "paper_protocol_parity": total == int(reported_paper_frames),
        "claim_scope": (
            "paper_comparable" if total == int(reported_paper_frames)
            else "same_attached_1493_frame_protocol_only"
        ),
        "items": items,
    }
    atomic_write_json(output, report)
    return report

