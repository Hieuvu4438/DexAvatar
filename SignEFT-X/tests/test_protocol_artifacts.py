from __future__ import annotations

import json
import os
from pathlib import Path

from signeft.data.manifest import read_manifest
from signeft.io_utils import sha256_file


ROOT = Path(__file__).parents[1]
EVALUATOR_SHA = "2722b5cd30d4baba23599a455cab483b143e6595d292f02de9643af4eebd5300"


def test_locked_protocol_and_manifest_are_exact_and_ordered():
    protocol = json.loads((ROOT / "runs" / "protocol_lock.json").read_text(encoding="utf-8"))
    records = read_manifest(ROOT / "manifests" / "trv2v.jsonl")
    assert protocol["status"] == "ok"
    assert protocol["sign_count"] == 57
    assert protocol["frame_count"] == 1493
    assert protocol["reported_paper_frame_count"] == 2872
    assert protocol["paper_protocol_parity"] is False
    assert protocol["claim_scope"] == "same_attached_1493_frame_protocol_only"
    assert len(records) == 1493
    assert len({record.record_id for record in records}) == len(records)
    assert len({(record.sign_id, record.frame_index) for record in records}) == len(records)
    by_sign: dict[str, list] = {}
    for record in records:
        by_sign.setdefault(record.sign_id, []).append(record)
    assert len(by_sign) == 57
    for sign_records in by_sign.values():
        assert [item.frame_index for item in sign_records] == list(range(len(sign_records)))
        assert len({item.source_frame_id for item in sign_records}) == len(sign_records)


def test_author_evaluator_is_hash_locked_and_read_only():
    protocol = json.loads((ROOT / "runs" / "protocol_lock.json").read_text(encoding="utf-8"))
    evaluator = Path(protocol["evaluator"])
    assert evaluator.is_file()
    assert sha256_file(evaluator) == EVALUATOR_SHA == protocol["evaluator_sha256"]
    assert protocol["evaluator_read_only"] is True
    assert os.stat(evaluator).st_mode & 0o222 == 0


def test_c0_materializes_every_manifest_frame_once():
    summary = json.loads(
        (ROOT / "runs" / "signeft_c0_a3f" / "official_meshes" / "materialization_summary.json")
        .read_text(encoding="utf-8")
    )
    preflight = json.loads(
        (ROOT / "runs" / "signeft_c0_a3f" / "preflight.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == preflight["status"] == "ok"
    assert summary["signs"] == preflight["signs"] == 57
    assert summary["frames"] == preflight["frames"] == 1493
    assert sum(item["frames"] for item in summary["items"]) == 1493
