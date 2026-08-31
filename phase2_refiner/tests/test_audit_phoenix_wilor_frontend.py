import json
import pickle
from pathlib import Path

import pytest

from phase2_refiner.data.audit_phoenix_wilor_frontend import (
    audit_once,
    require_all_verified,
)
from phase2_refiner.provenance import sha256_file


def _write_shard(run_root: Path, split: str, index: int, key: str) -> None:
    shard_root = run_root / "wilor_shards" / split
    output = run_root / "wilor_outputs" / split / f"shard_{index:04d}"
    (output / "hamer").mkdir(parents=True)
    (output / "wilor").mkdir()
    manifest = shard_root / f"shard_{index:04d}.json"
    manifest.write_text(json.dumps({"records": [{"image_key": key}]}))
    with (output / "hamer" / "hamer.pkl").open("wb") as handle:
        pickle.dump({}, handle)
    with (output / "wilor" / "wilor.pkl").open("wb") as handle:
        pickle.dump(
            {
                "images": {key: {}},
                "meta": {
                    "frame_manifest_sha256": sha256_file(manifest),
                    "frame_manifest_sources_verified": True,
                },
            },
            handle,
        )


def test_incremental_wilor_audit_writes_hash_ledger(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    shard_root = run_root / "wilor_shards" / "train"
    shard_root.mkdir(parents=True)
    (shard_root / "shard_report.json").write_text(
        json.dumps({"shards": [{"frames": 1}, {"frames": 1}]})
    )
    _write_shard(run_root, "train", 0, "a.png")
    output = tmp_path / "audit.json"

    first = audit_once(
        run_root=run_root,
        output=output,
        splits=("train",),
        minimum_age_seconds=0,
    )
    assert first["splits"]["train"]["verified_shards"] == 1
    assert not first["splits"]["train"]["all_verified"]
    record = first["verified"]["train/shard_0000"]
    assert record["raw_keys"] == 1
    assert record["hamer_dropouts"] == 1
    assert len(record["wilor_sha256"]) == 64

    _write_shard(run_root, "train", 1, "b.png")
    second = audit_once(
        run_root=run_root,
        output=output,
        splits=("train",),
        minimum_age_seconds=0,
    )
    assert second["splits"]["train"]["verified_shards"] == 2
    assert second["all_verified"]
    assert second["verified"]["train/shard_0000"] == record
    require_all_verified(second)


def test_require_all_verified_rejects_incomplete_audit() -> None:
    with pytest.raises(RuntimeError, match="train"):
        require_all_verified(
            {
                "all_verified": False,
                "splits": {"train": {"all_verified": False}},
            }
        )
