from __future__ import annotations

import json

from signal4d.data.manifest import ClipManifest, load_manifest, write_manifest
from signal4d.data.split import freeze_clip_splits


def test_frozen_splits_are_deterministic_and_disjoint(tmp_path) -> None:
    source = tmp_path / "all.jsonl"
    rows = [
        ClipManifest(
            dataset="test",
            clip_id=f"clip_{index}",
            split="development",
            fps=15,
            frame_ids=[0, 1],
            image_relpaths=[f"{index}/0.png", f"{index}/1.png"],
        )
        for index in range(7)
    ]
    write_manifest(rows, source)
    first = freeze_clip_splits(source, tmp_path / "first", 2, 2, 9, ("clip_6",))
    second = freeze_clip_splits(source, tmp_path / "second", 2, 2, 9, ("clip_6",))
    assert first["manifest_sha256"] == second["manifest_sha256"]
    clip_sets = []
    for split in ("calibration", "development", "test"):
        manifest = load_manifest(tmp_path / "first" / f"sgnify_{split}.jsonl")
        clip_sets.append({row.clip_id for row in manifest})
        assert all(row.split == split for row in manifest)
    assert not (
        clip_sets[0] & clip_sets[1] | clip_sets[0] & clip_sets[2] | clip_sets[1] & clip_sets[2]
    )
    metadata = json.loads((tmp_path / "first" / "split_freeze.json").read_text())
    assert metadata["clip_disjoint"] is True
    assert metadata["signer_disjoint"] is False
    assert metadata["extra_development_quarantine"] == ["clip_6"]
