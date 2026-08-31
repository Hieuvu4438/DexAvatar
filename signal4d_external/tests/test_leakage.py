from __future__ import annotations

import json

import pytest

from signal4d_external.leakage import audit_manifest


def test_manifest_requires_explicit_sgnify_exclusion(tmp_path) -> None:
    clip = tmp_path / "clip.npz"
    clip.write_bytes(b"placeholder")
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps({"dataset": "How2Sign", "clips": [str(clip)]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sgnify_excluded"):
        audit_manifest(manifest, scan_clips=False)


def test_manifest_only_audit_accepts_allowlisted_external_split(tmp_path) -> None:
    clip = tmp_path / "clip.npz"
    clip.write_bytes(b"placeholder")
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "How2Sign",
                "sgnify_excluded": True,
                "clips": [str(clip)],
            }
        ),
        encoding="utf-8",
    )
    report = audit_manifest(manifest, scan_clips=False)
    assert report["sgnify_reads"] == 0
    assert report["clips"] == 1
