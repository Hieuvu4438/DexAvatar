import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from phase2_refiner.data.cache_schema import save_cache_clip
from phase2_refiner.data.sample_signavatars_target_audit import build_sample
from phase2_refiner.tests.test_cache import make_clip


def _audit_fixture(tmp_path: Path, count: int = 9) -> tuple[Path, Path]:
    clips = []
    signer_map = {}
    for index in range(count):
        clip = make_clip(4)
        clip.clip_id = f"clip-{index}"
        clip.frame_names = np.asarray([f"frame_{value:03d}" for value in range(4)])
        clip.track_valid = np.ones((4, 51), dtype=bool)
        clip.in_frame = np.ones((4, 51), dtype=bool)
        clip.in_frame[: index % 3, 21:51] = False
        clip.keypoints_2d[:, 21:51, 0] = np.arange(4)[:, None] * (index + 1) * 0.003
        source_clip = f"source-{index}"
        clip.metadata_json = json.dumps(
            {"source_clip": source_clip, "source_group": f"group-{index}"}
        )
        path = tmp_path / f"clip-{index}.npz"
        save_cache_clip(path, clip)
        clips.append(str(path))
        signer_map[source_clip] = f"signer-{index % 3}"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"clips": clips}))
    signers = tmp_path / "signers.json"
    signers.write_text(json.dumps(signer_map))
    return manifest, signers


def test_audit_sample_is_deterministic_and_source_disjoint(tmp_path: Path) -> None:
    manifest, signers = _audit_fixture(tmp_path)
    first = build_sample(
        argparse.Namespace(
            manifest=manifest,
            signer_map=signers,
            output=tmp_path / "sample-1.json",
            sample_size=6,
            seed=7,
        )
    )
    second = build_sample(
        argparse.Namespace(
            manifest=manifest,
            signer_map=signers,
            output=tmp_path / "sample-2.json",
            sample_size=6,
            seed=7,
        )
    )
    assert [row["clip_id"] for row in first["clips"]] == [
        row["clip_id"] for row in second["clips"]
    ]
    assert first["source_group_disjoint"] is True
    assert set(first["stratified_by"]) == {
        "signer",
        "hand_activity",
        "hand_size",
        "truncation",
        "motion",
    }


def test_audit_sample_rejects_missing_signer_identity(tmp_path: Path) -> None:
    manifest, signers = _audit_fixture(tmp_path, count=2)
    signers.write_text("{}")
    with pytest.raises(ValueError, match="Signer map lacks"):
        build_sample(
            argparse.Namespace(
                manifest=manifest,
                signer_map=signers,
                output=tmp_path / "sample.json",
                sample_size=1,
                seed=7,
            )
        )
