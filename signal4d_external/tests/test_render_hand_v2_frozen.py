from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from signal4d_external import render_hand_v2_frozen


def _write_protocol_tree(tmp_path: Path, decision: str = "PASS") -> argparse.Namespace:
    root = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    result_dir = root / "clip" / "smplifyx" / "results"
    source_dir = baseline / "clip" / "smplifyx" / "results"
    result_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    (result_dir / "low_00001.pkl").write_bytes(b"candidate")
    (source_dir / "low_00001.pkl").write_bytes(b"source")
    (root / "freeze_audit.json").write_text(
        json.dumps(
            {
                "decision": decision,
                "sgnify_target_reads": 0,
                "baseline_root": str(baseline.resolve()),
                "result_tree_sha256": "frozen-tree",
            }
        ),
        encoding="utf-8",
    )
    (root / "run_manifest.json").write_text(
        json.dumps({"frames": 1, "clips": [{"clip_id": "clip", "frames": 1}]}),
        encoding="utf-8",
    )
    return argparse.Namespace(
        output_root=root,
        baseline_root=baseline,
        model_folder=tmp_path / "models",
        device="cpu",
    )


def test_render_frozen_tree_validates_coverage_and_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _write_protocol_tree(tmp_path)
    observed_sources: list[str] = []

    def fake_render(
        result_dir: Path,
        mesh_dir: Path,
        source_paths: list[str],
        model_folder: Path,
        device: str,
    ) -> int:
        del result_dir, model_folder, device
        observed_sources.extend(source_paths)
        mesh_dir.mkdir(parents=True)
        (mesh_dir / "low_00001.obj").write_text("v 0 0 0\n", encoding="utf-8")
        return 1

    monkeypatch.setattr(render_hand_v2_frozen, "EXPECTED_FRAMES", 1)
    monkeypatch.setattr(
        render_hand_v2_frozen, "render_source_anchored_directory", fake_render
    )
    report = render_hand_v2_frozen.run(args)

    assert report["frames"] == 1
    assert report["sgnify_target_reads"] == 0
    assert observed_sources == [
        str((args.baseline_root / "clip/smplifyx/results/low_00001.pkl").resolve())
    ]
    assert (args.output_root / "render_manifest.json").is_file()
    assert not (args.output_root / ".render_incomplete").exists()


def test_render_rejects_nonpassing_freeze_before_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _write_protocol_tree(tmp_path, decision="FAIL")
    monkeypatch.setattr(render_hand_v2_frozen, "EXPECTED_FRAMES", 1)
    with pytest.raises(ValueError, match="not frozen target-free"):
        render_hand_v2_frozen.run(args)


def test_render_rejects_wrong_mesh_names_and_leaves_incomplete_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _write_protocol_tree(tmp_path)

    def fake_render(
        result_dir: Path,
        mesh_dir: Path,
        source_paths: list[str],
        model_folder: Path,
        device: str,
    ) -> int:
        del result_dir, source_paths, model_folder, device
        mesh_dir.mkdir(parents=True)
        (mesh_dir / "wrong.obj").write_text("v 0 0 0\n", encoding="utf-8")
        return 1

    monkeypatch.setattr(render_hand_v2_frozen, "EXPECTED_FRAMES", 1)
    monkeypatch.setattr(
        render_hand_v2_frozen, "render_source_anchored_directory", fake_render
    )
    with pytest.raises(ValueError, match="Mesh coverage mismatch"):
        render_hand_v2_frozen.run(args)
    assert (args.output_root / ".render_incomplete").is_file()
    assert not (args.output_root / "render_manifest.json").exists()
