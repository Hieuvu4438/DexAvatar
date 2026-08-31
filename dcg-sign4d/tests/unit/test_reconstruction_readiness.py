from __future__ import annotations

import json

import yaml

from dcg_sign4d.inference.readiness import audit_reconstruction_readiness


def test_current_author_required_config_is_machine_blocked(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "clip_id": "clip",
                "video_path": str(tmp_path / "missing.mp4"),
                "fps_native": 30,
                "frame_count": 2,
                "width": 16,
                "height": 16,
                "signer_id": "s1",
                "split": "test",
                "camera_id": "c1",
                "dataset_name": "fixture",
                "dataset_version": "v1",
                "license_id": "fixture",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = yaml.safe_load(
        """
experiment:
  name: dcg
  development_only: false
data:
  window_length: AUTHOR_REQUIRED
geometry:
  patch_map: AUTHOR_REQUIRED
contact:
  checkpoint: AUTHOR_REQUIRED
diffusion:
  checkpoint: AUTHOR_REQUIRED
ranking:
  use_ground_truth: false
"""
    )
    report = audit_reconstruction_readiness(config, manifest)
    assert report["status"] == "BLOCKED"
    by_name = {row["name"]: row for row in report["checks"]}
    assert by_name["author_freezes"]["status"] == "BLOCKED"
    assert by_name["manifest"]["status"] == "PASS"
