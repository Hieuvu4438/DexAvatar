import hashlib
import json

import pytest

from cusp_sl.wilor_artifact import validate_wilor_raw_v3


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(manifest):
    digest = "a" * 64
    return {
        "meta": {
            "format": "wilor_raw_v3",
            "frame_manifest": str(manifest),
            "frame_manifest_sha256": _hash(manifest),
            "frame_manifest_sources_verified": True,
            "exporter_sha256": digest,
            "wilor_checkpoint_sha256": digest,
            "detector_checkpoint_sha256": digest,
            "model_config_sha256": digest,
            "base_focal_length": 5000.0,
            "wilor_repository_commit": "c" * 40,
            "frame_count": 2,
            "detector_dropout_frames": 1,
        },
        "images": {"a.png": {"hands": []}, "b.png": {"hands": [{}]}},
    }


def test_wilor_artifact_requires_exact_manifest_coverage(tmp_path):
    manifest = tmp_path / "frames.json"
    manifest.write_text(json.dumps({"records": [
        {"image_key": "a.png"}, {"image_key": "b.png"}
    ]}))
    artifact = _artifact(manifest)
    images, meta = validate_wilor_raw_v3(
        artifact, expected_frame_manifest=manifest
    )
    assert set(images) == {"a.png", "b.png"}
    assert meta["frame_count"] == 2

    artifact["images"].pop("b.png")
    with pytest.raises(ValueError, match="exactly cover"):
        validate_wilor_raw_v3(artifact)


def test_wilor_artifact_rejects_unverified_sources(tmp_path):
    manifest = tmp_path / "frames.json"
    manifest.write_text(json.dumps({"records": [
        {"image_key": "a.png"}, {"image_key": "b.png"}
    ]}))
    artifact = _artifact(manifest)
    artifact["meta"]["frame_manifest_sources_verified"] = False
    with pytest.raises(ValueError, match="verified manifest sources"):
        validate_wilor_raw_v3(artifact)
