"""Strict validation of locally generated WiLoR raw-v3 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_wilor_raw_v3(
    wilor: dict, *, expected_frame_manifest: Path | None = None
) -> tuple[dict[str, dict], dict]:
    meta = wilor.get("meta", {})
    if meta.get("format") != "wilor_raw_v3":
        raise ValueError("Consumer requires a wilor_raw_v3 artifact")
    if meta.get("frame_manifest_sources_verified") is not True:
        raise ValueError("WiLoR artifact lacks verified manifest sources")
    for field in (
        "exporter_sha256",
        "wilor_checkpoint_sha256",
        "detector_checkpoint_sha256",
        "model_config_sha256",
    ):
        value = str(meta.get(field, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"WiLoR artifact lacks valid {field}")
    if float(meta.get("base_focal_length", 0.0)) <= 0.0:
        raise ValueError("WiLoR artifact lacks a valid base focal length")
    commit = str(meta.get("wilor_repository_commit", ""))
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("WiLoR artifact lacks a valid repository commit")

    manifest_path = Path(str(meta.get("frame_manifest", "")))
    if not manifest_path.is_file():
        raise FileNotFoundError(f"WiLoR frame manifest is unavailable: {manifest_path}")
    manifest_hash = sha256(manifest_path)
    if manifest_hash != meta.get("frame_manifest_sha256"):
        raise ValueError("WiLoR frame-manifest hash mismatch")
    if expected_frame_manifest is not None:
        expected_hash = sha256(expected_frame_manifest)
        if manifest_hash != expected_hash:
            raise ValueError("WiLoR artifact belongs to a different frame manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_keys = [str(record["image_key"]) for record in manifest["records"]]
    images = wilor.get("images", {})
    if set(images) != set(expected_keys) or len(images) != len(expected_keys):
        raise ValueError("WiLoR image records do not exactly cover the frame manifest")
    if int(meta.get("frame_count", -1)) != len(expected_keys):
        raise ValueError("WiLoR metadata frame count is inconsistent")
    dropout = 0
    for key in expected_keys:
        hands = images[key].get("hands") if isinstance(images[key], dict) else None
        if not isinstance(hands, list):
            raise ValueError(f"WiLoR frame {key} lacks an explicit hands list")
        dropout += not hands
    if int(meta.get("detector_dropout_frames", -1)) != dropout:
        raise ValueError("WiLoR detector-dropout count is inconsistent")
    return images, meta
