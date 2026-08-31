import json

import pytest

from dcg_sign4d.data.manifest import ManifestItem, load_manifest


def item(**updates):
    payload = {
        "clip_id": "signer001_clip0001",
        "video_path": "data/raw/clip.mp4",
        "fps_native": 30.0,
        "frame_count": 180,
        "width": 1920,
        "height": 1080,
        "signer_id": "signer001",
        "split": "train",
        "camera_id": "cam0",
        "dataset_name": "fixture",
        "dataset_version": "1.0",
        "license_id": "fixture-only",
    }
    payload.update(updates)
    return ManifestItem.model_validate(payload)


def test_native_timestamp_round_trip():
    value = item()
    assert value.effective_frame_count == 180
    assert value.timestamp_sec(30) == 1.0


def test_exact_resample_mapping():
    value = item(fps_effective=15.0, frame_mapping=(0, 2, 4, 6))
    assert value.effective_frame_count == 4
    assert value.timestamp_sec(3) == 0.2


@pytest.mark.parametrize(
    "updates",
    [
        {"dataset_version": "AUTHOR_REQUIRED"},
        {"license_id": "UNKNOWN"},
        {"fps_effective": 15.0},
        {"fps_effective": 15.0, "frame_mapping": (0, 2, 2)},
        {"fps_effective": 15.0, "frame_mapping": (0, 200)},
    ],
)
def test_rejects_unsafe_manifest(updates):
    with pytest.raises(ValueError):
        item(**updates)


def test_load_rejects_duplicate(tmp_path):
    value = item().model_dump(mode="json")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(value) + "\n" + json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(manifest)
