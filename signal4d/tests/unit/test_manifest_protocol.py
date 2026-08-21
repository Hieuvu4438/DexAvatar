import pytest

from signal4d.data.manifest import ClipManifest, load_manifest, write_manifest
from signal4d.protocol import ProtocolGuard, ProtocolViolation


def _row(**updates):
    values = dict(
        dataset="toy",
        clip_id="clip",
        split="development",
        fps=25,
        frame_ids=[3, 4, 5],
        image_relpaths=["3.png", "4.png", "5.png"],
        frame_start=3,
        frame_end_exclusive=6,
    )
    values.update(updates)
    return ClipManifest(**values)


def test_manifest_roundtrip(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest([_row()], path)
    assert load_manifest(path)[0].frame_ids == [3, 4, 5]


def test_manifest_rejects_inclusive_and_reorder() -> None:
    with pytest.raises(ValueError, match="end_exclusive"):
        _row(frame_end_exclusive=5)
    with pytest.raises(ValueError, match="sorted"):
        _row(frame_ids=[3, 5, 4])


def test_explicit_missing_frames_are_preserved() -> None:
    row = _row(
        frame_ids=[3, 5, 9],
        image_relpaths=["3.png", "5.png", "9.png"],
        is_contiguous=False,
        frame_start=None,
        frame_end_exclusive=None,
    )
    assert row.frame_ids == [3, 5, 9]


def test_test_split_cannot_be_calibration() -> None:
    with pytest.raises(ValueError, match="test clips"):
        _row(split="test", allowed_for_calibration=True)
    guard = ProtocolGuard(track="clean")
    with pytest.raises(ProtocolViolation):
        guard.validate_action("calibrate", ["calibration", "test"])
