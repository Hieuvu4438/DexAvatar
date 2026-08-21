import json

from signal4d.data.manifest import build_sgnify_manifest


def test_available_frame_manifest_does_not_invent_inclusive_frames(tmp_path) -> None:
    frames = tmp_path / "frames" / "Sign"
    frames.mkdir(parents=True)
    for value in (10, 12, 14):
        (frames / f"low_{value:03d}.png").write_bytes(b"x")
    segments = tmp_path / "segments.json"
    segments.write_text(json.dumps({"Sign": [10, 14]}))
    rows = build_sgnify_manifest(tmp_path / "frames", segments, fps=15)
    assert rows[0].frame_ids == [10, 12, 14]
    assert not rows[0].is_contiguous
