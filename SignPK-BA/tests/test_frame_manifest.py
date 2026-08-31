from pathlib import Path

from signpk.data.frame_manifest import SignManifest, build_manifest


def test_build_manifest_uses_explicit_x2_ids(tmp_path: Path):
    frames = tmp_path / "frames" / "Test"
    gt = tmp_path / "gt" / "Test"
    frames.mkdir(parents=True)
    gt.mkdir(parents=True)
    for video_id in (11, 13, 15):
        (frames / f"low_{video_id}.png").write_bytes(b"image")
        (gt / f"{video_id * 2:05d}.obj").write_text("", encoding="utf-8")
    manifest = build_manifest(
        "Test", (11, 15), tmp_path / "frames", tmp_path / "gt", "~0", fps=25, strict_gt=True
    )
    assert manifest.frame_ids == (11, 13, 15)
    assert manifest.gt_ids == (22, 26, 30)
    assert [row.prediction_frame_id for row in manifest.records] == [22, 26, 30]
    path = tmp_path / "manifest.json"
    manifest.save(path)
    assert SignManifest.load(path, validate_paths=True) == manifest


def test_duplicate_ids_are_rejected(tmp_path: Path):
    frames = tmp_path / "frames" / "Test"
    gt = tmp_path / "gt" / "Test"
    frames.mkdir(parents=True)
    gt.mkdir(parents=True)
    (frames / "low_1.png").write_bytes(b"x")
    (frames / "copy_1.png").write_bytes(b"x")
    (gt / "00002.obj").write_text("", encoding="utf-8")
    try:
        build_manifest("Test", (1, 1), tmp_path / "frames", tmp_path / "gt", "0", image_glob="*.png")
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate video IDs were accepted")

