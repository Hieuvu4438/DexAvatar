import argparse
import gzip
import pickle
import zipfile

from phase2_refiner.data.audit_csl_daily_release import audit


def test_csl_release_audit_matches_metadata_rgb_keypoints_and_pose(tmp_path) -> None:
    metadata_root = tmp_path / "metadata"
    rgb_root = tmp_path / "rgb"
    keypoint_root = tmp_path / "keypoints"
    metadata_root.mkdir()
    rgb_root.mkdir()
    keypoint_root.mkdir()
    rows = {
        "train": [{"name": "A", "signer": 0, "gloss": "g", "text": "t", "num_frames": 2}],
        "val": [{"name": "B", "signer": 1, "gloss": "g", "text": "t", "num_frames": 1}],
        "test": [{"name": "C", "signer": 0, "gloss": "g", "text": "t", "num_frames": 1}],
    }
    for split, payload in rows.items():
        with gzip.open(metadata_root / f"csl_clean.{split}", "wb") as handle:
            pickle.dump(payload, handle)
        for row in payload:
            (rgb_root / f"{row['name']}.mp4").touch()
            (keypoint_root / f"{row['name']}.pkl").touch()
    pose_zip = tmp_path / "poses.zip"
    with zipfile.ZipFile(pose_zip, "w") as archive:
        archive.writestr("csl-daily_pose/A/000000.pkl", b"a")
        archive.writestr("csl-daily_pose/A/000001.pkl", b"a")
        archive.writestr("csl-daily_pose/B/000000.pkl", b"b")
        archive.writestr("csl-daily_pose/C/000000.pkl", b"c")
    report = audit(
        argparse.Namespace(
            metadata_root=metadata_root,
            pose_zip=pose_zip,
            rgb_root=rgb_root,
            keypoint_root=keypoint_root,
        )
    )
    assert report["decision"] == "PASS"
    assert report["metadata_clips"] == 3
    assert report["pose_files"] == 4
