import json

import pytest

from dcg_sign4d.observations.sapiens_adapter import load_sapiens_clip


def _write_frame(root, frame, instances):
    path = root / "clip/sapiens_1b" / f"low_{frame}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"instance_info": instances}), encoding="utf-8")


def test_sapiens_raw_adapter_preserves_frames_timestamps_and_missingness(tmp_path):
    instance = {"keypoints": [[1.0, 2.0]] * 133, "keypoint_scores": [0.8] * 133}
    _write_frame(tmp_path, 10, [instance])
    _write_frame(tmp_path, 12, [])
    batch, paths = load_sapiens_clip("clip", [10, 12], fps=10, detector_root=tmp_path)
    assert [path.name for path in paths] == ["low_10.json", "low_12.json"]
    assert batch.frame_ids.tolist() == [10, 12]
    assert batch.timestamps_sec.tolist() == [1.0, 1.2]
    assert batch.frame_available.tolist() == [True, False]
    assert int(batch.keypoint_available.sum()) == 133


def test_sapiens_raw_adapter_rejects_ambiguous_people(tmp_path):
    instance = {"keypoints": [[1.0, 2.0]] * 133, "keypoint_scores": [0.8] * 133}
    _write_frame(tmp_path, 1, [instance, instance])
    with pytest.raises(ValueError, match="multi-person"):
        load_sapiens_clip("clip", [1], fps=15, detector_root=tmp_path)


def test_sapiens_raw_adapter_does_not_clamp_unbounded_detector_score(tmp_path):
    instance = {"keypoints": [[1.0, 2.0]] * 133, "keypoint_scores": [1.08] * 133}
    _write_frame(tmp_path, 1, [instance])
    batch, _ = load_sapiens_clip("clip", [1], fps=15, detector_root=tmp_path)
    assert float(batch.raw_score.max()) == pytest.approx(1.08)
