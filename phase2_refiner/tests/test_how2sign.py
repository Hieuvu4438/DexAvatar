import numpy as np

from phase2_refiner.data.build_how2sign_cache import (
    _mapped_keypoints,
    _quality,
    _source_group,
)


def test_how2sign_source_group_preserves_underscore_in_video_id() -> None:
    assert _source_group("_G0MZFLIHa0_5-5-rgb_front") == "_G0MZFLIHa0"
    assert _source_group("FZd8Iv9ACVw_3_4-8-rgb_front") == "FZd8Iv9ACVw"


def test_how2sign_wholebody_mapping_is_finite_and_bounded() -> None:
    keypoints = np.full((3, 133, 2), 0.5, dtype=np.float32)
    scores = np.full((3, 133), 8.0, dtype=np.float32)
    mapped, confidence, valid = _mapped_keypoints(keypoints, scores)
    assert mapped.shape == (3, 51, 2)
    assert confidence.shape == (3, 51)
    assert valid.shape == (3, 51)
    assert valid.all()
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))


def test_how2sign_quality_rejects_catastrophic_pose_fraction() -> None:
    pose = np.zeros((10, 51, 3), dtype=np.float32)
    assert _quality(pose)["passed"]
    pose[:2, 0, 0] = 10.0
    assert not _quality(pose)["passed"]
