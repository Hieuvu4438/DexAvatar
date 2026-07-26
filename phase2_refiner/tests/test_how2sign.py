import numpy as np
import torch

from phase2_refiner.data.build_how2sign_cache import (
    _mapped_keypoints,
    _quality,
    _source_group,
)
from phase2_refiner.data.refine_how2sign_targets import _bounded_delta
from phase2_refiner.data.cache_schema import CacheClip
from phase2_refiner.data.dataset import _keypoints_in_model_coordinates


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


def test_how2sign_hand_tracks_follow_smplx_pose_order() -> None:
    keypoints = np.zeros((1, 133, 2), dtype=np.float32)
    scores = np.full((1, 133), 8.0, dtype=np.float32)
    keypoints[0, :, 0] = np.arange(133, dtype=np.float32)
    mapped, _, _ = _mapped_keypoints(keypoints, scores)
    assert mapped[0, 21:36, 0].tolist() == [
        96,
        97,
        98,
        100,
        101,
        102,
        108,
        109,
        110,
        104,
        105,
        106,
        92,
        93,
        94,
    ]
    assert mapped[0, 36:51, 0].tolist() == [
        117,
        118,
        119,
        121,
        122,
        123,
        129,
        130,
        131,
        125,
        126,
        127,
        113,
        114,
        115,
    ]


def test_how2sign_quality_rejects_catastrophic_pose_fraction() -> None:
    pose = np.zeros((10, 51, 3), dtype=np.float32)
    assert _quality(pose)["passed"]
    pose[:2, 0, 0] = 10.0
    assert not _quality(pose)["passed"]


def test_temporal_teacher_bounds_vector_correction() -> None:
    raw = torch.randn(2, 4, 51, 3) * 100.0
    limits = torch.full((1, 1, 51, 1), 0.2)
    refine = torch.ones(2, 51, dtype=torch.bool)
    refine[:, 0] = False
    delta = _bounded_delta(raw, limits, refine)
    assert torch.linalg.vector_norm(delta, dim=-1).max() <= 0.2 + 1e-6
    assert torch.count_nonzero(delta[:, :, 0]) == 0


def test_legacy_how2sign_keypoints_are_standardized_to_lane_coordinates() -> None:
    clip = object.__new__(CacheClip)
    clip.keypoints_2d = np.asarray([[[0.0, 0.5], [1.0, 0.25]]], dtype=np.float32)
    clip.metadata_json = '{"dataset":"How2Sign"}'
    converted = _keypoints_in_model_coordinates(clip, slice(None))
    np.testing.assert_allclose(
        converted,
        np.asarray([[[-1.0, 0.0], [1.0, -0.5]]], dtype=np.float32),
    )
