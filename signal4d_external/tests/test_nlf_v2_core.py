from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from signal4d_external.nlf_v2_core import (
    FEATURE_COLUMNS,
    cache_pose_from_full,
    frame_features,
    geodesic_blend,
    global_rotations,
    nlf_body_candidate,
    nlf_observation_contract,
    tals,
    viterbi_benefit_selection,
)


def _cache() -> SimpleNamespace:
    frames = 2
    return SimpleNamespace(
        global_orient=np.zeros((frames, 3), np.float32),
        init_axis_angle=np.zeros((frames, 51, 3), np.float32),
        jaw_pose=np.zeros((frames, 3), np.float32),
        leye_pose=np.zeros((frames, 3), np.float32),
        reye_pose=np.zeros((frames, 3), np.float32),
        keypoints_2d=np.full((frames, 51, 2), 0.5, np.float32),
        keypoint_valid=np.ones((frames, 51), bool),
        u0_reliability=np.ones((frames, 51), np.float32),
        reprojection_residual_2d=np.zeros((frames, 51, 2), np.float32),
        image_size=np.repeat(np.asarray([[100, 200]], np.int32), frames, axis=0),
    )


def _observation() -> dict[str, np.ndarray]:
    joints = np.zeros((55, 2), np.float32)
    joints[:, 0] = 100.0
    joints[:, 1] = 50.0
    return {
        "pose": np.zeros(165, np.float32),
        "joints2d": joints,
        "joint_uncertainties": np.full(55, 50.0, np.float32),
        "boxes": np.asarray([10, 5, 100, 80, 0.9], np.float32),
    }


def test_geodesic_blend_has_exact_endpoints() -> None:
    first = np.eye(3, dtype=np.float32)[None]
    second = Rotation.from_rotvec([[0.2, -0.1, 0.4]]).as_matrix().astype(np.float32)
    np.testing.assert_allclose(geodesic_blend(first, second, 0.0), first, atol=1e-6)
    np.testing.assert_allclose(geodesic_blend(first, second, 1.0), second, atol=1e-6)


def test_nlf_candidate_preserves_global_wrists_and_hands() -> None:
    cache = _cache()
    parents = np.asarray(
        [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        + [15] * 3
        + [20] * 15
        + [21] * 15,
        dtype=np.int64,
    )
    pose = np.zeros((55, 3), np.float32)
    pose[16:22, 1] = 0.5
    candidate = nlf_body_candidate(cache, 0, pose, parents, 0.75)
    reference = nlf_body_candidate(cache, 0, np.zeros_like(pose), parents, 0.0)
    reference_global = global_rotations(reference, parents)
    candidate_global = global_rotations(candidate, parents)
    np.testing.assert_allclose(candidate_global[20:22], reference_global[20:22], atol=1e-5)
    np.testing.assert_array_equal(candidate[22:55], reference[22:55])
    assert cache_pose_from_full(candidate).shape == (51, 3, 3)


def test_features_are_target_free_and_finite() -> None:
    cache = _cache()
    parents = np.asarray([-1] + [0] * 54, dtype=np.int64)
    candidate = nlf_body_candidate(cache, 0, _observation()["pose"], parents, 0.5)
    features = frame_features(cache, 0, _observation(), candidate)
    assert tuple(features) == FEATURE_COLUMNS
    assert all(np.isfinite(value) for value in features.values())
    assert all("target" not in name for name in features)


def test_velocity_features_are_normalized_to_15fps() -> None:
    cache = _cache()
    parents = np.asarray([-1] + [0] * 54, dtype=np.int64)
    candidate = nlf_body_candidate(cache, 0, _observation()["pose"], parents, 0.5)
    previous = cache_pose_from_full(candidate)
    cache.init_axis_angle[0, 15:21, 2] = 0.2
    candidate = nlf_body_candidate(cache, 0, _observation()["pose"], parents, 0.5)
    one_step = frame_features(
        cache,
        0,
        _observation(),
        candidate,
        previous_initializer=previous,
        elapsed_seconds=1.0 / 15.0,
    )
    two_steps = frame_features(
        cache,
        0,
        _observation(),
        candidate,
        previous_initializer=previous,
        elapsed_seconds=2.0 / 15.0,
    )
    assert one_step["initializer_velocity_arms_deg"] > 0
    np.testing.assert_allclose(
        two_steps["initializer_velocity_arms_deg"],
        one_step["initializer_velocity_arms_deg"] / 2,
    )
    assert two_steps["time_gap_reference_units"] == 2.0


def test_tals_ignores_small_noise() -> None:
    np.testing.assert_allclose(tals(np.asarray([0.01, 0.02, 0.05])), [0.0, 0.0, 0.03])


def test_nlf_observation_contract_ignores_device_and_manifest() -> None:
    base = {
        "model_sha256": "abc",
        "nlf_source_commit": "def",
        "manifests": ["external"],
        "settings": {
            "device": "cuda:0",
            "num_aug": 1,
            "detector_threshold": 0.3,
            "selection": "max_box_area_times_score",
        },
    }
    target = {
        **base,
        "manifests": ["target"],
        "settings": {**base["settings"], "device": "cuda:1"},
    }
    assert nlf_observation_contract(base) == nlf_observation_contract(target)


def test_viterbi_prefers_contiguous_benefit_and_base_on_tie() -> None:
    selected = viterbi_benefit_selection(
        np.asarray([0.0, -2.0, -2.0, 0.0]), margin=0.0, transition_penalty=0.25
    )
    # Switching twice costs more than retaining the candidate at neutral ends.
    np.testing.assert_array_equal(selected, [True, True, True, True])
    np.testing.assert_array_equal(
        viterbi_benefit_selection(np.zeros(3), 0.0, 0.0),
        [False, False, False],
    )


def test_viterbi_reduces_switch_cost_across_long_time_gap() -> None:
    delta = np.asarray([-1.0, 0.4])
    contiguous = viterbi_benefit_selection(delta, 0.0, 1.0)
    long_gap = viterbi_benefit_selection(delta, 0.0, 1.0, np.asarray([1.0, 0.1]))
    np.testing.assert_array_equal(contiguous, [True, True])
    np.testing.assert_array_equal(long_gap, [True, False])
