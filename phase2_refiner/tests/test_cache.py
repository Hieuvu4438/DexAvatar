from pathlib import Path

import numpy as np
import pytest

from phase2_refiner.data.cache_schema import (
    SCHEMA_VERSION,
    CacheClip,
    load_cache_clip,
    save_cache_clip,
)
from phase2_refiner.data.dataset import (
    REPROJECTION_RESIDUAL_2D,
    REPROJECTION_RESIDUAL_SCALE,
    TOKEN_FEATURE_DIM,
    TOKEN_FEATURE_DIM_WITH_REPROJECTION,
    features_from_clip,
)
from phase2_refiner.data.materialize_exact_a1_cache import (
    A1_COMPONENT_SHA256,
    G1_EVALUATION_SHA256,
    LOCKED_VIEW_SHA256,
    STACK_ID,
    validate_exact_a1_provenance,
    validate_exact_a1_result_binding,
)


def make_clip(frames: int = 5) -> CacheClip:
    return CacheClip(
        clip_id="synthetic",
        frame_names=np.asarray([f"low_{i:03d}" for i in range(frames)]),
        init_axis_angle=np.zeros((frames, 51, 3), np.float32),
        observation_features=np.zeros((frames, 51, 8), np.float32),
        keypoints_2d=np.zeros((frames, 51, 2), np.float32),
        keypoint_valid=np.zeros((frames, 51), bool),
        refine_mask=np.ones(51, bool),
        betas=np.zeros(10, np.float32),
        global_orient=np.zeros((frames, 3), np.float32),
        transl=np.zeros((frames, 3), np.float32),
        jaw_pose=np.zeros((frames, 3), np.float32),
        leye_pose=np.zeros((frames, 3), np.float32),
        reye_pose=np.zeros((frames, 3), np.float32),
        expression=np.zeros((frames, 10), np.float32),
        source_paths=np.asarray([f"/source/{i}.pkl" for i in range(frames)]),
    )


def test_cache_round_trip(tmp_path: Path) -> None:
    clip = make_clip()
    path = tmp_path / "clip.npz"
    save_cache_clip(path, clip)
    loaded = load_cache_clip(path)
    assert loaded.clip_id == clip.clip_id
    assert np.array_equal(loaded.frame_names, clip.frame_names)
    assert np.array_equal(loaded.init_axis_angle, clip.init_axis_angle)
    assert loaded.target_axis_angle is None
    assert loaded.frame_numbers.tolist() == list(range(5))
    assert loaded.torso_to_camera.shape == (5, 4, 4)
    with np.load(path, allow_pickle=False) as data:
        assert int(data["schema_version"]) == SCHEMA_VERSION


def test_partial_rotation_target_mask_round_trip(tmp_path: Path) -> None:
    clip = make_clip()
    clip.target_axis_angle = clip.init_axis_angle.copy()
    clip.target_rotation_valid = np.zeros((5, 51), dtype=bool)
    clip.target_rotation_valid[:, 21:36] = True
    path = tmp_path / "partial.npz"
    save_cache_clip(path, clip)
    loaded = load_cache_clip(path)
    assert np.array_equal(loaded.target_rotation_valid, clip.target_rotation_valid)


def test_missing_observations_produce_finite_explicitly_masked_tokens() -> None:
    clip = make_clip()
    clip.validate()
    features, _ = features_from_clip(clip)
    assert features.shape == (5, 51, TOKEN_FEATURE_DIM)
    assert np.isfinite(features.numpy()).all()
    assert np.count_nonzero(features[..., 26:28].numpy()) == 0


def test_v3_reprojection_layout_is_opt_in_and_backward_compatible() -> None:
    clip = make_clip()
    clip.keypoint_valid[:] = True
    clip.reprojection_residual_2d = np.full((5, 51, 2), 0.25, np.float32)
    legacy, _ = features_from_clip(clip)
    enriched, _ = features_from_clip(
        clip, input_dim=TOKEN_FEATURE_DIM_WITH_REPROJECTION
    )
    assert legacy.shape[-1] == TOKEN_FEATURE_DIM
    assert enriched.shape[-1] == TOKEN_FEATURE_DIM_WITH_REPROJECTION
    np.testing.assert_allclose(
        enriched[..., REPROJECTION_RESIDUAL_2D].numpy(),
        0.25 * REPROJECTION_RESIDUAL_SCALE,
    )


def test_cache_rejects_non_finite() -> None:
    clip = make_clip()
    clip.init_axis_angle[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        clip.validate()


def test_exact_a1_provenance_is_tied_to_locked_g1_evidence() -> None:
    payload = {
        "stack_id": STACK_ID,
        "locked_view_manifest_sha256": LOCKED_VIEW_SHA256,
        "g1_evaluation_sha256": G1_EVALUATION_SHA256,
        "component_hashes": A1_COMPONENT_SHA256,
    }
    validate_exact_a1_provenance(payload)
    payload["locked_view_manifest_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="locked_view_manifest_sha256"):
        validate_exact_a1_provenance(payload)


def test_exact_a1_result_hashes_are_bound_to_cached_sources() -> None:
    paths = np.asarray(["/tmp/frame_1.pkl", "/tmp/frame_2.pkl"])
    digests = np.asarray(["a" * 64, "b" * 64])
    result_hashes = {"frame_1.pkl": "a" * 64, "frame_2.pkl": "b" * 64}
    validate_exact_a1_result_binding(paths, digests, result_hashes)
    result_hashes["frame_2.pkl"] = "c" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_exact_a1_result_binding(paths, digests, result_hashes)
