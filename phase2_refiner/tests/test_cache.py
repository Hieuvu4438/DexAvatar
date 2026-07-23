from pathlib import Path

import numpy as np
import pytest

from phase2_refiner.data.cache_schema import CacheClip, load_cache_clip, save_cache_clip


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


def test_cache_rejects_non_finite() -> None:
    clip = make_clip()
    clip.init_axis_angle[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        clip.validate()
