import json
from pathlib import Path

import numpy as np

from cusp_sl.prepare_targetless_development import strip_targets
from phase2_refiner.data.cache_schema import CacheClip


def _clip():
    frames = 2
    return CacheClip(
        clip_id="clip",
        frame_names=np.asarray(["0", "1"]),
        init_axis_angle=np.zeros((frames, 51, 3), np.float32),
        observation_features=np.zeros((frames, 51, 8), np.float32),
        keypoints_2d=np.zeros((frames, 51, 2), np.float32),
        keypoint_valid=np.ones((frames, 51), bool),
        refine_mask=np.ones(51, bool),
        betas=np.zeros(10, np.float32),
        global_orient=np.zeros((frames, 3), np.float32),
        transl=np.zeros((frames, 3), np.float32),
        jaw_pose=np.zeros((frames, 3), np.float32),
        leye_pose=np.zeros((frames, 3), np.float32),
        reye_pose=np.zeros((frames, 3), np.float32),
        expression=np.zeros((frames, 10), np.float32),
        source_paths=np.asarray(["a", "b"]),
        target_axis_angle=np.ones((frames, 51, 3), np.float32),
        target_rotation_valid=np.ones((frames, 51), bool),
        target_joint_positions=np.ones((frames, 51, 3), np.float32),
        target_joint_valid=np.ones((frames, 51), bool),
        target_quality=np.ones((frames, 51), np.float32),
    )


def test_strip_targets_removes_all_target_arrays_and_quality():
    stripped = strip_targets(
        _clip(), source_path=Path("source.npz"), source_hash="a" * 64
    )
    assert stripped.target_axis_angle is None
    assert stripped.target_rotation_valid is None
    assert stripped.target_joint_positions is None
    assert stripped.target_joint_valid is None
    assert not stripped.target_quality.any()
    contract = json.loads(stripped.metadata_json)["development_inference_contract"]
    assert contract["target_arrays_removed"]
