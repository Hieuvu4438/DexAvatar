from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from signal4d_external.materialize_nlf_v2 import _selected_params


def test_nlf_body_fusion_preserves_v2h_hands_and_updates_body_splits() -> None:
    left = np.arange(45, dtype=np.float32).reshape(1, 45) / 100.0
    right = -left.copy()
    baseline = {
        "body_pose": np.zeros((1, 63), np.float32),
        "body_pose_fore": np.zeros((1, 45), np.float32),
        "body_pose_op": np.zeros((1, 18), np.float32),
        "left_hand_pose": left,
        "right_hand_pose": right,
        "global_orient": np.asarray([[0.1, 0.2, 0.3]], np.float32),
        "betas": np.zeros((1, 10), np.float32),
    }
    rotvec = np.zeros((55, 3), np.float32)
    rotvec[1:22, 1] = 0.2
    candidate = Rotation.from_rotvec(rotvec).as_matrix().astype(np.float32)
    result = _selected_params(baseline, candidate)
    np.testing.assert_array_equal(result["left_hand_pose"], left)
    np.testing.assert_array_equal(result["right_hand_pose"], right)
    np.testing.assert_array_equal(result["global_orient"], baseline["global_orient"])
    np.testing.assert_allclose(result["body_pose"].reshape(21, 3)[:, 1], 0.2)
    np.testing.assert_array_equal(result["body_pose_fore"], result["body_pose"][:, :45])
    np.testing.assert_array_equal(result["body_pose_op"], result["body_pose"][:, 45:])
