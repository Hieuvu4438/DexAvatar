import numpy as np

from cusp_sl.geometry import axis_angle_to_matrix, geodesic_distance
from cusp_sl.prepare_wilor_predictions import wilor_candidate
from cusp_sl.retargeting import (
    mirror_canonical_right_rotations,
    smplx_body_world_rotations,
)


def _hand(is_right: bool, confidence: float, angle: float) -> dict:
    fingers = axis_angle_to_matrix(np_to_tensor(np.full((15, 3), angle, np.float32)))
    wrist = axis_angle_to_matrix(np_to_tensor(np.asarray([angle, 0.0, 0.0])))
    return {
        "is_right": float(is_right),
        "detector_confidence": confidence,
        "pred_mano_pose_rotmat": fingers.numpy(),
        "pred_mano_global_orient_rotmat": wrist.numpy().reshape(1, 3, 3),
    }


def np_to_tensor(value):
    import torch

    return torch.from_numpy(np.asarray(value, dtype=np.float32))


def test_wilor_candidate_fallback_and_highest_confidence_selection():
    base = np.zeros((2, 51, 3), np.float32)
    root = np.zeros((2, 3), np.float32)
    records = [
        {"hands": [_hand(True, 0.5, 0.1), _hand(True, 0.9, 0.3)]},
        {"hands": []},
    ]
    selected, detected = wilor_candidate(
        base, root, records, geometric_wrist_alignment=False
    )
    assert detected == {"left": 0, "right": 1}
    np.testing.assert_allclose(selected[1], base[1], atol=1e-6)
    expected = np.full((15, 3), 0.3, np.float32)
    np.testing.assert_allclose(selected[0, 36:51], expected, atol=1e-5)


def test_wilor_candidate_geometric_left_alignment_undoes_mirror():
    base = np.zeros((1, 51, 3), np.float32)
    root = np.zeros((1, 3), np.float32)
    hand = _hand(False, 0.8, 0.25)
    selected, detected = wilor_candidate(
        base, root, [{"hands": [hand]}], geometric_wrist_alignment=True
    )
    rotations = axis_angle_to_matrix(np_to_tensor(selected))
    world = smplx_body_world_rotations(
        axis_angle_to_matrix(np_to_tensor(root)), rotations[:, :21]
    )
    canonical = np_to_tensor(hand["pred_mano_global_orient_rotmat"]).reshape(3, 3)
    target = mirror_canonical_right_rotations(canonical)
    assert geodesic_distance(world[0, 19], target) < 1e-5
    assert detected == {"left": 1, "right": 0}
