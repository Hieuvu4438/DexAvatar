import numpy as np
import torch

from signeft.observations.heatmaps import distribution_payload
from signeft.observations.nlf import tta_covariance, undo_image_rotation


def test_pose_payload_is_real_distribution_contract():
    raw = torch.zeros(4, 256, 192)
    raw[2, 120, 80] = 2.0
    payload = distribution_payload(
        raw, torch.tensor([2]), ("left_elbow",), np.eye(3, dtype=np.float32)
    )
    assert payload["heatmap_q"].shape == (1, 64, 48)
    assert payload["heatmap_q"].dtype == np.uint8
    assert payload["valid"].tolist() == [True]
    assert np.isfinite(payload["cov2d"]).all()
    assert payload["heatmap_q"].max() == 255


def test_nlf_rotation_tta_undo_matches_opencv_convention():
    angle = 10.0
    theta = np.deg2rad(angle)
    forward = np.asarray(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    original = np.asarray([[1.0, 2.0, 3.0], [-2.0, 1.0, 4.0]], dtype=np.float32)
    augmented = original.copy()
    augmented[:, :2] = original[:, :2] @ forward.T
    assert np.allclose(undo_image_rotation(augmented, angle), original, atol=1e-6)


def test_nlf_covariance_uses_explicit_augmentation_axis():
    joints = np.zeros((5, 2, 3), dtype=np.float32)
    joints[:, 1, 0] = np.arange(5, dtype=np.float32)
    covariance = tta_covariance(joints, np.ones((5, 2), dtype=bool))
    assert np.allclose(covariance[0], 0)
    assert np.isclose(covariance[1, 0, 0], 2.5)
    assert np.allclose(covariance[1, 1:, :], 0)

