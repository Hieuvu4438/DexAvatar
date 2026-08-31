import numpy as np

from signpccx.optimization.identity import farthest_point_indices, huber_location, pose_diversity_feature


def test_huber_location_rejects_shape_outlier():
    values = np.zeros((9, 10), dtype=np.float32)
    values[:8] = 0.25
    values[8] = 100.0
    estimate = huber_location(values)
    assert np.max(np.abs(estimate - 0.25)) < 1e-2


def test_farthest_point_selection_is_deterministic_and_unique():
    features = np.asarray([[0.0], [1.0], [2.0], [10.0], [11.0]])
    first = farthest_point_indices(features, 3)
    second = farthest_point_indices(features, 3)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 3


def test_pose_feature_is_finite():
    joints = np.zeros((55, 3), dtype=np.float32)
    joints[16] = [-0.2, 1.0, 0.0]
    joints[17] = [0.2, 1.0, 0.0]
    joints[18] = [-0.4, 0.8, 0.1]
    joints[19] = [0.4, 0.8, 0.1]
    joints[20] = [-0.5, 0.5, 0.2]
    joints[21] = [0.5, 0.5, 0.2]
    feature = pose_diversity_feature(joints)
    assert feature.shape == (9,)
    assert np.isfinite(feature).all()
