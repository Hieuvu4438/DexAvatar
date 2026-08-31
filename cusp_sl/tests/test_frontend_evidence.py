import numpy as np

from cusp_sl.evaluate_frontend_evidence import visible_motion, weighted_huber


def test_weighted_huber_ignores_zero_confidence():
    residual = np.asarray([[[0.03, 0.0], [10.0, 0.0]]], dtype=np.float32)
    confidence = np.asarray([[1.0, 0.0]], dtype=np.float32)
    value, weight = weighted_huber(residual, confidence, 0.03)
    assert np.isclose(value, 0.015)
    assert weight == 1.0


def test_visible_motion_is_residual_difference_with_adjacent_confidence():
    residual = np.asarray(
        [[[0.0, 0.0]], [[0.2, 0.0]], [[0.5, 0.0]]], dtype=np.float32
    )
    confidence = np.asarray([[1.0], [0.5], [1.0]], dtype=np.float32)
    value, weight = visible_motion(residual, confidence)
    assert np.isclose(value, 0.25)
    assert np.isclose(weight, 1.0)
