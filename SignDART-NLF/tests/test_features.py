import numpy as np

from signdart.features import FEATURE_NAMES, candidate_features


def test_candidate_features_are_finite_and_match_schema():
    joints = np.zeros((2, 55, 3), dtype=np.float64)
    for side_ids in ((13, 16, 18, 20), (14, 17, 19, 21)):
        joints[:, list(side_ids), 0] = np.arange(4)[None]
    joints[1, [16, 18, 20], 2] = [0.1, 0.2, 0.3]
    nlf = joints[0] * 1000.0
    metrics = np.zeros((2, 5))
    features = candidate_features(
        joints, metrics, nlf, nlf, np.ones(55), "left",
        np.asarray([0, 0, 100, 200, 0.9]), 500, 300,
    )
    assert features.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
