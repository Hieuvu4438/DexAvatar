import numpy as np

from cusp_sl.prepare_wilor_caches import hand_observation_metadata


def _hand(confidence, center, size):
    return {
        "is_right": 1.0,
        "detector_confidence": confidence,
        "box_center": np.asarray(center, dtype=np.float32),
        "box_size": size,
    }


def test_hand_observation_metadata_selects_confident_box_and_marks_duplicates():
    low = _hand(0.4, (20, 20), 40)
    high = _hand(0.9, (50, 50), 40)
    hand, scale, truncation, duplicate = hand_observation_metadata(
        [low, high], is_right=True, width=100, height=80
    )
    assert hand is high
    assert scale == 0.4
    assert truncation == 0.0
    assert duplicate


def test_hand_observation_metadata_reports_dropout_and_truncation():
    hand, scale, truncation, duplicate = hand_observation_metadata(
        [], is_right=False, width=100, height=80
    )
    assert (hand, scale, truncation, duplicate) == (None, 0.0, 1.0, False)

    hand, _, truncation, _ = hand_observation_metadata(
        [_hand(0.9, (5, 40), 40)], is_right=True, width=100, height=80
    )
    assert hand is not None
    assert truncation > 0.0
