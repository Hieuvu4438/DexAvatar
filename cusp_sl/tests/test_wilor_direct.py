import pytest

from cusp_sl.evaluate_wilor_direct_development import select_detection


def test_select_detection_uses_side_and_highest_detector_confidence():
    hands = [
        {"is_right": 0.0, "detector_confidence": 0.7},
        {"is_right": 1.0, "detector_confidence": 0.8},
        {"is_right": 1.0, "detector_confidence": 0.9},
    ]
    assert select_detection(hands, is_right=False) is hands[0]
    assert select_detection(hands, is_right=True) is hands[2]


def test_select_detection_requires_v3_confidence_and_allows_missing_side():
    assert select_detection([], is_right=True) is None
    with pytest.raises(ValueError, match="v3 detector_confidence"):
        select_detection([{"is_right": 1.0}], is_right=True)
