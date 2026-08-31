from copy import deepcopy

import pytest

from dcg_sign4d.contact.agreement import AgreementThresholds, audit_double_annotations


def records():
    result = []
    for event_id, edge, interval in (
        ("e0", ["left_tip", "face"], (2, 3, 5, 6)),
        ("e1", ["right_tip", "left_tip"], (10, 11, 13, 14)),
    ):
        for annotator, offset in (("ann_a", 0), ("ann_b", 1)):
            onset, hold_start, hold_end, release = interval
            result.append(
                {
                    "event_id": event_id,
                    "clip_id": "clip",
                    "edge": edge,
                    "onset_frame": onset + offset,
                    "hold_start": hold_start + offset,
                    "hold_end": hold_end + offset,
                    "release_frame": release + offset,
                    "uncertain": False,
                    "annotator_id": annotator,
                    "confidence": 0.9,
                }
            )
    return result


def thresholds():
    return AgreementThresholds(
        minimum_events=2,
        minimum_edge_agreement=1.0,
        minimum_mean_interval_iou=0.6,
        maximum_mean_boundary_error_frames=1.0,
    )


def test_double_annotation_agreement_passes_frozen_thresholds():
    report = audit_double_annotations(records(), thresholds())
    assert report["agreement_gate_pass"] is True
    assert report["edge_agreement"] == 1.0
    assert report["mean_boundary_error_frames"] == 1.0
    assert report["matched_edge_events"] == 2


def test_edge_disagreement_fails_and_same_annotator_is_rejected():
    mismatched = deepcopy(records())
    mismatched[1]["edge"] = ["left_tip", "torso"]
    report = audit_double_annotations(mismatched, thresholds())
    assert report["agreement_gate_pass"] is False
    assert report["checks"]["edge_agreement"] is False

    duplicate = deepcopy(records())
    duplicate[1]["annotator_id"] = "ann_a"
    with pytest.raises(ValueError, match="two distinct annotators"):
        audit_double_annotations(duplicate, thresholds())
