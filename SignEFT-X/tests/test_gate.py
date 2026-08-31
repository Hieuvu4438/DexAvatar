from signeft.gating.evidence_gate import FamilyDelta, accept_ubody
import math


def test_one_family_win_rejects():
    accepted, _, _, reason = accept_ubody([
        FamilyDelta("pose2d", -3.0, 1.0),
        FamilyDelta("nlf3d", 0.0, 1.0),
    ], changes_depth=False, geometry_ok=True, trust_ok=True)
    assert not accepted
    assert reason == "GATE_NOT_ENOUGH_WINS"


def test_depth_change_requires_3d_win():
    accepted, _, _, reason = accept_ubody([
        FamilyDelta("pose2d", -3.0, 1.0),
        FamilyDelta("hand_expert", -3.0, 1.0),
    ], changes_depth=True, geometry_ok=True, trust_ok=True)
    assert not accepted
    assert reason == "GATE_NO_3D_WIN"


def test_any_family_regression_rejects():
    accepted, _, losers, reason = accept_ubody([
        FamilyDelta("pose2d", -3.0, 1.0),
        FamilyDelta("nlf3d", -3.0, 1.0),
        FamilyDelta("dense_rgb", 2.0, 1.0),
    ], changes_depth=True, geometry_ok=True, trust_ok=True)
    assert not accepted
    assert losers == ["dense_rgb"]
    assert reason == "GATE_FAMILY_REGRESSION"


def test_two_family_with_3d_accepts():
    accepted, winners, losers, reason = accept_ubody([
        FamilyDelta("pose2d", -3.0, 1.0),
        FamilyDelta("nlf3d", -3.0, 1.0),
    ], changes_depth=True, geometry_ok=True, trust_ok=True)
    assert accepted
    assert winners == ["pose2d", "nlf3d"]
    assert not losers
    assert reason == "ACCEPTED"


def test_nonfinite_evidence_rejects_without_exception():
    accepted, winners, losers, reason = accept_ubody([
        FamilyDelta("pose2d", math.nan, 1.0),
        FamilyDelta("nlf3d", -3.0, 1.0),
    ], changes_depth=True, geometry_ok=True, trust_ok=True)
    assert not accepted
    assert not winners and not losers
    assert reason == "GATE_NONFINITE_EVIDENCE"


def test_negative_noise_rejects_without_exception():
    accepted, _, _, reason = accept_ubody([
        FamilyDelta("pose2d", -3.0, -1.0),
        FamilyDelta("nlf3d", -3.0, 1.0),
    ], changes_depth=True, geometry_ok=True, trust_ok=True)
    assert not accepted
    assert reason == "GATE_INVALID_SIGMA"
