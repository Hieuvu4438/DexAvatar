import numpy as np

from signdart.selector import (
    branch_scores,
    compose_selected_pose,
    conservative_consensus_decision,
)


def test_branch_score_selects_matching_bone_directions_and_composes_sides():
    base = np.zeros((22, 3), dtype=np.float64)
    ids = (13, 16, 18, 20)
    base[list(ids)] = np.asarray([
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
        [2.0, 0.0, 1.0], [3.0, 0.0, 1.0],
    ])
    alternate = base.copy()
    alternate[list(ids)] = np.asarray([
        [0.0, 0.0, 1.0], [0.0, 1.0, 1.0],
        [0.0, 2.0, 1.0], [0.0, 3.0, 1.0],
    ])
    uncertainty = np.ones(55)
    nlf = np.zeros((55, 3))
    nlf[list(ids)] = alternate[list(ids)] * 1000.0
    scores = branch_scores(
        np.stack((base, alternate)), nlf, nlf, uncertainty, "left"
    )
    assert int(np.argmin(scores)) == 1

    pose = np.zeros((21, 3), dtype=np.float32)
    left = np.ones((21, 3), dtype=np.float32)
    right = np.full((21, 3), 2.0, dtype=np.float32)
    composed = compose_selected_pose(pose, left, right).reshape(21, 3)
    assert np.all(composed[[12, 15, 17, 19]] == 1.0)
    assert np.all(composed[[13, 16, 18, 20]] == 2.0)


def _consensus_fixture():
    incumbent = np.zeros((55, 3), dtype=np.float64)
    alternate = incumbent.copy()
    ids = (13, 16, 18, 20)
    incumbent[list(ids)] = np.asarray([
        [0.0, 0.0, 0.0], [100.0, 0.0, 0.0],
        [200.0, 0.0, 0.0], [300.0, 0.0, 0.0],
    ])
    alternate[list(ids)] = np.asarray([
        [0.0, 0.0, 0.0], [0.0, 100.0, 0.0],
        [0.0, 200.0, 0.0], [0.0, 300.0, 0.0],
    ])
    uncertainty = np.full(55, 1.0, dtype=np.float64)
    return np.stack((incumbent, alternate)), alternate, uncertainty


def test_conservative_consensus_accepts_high_confidence_agreement():
    candidates, evidence, uncertainty = _consensus_fixture()
    selected, diagnostics = conservative_consensus_decision(
        candidates, evidence, evidence, uncertainty, "left"
    )
    assert selected == 1
    assert diagnostics["reason"] == "consensus_branch_accepted"


def test_conservative_consensus_abstains_when_estimators_disagree():
    candidates, alternate, uncertainty = _consensus_fixture()
    selected, diagnostics = conservative_consensus_decision(
        candidates, candidates[0], alternate, uncertainty, "left"
    )
    assert selected == 0
    assert diagnostics["reason"] == "estimator_branch_disagreement"


def test_conservative_consensus_abstains_when_uncertainty_is_too_large():
    candidates, evidence, _ = _consensus_fixture()
    selected, diagnostics = conservative_consensus_decision(
        candidates, evidence, evidence, np.full(55, 1000.0), "left"
    )
    assert selected == 0
    assert diagnostics["reason"] == "insufficient_likelihood_gain"
