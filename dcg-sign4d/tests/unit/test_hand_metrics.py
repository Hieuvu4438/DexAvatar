import numpy as np

from dcg_sign4d.evaluation.hand_metrics import HandPlacementMetrics


def fixture():
    # V=8: pelvis vertices 0/1, left 2/3, right 4/5, body 0..7.
    regressor = np.zeros((22, 8), dtype=np.float64)
    regressor[1, 0] = 1
    regressor[2, 1] = 1
    regressor[20, 2:4] = 0.5
    regressor[21, 4:6] = 0.5
    target = np.zeros((8, 3), dtype=np.float64)
    target[0, 0], target[1, 0] = -0.1, 0.1
    target[2:4, 0] = [-0.6, -0.5]
    target[4:6, 0] = [0.5, 0.6]
    target[6:, 1] = [0.2, 0.3]
    evaluator = HandPlacementMetrics(
        regressor,
        np.array([2, 3]),
        np.array([4, 5]),
        np.arange(8),
    )
    return evaluator, target


def test_global_translation_is_removed_by_all_alignments():
    evaluator, target = fixture()
    result = evaluator.evaluate_frame(target + np.array([3.0, -2.0, 8.0]), target)
    assert max(abs(value) for value in result.values()) < 1e-9


def test_rigid_hand_placement_error_is_not_mislabeled_articulation():
    evaluator, target = fixture()
    source = target.copy()
    source[[2, 3], 1] += 0.05
    result = evaluator.evaluate_frame(source, target)
    assert np.isclose(result["root_aligned_left_hand_pve_mm"], 50.0)
    assert result["wrist_aligned_left_hand_pve_mm"] < 1e-9
    assert result["legacy_region_tr_left_hand_pve_mm"] < 1e-9


def test_local_finger_articulation_survives_wrist_alignment():
    evaluator, target = fixture()
    source = target.copy()
    source[2, 1] += 0.05
    result = evaluator.evaluate_frame(source, target)
    assert result["wrist_aligned_left_hand_pve_mm"] > 0
