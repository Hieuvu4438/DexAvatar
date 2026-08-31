import torch

from cusp_sl.evaluate_prediction_development import rotation_group_metrics
from cusp_sl.geometry import axis_angle_to_matrix


def test_rotation_group_metrics_keeps_base_in_oracle_candidates():
    base = axis_angle_to_matrix(torch.zeros(2, 51, 3))
    target_axis = torch.zeros(2, 51, 3)
    target_axis[:, 21:, 0] = 0.2
    target = axis_angle_to_matrix(target_axis)
    selected = base.clone()
    better = target.clone()
    candidates = torch.stack((base, better))
    metrics = rotation_group_metrics(
        base,
        selected,
        candidates,
        target,
        torch.ones(2, 51, dtype=torch.bool),
        torch.ones(51, dtype=torch.bool),
    )
    assert metrics["selected_hands_degrees"] > 0.0
    assert metrics["oracle_hands_degrees"] < 1e-3
    assert metrics["base_body_degrees"] < 1e-3


def test_rotation_group_metrics_excludes_invalid_candidate_from_oracle():
    base = axis_angle_to_matrix(torch.zeros(1, 51, 3))
    target = axis_angle_to_matrix(torch.full((1, 51, 3), 0.1))
    invalid_but_best = target.clone()
    candidates = torch.stack((base, invalid_but_best))
    metrics = rotation_group_metrics(
        base,
        base,
        candidates,
        target,
        torch.ones(1, 51, dtype=torch.bool),
        torch.ones(51, dtype=torch.bool),
        torch.tensor([True, False]),
    )
    assert abs(metrics["oracle_overall_degrees"] - metrics["base_overall_degrees"]) < 1e-6
