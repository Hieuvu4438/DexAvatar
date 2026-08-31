import torch

from cusp_sl.inference import (
    candidate_seed,
    center_weights,
    generated_candidate_count,
    random_candidate_index,
    starts,
    variant_uses_energy,
)
from cusp_sl.models import ReliabilityCalibrator, SelectiveResidualFlow
from cusp_sl.geometry import axis_angle_to_matrix
from cusp_sl.train_deterministic import deterministic_residual


def test_model_shapes_and_sample_identity():
    b, t, j, f = 2, 8, 51, 45
    features = torch.randn(b, t, j, f)
    q = ReliabilityCalibrator(f, hidden_size=32, temporal_layers=1)
    logits = q(features)
    assert logits.shape == (b, t, j)
    flow = SelectiveResidualFlow(f + 1, hidden_size=48, layers=1, heads=4)
    condition = torch.cat((features, logits.sigmoid()[..., None]), dim=-1)
    base = axis_angle_to_matrix(torch.randn(b, t, j, 3) * 0.1)
    valid = torch.ones(b, t, dtype=torch.bool)
    _, rotation = flow.sample(condition, base, torch.zeros_like(logits), valid, steps=2)
    torch.testing.assert_close(rotation, base, rtol=0, atol=0)
    point = deterministic_residual(flow, condition, valid)
    assert point.shape == (b, t, j, 3)


def test_handflow_compatible_window_contract():
    assert starts(length=16, window=16, overlap=2) == [0]
    assert starts(length=17, window=16, overlap=2) == [0, 14]
    assert starts(length=29, window=16, overlap=2) == [0, 14]
    assert starts(length=32, window=16, overlap=2) == [0, 14, 28]
    weights = center_weights(5, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(
        weights, torch.tensor([0.01, 0.5, 1.0, 0.5, 0.01])
    )


def test_candidate_count_matches_declared_ablation_budget():
    assert generated_candidate_count("a4_k1", "flow", 4) == 1
    assert generated_candidate_count("a3_deterministic", "deterministic", 4) == 1
    assert generated_candidate_count("a5_random", "flow", 4) == 4
    assert generated_candidate_count("a7_geometry", "flow", 4) == 4
    assert not variant_uses_energy("a4_k1")
    assert not variant_uses_energy("a5_random")
    assert variant_uses_energy("a7_geometry")
    first = random_candidate_index("clip-a", 42, 4)
    assert first == random_candidate_index("clip-a", 42, 4)
    assert 1 <= first <= 4
    assert candidate_seed(42, "clip-a", 0) == candidate_seed(42, "clip-a", 0)
    assert candidate_seed(42, "clip-a", 0) != candidate_seed(42, "clip-b", 0)


def test_handflow_stride_grid_never_exceeds_declared_overlap():
    length, window, overlap = 32, 16, 2
    locations = starts(length, window, overlap)
    coverage = torch.zeros(length, dtype=torch.int64)
    for start in locations:
        coverage[start : min(start + window, length)] += 1
    assert coverage.min() == 1
    assert coverage.max() == 2
    assert int((coverage == 2).sum()) == 2 * overlap
