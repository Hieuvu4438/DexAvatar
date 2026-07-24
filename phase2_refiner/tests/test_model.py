import torch

from phase2_refiner.data.corruptions import apply_burst_corruption
from phase2_refiner.data.dataset import TOKEN_FEATURE_DIM
from phase2_refiner.geometry.rotations import axis_angle_to_matrix
from phase2_refiner.infer import _apply_safety_fallback, _predict_sequence
from phase2_refiner.losses import RefinerLoss
from phase2_refiner.models import WholeSequenceRefiner


def small_model(
    max_frames: int = 8, causal: bool = False, predict_uncertainty: bool = False
) -> WholeSequenceRefiner:
    return WholeSequenceRefiner(
        input_dim=TOKEN_FEATURE_DIM,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        max_frames=max_frames,
        dropout=0.0,
        causal=causal,
        predict_uncertainty=predict_uncertainty,
    ).eval()


def test_zero_initialized_model_is_identity() -> None:
    model = small_model()
    features = torch.randn(2, 8, 51, TOKEN_FEATURE_DIM)
    initial = axis_angle_to_matrix(torch.randn(2, 8, 51, 3) * 0.1)
    frame_valid = torch.ones(2, 8, dtype=torch.bool)
    refine_mask = torch.ones(2, 51, dtype=torch.bool)
    output = model(features, initial, frame_valid, refine_mask)
    assert torch.equal(output["matrix"], initial)
    assert torch.count_nonzero(output["raw_delta"]) == 0


def test_padding_does_not_change_valid_outputs() -> None:
    model = small_model()
    features = torch.randn(1, 8, 51, TOKEN_FEATURE_DIM)
    initial = axis_angle_to_matrix(torch.randn(1, 8, 51, 3) * 0.1)
    mask4 = torch.tensor([[True] * 4 + [False] * 4])
    refine_mask = torch.ones(1, 51, dtype=torch.bool)
    padded = model(features, initial, mask4, refine_mask)["matrix"][:, :4]
    short = model(features[:, :4], initial[:, :4], mask4[:, :4], refine_mask)["matrix"]
    assert torch.equal(padded, short)


def test_burst_corruption_marks_missing_interval() -> None:
    torch.manual_seed(3)
    features = torch.randn(2, 12, 51, TOKEN_FEATURE_DIM)
    initial = axis_angle_to_matrix(torch.zeros(2, 12, 51, 3))
    frame_valid = torch.ones(2, 12, dtype=torch.bool)
    corrupted, matrix, mask = apply_burst_corruption(
        features,
        initial,
        frame_valid,
        probability=1.0,
        min_duration=4,
        max_duration=4,
        modes=["left_hand"],
    )
    assert mask.sum() >= 4 * 9
    assert torch.all(corrupted[..., 20][mask] == 1.0)
    assert not torch.equal(matrix, initial)


def test_sliding_window_identity() -> None:
    model = small_model(max_frames=8)
    features = torch.randn(17, 51, TOKEN_FEATURE_DIM)
    initial = axis_angle_to_matrix(torch.randn(17, 51, 3) * 0.1)
    output = _predict_sequence(
        model, features, initial, torch.ones(51, dtype=torch.bool), torch.device("cpu")
    )
    assert torch.allclose(output["matrix"], initial, atol=1e-6)


def test_training_step_has_finite_gradients() -> None:
    model = small_model()
    model.train()
    features = torch.randn(1, 8, 51, TOKEN_FEATURE_DIM)
    initial = axis_angle_to_matrix(torch.randn(1, 8, 51, 3) * 0.1)
    target = axis_angle_to_matrix(torch.randn(1, 8, 51, 3) * 0.1)
    frame_valid = torch.ones(1, 8, dtype=torch.bool)
    refine_mask = torch.ones(1, 51, dtype=torch.bool)
    prediction = model(features, initial, frame_valid, refine_mask)
    losses = RefinerLoss()(
        prediction, initial, target, frame_valid, refine_mask, torch.ones(1, 8, 51)
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert model.delta_head.weight.grad is not None
    assert torch.isfinite(model.delta_head.weight.grad).all()
    assert torch.count_nonzero(model.delta_head.weight.grad) > 0


def test_safety_fallback_is_groupwise() -> None:
    initial = axis_angle_to_matrix(torch.zeros(2, 51, 3))
    output = initial.clone()
    output[0, 25] = torch.nan
    safe, fallback = _apply_safety_fallback(output, initial)
    assert fallback[0].tolist() == [False, True, False]
    assert not fallback[1].any()
    assert torch.equal(safe[0, 21:36], initial[0, 21:36])
    assert torch.isfinite(safe).all()


def test_causal_mode_cannot_see_future_features() -> None:
    torch.manual_seed(11)
    causal = small_model(causal=True)
    with torch.no_grad():
        causal.delta_head.weight.normal_(std=0.02)
    features = torch.randn(1, 8, 51, TOKEN_FEATURE_DIM)
    features[..., 29] = 1.0
    changed = features.clone()
    changed[:, 4:] += torch.randn_like(changed[:, 4:]) * 10.0
    initial = axis_angle_to_matrix(torch.randn(1, 8, 51, 3) * 0.1)
    valid = torch.ones(1, 8, dtype=torch.bool)
    refine = torch.ones(1, 51, dtype=torch.bool)
    first = causal(features, initial, valid, refine)["matrix"][:, :4]
    second = causal(changed, initial, valid, refine)["matrix"][:, :4]
    assert torch.allclose(first, second, atol=1e-6)


def test_u1_returns_bounded_observation_uncertainty() -> None:
    model = small_model(predict_uncertainty=True)
    features = torch.randn(1, 8, 51, TOKEN_FEATURE_DIM)
    initial = axis_angle_to_matrix(torch.zeros(1, 8, 51, 3))
    output = model(
        features,
        initial,
        torch.ones(1, 8, dtype=torch.bool),
        torch.ones(1, 51, dtype=torch.bool),
    )
    assert output["log_variance"].shape == (1, 8, 51, 1)
    assert output["observation_log_variance"].shape == (1, 8, 51, 2)
    assert torch.all((output["reliability"] >= 0) & (output["reliability"] <= 1))
