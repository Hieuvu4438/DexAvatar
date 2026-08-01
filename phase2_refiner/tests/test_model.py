import torch

from phase2_refiner.data.corruptions import apply_burst_corruption
from phase2_refiner.data.dataset import (
    TOKEN_FEATURE_DIM,
    TOKEN_FEATURE_DIM_WITH_REPROJECTION,
)
from phase2_refiner.geometry.rotations import axis_angle_to_matrix, geodesic_distance
from phase2_refiner.infer import _apply_safety_fallback, _predict_sequence
from phase2_refiner.losses import RefinerLoss
from phase2_refiner.models import WholeSequenceRefiner
from phase2_refiner.models.pretrained import load_compatible_initialization


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


def test_spatial_initialization_expands_append_only_input_features(tmp_path) -> None:
    source = small_model()
    with torch.no_grad():
        source.token_embedding.input_projection.weight.normal_()
    checkpoint = tmp_path / "spatial.pt"
    torch.save({"model": source.state_dict()}, checkpoint)

    target = WholeSequenceRefiner(
        input_dim=TOKEN_FEATURE_DIM_WITH_REPROJECTION,
        use_reprojection_skip=True,
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        max_frames=8,
        dropout=0.0,
    )
    provenance = load_compatible_initialization(target, checkpoint)
    source_weight = source.token_embedding.input_projection.weight
    target_weight = target.token_embedding.input_projection.weight

    assert torch.equal(target_weight[:, :TOKEN_FEATURE_DIM], source_weight)
    assert torch.count_nonzero(target_weight[:, TOKEN_FEATURE_DIM:]) == 0
    assert provenance["missing_tensors"] == 2
    assert set(provenance["missing_tensor_names"]) == {
        "reprojection_skip.weight",
        "reprojection_skip.bias",
    }
    assert provenance["adapted_tensors"] == [
        {
            "tensor": "token_embedding.input_projection.weight",
            "source_shape": [32, TOKEN_FEATURE_DIM],
            "target_shape": [32, TOKEN_FEATURE_DIM_WITH_REPROJECTION],
            "policy": "copy learned prefix; zero-initialize appended features",
        }
    ]


def test_reprojection_skip_is_identity_safe_and_receives_direct_gradient() -> None:
    legacy = WholeSequenceRefiner(
        input_dim=TOKEN_FEATURE_DIM_WITH_REPROJECTION,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        max_frames=4,
        dropout=0.0,
    )
    assert legacy.reprojection_skip is None

    model = WholeSequenceRefiner(
        input_dim=TOKEN_FEATURE_DIM_WITH_REPROJECTION,
        use_reprojection_skip=True,
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        max_frames=4,
        dropout=0.0,
    )
    features = torch.zeros(1, 4, 51, TOKEN_FEATURE_DIM_WITH_REPROJECTION)
    features[..., 43:45] = torch.randn(1, 4, 51, 2)
    initial = axis_angle_to_matrix(torch.zeros(1, 4, 51, 3))
    valid = torch.ones(1, 4, dtype=torch.bool)
    refine = torch.ones(1, 51, dtype=torch.bool)

    identity = model(features, initial, valid, refine)["matrix"]
    assert torch.equal(identity, initial)

    target = axis_angle_to_matrix(torch.full((1, 4, 51, 3), 0.05))
    prediction = model(features, initial, valid, refine)
    loss = geodesic_distance(prediction["matrix"], target).mean()
    loss.backward()
    assert model.reprojection_skip is not None
    assert model.reprojection_skip.weight.grad is not None
    assert torch.count_nonzero(model.reprojection_skip.weight.grad) > 0


def test_uncertainty_head_can_be_reconstruction_neutral() -> None:
    torch.manual_seed(7)
    deterministic = WholeSequenceRefiner(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        max_frames=8,
        dropout=0.0,
        predict_uncertainty=False,
    ).eval()
    uncertainty = WholeSequenceRefiner(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        max_frames=8,
        dropout=0.0,
        predict_uncertainty=True,
        uncertainty_feedback=False,
    ).eval()
    compatible = {
        key: value
        for key, value in deterministic.state_dict().items()
        if key in uncertainty.state_dict()
        and uncertainty.state_dict()[key].shape == value.shape
    }
    uncertainty.load_state_dict(compatible, strict=False)
    features = torch.randn(2, 6, 51, TOKEN_FEATURE_DIM)
    features[..., 29] = torch.rand(2, 6, 51)
    initial = axis_angle_to_matrix(torch.randn(2, 6, 51, 3) * 0.1)
    valid = torch.ones(2, 6, dtype=torch.bool)
    refine = torch.ones(2, 51, dtype=torch.bool)
    with torch.no_grad():
        reference = deterministic(features, initial, valid, refine)
        candidate = uncertainty(features, initial, valid, refine)
    assert torch.equal(reference["matrix"], candidate["matrix"])
    assert torch.equal(reference["raw_delta"], candidate["raw_delta"])
    assert torch.equal(reference["reliability"], candidate["reliability"])
    assert "log_variance" in candidate
