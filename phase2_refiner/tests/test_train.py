import pytest
import torch

import phase2_refiner.train as train_module
from phase2_refiner.train import (
    ExponentialMovingAverage,
    _geometry_loss_arguments,
    regional_validation_selection_score,
)


def test_regional_validation_selection_score_matches_contract() -> None:
    score, ratios = regional_validation_selection_score(
        {"ubody": 10.0, "lhand": 20.0, "rhand": 40.0},
        {"ubody": 9.0, "lhand": 20.4, "rhand": 36.0},
    )
    assert ratios == {"ubody": 0.9, "lhand": 1.02, "rhand": 0.9}
    assert score == pytest.approx((0.9 + 1.02 + 0.9) / 3.0 + 0.5 * 0.01)


def test_regional_validation_selection_score_rejects_empty_region() -> None:
    with pytest.raises(ValueError, match="lhand"):
        regional_validation_selection_score(
            {"ubody": 1.0, "lhand": 0.0, "rhand": 1.0},
            {"ubody": 1.0, "lhand": 0.0, "rhand": 1.0},
        )


def test_ema_validation_context_restores_training_weights() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    ema = ExponentialMovingAverage(model, decay=0.9)
    ema.shadow["weight"].fill_(2.0)
    original = model.weight.detach().clone()

    with ema.average_parameters(model):
        assert torch.equal(model.weight, torch.full_like(model.weight, 2.0))

    assert torch.equal(model.weight, original)


def test_geometry_arguments_decode_the_effective_initializer(monkeypatch) -> None:
    decoded_matrices = []

    def fake_decode(_model, matrix, *_args, **_kwargs):
        decoded_matrices.append(matrix.clone())
        shape = (*matrix.shape[:2], 3, 3)
        return torch.zeros(shape), None

    monkeypatch.setattr(train_module, "decode_smplx_sequence", fake_decode)
    original = torch.zeros(1, 2, 51, 3, 3)
    effective = torch.ones_like(original)
    batch = {
        "initial_matrix": original,
        "target_matrix": original + 2.0,
        "betas": torch.zeros(1, 10),
        "global_orient": torch.zeros(1, 2, 3),
        "transl": torch.zeros(1, 2, 3),
        "jaw_pose": torch.zeros(1, 2, 3),
        "leye_pose": torch.zeros(1, 2, 3),
        "reye_pose": torch.zeros(1, 2, 3),
        "expression": torch.zeros(1, 2, 10),
    }
    prediction = {"matrix": original + 3.0}

    _geometry_loss_arguments(
        {
            "model": object(),
            "region_masks": {
                "ubody": torch.tensor([0]),
                "lhand": torch.tensor([1]),
                "rhand": torch.tensor([2]),
            },
        },
        prediction,
        batch,
        effective,
        torch.device("cpu"),
    )

    assert torch.equal(decoded_matrices[1], effective)
