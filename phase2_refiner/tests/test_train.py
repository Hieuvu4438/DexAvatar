import pytest
import torch

from phase2_refiner.train import (
    ExponentialMovingAverage,
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
