from types import SimpleNamespace

import torch

from cusp_sl.calibrate_gate import calibration_residual


class ConstantDeterministic(torch.nn.Module):
    def forward(self, state, time, condition, frame_valid):
        assert torch.count_nonzero(state) == 0
        assert torch.count_nonzero(time) == 0
        assert bool(frame_valid.all())
        return torch.ones_like(state) * 2.0


class OffsetNormalizer:
    def denormalize(self, value):
        return value + 3.0


def test_calibration_deterministic_path_matches_zero_state_inference_contract():
    condition = torch.zeros(1, 4, 2, 7)
    base = torch.eye(3).expand(1, 4, 2, 3, 3).clone()
    result = calibration_residual(
        ConstantDeterministic(),
        condition,
        base,
        SimpleNamespace(),
        "clip",
        torch.device("cpu"),
        OffsetNormalizer(),
        "deterministic",
    )
    torch.testing.assert_close(result, torch.full((1, 4, 2, 3), 5.0))


def test_calibration_rejects_unknown_generator_kind():
    condition = torch.zeros(1, 1, 1, 4)
    base = torch.eye(3).expand(1, 1, 1, 3, 3).clone()
    try:
        calibration_residual(
            ConstantDeterministic(), condition, base, SimpleNamespace(),
            "clip", torch.device("cpu"), OffsetNormalizer(), "unknown",
        )
    except ValueError as error:
        assert "Unsupported generator kind" in str(error)
    else:
        raise AssertionError("Unknown generator kind must be rejected")
