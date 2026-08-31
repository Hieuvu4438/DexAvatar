import torch

from signpk.data.cache_schema import BodyObservation, CouplerPrediction, HandObservation
from signpk.geometry.rotations import so3_exp
from signpk.optimization.clip_ba import ClipBundleAdjuster
from signpk.optimization.factors import FactorInputs, compute_factors
from signpk.optimization.smplx_layer import SMPLXOutput
from signpk.optimization.state import SequenceState


def _hand(frames):
    identity = torch.eye(3)
    return HandObservation(
        identity.expand(frames, 16, 3, 3).clone(),
        torch.zeros(frames, 10),
        torch.zeros(frames, 778, 3),
        torch.zeros(frames, 21, 3),
        identity.expand(frames, 3, 3).clone(),
        torch.zeros(frames, 3),
        torch.ones(frames, 4),
        torch.zeros(frames, 21, 2),
        torch.zeros(frames, 21),
        torch.zeros(frames),
        torch.zeros(frames, dtype=torch.bool),
    )


def test_staged_ba_keeps_matching_best_state():
    frames = 2
    identity = torch.eye(3)
    base_upper = so3_exp(torch.full((frames, 14, 3), 0.03))
    state = SequenceState(
        base_upper,
        identity.expand(frames, 21, 3, 3).clone(),
        identity.expand(frames, 15, 3, 3).clone(),
        identity.expand(frames, 15, 3, 3).clone(),
        torch.zeros(10),
        torch.zeros(frames, 3),
    )
    body = BodyObservation(
        identity.expand(frames, 3, 3).clone(),
        identity.expand(frames, 21, 3, 3).clone(),
        torch.zeros(frames, 10),
        torch.zeros(frames, 10475, 3),
        torch.zeros(frames, 55, 3),
        torch.zeros(frames, 42, 2),
        torch.zeros(frames, 42),
        torch.zeros(frames, 3),
        torch.ones(frames, 2),
        torch.zeros(frames, 2),
    )
    hand = _hand(frames)
    pkc = CouplerPrediction(
        base_upper[:, 0],
        base_upper,
        identity.expand(frames, 15, 3, 3).clone(),
        identity.expand(frames, 15, 3, 3).clone(),
        torch.zeros(frames, 44, 3),
        torch.zeros(frames, 2, 3),
        {
            "upper": torch.zeros(frames, 14),
            "left": torch.zeros(frames, 15),
            "right": torch.zeros(frames, 15),
            "palm": torch.zeros(frames, 2),
        },
        torch.ones(frames, 1),
        torch.zeros(frames, 1),
    )
    inputs = FactorInputs(
        body,
        hand,
        hand,
        None,
        None,
        pkc,
        torch.arange(778),
        torch.arange(778, 1556),
        torch.zeros(frames, 3),
        torch.arange(frames).float(),
    )

    def model(current):
        # Geometry is irrelevant to the active h4 rotation factor, but the
        # sanity gate still verifies standard topology and finiteness.
        return SMPLXOutput(
            current.translation[:, None].expand(-1, 10475, -1),
            torch.zeros(frames, 55, 3),
            torch.zeros(frames, 21, 3),
            torch.zeros(frames, 21, 3),
        )

    initial = compute_factors(model(state), state, inputs)["h4w"].item()
    result = ClipBundleAdjuster(
        model,
        [
            {
                "name": "upper",
                "iterations": 15,
                "optimizer": "adam",
                "learning_rate": 0.02,
                "variables": ["root", "upper_body"],
                "weights": {"h4w": 1.0, "residual": 0.001},
            }
        ],
    ).optimize(state, inputs)
    final = compute_factors(result.output, result.state, inputs)["h4w"].item()
    assert final < initial
    accepted_losses = [record.loss for record in result.records if record.accepted]
    assert accepted_losses and accepted_losses[-1] == min(accepted_losses)
