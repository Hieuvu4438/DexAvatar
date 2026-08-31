import torch

from signpk.optimization.state import SequenceState


def test_zero_residual_reproduces_pkc_and_is_differentiable():
    frames = 3
    identity = torch.eye(3)
    state = SequenceState(
        identity.expand(frames, 14, 3, 3).clone(),
        identity.expand(frames, 21, 3, 3).clone(),
        identity.expand(frames, 15, 3, 3).clone(),
        identity.expand(frames, 15, 3, 3).clone(),
        torch.zeros(10),
        torch.zeros(frames, 3),
    )
    rotations = state.rotations()
    torch.testing.assert_close(rotations.upper, state.base_upper)
    loss = rotations.upper[..., 0, 1].sum() + rotations.left_hand[..., 1, 2].sum()
    loss.backward()
    assert state.root_delta.grad is not None
    assert state.left_hand_delta.grad is not None

