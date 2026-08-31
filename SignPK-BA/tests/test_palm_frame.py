import torch

from signpk.geometry.palm_frame import make_palm_frame


def _hand():
    joints = torch.zeros(21, 3)
    joints[5] = torch.tensor([1.0, 1.0, 0.0])
    joints[9] = torch.tensor([0.0, 1.2, 0.0])
    joints[17] = torch.tensor([-1.0, 1.0, 0.0])
    return joints


def test_palm_frame_is_right_handed():
    frame, wrist, valid = make_palm_frame(_hand()[None], "right")
    assert valid.item()
    torch.testing.assert_close(frame.transpose(-1, -2) @ frame, torch.eye(3)[None], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.det(frame), torch.ones(1), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(wrist, torch.zeros(1, 3))


def test_left_semantic_normal_matches_mirrored_right():
    right = _hand()
    left = right.clone()
    left[:, 0] *= -1
    right_frame, _, _ = make_palm_frame(right[None], "right")
    left_frame, _, _ = make_palm_frame(left[None], "left")
    assert torch.det(left_frame).item() > 0
    # Semantic handedness correction makes the camera-facing normal agree.
    torch.testing.assert_close(left_frame[..., :, 2], right_frame[..., :, 2], atol=1e-6, rtol=1e-6)


def test_degenerate_palm_is_invalid_and_finite():
    joints = torch.zeros(2, 21, 3)
    frame, _, valid = make_palm_frame(joints, "left")
    assert not valid.any()
    assert torch.isfinite(frame).all()

