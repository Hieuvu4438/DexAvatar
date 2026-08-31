import torch


def test_right_canonical_rotation_mirror_is_proper_rotation():
    angle = torch.tensor(0.4)
    right = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0],
            [torch.sin(angle), torch.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    reflection = torch.tensor([-1.0, 1.0, 1.0])
    left = right * reflection[:, None] * reflection[None, :]
    torch.testing.assert_close(left.T @ left, torch.eye(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(torch.det(left), torch.tensor(1.0), atol=1e-6, rtol=1e-6)
    # One conjugation reverses z rotation; a second would silently restore it.
    assert left[0, 1] > 0 and right[0, 1] < 0

