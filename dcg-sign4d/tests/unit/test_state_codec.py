import torch

from dcg_sign4d.diffusion.state_codec import StateCodec, TrajectoryState, rotation_6d_to_matrix


def state():
    torch.manual_seed(8)
    return TrajectoryState(
        root_rot6d=torch.randn(2, 4, 6),
        root_translation=torch.randn(2, 4, 3),
        root_velocity=torch.randn(2, 4, 3),
        body_rot6d=torch.randn(2, 4, 21, 6),
        left_hand_rot6d=torch.randn(2, 4, 15, 6),
        right_hand_rot6d=torch.randn(2, 4, 15, 6),
        face_state=torch.randn(2, 4, 13),
        beta=torch.randn(2, 10),
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
    )


def test_codec_round_trip_is_exact_within_float_tolerance():
    original = state()
    encoded, context = StateCodec().encode(original)
    decoded = StateCodec().decode(encoded, context)
    for name in original.__dataclass_fields__:
        first, second = getattr(original, name), getattr(decoded, name)
        if first is None:
            assert second is None
        elif first.dtype == torch.bool:
            assert torch.equal(first, second)
        else:
            assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_codec_uses_first_valid_root_relative_translation():
    original = state()
    encoded, context = StateCodec().encode(original)
    assert torch.equal(encoded[:, 0, 6:9], torch.zeros_like(encoded[:, 0, 6:9]))
    assert torch.equal(context.root_origin, original.root_translation[:, 0])


def test_training_normalizer_round_trip_and_payload():
    original = state()
    codec = StateCodec.fit(original)
    encoded, context = codec.encode(original)
    decoded = codec.decode(encoded, context)
    assert torch.allclose(decoded.root_translation, original.root_translation, atol=1e-6)
    restored = StateCodec.from_payload(codec.to_payload())
    restored_encoded, _ = restored.encode(original)
    assert torch.allclose(restored_encoded, encoded, atol=1e-5)


def test_rotation_conversion_is_orthonormal_and_proper():
    matrices = rotation_6d_to_matrix(state().body_rot6d)
    identity = torch.eye(3).expand_as(matrices)
    assert torch.allclose(matrices @ matrices.transpose(-1, -2), identity, atol=1e-5)
    assert torch.allclose(
        torch.linalg.det(matrices), torch.ones_like(torch.linalg.det(matrices)), atol=1e-5
    )
