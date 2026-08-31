import torch

from signpk.models.explicit_tokens import ExplicitTokenBatch
from signpk.models.palm_kinematic_coupler import PalmKinematicCoupler


def test_pkc_shapes_and_zero_initialized_residual():
    batch, time = 2, 5
    identity = torch.eye(3)
    tokens = ExplicitTokenBatch(
        body=torch.randn(batch, time, 14, 12),
        left=torch.randn(batch, time, 15, 54),
        right=torch.randn(batch, time, 15, 54),
        relation=torch.randn(batch, time, 20),
        timestamps=torch.arange(time).float()[None].repeat(batch, 1) / 25,
        upper_base_rotmat=identity.expand(batch, time, 14, 3, 3).clone(),
        left_base_rotmat=identity.expand(batch, time, 15, 3, 3).clone(),
        right_base_rotmat=identity.expand(batch, time, 15, 3, 3).clone(),
        left_valid=torch.ones(batch, time, dtype=torch.bool),
        right_valid=torch.ones(batch, time, dtype=torch.bool),
        disagreement=torch.rand(batch, time, 2, 2),
        left_observer_feature=torch.randn(batch, time, 1024),
        right_observer_feature=torch.randn(batch, time, 1024),
        left_h4w_feature=torch.randn(batch, time, 16),
        right_h4w_feature=torch.randn(batch, time, 16),
    )
    model = PalmKinematicCoupler(
        hidden_dim=32,
        temporal_layers=1,
        attention_heads=4,
        dropout=0,
        h4w_hand_feature_dim=16,
    )
    output = model(tokens)
    assert output.upper_rotmat.shape == (batch, 14, 3, 3)
    assert output.angular_velocity.shape == (batch, 44, 3)
    torch.testing.assert_close(
        output.upper_rotmat, tokens.upper_base_rotmat[:, time // 2], atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        output.left_rotmat, tokens.left_base_rotmat[:, time // 2], atol=1e-6, rtol=1e-6
    )
    assert ((0 <= output.phase_gate) & (output.phase_gate <= 1)).all()
    assert ((0 <= output.interaction_gate) & (output.interaction_gate <= 1)).all()
