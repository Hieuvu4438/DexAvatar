import torch

from signpk.data.training_augmentation import augment_training_tokens
from signpk.models.explicit_tokens import ExplicitTokenBatch


def test_training_observation_dropout_preserves_base_rotations():
    batch, time = 2, 3
    identity = torch.eye(3)
    tokens = ExplicitTokenBatch(
        body=torch.ones(batch, time, 14, 12),
        left=torch.ones(batch, time, 15, 54),
        right=torch.ones(batch, time, 15, 54),
        relation=torch.ones(batch, time, 20),
        timestamps=torch.arange(time).float()[None].expand(batch, -1),
        upper_base_rotmat=identity.expand(batch, time, 14, 3, 3).clone(),
        left_base_rotmat=identity.expand(batch, time, 15, 3, 3).clone(),
        right_base_rotmat=identity.expand(batch, time, 15, 3, 3).clone(),
        left_valid=torch.ones(batch, time, dtype=torch.bool),
        right_valid=torch.ones(batch, time, dtype=torch.bool),
        disagreement=torch.zeros(batch, time, 2, 2),
        left_observer_feature=torch.ones(batch, time, 1024),
        right_observer_feature=torch.ones(batch, time, 1024),
    )
    augmented = augment_training_tokens(tokens, observation_dropout=1.0)
    assert not augmented.left_valid.any() and not augmented.right_valid.any()
    assert augmented.left.count_nonzero() == 0
    assert augmented.relation.count_nonzero() == 0
    torch.testing.assert_close(augmented.upper_base_rotmat, tokens.upper_base_rotmat)
