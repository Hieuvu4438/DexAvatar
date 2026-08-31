from dataclasses import replace

import torch

from cusp_sl.config import load_config
from cusp_sl.geometry import axis_angle_to_matrix
from cusp_sl.train_flow import prepare_batch


class ZeroReliability(torch.nn.Module):
    def forward(self, features):
        return torch.zeros(features.shape[:-1], device=features.device)


def test_real_pairs_receive_flow_supervision():
    config = load_config("cusp_sl/configs/cusp_sl_v1.yaml")
    training = replace(
        config.training, real_fraction=1.0,
        synthetic_fraction=0.0, clean_fraction=0.0,
    )
    config = replace(config, training=training)
    initial = axis_angle_to_matrix(torch.zeros(1, 2, 51, 3))
    target_angle = torch.zeros(1, 2, 51, 3)
    target_angle[..., 21, 0] = 0.1
    batch = {
        "features": torch.zeros(1, 2, 51, config.data.input_dim),
        "initial_matrix": initial,
        "target_matrix": axis_angle_to_matrix(target_angle),
        "frame_valid": torch.ones(1, 2, dtype=torch.bool),
        "target_rotation_valid": torch.ones(1, 2, 51, dtype=torch.bool),
        "refine_mask": torch.ones(1, 51, dtype=torch.bool),
        "target_quality": torch.ones(1, 2, 51),
    }
    _, _, target, weight = prepare_batch(
        batch, config, ZeroReliability(), temperature=1.0
    )
    assert weight.sum() == 102
    torch.testing.assert_close(target[..., 21, 0], torch.full((1, 2), 0.1))


def test_frame_valid_mask_can_expand_to_all_joint_tokens():
    frame_valid = torch.tensor([[True, False]])
    target_valid = torch.ones(1, 2, 51, dtype=torch.bool)
    valid = frame_valid[:, :, None].expand_as(target_valid).clone()
    valid &= target_valid
    assert valid.shape == (1, 2, 51)
    assert int(valid.sum()) == 51
