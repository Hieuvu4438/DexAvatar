import torch

from dcg_sign4d.geometry.so3 import log_map
from dcg_sign4d.initialization.dexavatar_adapter import axis_angle_to_matrix


def test_log_exp_round_trip_away_from_branch_cut():
    vectors = torch.tensor([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [-1.0, 0.2, 0.1]])
    recovered = log_map(axis_angle_to_matrix(vectors))
    assert torch.allclose(recovered, vectors, atol=1e-5)
