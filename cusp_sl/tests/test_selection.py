import torch

from cusp_sl.geometry import axis_angle_to_matrix
from cusp_sl.selection import EnergyStatistics, candidate_energy, select_candidates


def test_base_candidate_can_win_and_weights_normalize():
    rotations = axis_angle_to_matrix(torch.zeros(1, 3, 2, 51, 3))
    terms = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [2.0, 2.0, 2.0, 0.0]]])
    stats = EnergyStatistics.fit(terms.reshape(-1, 4))
    energy = candidate_energy(terms, stats, torch.tensor([1.0, 1.0, 1.0, 0.0]))
    result = select_candidates(rotations, energy, torch.ones_like(energy, dtype=torch.bool), 1.0)
    assert result["index"].item() == 0
    torch.testing.assert_close(result["weights"].sum(dim=1), torch.ones(1))

