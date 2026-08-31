import torch

from signeft.losses.heatmap_nll import normalize_heatmaps, sample_heatmap_nll
from signeft.losses.nlf_bones import robust_unit_bones


def test_heatmap_nll_bilinear_sampling_prefers_peak():
    heatmap = torch.zeros(1, 1, 4, 4)
    heatmap[0, 0, 2, 1] = 1.0
    normalized, valid_mass = normalize_heatmaps(heatmap)
    at_peak = sample_heatmap_nll(normalized, torch.tensor([[[1.0, 2.0]]]), valid_mass)
    away = sample_heatmap_nll(normalized, torch.tensor([[[2.0, 2.0]]]), valid_mass)
    assert at_peak < away


def test_zero_mass_heatmap_is_explicitly_invalid():
    _, valid = normalize_heatmaps(torch.zeros(2, 3, 4, 5))
    assert not valid.any()


def test_nlf_bones_return_direction_length_and_validity():
    joints = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    unit, length, valid = robust_unit_bones(joints, ((0, 1),), torch.ones(1, 2, dtype=torch.bool))
    assert torch.allclose(unit, torch.tensor([[[0.0, 1.0, 0.0]]]))
    assert torch.allclose(length, torch.tensor([[2.0]]))
    assert valid.item()

