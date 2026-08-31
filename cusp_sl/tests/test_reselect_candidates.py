import numpy as np
import torch

from cusp_sl.reselect_candidates import reselect_payload
from cusp_sl.selection import EnergyStatistics


def test_reselection_preserves_candidates_and_selects_minimum_energy():
    identity = np.eye(3, dtype=np.float32)
    candidates = np.tile(identity, (3, 2, 1, 1, 1))
    arrays = {
        "candidate_rotation": candidates,
        "energy_terms": np.asarray(
            [[2.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "candidate_valid": np.asarray([True, True, True]),
        "selected_rotation": candidates[0],
        "selected_axis_angle": np.zeros((2, 1, 3), dtype=np.float32),
        "selected_index": np.asarray(0),
    }
    result, index = reselect_payload(
        arrays,
        EnergyStatistics(torch.zeros(4), torch.ones(4)),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        1.0,
    )
    assert index == 1
    np.testing.assert_array_equal(result["candidate_rotation"], candidates)
    np.testing.assert_array_equal(result["energy_terms"], arrays["energy_terms"])
    assert int(result["selected_index"]) == 1


def test_reselection_never_selects_invalid_candidate():
    identity = np.eye(3, dtype=np.float32)
    candidates = np.tile(identity, (2, 1, 1, 1, 1))
    arrays = {
        "candidate_rotation": candidates,
        "energy_terms": np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "candidate_valid": np.asarray([True, False]),
    }
    _, index = reselect_payload(
        arrays,
        EnergyStatistics(torch.zeros(4), torch.ones(4)),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        1.0,
    )
    assert index == 0
