import numpy as np
import pytest

from cusp_sl.evaluate_selection_evidence import compare_payloads, target_reads_prohibited


def payload() -> dict[str, np.ndarray]:
    identity = np.eye(3, dtype=np.float32)
    return {
        "candidate_rotation": np.tile(identity, (2, 3, 1, 1, 1)),
        "candidate_residual": np.zeros((2, 3, 1, 3), dtype=np.float32),
        "candidate_valid": np.asarray([True, True]),
        "energy_terms": np.asarray(
            [[0.4, 0.2, 0.1, 0.0], [0.3, 0.25, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "energy": np.asarray([1.0, 0.5], dtype=np.float32),
        "selected_index": np.asarray(1),
    }


def test_compare_payloads_reports_exact_selected_minus_base_inputs():
    source = payload()
    selected = {key: value.copy() for key, value in source.items()}
    index, values = compare_payloads(source, selected)
    assert index == 1
    assert np.isclose(values["base_observation"], 0.4)
    assert np.isclose(values["selected_observation"], 0.3)
    assert np.isclose(values["base_composite"], 1.0)
    assert np.isclose(values["selected_composite"], 0.5)


def test_compare_payloads_rejects_candidate_mutation():
    source = payload()
    selected = {key: value.copy() for key, value in source.items()}
    selected["candidate_residual"][1, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="modified frozen array"):
        compare_payloads(source, selected)


def test_compare_payloads_rejects_invalid_selection():
    source = payload()
    source["candidate_valid"][1] = False
    selected = {key: value.copy() for key, value in source.items()}
    with pytest.raises(ValueError, match="Selected invalid"):
        compare_payloads(source, selected)


def test_target_reads_prohibition_accepts_only_explicit_false_or_integer_zero():
    assert target_reads_prohibited({"target_reads_permitted": False})
    assert target_reads_prohibited({"target_reads_permitted": 0})
    assert not target_reads_prohibited({"target_reads_permitted": True})
    assert not target_reads_prohibited({"target_reads_permitted": "0"})
    assert not target_reads_prohibited({})
