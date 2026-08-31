import json

import pytest
import torch

from dcg_sign4d.observations.calibration import (
    TemperatureScaler,
    load_frozen_temperature,
)


def test_temperature_scaling_improves_overconfident_fixture(tmp_path):
    # Correct class ranking but excessive confidence on deliberately noisy labels.
    logits = torch.tensor([[8.0, -8.0]] * 80 + [[-8.0, 8.0]] * 20)
    labels = torch.tensor([0] * 60 + [1] * 20 + [1] * 15 + [0] * 5)
    scaler = TemperatureScaler()
    report = scaler.fit(logits, labels, split="calibration", bins=10)
    assert report.calibrated_nll < report.raw_nll
    assert report.passes_no_worse_gate
    scaler.save(tmp_path / "temperature.json", report)


def test_test_split_cannot_fit_calibrator():
    try:
        TemperatureScaler().fit(torch.randn(4, 2), torch.zeros(4, dtype=torch.long), split="test")
    except ValueError as exc:
        assert "restricted" in str(exc)
    else:
        raise AssertionError("test split was accepted")


def test_frozen_json_calibrator_is_executable_only_after_gate_pass(tmp_path):
    artifact = tmp_path / "calibrator.json"
    payload = {
        "schema_version": "dcg_temperature_calibration_v1",
        "temperature": 2.0,
        "gate_status": "PASS",
    }
    artifact.write_text(json.dumps(payload))
    scaler = load_frozen_temperature(artifact)
    assert torch.allclose(scaler(torch.tensor([[2.0, -2.0]])), torch.tensor([[1.0, -1.0]]))
    assert not any(parameter.requires_grad for parameter in scaler.parameters())

    payload["gate_status"] = "FAIL"
    artifact.write_text(json.dumps(payload))
    with pytest.raises(PermissionError, match="did not pass"):
        load_frozen_temperature(artifact)

    payload["gate_status"] = "PASS"
    payload["development_only"] = True
    artifact.write_text(json.dumps(payload))
    with pytest.raises(PermissionError, match="development calibrator"):
        load_frozen_temperature(artifact)
    load_frozen_temperature(artifact, allow_development=True)
