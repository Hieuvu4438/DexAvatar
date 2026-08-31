import hashlib
import json

import pytest

from cusp_sl.calibrate_gate import fold
from cusp_sl.gate_artifact import load_gate_thresholds


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_group_fold_is_stable_and_not_clip_keyed():
    assert fold("same-video") == fold("same-video")
    assert fold("same-video") in {"fit", "audit"}


def test_gate_artifact_is_bound_to_config_and_checkpoints(tmp_path):
    config = tmp_path / "config.yaml"
    q = tmp_path / "q.pt"
    generator = tmp_path / "g.pt"
    for path, value in ((config, b"config"), (q, b"q"), (generator, b"g")):
        path.write_bytes(value)
    artifact = tmp_path / "gate.json"
    payload = {
        "role": "development_gate_calibration_with_hash_disjoint_audit",
        "selection_control": "fixed_seed_k1",
        "split_unit": "source_group",
        "config_sha256": _sha(config),
        "reliability_checkpoint_sha256": _sha(q),
        "flow_checkpoint_sha256": _sha(generator),
        "best_fit": {"tau_low": 0.2, "tau_high": 0.8},
    }
    artifact.write_text(json.dumps(payload))
    low, high, loaded = load_gate_thresholds(
        artifact,
        config_path=config,
        reliability_checkpoint=q,
        generator_checkpoint=generator,
    )
    assert (low, high) == (0.2, 0.8)
    assert loaded == payload

    q.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        load_gate_thresholds(
            artifact,
            config_path=config,
            reliability_checkpoint=q,
            generator_checkpoint=generator,
        )


def test_gate_artifact_enforces_deterministic_control_label(tmp_path):
    config = tmp_path / "config.yaml"
    q = tmp_path / "q.pt"
    generator = tmp_path / "g.pt"
    for path, value in ((config, b"config"), (q, b"q"), (generator, b"g")):
        path.write_bytes(value)
    artifact = tmp_path / "gate.json"
    payload = {
        "role": "development_gate_calibration_with_hash_disjoint_audit",
        "selection_control": "deterministic_point_estimate",
        "generator_kind": "deterministic",
        "split_unit": "source_group",
        "config_sha256": _sha(config),
        "reliability_checkpoint_sha256": _sha(q),
        "flow_checkpoint_sha256": _sha(generator),
        "best_fit": {"tau_low": 0.2, "tau_high": 0.8},
    }
    artifact.write_text(json.dumps(payload))
    low, high, _ = load_gate_thresholds(
        artifact,
        config_path=config,
        reliability_checkpoint=q,
        generator_checkpoint=generator,
        generator_kind="deterministic",
    )
    assert (low, high) == (0.2, 0.8)
    with pytest.raises(ValueError, match="selection control/generator kind"):
        load_gate_thresholds(
            artifact,
            config_path=config,
            reliability_checkpoint=q,
            generator_checkpoint=generator,
            generator_kind="flow",
        )
