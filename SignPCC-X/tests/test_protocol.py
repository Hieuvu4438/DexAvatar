from pathlib import Path

import numpy as np
import yaml

from signpccx.evaluation.parse_metrics import parse_global_metrics
from signpccx.io import load_config


def centered_error(prediction, target, indices):
    pred = prediction[indices] - prediction[indices].mean(0)
    truth = target[indices] - target[indices].mean(0)
    return np.linalg.norm(pred - truth, axis=-1).mean()


def test_translation_invariance_and_rotation_sensitivity():
    rng = np.random.default_rng(7)
    truth = rng.normal(size=(20, 3))
    indices = np.arange(5, 15)
    assert centered_error(truth + [10.0, -4.0, 2.0], truth, indices) < 1e-12
    rotated = truth * [1.0, -1.0, -1.0]
    assert centered_error(rotated, truth, indices) > 0.1


def test_metric_parser():
    text = """
    [m]: Tr Left Hand: 12.3 (mm)
    [m]: Tr Right Hand: 13.4 (mm)
    [m]: Tr Above Pelvis Upper Body: 29.5 (mm)
    """
    assert parse_global_metrics(text) == {
        "tr left hand": 12.3,
        "tr right hand": 13.4,
        "tr above pelvis upper body": 29.5,
    }


def test_default_config_has_no_temporal_pose_loss():
    config_path = Path(__file__).parents[1] / "configs" / "signpccx_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["temporal"] == {
        "pose_smoothing": False,
        "velocity_loss": False,
        "acceleration_loss": False,
    }


def test_best_config_is_gt_free_and_non_temporal():
    config_path = Path(__file__).parents[1] / "configs" / "ablations" / "a3f_external_v1_identity200.yaml"
    config = load_config(config_path)
    assert config["experiment"]["name"] == "signpccx_a3f_external_v1_identity200"
    assert config["canonical_refit"]["ground_truth_in_objective"] is False
    assert config["canonical_refit"]["evaluator_upper_body_mask_in_objective"] is False
    assert not any(config["temporal"].values())
    assert config["identity"]["scope"] == "signer"


def test_config_inheritance_keeps_child_relative_paths():
    path = Path(__file__).parents[1] / "configs" / "ablations" / "a3f_external_v1_identity200.yaml"
    config = load_config(path)
    assert config["experiment"]["name"] == "signpccx_a3f_external_v1_identity200"
    assert config["identity"]["calibration_frames"] == 200
    assert config["paths"]["run_root"].endswith("signpccx_a3f_external_v1_identity200")
    assert config["canonical_refit"]["ground_truth_in_objective"] is False
