from pathlib import Path

import numpy as np
import torch

from signpccx.io import load_config
from signpccx.model.smplx_state import SMPLX_BODY_NAMES, validate_body_slots
from signpccx.optimization.hypotheses import _axis_angle_to_matrix, _matrix_to_axis_angle
from signpccx.optimization.losses import safe_project, signed_point_to_triangle


def test_signed_negative_z_projection_matches_h4w_convention():
    points = torch.tensor([[[1.0, 2.0, -10.0]]])
    intrinsics = torch.tensor([[-1000.0, 0.0, 96.0], [0.0, 1000.0, 128.0], [0.0, 0.0, 1.0]])
    expected = torch.tensor([[[196.0, -72.0]]])
    assert torch.allclose(safe_project(points, intrinsics), expected, atol=1e-6)


def test_axis_angle_roundtrip_identity_and_near_pi():
    vectors = torch.tensor([
        [0.0, 0.0, 0.0],
        [np.pi - 1e-5, 0.0, 0.0],
        [0.0, (np.pi - 1e-5) / np.sqrt(2), (np.pi - 1e-5) / np.sqrt(2)],
    ], dtype=torch.float64)
    matrices = _axis_angle_to_matrix(vectors)
    recovered = _axis_angle_to_matrix(_matrix_to_axis_angle(matrices))
    assert torch.max(torch.abs(matrices - recovered)) < 1e-5


def test_body_slot_names_runtime_contract():
    validate_body_slots(SMPLX_BODY_NAMES)
    wrong = list(SMPLX_BODY_NAMES)
    wrong[20] = "right_wrist"
    try:
        validate_body_slots(tuple(wrong))
    except RuntimeError:
        pass
    else:
        raise AssertionError("body slot mismatch was not rejected")


def test_signed_point_triangle_penetration_gradient_pushes_outward():
    triangle = torch.tensor([[[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]])
    point = torch.tensor([[0.0, 0.0, -0.01]], requires_grad=True)
    signed = signed_point_to_triangle(point, triangle)
    loss = torch.relu(-signed).square().mean()
    loss.backward()
    assert signed.item() < 0
    assert torch.isfinite(point.grad).all()
    # Gradient descent subtracts a negative z gradient, moving toward +z/outside.
    assert point.grad[0, 2] < 0


def test_full_configs_encode_exact_a1_a5_matrix_and_stage_counts():
    root = Path(__file__).resolve().parents[1] / "configs" / "full"
    expected = {
        "a1_h4wpp.yaml": (False, False, False, False),
        "a2_shared_beta.yaml": (True, False, False, False),
        "a3_shared_beta_camera.yaml": (True, True, False, False),
        "a4_palm_best_of_k.yaml": (True, True, True, False),
        "a5_contact_penetration.yaml": (True, True, True, True),
    }
    for name, flags in expected.items():
        config = load_config(root / name)
        method = config["method"]
        assert tuple(method[key] for key in ("shared_beta", "shared_camera", "hypotheses", "contact")) == flags
        stages = config["optimization"]["stages"]
        assert stages["camera_root"]["steps"] == 60
        assert stages["upper_body"]["steps"] == 100
        assert stages["hand_candidate"]["steps"] == 25
        assert stages["bimanual_contact"]["steps"] == 100
        assert stages["lbfgs"]["steps"] == 20
        assert stages["canonical"]["steps"] == 40
        assert stages["camera_root"]["min_steps"] == 50
        assert stages["upper_body"]["min_steps"] == 75
        assert stages["hand_candidate"]["min_steps"] == 25
        assert stages["bimanual_contact"]["min_steps"] == 80
        assert stages["canonical"]["min_steps"] == 20
        assert not config["temporal"]["pose_smoothing"]
