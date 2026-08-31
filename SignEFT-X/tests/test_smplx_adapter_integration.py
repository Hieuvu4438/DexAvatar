from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from signeft.data.manifest import read_manifest
from signeft.io_utils import array_sha256
from signeft.model.kinematics import translation_aligned_hand_distance
from signeft.model.smplx_adapter import BaselineBatch, TrustRegionSMPLX, load_mano_vertex_ids
from signeft.optim.core import ALL_ACTIVE


ROOT = Path(__file__).parents[1]
MODEL_ROOT = ROOT.parent / "SMPLer-X" / "common" / "utils" / "human_model_files"
EXPECTED_FACE_HASH = "2cb81d8e6c789896d764805d58fb44bdce62424bab97b519bbd6c1668d66ce2b"


def _adapter(*, wrist_protection: bool) -> tuple[BaselineBatch, TrustRegionSMPLX]:
    record = read_manifest(ROOT / "manifests" / "splits" / "engineering12.jsonl")[0]
    baseline = BaselineBatch.from_npz([Path(record.a3f_state_path)], "cpu")
    return baseline, TrustRegionSMPLX(
        MODEL_ROOT, baseline, ALL_ACTIVE, wrist_protection=wrist_protection,
    )


def _descendants(model: TrustRegionSMPLX, joint: int) -> list[int]:
    result = []
    for candidate in range(len(model.model.parents)):
        ancestor = candidate
        while ancestor >= 0:
            if ancestor == joint:
                result.append(candidate)
                break
            ancestor = int(model.model.parents[ancestor])
    return result


def test_exact_neutral_topology_and_a3f_decode_parity():
    baseline, model = _adapter(wrist_protection=False)
    with torch.no_grad():
        output = model()
    assert output["vertices"].shape == (1, 10475, 3)
    assert np.asarray(model.model.faces).shape == (20908, 3)
    assert array_sha256(np.asarray(model.model.faces, dtype=np.int64)) == EXPECTED_FACE_HASH
    assert torch.isfinite(output["vertices"]).all()
    assert torch.max(torch.abs(output["vertices"] - baseline.cached_vertices)) < 2e-6


def test_each_five_degree_residual_moves_its_descendants_only():
    _, model = _adapter(wrist_protection=False)
    with torch.no_grad():
        baseline_vertices = model()["vertices"]
        for parameter_index, slot in enumerate(model.active_slots):
            model.delta.zero_()
            model.delta[0, parameter_index, 2] = np.deg2rad(5.0)
            displacement = torch.linalg.vector_norm(
                model()["vertices"] - baseline_vertices, dim=-1,
            )[0]
            descendants = _descendants(model, slot + 1)
            descendant_weight = model.model.lbs_weights[:, descendants].sum(1)
            influenced = descendant_weight > 0.1
            unrelated = descendant_weight < 1e-8
            assert int(influenced.sum()) > 1000
            assert torch.median(displacement[influenced]) > 0.005
            # Pose blend shapes can create a small non-local numerical tail;
            # 99% of vertices with zero descendant skinning weight stay <1 mm.
            assert torch.quantile(displacement[unrelated], 0.99) < 0.001


def test_frozen_state_and_analytic_wrist_protection_on_real_decoder():
    baseline, model = _adapter(wrist_protection=True)
    frozen = {
        name: getattr(model, f"base_{name}").clone()
        for name in (
            "betas", "global_orient", "left_hand_pose", "right_hand_pose",
            "jaw_pose", "leye_pose", "reye_pose", "expression", "transl",
        )
    }
    camera = model.cameras.clone()
    with torch.no_grad():
        model.delta[0, ALL_ACTIVE.index("left_shoulder"), 2] = np.deg2rad(5.0)
        output = model()
        globals_ = model._global_rotations(model.root_R0, output["body_rotations"])
    for name, value in frozen.items():
        assert torch.equal(getattr(model, f"base_{name}"), value)
    assert torch.equal(model.cameras, camera)
    assert torch.linalg.matrix_norm(globals_[:, 20] - model.wrist_global_left0) < 1e-5
    assert torch.linalg.matrix_norm(globals_[:, 21] - model.wrist_global_right0) < 1e-5
    left_ids, right_ids = load_mano_vertex_ids(MODEL_ROOT)
    left_drift = translation_aligned_hand_distance(
        output["vertices"][:, left_ids], baseline.cached_vertices[:, left_ids],
    )
    right_drift = translation_aligned_hand_distance(
        output["vertices"][:, right_ids], baseline.cached_vertices[:, right_ids],
    )
    assert left_drift < 0.00025
    assert right_drift < 0.00025
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {"delta"}


def test_wrist_position_keeps_arm_gradient_while_fingers_are_frozen():
    _, model = _adapter(wrist_protection=True)
    output = model()
    target = output["joints"][:, 20:22].detach() + torch.tensor(
        [[[0.02, -0.03, 0.01], [-0.01, 0.02, 0.03]]], dtype=torch.float32,
    )
    torch.sum((output["joints"][:, 20:22] - target).square()).backward()
    for name in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow"):
        gradient = model.delta.grad[0, ALL_ACTIVE.index(name)]
        assert torch.linalg.vector_norm(gradient) > 1e-5
    assert not model.base_left_hand_pose.requires_grad
    assert not model.base_right_hand_pose.requires_grad


def test_numerical_wrist_projection_backward_has_no_inplace_version_error():
    baseline, model = _adapter(wrist_protection=True)
    with torch.no_grad():
        model.delta[0, ALL_ACTIVE.index("left_shoulder"), 2] = np.deg2rad(5.0)
    model.delta.requires_grad_(False)
    model.wrist_projection_delta.requires_grad_(True)
    left_ids, right_ids = load_mano_vertex_ids(MODEL_ROOT)
    output = model()
    objective = translation_aligned_hand_distance(
        output["vertices"][:, left_ids], baseline.cached_vertices[:, left_ids],
    ) + translation_aligned_hand_distance(
        output["vertices"][:, right_ids], baseline.cached_vertices[:, right_ids],
    )
    objective.mean().backward()
    assert model.wrist_projection_delta.grad is not None
    assert torch.isfinite(model.wrist_projection_delta.grad).all()
