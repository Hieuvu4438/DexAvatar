from pathlib import Path

import numpy as np

from signpk.data.frame_manifest import FrameRecord, SignManifest
from signpk.evaluation.trv2v_audited import AuditedTRV2VEvaluator
from signpk.geometry.topology import write_obj


def _manifest(tmp_path: Path):
    rgb = tmp_path / "frame.png"
    rgb.write_bytes(b"x")
    gt = tmp_path / "gt.obj"
    records = (FrameRecord(0, 5, 10, 10, 0.2, rgb, gt),)
    return SignManifest("Test", 5, 5, "~0", "unknown", "x2", "reflect", records), gt


def test_strict_pairing_and_translation_invariance(tmp_path: Path):
    manifest, gt_path = _manifest(tmp_path)
    vertices = np.zeros((4, 3))
    vertices[1:] = np.eye(3)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    write_obj(gt_path, vertices, faces)
    prediction = tmp_path / "prediction_000010.obj"
    write_obj(prediction, vertices + 7, faces)
    evaluator = AuditedTRV2VEvaluator(
        {"UBody(-F)": np.arange(4), "LHand": np.array([0, 1]), "RHand": np.array([2, 3])},
        vertex_count=4,
    )
    summary, _ = evaluator.evaluate_sign(manifest, {10: prediction})
    assert summary["UBody(-F)"].mean_mm == 0


def test_wrong_id_fails_even_when_count_matches(tmp_path: Path):
    manifest, gt_path = _manifest(tmp_path)
    vertices = np.zeros((4, 3))
    faces = np.array([[0, 1, 2]])
    write_obj(gt_path, vertices, faces)
    wrong = tmp_path / "prediction_11.obj"
    write_obj(wrong, vertices, faces)
    evaluator = AuditedTRV2VEvaluator(
        {"UBody(-F)": np.arange(4), "LHand": np.array([0]), "RHand": np.array([1])},
        vertex_count=4,
    )
    try:
        evaluator.evaluate_sign(manifest, {11: wrong})
    except ValueError as error:
        assert "frame-ID mismatch" in str(error)
    else:
        raise AssertionError("wrong frame ID was silently paired")


def test_rotation_and_topology_changes_are_detected(tmp_path: Path):
    manifest, gt_path = _manifest(tmp_path)
    vertices = np.zeros((4, 3))
    vertices[1:] = np.eye(3)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    write_obj(gt_path, vertices, faces)
    rotated = vertices.copy()
    rotated[:2] = rotated[:2] @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    prediction = tmp_path / "prediction_10.obj"
    write_obj(prediction, rotated, faces)
    evaluator = AuditedTRV2VEvaluator(
        {"UBody(-F)": np.arange(4), "LHand": np.array([0, 1]), "RHand": np.array([2, 3])},
        vertex_count=4,
    )
    summary, _ = evaluator.evaluate_sign(manifest, {10: prediction})
    assert summary["LHand"].mean_mm > 0

    write_obj(prediction, vertices, faces[::-1].copy())
    try:
        evaluator.evaluate_sign(manifest, {10: prediction})
    except AssertionError:
        pass
    else:
        raise AssertionError("changed face ordering passed strict topology validation")


def test_class0_excludes_left_hand_from_left_and_upper(tmp_path: Path):
    manifest, gt_path = _manifest(tmp_path)
    manifest = SignManifest(
        manifest.sign_name,
        manifest.segment_start,
        manifest.segment_end,
        "0",
        manifest.dominant_hand,
        manifest.sampling_policy,
        manifest.boundary_padding,
        manifest.records,
    )
    vertices = np.zeros((4, 3))
    vertices[1:] = np.eye(3)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    write_obj(gt_path, vertices, faces)
    changed = vertices.copy()
    changed[:2] *= 10
    prediction = tmp_path / "prediction_10.obj"
    write_obj(prediction, changed, faces)
    evaluator = AuditedTRV2VEvaluator(
        {"UBody(-F)": np.arange(4), "LHand": np.array([0, 1]), "RHand": np.array([2, 3])},
        vertex_count=4,
    )
    summary, _ = evaluator.evaluate_sign(manifest, {10: prediction})
    assert summary["LHand"].frames == 0
    assert summary["UBody(-F)"].mean_mm == 0
