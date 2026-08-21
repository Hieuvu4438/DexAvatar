from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.cache import ObservationBatch
from ..data.manifest import ClipManifest
from ..geometry.so3 import exp_map
from ..utils.hashing import sha256_file

SMPLX_JOINT_COUNT = 55
LEFT_WRIST = 20
RIGHT_WRIST = 21
LEFT_HAND_START = 25
RIGHT_HAND_START = 40
MANO_NON_TIP_INDICES = (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19)


def mirror_left_mano_points(points: torch.Tensor) -> torch.Tensor:
    """Undo WiLoR's right-hand canonicalization for a detected left hand."""
    if points.shape[-1] != 3:
        raise ValueError("MANO points must end in xyz")
    mirrored = points.clone()
    mirrored[..., 0] *= -1
    return mirrored


def mirror_left_mano_rotations(rotations: torch.Tensor) -> torch.Tensor:
    """Map canonical-right MANO rotations to the SMPL-X left-hand convention."""
    if rotations.shape[-2:] != (3, 3):
        raise ValueError("MANO rotations must end in 3x3")
    reflection = rotations.new_tensor([-1.0, 1.0, 1.0])
    return rotations * reflection[..., :, None] * reflection[..., None, :]


def _trusted_pickle(path: Path) -> Any:
    """Load only user-owned local artifacts and record their hashes in provenance."""
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _axis_angle_rotations(parameters: dict[str, np.ndarray]) -> torch.Tensor:
    pieces = [
        np.asarray(parameters["global_orient"]).reshape(1, 3),
        np.asarray(parameters["body_pose"]).reshape(21, 3),
        np.asarray(parameters.get("jaw_pose", np.zeros(3))).reshape(1, 3),
        np.asarray(parameters.get("leye_pose", np.zeros(3))).reshape(1, 3),
        np.asarray(parameters.get("reye_pose", np.zeros(3))).reshape(1, 3),
        np.asarray(parameters["left_hand_pose"]).reshape(15, 3),
        np.asarray(parameters["right_hand_pose"]).reshape(15, 3),
    ]
    axis_angle = torch.from_numpy(np.concatenate(pieces, axis=0)).float()
    if axis_angle.shape != (SMPLX_JOINT_COUNT, 3):
        raise ValueError(f"unexpected SMPL-X pose layout: {axis_angle.shape}")
    return exp_map(axis_angle)


def _forward_smplx_parameters(
    parameter_rows: list[dict[str, np.ndarray]], model_path: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        import smplx
    except ImportError as exc:
        raise RuntimeError("SGNify preprocessing requires the local smplx environment") from exc
    model = smplx.SMPLX(
        str(model_path),
        gender="neutral",
        ext=model_path.suffix.lstrip("."),
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        num_expression_coeffs=10,
    ).to(device)

    def tensor(key: str, width: int) -> torch.Tensor:
        return (
            torch.from_numpy(
                np.stack(
                    [
                        np.asarray(row.get(key, np.zeros(width))).reshape(width)
                        for row in parameter_rows
                    ]
                )
            )
            .float()
            .to(device)
        )

    with torch.inference_mode():
        output = model(
            global_orient=tensor("global_orient", 3),
            body_pose=tensor("body_pose", 63),
            left_hand_pose=tensor("left_hand_pose", 45),
            right_hand_pose=tensor("right_hand_pose", 45),
            jaw_pose=tensor("jaw_pose", 3),
            leye_pose=tensor("leye_pose", 3),
            reye_pose=tensor("reye_pose", 3),
            betas=tensor("betas", 10),
            expression=tensor("expression", 10),
            transl=tensor("transl", 3),
            return_verts=True,
        )
    rotations = torch.stack([_axis_angle_rotations(row) for row in parameter_rows]).to(device)
    betas = tensor("betas", 10)
    translation = tensor("transl", 3)
    return output.joints[:, :SMPLX_JOINT_COUNT], rotations, betas, translation


def build_sgnify_observation_cache(
    manifest: ClipManifest,
    body_root: str | Path,
    wilor_root: str | Path,
    model_path: str | Path,
    body_subpath: str = "smplerx/smplx",
    body_source_name: str = "smplerx",
    legacy_root: str | Path | None = None,
    legacy_subpath: str = "smplifyx/results",
    legacy_source_name: str = "legacy_dexavatar",
    device: str = "cpu",
) -> tuple[ObservationBatch, dict[str, object]]:
    body_root = Path(body_root)
    wilor_root = Path(wilor_root)
    model_path = Path(model_path)
    body_rows: list[dict[str, np.ndarray]] = []
    source_hashes: dict[str, str] = {}
    for frame_id in manifest.frame_ids:
        path = body_root / manifest.clip_id / body_subpath / f"low_{frame_id:03d}.pkl"
        if not path.is_file():
            raise FileNotFoundError(f"Body initialization missing for manifest frame: {path}")
        body_rows.append(_trusted_pickle(path))
        source_hashes[str(path)] = sha256_file(path)
    torch_device = torch.device(device)
    body_joints, body_rotations, betas, translation = _forward_smplx_parameters(
        body_rows, model_path, torch_device
    )
    camera_x_180 = body_joints.new_tensor([1.0, -1.0, -1.0])
    body_joints = body_joints * camera_x_180
    t, joint_count = body_joints.shape[:2]
    sources = 3 if legacy_root is not None else 2
    joints = torch.zeros((t, sources, joint_count, 3), device=torch_device)
    rotations = torch.eye(3, device=torch_device).expand(t, sources, joint_count, 3, 3).clone()
    valid_3d = torch.zeros((t, sources, joint_count), dtype=torch.bool, device=torch_device)
    valid_rot = torch.zeros_like(valid_3d)
    features = torch.zeros((t, sources, joint_count, 8), device=torch_device)
    keypoints_2d = torch.zeros((t, sources, joint_count, 2), device=torch_device)
    valid_2d = torch.zeros((t, sources, joint_count), dtype=torch.bool, device=torch_device)
    camera_k = torch.eye(3, device=torch_device).expand(t, 3, 3).clone()
    image_size = torch.zeros((t, 2), device=torch_device)
    for frame_index, parameters in enumerate(body_rows):
        focal = np.asarray(parameters["focal"]).reshape(2)
        principal = np.asarray(parameters["princpt"]).reshape(2)
        camera_k[frame_index, 0, 0] = float(focal[0])
        camera_k[frame_index, 1, 1] = float(focal[1])
        camera_k[frame_index, 0, 2] = float(principal[0])
        camera_k[frame_index, 1, 2] = float(principal[1])
        image_size[frame_index] = image_size.new_tensor(
            [2 * float(principal[1]), 2 * float(principal[0])]
        )
    joints[:, 0] = body_joints
    rotations[:, 0] = body_rotations
    valid_3d[:, 0] = True
    valid_rot[:, 0] = True
    features[:, 0, :, 0] = 0.35  # source prior risk; not a calibrated score

    legacy_betas_mean: list[float] | None = None
    if legacy_root is not None:
        legacy_root = Path(legacy_root)
        available_rows: list[dict[str, np.ndarray]] = []
        available_indices: list[int] = []
        for frame_index, frame_id in enumerate(manifest.frame_ids):
            path = legacy_root / manifest.clip_id / legacy_subpath / f"low_{frame_id:03d}.pkl"
            if path.is_file():
                available_rows.append(_trusted_pickle(path))
                available_indices.append(frame_index)
                source_hashes[str(path)] = sha256_file(path)
        if available_rows:
            legacy_joints, legacy_rotations, legacy_betas, _ = _forward_smplx_parameters(
                available_rows, model_path, torch_device
            )
            legacy_betas_mean = legacy_betas.mean(0).detach().cpu().tolist()
            index = torch.tensor(available_indices, device=torch_device)
            joints[index, 2] = legacy_joints * camera_x_180
            rotations[index, 2] = legacy_rotations
            valid_3d[index, 2] = True
            valid_rot[index, 2] = True
            features[index, 2, :, 0] = 0.15

    wilor_path = wilor_root / manifest.clip_id / "wilor" / "wilor.pkl"
    if not wilor_path.is_file():
        raise FileNotFoundError(wilor_path)
    wilor = _trusted_pickle(wilor_path)
    source_hashes[str(wilor_path)] = sha256_file(wilor_path)
    image_records = wilor.get("images", {})
    for frame_index, frame_id in enumerate(manifest.frame_ids):
        record = image_records.get(f"low_{frame_id:03d}.png", {})
        hands = record.get("hands", []) if isinstance(record, dict) else []
        for hand in hands:
            right = bool(round(float(hand["is_right"])))
            wrist_index = RIGHT_WRIST if right else LEFT_WRIST
            hand_start = RIGHT_HAND_START if right else LEFT_HAND_START
            hand_joints = (
                torch.from_numpy(np.asarray(hand["pred_keypoints_3d"])).float().to(torch_device)
            )
            if hand_joints.shape != (21, 3):
                continue
            if not right:
                hand_joints = mirror_left_mano_points(hand_joints)
            selected = hand_joints[list(MANO_NON_TIP_INDICES)]
            aligned = (selected - selected[0]) * camera_x_180 + body_joints[
                frame_index, wrist_index
            ]
            destination = torch.tensor(
                [wrist_index, *range(hand_start, hand_start + 15)], device=torch_device
            )
            joints[frame_index, 1, destination] = aligned
            valid_3d[frame_index, 1, destination] = True
            hand_keypoints_2d = (
                torch.from_numpy(np.asarray(hand["pred_keypoints_2d"])).float().to(torch_device)
            )
            if hand_keypoints_2d.shape == (21, 2):
                hand_keypoints_2d[:, 0] *= 1.0 if right else -1.0
                hand_keypoints_2d *= float(hand["box_size"])
                hand_keypoints_2d += (
                    torch.from_numpy(np.asarray(hand["box_center"])).float().to(torch_device)
                )
                keypoints_2d[frame_index, 1, destination] = hand_keypoints_2d[
                    list(MANO_NON_TIP_INDICES)
                ]
                valid_2d[frame_index, 1, destination] = True
            hand_rotations = (
                torch.from_numpy(np.asarray(hand["pred_mano_pose_rotmat"])).float().to(torch_device)
            )
            if hand_rotations.shape == (15, 3, 3):
                if not right:
                    hand_rotations = mirror_left_mano_rotations(hand_rotations)
                rotations[frame_index, 1, hand_start : hand_start + 15] = hand_rotations
                valid_rot[frame_index, 1, hand_start : hand_start + 15] = True
            box_size = float(hand.get("box_size", 0.0))
            features[frame_index, 1, destination, 0] = 0.2
            features[frame_index, 1, destination, 1] = 1.0 / max(box_size, 1.0)
            features[frame_index, 1, destination, 2] = abs(
                float(hand["is_right"]) - round(float(hand["is_right"]))
            )

    missing_run = torch.zeros((t, sources, joint_count), device=torch_device)
    for frame_index in range(t):
        missing_run[frame_index] = (~valid_3d[frame_index]).float()
        if frame_index:
            missing_run[frame_index] *= 1 + missing_run[frame_index - 1]
    features[..., 3] = missing_run
    for source in range(sources):
        features[:, source, :, 4] = torch.linalg.vector_norm(
            joints[:, 0] - joints[:, source], dim=-1
        )
    features[..., 5] = (~valid_3d).float()
    features[..., 6] = torch.linspace(0, 1, t, device=torch_device)[:, None, None]
    features[..., 7] = torch.arange(sources, device=torch_device)[None, :, None]
    batch = ObservationBatch(
        frame_ids=torch.tensor(manifest.frame_ids, dtype=torch.int64, device=torch_device),
        joints_3d=joints,
        valid_3d=valid_3d,
        features=features,
        keypoints_2d=keypoints_2d,
        valid_2d=valid_2d,
        rotations=rotations,
        valid_rot=valid_rot,
        camera_K=camera_k,
        image_size=image_size,
    )
    metadata = {
        "schema_version": "1.0",
        "clip_id": manifest.clip_id,
        "sources": [
            {"source_id": 0, "name": body_source_name, "role": "body_initializer"},
            {"source_id": 1, "name": "wilor", "role": "hand_hypothesis"},
        ]
        + (
            [{"source_id": 2, "name": legacy_source_name, "role": "fitted_hypothesis"}]
            if legacy_root is not None
            else []
        ),
        "camera_convention": "opencv_x_right_y_down_z_forward",
        "length_unit": "meter",
        "rotation_convention": "matrix_local_parent_to_child",
        "smplx_model_sha256": sha256_file(model_path),
        "source_hashes": source_hashes,
        "betas_mean": betas.mean(0).detach().cpu().tolist(),
        "legacy_betas_mean": legacy_betas_mean,
        "translation": translation.detach().cpu().tolist(),
        "uncertainty_status": "features_only_uncalibrated",
    }
    batch.validate_against(manifest)
    return batch, metadata
