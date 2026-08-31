from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from signpccx.data.manifest import FrameRecord


ANCHOR_JOINT_IDS = np.asarray(
    [-1, 12, 16, 17, 18, 19, 20, 21,
     -1, 37, 25, 28, 34, 31, 66, 67, 68, 69, 70,
     -1, 52, 40, 43, 49, 46, 71, 72, 73, 74, 75],
    dtype=np.int64,
)
LEFT_ANCHORS = tuple(range(8, 19))
RIGHT_ANCHORS = tuple(range(19, 30))
BODY_ANCHORS = tuple(range(8))


@dataclass(frozen=True)
class FrameObservation:
    record: FrameRecord
    cache_path: Path
    arrays: dict[str, np.ndarray]
    image_wh: tuple[int, int]
    smplerx: dict[str, np.ndarray] | None
    hamer_fingers: dict[str, np.ndarray]

    @property
    def crop_hw(self) -> tuple[int, int]:
        # H4W++ body crop is fixed at width=192, height=256.
        return (256, 192)


_REQUIRED = {
    "K_crop": (3, 3), "image_to_crop": (2, 3), "crop_to_image": (2, 3),
    "smplx_beta": (10,), "smplx_root_pose_aa": (3,), "smplx_body_pose_aa": (63,),
    "smplx_left_hand_pose_aa": (45,), "smplx_right_hand_pose_aa": (45,),
    "smplx_jaw_pose_aa": (3,), "smplx_expression": (10,), "smplx_translation": (3,),
    "mesh_parametric_init": (10475, 3), "mesh_hybrid_init": (10475, 3),
    "smplx_joints_parametric": (144, 3), "anchor_uv_observed": (30, 2),
    "anchor_uv_confidence": (30,), "anchor_valid": (30,), "init_anchor_cam": (30, 3),
    "dwpose_keypoints": (137, 3), "wilor_left_hand_pose_aa": (45,),
    "wilor_right_hand_pose_aa": (45,), "wilor_left_joints": (21, 3),
    "wilor_right_joints": (21, 3),
}


def _validate_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    for key, shape in _REQUIRED.items():
        if key not in arrays:
            raise KeyError(f"{path}: missing {key}")
        value = arrays[key]
        if value.shape != shape:
            raise ValueError(f"{path}: {key} shape {value.shape} != {shape}")
        if value.dtype == object:
            raise TypeError(f"{path}: object dtype for {key}")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise FloatingPointError(f"{path}: non-finite {key}")


def _matrix_to_axis_angle_numpy(matrices: np.ndarray) -> np.ndarray:
    import torch
    from signpccx.optimization.hypotheses import _matrix_to_axis_angle

    tensor = torch.as_tensor(matrices, dtype=torch.float32)
    return _matrix_to_axis_angle(tensor).cpu().numpy().astype(np.float32)


def _load_hamer(path: Path, frame_id: int) -> dict[str, np.ndarray]:
    if not path.is_file():
        return {}
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # trusted local inference artifact
    entry = payload.get(f"low_{frame_id}.png")
    if not isinstance(entry, (list, tuple)) or len(entry) < 4:
        return {}
    rotations = entry[0].get("pred_mano_params", {}).get("hand_pose")
    if rotations is None:
        return {}
    rotations = np.asarray(rotations.detach().cpu() if hasattr(rotations, "detach") else rotations)
    flags = np.asarray(entry[3].detach().cpu() if hasattr(entry[3], "detach") else entry[3]).reshape(-1)
    poses = _matrix_to_axis_angle_numpy(rotations)
    result: dict[str, np.ndarray] = {}
    for index, flag in enumerate(flags.astype(int)):
        side = "right" if flag == 1 else "left"
        pose = poses[index].copy()
        if side == "left":
            pose[:, 1:] *= -1.0
        result.setdefault(side, pose.reshape(45))
    return result


def _load_smplerx(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        raw = pickle.load(handle)  # trusted local inference artifact
    keys = ("global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "transl")
    result = {key: np.asarray(raw[key], dtype=np.float32).reshape(-1) for key in keys}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError(f"{path}: non-finite SMPLer-X state")
    return result


def load_frame_observation(
    record: FrameRecord,
    cache_root: Path,
    initializer_root: Path | None = None,
) -> FrameObservation:
    from PIL import Image

    path = cache_root / "clips" / record.sign / f"{record.source_frame_id:06d}.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    _validate_arrays(path, arrays)
    with Image.open(record.source_path) as image:
        image_wh = tuple(map(int, image.size))
    smplerx = None
    hamer: dict[str, np.ndarray] = {}
    if initializer_root is not None:
        base = initializer_root / record.sign
        smplerx = _load_smplerx(base / "smplerx" / "smplx" / f"low_{record.source_frame_id}.pkl")
        hamer = _load_hamer(base / "hamer" / "hamer.pkl", record.source_frame_id)
    return FrameObservation(record, path, arrays, image_wh, smplerx, hamer)


def observation_initial_state(observation: FrameObservation, device: str):
    import torch
    from signpccx.model.smplx_state import FrameState

    a = observation.arrays
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    return FrameState({
        "global_orient": tensor(a["smplx_root_pose_aa"]),
        "body_pose": tensor(a["smplx_body_pose_aa"]),
        "left_hand_pose": tensor(a["smplx_left_hand_pose_aa"]),
        "right_hand_pose": tensor(a["smplx_right_hand_pose_aa"]),
        "jaw_pose": tensor(a["smplx_jaw_pose_aa"]),
        "expression": tensor(a["smplx_expression"]),
        "transl": tensor(a["smplx_translation"]),
    }).to(device)


def validate_h4wpp_frame(
    observation: FrameObservation,
    model_root: Path,
    output_root: Path,
    device: str = "cpu",
) -> dict[str, object]:
    """Forward the cached SMPL-X state and write a full-image anchor overlay."""
    import os
    import smplx
    import torch
    from PIL import Image, ImageDraw
    from signpccx.io import atomic_write_json, sha256_file
    from signpccx.optimization.losses import safe_project

    state = observation_initial_state(observation, device)
    beta = torch.as_tensor(
        observation.arrays["smplx_beta"], dtype=torch.float32, device=device
    ).reshape(1, 10)
    model = smplx.create(
        str(model_root), model_type="smplx", gender="neutral", num_betas=10,
        use_pca=False, use_face_contour=True,
    ).to(device)
    model.eval()
    with torch.no_grad():
        output = model(**state.smplx_kwargs(beta))
    target = torch.as_tensor(
        observation.arrays["mesh_parametric_init"], dtype=torch.float32, device=device
    ).unsqueeze(0)
    residual = torch.linalg.vector_norm(output.vertices - target, dim=-1)
    if float(residual.max()) >= 0.001:
        raise RuntimeError(f"H4W++ canonical forward mismatch: max={float(residual.max()) * 1000:.3f} mm")

    # Keep the same explicit anchor construction as the optimizer contract.
    body_ids = [12, 16, 17, 18, 19, 20, 21]
    left_ids = [20, 37, 25, 28, 34, 31, 66, 67, 68, 69, 70]
    right_ids = [21, 52, 40, 43, 49, 46, 71, 72, 73, 74, 75]
    left = output.joints[:, left_ids]
    right = output.joints[:, right_ids]
    chest = (output.joints[:, 16:17] + output.joints[:, 17:18]) * 0.5
    anchors = torch.cat((
        chest, output.joints[:, body_ids], left[:, 1:6].mean(1, keepdim=True), left[:, 1:],
        right[:, 1:6].mean(1, keepdim=True), right[:, 1:],
    ), dim=1)
    intrinsic = torch.as_tensor(observation.arrays["K_crop"], dtype=torch.float32, device=device)
    projected_crop = safe_project(anchors, intrinsic).cpu().numpy()[0]
    projected_full = _apply_affine(observation.arrays["crop_to_image"], projected_crop)
    observed_full = _apply_affine(
        observation.arrays["crop_to_image"], observation.arrays["anchor_uv_observed"]
    )
    valid = observation.arrays["anchor_valid"] & (observation.arrays["anchor_uv_confidence"] > 0)
    reprojection = np.linalg.norm(
        projected_crop[valid] - observation.arrays["anchor_uv_observed"][valid], axis=1
    )
    cache_projection_error = np.linalg.norm(
        projected_crop - observation.arrays["anchor_uv"], axis=1
    )
    image = Image.open(observation.record.source_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for point in observed_full[valid]:
        x, y = map(float, point)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 0))
    for point in projected_full[valid]:
        x, y = map(float, point)
        draw.line((x - 3, y, x + 3, y), fill=(255, 0, 0), width=1)
        draw.line((x, y - 3, x, y + 3), fill=(255, 0, 0), width=1)
    output_root.mkdir(parents=True, exist_ok=True)
    overlay = output_root / f"{observation.record.sign}_{observation.record.source_frame_id:06d}_overlay.png"
    temporary = overlay.with_suffix(".png.tmp")
    image.save(temporary, format="PNG")
    os.replace(temporary, overlay)
    report = {
        "schema_version": "signpccx.h4wpp-one-frame-gate.v1",
        "sign": observation.record.sign, "frame_id": observation.record.source_frame_id,
        "mesh_mean_v2v_mm": float(residual.mean() * 1000),
        "mesh_max_v2v_mm": float(residual.max() * 1000),
        "anchor_reprojection_mean_px": float(reprojection.mean()),
        "anchor_reprojection_max_px": float(reprojection.max()),
        "cache_projection_parity_mean_px": float(cache_projection_error.mean()),
        "cache_projection_parity_max_px": float(cache_projection_error.max()),
        "valid_anchors": int(valid.sum()), "overlay": str(overlay.resolve()),
        "overlay_sha256": sha256_file(overlay), "status": "ok",
    }
    atomic_write_json(output_root / f"{observation.record.sign}_{observation.record.source_frame_id:06d}.json", report)
    return report


def _apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((
        np.asarray(points, dtype=np.float32), np.ones((len(points), 1), dtype=np.float32)
    ), axis=1)
    return (np.asarray(affine, dtype=np.float32) @ homogeneous.T).T
