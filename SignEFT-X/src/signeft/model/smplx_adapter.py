from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Sequence

import numpy as np
import torch
from torch import nn

from signeft.model.kinematics import (
    apply_lie_residual, compensate_wrist, so3_exp_map, so3_log_map,
)


BODY_JOINT_NAMES = (
    "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)
FULL_JOINT_NAMES = ("pelvis",) + BODY_JOINT_NAMES
BOUNDARY_X180 = torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32)
RADIUS_DEG = {
    "spine1": 5.0, "spine2": 5.0, "spine3": 5.0, "neck": 5.0,
    "left_collar": 7.0, "right_collar": 7.0,
    "left_shoulder": 10.0, "right_shoulder": 10.0,
    "left_elbow": 8.0, "right_elbow": 8.0,
}


def resolve_joint_indices(names: Sequence[str], required: Sequence[str]) -> dict[str, int]:
    index = {name: i for i, name in enumerate(names)}
    missing = [name for name in required if name not in index]
    if missing:
        raise KeyError(f"missing SMPL-X joints: {missing}")
    result = {name: index[name] for name in required}
    if len(set(result.values())) != len(result):
        raise ValueError("joint mapping is not one-to-one")
    return result


@dataclass
class BaselineBatch:
    arrays: dict[str, torch.Tensor]
    cached_vertices: torch.Tensor
    cameras: torch.Tensor

    @classmethod
    def from_npz(cls, paths: Sequence[Path], device: str) -> "BaselineBatch":
        keys = (
            "betas", "global_orient", "body_pose", "left_hand_pose", "right_hand_pose",
            "jaw_pose", "leye_pose", "reye_pose", "expression", "transl",
        )
        loaded: list[dict[str, np.ndarray]] = []
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                if str(archive["coord_frame"]) != "evaluator_camera" or str(archive["unit"]) != "meter":
                    raise RuntimeError(f"baseline coordinate contract mismatch: {path}")
                loaded.append({
                    key: np.asarray(archive[key], dtype=np.float32)
                    for key in (*keys, "vertices", "K")
                })
        arrays = {
            key: torch.as_tensor(np.concatenate([item[key] for item in loaded], axis=0), device=device)
            for key in keys
        }
        cached_vertices = torch.as_tensor(
            np.stack([item["vertices"] for item in loaded]), dtype=torch.float32, device=device
        )
        cameras = torch.as_tensor(
            np.stack([item["K"] for item in loaded]), dtype=torch.float32, device=device
        )
        return cls(arrays=arrays, cached_vertices=cached_vertices, cameras=cameras)


class TrustRegionSMPLX(nn.Module):
    """SMPL-X decoder with bounded left-composed body residuals."""

    def __init__(
        self,
        model_root: Path,
        baseline: BaselineBatch,
        active_names: Sequence[str],
        *,
        wrist_protection: bool,
    ) -> None:
        super().__init__()
        import smplx

        mapping = resolve_joint_indices(BODY_JOINT_NAMES, active_names)
        self.active_names = tuple(active_names)
        self.active_slots = tuple(mapping[name] for name in self.active_names)
        self.wrist_protection = bool(wrist_protection)
        self.model = smplx.create(
            str(model_root), model_type="smplx", gender="neutral", num_betas=10,
            use_pca=False, use_face_contour=True,
        ).to(baseline.cached_vertices.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.delta = nn.Parameter(torch.zeros(
            (baseline.cached_vertices.shape[0], len(self.active_names), 3),
            dtype=torch.float32, device=baseline.cached_vertices.device,
        ))
        self.wrist_projection_delta = nn.Parameter(torch.zeros(
            (baseline.cached_vertices.shape[0], 2, 3),
            dtype=torch.float32, device=baseline.cached_vertices.device,
        ), requires_grad=False)
        for key, value in baseline.arrays.items():
            self.register_buffer(f"base_{key}", value.detach().clone())
        self.register_buffer("cached_vertices", baseline.cached_vertices.detach().clone())
        self.register_buffer("cameras", baseline.cameras.detach().clone())
        radii = [np.deg2rad(RADIUS_DEG[name]) for name in self.active_names]
        self.register_buffer(
            "radii", torch.as_tensor(
                radii, dtype=torch.float32, device=baseline.cached_vertices.device
            )
        )
        self.register_buffer("boundary", BOUNDARY_X180.to(baseline.cached_vertices.device))
        parents = self.model.parents.detach().cpu().tolist()[:22]
        self.parents = tuple(int(item) for item in parents)
        if self.parents[20] != 18 or self.parents[21] != 19:
            raise RuntimeError(f"unexpected SMPL-X wrist parents: {self.parents[20:22]}")
        self._make_baseline_rotations()

    def _make_baseline_rotations(self) -> None:
        self.register_buffer("root_R0", so3_exp_map(self.base_global_orient.reshape(-1, 1, 3)))
        self.register_buffer("body_R0", so3_exp_map(self.base_body_pose.reshape(-1, 21, 3)))
        self.register_buffer("left_R0", so3_exp_map(self.base_left_hand_pose.reshape(-1, 15, 3)))
        self.register_buffer("right_R0", so3_exp_map(self.base_right_hand_pose.reshape(-1, 15, 3)))
        self.register_buffer("jaw_R0", so3_exp_map(self.base_jaw_pose.reshape(-1, 1, 3)))
        self.register_buffer("leye_R0", so3_exp_map(self.base_leye_pose.reshape(-1, 1, 3)))
        self.register_buffer("reye_R0", so3_exp_map(self.base_reye_pose.reshape(-1, 1, 3)))
        with torch.no_grad():
            baseline_global = self._global_rotations(self.root_R0, self.body_R0)
        self.register_buffer("wrist_global_left0", baseline_global[:, 20].clone())
        self.register_buffer("wrist_global_right0", baseline_global[:, 21].clone())

    def _global_rotations(self, root: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        local = torch.cat((root, body), dim=1)
        globals_: list[torch.Tensor] = [local[:, 0]]
        for joint in range(1, 22):
            globals_.append(globals_[self.parents[joint]] @ local[:, joint])
        return torch.stack(globals_, dim=1)

    def body_rotations(self) -> tuple[torch.Tensor, torch.Tensor]:
        body_items = [self.body_R0[:, slot] for slot in range(self.body_R0.shape[1])]
        bounded_items = []
        for parameter_index, slot in enumerate(self.active_slots):
            rotation, bounded = apply_lie_residual(
                self.body_R0[:, slot], self.delta[:, parameter_index], self.radii[parameter_index]
            )
            body_items[slot] = rotation
            bounded_items.append(bounded)
        bounded = torch.stack(bounded_items, dim=1) if bounded_items else self.delta
        body = torch.stack(body_items, dim=1)
        if self.wrist_protection:
            global_before_wrist = self._global_rotations(self.root_R0, body)
            protected = (
                compensate_wrist(global_before_wrist[:, 18], self.wrist_global_left0),
                compensate_wrist(global_before_wrist[:, 19], self.wrist_global_right0),
            )
            for side_index, (wrist_name, compensated) in enumerate(zip(
                ("left_wrist", "right_wrist"), protected,
            )):
                slot = BODY_JOINT_NAMES.index(wrist_name)
                body_items[slot], _ = apply_lie_residual(
                    compensated, self.wrist_projection_delta[:, side_index],
                    np.deg2rad(1.0),
                )
            body = torch.stack(body_items, dim=1)
        return body, bounded

    def forward(self) -> dict[str, torch.Tensor]:
        body, bounded = self.body_rotations()
        body_axis_angle = self.base_body_pose.reshape(-1, 21, 3).clone()
        converted_slots = set(self.active_slots)
        if self.wrist_protection:
            converted_slots.update((BODY_JOINT_NAMES.index("left_wrist"), BODY_JOINT_NAMES.index("right_wrist")))
        for slot in sorted(converted_slots):
            body_axis_angle[:, slot] = so3_log_map(body[:, slot])
        output = self.model(
            betas=self.base_betas,
            global_orient=self.base_global_orient,
            body_pose=body_axis_angle.reshape(-1, 63),
            left_hand_pose=self.base_left_hand_pose,
            right_hand_pose=self.base_right_hand_pose,
            jaw_pose=self.base_jaw_pose,
            leye_pose=self.base_leye_pose,
            reye_pose=self.base_reye_pose,
            expression=self.base_expression,
            transl=self.base_transl,
            return_verts=True,
        )
        return {
            "vertices": output.vertices * self.boundary,
            "joints": output.joints * self.boundary,
            "body_rotations": body,
            "body_axis_angle": body_axis_angle,
            "bounded_delta": bounded,
        }


def load_mano_vertex_ids(model_root: Path) -> tuple[np.ndarray, np.ndarray]:
    path = model_root / "smplx" / "MANO_SMPLX_vertex_ids.pkl"
    with path.open("rb") as handle:
        value = pickle.load(handle)
    left = np.asarray(value["left_hand"], dtype=np.int64)
    right = np.asarray(value["right_hand"], dtype=np.int64)
    if left.shape != (778,) or right.shape != (778,):
        raise RuntimeError("MANO/SMPL-X vertex mapping contract changed")
    return left, right


def semantic_vertex_ids(model: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive off-target regions from neutral SMPL-X skinning weights."""
    weights = model.lbs_weights
    lower_joints = torch.as_tensor((1, 2, 4, 5, 7, 8, 10, 11), device=weights.device)
    face_joints = torch.as_tensor((15, 22, 23, 24), device=weights.device)
    lower = torch.where(weights.index_select(1, lower_joints).sum(1) > 0.5)[0]
    face = torch.where(weights.index_select(1, face_joints).sum(1) > 0.5)[0]
    if len(lower) < 1000 or len(face) < 1000:
        raise RuntimeError("SMPL-X semantic region derivation failed")
    return face, lower
