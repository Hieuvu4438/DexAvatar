from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from signpk.data.cache_schema import BodyObservation, HandObservation
from signpk.geometry.rotations import matrix_to_rotation_6d, so3_distance, so3_log


# Combined [root, body joints 1..21] indices in the SMPL-X kinematic tree.
UPPER_BODY_INDICES = (0, 3, 6, 9, 12, 15, 13, 14, 16, 17, 18, 19, 20, 21)

# MANO pose joints follow index/middle/pinky/ring/thumb, whereas the canonical
# 21-keypoint layout follows thumb/index/middle/ring/pinky.
MANO_POSE_KEYPOINT_INDICES = (5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3)
MANO_POSE_PARENT_KEYPOINT_INDICES = (0, 5, 6, 0, 9, 10, 0, 17, 18, 0, 13, 14, 0, 1, 2)
FINGERTIP_INDICES = (4, 8, 12, 16, 20)


@dataclass
class ExplicitTokenBatch:
    body: Tensor  # [B,T,14,12]
    left: Tensor  # [B,T,15,54]
    right: Tensor  # [B,T,15,54]
    relation: Tensor  # [B,T,20]
    timestamps: Tensor  # [B,T]
    upper_base_rotmat: Tensor  # [B,T,14,3,3]
    left_base_rotmat: Tensor  # [B,T,15,3,3]
    right_base_rotmat: Tensor  # [B,T,15,3,3]
    left_valid: Tensor  # [B,T]
    right_valid: Tensor  # [B,T]
    disagreement: Tensor  # [B,T,2,2]
    left_observer_feature: Tensor | None = None  # [B,T,F_omni]
    right_observer_feature: Tensor | None = None  # [B,T,F_omni]
    left_h4w_feature: Tensor | None = None  # [B,T,F_h4w_hand]
    right_h4w_feature: Tensor | None = None  # [B,T,F_h4w_hand]
    body_observer_feature: Tensor | None = None  # [B,T,F_h4w] or [B,T,14,F_h4w]

    def validate(self) -> None:
        batch, time = self.timestamps.shape
        expected = {
            "body": (batch, time, 14, 12),
            "left": (batch, time, 15, 54),
            "right": (batch, time, 15, 54),
            "relation": (batch, time, 20),
            "upper_base_rotmat": (batch, time, 14, 3, 3),
            "left_base_rotmat": (batch, time, 15, 3, 3),
            "right_base_rotmat": (batch, time, 15, 3, 3),
            "left_valid": (batch, time),
            "right_valid": (batch, time),
            "disagreement": (batch, time, 2, 2),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name}: expected {shape}, got {tuple(value.shape)}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN/Inf")
        for name in (
            "left_observer_feature",
            "right_observer_feature",
            "left_h4w_feature",
            "right_h4w_feature",
        ):
            value = getattr(self, name)
            if value is not None:
                if value.ndim != 3 or tuple(value.shape[:2]) != (batch, time):
                    raise ValueError(f"{name} must have shape [B,T,F]")
                if not torch.isfinite(value).all():
                    raise ValueError(f"{name} contains NaN/Inf")
        if self.body_observer_feature is not None:
            value = self.body_observer_feature
            valid_shape = (value.ndim == 3 and tuple(value.shape[:2]) == (batch, time)) or (
                value.ndim == 4 and tuple(value.shape[:3]) == (batch, time, 14)
            )
            if not valid_shape:
                raise ValueError("body_observer_feature must have shape [B,T,F] or [B,T,14,F]")
            if not torch.isfinite(value).all():
                raise ValueError("body_observer_feature contains NaN/Inf")


def _finite_difference(values: Tensor, timestamps: Tensor) -> Tensor:
    velocity = torch.zeros_like(values)
    dt = (timestamps[:, 1:] - timestamps[:, :-1]).clamp_min(1e-6)
    while dt.ndim < values.ndim:
        dt = dt.unsqueeze(-1)
    velocity[:, 1:] = (values[:, 1:] - values[:, :-1]) / dt
    velocity[:, 0] = velocity[:, 1] if values.shape[1] > 1 else 0
    return velocity


def _bbox_iou(left: Tensor, right: Tensor) -> Tensor:
    top_left = torch.maximum(left[..., :2], right[..., :2])
    bottom_right = torch.minimum(left[..., 2:], right[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
    left_area = (left[..., 2:] - left[..., :2]).clamp_min(0).prod(-1)
    right_area = (right[..., 2:] - right[..., :2]).clamp_min(0).prod(-1)
    return intersection / (left_area + right_area - intersection).clamp_min(1e-8)


def observer_disagreement(h4w: HandObservation, omni: HandObservation) -> Tensor:
    h4w_vertices = h4w.vertices_local - h4w.vertices_local.mean(-2, keepdim=True)
    omni_vertices = omni.vertices_local - omni.vertices_local.mean(-2, keepdim=True)
    if h4w_vertices.shape[-2] != omni_vertices.shape[-2]:
        vertex = torch.linalg.vector_norm(h4w.joints_local - omni.joints_local, dim=-1).mean(-1)
    else:
        vertex = torch.linalg.vector_norm(h4w_vertices - omni_vertices, dim=-1).mean(-1)
    palm = so3_distance(h4w.palm_rotmat, omni.palm_rotmat)
    return torch.stack([vertex, palm], dim=-1)


class ExplicitTokenBuilder:
    """Build the fixed-dimensional palm-kinematic streams used by PKC."""

    def __init__(self, torso_joint_indices: tuple[int, int, int] = (7, 8, 9)):
        self.neck_index, self.left_shoulder_index, self.right_shoulder_index = torso_joint_indices

    def _body_tokens(self, body: BodyObservation) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return semantic body tokens and wrist-relative geometry.

        H4W++'s first 25 regressed keypoints use its documented body layout:
        pelvis=0, neck=7, shoulders=8/9, elbows=10/11, wrists=12/13,
        nose=24. The SMPL-X spine/collar locations are interpolated explicitly.
        """

        if body.joints3d.shape[-2] < 25:
            raise ValueError("body observer must expose the 25 H4W++ body keypoints")
        joints = body.joints3d
        pelvis = joints[..., 0, :]
        neck = joints[..., self.neck_index, :]
        left_shoulder = joints[..., self.left_shoulder_index, :]
        right_shoulder = joints[..., self.right_shoulder_index, :]
        torso_scale = torch.linalg.vector_norm(left_shoulder - right_shoulder, dim=-1).clamp_min(
            1e-4
        )
        spine = neck - pelvis
        semantic_positions = torch.stack(
            [
                pelvis,
                pelvis + 0.25 * spine,
                pelvis + 0.50 * spine,
                pelvis + 0.75 * spine,
                neck,
                joints[..., 24, :],
                0.5 * (neck + left_shoulder),
                0.5 * (neck + right_shoulder),
                left_shoulder,
                right_shoulder,
                joints[..., 10, :],
                joints[..., 11, :],
                joints[..., 12, :],
                joints[..., 13, :],
            ],
            dim=-2,
        )
        relative_positions = (semantic_positions - pelvis[..., None, :]) / torso_scale[
            ..., None, None
        ]
        combined = torch.cat([body.root_rotmat[:, :, None], body.body_rotmat], dim=2)
        upper = combined[:, :, UPPER_BODY_INDICES]
        shape_norm = torch.linalg.vector_norm(body.shape, dim=-1)
        confidence = torch.ones_like(shape_norm)
        scalars = torch.stack([torso_scale, shape_norm, confidence], dim=-1)
        scalars = scalars[:, :, None].expand(-1, -1, len(UPPER_BODY_INDICES), -1)
        tokens = torch.cat([matrix_to_rotation_6d(upper), relative_positions, scalars], dim=-1)
        return tokens, upper, torso_scale, semantic_positions

    @staticmethod
    def _body_relative_hand_location(
        semantic_positions: Tensor,
        torso_scale: Tensor,
        side: str,
    ) -> Tensor:
        wrist_index = 12 if side == "left" else 13
        shoulder_index = 8 if side == "left" else 9
        wrist = semantic_positions[..., wrist_index, :]
        neck = semantic_positions[..., 4, :]
        shoulder = semantic_positions[..., shoulder_index, :]
        sternum = (
            semantic_positions[..., 4, :]
            + semantic_positions[..., 8, :]
            + semantic_positions[..., 9, :]
        ) / 3
        normalized = torch.cat([wrist - neck, wrist - shoulder, wrist - sternum], dim=-1)
        normalized = normalized / torso_scale[..., None]
        relative_depth = ((wrist[..., 2] - semantic_positions[..., 0, 2]) / torso_scale)[..., None]
        return torch.cat([normalized, relative_depth], dim=-1)

    @staticmethod
    def _normalized_bbox(bbox: Tensor, principal_point: Tensor) -> Tensor:
        image_size = (2 * principal_point).clamp_min(1.0)
        center = 0.5 * (bbox[..., :2] + bbox[..., 2:]) / image_size
        side_lengths = (bbox[..., 2:] - bbox[..., :2]).clamp_min(0)
        scale = torch.sqrt(side_lengths.prod(-1).clamp_min(1e-8)) / image_size.amax(-1)
        return torch.cat([center, scale[..., None]], dim=-1)

    def _hand_tokens(
        self,
        h4w: HandObservation,
        omni: HandObservation,
        timestamps: Tensor,
        disagreement: Tensor,
        body_relative_location: Tensor,
        normalized_bbox: Tensor,
    ) -> Tensor:
        rotation = matrix_to_rotation_6d(omni.pose_rotmat[:, :, 1:])
        joints = omni.joints_local[:, :, MANO_POSE_KEYPOINT_INDICES]
        parents = omni.joints_local[:, :, MANO_POSE_PARENT_KEYPOINT_INDICES]
        scale = torch.linalg.vector_norm(omni.joints_local[:, :, 9], dim=-1).clamp_min(1e-4)
        joints = joints / scale[..., None, None]
        joint_velocity = _finite_difference(joints, timestamps)
        palm = matrix_to_rotation_6d(omni.palm_rotmat)[:, :, None].expand(-1, -1, 15, -1)
        scalar = torch.stack(
            [
                omni.confidence,
                omni.valid.float(),
                omni.padding_ratio
                if omni.padding_ratio is not None
                else torch.zeros_like(omni.confidence),
                torch.linalg.vector_norm(omni.shape, dim=-1),
                disagreement[..., 0],
                disagreement[..., 1],
            ],
            dim=-1,
        )[:, :, None].expand(-1, -1, 15, -1)
        shape = omni.shape[:, :, None].expand(-1, -1, 15, -1)
        location = body_relative_location[:, :, None].expand(-1, -1, 15, -1)
        bbox = normalized_bbox[:, :, None].expand(-1, -1, 15, -1)
        tips = omni.joints_local[:, :, FINGERTIP_INDICES]
        tip_distances = (
            torch.stack(
                [
                    torch.linalg.vector_norm(tips[..., 0, :] - tips[..., 1, :], dim=-1),
                    torch.linalg.vector_norm(tips[..., 0, :] - tips[..., 2, :], dim=-1),
                    torch.linalg.vector_norm(tips[..., 0, :] - tips[..., 3, :], dim=-1),
                    torch.linalg.vector_norm(tips[..., 0, :] - tips[..., 4, :], dim=-1),
                    torch.linalg.vector_norm(tips[..., 1, :] - tips[..., 4, :], dim=-1),
                ],
                dim=-1,
            )
            / scale[..., None]
        )
        tip_distances = tip_distances[:, :, None].expand(-1, -1, 15, -1)
        bone = torch.nn.functional.normalize(
            omni.joints_local[:, :, MANO_POSE_KEYPOINT_INDICES] - parents,
            dim=-1,
            eps=1e-8,
        )
        palm_x = omni.palm_rotmat[..., :, 0][:, :, None]
        palm_y = omni.palm_rotmat[..., :, 1][:, :, None]
        flexion_splay = torch.stack(
            [(bone * palm_y).sum(-1), (bone * palm_x).sum(-1)],
            dim=-1,
        )
        return torch.cat(
            [
                rotation,
                joints,
                joint_velocity,
                palm,
                scalar,
                shape,
                location,
                bbox,
                tip_distances,
                flexion_splay,
            ],
            dim=-1,
        )

    def build(
        self,
        body: BodyObservation,
        h4w_left: HandObservation,
        h4w_right: HandObservation,
        omni_left: HandObservation,
        omni_right: HandObservation,
        root_rel: Tensor,
        timestamps: Tensor,
        handedness_class: Tensor | None = None,
    ) -> ExplicitTokenBatch:
        """Build streams from batched observations.

        Observations may be unbatched (`[T,...]`); a batch dimension is added.
        """

        if body.root_rotmat.ndim == 3:
            body = _batch_body(body)
            h4w_left, h4w_right = _batch_hand(h4w_left), _batch_hand(h4w_right)
            omni_left, omni_right = _batch_hand(omni_left), _batch_hand(omni_right)
            root_rel = root_rel.unsqueeze(0)
            timestamps = timestamps.unsqueeze(0)
        body_tokens, upper, torso_scale, semantic_positions = self._body_tokens(body)
        left_disagreement = observer_disagreement(h4w_left, omni_left)
        right_disagreement = observer_disagreement(h4w_right, omni_right)
        disagreement = torch.stack([left_disagreement, right_disagreement], dim=2)
        left_tokens = self._hand_tokens(
            h4w_left,
            omni_left,
            timestamps,
            left_disagreement,
            self._body_relative_hand_location(semantic_positions, torso_scale, "left"),
            self._normalized_bbox(omni_left.bbox_xyxy, body.principal_point),
        )
        right_tokens = self._hand_tokens(
            h4w_right,
            omni_right,
            timestamps,
            right_disagreement,
            self._body_relative_hand_location(semantic_positions, torso_scale, "right"),
            self._normalized_bbox(omni_right.bbox_xyxy, body.principal_point),
        )

        relative_palm = so3_log(omni_right.palm_rotmat.transpose(-1, -2) @ omni_left.palm_rotmat)
        iou = _bbox_iou(omni_left.bbox_xyxy, omni_right.bbox_xyxy)
        root_distance = torch.linalg.vector_norm(root_rel, dim=-1)
        minimum_joint_distance = (
            torch.cdist(omni_left.joints_world_rel, omni_right.joints_world_rel).amin(dim=(-2, -1))
            if omni_left.joints_world_rel is not None and omni_right.joints_world_rel is not None
            else root_distance
        )
        fingertip_distance = torch.linalg.vector_norm(
            omni_left.joints_local[..., FINGERTIP_INDICES, :].mean(-2)
            - omni_right.joints_local[..., FINGERTIP_INDICES, :].mean(-2),
            dim=-1,
        )
        wrist_velocity_left = _finite_difference(omni_left.wrist_world_rel, timestamps)
        wrist_velocity_right = _finite_difference(omni_right.wrist_world_rel, timestamps)
        speed_left = torch.linalg.vector_norm(wrist_velocity_left, dim=-1)
        speed_right = torch.linalg.vector_norm(wrist_velocity_right, dim=-1)
        angular_left = torch.zeros_like(omni_left.wrist_world_rel)
        angular_right = torch.zeros_like(omni_right.wrist_world_rel)
        if timestamps.shape[1] > 1:
            dt = (timestamps[:, 1:] - timestamps[:, :-1]).clamp_min(1e-6)
            angular_left[:, 1:] = (
                so3_log(
                    omni_left.palm_rotmat[:, :-1].transpose(-1, -2) @ omni_left.palm_rotmat[:, 1:]
                )
                / dt[..., None]
            )
            angular_right[:, 1:] = (
                so3_log(
                    omni_right.palm_rotmat[:, :-1].transpose(-1, -2) @ omni_right.palm_rotmat[:, 1:]
                )
                / dt[..., None]
            )
            angular_left[:, 0], angular_right[:, 0] = angular_left[:, 1], angular_right[:, 1]
        angular_speed_left = torch.linalg.vector_norm(angular_left, dim=-1)
        angular_speed_right = torch.linalg.vector_norm(angular_right, dim=-1)
        if handedness_class is None:
            handedness = torch.zeros_like(iou)
        else:
            handedness = handedness_class.to(iou).view(-1, 1).expand_as(iou)
        interaction_prior = torch.exp(-root_distance / 0.12) * (omni_left.valid & omni_right.valid)
        relation = torch.cat(
            [
                root_rel,
                relative_palm,
                iou[..., None],
                torch.stack([root_distance, minimum_joint_distance, fingertip_distance], dim=-1),
                torch.stack([omni_left.confidence, omni_right.confidence], dim=-1),
                torch.stack([omni_left.valid.float(), omni_right.valid.float()], dim=-1),
                torch.stack([speed_left, speed_right], dim=-1),
                torch.stack([angular_speed_left, angular_speed_right], dim=-1),
                handedness[..., None],
                interaction_prior[..., None],
            ],
            dim=-1,
        )
        result = ExplicitTokenBatch(
            body=body_tokens,
            left=left_tokens,
            right=right_tokens,
            relation=relation,
            timestamps=timestamps,
            upper_base_rotmat=upper,
            left_base_rotmat=body_device_pose(h4w_left.pose_rotmat[:, :, 1:], body_tokens),
            right_base_rotmat=body_device_pose(h4w_right.pose_rotmat[:, :, 1:], body_tokens),
            left_valid=omni_left.valid,
            right_valid=omni_right.valid,
            disagreement=disagreement,
            left_observer_feature=omni_left.temporal_token,
            right_observer_feature=omni_right.temporal_token,
            left_h4w_feature=h4w_left.temporal_token,
            right_h4w_feature=h4w_right.temporal_token,
            body_observer_feature=body.body_features,
        )
        result.validate()
        return result


def body_device_pose(value: Tensor, reference: Tensor) -> Tensor:
    return value.to(device=reference.device, dtype=reference.dtype)


def _batch_hand(hand: HandObservation) -> HandObservation:
    values = {}
    for name in hand.__dataclass_fields__:
        value = getattr(hand, name)
        values[name] = None if value is None else value.unsqueeze(0)
    return HandObservation(**values)


def _batch_body(body: BodyObservation) -> BodyObservation:
    values = {}
    for name in body.__dataclass_fields__:
        value = getattr(body, name)
        values[name] = None if value is None else value.unsqueeze(0)
    return BodyObservation(**values)
