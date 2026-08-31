from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from signpk.data.cache_schema import BodyObservation, CouplerPrediction, HandObservation
from signpk.geometry.coordinates import CameraParameters, CoordinateAdapter
from signpk.geometry.palm_frame import make_palm_frame
from signpk.geometry.robustifiers import charbonnier, geman_mcclure, masked_mean
from signpk.geometry.rotations import so3_distance, so3_log
from signpk.losses.interaction import hand_penetration_loss
from signpk.models.gates import uncertainty_weight

from .smplx_layer import SMPLXOutput
from .state import SequenceState


@dataclass
class FactorInputs:
    h4w_body: BodyObservation
    h4w_left: HandObservation
    h4w_right: HandObservation
    omni_left: HandObservation | None
    omni_right: HandObservation | None
    pkc: CouplerPrediction
    left_vertex_indices: Tensor
    right_vertex_indices: Tensor
    root_rel: Tensor
    timestamps: Tensor
    dex_priors: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None
    faces: Tensor | None = None
    one_hand_dominant: str | None = None


def validate_factor_inputs(inputs: FactorInputs, vertex_count: int = 10475) -> None:
    frames = inputs.h4w_body.root_rotmat.shape[0]
    if inputs.root_rel.shape != (frames, 3) or inputs.timestamps.shape != (frames,):
        raise ValueError("factor root_rel/timestamps do not match the clip length")
    for name, indices in (
        ("left", inputs.left_vertex_indices),
        ("right", inputs.right_vertex_indices),
    ):
        if indices.ndim != 1 or indices.numel() != 778:
            raise ValueError(f"{name} MANO-SMPL-X correspondence must contain 778 vertices")
        if indices.unique().numel() != indices.numel():
            raise ValueError(f"{name} MANO-SMPL-X correspondence contains duplicates")
        if int(indices.min()) < 0 or int(indices.max()) >= vertex_count:
            raise ValueError(f"{name} MANO-SMPL-X correspondence is out of range")
    if torch.isin(inputs.left_vertex_indices, inputs.right_vertex_indices).any():
        raise ValueError("left/right MANO-SMPL-X correspondences overlap")
    if inputs.faces is not None:
        if inputs.faces.ndim != 2 or inputs.faces.shape[1] != 3:
            raise ValueError("SMPL-X faces must have shape [F,3]")
        if int(inputs.faces.min()) < 0 or int(inputs.faces.max()) >= vertex_count:
            raise ValueError("SMPL-X faces are out of range")
    if inputs.one_hand_dominant not in {None, "left", "right"}:
        raise ValueError("one_hand_dominant must be left, right, or None")


def _routing_weight(inputs: FactorInputs, side: str, weak_weight: float = 0.25) -> float:
    if inputs.one_hand_dominant is None or inputs.one_hand_dominant == side:
        return 1.0
    return weak_weight


def _rotation_factor(prediction: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
    values = so3_distance(prediction, target, squared=True)
    if weight is not None:
        values = values * weight
    return values.mean()


def _hand_vertex_factor(
    vertices: Tensor,
    indices: Tensor,
    observation: HandObservation,
    scale: float,
) -> Tensor:
    predicted = vertices[:, indices]
    predicted = predicted - predicted.mean(-2, keepdim=True)
    target = observation.vertices_local
    target = target - target.mean(-2, keepdim=True)
    if predicted.shape[-2] != target.shape[-2]:
        # The official MANO-SMPL-X correspondence arrays are expected to have
        # 778 entries. Refuse to invent a nearest-neighbour correspondence.
        raise ValueError(f"hand correspondence mismatch: {predicted.shape} vs {target.shape}")
    distance = torch.linalg.vector_norm(predicted - target, dim=-1)
    robust = geman_mcclure(distance, scale)
    return masked_mean(robust, observation.valid)


def _hand_reprojection(
    joints: Tensor,
    observation: HandObservation,
    camera: CameraParameters,
    scale_px: float,
) -> Tensor:
    projected = CoordinateAdapter.project(joints, camera)
    distance = torch.linalg.vector_norm(projected - observation.keypoints2d, dim=-1)
    weighted = geman_mcclure(distance, scale_px) * observation.keypoint_confidence
    return masked_mean(weighted, observation.valid)


def compute_factors(
    output: SMPLXOutput,
    state: SequenceState,
    inputs: FactorInputs,
    scales: dict[str, float] | None = None,
) -> dict[str, Tensor]:
    """Compute normalized test-time factors; absent observations are omitted."""

    scales = scales or {}
    rotations = state.rotations()
    factors: dict[str, Tensor] = {}
    h4w_upper = torch.cat(
        [inputs.h4w_body.root_rotmat[:, None], inputs.h4w_body.body_rotmat], dim=1
    )[:, (0, 3, 6, 9, 12, 15, 13, 14, 16, 17, 18, 19, 20, 21)]
    upper_routing = h4w_upper.new_ones(14)
    if inputs.one_hand_dominant == "right":
        upper_routing[[6, 8, 10, 12]] = 0.25
    elif inputs.one_hand_dominant == "left":
        upper_routing[[7, 9, 11, 13]] = 0.25
    factors["h4w"] = _rotation_factor(
        rotations.upper,
        h4w_upper,
        uncertainty_weight(inputs.pkc.log_variance["upper"]) * upper_routing,
    )
    factors["pkc"] = (
        _rotation_factor(
            rotations.upper,
            inputs.pkc.pose_rotmat[:, :14],
            uncertainty_weight(inputs.pkc.log_variance["upper"]) * upper_routing,
        )
        + _routing_weight(inputs, "left")
        * _rotation_factor(
            rotations.left_hand,
            inputs.pkc.pose_rotmat[:, 14:29],
            uncertainty_weight(inputs.pkc.log_variance["left"]),
        )
        + _routing_weight(inputs, "right")
        * _rotation_factor(
            rotations.right_hand,
            inputs.pkc.pose_rotmat[:, 29:44],
            uncertainty_weight(inputs.pkc.log_variance["right"]),
        )
    ) / 3
    median_shape = inputs.h4w_body.shape.median(0).values
    factors["shape"] = (
        charbonnier(state.beta - median_shape).mean() + 1e-3 * state.beta.square().mean()
    )
    camera = CameraParameters(
        focal_length=inputs.h4w_body.focal_length,
        principal_point=inputs.h4w_body.principal_point,
        translation=None,
    )
    reprojection_terms = []
    if inputs.h4w_left.keypoint_confidence.any():
        reprojection_terms.append(
            _routing_weight(inputs, "left")
            * _hand_reprojection(
                output.left_hand_joints, inputs.h4w_left, camera, scales.get("hand_2d_px", 6.0)
            )
        )
    if inputs.h4w_right.keypoint_confidence.any():
        reprojection_terms.append(
            _routing_weight(inputs, "right")
            * _hand_reprojection(
                output.right_hand_joints, inputs.h4w_right, camera, scales.get("hand_2d_px", 6.0)
            )
        )
    if reprojection_terms:
        factors["reprojection_2d"] = torch.stack(reprojection_terms).mean()

    omni_terms, palm_terms = [], []
    predicted_palms: dict[str, Tensor] = {}
    for side, observation, vertex_indices, hand_joints in (
        ("left", inputs.omni_left, inputs.left_vertex_indices, output.left_hand_joints),
        ("right", inputs.omni_right, inputs.right_vertex_indices, output.right_hand_joints),
    ):
        if observation is None:
            continue
        omni_terms.append(
            _routing_weight(inputs, side)
            * _hand_vertex_factor(
                output.vertices, vertex_indices, observation, scales.get("hand_vertex_m", 0.008)
            )
        )
        palm, _, palm_valid = make_palm_frame(hand_joints, side)
        predicted_palms[side] = palm
        palm_weight = uncertainty_weight(
            inputs.pkc.log_variance["palm"][:, 0 if side == "left" else 1]
        )
        palm_terms.append(
            _routing_weight(inputs, side)
            * masked_mean(
                so3_distance(palm, observation.palm_rotmat, squared=True) * palm_weight,
                observation.valid & palm_valid,
            )
        )
    if omni_terms:
        factors["omni_hand"] = torch.stack(omni_terms).mean()
    if palm_terms:
        factors["palm"] = torch.stack(palm_terms).mean()

    full_rotations = torch.cat([rotations.upper, rotations.left_hand, rotations.right_hand], dim=1)
    if state.num_frames > 1:
        dt = (inputs.timestamps[1:] - inputs.timestamps[:-1]).clamp_min(1e-6)
        observed_velocity = (
            so3_log(full_rotations[:-1].transpose(-1, -2) @ full_rotations[1:]) / dt[:, None, None]
        )
        target_velocity = inputs.pkc.angular_velocity[:-1]
        phase = inputs.pkc.phase_gate[:-1].view(-1, 1)
        alpha = 0.25 * (1 - phase) + phase
        factors["motion"] = (
            charbonnier(observed_velocity - target_velocity).mean(-1) * alpha
        ).mean()

    if inputs.omni_left is not None and inputs.omni_right is not None:
        wrist_delta = output.left_hand_joints[:, 0] - output.right_hand_joints[:, 0]
        target_delta = inputs.root_rel
        valid = inputs.omni_left.valid & inputs.omni_right.valid
        gate = inputs.pkc.interaction_gate.squeeze(-1)
        distance = torch.linalg.vector_norm(wrist_delta - target_delta, dim=-1)
        factors["relation"] = masked_mean(
            geman_mcclure(distance, scales.get("wrist_relation_m", 0.015)) * gate,
            valid,
        )
        if "left" in predicted_palms and "right" in predicted_palms:
            predicted_relative_palm = (
                predicted_palms["right"].transpose(-1, -2) @ predicted_palms["left"]
            )
            target_relative_palm = (
                inputs.omni_right.palm_rotmat.transpose(-1, -2) @ inputs.omni_left.palm_rotmat
            )
            factors["relation"] = factors["relation"] + masked_mean(
                so3_distance(
                    predicted_relative_palm,
                    target_relative_palm,
                    squared=True,
                )
                * gate,
                valid,
            )
        if inputs.faces is not None:
            factors["penetration"] = hand_penetration_loss(
                output.vertices,
                inputs.faces,
                inputs.left_vertex_indices,
                inputs.right_vertex_indices,
                gate,
                valid,
            )
    residuals = state.residual_radians()
    factors["residual"] = torch.stack(
        [value.square().mean() for value in residuals.values()]
    ).mean()
    if inputs.dex_priors is not None:
        factors["sign_prior"] = inputs.dex_priors(
            rotations.body, rotations.left_hand, rotations.right_hand
        )
    return factors


def normalized_weighted_sum(factors: dict[str, Tensor], weights: dict[str, float]) -> Tensor:
    active = [
        (name, value, float(weights.get(name, 0.0)))
        for name, value in factors.items()
        if weights.get(name, 0.0) != 0
    ]
    if not active:
        raise ValueError("BA stage has no active factors")
    return sum(weight * value for _, value, weight in active)
