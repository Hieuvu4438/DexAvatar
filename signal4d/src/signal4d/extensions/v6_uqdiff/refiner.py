from __future__ import annotations

import copy
from dataclasses import dataclass

import torch

from ...config import MethodConfig
from ...data.cache import ObservationBatch
from ...factors.observation_3d import observation_3d_factor
from ...factors.rotation_observation import rotation_observation_factor
from ...factors.temporal import adaptive_weights, temporal_position_factor, temporal_rotation_factor
from ...geometry.so3 import geodesic_distance, matrix_to_rotation_6d
from ...io.predictions import PredictionArtifact
from ...models.smplx_wrapper import SMPLXOutput, SMPLXWrapper
from ...optimization.smplx_solver import _change_probability, _sigma
from ...optimization.state import SequenceState
from .config import V6Config
from .diffusion_prior import (
    canonical_joint_indices,
    euclidean_dposer_loss,
    geodesic_dposer_loss,
    normalized_dimension_mask,
    uncertainty_change_weights,
)
from .dposer_bridge import DenoisedTarget, DPoserXBridge
from .retraction import stable_exp_map
from .safe_gate import (
    apply_rotation_gate,
    assert_only_open_rotations_changed,
    safe_acceptance_mask,
)
from .seam import wrist_mcp_seam_loss


@dataclass(frozen=True)
class ShapeSelection:
    betas: torch.Tensor
    source: str
    vertex_error_m: float


@dataclass(frozen=True)
class RefinementResult:
    prediction: PredictionArtifact
    diagnostics: dict[str, object]


def _state(
    rotations: torch.Tensor, translation_internal: torch.Tensor, betas: torch.Tensor
) -> SequenceState:
    rotation_6d = matrix_to_rotation_6d(rotations)
    state = SequenceState(
        global_rot6d=rotation_6d[:, 0],
        body_rot6d=rotation_6d[:, 1:22],
        left_hand_rot6d=rotation_6d[:, 25:40],
        right_hand_rot6d=rotation_6d[:, 40:55],
        translation=translation_internal,
        betas=betas,
    )
    state.validate()
    return state


def compose_tangent_update(
    base: torch.Tensor,
    body_delta: torch.Tensor,
    body_indices: tuple[int, ...],
    left_hand_delta: torch.Tensor | None,
    right_hand_delta: torch.Tensor | None,
) -> torch.Tensor:
    """Right-retract open tangents while retaining all closed rotations bitwise."""
    result = base.clone()
    canonical_body = tuple(index + 1 for index in body_indices)
    result[:, canonical_body] = base[:, canonical_body] @ stable_exp_map(body_delta)
    if left_hand_delta is not None:
        result[:, 25:40] = base[:, 25:40] @ stable_exp_map(left_hand_delta)
    if right_hand_delta is not None:
        result[:, 40:55] = base[:, 40:55] @ stable_exp_map(right_hand_delta)
    return result


@torch.inference_mode()
def select_v5_shape(
    model: SMPLXWrapper,
    base: PredictionArtifact,
    metadata: dict[str, object],
    translation_internal: torch.Tensor,
) -> ShapeSelection:
    if base.rotations is None or base.vertices is None:
        raise ValueError("V6 requires V5 rotations and vertices")
    candidates = {
        "smplerx": metadata.get("betas_mean"),
        "legacy_biomech": metadata.get("legacy_betas_mean"),
    }
    results: list[ShapeSelection] = []
    for source, value in candidates.items():
        if value is None:
            continue
        betas = torch.as_tensor(value, dtype=base.vertices.dtype, device=base.vertices.device)[None]
        output = model(_state(base.rotations, translation_internal, betas))
        error = torch.linalg.vector_norm(output.vertices - base.vertices, dim=-1).mean()
        results.append(ShapeSelection(betas, source, float(error)))
    if not results:
        raise ValueError("cache metadata has no candidate V5 shape")
    return min(results, key=lambda item: item.vertex_error_m)


def _enabled_observations(
    batch: ObservationBatch, config: MethodConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_3d = batch.valid_3d.clone()
    if batch.valid_rot is None:
        raise ValueError("V6 requires canonical rotation observations")
    valid_rot = batch.valid_rot.clone()
    if config.observation_sources is not None:
        enabled = torch.zeros(valid_3d.shape[1], dtype=torch.bool, device=valid_3d.device)
        enabled[config.observation_sources] = True
        valid_3d &= enabled[None, :, None]
        valid_rot &= enabled[None, :, None]
    return valid_3d, valid_rot


def _data_terms(
    output: SMPLXOutput,
    rotations: torch.Tensor,
    batch: ObservationBatch,
    valid_3d: torch.Tensor,
    valid_rot: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[object, object]:
    observation = observation_3d_factor(
        output.joints[:, :55], batch.joints_3d, valid_3d, sigma
    )
    rotation = rotation_observation_factor(
        rotations, batch.rotations, valid_rot, sigma  # type: ignore[arg-type]
    )
    return observation, rotation


def _frame_objective(
    observation: object,
    rotation: object,
    config: V6Config,
) -> torch.Tensor:
    return (
        config.refinement.observation_weight * observation.per_frame
        + config.refinement.rotation_observation_weight * rotation.per_frame
    )


def _time_for_step(config: V6Config, step: int) -> float:
    if config.dposer.time_strategy == "fixed" or config.refinement.max_steps == 1:
        return 0.5 * (config.dposer.time_min + config.dposer.time_max)
    fraction = step / (config.refinement.max_steps - 1)
    return config.dposer.time_max + fraction * (
        config.dposer.time_min - config.dposer.time_max
    )


def refine_v5_clip(
    batch: ObservationBatch,
    metadata: dict[str, object],
    base: PredictionArtifact,
    base_method: MethodConfig,
    config: V6Config,
    model: SMPLXWrapper,
    dposer: DPoserXBridge | None,
    fps: float,
) -> RefinementResult:
    batch.validate()
    base.validate()
    if base.rotations is None or base.vertices is None:
        raise ValueError("V6 requires complete V5 SMPL-X predictions")
    if not torch.equal(batch.frame_ids, base.frame_ids):
        raise ValueError("V5 and observation frame IDs differ")
    if config.dposer.enabled != (dposer is not None):
        raise ValueError("DPoser-X bridge presence does not match V6 config")

    camera_x_180 = base.translation.new_tensor([1.0, -1.0, -1.0])
    translation_internal = base.translation * camera_x_180
    shape = select_v5_shape(model, base, metadata, translation_internal)
    base_rotations = base.rotations.detach()
    frame_count = base.frame_ids.numel()
    open_body = config.open_body_indices
    open_canonical = canonical_joint_indices(
        open_body,
        config.refinement.optimize_left_hand,
        config.refinement.optimize_right_hand,
    )

    body_delta = torch.zeros(
        (frame_count, len(open_body), 3), device=base_rotations.device, requires_grad=True
    )
    left_delta = (
        torch.zeros((frame_count, 15, 3), device=base_rotations.device, requires_grad=True)
        if config.refinement.optimize_left_hand
        else None
    )
    right_delta = (
        torch.zeros((frame_count, 15, 3), device=base_rotations.device, requires_grad=True)
        if config.refinement.optimize_right_hand
        else None
    )
    parameters = [body_delta]
    parameters.extend(value for value in (left_delta, right_delta) if value is not None)
    optimizer = torch.optim.Adam(parameters, lr=config.refinement.learning_rate)

    sigma = _sigma(batch, base_method)
    change = _change_probability(batch, fps, config.refinement.change_aware)
    uncertainty = sigma.mean((1, 3))
    temporal_weight = adaptive_weights(
        uncertainty,
        change,
        gamma=base_method.change_point.gamma,
    )
    prior_joint_weight = uncertainty_change_weights(
        uncertainty,
        change,
        uncertainty_aware=config.refinement.uncertainty_aware,
        change_aware=config.refinement.change_aware,
        change_gamma=base_method.change_point.gamma,
    )
    valid_3d, valid_rot = _enabled_observations(batch, base_method)
    with torch.inference_mode():
        base_output = model(_state(base_rotations, translation_internal, shape.betas))
        base_observation, base_rotation = _data_terms(
            base_output, base_rotations, batch, valid_3d, valid_rot, sigma
        )
        base_frame_objective = _frame_objective(
            base_observation, base_rotation, config
        )

    generator = torch.Generator(device=base_rotations.device)
    generator.manual_seed(config.dposer.noise_seed)
    target: DenoisedTarget | None = None
    dimension_mask = normalized_dimension_mask(
        open_body,
        config.refinement.optimize_left_hand,
        config.refinement.optimize_right_hand,
        device=base_rotations.device,
    )
    best_loss = float("inf")
    best_parameters = [value.detach().clone() for value in parameters]
    history: list[dict[str, float]] = []

    for step in range(config.refinement.max_steps):
        optimizer.zero_grad(set_to_none=True)
        rotations = compose_tangent_update(
            base_rotations, body_delta, open_body, left_delta, right_delta
        )
        output = model(_state(rotations, translation_internal, shape.betas))
        observation, rotation = _data_terms(
            output, rotations, batch, valid_3d, valid_rot, sigma
        )
        temporal = temporal_position_factor(
            output.joints[:, :55], fps, temporal_weight, delta=5.0
        )
        temporal_rotation = temporal_rotation_factor(
            rotations, fps, temporal_weight, delta=2.0
        )
        anchor_distance = geodesic_distance(
            base_rotations[:, open_canonical], rotations[:, open_canonical]
        )
        anchor = anchor_distance.square().mean()

        diffusion = rotations.sum() * 0
        if dposer is not None:
            if target is None or step % config.dposer.refresh_denoised_target_every == 0:
                time_value = _time_for_step(config, step)
                time = rotations.new_full((frame_count,), time_value)
                target = dposer.denoise_target(rotations.detach(), time, generator)
            if config.dposer.mode == "euclidean":
                normalized = dposer.normalizer.normalize_parts(
                    dposer.rotations_to_parts(rotations)
                )
                frame_weight = prior_joint_weight[:, open_canonical].mean(1)
                diffusion = euclidean_dposer_loss(
                    normalized,
                    target.normalized,
                    target.snr,
                    dimension_mask,
                    frame_weight,
                )
            else:
                diffusion = geodesic_dposer_loss(
                    rotations,
                    target.rotations,
                    target.snr,
                    open_canonical,
                    prior_joint_weight,
                )
        seam = wrist_mcp_seam_loss(
            rotations,
            base_rotations,
            frame_weight=(1 - change).clamp_min(0.1),
        )
        loss = config.refinement.observation_weight * observation.loss
        loss += config.refinement.rotation_observation_weight * rotation.loss
        loss += config.refinement.v5_anchor_weight * anchor
        loss += config.refinement.temporal_weight * temporal.loss
        loss += config.refinement.temporal_rotation_weight * temporal_rotation.loss
        loss += config.refinement.diffusion_weight * diffusion
        loss += config.refinement.seam_weight * seam
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite V6 loss at step {step}")
        value = float(loss.detach())
        history.append(
            {
                "step": float(step),
                "total": value,
                "observation": float(observation.loss.detach()),
                "rotation": float(rotation.loss.detach()),
                "anchor": float(anchor.detach()),
                "temporal": float(temporal.loss.detach()),
                "temporal_rotation": float(temporal_rotation.loss.detach()),
                "diffusion": float(diffusion.detach()),
                "seam": float(seam.detach()),
            }
        )
        if value < best_loss:
            best_loss = value
            best_parameters = [item.detach().clone() for item in parameters]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.refinement.grad_clip_norm)
        optimizer.step()

    body_best = best_parameters[0]
    offset = 1
    left_best = best_parameters[offset] if left_delta is not None else None
    offset += int(left_delta is not None)
    right_best = best_parameters[offset] if right_delta is not None else None
    raw_rotations = compose_tangent_update(
        base_rotations, body_best, open_body, left_best, right_best
    )
    assert_only_open_rotations_changed(base_rotations, raw_rotations, open_canonical)
    with torch.inference_mode():
        raw_output = model(_state(raw_rotations, translation_internal, shape.betas))
        candidate_observation, candidate_rotation = _data_terms(
            raw_output, raw_rotations, batch, valid_3d, valid_rot, sigma
        )
        candidate_frame_objective = _frame_objective(
            candidate_observation, candidate_rotation, config
        )

    delta = geodesic_distance(
        base_rotations[:, open_canonical], raw_rotations[:, open_canonical]
    )
    uncertainty_open = uncertainty[:, open_canonical].mean(1)
    uncertainty_ratio = uncertainty_open / uncertainty_open.median().clamp_min(1e-6)
    if config.safe_gate.enabled:
        accept = safe_acceptance_mask(
            base_frame_objective,
            candidate_frame_objective,
            delta.max(1).values,
            uncertainty_ratio,
            require_objective_improvement=config.safe_gate.require_objective_improvement,
            max_rotation_delta_rad=config.safe_gate.max_rotation_delta_rad,
            max_uncertainty_ratio=config.safe_gate.max_uncertainty_ratio,
            transition_radius=config.safe_gate.transition_radius,
        )
    else:
        accept = torch.ones(frame_count, dtype=torch.bool, device=base_rotations.device)
    final_rotations = apply_rotation_gate(base_rotations, raw_rotations, accept)
    assert_only_open_rotations_changed(base_rotations, final_rotations, open_canonical)
    with torch.inference_mode():
        final_output = model(_state(final_rotations, translation_internal, shape.betas))
    vertices = torch.where(accept[:, None, None], final_output.vertices, base.vertices)
    joints = torch.where(accept[:, None, None], final_output.joints[:, :55], base.joints_3d)

    prediction = PredictionArtifact(
        frame_ids=base.frame_ids,
        joints_3d=joints,
        rotations=final_rotations,
        translation=base.translation,
        vertices=vertices,
        risk_score=base.risk_score,
        abstain=base.abstain,
        uncertainty=base.uncertainty,
        contact_probability=base.contact_probability,
        contacts=base.contacts,
    )
    prediction.validate()
    diagnostics: dict[str, object] = {
        "best_loss": best_loss,
        "shape_source": shape.source,
        "v5_shape_reproduction_vertex_error_m": shape.vertex_error_m,
        "accepted_frames": int(accept.sum()),
        "total_frames": frame_count,
        "acceptance_mask": accept.detach().cpu().tolist(),
        "max_rotation_delta_rad": float(delta.max()),
        "mean_rotation_delta_rad": float(delta.mean()),
        "base_frame_objective_mean": float(base_frame_objective.mean()),
        "candidate_frame_objective_mean": float(candidate_frame_objective.mean()),
        "history": copy.deepcopy(history),
        "open_canonical_joint_indices": list(open_canonical),
        "gt_used": False,
    }
    return RefinementResult(prediction, diagnostics)

