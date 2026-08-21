from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from ..config import MethodConfig
from ..data.cache import ObservationBatch
from ..factors.collision import collision_factor
from ..factors.contact import contact_factor
from ..factors.observation_2d import observation_2d_factor
from ..factors.observation_3d import observation_3d_factor
from ..factors.prior import pose_prior_factor
from ..factors.rotation_observation import rotation_observation_factor
from ..factors.temporal import (
    adaptive_weights,
    temporal_position_factor,
    temporal_rotation_factor,
)
from ..geometry.so3 import matrix_to_rotation_6d
from ..io.predictions import PredictionArtifact
from ..models.change_point import rule_based_change_probability
from ..models.contact_proposer import ContactEdgeSpec, decode_hysteresis, propose_contacts
from ..models.smplx_wrapper import SMPLXWrapper
from ..models.uncertainty import UncertaintyCalibrator
from ..utils.hashing import sha256_file
from ..utils.logging import JsonlLogger
from .consensus import weighted_karcher_mean
from .state import SequenceState
from .window import Window, hann_weights, plan_windows


@dataclass
class PoseWindowFit:
    window: Window
    rotations: torch.Tensor
    translation: torch.Tensor
    betas: torch.Tensor
    uncertainty: torch.Tensor
    contact_logits: torch.Tensor | None
    diagnostics: dict[str, object]


def _combined_rotations(state: SequenceState) -> torch.Tensor:
    values = state.rotations()
    t = state.translation.shape[0]
    identity = torch.eye(3, dtype=state.translation.dtype, device=state.translation.device)
    face = identity.expand(t, 3, 3, 3)
    return torch.cat(
        (
            values["global_orient"][:, None],
            values["body_pose"],
            face,
            values["left_hand_pose"],
            values["right_hand_pose"],
        ),
        dim=1,
    )


def _make_state(
    rotations: torch.Tensor,
    translation: torch.Tensor,
    betas: torch.Tensor,
    trainable: bool,
) -> SequenceState:
    def value(tensor: torch.Tensor) -> torch.Tensor:
        result = tensor.detach().clone()
        result.requires_grad_(trainable)
        return result

    rotation_6d = matrix_to_rotation_6d(rotations)
    state = SequenceState(
        global_rot6d=value(rotation_6d[:, 0]),
        body_rot6d=value(rotation_6d[:, 1:22]),
        left_hand_rot6d=value(rotation_6d[:, 25:40]),
        right_hand_rot6d=value(rotation_6d[:, 40:55]),
        translation=value(translation),
        betas=value(betas),
    )
    state.validate()
    return state


def _m0_pose(batch: ObservationBatch) -> torch.Tensor:
    if batch.rotations is None or batch.valid_rot is None:
        raise ValueError("SMPL-X solver requires canonical rotation observations")
    pose = batch.rotations[:, 0].clone()
    wilor_valid = batch.valid_rot[:, 1]
    return torch.where(wilor_valid[..., None, None], batch.rotations[:, 1], pose)


def _coherent_legacy_selection(batch: ObservationBatch, sigma: torch.Tensor) -> torch.Tensor:
    if batch.valid_rot is None or batch.joints_3d.shape[1] < 3:
        return torch.zeros(batch.frame_ids.numel(), dtype=torch.bool, device=sigma.device)
    source0 = sigma.mean(-1)[:, 0]
    source1 = sigma.mean(-1)[:, 1]
    source2 = sigma.mean(-1)[:, 2]
    wilor_valid = batch.valid_rot[:, 1]
    hybrid = torch.cat(
        (
            source0[:, :25],
            torch.where(wilor_valid[:, 25:40], source1[:, 25:40], source0[:, 25:40]),
            torch.where(wilor_valid[:, 40:55], source1[:, 40:55], source0[:, 40:55]),
        ),
        dim=-1,
    )
    hybrid_score = torch.stack(
        (
            hybrid[:, :25].mean(-1),
            hybrid[:, 25:40].mean(-1),
            hybrid[:, 40:55].mean(-1),
        ),
        dim=-1,
    ).mean(-1)
    legacy_score = torch.stack(
        (
            source2[:, :25].mean(-1),
            source2[:, 25:40].mean(-1),
            source2[:, 40:55].mean(-1),
        ),
        dim=-1,
    ).mean(-1)
    legacy_complete = batch.valid_rot[:, 2].all(-1)
    return legacy_complete & (legacy_score < hybrid_score)


def _initial_pose(
    batch: ObservationBatch, sigma: torch.Tensor, initializer_mode: str
) -> torch.Tensor:
    if batch.rotations is None or batch.valid_rot is None:
        raise ValueError("SMPL-X solver requires canonical rotation observations")
    if initializer_mode == "region_uncertainty":
        pose = batch.rotations[:, 0].clone()
        for start, end in ((0, 25), (25, 40), (40, 55)):
            source = _region_source_selection(batch, sigma, start, end)
            gather_index = source[:, None, None, None, None].expand(-1, 1, end - start, 3, 3)
            pose[:, start:end] = batch.rotations[:, :, start:end].gather(1, gather_index).squeeze(1)
        return pose
    pose = _m0_pose(batch)
    if initializer_mode in {"coherent_uncertainty", "legacy_full"}:
        if batch.joints_3d.shape[1] < 3:
            return pose
        legacy = (
            batch.valid_rot[:, 2].all(-1)
            if initializer_mode == "legacy_full"
            else _coherent_legacy_selection(batch, sigma)
        )
        return torch.where(legacy[:, None, None, None], batch.rotations[:, 2], pose)
    return pose


def _region_source_selection(
    batch: ObservationBatch, sigma: torch.Tensor, start: int, end: int
) -> torch.Tensor:
    if batch.valid_rot is None:
        raise ValueError("rotation validity is required")
    valid = batch.valid_rot[:, :, start:end]
    complete = valid.all(-1)
    score = sigma.mean(-1)[:, :, start:end].mean(-1)
    score = score.masked_fill(~complete, float("inf"))
    if torch.isinf(score).all(1).any():
        raise ValueError("no complete rotation hypothesis for a canonical region")
    return score.argmin(1)


def _change_probability(batch: ObservationBatch, fps: float, enabled: bool) -> torch.Tensor:
    if not enabled:
        return batch.joints_3d.new_zeros(batch.frame_ids.numel())
    joints = batch.joints_3d[:, 0]
    velocity = torch.zeros_like(joints)
    velocity[1:] = (joints[1:] - joints[:-1]) * fps
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    acceleration = torch.zeros_like(speed)
    acceleration[1:] = (speed[1:] - speed[:-1]).abs() * fps
    hand_speed = speed[:, 20:]
    features = torch.stack(
        (
            speed.mean(-1),
            speed.max(-1).values,
            acceleration.mean(-1),
            acceleration.max(-1).values,
            hand_speed.mean(-1),
            hand_speed.max(-1).values,
        ),
        dim=-1,
    )
    return rule_based_change_probability(features)


def _sigma(batch: ObservationBatch, config: MethodConfig) -> torch.Tensor:
    base = batch.joints_3d.new_full(batch.joints_3d.shape, config.uncertainty.sigma_min)
    if config.uncertainty.mode == "calibrated":
        artifact = Path(str(config.uncertainty.artifact))
        metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
        weights_path = artifact / "weights.safetensors"
        if sha256_file(weights_path) != metrics["weights_sha256"]:
            raise ValueError(f"uncertainty artifact hash mismatch: {weights_path}")
        if int(metrics["feature_dim"]) != batch.features.shape[-1]:
            raise ValueError("uncertainty artifact feature dimension does not match cache")
        calibrator = UncertaintyCalibrator(
            int(metrics["feature_dim"]),
            hidden_dim=int(metrics["hidden_dim"]),
            sigma_min=float(metrics["sigma_min"]),
            sigma_max=float(metrics["sigma_max"]),
        ).to(batch.features.device)
        calibrator.load_state_dict(load_file(weights_path, device=str(batch.features.device)))
        calibrator.eval()
        with torch.no_grad():
            base = calibrator(batch.features, batch.valid_3d)["sigma_xyz"]
        calibration = json.loads((artifact / "group_scales.json").read_text(encoding="utf-8"))
        scales = batch.features.new_ones(batch.joints_3d.shape[1:3])
        for source in range(scales.shape[0]):
            for joint in range(scales.shape[1]):
                region = "body" if joint < 25 else "left_hand" if joint < 40 else "right_hand"
                scales[source, joint] = float(
                    calibration["scales"].get(f"source_{source}:{region}", 1.0)
                )
        base = (base * scales[None, ..., None]).clamp(
            config.uncertainty.sigma_min, config.uncertainty.sigma_max
        )
    elif config.uncertainty.mode != "constant":
        raw = torch.sigmoid(batch.features[..., :5].mean(-1))
        base = (
            config.uncertainty.sigma_min
            + (config.uncertainty.sigma_max - config.uncertainty.sigma_min) * raw
        )[..., None].expand_as(batch.joints_3d)
    return torch.where(
        batch.valid_3d[..., None],
        base,
        torch.full_like(base, config.uncertainty.sigma_max),
    )


def _fit_window(
    batch: ObservationBatch,
    model: SMPLXWrapper,
    window: Window,
    initial_rotation: torch.Tensor,
    initial_translation: torch.Tensor,
    betas: torch.Tensor,
    sigma: torch.Tensor,
    change_probability: torch.Tensor,
    config: MethodConfig,
    fps: float,
    logger: JsonlLogger | None,
) -> PoseWindowFit:
    frame_slice = slice(window.start, window.end)
    state = _make_state(
        initial_rotation[frame_slice], initial_translation[frame_slice], betas, trainable=True
    )
    parameters = []
    if config.solver.optimize_global:
        parameters.append(state.global_rot6d)
    if config.solver.optimize_body:
        parameters.append(state.body_rot6d)
        if config.solver.body_joint_indices is not None:
            gradient_mask = torch.zeros_like(state.body_rot6d)
            gradient_mask[:, config.solver.body_joint_indices] = 1
            state.body_rot6d.register_hook(lambda gradient: gradient * gradient_mask)
    if config.solver.optimize_hands:
        if config.solver.optimize_left_hand:
            parameters.append(state.left_hand_rot6d)
        if config.solver.optimize_right_hand:
            parameters.append(state.right_hand_rot6d)
    if config.solver.optimize_translation:
        parameters.append(state.translation)
    if not parameters:
        raise ValueError("at least one SMPL-X parameter group must be trainable")
    best_loss = float("inf")
    best_state = state.detached_clone()
    best_contact_logits = None
    stale = 0
    local_sigma = sigma[frame_slice]
    local_valid_3d = batch.valid_3d[frame_slice].clone()
    local_valid_rot = batch.valid_rot[frame_slice].clone()  # type: ignore[index]
    if config.observation_sources is not None:
        enabled = torch.zeros(
            local_valid_3d.shape[1], dtype=torch.bool, device=local_valid_3d.device
        )
        enabled[config.observation_sources] = True
        local_valid_3d &= enabled[None, :, None]
        local_valid_rot &= enabled[None, :, None]
    local_uncertainty = local_sigma.mean((1, 3))
    weights = adaptive_weights(
        local_uncertainty,
        change_probability[frame_slice],
        gamma=config.change_point.gamma,
    )
    max_steps = config.solver.max_steps
    contact_candidates = None
    contact_logits = None
    collision_pairs: tuple[tuple[int, int], ...] = ()
    if config.contact.enabled:
        with torch.no_grad():
            initial_joints = model(state).joints[:, :55]
            contact_candidates = propose_contacts(
                initial_joints,
                _default_contact_edges(),
                uncertainty=local_uncertainty,
                proposal_radius_m=config.contact.proposal_radius_m,
            )
            initial_logits = torch.logit(contact_candidates.probability.clamp(1e-5, 1 - 1e-5))
        contact_logits = torch.nn.Parameter(initial_logits)
        parameters.append(contact_logits)
        best_contact_logits = contact_logits.detach().clone()
        contact_pairs = {(edge.joint_a, edge.joint_b) for edge in contact_candidates.edges}
        candidate_pairs = ((27, 45), (30, 48), (33, 51), (36, 54), (39, 42))
        collision_pairs = tuple(pair for pair in candidate_pairs if pair not in contact_pairs)
    optimizer = torch.optim.Adam(parameters, lr=config.solver.learning_rate)
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(state)
        predicted_joints = output.joints[:, :55]
        observation = observation_3d_factor(
            predicted_joints,
            batch.joints_3d[frame_slice],
            local_valid_3d,
            local_sigma,
        )
        rotation_result = rotation_observation_factor(
            _combined_rotations(state),
            batch.rotations[frame_slice],  # type: ignore[index]
            local_valid_rot,
            local_sigma,
        )
        temporal = temporal_position_factor(predicted_joints, fps, weights, delta=5.0)
        temporal_rotation = temporal_rotation_factor(
            _combined_rotations(state), fps, weights, delta=2.0
        )
        pose_prior = pose_prior_factor(_combined_rotations(state), initial_rotation[frame_slice])
        observation_2d = None
        if (
            batch.keypoints_2d is not None
            and batch.valid_2d is not None
            and batch.camera_K is not None
            and batch.image_size is not None
        ):
            camera_k = batch.camera_K
            if camera_k.ndim == 3:
                camera_k = camera_k[frame_slice]
            image_size = batch.image_size
            if image_size.ndim == 2:
                image_size = image_size[frame_slice]
            observation_2d = observation_2d_factor(
                predicted_joints * predicted_joints.new_tensor([1.0, -1.0, -1.0]),
                batch.keypoints_2d[frame_slice],
                batch.valid_2d[frame_slice],
                camera_k,
                image_size,
            )
        contact_result = None
        collision_result = None
        if contact_candidates is not None and contact_logits is not None:
            contact_result = contact_factor(
                predicted_joints,
                contact_logits,
                contact_candidates,
                change_probability[frame_slice],
            )
            collision_result = collision_factor(predicted_joints, collision_pairs)
        loss = config.factors.get("observation", 1.0) * observation.loss
        loss += 0.1 * rotation_result.loss
        loss += config.factors.get("temporal", 0.0) * temporal.loss
        loss += config.factors.get("temporal_rotation", 0.0) * temporal_rotation.loss
        loss += config.factors.get("prior", 0.0) * pose_prior.loss
        if observation_2d is not None:
            loss += config.factors.get("observation_2d", 0.0) * observation_2d.loss
        if contact_result is not None and collision_result is not None:
            loss += config.factors.get("contact", 0.0) * contact_result.loss
            loss += config.factors.get("collision", 0.0) * collision_result.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite SMPL-X fit in window {window}")
        value = float(loss.detach())
        if value < best_loss * (1 - config.solver.relative_tolerance):
            best_loss = value
            best_state = state.detached_clone()
            best_contact_logits = (
                contact_logits.detach().clone() if contact_logits is not None else None
            )
            stale = 0
        else:
            stale += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.solver.grad_clip_norm)
        optimizer.step()
        if logger is not None and (step == 0 or step % 20 == 0):
            logger.write(
                {
                    "window": [window.start, window.end],
                    "stage": config.method_name,
                    "step": step,
                    "total_loss": value,
                    "factor_loss": {
                        "observation_3d": float(observation.loss.detach()),
                        "rotation_observation": float(rotation_result.loss.detach()),
                        "temporal": float(temporal.loss.detach()),
                        "temporal_rotation": float(temporal_rotation.loss.detach()),
                        "pose_prior": float(pose_prior.loss.detach()),
                        "observation_2d": (
                            float(observation_2d.loss.detach()) if observation_2d else 0.0
                        ),
                        "contact": (float(contact_result.loss.detach()) if contact_result else 0.0),
                        "collision": (
                            float(collision_result.loss.detach()) if collision_result else 0.0
                        ),
                    },
                }
            )
        if stale >= config.solver.patience:
            break
    return PoseWindowFit(
        window=window,
        rotations=_combined_rotations(best_state),
        translation=best_state.translation,
        betas=best_state.betas,
        uncertainty=local_uncertainty,
        contact_logits=best_contact_logits,
        diagnostics={"best_loss": best_loss, "steps": step + 1},
    )


def _merge_pose(
    results: list[PoseWindowFit], total_frames: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = results[0].rotations.device
    dtype = results[0].rotations.dtype
    joint_count = results[0].rotations.shape[1]
    rotations = (
        torch.eye(3, device=device, dtype=dtype).expand(total_frames, joint_count, 3, 3).clone()
    )
    translation = torch.zeros((total_frames, 3), device=device, dtype=dtype)
    uncertainty = torch.zeros((total_frames, joint_count), device=device, dtype=dtype)
    for frame in range(total_frames):
        local_rotations = []
        local_translation = []
        local_uncertainty = []
        local_weights = []
        for result in results:
            if result.window.start <= frame < result.window.end:
                index = frame - result.window.start
                taper = hann_weights(result.window.length, device, dtype)[index]
                weight = taper / result.uncertainty[index].clamp_min(1e-6)
                local_rotations.append(result.rotations[index])
                local_translation.append(result.translation[index])
                local_uncertainty.append(result.uncertainty[index])
                local_weights.append(weight)
        stacked_weights = torch.stack(local_weights)
        rotations[frame] = weighted_karcher_mean(torch.stack(local_rotations), stacked_weights)
        scalar_weight = stacked_weights.mean(-1)
        translation[frame] = (torch.stack(local_translation) * scalar_weight[:, None]).sum(
            0
        ) / scalar_weight.sum()
        uncertainty[frame] = (torch.stack(local_uncertainty) * stacked_weights).sum(
            0
        ) / stacked_weights.sum(0)
    # Shape is frozen during every window fit, so all windows carry the same value.
    # Reuse it directly and avoid CUDA median's non-deterministic indexed kernel.
    betas = results[0].betas
    return rotations, translation, betas


def _merge_contact_logits(results: list[PoseWindowFit], total_frames: int) -> torch.Tensor | None:
    if not results or results[0].contact_logits is None:
        return None
    template = results[0].contact_logits
    merged = template.new_zeros((total_frames, template.shape[1]))
    denominator = template.new_zeros(total_frames)
    for result in results:
        if result.contact_logits is None:
            raise ValueError("contact logits must be present in every M2 window")
        weight = hann_weights(
            result.window.length, result.contact_logits.device, result.contact_logits.dtype
        )
        merged[result.window.start : result.window.end] += result.contact_logits * weight[:, None]
        denominator[result.window.start : result.window.end] += weight
    return merged / denominator.clamp_min(1e-8)[:, None]


def _default_contact_edges() -> tuple[ContactEdgeSpec, ...]:
    # Corresponding finger-chain joints plus wrist-to-wrist; no all-to-all graph.
    pairs = [
        ("wrist_wrist", 20, 21, 0.006),
        ("index_index", 27, 42, 0.006),
        ("middle_middle", 30, 45, 0.006),
        ("pinky_pinky", 33, 48, 0.006),
        ("ring_ring", 36, 51, 0.006),
        ("thumb_thumb", 39, 54, 0.006),
        ("left_index_head", 27, 15, 0.018),
        ("right_index_head", 42, 15, 0.018),
        ("left_wrist_chest", 20, 9, 0.025),
        ("right_wrist_chest", 21, 9, 0.025),
        ("left_hand_right_shoulder", 27, 17, 0.018),
        ("right_hand_left_shoulder", 42, 16, 0.018),
    ]
    return tuple(
        ContactEdgeSpec(
            edge_id=edge_id,
            joint_a=left,
            joint_b=right,
            target_distance_m=target,
            enter_threshold_m=0.025,
            exit_threshold_m=0.04,
        )
        for edge_id, left, right, target in pairs
    )


def fit_smplx_sequence(
    batch: ObservationBatch,
    metadata: dict[str, object],
    config: MethodConfig,
    model_path: str | Path,
    fps: float,
    log_path: str | None = None,
    warm_start: PredictionArtifact | None = None,
) -> tuple[PredictionArtifact, dict[str, object]]:
    batch.validate()
    contact_registry_hash = None
    if config.contact.enabled:
        if not config.contact.registry:
            raise ValueError("M2 requires a frozen contact registry")
        registry_path = Path(config.contact.registry)
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        configured_edges = [edge["edge_id"] for edge in registry["edges"]]
        expected_edges = [edge.edge_id for edge in _default_contact_edges()]
        if configured_edges != expected_edges:
            raise ValueError("contact registry does not match the compiled canonical edge order")
        contact_registry_hash = sha256_file(registry_path)
    device = batch.joints_3d.device
    model = SMPLXWrapper(model_path).to(device)
    sigma = _sigma(batch, config)
    initial_rotation = _initial_pose(batch, sigma, initializer_mode=config.initializer_mode)
    translation = torch.tensor(metadata["translation"], device=device, dtype=batch.joints_3d.dtype)
    if warm_start is not None:
        if not torch.equal(warm_start.frame_ids.cpu(), batch.frame_ids.cpu()):
            raise ValueError("warm-start frame_ids do not match the observation batch")
        if warm_start.rotations is None or warm_start.rotations.shape != initial_rotation.shape:
            raise ValueError("warm-start rotations are missing or have an incompatible shape")
        if warm_start.translation.shape != translation.shape:
            raise ValueError("warm-start translation has an incompatible shape")
        initial_rotation = warm_start.rotations.to(device=device, dtype=initial_rotation.dtype)
        translation = warm_start.translation.to(device=device, dtype=translation.dtype)
        translation = translation * translation.new_tensor([1.0, -1.0, -1.0])
    use_legacy_shape = False
    if config.uncertainty.mode == "calibrated" and metadata.get("legacy_betas_mean"):
        legacy_selection = (
            batch.valid_rot[:, 2].all(-1)
            if config.initializer_mode == "legacy_full"
            else _coherent_legacy_selection(batch, sigma)
        )
        use_legacy_shape = int(legacy_selection.sum()) > batch.frame_ids.numel() // 2
    betas_key = "legacy_betas_mean" if use_legacy_shape else "betas_mean"
    betas = torch.tensor(metadata[betas_key], device=device, dtype=batch.joints_3d.dtype)[None]
    change = _change_probability(batch, fps, config.change_point.mode != "disabled")
    windows = plan_windows(
        batch.frame_ids.numel(), config.window.length, config.window.stride, change
    )
    logger = JsonlLogger(log_path, reset=True) if log_path else None
    results = [
        _fit_window(
            batch,
            model,
            window,
            initial_rotation,
            translation,
            betas,
            sigma,
            change,
            config,
            fps,
            logger,
        )
        for window in windows
    ]
    rotations, translation, betas = _merge_pose(results, batch.frame_ids.numel())
    merged_contact_logits = _merge_contact_logits(results, batch.frame_ids.numel())
    final_state = _make_state(rotations, translation, betas, trainable=False)
    with torch.inference_mode():
        output = model(final_state)
    uncertainty = sigma.mean((1, 3))
    risk = torch.stack(
        (
            uncertainty[:, :25].mean(-1),
            uncertainty[:, 25:40].mean(-1),
            uncertainty[:, 40:55].mean(-1),
        ),
        dim=-1,
    )
    contact_probability = None
    contacts = None
    if config.contact.enabled:
        candidates = propose_contacts(
            output.joints[:, :55],
            _default_contact_edges(),
            uncertainty=uncertainty,
            proposal_radius_m=config.contact.proposal_radius_m,
        )
        contact_probability = (
            torch.sigmoid(merged_contact_logits)
            if merged_contact_logits is not None
            else candidates.probability
        )
        contacts = decode_hysteresis(
            contact_probability,
            candidates.distance,
            config.contact.enter_probability,
            config.contact.exit_probability,
            config.contact.enter_distance_m,
            config.contact.exit_distance_m,
        )
    if config.uncertainty.mode == "calibrated":
        calibration_metrics = json.loads(
            (Path(str(config.uncertainty.artifact)) / "metrics.json").read_text(encoding="utf-8")
        )
        thresholds = risk.new_tensor(
            [
                calibration_metrics["abstention_thresholds_m"]["body"],
                calibration_metrics["abstention_thresholds_m"]["left_hand"],
                calibration_metrics["abstention_thresholds_m"]["right_hand"],
            ]
        )
        abstain = risk > thresholds
    else:
        abstain = torch.zeros_like(risk, dtype=torch.bool)
    prediction = PredictionArtifact(
        frame_ids=batch.frame_ids,
        joints_3d=output.joints[:, :55],
        rotations=rotations,
        translation=translation * translation.new_tensor([1.0, -1.0, -1.0]),
        vertices=output.vertices,
        risk_score=risk,
        abstain=abstain,
        uncertainty=uncertainty,
        contact_probability=contact_probability,
        contacts=contacts,
    )
    prediction.validate()
    diagnostics = {
        "model_sha256": model.model_hash,
        "windows": [[item.window.start, item.window.end] for item in results],
        "window_diagnostics": [copy.deepcopy(item.diagnostics) for item in results],
        "change_probability": change.detach().cpu().tolist(),
        "uncertainty_status": (
            "calibrated" if config.uncertainty.artifact else "proxy_uncalibrated"
        ),
        "warm_start": warm_start is not None,
        "contact_registry_sha256": contact_registry_hash,
    }
    if config.uncertainty.mode == "calibrated":
        legacy_selection = (
            batch.valid_rot[:, 2].all(-1)
            if config.initializer_mode == "legacy_full"
            else _coherent_legacy_selection(batch, sigma)
        )
        diagnostics["initializer_hypothesis_counts"] = {
            "m0_hybrid": int((~legacy_selection).sum()),
            "legacy_biomech_full": int(legacy_selection.sum()),
        }
        diagnostics["shape_source"] = "legacy_biomech" if use_legacy_shape else "smplerx"
    return prediction, diagnostics
