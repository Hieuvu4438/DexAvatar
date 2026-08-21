from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import torch

from ..config import MethodConfig
from ..data.cache import ObservationBatch
from ..factors.contact import contact_factor
from ..factors.observation_3d import observation_3d_factor
from ..factors.temporal import adaptive_weights, temporal_position_factor
from ..io.predictions import PredictionArtifact
from ..models.change_point import rule_based_change_probability
from ..models.contact_proposer import (
    ContactEdgeSpec,
    decode_hysteresis,
    propose_contacts,
)
from ..models.uncertainty import heuristic_uncertainty
from ..utils.logging import JsonlLogger
from .consensus import merge_trajectories
from .window import Window, plan_windows


@dataclass
class WindowFit:
    window: Window
    joints: torch.Tensor
    uncertainty: torch.Tensor
    diagnostics: dict[str, object]
    contact_logits: torch.Tensor | None = None


def _source_sigma(observations: ObservationBatch, config: MethodConfig) -> torch.Tensor:
    if config.uncertainty.mode == "constant":
        sigma = torch.full_like(observations.joints_3d, config.uncertainty.sigma_min)
        return torch.where(
            observations.valid_3d[..., None],
            sigma,
            torch.full_like(sigma, config.uncertainty.sigma_max),
        )
    return heuristic_uncertainty(
        observations.features,
        observations.valid_3d,
        config.uncertainty.sigma_min,
        config.uncertainty.sigma_max,
    )["sigma_xyz"]


def _initialize_joints(observations: ObservationBatch, sigma: torch.Tensor) -> torch.Tensor:
    weight = observations.valid_3d[..., None] / sigma.square().clamp_min(1e-8)
    denominator = weight.sum(dim=1)
    if (denominator == 0).any():
        # Missing joints are initialized by temporal nearest-neighbor propagation, never dropped.
        denominator = denominator.clamp_min(1e-8)
    joints = (observations.joints_3d * weight).sum(dim=1) / denominator.clamp_min(1e-8)
    missing = observations.valid_3d.sum(dim=1) == 0
    for frame in range(joints.shape[0]):
        if missing[frame].any():
            if frame > 0:
                joints[frame, missing[frame]] = joints[frame - 1, missing[frame]]
            else:
                future = (~missing[:, missing[frame]]).float().argmax(dim=0)
                columns = torch.where(missing[frame])[0]
                joints[frame, columns] = torch.stack(
                    [joints[int(future[index]), joint] for index, joint in enumerate(columns)]
                )
    return joints


def _change_probability(joints: torch.Tensor, fps: float, enabled: bool) -> torch.Tensor:
    if not enabled or joints.shape[0] < 3:
        return joints.new_zeros(joints.shape[0])
    velocity = torch.zeros_like(joints)
    velocity[1:] = (joints[1:] - joints[:-1]) * fps
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    acceleration = torch.zeros_like(speed)
    acceleration[1:] = (speed[1:] - speed[:-1]).abs() * fps
    # Six aggregate cues: max/mean speed, acceleration and distal-joint motion.
    distal = speed[:, -min(10, speed.shape[1]) :]
    features = torch.stack(
        (
            speed.mean(-1),
            speed.max(-1).values,
            acceleration.mean(-1),
            acceleration.max(-1).values,
            distal.mean(-1),
            distal.max(-1).values,
        ),
        dim=-1,
    )
    return rule_based_change_probability(features)


def _default_contact_edges(joint_count: int) -> tuple[ContactEdgeSpec, ...]:
    if joint_count < 4:
        return ()
    midpoint = joint_count // 2
    candidates = [
        (max(0, midpoint - 1), midpoint),
        (max(0, midpoint - 2), min(joint_count - 1, midpoint + 1)),
    ]
    return tuple(
        ContactEdgeSpec(
            edge_id=f"canonical_{first}_{second}",
            joint_a=first,
            joint_b=second,
            target_distance_m=0.006,
            enter_threshold_m=0.015,
            exit_threshold_m=0.03,
        )
        for first, second in candidates
    )


def _fit_window(
    observations: ObservationBatch,
    window: Window,
    sigma: torch.Tensor,
    initial: torch.Tensor,
    change_probability: torch.Tensor,
    config: MethodConfig,
    fps: float,
    logger: JsonlLogger | None,
) -> WindowFit:
    local_obs = observations.joints_3d[window.start : window.end]
    local_valid = observations.valid_3d[window.start : window.end]
    local_sigma = sigma[window.start : window.end]
    local_change = change_probability[window.start : window.end]
    local_joint_uncertainty = local_sigma.mean(dim=(1, 3))

    joints = torch.nn.Parameter(initial[window.start : window.end].clone())
    edges = _default_contact_edges(joints.shape[1]) if config.contact.enabled else ()
    candidates = (
        propose_contacts(
            joints.detach(),
            edges,
            uncertainty=local_joint_uncertainty,
            proposal_radius_m=config.contact.proposal_radius_m,
        )
        if edges
        else None
    )
    contact_logits = None
    parameters: list[torch.nn.Parameter] = [joints]
    if candidates is not None:
        prior = candidates.probability.clamp(1e-4, 1 - 1e-4)
        contact_logits = torch.nn.Parameter(torch.logit(prior))
        parameters.append(contact_logits)

    optimizer = torch.optim.Adam(parameters, lr=config.solver.learning_rate)
    best_loss = float("inf")
    best_joints = joints.detach().clone()
    best_logits = contact_logits.detach().clone() if contact_logits is not None else None
    stale = 0
    recoveries = 0
    started = time.monotonic()

    temporal_uq = local_joint_uncertainty
    temporal_weight = adaptive_weights(
        temporal_uq,
        local_change,
        base_weight=1.0,
        gamma=config.change_point.gamma,
    )
    for step in range(config.solver.max_steps):
        optimizer.zero_grad(set_to_none=True)
        observation = observation_3d_factor(joints, local_obs, local_valid, local_sigma)
        temporal = temporal_position_factor(joints, fps, temporal_weight)
        loss = config.factors.get("observation", 1.0) * observation.loss
        loss = loss + config.factors.get("temporal", 0.0) * temporal.loss
        contact_result = None
        if contact_logits is not None and candidates is not None:
            contact_result = contact_factor(joints, contact_logits, candidates, local_change)
            loss = loss + config.factors.get("contact", 0.0) * contact_result.loss

        if not torch.isfinite(loss):
            recoveries += 1
            with torch.no_grad():
                joints.copy_(best_joints)
                if contact_logits is not None and best_logits is not None:
                    contact_logits.copy_(best_logits)
            for group in optimizer.param_groups:
                group["lr"] *= 0.5
            if recoveries > config.solver.retries_on_nonfinite:
                raise RuntimeError(f"non-finite optimization in window {window}")
            continue

        value = float(loss.detach())
        if value < best_loss * (1 - config.solver.relative_tolerance):
            best_loss = value
            best_joints = joints.detach().clone()
            best_logits = contact_logits.detach().clone() if contact_logits is not None else None
            stale = 0
        else:
            stale += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.solver.grad_clip_norm)
        optimizer.step()
        if logger is not None and (step == 0 or step % 20 == 0 or stale >= config.solver.patience):
            logger.write(
                {
                    "window": [window.start, window.end],
                    "stage": config.method_name,
                    "step": step,
                    "total_loss": value,
                    "factor_loss": {
                        "observation": float(observation.loss.detach()),
                        "temporal": float(temporal.loss.detach()),
                        "contact": float(contact_result.loss.detach())
                        if contact_result is not None
                        else 0.0,
                    },
                    "valid_residuals": {"observation": observation.valid_count},
                    "nonfinite_count": recoveries,
                    "elapsed_s": time.monotonic() - started,
                }
            )
        if stale >= config.solver.patience:
            break

    return WindowFit(
        window=window,
        joints=best_joints,
        uncertainty=local_joint_uncertainty,
        contact_logits=best_logits,
        diagnostics={"best_loss": best_loss, "steps": step + 1, "recoveries": recoveries},
    )


def _aggregate_region_risk(joint_uncertainty: torch.Tensor) -> torch.Tensor:
    joint_count = joint_uncertainty.shape[1]
    boundaries = [0, max(1, joint_count // 3), max(2, 2 * joint_count // 3), joint_count]
    return torch.stack(
        [
            joint_uncertainty[:, boundaries[index] : boundaries[index + 1]].mean(-1)
            for index in range(3)
        ],
        dim=-1,
    )


def fit_sequence(
    observations: ObservationBatch,
    config: MethodConfig,
    fps: float,
    log_path: str | None = None,
) -> tuple[PredictionArtifact, dict[str, object]]:
    observations.validate()
    sigma = _source_sigma(observations, config)
    initial = _initialize_joints(observations, sigma)
    change = _change_probability(initial, fps, config.change_point.mode != "disabled")
    windows = plan_windows(
        initial.shape[0],
        config.window.length,
        config.window.stride,
        transitions=change,
    )
    logger = JsonlLogger(log_path, reset=True) if log_path else None
    results = [
        _fit_window(observations, window, sigma, initial, change, config, fps, logger)
        for window in windows
    ]
    merged, merged_uncertainty = merge_trajectories(
        windows,
        [result.joints for result in results],
        [result.uncertainty for result in results],
        initial.shape[0],
    )

    contact_probability = None
    contacts = None
    if config.contact.enabled:
        candidates = propose_contacts(
            merged,
            _default_contact_edges(merged.shape[1]),
            uncertainty=merged_uncertainty,
            proposal_radius_m=config.contact.proposal_radius_m,
        )
        contact_probability = candidates.probability
        contacts = decode_hysteresis(
            candidates.probability,
            candidates.distance,
            config.contact.enter_probability,
            config.contact.exit_probability,
            config.contact.enter_distance_m,
            config.contact.exit_distance_m,
        )

    risk = _aggregate_region_risk(merged_uncertainty)
    threshold = torch.quantile(risk.detach().reshape(-1), 0.9)
    prediction = PredictionArtifact(
        frame_ids=observations.frame_ids.clone(),
        joints_3d=merged,
        rotations=None,
        translation=torch.zeros((merged.shape[0], 3), dtype=merged.dtype, device=merged.device),
        vertices=None,
        risk_score=risk,
        abstain=risk > threshold,
        uncertainty=merged_uncertainty,
        contact_probability=contact_probability,
        contacts=contacts,
    )
    prediction.validate()
    diagnostics = {
        "windows": [[result.window.start, result.window.end] for result in results],
        "window_diagnostics": [copy.deepcopy(result.diagnostics) for result in results],
        "change_probability": change.detach().cpu().tolist(),
        "calibration_status": "proxy" if config.uncertainty.mode != "constant" else "constant",
    }
    return prediction, diagnostics
