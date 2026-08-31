"""Objectives for proposal Stage 2 contact and Stage 3 trajectory training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from dcg_sign4d.contact.losses import (
    balanced_event_loss,
    invalid_transition_loss,
    masked_duration_loss,
    masked_event_brier_loss,
)
from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.contact.proposal import ContactProposalOutput
from dcg_sign4d.diffusion.contact_encoder import ConditioningMode
from dcg_sign4d.diffusion.schedule import DiffusionSchedule
from dcg_sign4d.diffusion.state_codec import StateCodec, TrajectoryState
from dcg_sign4d.diffusion.training import denoising_loss
from dcg_sign4d.observations.schema import ObservationBatch


@dataclass(frozen=True)
class ContactObjective:
    total: Tensor
    event: Tensor
    duration: Tensor
    transition: Tensor
    calibration: Tensor


def contact_objective(
    output: ContactProposalOutput,
    *,
    event_state: Tensor,
    duration_frames: Tensor,
    edge_valid: Tensor,
    frame_valid: Tensor,
    uncertain: Tensor,
    class_counts: Tensor,
    event_weight: float = 1.0,
    duration_weight: float = 1.0,
    transition_weight: float = 1.0,
    calibration_weight: float = 0.0,
    sample_weight: Tensor | None = None,
) -> ContactObjective:
    """Masked multi-task objective; uncertain labels have exactly zero direct loss."""

    if event_state.shape != output.event_logits.shape[:-1]:
        raise ValueError("event state shape mismatch")
    batch, time, edges = event_state.shape
    if edge_valid.shape != (batch, edges) or frame_valid.shape != (batch, time):
        raise ValueError("edge/frame validity shape mismatch")
    valid = edge_valid[:, None, :].expand_as(event_state)
    valid = valid & frame_valid[:, :, None]
    event = balanced_event_loss(
        output.event_logits,
        event_state,
        valid,
        uncertain,
        class_counts,
        sample_weight=sample_weight,
    )
    duration = masked_duration_loss(
        output.duration_logits,
        duration_frames,
        valid,
        uncertain,
        sample_weight=sample_weight,
    )
    transition = invalid_transition_loss(
        output.event_logits, frame_valid, edge_valid, sample_weight
    )
    calibration = masked_event_brier_loss(
        output.event_logits, event_state, valid, uncertain, sample_weight
    )
    total = event_weight * event + duration_weight * duration + transition_weight * transition
    total = total + calibration_weight * calibration
    return ContactObjective(
        total=total,
        event=event,
        duration=duration,
        transition=transition,
        calibration=calibration,
    )


@dataclass(frozen=True)
class DiffusionObjective:
    total: Tensor
    predicted_noise: Tensor
    target_noise: Tensor
    timesteps: Tensor


def diffusion_objective(
    denoiser: nn.Module,
    schedule: DiffusionSchedule,
    codec: StateCodec,
    token_encoder: nn.Module,
    trajectory: TrajectoryState,
    graph: ContactGraphBatch,
    observations: ObservationBatch,
    *,
    conditioning_mode: ConditioningMode,
    channel_weights: Tensor,
    generator: torch.Generator,
    supervision_mask: Tensor | None = None,
    timesteps: Tensor | None = None,
    noise: Tensor | None = None,
) -> DiffusionObjective:
    """One epsilon-prediction objective with part-level masking support."""

    clean, _ = codec.encode(trajectory)
    batch = clean.shape[0]
    if timesteps is None:
        timesteps = torch.randint(
            schedule.steps,
            (batch,),
            generator=generator,
            device=clean.device,
        )
    if timesteps.shape != (batch,) or timesteps.dtype != torch.long:
        raise ValueError("timesteps must be long [B]")
    if noise is None:
        noise = torch.randn(
            clean.shape,
            generator=generator,
            device=clean.device,
            dtype=clean.dtype,
        )
    if noise.shape != clean.shape:
        raise ValueError("noise shape mismatch")
    noisy = schedule.q_sample(clean, timesteps, noise)
    tokens = token_encoder(graph, conditioning_mode)
    observations.validate()
    frame_reliability = observations.keypoint_reliability.mean(-1)
    predicted = denoiser(
        noisy,
        timesteps,
        tokens,
        trajectory.valid_mask,
        frame_reliability,
        shape=trajectory.beta,
    )
    total = denoising_loss(
        predicted,
        noise,
        trajectory.valid_mask,
        channel_weights,
        supervision_mask,
    )
    return DiffusionObjective(total, predicted, noise, timesteps)
