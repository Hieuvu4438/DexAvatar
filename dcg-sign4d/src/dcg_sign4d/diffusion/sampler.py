"""DPS-style observation/contact-guided reverse diffusion sampler."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.guidance.base import GuidanceTerm
from dcg_sign4d.observations.schema import ObservationBatch

from .schedule import DiffusionSchedule
from .state_codec import StateCodec, TrajectoryState


@dataclass
class SamplerDiagnostics:
    seed: int
    steps: int
    term_losses: dict[str, list[float]] = field(default_factory=dict)
    gradient_norms: dict[str, list[float]] = field(default_factory=dict)
    clip_count: int = 0
    nan_guard_count: int = 0
    aborted: bool = False


class GuidedTrajectorySampler:
    def __init__(
        self,
        denoiser: nn.Module,
        schedule: DiffusionSchedule,
        codec: StateCodec,
        token_encoder: nn.Module,
        guidance_terms: tuple[GuidanceTerm, ...] = (),
        guidance_scale: float = 0.0,
        gradient_clip_norm: float = 1.0,
        trust_region_norm: float = 10.0,
        conditioning_mode: str = "dynamic",
    ) -> None:
        self.denoiser = denoiser
        self.schedule = schedule
        self.codec = codec
        self.token_encoder = token_encoder
        self.guidance_terms = guidance_terms
        self.guidance_scale = guidance_scale
        self.gradient_clip_norm = gradient_clip_norm
        self.trust_region_norm = trust_region_norm
        self.conditioning_mode = conditioning_mode

    def sample(
        self,
        initial: TrajectoryState,
        graph: ContactGraphBatch,
        observations: ObservationBatch,
        seed: int,
        num_steps: int,
        guidance_scale_override: float | None = None,
    ) -> tuple[TrajectoryState, SamplerDiagnostics]:
        clean_initial, context = self.codec.encode(initial)
        if not 1 <= num_steps <= self.schedule.steps:
            raise ValueError("num_steps outside the diffusion schedule")
        generator = torch.Generator(device=clean_initial.device).manual_seed(seed)
        first_step = num_steps - 1
        timestep = torch.full(
            (clean_initial.shape[0],), first_step, dtype=torch.long, device=clean_initial.device
        )
        noise = torch.randn(
            clean_initial.shape,
            generator=generator,
            device=clean_initial.device,
            dtype=clean_initial.dtype,
        )
        noisy = self.schedule.q_sample(clean_initial, timestep, noise)
        contact_tokens = self.token_encoder(graph, self.conditioning_mode)
        diagnostics = SamplerDiagnostics(seed=seed, steps=num_steps)
        frame_reliability = observations.keypoint_reliability.mean(-1)
        guidance_scale = (
            self.guidance_scale if guidance_scale_override is None else guidance_scale_override
        )
        if guidance_scale < 0:
            raise ValueError("guidance scale cannot be negative")

        for step in reversed(range(num_steps)):
            timestep = torch.full(
                (clean_initial.shape[0],), step, dtype=torch.long, device=clean_initial.device
            )
            noisy = noisy.detach().requires_grad_(bool(self.guidance_terms and guidance_scale))
            predicted_noise = self.denoiser(
                noisy,
                timestep,
                contact_tokens,
                initial.valid_mask,
                frame_reliability,
                shape=initial.beta,
            )
            clean = self.schedule.predict_clean(noisy, timestep, predicted_noise)
            delta = clean - clean_initial
            norm = torch.linalg.vector_norm(delta.flatten(1), dim=-1).clamp_min(1e-8)
            factor = (self.trust_region_norm / norm).clamp_max(1.0)
            clean = clean_initial + delta * factor[:, None, None]

            total_gradient = torch.zeros_like(noisy)
            if self.guidance_terms and guidance_scale:
                clean_state = self.codec.decode(clean, context)
                for term in self.guidance_terms:
                    loss = term.loss(clean_state, observations, graph)
                    gradient = torch.autograd.grad(
                        loss, noisy, retain_graph=True, allow_unused=True
                    )[0]
                    if gradient is None:
                        gradient = torch.zeros_like(noisy)
                    if not torch.isfinite(gradient).all():
                        diagnostics.nan_guard_count += 1
                        gradient = torch.zeros_like(gradient)
                    gradient_norm = float(torch.linalg.vector_norm(gradient).detach())
                    diagnostics.term_losses.setdefault(term.name, []).append(float(loss.detach()))
                    diagnostics.gradient_norms.setdefault(term.name, []).append(gradient_norm)
                    if gradient_norm > self.gradient_clip_norm:
                        gradient = gradient * (self.gradient_clip_norm / max(gradient_norm, 1e-12))
                        diagnostics.clip_count += 1
                    # Lower guidance at noisier (larger-index) steps.
                    schedule_weight = 1 - step / max(num_steps, 1)
                    total_gradient = total_gradient + gradient * schedule_weight

            mean, variance = self.schedule.posterior_mean_variance(noisy, clean, timestep)
            mean = mean - guidance_scale * total_gradient
            if step > 0:
                reverse_noise = torch.randn(
                    noisy.shape,
                    generator=generator,
                    device=noisy.device,
                    dtype=noisy.dtype,
                )
                noisy = mean + variance.sqrt() * reverse_noise
            else:
                noisy = mean
            if not torch.isfinite(noisy).all():
                diagnostics.aborted = True
                raise FloatingPointError("reverse diffusion produced NaN/Inf")

        return self.codec.decode(noisy.detach(), context), diagnostics
