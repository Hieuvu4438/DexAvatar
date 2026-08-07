"""Deterministic/replayable K-hypothesis sub-VP posterior sampling."""

from __future__ import annotations

import torch

from phase3_posterior.geometry.relation_anchors import mask_relation_inputs
from phase3_posterior.losses.diffusion import SubVPSDE


def _observed_marginal(
    sde: SubVPSDE,
    clean: torch.Tensor,
    time: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    mean, std = sde.marginal(clean, time)
    return mean + std[:, None, None, None] * noise


@torch.no_grad()
def sample_candidates(
    model,
    batch: dict[str, torch.Tensor],
    sde: SubVPSDE,
    candidates: int = 4,
    steps: int = 50,
    seed: int = 42,
    observation_strength: float = 0.15,
    candidate_noise: torch.Tensor | None = None,
    condition_mask: torch.Tensor | None = None,
    rotation_hint_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return K candidates with candidate zero fixed to the initializer.

    When ``condition_mask`` is supplied, this performs conditional diffusion
    inpainting: observed joints follow one fixed forward-noise path throughout
    reverse integration and are restored exactly at the end. Hidden joints are
    not attracted to the initializer, and relational edges touching a hidden
    endpoint are removed from conditioning.
    """
    initial = batch["initial_state"]
    if condition_mask is not None:
        if condition_mask.shape != initial.shape[:-1]:
            raise ValueError("condition_mask must have shape (B,T,51)")
        condition_mask = condition_mask.to(
            device=initial.device, dtype=torch.bool
        ).clone()
        condition_mask &= batch["frame_valid"][..., None]
        conditioned_edges, conditioned_edge_valid = mask_relation_inputs(
            batch["edge_features"],
            batch["edge_valid"],
            batch["edge_index"],
            condition_mask,
        )
    else:
        conditioned_edges = batch["edge_features"]
        conditioned_edge_valid = batch["edge_valid"]
    if rotation_hint_mask is not None:
        if condition_mask is None:
            raise ValueError("rotation hints require a condition mask")
        if rotation_hint_mask.shape != initial.shape[:-1]:
            raise ValueError("rotation_hint_mask must have shape (B,T,51)")
        rotation_hint_mask = rotation_hint_mask.to(
            device=initial.device, dtype=torch.bool
        ) & batch["frame_valid"][..., None]
        if torch.any(rotation_hint_mask & condition_mask):
            raise ValueError("rotation hints and trusted conditioning must be disjoint")
    outputs = [initial.clone()]
    for candidate in range(1, candidates):
        generator = torch.Generator(device=initial.device).manual_seed(
            seed + candidate * 1_000_003
        )
        if candidate_noise is None:
            state = torch.randn(
                initial.shape,
                device=initial.device,
                dtype=initial.dtype,
                generator=generator,
            )
        else:
            if candidate_noise.shape != (
                initial.shape[0],
                candidates - 1,
                *initial.shape[1:],
            ):
                raise ValueError("candidate_noise has invalid shape")
            state = candidate_noise[:, candidate - 1].clone()
        observed_noise = torch.randn(
            initial.shape,
            device=initial.device,
            dtype=initial.dtype,
            generator=generator,
        )
        timeline = torch.linspace(1.0, sde.eps, steps + 1, device=initial.device)
        for step in range(steps):
            time = timeline[step].expand(initial.shape[0])
            if condition_mask is not None:
                observed = _observed_marginal(
                    sde, initial, time, observed_noise
                )
                state = torch.where(condition_mask[..., None], observed, state)
            dt = timeline[step + 1] - timeline[step]
            result = model(
                state,
                time,
                batch["features"],
                batch["frame_valid"],
                conditioned_edges,
                batch["edge_index"],
                conditioned_edge_valid,
                condition_mask,
                rotation_hint_mask=rotation_hint_mask,
            )
            drift = sde.probability_flow_drift(state, result["score"], time)
            state = state + drift * dt
            reliability = batch["features"][..., 29:30].clamp(0, 1)
            if condition_mask is not None:
                reliability = reliability * condition_mask[..., None]
            state = state + (
                observation_strength * dt.abs() * reliability * (initial - state)
            )
            if condition_mask is not None:
                next_time = timeline[step + 1].expand(initial.shape[0])
                next_observed = _observed_marginal(
                    sde, initial, next_time, observed_noise
                )
                state = torch.where(
                    condition_mask[..., None], next_observed, state
                )
            state = torch.where(batch["frame_valid"][..., None, None], state, initial)
        if condition_mask is not None:
            state = torch.where(condition_mask[..., None], initial, state)
        outputs.append(state)
    return torch.stack(outputs, dim=1)
