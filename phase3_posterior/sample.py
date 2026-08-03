"""Deterministic/replayable K-hypothesis sub-VP posterior sampling."""

from __future__ import annotations

import torch

from phase3_posterior.losses.diffusion import SubVPSDE


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
) -> torch.Tensor:
    """Return K candidates with candidate zero fixed to the initializer."""
    initial = batch["initial_state"]
    outputs = [initial.clone()]
    for candidate in range(1, candidates):
        if candidate_noise is None:
            generator = torch.Generator(device=initial.device).manual_seed(
                seed + candidate * 1_000_003
            )
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
        timeline = torch.linspace(1.0, sde.eps, steps + 1, device=initial.device)
        for step in range(steps):
            time = timeline[step].expand(initial.shape[0])
            dt = timeline[step + 1] - timeline[step]
            result = model(
                state,
                time,
                batch["features"],
                batch["frame_valid"],
                batch["edge_features"],
                batch["edge_index"],
                batch["edge_valid"],
            )
            beta = sde.beta(time)[:, None, None, None]
            drift = -0.5 * beta * state - beta * result["score"]
            state = state + drift * dt
            reliability = batch["features"][..., 29:30].clamp(0, 1)
            state = state + observation_strength * reliability * (initial - state)
            state = torch.where(batch["frame_valid"][..., None, None], state, initial)
        outputs.append(state)
    return torch.stack(outputs, dim=1)
