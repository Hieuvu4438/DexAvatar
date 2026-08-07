"""Continuous sub-VP perturbation and region-balanced score matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SubVPSDE:
    beta_min: float = 0.1
    beta_max: float = 20.0
    eps: float = 1e-3

    def beta(self, time: torch.Tensor) -> torch.Tensor:
        return self.beta_min + time * (self.beta_max - self.beta_min)

    def diffusion_squared(self, time: torch.Tensor) -> torch.Tensor:
        """Return the exact public sub-VP diffusion coefficient squared."""
        discount = 1.0 - torch.exp(
            -2.0 * self.beta_min * time
            - (self.beta_max - self.beta_min) * time.square()
        )
        return self.beta(time) * discount

    def probability_flow_drift(
        self, state: torch.Tensor, score: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        beta = self.beta(time)[:, None, None, None]
        diffusion_squared = self.diffusion_squared(time)[:, None, None, None]
        return -0.5 * beta * state - 0.5 * diffusion_squared * score

    def log_mean_coeff(self, time: torch.Tensor) -> torch.Tensor:
        return (
            -0.25 * time**2 * (self.beta_max - self.beta_min)
            - 0.5 * time * self.beta_min
        )

    def marginal(
        self, clean: torch.Tensor, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_mean = self.log_mean_coeff(time)
        mean = torch.exp(log_mean)[:, None, None, None] * clean
        # This intentionally matches the public DPoser sub-VP implementation.
        std = (1.0 - torch.exp(2.0 * log_mean)).clamp_min(1e-6)
        return mean, std

    def snr(self, time: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio under this implementation's perturbation scale."""
        alpha = torch.exp(self.log_mean_coeff(time))
        std = (1.0 - alpha.square()).clamp_min(1e-6)
        return alpha.square() / std.square()

    def clipped_auxiliary_weight(
        self, time: torch.Tensor, gamma: float = 5.0
    ) -> torch.Tensor:
        """Suppress unstable x0 geometry gradients at high-noise timesteps."""
        if gamma <= 0:
            raise ValueError("auxiliary SNR gamma must be positive")
        return self.snr(time).clamp(max=gamma) / gamma

    def perturb(
        self, clean: torch.Tensor, time: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise
        mean, std = self.marginal(clean, time)
        return mean + std[:, None, None, None] * noise, noise, std

    def x0_from_score(
        self, noisy: torch.Tensor, score: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        log_mean = self.log_mean_coeff(time)
        alpha = torch.exp(log_mean)[:, None, None, None]
        std = (1.0 - torch.exp(2.0 * log_mean))[:, None, None, None].clamp_min(1e-6)
        return (noisy + std.square() * score) / alpha.clamp_min(1e-6)


def region_balanced_score_loss(
    score: torch.Tensor,
    noise: torch.Tensor,
    std: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    residual = (score * std[:, None, None, None] + noise).square().mean(dim=-1)
    regions = {
        "body": slice(0, 21),
        "left_hand": slice(21, 36),
        "right_hand": slice(36, 51),
    }
    values: dict[str, torch.Tensor] = {}
    weights = (
        torch.ones(residual.shape[0], device=residual.device)
        if sample_weight is None
        else sample_weight
    )
    for name, region in regions.items():
        mask = valid[..., region].to(residual.dtype) * weights[:, None, None]
        values[name] = (residual[..., region] * mask).sum() / mask.sum().clamp_min(1.0)
    return torch.stack(tuple(values.values())).mean(), values
