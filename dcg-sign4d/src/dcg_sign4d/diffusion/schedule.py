"""DPoser-X-compatible variance-preserving discrete diffusion schedule."""

from __future__ import annotations

import torch
from torch import Tensor


class DiffusionSchedule:
    def __init__(self, steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
        if steps < 2 or not 0 < beta_start < beta_end < 1:
            raise ValueError("invalid diffusion schedule")
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    @staticmethod
    def _extract(values: Tensor, timesteps: Tensor, target: Tensor) -> Tensor:
        return values.to(target.device)[timesteps].reshape(-1, *((1,) * (target.ndim - 1)))

    def q_sample(self, clean: Tensor, timesteps: Tensor, noise: Tensor) -> Tensor:
        alpha_bar = self._extract(self.alpha_bars, timesteps, clean)
        return alpha_bar.sqrt() * clean + (1 - alpha_bar).sqrt() * noise

    def predict_clean(self, noisy: Tensor, timesteps: Tensor, predicted_noise: Tensor) -> Tensor:
        alpha_bar = self._extract(self.alpha_bars, timesteps, noisy)
        return (noisy - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()

    def posterior_mean_variance(
        self, noisy: Tensor, clean: Tensor, timesteps: Tensor
    ) -> tuple[Tensor, Tensor]:
        beta = self._extract(self.betas, timesteps, noisy)
        alpha = self._extract(self.alphas, timesteps, noisy)
        alpha_bar = self._extract(self.alpha_bars, timesteps, noisy)
        previous_index = (timesteps - 1).clamp_min(0)
        previous_bar = self._extract(self.alpha_bars, previous_index, noisy)
        previous_bar = torch.where(
            timesteps.reshape(-1, *((1,) * (noisy.ndim - 1))) == 0,
            torch.ones_like(previous_bar),
            previous_bar,
        )
        denominator = (1 - alpha_bar).clamp_min(1e-12)
        first = beta * previous_bar.sqrt() / denominator
        second = (1 - previous_bar) * alpha.sqrt() / denominator
        mean = first * clean + second * noisy
        variance = beta * (1 - previous_bar) / denominator
        return mean, variance.clamp_min(0)
