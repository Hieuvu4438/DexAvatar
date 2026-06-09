"""
PHD Diffusion Process
=====================
Forward diffusion process, noise schedule, và sampling cho PHD body pose prior.

Implements:
- Forward process: x_0 → x_t (add noise)
- Reverse process: x_t → x_0 (denoise)
- DDPM sampling
- DDIM sampling (faster)
- Training loss computation

References:
- Ho et al., "Denoising Diffusion Probabilistic Models" (DDPM)
- Song et al., "Denoising Diffusion Implicit Models" (DDIM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PHDDiffusion:
    """
    Diffusion process cho PHD body pose prior.

    Args:
        score_network: PHDScoreNetwork instance
        num_timesteps: số diffusion steps (T)
        beta_start: noise schedule start
        beta_end: noise schedule end
        beta_schedule: 'linear' hoặc 'cosine'
        prediction_type: 'noise' (predict ε) hoặc 'x0' (predict x_0)
    """

    def __init__(
        self,
        score_network,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule='cosine',
        prediction_type='noise',
    ):
        self.score_network = score_network
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type

        # Noise schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        """Cosine noise schedule (improved schedule từ Nichol & Dhariwal)."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def _move_to_device(self, device):
        """Move all tensors to device."""
        for attr_name in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                          'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                          'sqrt_recip_alphas', 'posterior_variance',
                          'posterior_log_variance', 'posterior_mean_coef1',
                          'posterior_mean_coef2']:
            tensor = getattr(self, attr_name)
            setattr(self, attr_name, tensor.to(device))

    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion: x_0 → x_t
        x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε

        Args:
            x_0: (B, D) - clean body pose
            t: (B,) - timestep
            noise: (B, D) - optional noise

        Returns:
            x_t: (B, D) - noisy body pose
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        self._move_to_device(x_0.device)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def predict_x0_from_noise(self, x_t, t, noise_pred):
        """
        Predict x_0 từ predicted noise.
        x_0 = (x_t - √(1-ᾱ_t) * ε) / √(ᾱ_t)
        """
        self._move_to_device(x_t.device)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Posterior q(x_{t-1} | x_t, x_0) mean và variance.
        """
        self._move_to_device(x_t.device)
        coef1 = self.posterior_mean_coef1[t].unsqueeze(-1)
        coef2 = self.posterior_mean_coef2[t].unsqueeze(-1)
        mean = coef1 * x_0 + coef2 * x_t
        variance = self.posterior_variance[t].unsqueeze(-1)
        log_variance = self.posterior_log_variance[t].unsqueeze(-1)
        return mean, variance, log_variance

    def training_loss(self, x_0, condition=None):
        """
        Compute training loss cho diffusion model.

        L = ||ε - ε_θ(x_t, t, condition)||²

        Args:
            x_0: (B, D) - clean body pose
            condition: (B, C) - optional conditioning

        Returns:
            loss: scalar
            dict: additional info (for logging)
        """
        B = x_0.shape[0]
        device = x_0.device
        self._move_to_device(device)

        # Random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=device)

        # Random noise
        noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        noise_pred = self.score_network(x_t, t, condition)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        info = {
            't': t,
            'noise_pred': noise_pred.detach(),
            'noise': noise,
        }

        return loss, info

    @torch.no_grad()
    def p_sample(self, x_t, t, condition=None):
        """
        Single reverse step: x_t → x_{t-1}
        DDPM sampling.

        Args:
            x_t: (B, D)
            t: int hoặc (B,) tensor
            condition: (B, C) optional

        Returns:
            x_{t-1}: (B, D)
        """
        device = x_t.device
        self._move_to_device(device)

        B = x_t.shape[0]
        if isinstance(t, int):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        else:
            t_batch = t

        # Predict noise
        noise_pred = self.score_network(x_t, t_batch, condition)

        # Get x_0 prediction
        if self.prediction_type == 'noise':
            x_0_pred = self.predict_x0_from_noise(x_t, t_batch, noise_pred)
        else:
            x_0_pred = noise_pred  #直接预测 x_0

        # Clamp x_0 (optional, helps stability)
        x_0_pred = torch.clamp(x_0_pred, -5.0, 5.0)

        # Posterior mean
        mean, variance, log_variance = self.q_posterior_mean_variance(
            x_0_pred, x_t, t_batch)

        # Sample
        noise = torch.randn_like(x_t)
        # No noise at t=0
        nonzero_mask = (t_batch != 0).float().unsqueeze(-1)
        x_prev = mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise

        return x_prev

    @torch.no_grad()
    def sample(self, shape, condition=None, device='cuda', return_trajectory=False):
        """
        Full reverse sampling: x_T → x_0

        Args:
            shape: tuple - (B, D)
            condition: (B, C) optional conditioning
            device: str
            return_trajectory: return full denoising trajectory

        Returns:
            x_0: (B, D) - sampled body poses
            trajectory: list of x_t (if return_trajectory=True)
        """
        # Start from pure noise
        x = torch.randn(shape, device=device)
        trajectory = [x] if return_trajectory else None

        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t, condition)
            if return_trajectory:
                trajectory.append(x)

        if return_trajectory:
            return x, trajectory
        return x

    @torch.no_grad()
    def ddim_sample(self, shape, condition=None, device='cuda',
                    ddim_steps=50, eta=0.0):
        """
        DDIM sampling (faster, deterministic when eta=0).

        Args:
            shape: tuple - (B, D)
            condition: optional conditioning
            device: str
            ddim_steps: số steps thực sự (ít hơn num_timesteps)
            eta: stochasticity (0=deterministic, 1=DDPM)

        Returns:
            x_0: (B, D) - sampled body poses
        """
        # Subsequence of timesteps
        step_size = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        # Start from noise
        x = torch.randn(shape, device=device)
        self._move_to_device(device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.score_network(x, t_batch, condition)

            # Get x_0 prediction
            x_0_pred = self.predict_x0_from_noise(x, t_batch, noise_pred)
            x_0_pred = torch.clamp(x_0_pred, -5.0, 5.0)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
            else:
                t_prev = 0

            # DDIM update
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
            alpha_prev = alpha_prev.to(device)

            sigma = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))

            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * noise_pred
            noise = torch.randn_like(x) if t_prev > 0 else torch.zeros_like(x)

            x = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + sigma * noise

        return x
