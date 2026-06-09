"""
PHD Body Pose Prior
===================
Score-based diffusion prior cho body pose optimization.
Drop-in replacement cho SignBPoser trong DexAvatar fitting pipeline.

Usage:
    from signbposer.phd_prior import PHDBodyPrior

    # Load pretrained
    prior = PHDBodyPrior.from_checkpoint('checkpoints/phd_prior/best_model.pt')

    # Trong optimization loop:
    body_pose = smplerx_init.clone().requires_grad_(True)
    loss_prior = prior.prior_loss(body_pose, condition, t)
    loss_prior.backward()  # gradient guide body_pose toward realistic poses

Integration với DexAvatar:
    Old: pose_embedding (33-dim) → signbposer.decode() → body_pose
         L_prior = ||pose_embedding||²

    New: body_pose (63-dim) = direct optimization variable
         L_prior = score_guidance_loss(body_pose) từ PHD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# Import từ signbposer gốc để reuse rotation conversion
from signbposer import SignbPoser


class PHDBodyPrior(nn.Module):
    """
    PHD Body Prior: drop-in replacement cho SignBPoser.

    Sử dụng score-based diffusion để guide body pose optimization.
    Score function ∇_x log p(x) guide optimization về phía realistic poses.

    Args:
        diffusion: PHDDiffusion instance (chứa score_network + noise schedule)
        guidance_scale: weight cho prior loss
        num_inference_steps: số diffusion steps cho sampling
        timestep_strategy: 'random' hoặc 'fixed' cho optimization
        fixed_timestep: timestep cố định (nếu strategy='fixed')
    """

    def __init__(
        self,
        diffusion,
        guidance_scale=1.0,
        num_inference_steps=50,
        timestep_strategy='random',
        fixed_timestep=50,
    ):
        super().__init__()
        self.diffusion = diffusion
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.timestep_strategy = timestep_strategy
        self.fixed_timestep = fixed_timestep

    @classmethod
    def from_checkpoint(cls, checkpoint_path, device='cuda'):
        """
        Load PHD prior từ pretrained checkpoint.

        Args:
            checkpoint_path: path to .pt checkpoint
            device: device to load onto

        Returns:
            PHDBodyPrior instance
        """
        from .phd_score_network import PHDScoreNetwork
        from .phd_diffusion import PHDDiffusion

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Reconstruct score network
        config = checkpoint.get('config', {})
        score_network = PHDScoreNetwork(
            pose_dim=config.get('pose_dim', 63),
            condition_dim=config.get('condition_dim', 1024),
            hidden_dim=config.get('hidden_dim', 1024),
            num_layers=config.get('num_layers', 4),
            time_embed_dim=config.get('time_embed_dim', 256),
            dropout=config.get('dropout', 0.1),
            use_condition=config.get('use_condition', True),
        )
        score_network.load_state_dict(checkpoint['score_network_state_dict'])

        # Reconstruct diffusion
        diffusion = PHDDiffusion(
            score_network=score_network,
            num_timesteps=config.get('num_timesteps', 1000),
            beta_schedule=config.get('beta_schedule', 'cosine'),
        )

        # Create prior
        prior = cls(
            diffusion=diffusion,
            guidance_scale=config.get('guidance_scale', 1.0),
            num_inference_steps=config.get('num_inference_steps', 50),
        )

        prior.to(device)
        prior.eval()
        print(f"[PHDBodyPrior] Loaded checkpoint from {checkpoint_path}")

        return prior

    def decode(self, pose_input, output_type='matrot'):
        """
        Compatible interface với SignBPoser.decode().
        Pass-through vì body_pose đã ở đúng format.
        """
        assert output_type in ['matrot', 'aa']
        body_pose = pose_input.view(-1, 21, 3)

        if output_type == 'matrot':
            return SignbPoser.aa2matrot(body_pose).view(-1, 1, 21, 9)
        else:
            return body_pose.view(-1, 1, 21, 3)

    def encode(self, body_pose):
        """No-op encode (compatible interface)."""
        return body_pose

    def forward(self, body_pose, output_type='matrot'):
        """Forward pass (compatible interface)."""
        return {'pose_aa': self.decode(body_pose, output_type='aa').view(-1, 63),
                'mean': body_pose, 'std': torch.zeros_like(body_pose)}

    def sample_poses(self, num_poses, output_type='aa', seed=None):
        """Sample body poses từ diffusion prior."""
        if seed is not None:
            torch.manual_seed(seed)
        device = next(self.parameters()).device
        x_0 = self.diffusion.sample(
            shape=(num_poses, 63),
            device=device,
        )
        return x_0

    def _get_timestep(self, batch_size, device):
        """Get timestep cho prior loss computation."""
        if self.timestep_strategy == 'random':
            t = torch.randint(
                0, self.diffusion.num_timesteps, (batch_size,), device=device)
        else:
            t = torch.full(
                (batch_size,), self.fixed_timestep, device=device, dtype=torch.long)
        return t

    def prior_loss(self, body_pose, condition=None, t=None):
        """
        Compute prior loss: score-based guidance.

        Encourage body_pose nằm trong high-density region của learned distribution.

        Args:
            body_pose: (B, 63) - body pose (requires_grad=True)
            condition: (B, condition_dim) - optional conditioning
            t: (B,) - optional fixed timestep

        Returns:
            loss: scalar
        """
        B = body_pose.shape[0]
        device = body_pose.device

        if t is None:
            t = self._get_timestep(B, device)

        # Add noise to body_pose
        noise = torch.randn_like(body_pose)
        x_t = self.diffusion.q_sample(body_pose, t, noise)

        # Predict noise
        noise_pred = self.diffusion.score_network(x_t, t, condition)

        # Prior loss: encourage small predicted noise (→ body_pose is likely)
        # Large predicted noise → body_pose is far from data distribution
        loss = torch.norm(noise_pred, p=2, dim=-1).mean()

        return loss * self.guidance_scale

    def score_guidance(self, body_pose, condition=None, t=None):
        """
        Compute score gradient cho optimization.
        ∇_body_pose log p(body_pose | condition)

        Args:
            body_pose: (1, 63) - body pose (requires_grad=True)
            condition: optional conditioning
            t: optional timestep

        Returns:
            score_grad: (1, 63) - gradient direction
        """
        if t is None:
            t = self._get_timestep(body_pose.shape[0], body_pose.device)

        body_pose_detached = body_pose.detach().requires_grad_(True)

        noise_pred = self.diffusion.score_network(body_pose_detached, t, condition)

        # Score = -ε / √(1-ᾱ_t)
        self.diffusion._move_to_device(body_pose.device)
        sqrt_coeff = self.diffusion.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        score = -noise_pred / sqrt_coeff

        score.sum().backward()

        return body_pose_detached.grad

    def reconstruction_loss(self, body_pose, body_pose_gt):
        """
        Reconstruction loss: pull body_pose toward ground truth.
        Dùng kết hợp với prior loss.
        """
        return F.l1_loss(body_pose, body_pose_gt.detach())

    def init_prior_loss(self, body_pose, smplerx_init, core_weight=1.0, noncore_weight=1.0):
        """
        Initialization prior: L1 distance vs SMPLer-X init.
        Split vào core (11 joints) và non-core (10 joints).
        """
        core_loss = F.l1_loss(
            body_pose[:, :11*3], smplerx_init[:, :11*3].detach())
        noncore_loss = F.l1_loss(
            body_pose[:, 11*3:], smplerx_init[:, 11*3:].detach())
        return core_weight * core_loss + noncore_weight * noncore_loss
