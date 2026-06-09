"""
PHD Score Network
=================
Score-based diffusion network cho body pose prior.
Learns s_θ(x_t, t, condition) ≈ ∇_x log p(x_t | condition)

Architecture:
- Timestep embedding: sinusoidal → MLP
- Condition embedding: SMPLer-X task tokens hoặc 2D keypoint features
- Denoising network: MLP-based với residual connections
- Output: predicted noise ε (63-dim)

References:
- PHD: Pose from Human Diffusion
- Song et al., "Score-Based Generative Modeling through SDEs"
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding cho diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    """Residual MLP block với LayerNorm."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.act(x + self.dropout(self.net(x)))


class PHDScoreNetwork(nn.Module):
    """
    Score-based diffusion network cho body pose prior.

    Predicts noise ε_θ(x_t, t, condition) tại mỗi diffusion timestep.
    Score function: ∇_x log p(x_t) ≈ -ε_θ(x_t, t) / √(1-ᾱ_t)

    Args:
        pose_dim: dimension của body pose (63 = 21 joints × 3 AA)
        condition_dim: dimension của conditioning signal (1024 cho SMPLer-X tokens)
        hidden_dim: hidden dimension
        num_layers: số residual blocks
        time_embed_dim: timestep embedding dimension
        dropout: dropout rate
        use_condition: có sử dụng conditioning không
    """

    def __init__(
        self,
        pose_dim=63,
        condition_dim=1024,
        hidden_dim=1024,
        num_layers=4,
        time_embed_dim=256,
        dropout=0.1,
        use_condition=True,
    ):
        super().__init__()

        self.pose_dim = pose_dim
        self.use_condition = use_condition

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition embedding (SMPLer-X features)
        if use_condition:
            self.condition_embed = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Pose input projection
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.SiLU(),
        )

        # Combine embeddings (always 3: pose + timestep + condition)
        # When no condition, use zeros
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Denoising network: stack of residual blocks
        self.denoise_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, pose_dim),
        )

        # Initialize last layer to small values
        nn.init.xavier_uniform_(self.output_proj[-1].weight, gain=0.01)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(self, x_t, t, condition=None):
        """
        Predict noise tại timestep t.

        Args:
            x_t: (B, pose_dim) - noisy body pose at timestep t
            t: (B,) - diffusion timestep (0 to T-1)
            condition: (B, condition_dim) - optional conditioning signal

        Returns:
            noise_pred: (B, pose_dim) - predicted noise ε
        """
        # Embed timestep
        t_emb = self.time_embed(t)  # (B, hidden_dim)

        # Embed pose
        x_emb = self.pose_proj(x_t)  # (B, hidden_dim)

        # Condition embedding
        if self.use_condition and condition is not None:
            c_emb = self.condition_embed(condition)  # (B, hidden_dim)
        else:
            # Use zeros when no condition provided
            c_emb = torch.zeros_like(t_emb)

        # Always combine 3 embeddings: pose + timestep + condition
        h = torch.cat([x_emb, t_emb, c_emb], dim=-1)
        h = self.combine(h)  # (B, hidden_dim)

        # Denoise through residual blocks
        for block in self.denoise_blocks:
            h = block(h)

        # Predict noise
        noise_pred = self.output_proj(h)  # (B, pose_dim)

        return noise_pred

    def score(self, x_t, t, condition=None):
        """
        Compute score function: ∇_x log p(x_t | condition)
        Score ≈ -ε_θ(x_t, t) / √(1-ᾱ_t)

        Args:
            x_t: (B, pose_dim) - noisy body pose
            t: (B,) - timestep
            condition: optional conditioning

        Returns:
            score: (B, pose_dim) - score gradient
        """
        noise_pred = self.forward(x_t, t, condition)
        # Score sẽ được normalize bởi caller (cần sqrt_one_minus_alpha_cumprod)
        return noise_pred
