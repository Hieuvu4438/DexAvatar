"""
SignHVQVAE: SOKE VQVAE wrapper for per-frame SMPL-X hand pose.

This module is the user-facing API; the vendored SOKE bits live in
`vendored_soke/`. We re-export the model class as `SignHVQVAE` and provide
helpers that match the SignHPoser contract so the DexAvatar fitting loop can
swap them with minimal changes.

Differences vs. the original SignHPoser (signhposer.py):
  * `decode(Zin, output_type)` is the discrete-codebook path; the L-BFGS
    fitting loop should use `decode_aa(pose_embedding)` instead, which goes
    through the continuous pre-quantization embedding and is fully
    differentiable.
  * `recon_loss(pose_aa)` exposes the standard VQVAE reconstruction loss
    (output + commitment) so the fitting loop can add it as a regularizer.
  * `forward()` returns the standard VQVAE outputs (recon, commit_loss,
    perplexity) for use in training scripts.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vendored_soke.mgpt_vq import VQVae


def _min_T(stride_t: int, down_t: int) -> int:
    """Smallest T that the temporal conv stack can downsample without empty output."""
    return stride_t ** down_t


class SignHVQVAE(nn.Module):
    """SOKE VQVAE wrapped to match the SignHPoser contract.

    Args:
        nfeats: 45 for SMPL-X 15-joint hand axis-angle.
        code_num: 192 (SOKE hand192 config).
        code_dim: 512 (SOKE default).
        latent_dim: dim of the upstream L-BFGS variable (matches existing
            SignHPoser 23-dim `lhand_embedding3d`).
        down_t / stride_t: temporal conv config (SOKE default 3 / 2).
        width / depth / dilation_growth_rate: Resnet1D config.
        quantizer: 'ema_reset' | 'ema' | 'reset' | 'orig'.
    """

    def __init__(self,
                 nfeats: int = 45,
                 code_num: int = 192,
                 code_dim: int = 512,
                 latent_dim: int = 23,
                 down_t: int = 3,
                 stride_t: int = 2,
                 width: int = 512,
                 depth: int = 3,
                 dilation_growth_rate: int = 3,
                 quantizer: str = "ema_reset"):
        super().__init__()
        self.nfeats = nfeats
        self.num_joints = nfeats // 3
        self.latent_dim = latent_dim
        self.code_num = code_num
        self.code_dim = code_dim
        self.down_t = down_t
        self.stride_t = stride_t
        self.min_T = _min_T(stride_t, down_t)

        # Core SOKE VQVAE.
        self.vqvae = VQVae(
            nfeats=nfeats,
            quantizer=quantizer,
            code_num=code_num,
            code_dim=code_dim,
            output_emb_width=code_dim,
            down_t=down_t,
            stride_t=stride_t,
            width=width,
            depth=depth,
            dilation_growth_rate=dilation_growth_rate,
        )

        # Adapter: (B, latent_dim) -> (B, code_dim) for the continuous-bottleneck path.
        # Note: the encoder output width is `code_dim` (we set output_emb_width=code_dim).
        self.latent_to_emb = nn.Sequential(
            nn.Linear(latent_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.SiLU(),
        )

    # ------------------------------------------------------------------
    # Decoding helpers
    # ------------------------------------------------------------------
    def _expand_to_T(self, pose_aa_flat: torch.Tensor) -> torch.Tensor:
        """Pad a (B, nfeats) per-frame pose to (B, T_min, nfeats) for the encoder."""
        B = pose_aa_flat.shape[0]
        return pose_aa_flat.unsqueeze(1).expand(B, self.min_T, self.nfeats).contiguous()

    def _center_frame(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Take the center frame of a (B, T, nfeats) tensor (B, nfeats)."""
        T = x_seq.shape[1]
        return x_seq[:, T // 2, :]

    def decode_aa(self, pose_embedding: torch.Tensor) -> torch.Tensor:
        """Continuous-bottleneck decode: (B, latent_dim) -> axis-angle (B, 1, J, 3).

        This is the **fitting-time** decoder — gradients flow through it.
        """
        B = pose_embedding.shape[0]
        if pose_embedding.dim() == 1:
            pose_embedding = pose_embedding.unsqueeze(0)
        z = self.latent_to_emb(pose_embedding)  # (B, code_dim)
        # The VQVAE decoder expects (N, code_dim, T'). We need to pad the
        # continuous embedding to T' = min_T // (stride_t ** down_t) = 1 for
        # the SOKE default config (stride=2, down=3 -> T_min=8, T'=1).
        z = z.unsqueeze(-1)  # (B, code_dim, 1)
        out = self.vqvae.decode_continuous(z)  # (B, T_min, nfeats)
        out = self._center_frame(out)  # (B, nfeats)
        return out.view(B, 1, self.num_joints, 3)

    def encode_aa(self, pose_aa: torch.Tensor) -> torch.Tensor:
        """Encode a per-frame axis-angle pose to continuous embedding (B, code_dim)."""
        B = pose_aa.shape[0]
        if pose_aa.dim() == 4:
            pose_aa = pose_aa.view(B, 1, self.nfeats)
        pose_aa = self._expand_to_T(pose_aa.view(B, self.nfeats))
        z = self.vqvae.encode_continuous(pose_aa)  # (B, code_dim, 1)
        return z.squeeze(-1)

    # ------------------------------------------------------------------
    # SignHPoser-compatible API
    # ------------------------------------------------------------------
    def encode(self, Pin: torch.Tensor):
        """(B, 1, J, 3) or (B, J, 3) -> Normal distribution (dummy mean/std).

        Provided for API parity; not used by the fitting loop (we use the
        continuous path through `decode_aa` instead).
        """
        z = self.encode_aa(Pin)
        # Return a Normal with zero std so rsample() == mean. The discrete
        # codebook path is intentionally avoided in the fitting loop.
        return torch.distributions.normal.Normal(z, torch.ones_like(z) * 1e-3)

    def decode(self, Zin: torch.Tensor, output_type: str = "aa") -> torch.Tensor:
        """Discrete-codebook decode (used only for sampling/post-fit).

        For fitting, prefer `decode_aa` which is differentiable.
        """
        if Zin.dim() == 2:
            Zin = Zin.unsqueeze(0)  # (1, N_tokens)
        if Zin.dim() == 1:
            Zin = Zin.unsqueeze(0)
        out = self.vqvae.decode(Zin)  # (1, T, nfeats)
        out = self._center_frame(out)
        out = out.view(1, 1, self.num_joints, 3)
        if output_type == "aa":
            return SignHVQVAE.matrot2aa(out.reshape(out.shape[0], out.shape[1], self.num_joints, 3, 3))
        return out

    def forward(self, pose_aa: torch.Tensor):
        """Standard VQVAE forward (used by training scripts).

        Input  : (B, T, nfeats) axis-angle.
        Output : (recon, commit_loss, perplexity).
        """
        if pose_aa.dim() == 4:
            B = pose_aa.shape[0]
            pose_aa = pose_aa.view(B, 1, self.nfeats)
        recon, commit_loss, perplexity = self.vqvae(pose_aa)
        return recon, commit_loss, perplexity

    def recon_loss(self, pose_aa: torch.Tensor,
                   lambda_commit: float = 0.02,
                   lambda_recon: float = 1.0) -> torch.Tensor:
        """Reconstruction + commitment loss for fitting-time regularizer."""
        recon, commit_loss, _ = self.forward(pose_aa)
        rec_l1 = (recon - pose_aa).abs().mean()
        return lambda_recon * rec_l1 + lambda_commit * commit_loss

    def sample_poses(self, num_poses: int, seed: Optional[int] = None) -> torch.Tensor:
        """Sample `num_poses` poses from the discrete codebook.

        Returns axis-angle of shape (1, 1, J, 3) — same as SignhPoser.sample_poses.
        """
        import numpy as np
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        device = self.vqvae.quantizer.codebook.device
        dtype = self.vqvae.quantizer.codebook.dtype
        # Sample one token from the codebook at random; replicate to T_min.
        idx = torch.randint(0, self.code_num, (1,), device=device)
        # Decode requires a token index, so build a (1, T) integer code.
        code = idx.view(1, 1)
        out = self.vqvae.decode(code)  # (1, T_min, nfeats)
        out = self._center_frame(out)
        out = out.view(1, 1, self.num_joints, 3)
        return out

    # ------------------------------------------------------------------
    # Rotation conversion (mirrors SignhPoser)
    # ------------------------------------------------------------------
    @staticmethod
    def matrot2aa(pose_matrot: torch.Tensor) -> torch.Tensor:
        """(B, 1, J, 3, 3) matrot -> (B, 1, J, 3) axis-angle."""
        try:
            import torchgeometry as tgm
        except ImportError as e:
            raise ImportError(
                "torchgeometry is required for axis-angle conversion. "
                "Install with: pip install torchgeometry"
            ) from e
        batch_size = pose_matrot.shape[0]
        homogen = F.pad(pose_matrot.reshape(-1, 3, 3), [0, 1])
        return tgm.rotation_matrix_to_angle_axis(homogen).view(batch_size, 1, -1, 3)

    @staticmethod
    def aa2matrot(pose_aa: torch.Tensor) -> torch.Tensor:
        try:
            import torchgeometry as tgm
        except ImportError as e:
            raise ImportError("torchgeometry is required for axis-angle conversion.") from e
        batch_size = pose_aa.shape[0]
        return tgm.angle_axis_to_rotation_matrix(pose_aa.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
