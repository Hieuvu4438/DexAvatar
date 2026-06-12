"""
SOKE VQVAE hand-pose prior — drop-in additive replacement for SignHPoser.

Wraps the SOKE (https://github.com/2000ZRL/SOKE) VQVAE for per-frame hand pose
(nfeats=45 = 15 SMPL-X hand joints * 3 axis-angle).

Interface (mirrors SignhPoser):
    SignHVQVAE(nfeats, code_num, code_dim, latent_dim, ...).decode_aa(pose_embedding)
    SignHVQVAE.recon_loss(pose_aa)
    SignHVQVAE.sample_poses(num)

The actual VQVAE is a 1-D-conv stack on the time axis, so a per-frame (T=1)
input is padded to T_min = stride_t ** down_t before the encoder runs.
Quantization is bypassed in the fitting path; the continuous pre-quantization
embedding is used as the differentiable bottleneck so gradients flow through
L-BFGS. The codebook itself is only used for sampling / post-fit refinement.
"""
from .vqvae_hand import SignHVQVAE
from .loaders import load_signhposer_vqvae

__all__ = ["SignHVQVAE", "load_signhposer_vqvae"]
