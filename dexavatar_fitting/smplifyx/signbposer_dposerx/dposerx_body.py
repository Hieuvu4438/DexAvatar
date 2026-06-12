"""
DPoserXBodyPrior: score-based diffusion prior for SMPL-X body pose.

Wraps a DPoser-X body checkpoint (21 joints * 3 axis-angle = 63 dims) so it
can be used as a body-pose prior inside the DexAvatar SMPLify-X fitting loop.

The interface mirrors the existing `PHDBodyPrior`:
    prior_loss(body_pose, condition=None, t=None) -> scalar loss

Implementation notes:
  * Imports `lib.algorithms.advanced.model.create_model` from the DPoser-X
    clone at /home/haipd/DexAvatar/DPoser-X/ (we mutate `sys.path` once).
  * Loads checkpoint with `lib.utils.generic.load_model(..., is_ema=True)`.
  * Uses the score-based noise-prediction MSE loss (PHD-style), which is
    stable in an inner-loop L-BFGS and matches the existing PHD branch.
"""
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the DPoser-X repo importable without polluting global sys.path too much.
_DPOSERX_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
    "DPoser-X",
)
# Path layout: .../DexAvatar/dexavatar_fitting/smplifyx/signbposer_dposerx/dposerx_body.py
# So we go up 4 levels to /home/haipd/DexAvatar/, then add DPoser-X/.
if os.path.isdir(_DPOSERX_REPO) and _DPOSERX_REPO not in sys.path:
    sys.path.insert(0, _DPOSERX_REPO)


class DPoserXBodyPrior(nn.Module):
    """Body pose prior wrapping a DPoser-X checkpoint.

    Args:
        config_path: path to the DPoser-X body config .py (e.g. `configs/body/subvp/timefc.py`).
        ckpt_path: path to the DPoser-X `.ckpt` file.
        body_normalizer_path: path to the `body_normalizer/` directory containing
            `axis_normalize1.pt` (min/max stats). Computed once by
            `scripts/fit_dposerx_normalizer.py` from sign-language data.
        device: torch device.
        batch_size: nominal batch size for timestep sampling.
        guidance_scale: weight for the prior loss.
        num_inference_steps: number of diffusion steps for post-fit denoising (unused in fitting).
        timestep_strategy: 'random' | 'fixed'.
        fixed_timestep: integer timestep for the 'fixed' strategy (0..N-1).
    """

    def __init__(self,
                 config_path: str,
                 ckpt_path: str,
                 body_normalizer_path: str,
                 device: str = "cuda",
                 batch_size: int = 32,
                 guidance_scale: float = 1.0,
                 num_inference_steps: int = 50,
                 timestep_strategy: str = "random",
                 fixed_timestep: int = 50):
        super().__init__()

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"DPoser-X config not found: {config_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"DPoser-X checkpoint not found: {ckpt_path}")
        if not os.path.isdir(body_normalizer_path):
            raise FileNotFoundError(
                f"DPoser-X body_normalizer dir not found: {body_normalizer_path}. "
                f"Run scripts/fit_dposerx_normalizer.py first."
            )

        # Import DPoser-X utilities (after sys.path tweak above).
        from lib.algorithms.advanced import sde_lib
        from lib.algorithms.advanced import utils as mutils
        from lib.algorithms.advanced.model import create_model
        from lib.dataset.body import N_POSES
        from lib.dataset.utils import Posenormalizer
        from lib.utils.generic import import_configs, load_model

        self._load_model = load_model  # for unit-testing
        self._mutils = mutils
        self._sde_lib = sde_lib

        config = import_configs(config_path)
        self._config = config
        self._device = device

        # Posenormalizer for axis-angle, min-max scheme (matches AMASS body config).
        self.Normalizer = Posenormalizer(
            data_path=body_normalizer_path,
            device=device,
            normalize=True,
            min_max=True,
            rot_rep="axis",
        )

        # Build the score model.
        POSE_DIM = 3 if config.data.rot_rep == "axis" else 6
        model = create_model(config.model, N_POSES, POSE_DIM)
        model.to(device)
        model.eval()
        load_model(model, config.model, ckpt_path, device, is_ema=True)
        self._model = model
        self._pose_dim = POSE_DIM
        self._n_poses = N_POSES
        self._pose_flat = N_POSES * POSE_DIM  # 63 for body

        # Build the SDE.
        sde_name = config.training.sde.lower()
        if sde_name == "subvpsde":
            sde = sde_lib.subVPSDE(beta_min=config.model.beta_min,
                                   beta_max=config.model.beta_max,
                                   N=config.model.num_scales)
        elif sde_name == "vpsde":
            sde = sde_lib.VPSDE(beta_min=config.model.beta_min,
                                beta_max=config.model.beta_max,
                                N=config.model.num_scales)
        elif sde_name == "vesde":
            sde = sde_lib.VESDE(sigma_min=config.model.sigma_min,
                                sigma_max=config.model.sigma_max,
                                N=config.model.num_scales)
        else:
            raise ValueError(f"Unknown SDE: {sde_name}")
        self.sde = sde

        # Wrapped score function.
        self.score_fn = mutils.get_score_fn(sde, model, train=False,
                                            continuous=config.training.continuous)

        self.batch_size = batch_size
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.timestep_strategy = timestep_strategy
        self.fixed_timestep = fixed_timestep

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_t(self, B: int, device: torch.device) -> torch.Tensor:
        """Sample a per-batch timestep t in [0, 1] (continuous)."""
        if self.timestep_strategy == "fixed":
            t = torch.full((B,), float(self.fixed_timestep) / max(self.sde.N - 1, 1),
                           device=device)
        else:
            eps = 1e-3
            t = torch.rand(B, device=device) * (self.sde.T - eps) + eps
        return t

    # ------------------------------------------------------------------
    # Prior loss
    # ------------------------------------------------------------------
    def prior_loss(self,
                   body_pose: torch.Tensor,
                   condition: Optional[torch.Tensor] = None,
                   t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Differentiable body-pose prior loss (PHD-style noise-prediction MSE).

        Args:
            body_pose: (B, 63) axis-angle SMPL-X body pose.
            condition: unused (DPoser-X body is unconditional).
            t: optional (B,) or scalar timestep in [0, 1]. Sampled if None.
        Returns:
            Scalar loss tensor, differentiable w.r.t. `body_pose`.
        """
        if body_pose.dim() == 3:
            B = body_pose.shape[0]
            body_pose = body_pose.reshape(B, -1)
        else:
            B = body_pose.shape[0]

        if body_pose.shape[-1] != self._pose_flat:
            raise ValueError(
                f"body_pose last dim must be {self._pose_flat} (={self._n_poses}*{self._pose_dim}), "
                f"got {body_pose.shape[-1]}"
            )

        device = body_pose.device
        # Normalize to the model's training scale.
        x0 = self.Normalizer.offline_normalize(body_pose)  # (B, pose_flat)

        # Sample t.
        if t is None:
            t = self._sample_t(B, device)
        elif t.dim() == 0:
            t = t.expand(B).to(device)
        else:
            t = t.to(device)

        # Forward SDE: x_t = alpha * x0 + sigma * z
        mean, std = self.sde.marginal_prob(x0, t)
        z = torch.randn_like(x0)
        x_t = mean + std[:, None] * z

        # Score: dPoser-X's get_score_fn returns the **negative score** directly
        # (i.e. -nabla_x log p), because of the line `score = -score / std`.
        # Concretely, the model output (with scale_by_sigma=True) is the
        # epsilon prediction * sigma. So `score_fn` returns -(eps / sigma).
        neg_score = self.score_fn(x_t, t, condition, mask=None)
        # Recover the epsilon prediction: eps = -neg_score * sigma
        eps_pred = -neg_score * std[:, None]

        # Standard DDPM noise-prediction loss.
        loss = F.mse_loss(eps_pred, z) * self.guidance_scale
        return loss

    # ------------------------------------------------------------------
    # Optional decode (multi-step denoising) — kept for parity, not used in fitting.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_to_pose(self,
                       body_pose: torch.Tensor,
                       num_steps: int = 10) -> torch.Tensor:
        """Multi-step denoising of a body pose (for post-fit refinement / sampling)."""
        from lib.algorithms.advanced.sde_lib import subVPSDE
        x = self.Normalizer.offline_normalize(body_pose)
        # Reverse-time SDE solver (mirror of DPoser-X multi_step_denoise).
        B = x.shape[0]
        device = x.device
        t_end = torch.full((B,), 1e-3, device=device)
        time_traj = torch.linspace(self.sde.T, 1e-3, num_steps + 1, device=device)
        for i in range(num_steps):
            t_current = time_traj[i]
            t_before = time_traj[i + 1]
            alpha_c, sigma_c = self.sde.return_alpha_sigma(t_current.expand(B))
            alpha_b, sigma_b = self.sde.return_alpha_sigma(t_before.expand(B))
            score = self.score_fn(x, t_current.expand(B), condition=None, mask=None)
            # model output = score * sigma, so eps = -score * sigma
            eps = -score * sigma_c[:, None]
            x = (alpha_b[:, None] / alpha_c[:, None] * (x - sigma_c[:, None] * eps)
                 + sigma_b[:, None] * eps)
        return self.Normalizer.offline_denormalize(x)

    def forward(self, body_pose: torch.Tensor,
                condition: Optional[torch.Tensor] = None,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.prior_loss(body_pose, condition, t)
