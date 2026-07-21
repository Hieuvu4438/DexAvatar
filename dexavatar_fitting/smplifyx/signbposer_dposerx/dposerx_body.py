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
  * Default loss mode 'x0_prediction' matches the original DPoser-X paper
    (ICCV 2025 Oral): one-step Tweedie denoising + SNR-weighted MSE.
    Alternative 'noise_prediction' uses simple eps-prediction MSE.
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
            ``axis_normalize1.pt`` (min/max stats). Computed once by
            ``scripts/fit_dposerx_normalizer.py``.  **Important:** DPoser-X
            was trained on AMASS; for best results use the original AMASS
            body normalizer at ``DPoser-X/data/body_data/body_normalizer/``.
        device: torch device.
        batch_size: nominal batch size for timestep sampling.
        guidance_scale: weight for the prior loss.
        num_inference_steps: number of diffusion steps for post-fit denoising (unused in fitting).
        timestep_strategy: 'random' | 'fixed'.
        fixed_timestep: integer timestep for the 'fixed' strategy (0..N-1).
        loss_mode: 'x0_prediction' (default, matches DPoser-X paper) or
            'noise_prediction' (legacy eps-prediction MSE).
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
                 fixed_timestep: int = 50,
                 loss_mode: str = "x0_prediction"):
        super().__init__()

        if loss_mode not in ("x0_prediction", "noise_prediction"):
            raise ValueError(
                f"loss_mode must be 'x0_prediction' or 'noise_prediction', got '{loss_mode}'"
            )

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

        # DPoser-X's import_configs uses importlib.import_module + a final
        # `getattr(_, function_name)` call. The expected format is therefore
        # `configs.body.subvp.timefc.get_config` (note the trailing
        # `.get_config` — the function inside the module). Accept either:
        #   - the full dotted path ending in `.get_config`
        #   - a filesystem path to the .py file
        if os.sep in config_path or config_path.endswith(".py"):
            norm = os.path.normpath(config_path)
            parts = norm.split(os.sep)
            if "configs" not in parts:
                raise ValueError(
                    f"Cannot derive module path from {config_path}. "
                    f"Pass the dotted module path instead "
                    f"(e.g. 'configs.body.subvp.timefc.get_config')."
                )
            i = parts.index("configs")
            rel = parts[i:]
            rel[-1] = rel[-1][:-3] if rel[-1].endswith(".py") else rel[-1]
            config_module = ".".join(rel) + ".get_config"
        elif not config_path.endswith(".get_config"):
            config_module = config_path + ".get_config"
        else:
            config_module = config_path

        config = import_configs(config_module)
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
        # DPoser-X's load_model uses torch.load (which defaults to
        # weights_only=True in torch>=2.6 and rejects numpy scalars).
        # We trust the Hugging Face checkpoint, so we monkey-patch
        # `torch.load` to weights_only=False for this single call. This is
        # safe because the only user-provided path is `--dposerx_ckpt`.
        import torch as _torch
        _orig_load = _torch.load
        def _load_trusted(*a, **kw):
            kw["weights_only"] = False
            return _orig_load(*a, **kw)
        _torch.load = _load_trusted
        try:
            load_model(model, config.model, ckpt_path, device, is_ema=True)
        finally:
            _torch.load = _orig_load
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
        self.loss_mode = loss_mode
        # L-BFGS requires a deterministic closure. Reuse one diffusion noise
        # sample while fitting a pose at a fixed timestep instead of drawing a
        # different target at every line-search evaluation.
        self._fixed_prior_noise = None

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

    def set_fixed_timestep(self, t):
        """Override the fixed timestep at runtime (used for per-stage annealing).

        With timestep_strategy='fixed', _sample_t reads self.fixed_timestep; this
        lets the fitting loop schedule a coarse-to-fine noise level across stages
        (e.g. [400, 200, 100, 50]) without changing the constructor or touching
        the loss code. No-op for other strategies.
        """
        t = int(t)
        if t != self.fixed_timestep:
            self._fixed_prior_noise = None
        self.fixed_timestep = t

    # ------------------------------------------------------------------
    # Prior loss
    # ------------------------------------------------------------------
    def prior_loss(self,
                   body_pose: torch.Tensor,
                   condition: Optional[torch.Tensor] = None,
                   t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Differentiable body-pose prior loss.

        Two modes (set via ``loss_mode`` in ``__init__``):

        ``'x0_prediction'`` (default, matches DPoser-X paper)
            One-step Tweedie denoising to estimate x̂₀ from x_t, then
            SNR-weighted MSE(x₀, x̂₀).  Gradients flow only through the
            MSE term (x̂₀ is detached), which is simpler and more stable
            in inner-loop L-BFGS.  This is exactly how the original
            DPoser-X ``DPoser_loss`` in ``smplify.py`` works.

        ``'noise_prediction'``
            Classic eps-prediction MSE: the score network predicts the
            noise ``z`` added during SDE perturbation.  Gradients flow
            through the full score network.

        Args:
            body_pose: (B, 63) axis-angle SMPL-X body pose.
            condition: unused (DPoser-X body is unconditional).
            t: optional (B,) or scalar timestep in [0, 1]. Sampled if None.

        Returns:
            Scalar loss tensor, differentiable w.r.t. ``body_pose``.
        """
        if body_pose.dim() == 3:
            B = body_pose.shape[0]
            body_pose = body_pose.reshape(B, -1)
        else:
            B = body_pose.shape[0]

        if body_pose.shape[-1] != self._pose_flat:
            raise ValueError(
                f"body_pose last dim must be {self._pose_flat} "
                f"(={self._n_poses}*{self._pose_dim}), "
                f"got {body_pose.shape[-1]}"
            )

        device = body_pose.device
        _zero = torch.zeros(1, device=device, dtype=body_pose.dtype)

        # Early bail if input body_pose already has NaN (corrupted params).
        if torch.isnan(body_pose).any():
            return _zero

        # Normalize to the model's training scale.
        x0 = self.Normalizer.offline_normalize(body_pose)  # (B, pose_flat)
        if torch.isnan(x0).any():
            return _zero

        # Sample t.
        if t is None:
            t = self._sample_t(B, device)
        elif t.dim() == 0:
            t = t.expand(B).to(device)
        else:
            t = t.to(device)

        # Forward SDE perturbation:  x_t = alpha * x0 + sigma * z
        mean, std = self.sde.marginal_prob(x0, t)
        if self.timestep_strategy == "fixed":
            if (self._fixed_prior_noise is None
                    or self._fixed_prior_noise.shape != x0.shape
                    or self._fixed_prior_noise.device != x0.device
                    or self._fixed_prior_noise.dtype != x0.dtype):
                self._fixed_prior_noise = torch.randn_like(x0)
            z = self._fixed_prior_noise
        else:
            z = torch.randn_like(x0)
        x_t = mean + std[:, None] * z

        if self.loss_mode == "x0_prediction":
            # --- Paper-matched: x₀-prediction loss with SNR weighting ---
            #
            # Tweedie's formula:  E[x₀ | x_t] = (x_t + σ² * score) / α
            # where α, σ² come from ``return_alpha_sigma`` (variance for
            # subVPSDE, std² for VPSDE — both are consistent with the
            # ``score_fn`` output convention).
            score = self.score_fn(x_t, t, condition, mask=None)
            alpha, sigma_sq = self.sde.return_alpha_sigma(t)

            # ``return_alpha_sigma`` returns alpha as (B, 1) but sigma as
            # (B,). Flatten both before adding the feature dimension below;
            # otherwise alpha[:, None] creates (B, 1, 1) and broadcasts the
            # denoising loss to an incorrect three-dimensional tensor.
            alpha = alpha.reshape(B)
            sigma_sq = sigma_sq.reshape(B)

            # One-step denoised estimate of x₀ (detached, matching paper).
            x0_pred = (x_t + sigma_sq[:, None] * score) / alpha[:, None]

            if torch.isnan(x0_pred).any():
                return _zero

            # SNR-based weighting: higher SNR → more reliable denoising →
            # higher weight.  Matches DPoser-X smplify.py L95.
            snr = alpha[:, None] / torch.sqrt(sigma_sq[:, None])
            weight = 0.5 * torch.sqrt(1.0 + snr ** 2)  # (B, 1)

            # MSE between clean x₀ and denoised estimate (x₀_pred detached
            # so gradients are simply weight * (x₀ - x̂₀), same as paper).
            loss_per_dim = F.mse_loss(x0, x0_pred.detach(), reduction="none")  # (B, D)
            loss = (weight * loss_per_dim).sum() / B
            loss = loss * self.guidance_scale
            return loss

        else:
            # --- Noise-prediction loss (legacy fallback) ---
            neg_score = self.score_fn(x_t, t, condition, mask=None)
            eps_pred = -neg_score * std[:, None]

            if torch.isnan(eps_pred).any():
                return _zero

            loss = F.mse_loss(eps_pred, z) * self.guidance_scale
            return loss

    # ------------------------------------------------------------------
    # Optional decode (multi-step denoising) — kept for parity, not used in fitting.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_to_pose(self,
                       body_pose: torch.Tensor,
                       num_steps: int = 10) -> torch.Tensor:
        """Multi-step denoising of a body pose (for post-fit refinement / sampling).

        Works around a GroupNorm batch_size=1 issue by duplicating the
        input to batch_size=2 when needed, then taking the first output.
        """
        x = self.Normalizer.offline_normalize(body_pose)
        B = x.shape[0]
        device = x.device

        # Work around GroupNorm requiring >1 channels in some PyTorch
        # versions by ensuring at least batch_size=2.
        _orig_B = B
        if B == 1:
            x = x.repeat(2, 1)
            B = 2

        t_end = torch.full((B,), 1e-3, device=device)
        time_traj = torch.linspace(self.sde.T, 1e-3, num_steps + 1, device=device)
        for i in range(num_steps):
            t_current = time_traj[i]
            t_before = time_traj[i + 1]
            alpha_c, sigma_c = self.sde.return_alpha_sigma(t_current.expand(B))
            alpha_b, sigma_b = self.sde.return_alpha_sigma(t_before.expand(B))
            # Ensure 2D input (B, D) for the score network.
            x_flat = x.reshape(B, -1)
            score = self.score_fn(x_flat, t_current.expand(B), condition=None, mask=None)
            # score to noise prediction (matches DPoser-X multi_step_denoise).
            # alpha is (B,1), sigma is (B,) — do NOT add [:,None] to alpha.
            eps = -score.reshape(B, -1) * sigma_c[:, None]
            x = (alpha_b / alpha_c * (x_flat - sigma_c[:, None] * eps)
                 + sigma_b[:, None] * eps)

        # Take only the first (original) output.
        if _orig_B == 1:
            x = x[:1]
        return self.Normalizer.offline_denormalize(x)

    def forward(self, body_pose: torch.Tensor,
                condition: Optional[torch.Tensor] = None,
                t: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.prior_loss(body_pose, condition, t)
