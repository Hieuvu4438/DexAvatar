"""Loader for DPoser-X body-pose prior.

Returns an instantiated `DPoserXBodyPrior` ready to drop into the fitting loop.
"""
import os
import torch

from .dposerx_body import DPoserXBodyPrior


def load_signbposer_dposerx(config_path: str,
                            ckpt_path: str,
                            body_normalizer_path: str,
                            device: str = "cuda",
                            guidance_scale: float = 1.0,
                            timestep_strategy: str = "random",
                            fixed_timestep: int = 50,
                            loss_mode: str = "x0_prediction"):
    """Instantiate a DPoserXBodyPrior.

    Args:
        config_path: e.g. `/home/haipd/DexAvatar/DPoser-X/configs/body/subvp/timefc.py`
        ckpt_path:   e.g. `/home/haipd/DexAvatar/checkpoints/dposerx_body/body.ckpt`
        body_normalizer_path: e.g. `/home/haipd/DexAvatar/checkpoints/dposerx_body/body_normalizer`
        device: torch device.
        guidance_scale, timestep_strategy, fixed_timestep: prior-loss controls.
        loss_mode: 'x0_prediction' (default, matches DPoser-X paper) or
                   'noise_prediction' (legacy eps-prediction MSE).

    Returns:
        A `DPoserXBodyPrior` instance (already on `device`, in eval mode).
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isdir(body_normalizer_path):
        raise FileNotFoundError(f"body_normalizer dir not found: {body_normalizer_path}")

    prior = DPoserXBodyPrior(
        config_path=config_path,
        ckpt_path=ckpt_path,
        body_normalizer_path=body_normalizer_path,
        device=device,
        guidance_scale=guidance_scale,
        timestep_strategy=timestep_strategy,
        fixed_timestep=fixed_timestep,
        loss_mode=loss_mode,
    )
    return prior.to(device)
