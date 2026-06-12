"""
DPoser-X body-pose prior — drop-in additive replacement for SignBPoser.

Wraps DPoser-X (https://github.com/moonbow721/DPoser-X) as a body-pose prior
for the DexAvatar fitting pipeline. Designed to mirror the existing
`PHDBodyPrior` interface exactly so the `fitting.py` integration is a near
copy-paste of the PHD branch.

Interface:
    prior = DPoserXBodyPrior.from_checkpoint(config_path, ckpt_path, normalizer_dir, device)
    loss = prior.prior_loss(body_pose, condition=None, t=None)

The `prior_loss` returns a scalar differentiable loss w.r.t. `body_pose`,
following the PHD/MotionBERT noise-prediction MSE recipe.
"""
from .dposerx_body import DPoserXBodyPrior
from .loaders import load_signbposer_dposerx

__all__ = ["DPoserXBodyPrior", "load_signbposer_dposerx"]
