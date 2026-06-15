"""Loader for the SOKE VQVAE hand-pose checkpoint.

Loads a Lightning `.ckpt` from SOKE's `experiments/mgpt/vae/checkpoints/tokenizer.ckpt`
(or a fine-tuned version) into a `SignHVQVAE` instance.
"""
import os

import torch

from .vqvae_hand import SignHVQVAE
from .vendored_soke.neq_load_customized import neq_load_customized


# Default SOKE hand192 config (from SOKE/configs/vq/hand192.yaml).
DEFAULT_VQVAE_CONFIG = dict(
    nfeats=45,
    code_num=192,
    code_dim=512,
    down_t=2,           # hand192.yaml uses down_t=2 (not the SOKE default 3)
    stride_t=2,
    width=512,
    depth=3,
    dilation_growth_rate=3,
    quantizer="ema_reset",
)


def load_signhposer_vqvae(ckpt_path: str = "",
                          config_yaml: str = "",
                          latent_dim: int = 23,
                          map_location: str = "cpu",
                          verbose: bool = False):
    """Build a `SignHVQVAE` and (optionally) load weights from a Lightning .ckpt.

    Args:
        ckpt_path: path to a `.ckpt` file. If empty, returns an untrained model.
        config_yaml: optional YAML path; if absent, uses SOKE hand192 defaults.
        latent_dim: matches the upstream SignHPoser 23-dim latent (default 23).
        map_location: torch device mapping for the load.
        verbose: print key/shape diagnostics from `neq_load_customized`.

    Returns:
        (model, config_dict) tuple.
    """
    cfg = dict(DEFAULT_VQVAE_CONFIG)
    if config_yaml and os.path.exists(config_yaml):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required to read VQVAE config YAMLs.") from e
        with open(config_yaml) as f:
            yaml_cfg = yaml.safe_load(f)
        params = yaml_cfg.get("params", yaml_cfg)
        for k in ("nfeats", "code_num", "code_dim", "down_t", "stride_t",
                  "width", "depth", "dilation_growth_rate", "quantizer"):
            if k in params:
                cfg[k] = params[k]

    # If a checkpoint is provided and was saved by our training script, the
    # saved `config` dict is the source of truth (overrides the SOKE default
    # and any YAML). This is important for fine-tuned models that may have
    # shrunk the codebook (code_num=64 for our sign-data fine-tune).
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=map_location)
        saved_cfg = None
        if isinstance(state, dict) and "config" in state:
            saved_cfg = state["config"]
        if saved_cfg is not None:
            for k, v in saved_cfg.items():
                if k in ("nfeats", "code_num", "code_dim", "down_t", "stride_t",
                         "width", "depth", "dilation_growth_rate", "quantizer",
                         "latent_dim"):
                    cfg[k] = v
            if verbose:
                print(f"[load_signhposer_vqvae] using saved config: code_num={cfg.get('code_num')}, "
                      f"code_dim={cfg.get('code_dim')}, width={cfg.get('width')}")

    cfg["latent_dim"] = latent_dim
    model = SignHVQVAE(**cfg)

    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=map_location)
        # SOKE saves Lightning .ckpt with a 'state_dict' key. Sliced (hand-only)
        # weights have keys prefixed by `hand_vae.`.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Strip the prefix to match our `vqvae.` namespace.
        stripped = {}
        for k, v in state.items():
            if k.startswith("hand_vae."):
                stripped[k[len("hand_vae."):]] = v
            elif k.startswith("vqvae."):
                stripped[k[len("vqvae."):]] = v
            else:
                stripped[k] = v
        neq_load_customized(model.vqvae, stripped, verbose=verbose)
        if verbose:
            print(f"[load_signhposer_vqvae] loaded from {ckpt_path}")
    elif ckpt_path:
        print(f"[load_signhposer_vqvae] WARNING: ckpt not found: {ckpt_path} — returning random-init model")

    model.eval()
    return model, cfg
