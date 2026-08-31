from __future__ import annotations

import ast
import configparser
import importlib.util
from pathlib import Path

import torch
from torch import Tensor, nn

from signpk.geometry.rotations import matrix_to_axis_angle
from signpk.utils.config_hash import sha256_file


def _load_vae(experiment_root: Path, class_name: str, source_glob: str) -> tuple[nn.Module, dict[str, object]]:
    ini_paths = sorted(experiment_root.glob("*.ini"))
    checkpoint_paths = sorted(experiment_root.glob("snapshots/*.pt"), key=lambda path: path.stat().st_mtime)
    source_paths = sorted(experiment_root.glob(source_glob))
    if not ini_paths or not checkpoint_paths or not source_paths:
        raise FileNotFoundError(f"incomplete DexAvatar prior at {experiment_root}")
    parser = configparser.ConfigParser()
    parser.read(ini_paths[0])
    values = parser["All"]
    module_spec = importlib.util.spec_from_file_location(f"signpk_{class_name}", source_paths[0])
    if module_spec is None or module_spec.loader is None:
        raise ImportError(source_paths[0])
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = getattr(module, class_name)(
        num_neurons=values.getint("num_neurons"),
        latentD=values.getint("latentD"),
        data_shape=ast.literal_eval(values["data_shape"]),
    )
    state = torch.load(checkpoint_paths[-1], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "experiment_root": str(experiment_root),
        "config": str(ini_paths[0]),
        "checkpoint": str(checkpoint_paths[-1]),
        "checkpoint_sha256": sha256_file(checkpoint_paths[-1]),
        "latent_dim": values.getint("latentD"),
    }


class DexSignPriors(nn.Module):
    """Frozen SignBPoser/SignHPoser latent safeguards.

    This wrapper evaluates local rotations through the original encoders and
    returns only a weak normalized latent penalty. It never uses raw
    axis-angle subtraction as a rotation distance.
    """

    def __init__(self, body_root: str | Path, hand_root: str | Path):
        super().__init__()
        self.body, body_metadata = _load_vae(Path(body_root), "SignbPoser", "signbposer.py")
        self.hand, hand_metadata = _load_vae(Path(hand_root), "SignhPoser", "signhposer*.py")
        self.metadata = {"body": body_metadata, "hand": hand_metadata}

    @staticmethod
    def _posterior_penalty(model: nn.Module, rotations: Tensor) -> Tensor:
        axis_angle = matrix_to_axis_angle(rotations).unsqueeze(1)
        posterior = model.encode(axis_angle)
        return posterior.mean.square().mean()

    def forward(self, body_rotmat: Tensor, left_hand_rotmat: Tensor, right_hand_rotmat: Tensor) -> Tensor:
        return (
            self._posterior_penalty(self.body, body_rotmat)
            + self._posterior_penalty(self.hand, left_hand_rotmat)
            + self._posterior_penalty(self.hand, right_hand_rotmat)
        ) / 3

