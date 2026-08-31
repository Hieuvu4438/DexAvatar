"""Hash-pinned bridge to the official DPoser-X whole-body score model."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from ml_collections.config_dict.config_dict import ConfigDict
from torch import Tensor, nn

from dcg_sign4d.diffusion.dposer_normalizer import DPoserXWholeBodyNormalizer
from dcg_sign4d.diffusion.state_codec import TrajectoryState, rotation_6d_to_matrix
from dcg_sign4d.geometry.so3 import log_map
from dcg_sign4d.utils.hashing import file_sha256

REQUIRED_RUNTIME_FILES: tuple[str, ...] = (
    "pretrained_models/body/BaseMLP/last.ckpt",
    "pretrained_models/hand/BaseMLP/last.ckpt",
    "pretrained_models/face/BaseMLP/last.ckpt",
    "pretrained_models/wholebody/mixed/last.ckpt",
    "data/body_data/body_normalizer/axis_normalize2.pt",
    "data/hand_data/hand_normalizer/axis_normalize2.pt",
    "data/face_data/jaw_normalizer/axis_normalize2.pt",
    "data/face_data/expression_normalizer/axis_normalize2.pt",
)

SELECTED_UPSTREAM_FILES: tuple[str, ...] = (
    "configs/wholebody/subvp/mixed.py",
    "lib/algorithms/advanced/model.py",
    "lib/algorithms/advanced/model_wholebody.py",
    "lib/algorithms/advanced/module.py",
    "lib/algorithms/advanced/sde_lib.py",
    "lib/algorithms/advanced/utils.py",
    "lib/algorithms/ema.py",
    "lib/utils/generic.py",
)


@dataclass(frozen=True)
class DPoserXDenoisedTarget:
    normalized: Tensor
    score: Tensor
    time: Tensor


def validate_runtime_registry(
    runtime_root: str | Path, registry_path: str | Path
) -> dict[str, Any]:
    runtime_root = Path(runtime_root)
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    entries = registry.get("files")
    if not isinstance(entries, list):
        raise ValueError("DPoser-X registry must contain a files list")
    registered = {entry.get("path") for entry in entries if isinstance(entry, dict)}
    if registered != set(REQUIRED_RUNTIME_FILES):
        raise ValueError("DPoser-X registry does not contain the exact runtime file set")
    for entry in entries:
        path = runtime_root / entry["path"]
        if not path.is_file() or file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"DPoser-X runtime hash mismatch: {path}")
    return registry


def validate_upstream_source(source_root: str | Path, expected_commit: str) -> None:
    source_root = Path(source_root)
    commit = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected_commit:
        raise ValueError(f"DPoser-X commit mismatch: expected {expected_commit}, got {commit}")
    diff = subprocess.run(
        ["git", "-C", str(source_root), "diff", "--quiet", "HEAD", "--", *SELECTED_UPSTREAM_FILES],
        check=False,
    )
    if diff.returncode != 0:
        raise ValueError("selected DPoser-X source files have uncommitted modifications")


def _safe_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load legacy Lightning tensors without enabling arbitrary pickle globals."""
    safe_globals: list[Any] = [
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        (np.dtype, "numpy.dtype"),
        np.dtypes.Float64DType,
        ConfigDict,
    ]
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"invalid DPoser-X checkpoint container: {path}")
    return checkpoint


def _apply_ema_checkpoint(model: nn.Module, checkpoint: dict[str, Any]) -> None:
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("DPoser-X checkpoint has no state_dict")
    prefix = "model."
    model_state = {
        key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)
    }
    model.load_state_dict(model_state, strict=True)
    ema = checkpoint.get("model_ema")
    if not isinstance(ema, dict) or not isinstance(ema.get("shadow_params"), list):
        raise ValueError("DPoser-X checkpoint has no valid EMA tensor list")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    shadows = ema["shadow_params"]
    if len(parameters) != len(shadows):
        raise ValueError("DPoser-X EMA parameter count mismatch")
    for parameter, shadow in zip(parameters, shadows, strict=True):
        if parameter.shape != shadow.shape:
            raise ValueError("DPoser-X EMA parameter shape mismatch")
        parameter.data.copy_(shadow.to(parameter))


@contextmanager
def _patched_upstream_loader(model_module: ModuleType, device: torch.device):
    original: Callable[..., Any] = model_module.load_model

    def safe_load(model: nn.Module, config: Any, ckpt_path: str, *_: Any, **__: Any) -> None:
        del config
        _apply_ema_checkpoint(model, _safe_checkpoint(ckpt_path, device))

    model_module.load_model = safe_load
    try:
        yield
    finally:
        model_module.load_model = original


class OfficialDPoserXBridge(nn.Module):
    """Frozen official per-frame score backbone used inside the DCG trajectory model."""

    def __init__(
        self,
        *,
        source_root: str | Path,
        runtime_root: str | Path,
        registry_path: str | Path,
        expected_commit: str,
        device: torch.device,
    ) -> None:
        super().__init__()
        source_root = Path(source_root).resolve()
        runtime_root = Path(runtime_root).resolve()
        validate_upstream_source(source_root, expected_commit)
        registry = validate_runtime_registry(runtime_root, registry_path)
        expected_hashes = {entry["path"]: entry["sha256"] for entry in registry["files"]}
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
        config_module = importlib.import_module("configs.wholebody.subvp.mixed")
        model_module = importlib.import_module("lib.algorithms.advanced.model_wholebody")
        sde_module = importlib.import_module("lib.algorithms.advanced.sde_lib")
        score_module = importlib.import_module("lib.algorithms.advanced.utils")
        config = config_module.get_config()
        config.model.body_ckpt = str(runtime_root / REQUIRED_RUNTIME_FILES[0])
        config.model.hand_ckpt = str(runtime_root / REQUIRED_RUNTIME_FILES[1])
        config.model.face_ckpt = str(runtime_root / REQUIRED_RUNTIME_FILES[2])
        with _patched_upstream_loader(model_module, device):
            model = model_module.create_wholebody_model(config.model, [21, 15, 103], 3)
        _apply_ema_checkpoint(
            model, _safe_checkpoint(runtime_root / REQUIRED_RUNTIME_FILES[3], device)
        )
        model.to(device).eval().requires_grad_(False)
        self.model = model
        self.sde = sde_module.subVPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        self.score_fn = score_module.get_score_fn(
            self.sde, self.model, train=False, continuous=config.training.continuous
        )
        self.normalizer = DPoserXWholeBodyNormalizer.from_runtime_root(
            runtime_root, expected_hashes=expected_hashes, device=device
        )
        self.source_commit = expected_commit
        self.registry_status = registry["status"]

    @staticmethod
    def trajectory_parts(state: TrajectoryState) -> dict[str, Tensor]:
        state.validate()
        batch, time = state.valid_mask.shape

        def axis(rotation: Tensor) -> Tensor:
            return log_map(rotation_6d_to_matrix(rotation)).reshape(batch * time, -1)

        jaw = state.root_translation.new_zeros(batch * time, 3)
        expression = state.root_translation.new_zeros(batch * time, 100)
        if state.face_state is not None:
            face = state.face_state.reshape(batch * time, -1)
            if face.shape[-1] != 19:
                raise ValueError("DCG face_state must be jaw/eyes/expression [19]")
            jaw = face[:, :3]
            expression[:, :10] = face[:, 9:19]
        return {
            "body_pose": axis(state.body_rot6d),
            "left_hand_pose": axis(state.left_hand_rot6d),
            "right_hand_pose": axis(state.right_hand_rot6d),
            "jaw_pose": jaw,
            "expression": expression,
        }

    def normalize_trajectory(self, state: TrajectoryState) -> Tensor:
        normalized = self.normalizer.normalize_parts(self.trajectory_parts(state))
        return normalized.reshape(*state.valid_mask.shape, -1)

    def predict_noise(
        self, normalized: Tensor, timesteps: Tensor, *, trajectory_steps: int
    ) -> Tensor:
        """Published DPoser-X epsilon output at DCG's discrete trajectory time."""
        if normalized.ndim != 2 or normalized.shape[-1] != 256:
            raise ValueError("normalized DPoser-X input must be [N,256]")
        if timesteps.shape != (normalized.shape[0],) or timesteps.dtype != torch.long:
            raise ValueError("DPoser-X timesteps must be long [N]")
        if (
            trajectory_steps < 2
            or bool((timesteps < 0).any())
            or bool((timesteps >= trajectory_steps).any())
        ):
            raise ValueError("DPoser-X timestep outside trajectory schedule")
        upstream_labels = timesteps.float() * (999.0 / (trajectory_steps - 1))
        return self.model(normalized, upstream_labels, None, None)

    @torch.inference_mode()
    def denoise_target(
        self, normalized: Tensor, time: Tensor, *, generator: torch.Generator
    ) -> DPoserXDenoisedTarget:
        if normalized.ndim != 2 or normalized.shape[-1] != 256:
            raise ValueError("normalized DPoser-X input must be [N,256]")
        if time.shape != (normalized.shape[0],) or not bool(((time > 0) & (time < 1)).all()):
            raise ValueError("DPoser-X continuous time must be [N] strictly inside (0,1)")
        noise = torch.randn(
            normalized.shape,
            dtype=normalized.dtype,
            device=normalized.device,
            generator=generator,
        )
        mean, std = self.sde.marginal_prob(normalized, time)
        perturbed = mean + std[:, None] * noise
        score = self.score_fn(perturbed, time, None, None)
        alpha, sigma = self.sde.return_alpha_sigma(time)
        denoised = (perturbed + sigma.square()[:, None] * score) / alpha
        return DPoserXDenoisedTarget(denoised, score, time.detach().clone())
