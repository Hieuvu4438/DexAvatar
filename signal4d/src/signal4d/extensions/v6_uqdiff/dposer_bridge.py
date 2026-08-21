from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

from ...geometry.so3 import exp_map, log_map
from ...utils.hashing import sha256_file
from .config import DPoserConfig
from .normalizer import WholeBodyAxisNormalizer

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


@dataclass(frozen=True)
class DenoisedTarget:
    normalized: torch.Tensor
    rotations: torch.Tensor
    snr: torch.Tensor
    time: torch.Tensor


def one_step_target(
    perturbed: torch.Tensor,
    alpha: torch.Tensor,
    sigma_squared: torch.Tensor,
    score: torch.Tensor,
) -> torch.Tensor:
    """Exact one-step estimator used by the published DPoser-X fitter."""
    if alpha.ndim == 1:
        alpha = alpha[:, None]
    return (perturbed + sigma_squared[:, None] * score) / alpha


def validate_runtime_registry(root: str | Path, registry_path: str | Path) -> dict[str, object]:
    root = Path(root).resolve()
    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = registry.get("files")
    if not isinstance(entries, list):
        raise ValueError("DPoser-X registry must contain a files list")
    expected = {str(item) for item in REQUIRED_RUNTIME_FILES}
    registered = {str(item.get("path")) for item in entries if isinstance(item, dict)}
    if registered != expected:
        missing = sorted(expected - registered)
        extra = sorted(registered - expected)
        raise ValueError(f"DPoser-X registry mismatch; missing={missing}, extra={extra}")
    for item in entries:
        path = root / str(item["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != item.get("sha256"):
            raise ValueError(f"DPoser-X runtime hash mismatch: {path}")
    return registry


def _git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_upstream_source(root: str | Path, expected_commit: str) -> None:
    root = Path(root).resolve()
    actual_commit = _git_output(root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"DPoser-X commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *SELECTED_UPSTREAM_FILES],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("selected DPoser-X source files contain uncommitted modifications")


def _import_config(
    path: str,
) -> tuple[object, ModuleType, ModuleType, ModuleType, ModuleType]:
    module_name, function_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    config = getattr(module, function_name)()
    model_module = importlib.import_module("lib.algorithms.advanced.model_wholebody")
    sde_module = importlib.import_module("lib.algorithms.advanced.sde_lib")
    score_utils_module = importlib.import_module("lib.algorithms.advanced.utils")
    generic_module = importlib.import_module("lib.utils.generic")
    return config, model_module, sde_module, score_utils_module, generic_module


class DPoserXBridge(nn.Module):
    """Pinned, inference-only bridge to published DPoser-X code and weights."""

    def __init__(self, config: DPoserConfig, device: torch.device) -> None:
        super().__init__()
        root = Path(config.source_root).resolve()
        validate_upstream_source(root, config.source_commit)
        self.registry = validate_runtime_registry(root, config.checkpoint_registry)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        modules = _import_config(config.upstream_config)
        upstream, model_module, sde_module, score_utils_module, generic_module = modules
        upstream.model.body_ckpt = str(root / "pretrained_models/body/BaseMLP/last.ckpt")
        upstream.model.hand_ckpt = str(root / "pretrained_models/hand/BaseMLP/last.ckpt")
        upstream.model.face_ckpt = str(root / "pretrained_models/face/BaseMLP/last.ckpt")
        model = model_module.create_wholebody_model(upstream.model, [21, 15, 103], 3)
        generic_module.load_model(
            model,
            upstream.model,
            str(root / "pretrained_models/wholebody/mixed/last.ckpt"),
            str(device),
            is_ema=True,
        )
        model.to(device).eval()
        model.requires_grad_(False)
        self.model = model
        self.sde = sde_module.subVPSDE(
            beta_min=upstream.model.beta_min,
            beta_max=upstream.model.beta_max,
            N=upstream.model.num_scales,
        )
        self.score_fn = score_utils_module.get_score_fn(
            self.sde,
            self.model,
            train=False,
            continuous=upstream.training.continuous,
        )
        self.normalizer = WholeBodyAxisNormalizer.from_dposer_data(root / "data", device)
        self.device = device

    @staticmethod
    def rotations_to_parts(
        rotations: torch.Tensor, expression: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if rotations.ndim != 4 or rotations.shape[1:] != (55, 3, 3):
            raise ValueError("canonical rotations must have shape [B,55,3,3]")
        batch = rotations.shape[0]
        axis = log_map(rotations)
        if expression is None:
            expression = rotations.new_zeros((batch, 100))
        if expression.shape != (batch, 100):
            raise ValueError("DPoser-X expression context must have shape [B,100]")
        return {
            "body_pose": axis[:, 1:22].flatten(1),
            "left_hand_pose": axis[:, 25:40].flatten(1),
            "right_hand_pose": axis[:, 40:55].flatten(1),
            "jaw_pose": rotations.new_zeros((batch, 3)),
            "expression": expression,
        }

    @staticmethod
    def parts_to_rotations(parts: dict[str, torch.Tensor], template: torch.Tensor) -> torch.Tensor:
        result = template.detach().clone()
        result[:, 1:22] = exp_map(parts["body_pose"].reshape(-1, 21, 3))
        result[:, 25:40] = exp_map(parts["left_hand_pose"].reshape(-1, 15, 3))
        result[:, 40:55] = exp_map(parts["right_hand_pose"].reshape(-1, 15, 3))
        return result

    @torch.inference_mode()
    def denoise_target(
        self,
        rotations: torch.Tensor,
        time: torch.Tensor,
        generator: torch.Generator,
    ) -> DenoisedTarget:
        if time.shape != (rotations.shape[0],):
            raise ValueError("diffusion time must have shape [B]")
        normalized = self.normalizer.normalize_parts(self.rotations_to_parts(rotations))
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
        denoised = one_step_target(perturbed, alpha, sigma**2, score).detach()
        parts = self.normalizer.denormalize_parts(denoised)
        target_rotations = self.parts_to_rotations(parts, rotations)
        snr = (alpha / sigma[:, None].clamp_min(1e-8)).detach()
        return DenoisedTarget(denoised, target_rotations, snr, time.detach().clone())
