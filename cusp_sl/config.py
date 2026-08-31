"""Typed configuration and validation for the CUSP-SL implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    train_manifest: str
    val_manifest: str
    input_dim: int = 45
    window_size: int = 16
    fps: float = 24.0
    require_phase2r_semantics: bool = False


@dataclass(frozen=True)
class ReliabilityConfig:
    hidden_size: int = 96
    temporal_layers: int = 3
    body_tolerance_degrees: float = 15.0
    hand_tolerance_degrees: float = 20.0
    tau_low: float = 0.35
    tau_high: float = 0.75
    dilation: int = 1


@dataclass(frozen=True)
class FlowConfig:
    hidden_size: int = 192
    layers: int = 4
    heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    ode_steps: int = 3
    candidates: int = 4
    overlap: int = 2
    body_max_degrees: float = 25.0
    hand_max_degrees: float = 35.0
    normalization_statistics: str | None = None


@dataclass(frozen=True)
class FormConfig:
    enabled: bool = False
    video_feature_dim: int = 1024
    hidden_size: int = 256
    embedding_dim: int = 128
    temperature: float = 0.07
    counterfactual_margin: float = 0.15
    checkpoint: str | None = None


@dataclass(frozen=True)
class SelectionConfig:
    observation_weight: float = 1.0
    motion_weight: float = 0.5
    physical_weight: float = 0.25
    form_weight: float = 0.0
    energy_temperature: float = 1.0
    huber_delta: float = 0.03
    rom_threshold_degrees: float = 150.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    workers: int = 0
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    reliability_steps: int = 2500
    flow_steps: int = 10000
    validate_every: int = 500
    gradient_clip: float = 1.0
    precision: str = "bf16"
    real_fraction: float = 0.50
    synthetic_fraction: float = 0.35
    clean_fraction: float = 0.15


@dataclass(frozen=True)
class ProtocolConfig:
    baseline_root: str = "outputs/method_hamer"
    frames_root: str = "data/frames"
    gt_root: str = "data/smplx_gt"
    signs_file: str = "data/evaluation_from_author/data/data/signs.txt"
    segments_file: str = "data/evaluation_from_author/data/data/segment.json"
    assets_root: str = "data/evaluation_from_author/data/data"
    smplx_model_folder: str = "SMPLer-X/common/utils/human_model_files"
    expected_frames: int = 1493
    bootstrap_replicates: int = 10000


@dataclass(frozen=True)
class CUSPConfig:
    output_dir: str
    data: DataConfig
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    form: FormConfig = field(default_factory=FormConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)

    def validate(self) -> None:
        if self.data.input_dim not in (43, 45):
            raise ValueError("data.input_dim must be 43 or 45")
        if self.data.window_size < 2:
            raise ValueError("data.window_size must be at least 2")
        if not 0 <= self.reliability.tau_low < self.reliability.tau_high <= 1:
            raise ValueError("reliability thresholds must satisfy 0 <= low < high <= 1")
        if self.reliability.dilation < 0:
            raise ValueError("reliability.dilation must be non-negative")
        if self.flow.overlap < 0 or self.flow.overlap >= self.data.window_size:
            raise ValueError("flow.overlap must be in [0, window_size)")
        if self.flow.ode_steps < 2:
            raise ValueError("At least two Euler steps are required")
        if self.flow.candidates < 1:
            raise ValueError("flow.candidates must be positive")
        if self.flow.hidden_size % self.flow.heads:
            raise ValueError("flow.hidden_size must be divisible by flow.heads")
        if self.reliability.body_tolerance_degrees <= 0:
            raise ValueError("body reliability tolerance must be positive")
        if self.reliability.hand_tolerance_degrees <= 0:
            raise ValueError("hand reliability tolerance must be positive")
        if self.training.precision not in {"fp32", "bf16"}:
            raise ValueError("training.precision must be fp32 or bf16")
        mixture = (
            self.training.real_fraction + self.training.synthetic_fraction
            + self.training.clean_fraction
        )
        if abs(mixture - 1.0) > 1e-8:
            raise ValueError("real_fraction + synthetic_fraction + clean_fraction must equal one")
        if self.selection.form_weight > 0 and not self.form.enabled:
            raise ValueError("selection.form_weight requires form.enabled=true")


def _section(payload: dict[str, Any], name: str, cls):
    value = payload.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    return cls(**value)


def load_config(path: str | Path) -> CUSPConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("CUSP config must be a mapping")
    config = CUSPConfig(
        output_dir=str(payload["output_dir"]),
        data=_section(payload, "data", DataConfig),
        reliability=_section(payload, "reliability", ReliabilityConfig),
        flow=_section(payload, "flow", FlowConfig),
        form=_section(payload, "form", FormConfig),
        selection=_section(payload, "selection", SelectionConfig),
        training=_section(payload, "training", TrainingConfig),
        protocol=_section(payload, "protocol", ProtocolConfig),
    )
    config.validate()
    return config
