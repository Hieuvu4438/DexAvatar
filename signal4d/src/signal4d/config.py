from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .utils.hashing import sha256_json


class WindowConfig(BaseModel):
    length: int = Field(default=64, gt=0)
    stride: int = Field(default=32, gt=0)
    context: int = Field(default=8, ge=0)

    @model_validator(mode="after")
    def validate_stride(self) -> WindowConfig:
        if self.stride > self.length:
            raise ValueError("window.stride must be <= window.length")
        return self


class UncertaintyConfig(BaseModel):
    mode: str = "constant"
    sigma_min: float = Field(default=0.002, gt=0)
    sigma_max: float = Field(default=0.2, gt=0)
    artifact: str | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> UncertaintyConfig:
        if self.sigma_min >= self.sigma_max:
            raise ValueError("uncertainty sigma_min must be less than sigma_max")
        if self.mode == "calibrated" and not self.artifact:
            raise ValueError("calibrated uncertainty requires a frozen artifact directory")
        return self


class ChangePointConfig(BaseModel):
    mode: str = "rule_based"
    gamma: float = Field(default=2.0, ge=0)
    threshold: float = Field(default=0.7, ge=0, le=1)


class ContactConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    enabled: bool = False
    registry: str | None = None
    proposal_radius_m: float = Field(default=0.08, gt=0)
    enter_probability: float = Field(default=0.65, ge=0, le=1)
    exit_probability: float = Field(default=0.35, ge=0, le=1)
    enter_distance_m: float = Field(default=0.025, gt=0)
    exit_distance_m: float = Field(default=0.04, gt=0)

    @model_validator(mode="after")
    def validate_hysteresis(self) -> ContactConfig:
        if self.enter_probability <= self.exit_probability:
            raise ValueError("contact enter_probability must exceed exit_probability")
        if self.enter_distance_m >= self.exit_distance_m:
            raise ValueError("contact enter_distance_m must be below exit_distance_m")
        return self


class SolverConfig(BaseModel):
    learning_rate: float = Field(default=0.03, gt=0)
    max_steps: int = Field(default=120, gt=0)
    patience: int = Field(default=20, gt=0)
    grad_clip_norm: float = Field(default=10.0, gt=0)
    relative_tolerance: float = Field(default=1e-6, gt=0)
    retries_on_nonfinite: int = Field(default=2, ge=0)
    optimize_global: bool = True
    optimize_body: bool = True
    body_joint_indices: list[int] | None = None
    optimize_hands: bool = True
    optimize_left_hand: bool = True
    optimize_right_hand: bool = True
    optimize_translation: bool = True


class MethodConfig(BaseModel):
    schema_version: str = "1.0"
    method_name: str
    seed: int = 12345
    window: WindowConfig = WindowConfig()
    uncertainty: UncertaintyConfig = UncertaintyConfig()
    change_point: ChangePointConfig = ChangePointConfig()
    contact: ContactConfig = ContactConfig()
    factors: dict[str, float] = Field(default_factory=dict)
    observation_sources: list[int] | None = None
    initializer_mode: str = "m0_hybrid"
    solver: SolverConfig = SolverConfig()

    @model_validator(mode="after")
    def validate_method_contract(self) -> MethodConfig:
        if self.method_name.endswith("m2") and not self.contact.enabled:
            raise ValueError("M2 requires contact.enabled=true")
        if self.observation_sources is not None and (
            not self.observation_sources or min(self.observation_sources) < 0
        ):
            raise ValueError("observation_sources must contain non-negative source IDs")
        if self.initializer_mode not in {
            "m0_hybrid",
            "region_uncertainty",
            "coherent_uncertainty",
            "legacy_full",
        }:
            raise ValueError("unsupported initializer_mode")
        if self.solver.body_joint_indices is not None and any(
            index < 0 or index >= 21 for index in self.solver.body_joint_indices
        ):
            raise ValueError("body_joint_indices must be in [0, 20]")
        if not self.solver.optimize_hands and (
            self.solver.optimize_left_hand or self.solver.optimize_right_hand
        ):
            raise ValueError("per-hand optimization requires optimize_hands=true")
        return self

    @property
    def sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def load_method_config(path: str | Path) -> MethodConfig:
    return MethodConfig.model_validate(load_yaml(path))
