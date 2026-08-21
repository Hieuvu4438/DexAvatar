from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from ...utils.hashing import sha256_json
from .joint_map import body_joint_indices


class DPoserConfig(BaseModel):
    enabled: bool = True
    source_root: str = "DPoser-X"
    source_commit: str
    checkpoint_registry: str
    upstream_config: str = "configs.wholebody.subvp.mixed.get_config"
    data_root: str = "DPoser-X/data"
    mode: Literal["euclidean", "geodesic"] = "geodesic"
    time_strategy: Literal["fixed", "linear"] = "linear"
    time_min: float = Field(default=0.08, gt=0, lt=1)
    time_max: float = Field(default=0.12, gt=0, lt=1)
    noise_seed: int = 12345
    refresh_denoised_target_every: int = Field(default=5, gt=0)

    @model_validator(mode="after")
    def validate_time_interval(self) -> DPoserConfig:
        if self.time_min > self.time_max:
            raise ValueError("time_min must be <= time_max")
        return self


class RefinementConfig(BaseModel):
    open_body_joints: list[str]
    optimize_left_hand: bool = False
    optimize_right_hand: bool = False
    max_steps: int = Field(default=30, gt=0)
    learning_rate: float = Field(default=2e-4, gt=0)
    grad_clip_norm: float = Field(default=5.0, gt=0)
    observation_weight: float = Field(default=1.0, ge=0)
    rotation_observation_weight: float = Field(default=0.1, ge=0)
    v5_anchor_weight: float = Field(default=0.1, ge=0)
    temporal_weight: float = Field(default=1e-3, ge=0)
    temporal_rotation_weight: float = Field(default=1e-3, ge=0)
    diffusion_weight: float = Field(default=1e-3, ge=0)
    seam_weight: float = Field(default=0.0, ge=0)
    uncertainty_aware: bool = True
    change_aware: bool = True

    @model_validator(mode="after")
    def validate_joint_contract(self) -> RefinementConfig:
        body_joint_indices(self.open_body_joints)
        if not self.open_body_joints and not (
            self.optimize_left_hand or self.optimize_right_hand
        ):
            raise ValueError("V6 must open at least one named parameter group")
        return self


class SafeGateConfig(BaseModel):
    enabled: bool = True
    require_objective_improvement: bool = True
    minimum_objective_improvement: float = Field(default=0.0, ge=0)
    max_rotation_delta_rad: float = Field(default=0.35, gt=0)
    max_uncertainty_ratio: float = Field(default=1.5, gt=0)
    transition_radius: int = Field(default=2, ge=0)


class V6Config(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    method_name: str
    base_method_config: str
    warm_start_root: str
    dposer: DPoserConfig
    refinement: RefinementConfig
    safe_gate: SafeGateConfig = SafeGateConfig()

    @property
    def open_body_indices(self) -> tuple[int, ...]:
        return body_joint_indices(self.refinement.open_body_joints)

    @property
    def sha256(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


def load_v6_config(path: str | Path) -> V6Config:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {path}")
    return V6Config.model_validate(value)
