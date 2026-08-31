"""Validation-only temperature scaling and reliability diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


def expected_calibration_error(probabilities: Tensor, labels: Tensor, bins: int = 15) -> float:
    if probabilities.ndim != 2 or labels.shape != probabilities.shape[:1]:
        raise ValueError("probabilities [N,C] and labels [N] required")
    confidence, prediction = probabilities.max(-1)
    correct = prediction == labels
    boundaries = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    ece = torch.zeros((), device=probabilities.device)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        member = (confidence > lower) & (confidence <= upper)
        if bool(member.any()):
            gap = confidence[member].mean() - correct[member].float().mean()
            ece = ece + member.float().mean() * gap.abs()
    return float(ece)


@dataclass(frozen=True)
class CalibrationReport:
    temperature: float
    raw_ece: float
    calibrated_ece: float
    raw_nll: float
    calibrated_nll: float
    samples: int
    fit_split: str

    @property
    def passes_no_worse_gate(self) -> bool:
        return self.calibrated_ece <= self.raw_ece + 1e-12


class TemperatureScaler(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        *,
        split: str,
        bins: int = 15,
        max_iter: int = 100,
    ) -> CalibrationReport:
        if split not in {"calibration", "validation"}:
            raise ValueError("calibrator fitting is restricted to calibration/validation")
        if logits.ndim != 2 or labels.shape != logits.shape[:1] or labels.dtype != torch.long:
            raise ValueError("logits [N,C] and long labels [N] required")
        if not torch.isfinite(logits).all() or labels.numel() == 0:
            raise ValueError("calibration data must be finite and non-empty")
        detached_logits, detached_labels = logits.detach(), labels.detach()
        raw_nll = float(functional.cross_entropy(detached_logits, detached_labels))
        raw_ece = expected_calibration_error(detached_logits.softmax(-1), detached_labels, bins)
        optimizer = torch.optim.LBFGS(
            [self.log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = functional.cross_entropy(self(detached_logits), detached_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        calibrated_logits = self(detached_logits).detach()
        report = CalibrationReport(
            temperature=float(self.temperature.detach()),
            raw_ece=raw_ece,
            calibrated_ece=expected_calibration_error(
                calibrated_logits.softmax(-1), detached_labels, bins
            ),
            raw_nll=raw_nll,
            calibrated_nll=float(functional.cross_entropy(calibrated_logits, detached_labels)),
            samples=labels.numel(),
            fit_split=split,
        )
        return report

    def save(self, path: str | Path, report: CalibrationReport) -> None:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"immutable calibration artifact exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {**report.__dict__, "passes_no_worse_gate": report.passes_no_worse_gate}
        destination.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def load_frozen_temperature(
    path: str | Path,
    *,
    require_pass: bool = True,
    allow_development: bool = False,
) -> TemperatureScaler:
    """Load a JSON-only frozen calibrator without executable checkpoint deserialization."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "dcg_temperature_calibration_v1":
        raise ValueError("unknown calibration artifact schema")
    if require_pass and payload.get("gate_status") != "PASS":
        raise PermissionError("calibration artifact did not pass its preregistered gate")
    if payload.get("development_only", False) and not allow_development:
        raise PermissionError("development calibrator cannot enter a production run")
    temperature = float(payload["temperature"])
    scaler = TemperatureScaler(temperature)
    scaler.requires_grad_(False)
    return scaler
