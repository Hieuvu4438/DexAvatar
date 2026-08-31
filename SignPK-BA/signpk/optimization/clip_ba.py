from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .factors import (
    FactorInputs,
    compute_factors,
    normalized_weighted_sum,
    validate_factor_inputs,
)
from .fallback import passes_sanity_checks
from .smplx_layer import SMPLXOutput
from .state import SequenceState


@dataclass
class StageRecord:
    stage: str
    iteration: int
    loss: float
    factors: dict[str, float]
    accepted: bool
    fallback_reason: str | None = None


@dataclass
class OptimizationResult:
    state: SequenceState
    output: SMPLXOutput
    records: list[StageRecord] = field(default_factory=list)
    fallback_events: list[dict[str, Any]] = field(default_factory=list)


class ClipBundleAdjuster:
    def __init__(
        self,
        model: Callable[[SequenceState], SMPLXOutput],
        stages: list[dict[str, Any]],
        *,
        scales: dict[str, float] | None = None,
        grad_clip_norm: float = 1.0,
        sanity: dict[str, float] | None = None,
    ):
        self.model = model
        self.stages = stages
        self.scales = scales or {}
        self.grad_clip_norm = grad_clip_norm
        self.sanity = sanity or {}

    def _sanity(self, state: SequenceState, output: SMPLXOutput) -> tuple[bool, str | None]:
        return passes_sanity_checks(
            state,
            output,
            max_body_degrees=self.sanity.get("max_body_residual_degrees", 20.0),
            max_wrist_degrees=self.sanity.get("max_wrist_residual_degrees", 30.0),
            max_finger_degrees=self.sanity.get("max_finger_residual_degrees", 25.0),
        )

    def optimize(self, state: SequenceState, inputs: FactorInputs) -> OptimizationResult:
        validate_factor_inputs(inputs)
        records: list[StageRecord] = []
        fallbacks: list[dict[str, Any]] = []
        with torch.no_grad():
            zero_output = self.model(state)
            valid, reason = self._sanity(state, zero_output)
            if not valid:
                raise ValueError(f"BA-0 initialization failed: {reason}")

        for stage in self.stages:
            name = str(stage["name"])
            trainable = state.set_trainable(list(stage["variables"]))
            optimizer_name = str(stage.get("optimizer", "adam")).lower()
            if optimizer_name != "adam":
                raise ValueError(
                    f"unsupported staged optimizer {optimizer_name}; LBFGS is a separate finish step"
                )
            optimizer = torch.optim.Adam(trainable, lr=float(stage["learning_rate"]))
            stage_start = state.snapshot()
            best_snapshot = stage_start
            best_loss = float("inf")
            best_output: SMPLXOutput | None = None
            invalid_reason: str | None = None
            patience = int(stage.get("early_stop_patience", 0))
            relative_tolerance = float(stage.get("relative_tolerance", 0.0))
            stale_iterations = 0
            with torch.no_grad():
                stage_start_factors = compute_factors(zero_output, state, inputs, self.scales)
            starting_reprojection = stage_start_factors.get("reprojection_2d")
            for iteration in range(int(stage["iterations"])):
                optimizer.zero_grad(set_to_none=True)
                output = self.model(state)
                factors = compute_factors(output, state, inputs, self.scales)
                loss = normalized_weighted_sum(factors, stage["weights"])
                if not torch.isfinite(loss):
                    invalid_reason = "non_finite_loss"
                    break
                valid, reason = self._sanity(state, output)
                if valid and starting_reprojection is not None and "reprojection_2d" in factors:
                    ratio = factors["reprojection_2d"] / starting_reprojection.clamp_min(1e-12)
                    if ratio > float(self.sanity.get("max_reprojection_ratio", float("inf"))):
                        valid, reason = False, "reprojection_regression"
                value = float(loss.detach())
                previous_best = best_loss
                accepted = valid and value < best_loss
                if accepted:
                    # The objective, output and snapshot must describe the same
                    # pre-update state; otherwise a final Adam step can be
                    # accepted using stale sanity checks.
                    best_loss = value
                    best_snapshot = state.snapshot()
                    best_output = output
                meaningful = accepted and (
                    previous_best == float("inf")
                    or (previous_best - value) / max(abs(previous_best), 1e-12) > relative_tolerance
                )
                stale_iterations = 0 if meaningful else stale_iterations + 1
                if iteration == 0 or iteration == int(stage["iterations"]) - 1 or accepted:
                    records.append(
                        StageRecord(
                            stage=name,
                            iteration=iteration,
                            loss=value,
                            factors={key: float(item.detach()) for key, item in factors.items()},
                            accepted=accepted,
                            fallback_reason=reason,
                        )
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip_norm)
                optimizer.step()
                if patience and stale_iterations >= patience:
                    records.append(
                        StageRecord(
                            stage=name,
                            iteration=iteration,
                            loss=value,
                            factors={key: float(item.detach()) for key, item in factors.items()},
                            accepted=accepted,
                            fallback_reason="early_stop",
                        )
                    )
                    break
            if best_output is None:
                state.restore(stage_start)
                fallbacks.append(
                    {"stage": name, "reason": invalid_reason or "no_valid_improvement"}
                )
            else:
                state.restore(best_snapshot)
            with torch.no_grad():
                zero_output = self.model(state)

        with torch.no_grad():
            final_output = self.model(state)
            valid, reason = self._sanity(state, final_output)
            if not valid:
                raise RuntimeError(f"final BA state failed sanity checks: {reason}")
        return OptimizationResult(
            state=state, output=final_output, records=records, fallback_events=fallbacks
        )
