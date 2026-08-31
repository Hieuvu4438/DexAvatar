"""Gold annotation contract and hysteretic pseudo-event compiler."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .ontology import EventState


@dataclass(frozen=True)
class ContactAnnotation:
    clip_id: str
    edge: tuple[str, str]
    onset_frame: int
    hold_start: int
    hold_end: int
    release_frame: int
    uncertain: bool
    annotator_id: str
    confidence: float
    adjudication_result: str | None = None

    def validate(self) -> ContactAnnotation:
        if not self.clip_id or len(self.edge) != 2 or not all(self.edge):
            raise ValueError("annotation requires clip and patch pair")
        if not (0 <= self.onset_frame <= self.hold_start <= self.hold_end <= self.release_frame):
            raise ValueError("invalid onset/hold/release ordering")
        if not 0 <= self.confidence <= 1:
            raise ValueError("annotator confidence must be in [0,1]")
        if not self.annotator_id:
            raise ValueError("annotator_id is required")
        return self


@dataclass(frozen=True)
class PseudoContactLabels:
    event_state: Tensor
    uncertain_mask: Tensor


class HysteresisPseudoLabeler:
    def __init__(
        self,
        enter_threshold: float,
        exit_threshold: float,
        n_enter: int,
        n_exit: int,
        uncertainty_margin: float,
    ) -> None:
        if not 0 <= enter_threshold < exit_threshold:
            raise ValueError("enter threshold must be below exit threshold")
        if n_enter < 1 or n_exit < 1 or uncertainty_margin < 0:
            raise ValueError("invalid persistence or uncertainty margin")
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.n_enter = n_enter
        self.n_exit = n_exit
        self.uncertainty_margin = uncertainty_margin

    def compile(self, distance: Tensor, valid: Tensor | None = None) -> PseudoContactLabels:
        if distance.ndim != 2:
            raise ValueError("distance must be [T,E]")
        time, edges = distance.shape
        if valid is None:
            valid = torch.ones_like(distance, dtype=torch.bool)
        if valid.shape != distance.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be bool [T,E]")
        if not torch.isfinite(distance[valid]).all():
            raise ValueError("valid distances contain NaN/Inf")
        states = torch.full_like(distance, int(EventState.OFF), dtype=torch.long)
        uncertain = ~valid.clone()
        near_enter = (distance - self.enter_threshold).abs() <= self.uncertainty_margin
        near_exit = (distance - self.exit_threshold).abs() <= self.uncertainty_margin
        uncertain |= (near_enter | near_exit) & valid

        for edge in range(edges):
            active = False
            frame = 0
            while frame < time:
                if not bool(valid[frame, edge]):
                    active = False
                    frame += 1
                    continue
                if not active:
                    run_end = frame
                    while (
                        run_end < time
                        and bool(valid[run_end, edge])
                        and float(distance[run_end, edge]) < self.enter_threshold
                    ):
                        run_end += 1
                    if run_end - frame >= self.n_enter:
                        states[frame, edge] = EventState.ONSET
                        if frame + 1 < run_end:
                            states[frame + 1 : run_end, edge] = EventState.HOLD
                        active = True
                        frame = run_end
                    else:
                        frame += 1
                else:
                    run_end = frame
                    while (
                        run_end < time
                        and bool(valid[run_end, edge])
                        and float(distance[run_end, edge]) > self.exit_threshold
                    ):
                        run_end += 1
                    if run_end - frame >= self.n_exit:
                        states[frame, edge] = EventState.RELEASE
                        active = False
                        frame = run_end
                    else:
                        states[frame, edge] = EventState.HOLD
                        frame += 1
        return PseudoContactLabels(states, uncertain)
