from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.diffusion.state_codec import TrajectoryState


@dataclass(frozen=True)
class RoundResult:
    round_index: int
    trajectory: TrajectoryState
    graph: ContactGraphBatch
    diagnostics: dict[str, Any]
    runtime_objective: dict[str, float]


@dataclass(frozen=True)
class Hypothesis:
    identifier: int
    seed: int
    trajectory: TrajectoryState
    graph: ContactGraphBatch
    score: float
    ranking_terms: dict[str, float]
    diagnostics: dict[str, Any]
    status: str = "ok"
    rounds: tuple[RoundResult, ...] = ()
    retry_count: int = 0
