"""Guidance interface evaluated only on decoded clean-state estimates."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.diffusion.state_codec import TrajectoryState
from dcg_sign4d.observations.schema import ObservationBatch


class GuidanceTerm(Protocol):
    name: str

    def loss(
        self,
        clean_state: TrajectoryState,
        observations: ObservationBatch,
        graph: ContactGraphBatch,
    ) -> Tensor: ...
