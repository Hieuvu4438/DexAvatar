"""Positive/negative/penetration guidance from differentiable geometry."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.diffusion.state_codec import TrajectoryState
from dcg_sign4d.geometry.contact_geometry import ContactGeometry, GeometryOutput
from dcg_sign4d.observations.schema import ObservationBatch


class ContactGuidance:
    name = "contact"

    def __init__(
        self,
        geometry: ContactGeometry,
        geometry_from_state: Callable[[TrajectoryState], GeometryOutput],
    ) -> None:
        self.geometry = geometry
        self.geometry_from_state = geometry_from_state

    def loss(
        self,
        clean_state: TrajectoryState,
        observations: ObservationBatch,
        graph: ContactGraphBatch,
    ) -> Tensor:
        del observations
        return self.geometry.energy(self.geometry_from_state(clean_state), graph)["total"]
