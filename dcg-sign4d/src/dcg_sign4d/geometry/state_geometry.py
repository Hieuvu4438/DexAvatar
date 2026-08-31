"""Differentiable TrajectoryState-to-contact-geometry runtime assembly."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from dcg_sign4d.diffusion.state_codec import TrajectoryState

from .contact_geometry import ContactGeometry, GeometryOutput
from .mesh import vertex_normals
from .penetration import PenetrationOutput


class StateContactGeometry(nn.Module):
    """Join SMPL-X forward kinematics, mesh normals and signed selfcontact.

    This is the production callable consumed by alternating inference and
    contact guidance.  Shape remains live through the supplied SMPL-X adapter,
    so changing clip-shared ``beta`` can change every geometry feature.
    """

    def __init__(
        self,
        body_model: nn.Module,
        geometry: ContactGeometry,
        faces: Tensor,
        penetration: nn.Module | None,
        patch_reliability: Callable[[TrajectoryState], Tensor | None] | None = None,
    ) -> None:
        super().__init__()
        if faces.ndim != 2 or faces.shape[-1] != 3 or faces.dtype != torch.long:
            raise ValueError("faces must be long [F,3]")
        self.body_model = body_model
        self.geometry = geometry
        self.penetration = penetration
        self.patch_reliability = patch_reliability
        self.register_buffer("faces", faces)

    def forward(self, state: TrajectoryState) -> GeometryOutput:
        body = self.body_model(state)
        vertices = body.vertices
        normals = vertex_normals(vertices, self.faces)
        penetration_output: PenetrationOutput | None = None
        if self.penetration is not None:
            penetration_output = self.penetration(vertices, self.geometry.patch_map)
        reliability = self.patch_reliability(state) if self.patch_reliability else None
        return self.geometry.features(
            vertices,
            vertex_normals=normals,
            patch_reliability=reliability,
            signed_edge_distance=(
                penetration_output.signed_edge_distance if penetration_output else None
            ),
            penetration_area=(penetration_output.area if penetration_output else None),
        )
