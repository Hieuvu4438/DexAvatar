"""Differentiable patch geometry adapted from TUCH/selfcontact concepts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from dcg_sign4d.contact.ontology import ContactGraphBatch, EventState

from .patch_map import PatchMap


@dataclass(frozen=True)
class GeometryOutput:
    features: Tensor
    distance: Tensor
    normal_compatibility: Tensor
    relative_speed: Tensor
    penetration_depth: Tensor
    penetration_area: Tensor
    reliability: Tensor
    normal_available: bool
    penetration_available: bool


class ContactGeometry(nn.Module):
    """Patch geometry with explicit signed-penetration input.

    Pair distances and velocities are native differentiable PyTorch. Production
    signed penetration must come from the licensed audited selfcontact adapter;
    it is never inferred from mere proximity.
    """

    def __init__(
        self,
        patch_map: PatchMap,
        fps: float,
        separation_margin: float = 0.03,
        sigma_distance: float = 0.01,
        sigma_normal: float = 0.25,
        sigma_velocity: float = 0.1,
        normal_weight: float = 1.0,
        hold_velocity_weight: float = 1.0,
        penetration_area_weight: float = 1.0,
        *,
        allow_missing_penetration: bool = False,
    ):
        super().__init__()
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.patch_map = patch_map
        self.fps = fps
        self.separation_margin = separation_margin
        for name, value in (
            ("sigma_distance", sigma_distance),
            ("sigma_normal", sigma_normal),
            ("sigma_velocity", sigma_velocity),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("normal_weight", normal_weight),
            ("hold_velocity_weight", hold_velocity_weight),
            ("penetration_area_weight", penetration_area_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        self.sigma_distance = sigma_distance
        self.sigma_normal = sigma_normal
        self.sigma_velocity = sigma_velocity
        self.normal_weight = normal_weight
        self.hold_velocity_weight = hold_velocity_weight
        self.penetration_area_weight = penetration_area_weight
        self.allow_missing_penetration = allow_missing_penetration

    @staticmethod
    def robust_symmetric_distance(first: Tensor, second: Tensor) -> Tensor:
        pairwise = torch.cdist(first, second)
        left = pairwise.amin(dim=-1).mean(dim=-1)
        right = pairwise.amin(dim=-2).mean(dim=-1)
        return 0.5 * (left + right)

    def features(
        self,
        vertices: Tensor,
        *,
        vertex_normals: Tensor | None = None,
        patch_reliability: Tensor | None = None,
        signed_edge_distance: Tensor | None = None,
        penetration_area: Tensor | None = None,
    ) -> GeometryOutput:
        if vertices.ndim != 4 or vertices.shape[-1] != 3:
            raise ValueError("vertices must be [B,T,V,3]")
        if vertices.shape[2] != self.patch_map.mesh_vertex_count:
            raise ValueError("vertex count does not match patch map")
        batch, time = vertices.shape[:2]
        edge_count = len(self.patch_map.admissible_edges)
        distances: list[Tensor] = []
        normal_terms: list[Tensor] = []
        centroid_differences: list[Tensor] = []
        reliability_terms: list[Tensor] = []
        patch_names = tuple(self.patch_map.patches)
        patch_index = {name: idx for idx, name in enumerate(patch_names)}
        for source, target in self.patch_map.admissible_edges:
            source_vertices = vertices[:, :, self.patch_map.patches[source]]
            target_vertices = vertices[:, :, self.patch_map.patches[target]]
            distances.append(self.robust_symmetric_distance(source_vertices, target_vertices))
            centroid_differences.append(source_vertices.mean(-2) - target_vertices.mean(-2))
            if vertex_normals is None:
                normal_terms.append(torch.zeros(batch, time, device=vertices.device))
            else:
                source_normal = functional.normalize(
                    vertex_normals[:, :, self.patch_map.patches[source]].mean(-2), dim=-1
                )
                target_normal = functional.normalize(
                    vertex_normals[:, :, self.patch_map.patches[target]].mean(-2), dim=-1
                )
                normal_terms.append((source_normal * target_normal).sum(-1))
            if patch_reliability is None:
                reliability_terms.append(torch.ones(batch, time, device=vertices.device))
            else:
                source_rel = patch_reliability[:, :, patch_index[source]]
                target_rel = patch_reliability[:, :, patch_index[target]]
                reliability_terms.append(
                    torch.sqrt(source_rel.clamp_min(0) * target_rel.clamp_min(0))
                )
        distance = torch.stack(distances, dim=-1)
        normal = torch.stack(normal_terms, dim=-1)
        relative = torch.stack(centroid_differences, dim=-2)
        delta = torch.zeros_like(relative)
        delta[:, 1:] = (relative[:, 1:] - relative[:, :-1]) * self.fps
        relative_speed = torch.linalg.vector_norm(delta, dim=-1)
        reliability = torch.stack(reliability_terms, dim=-1)
        if signed_edge_distance is None:
            if not self.allow_missing_penetration:
                raise RuntimeError(
                    "signed penetration geometry is required; set "
                    "allow_missing_penetration=True only for development fixtures"
                )
            penetration = torch.zeros_like(distance)
            area = torch.zeros_like(distance)
        else:
            if signed_edge_distance.shape != (batch, time, edge_count):
                raise ValueError("signed_edge_distance must be [B,T,E]")
            if penetration_area is None:
                raise ValueError("signed penetration requires a separately reported area tensor")
            if penetration_area.shape != (batch, time, edge_count):
                raise ValueError("penetration_area must be [B,T,E]")
            if not torch.isfinite(penetration_area).all() or bool((penetration_area < 0).any()):
                raise ValueError("penetration_area must be finite and nonnegative")
            penetration = (-signed_edge_distance).clamp_min(0)
            area = penetration_area
        penetration_feature = penetration + self.penetration_area_weight * area
        features = torch.stack(
            (distance, normal, relative_speed, penetration_feature, reliability), dim=-1
        )
        return GeometryOutput(
            features,
            distance,
            normal,
            relative_speed,
            penetration,
            area,
            reliability,
            vertex_normals is not None,
            signed_edge_distance is not None,
        )

    def energy(
        self,
        geometry: GeometryOutput,
        graph: ContactGraphBatch,
        *,
        hard_negative: Tensor | None = None,
    ) -> dict[str, Tensor]:
        graph.validate()
        positive_weight = graph.event_probability[..., EventState.ONSET]
        positive_weight = positive_weight + graph.event_probability[..., EventState.HOLD]
        active_weight = positive_weight * geometry.reliability
        distance_loss = functional.smooth_l1_loss(
            geometry.distance / self.sigma_distance,
            torch.zeros_like(geometry.distance),
            reduction="none",
        )
        distance_positive = (active_weight * distance_loss).mean()
        normal_positive = geometry.distance.sum() * 0
        if geometry.normal_available:
            normal_loss = functional.smooth_l1_loss(
                (geometry.normal_compatibility + 1) / self.sigma_normal,
                torch.zeros_like(geometry.normal_compatibility),
                reduction="none",
            )
            normal_positive = self.normal_weight * (active_weight * normal_loss).mean()
        hold = graph.event_state == EventState.HOLD
        velocity_positive = geometry.distance.sum() * 0
        if bool(hold.any()):
            velocity_positive = self.hold_velocity_weight * functional.smooth_l1_loss(
                geometry.relative_speed[hold] / self.sigma_velocity,
                torch.zeros_like(geometry.relative_speed[hold]),
            )
        positive = distance_positive + normal_positive + velocity_positive
        if hard_negative is None or not bool(hard_negative.any()):
            negative = geometry.distance.sum() * 0
        else:
            negative = functional.smooth_l1_loss(
                (self.separation_margin - geometry.distance[hard_negative]).clamp_min(0),
                torch.zeros_like(geometry.distance[hard_negative]),
            )
        penetration_depth = geometry.penetration_depth.mean()
        penetration_area = self.penetration_area_weight * geometry.penetration_area.mean()
        penetration = penetration_depth + penetration_area
        return {
            "positive_distance": distance_positive,
            "positive_normal": normal_positive,
            "positive_velocity": velocity_positive,
            "positive": positive,
            "negative": negative,
            "penetration_depth": penetration_depth,
            "penetration_area": penetration_area,
            "penetration": penetration,
            "total": positive + negative + penetration,
        }
