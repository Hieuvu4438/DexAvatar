"""Audited boundary around the licensed official selfcontact penetration code."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from dcg_sign4d.utils.hashing import file_sha256

from .mesh import vertex_areas
from .patch_map import PatchMap


@dataclass(frozen=True)
class PenetrationOutput:
    """Per-edge penetration statistics.

    ``depth`` remains differentiable through the closest-point distances.  The
    inside/outside decision is discrete in the upstream selfcontact package and
    is therefore intentionally treated as a detached mask.
    """

    depth: Tensor
    area: Tensor
    signed_edge_distance: Tensor


def chunked_winding_exterior(
    winding_numbers: object,
    vertices: Tensor,
    triangles: Tensor,
    *,
    query_chunk_size: int,
) -> Tensor:
    """Evaluate the official winding-number function in exact query-point chunks."""

    if query_chunk_size < 1:
        raise ValueError("winding query chunk size must be positive")
    if vertices.ndim != 3 or triangles.ndim != 4 or vertices.shape[0] != triangles.shape[0]:
        raise ValueError("vertices/triangles must be [B,V,3] and [B,F,3,3]")
    exterior = torch.zeros(
        vertices.shape[:2], dtype=torch.bool, device=vertices.device
    )
    for start in range(0, vertices.shape[1], query_chunk_size):
        end = min(start + query_chunk_size, vertices.shape[1])
        exterior[:, start:end] = winding_numbers(  # type: ignore[operator]
            vertices[:, start:end], triangles
        ).le(0.99)
    return exterior


def chunked_official_minimum_distance(
    vertices: Tensor, geomask: Tensor, *, target_chunk_size: int
) -> Tensor:
    """Memory-bounded equivalent of upstream ``segment_vertices_scopti`` distance."""

    if vertices.ndim != 3 or vertices.shape[0] != 1 or vertices.shape[-1] != 3:
        raise ValueError("official deterministic distance requires vertices [1,V,3]")
    vertex_count = vertices.shape[1]
    if geomask.shape != (vertex_count, vertex_count) or geomask.dtype != torch.bool:
        raise ValueError("geomask must be boolean [V,V]")
    if target_chunk_size < 1:
        raise ValueError("distance target chunk size must be positive")
    flat = vertices[0]
    nearest_indices = []
    # Upstream also selects the nearest index under no_grad and only differentiates
    # the gathered winning distance. Chunking target columns is algebraically identical.
    with torch.no_grad():
        for start in range(0, vertex_count, target_chunk_size):
            end = min(start + target_chunk_size, vertex_count)
            pairwise = torch.norm(
                flat[:, None, :] - flat[None, start:end, :], dim=-1
            )
            pairwise.masked_fill_(~geomask[:, start:end], float("inf"))
            nearest_indices.append(pairwise.argmin(dim=0))
    nearest = torch.cat(nearest_indices)
    return torch.norm(flat - flat[nearest], dim=-1, keepdim=False)[None]


def _verify_registry(root: Path, registry_path: Path, expected_sha256: str) -> None:
    if file_sha256(registry_path) != expected_sha256:
        raise ValueError("selfcontact essentials registry hash mismatch")
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "dcg_selfcontact_essentials_v1":
        raise ValueError("unknown selfcontact essentials registry schema")
    if payload.get("scientific_status") != "FROZEN":
        raise PermissionError("selfcontact essentials registry is not scientifically frozen")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("selfcontact essentials registry contains no files")
    for row in files:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("selfcontact registry paths must stay below the essentials root")
        path = root / relative
        if not path.is_file() or file_sha256(path) != row.get("sha256"):
            raise ValueError(f"missing/mismatched selfcontact essential: {relative}")


class OfficialSelfContactPenetration(nn.Module):
    """Convert official selfcontact segmentation into patch-edge depth/area.

    The official implementation supplies differentiable closest-point distance
    and a detached generalized-winding-number inside mask.  This adapter only
    aggregates those outputs over the author-frozen patch map; it does not
    replace them with proximity heuristics.
    """

    def __init__(
        self,
        essentials_root: str | Path,
        *,
        source_root: str | Path,
        expected_commit: str,
        registry: str | Path,
        expected_registry_sha256: str,
        trusted_licensed_assets: bool,
        test_segments: bool = True,
        winding_query_chunk_size: int = 1000,
        distance_target_chunk_size: int = 1000,
    ) -> None:
        super().__init__()
        if not trusted_licensed_assets:
            raise PermissionError("licensed selfcontact essentials require explicit trust")
        root = Path(essentials_root)
        _verify_registry(root, Path(registry), expected_registry_sha256)
        source = Path(source_root).resolve()
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if commit != expected_commit:
            raise ValueError("selfcontact source commit mismatch")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        try:
            module = importlib.import_module("selfcontact")
        except ImportError as exc:
            raise RuntimeError(
                "install the pinned third_party/selfcontact package before reconstruction"
            ) from exc
        self.backend = module.SelfContact(
            essentials_folder=str(root),
            model_type="smplx",
            test_segments=test_segments,
            compute_hd=False,
        )
        mesh_module = importlib.import_module("selfcontact.utils.mesh")
        self._official_winding_numbers = mesh_module.winding_numbers
        self._official_get_intersection_mask = self.backend.get_intersection_mask
        self.winding_query_chunk_size = winding_query_chunk_size
        self.distance_target_chunk_size = distance_target_chunk_size
        self.backend.get_intersection_mask = self._memory_bounded_intersection_mask
        self.essentials_registry_sha256 = expected_registry_sha256
        self.test_segments = test_segments

    def _memory_bounded_intersection_mask(
        self, vertices: Tensor, triangles: Tensor, test_segments: bool = True
    ) -> Tensor:
        if test_segments:
            return self._official_get_intersection_mask(vertices, triangles, test_segments)
        return chunked_winding_exterior(
            self._official_winding_numbers,
            vertices,
            triangles,
            query_chunk_size=self.winding_query_chunk_size,
        )

    @staticmethod
    def aggregate(
        minimum_distance: Tensor,
        exterior: Tensor,
        patch_map: PatchMap,
        vertex_area: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Aggregate official per-vertex outputs to depth and affected area fraction."""

        if minimum_distance.ndim != 2 or exterior.shape != minimum_distance.shape:
            raise ValueError("minimum_distance/exterior must both be [N,V]")
        if exterior.dtype != torch.bool:
            raise ValueError("exterior must be boolean")
        if vertex_area.shape != minimum_distance.shape or bool((vertex_area < 0).any()):
            raise ValueError("vertex_area must be nonnegative [N,V]")
        if minimum_distance.shape[1] != patch_map.mesh_vertex_count:
            raise ValueError("selfcontact vertex topology does not match the patch map")
        inside = ~exterior
        depths = []
        areas = []
        for source, target in patch_map.admissible_edges:
            indices = torch.tensor(
                (*patch_map.patches[source], *patch_map.patches[target]),
                dtype=torch.long,
                device=minimum_distance.device,
            ).unique()
            edge_inside = inside[:, indices]
            count = edge_inside.sum(-1)
            depth = (minimum_distance[:, indices] * edge_inside).sum(-1)
            depth = depth / count.clamp_min(1)
            depth = torch.where(count > 0, depth, torch.zeros_like(depth))
            depths.append(depth)
            areas.append((vertex_area[:, indices] * edge_inside).sum(-1))
        return torch.stack(depths, -1), torch.stack(areas, -1)

    def forward(self, vertices: Tensor, patch_map: PatchMap) -> PenetrationOutput:
        if vertices.ndim != 4 or vertices.shape[-1] != 3:
            raise ValueError("vertices must be [B,T,V,3]")
        if vertices.shape[2] != patch_map.mesh_vertex_count:
            raise ValueError("vertices do not match patch-map topology")
        batch, time, vertex_count, _ = vertices.shape
        flattened = vertices.reshape(batch * time, vertex_count, 3)
        distances = []
        exteriors = []
        # The deterministic upstream routine is explicitly batch-size one.
        for frame in flattened:
            current = frame[None]
            with torch.no_grad():
                triangles = self.backend.triangles(current.detach())
                exterior = self.backend.get_intersection_mask(
                    current.detach(), triangles.detach(), self.test_segments
                )
            minimum = chunked_official_minimum_distance(
                current,
                self.backend.geomask,
                target_chunk_size=self.distance_target_chunk_size,
            )
            distances.append(minimum[0])
            exteriors.append(exterior[0])
        minimum_distance = torch.stack(distances)
        exterior = torch.stack(exteriors)
        area_per_vertex = vertex_areas(flattened, self.backend.faces.to(flattened.device))
        depth, area = self.aggregate(minimum_distance, exterior, patch_map, area_per_vertex)
        depth = depth.reshape(batch, time, -1)
        area = area.reshape(batch, time, -1)
        # ContactGeometry consumes the negative part as penetration depth.
        signed = torch.where(area > 0, -depth, torch.zeros_like(depth))
        return PenetrationOutput(depth=depth, area=area, signed_edge_distance=signed)
