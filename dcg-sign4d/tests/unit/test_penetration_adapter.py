from __future__ import annotations

import torch

from dcg_sign4d.geometry.patch_map import PatchMap
from dcg_sign4d.geometry.penetration import (
    OfficialSelfContactPenetration,
    chunked_official_minimum_distance,
    chunked_winding_exterior,
)
from dcg_sign4d.utils.hashing import canonical_hash


def _patch_map() -> PatchMap:
    payload = {
        "patch_map_version": "test",
        "smplx_model_version": "test",
        "mesh_vertex_count": 4,
        "patches": {"a": [0, 1], "b": [2, 3]},
        "admissible_edges": [["a", "b"]],
        "excluded_edges": [],
        "development_only": True,
    }
    return PatchMap(
        version="test",
        smplx_model_version="test",
        mesh_vertex_count=4,
        patches={"a": (0, 1), "b": (2, 3)},
        admissible_edges=(("a", "b"),),
        excluded_edges=(),
        development_only=True,
        content_hash=canonical_hash(payload),
    )


def test_official_penetration_aggregation_preserves_depth_gradient():
    distance = torch.tensor([[0.1, 0.2, 0.3, 0.4]], requires_grad=True)
    exterior = torch.tensor([[True, False, True, False]])
    vertex_area = torch.tensor([[0.5, 1.0, 1.5, 2.0]])
    depth, area = OfficialSelfContactPenetration.aggregate(
        distance, exterior, _patch_map(), vertex_area
    )
    assert torch.allclose(depth, torch.tensor([[0.3]]))
    assert torch.allclose(area, torch.tensor([[3.0]]))
    depth.sum().backward()
    assert torch.allclose(distance.grad, torch.tensor([[0.0, 0.5, 0.0, 0.5]]))


def test_winding_query_chunking_preserves_outputs_and_bounds_calls():
    vertices = torch.arange(21, dtype=torch.float32).reshape(1, 7, 3)
    triangles = torch.zeros(1, 2, 3, 3)
    observed_sizes = []

    def winding(points, _triangles):
        observed_sizes.append(points.shape[1])
        return points[..., 0] / 9.0

    exterior = chunked_winding_exterior(
        winding, vertices, triangles, query_chunk_size=3
    )
    expected = winding(vertices, triangles).le(0.99)
    assert torch.equal(exterior, expected)
    assert observed_sizes[:-1] == [3, 3, 1]


def test_chunked_official_distance_matches_upstream_value_and_gradient():
    torch.manual_seed(7)
    vertices = torch.randn(1, 8, 3, requires_grad=True)
    geomask = ~torch.eye(8, dtype=torch.bool)
    actual = chunked_official_minimum_distance(
        vertices, geomask, target_chunk_size=3
    )

    reference_vertices = vertices.detach().clone().requires_grad_(True)
    flat = reference_vertices[0]
    pairwise = torch.norm(flat[:, None, :] - flat[None, :, :], dim=-1)[None]
    with torch.no_grad():
        masked = pairwise.detach().clone()
        masked[:, ~geomask] = float("inf")
        indices = masked.argmin(dim=1)[0]
    expected = pairwise[:, torch.arange(8), indices]
    assert torch.allclose(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    assert torch.allclose(vertices.grad, reference_vertices.grad)
