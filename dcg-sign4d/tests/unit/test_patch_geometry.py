from pathlib import Path

import pytest
import torch

from dcg_sign4d.contact.ontology import EventState
from dcg_sign4d.geometry.contact_geometry import ContactGeometry
from dcg_sign4d.geometry.patch_map import PatchMap
from dcg_sign4d.synthetic import make_graph

ASSET = Path(__file__).resolve().parents[2] / "assets/patch_maps/synthetic_smoke.yaml"


def vertices():
    value = torch.zeros(1, 2, 8, 3)
    value[:, :, 0:2, 0] = torch.tensor([0.0, 0.01])
    value[:, :, 2:4, 0] = torch.tensor([0.02, 0.03])
    value[:, :, 4:6, 1] = torch.tensor([0.10, 0.11])
    value[:, :, 6:8, 2] = torch.tensor([0.20, 0.21])
    return value


def test_patch_map_is_versioned_and_complete():
    patch_map = PatchMap.load(ASSET)
    assert patch_map.development_only
    assert len(patch_map.content_hash) == 64


def test_distance_direction_rigid_invariance_and_gradient():
    patch_map = PatchMap.load(ASSET)
    geometry = ContactGeometry(patch_map, fps=30, allow_missing_penetration=True)
    value = vertices().requires_grad_()
    first = geometry.features(value).distance
    translated = value.detach() + torch.tensor([2.0, -1.0, 4.0])
    assert torch.allclose(first, geometry.features(translated).distance, atol=1e-6)
    moved = value.detach().clone()
    moved[:, :, 2:4, 0] += 0.1
    assert bool((geometry.features(moved).distance[..., 0] > first[..., 0]).all())
    first.sum().backward()
    assert torch.isfinite(value.grad).all()
    assert value.grad.abs().sum() > 0


def test_velocity_is_per_second_and_penetration_is_not_proximity():
    patch_map = PatchMap.load(ASSET)
    moving = vertices()
    moving[:, 1, 0:2, 0] += 0.03
    at_30 = ContactGeometry(patch_map, fps=30, allow_missing_penetration=True).features(moving)
    at_15 = ContactGeometry(patch_map, fps=15, allow_missing_penetration=True).features(moving)
    assert torch.allclose(at_30.relative_speed[:, 1], 2 * at_15.relative_speed[:, 1])
    assert at_30.penetration_depth.sum() == 0
    assert at_30.penetration_available is False
    signed = torch.zeros(1, 2, 2)
    signed[..., 0] = -0.005
    area = torch.zeros_like(signed)
    area[..., 0] = 0.25
    penetrated = ContactGeometry(patch_map, fps=30).features(
        moving, signed_edge_distance=signed, penetration_area=area
    )
    assert torch.allclose(penetrated.penetration_depth[..., 0], torch.full((1, 2), 0.005))
    assert torch.allclose(penetrated.penetration_area[..., 0], torch.full((1, 2), 0.25))
    assert penetrated.penetration_available is True


def test_signed_penetration_requires_separate_area_report():
    patch_map = PatchMap.load(ASSET)
    signed = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="separately reported area"):
        ContactGeometry(patch_map, fps=30).features(vertices(), signed_edge_distance=signed)


def test_missing_penetration_is_fail_closed_by_default():
    patch_map = PatchMap.load(ASSET)
    geometry = ContactGeometry(patch_map, fps=30)
    with pytest.raises(RuntimeError, match="signed penetration geometry is required"):
        geometry.features(vertices())


def test_geometry_distance_finite_difference_matches_autograd():
    patch_map = PatchMap.load(ASSET)
    geometry = ContactGeometry(patch_map, fps=30, allow_missing_penetration=True)
    displacement = torch.tensor(0.04, requires_grad=True)

    def objective(value):
        mesh = vertices()
        offset = torch.zeros_like(mesh)
        offset[:, :, 2:4, 0] = value
        return geometry.features(mesh + offset).distance[..., 0].mean()

    objective(displacement).backward()
    epsilon = 1e-4
    finite = objective(displacement.detach() + epsilon) - objective(displacement.detach() - epsilon)
    finite = finite / (2 * epsilon)
    assert torch.allclose(displacement.grad, finite, atol=2e-3, rtol=2e-3)


def test_normal_hold_and_penetration_terms_are_reported_separately():
    patch_map = PatchMap.load(ASSET)
    backend = ContactGeometry(patch_map, fps=30)
    mesh = vertices()
    normals_opposed = torch.ones_like(mesh)
    normals_opposed[..., 0:2, :] = torch.tensor([1.0, 0.0, 0.0])
    normals_opposed[..., 2:4, :] = torch.tensor([-1.0, 0.0, 0.0])
    signed = torch.zeros(1, 2, 2)
    signed[..., 0] = -0.005
    area = torch.zeros_like(signed)
    area[..., 0] = 0.25
    output = backend.features(
        mesh,
        vertex_normals=normals_opposed,
        signed_edge_distance=signed,
        penetration_area=area,
    )
    graph = make_graph(time=2, edges=2)
    graph.event_state[..., 0] = int(EventState.HOLD)
    graph.event_probability[..., 0, :] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    terms = backend.energy(output, graph)
    assert terms["positive_normal"] == 0
    assert terms["penetration_depth"] > 0
    assert terms["penetration_area"] > terms["penetration_depth"]
    assert torch.allclose(
        terms["penetration"], terms["penetration_depth"] + terms["penetration_area"]
    )
