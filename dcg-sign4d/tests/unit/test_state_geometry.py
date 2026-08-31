from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from dcg_sign4d.geometry.contact_geometry import ContactGeometry
from dcg_sign4d.geometry.patch_map import PatchMap
from dcg_sign4d.geometry.penetration import PenetrationOutput
from dcg_sign4d.geometry.smplx_adapter import SMPLXForwardOutput
from dcg_sign4d.geometry.state_geometry import StateContactGeometry
from dcg_sign4d.synthetic import make_state


class _ShapeBody(nn.Module):
    def forward(self, state):
        batch, time = state.valid_mask.shape
        vertices = torch.zeros(batch, time, 8, 3)
        vertices[:, :, :, 0] = torch.arange(8) * 0.01
        vertices[:, :, 0:2, 1] = state.beta[:, None, :1]
        joints = vertices[:, :, :2]
        return SMPLXForwardOutput(vertices.to(state.beta), joints.to(state.beta))


class _NoPenetration(nn.Module):
    def forward(self, vertices, patch_map):
        shape = (*vertices.shape[:2], len(patch_map.admissible_edges))
        zeros = vertices.new_zeros(shape)
        return PenetrationOutput(zeros, zeros, zeros)


def test_state_geometry_is_shape_aware_and_differentiable():
    patch = PatchMap.load("assets/patch_maps/synthetic_smoke.yaml")
    faces = torch.tensor([[0, 1, 4], [1, 2, 4], [2, 3, 5], [4, 5, 6], [5, 6, 7]], dtype=torch.long)
    adapter = StateContactGeometry(
        _ShapeBody(), ContactGeometry(patch, fps=30), faces, _NoPenetration()
    )
    state = make_state(time=2)
    first = adapter(replace(state, beta=torch.zeros_like(state.beta))).distance
    changed_beta = torch.ones_like(state.beta).requires_grad_()
    second = adapter(replace(state, beta=changed_beta)).distance
    assert not torch.allclose(first, second)
    second.sum().backward()
    assert changed_beta.grad is not None and torch.isfinite(changed_beta.grad).all()
