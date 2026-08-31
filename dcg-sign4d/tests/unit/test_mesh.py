import pytest
import torch

from dcg_sign4d.geometry.mesh import vertex_normals


def test_vertex_normals_are_unit_and_differentiable():
    vertices = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        requires_grad=True,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    normals = vertex_normals(vertices, faces)
    assert torch.allclose(normals, torch.tensor([[[0.0, 0.0, 1.0]] * 3]))
    normals.square().sum().backward()
    assert vertices.grad is not None


def test_vertex_normals_reject_bad_topology():
    vertices = torch.zeros(3, 3)
    with pytest.raises(ValueError):
        vertex_normals(vertices, torch.tensor([[0, 1, 3]], dtype=torch.long))
