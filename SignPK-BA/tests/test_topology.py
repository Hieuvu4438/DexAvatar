from pathlib import Path

import numpy as np
import torch

from signpk.geometry.topology import load_obj, load_reference_faces, validate_topology, write_obj
from signpk.optimization.smplx_layer import SMPLXLayer
from signpk.optimization.state import SequenceState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT.parent / "data/evaluation_from_author/data/data/SMPLX_NEUTRAL.npz"


def test_reference_topology_and_obj_order(tmp_path: Path):
    faces = load_reference_faces(MODEL_PATH)
    vertices = np.arange(10475 * 3, dtype=np.float64).reshape(10475, 3) * 1e-6
    validate_topology(vertices, faces, faces)
    path = tmp_path / "mesh.obj"
    write_obj(path, vertices, faces)
    restored_vertices, restored_faces = load_obj(path)
    np.testing.assert_allclose(restored_vertices, vertices, atol=5e-10)
    np.testing.assert_array_equal(restored_faces, faces)


def test_standard_smplx_forward_is_differentiable():
    identity = torch.eye(3)
    state = SequenceState(
        identity.expand(1, 14, 3, 3).clone(),
        identity.expand(1, 21, 3, 3).clone(),
        identity.expand(1, 15, 3, 3).clone(),
        identity.expand(1, 15, 3, 3).clone(),
        torch.zeros(10),
        torch.tensor([[0.0, 0.0, 5.0]]),
    )
    model = SMPLXLayer(MODEL_PATH)
    output = model(state)
    assert output.vertices.shape == (1, 10475, 3)
    output.vertices[:, ::100].square().mean().backward()
    assert state.beta.grad is not None and torch.isfinite(state.beta.grad).all()
    assert state.root_delta.grad is not None and torch.isfinite(state.root_delta.grad).all()

