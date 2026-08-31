import torch

from signpk.geometry.rotations import (
    compose_residual,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    so3_distance,
    so3_exp,
    so3_log,
)


def test_so3_roundtrip_and_composition():
    vectors = torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.05, 0.2]], dtype=torch.float64)
    matrices = so3_exp(vectors)
    torch.testing.assert_close(so3_log(matrices), vectors, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(compose_residual(matrices, torch.zeros_like(vectors)), matrices)
    assert torch.all(so3_distance(matrices, matrices) < 1e-5)


def test_rotation_6d_roundtrip_and_gradient():
    vector = torch.tensor([[0.2, -0.1, 0.05]], requires_grad=True)
    matrix = so3_exp(vector)
    recovered = rotation_6d_to_matrix(matrix_to_rotation_6d(matrix))
    torch.testing.assert_close(recovered, matrix, atol=1e-6, rtol=1e-6)
    recovered.sum().backward()
    assert torch.isfinite(vector.grad).all()

