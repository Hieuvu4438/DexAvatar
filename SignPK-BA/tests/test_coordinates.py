import torch

from signpk.geometry.coordinates import CameraParameters, CoordinateAdapter


def test_point_rotation_and_projection_conventions():
    rotation_x_180 = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    adapter = CoordinateAdapter({"native": rotation_x_180})
    point = torch.tensor([[1.0, 2.0, 4.0]])
    torch.testing.assert_close(adapter.points_to_canonical(point, "native"), torch.tensor([[1.0, -2.0, -4.0]]))
    rotation = torch.eye(3)[None]
    converted = adapter.rotations_to_canonical(rotation, "native")
    torch.testing.assert_close(converted, rotation)
    camera = CameraParameters(torch.tensor([100.0, 100.0]), torch.tensor([50.0, 40.0]))
    uv = adapter.project(torch.tensor([[[1.0, 2.0, 10.0]]]), camera)
    torch.testing.assert_close(uv, torch.tensor([[[60.0, 60.0]]]))

