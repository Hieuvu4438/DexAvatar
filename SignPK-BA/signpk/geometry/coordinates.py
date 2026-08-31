from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class CameraParameters:
    focal_length: Tensor
    principal_point: Tensor
    translation: Tensor | None = None


class CoordinateAdapter:
    """Explicit native-to-canonical point and rotation transforms.

    Matrices map native column vectors into canonical camera coordinates.
    Rotation bases are changed as ``Q R Q^T``.
    """

    def __init__(self, transforms: Mapping[str, Tensor | list[list[float]]]):
        self.transforms = {
            name: torch.as_tensor(value, dtype=torch.float32) for name, value in transforms.items()
        }
        for name, transform in self.transforms.items():
            if transform.shape != (3, 3):
                raise ValueError(f"coordinate transform {name} is not 3x3")
            if not torch.allclose(transform.transpose(0, 1) @ transform, torch.eye(3), atol=1e-5):
                raise ValueError(f"coordinate transform {name} is not orthogonal")

    def _transform(self, source: str, reference: Tensor) -> Tensor:
        if source not in self.transforms:
            raise KeyError(f"coordinate source {source!r} is not registered")
        return self.transforms[source].to(device=reference.device, dtype=reference.dtype)

    def points_to_canonical(self, xyz: Tensor, source: str) -> Tensor:
        transform = self._transform(source, xyz)
        return torch.einsum("ij,...j->...i", transform, xyz)

    def rotations_to_canonical(self, rotation: Tensor, source: str) -> Tensor:
        transform = self._transform(source, rotation)
        return transform @ rotation @ transform.transpose(-1, -2)

    @staticmethod
    def project(xyz_canonical: Tensor, camera: CameraParameters, eps: float = 1e-6) -> Tensor:
        points = xyz_canonical
        if camera.translation is not None:
            points = points + camera.translation[..., None, :]
        xy = points[..., :2] / points[..., 2:].clamp_min(eps)
        return xy * camera.focal_length[..., None, :] + camera.principal_point[..., None, :]

    def validate_reprojection(
        self,
        xyz: Tensor,
        uv: Tensor,
        camera: CameraParameters,
        tolerance_px: float,
        confidence: Tensor | None = None,
    ) -> Tensor:
        error = torch.linalg.vector_norm(self.project(xyz, camera) - uv, dim=-1)
        if confidence is not None:
            valid = confidence > 0
            mean = (error * confidence).sum(-1) / (confidence * valid).sum(-1).clamp_min(1e-8)
        else:
            mean = error.mean(-1)
        if torch.any(mean > tolerance_px):
            raise ValueError(f"reprojection error exceeds {tolerance_px}px: {mean.max().item():.2f}px")
        return mean


def infer_scale_from_bone(points: Tensor, first: int, second: int, expected_m: float) -> Tensor:
    measured = torch.linalg.vector_norm(points[..., first, :] - points[..., second, :], dim=-1)
    return expected_m / measured.clamp_min(1e-8)


def validate_human_scale(
    joints: Tensor,
    *,
    shoulder_pair: tuple[int, int] | None = None,
    wrist_middle_pair: tuple[int, int] | None = None,
) -> None:
    if shoulder_pair is not None:
        width = torch.linalg.vector_norm(joints[..., shoulder_pair[0], :] - joints[..., shoulder_pair[1], :], dim=-1)
        if torch.any((width < 0.15) | (width > 0.75)):
            raise ValueError("implausible shoulder width; check units/coordinates")
    if wrist_middle_pair is not None:
        length = torch.linalg.vector_norm(joints[..., wrist_middle_pair[0], :] - joints[..., wrist_middle_pair[1], :], dim=-1)
        if torch.any((length < 0.03) | (length > 0.20)):
            raise ValueError("implausible wrist-to-middle-MCP length; check units")

