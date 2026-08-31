"""Frozen per-frame camera contract and differentiable pinhole projection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from dcg_sign4d.diffusion.state_codec import TrajectoryState


@dataclass(frozen=True)
class CameraTrajectory:
    intrinsics: Tensor  # [B,T,3,3]
    world_to_camera: Tensor  # [B,T,4,4]
    image_size_wh: Tensor  # [B,T,2]
    valid_mask: Tensor  # [B,T]
    coordinate_convention: str

    def validate(self) -> CameraTrajectory:
        if self.intrinsics.ndim != 4 or self.intrinsics.shape[-2:] != (3, 3):
            raise ValueError("camera intrinsics must be [B,T,3,3]")
        batch, time = self.intrinsics.shape[:2]
        if self.world_to_camera.shape != (batch, time, 4, 4):
            raise ValueError("world_to_camera must be [B,T,4,4]")
        if self.image_size_wh.shape != (batch, time, 2):
            raise ValueError("image_size_wh must be [B,T,2]")
        if self.valid_mask.shape != (batch, time) or self.valid_mask.dtype != torch.bool:
            raise ValueError("camera valid_mask must be bool [B,T]")
        for value in (self.intrinsics, self.world_to_camera, self.image_size_wh):
            if not torch.isfinite(value[self.valid_mask]).all():
                raise ValueError("valid camera parameters contain NaN/Inf")
        if bool((self.image_size_wh[self.valid_mask] <= 0).any()):
            raise ValueError("camera image size must be positive")
        if not self.coordinate_convention:
            raise ValueError("camera coordinate convention is required")
        return self

    def project(self, points: Tensor, *, minimum_depth: float = 1e-4) -> Tensor:
        """Project [B,T,J,3] world/canonical points to image pixels."""

        self.validate()
        if points.ndim != 4 or points.shape[:2] != self.valid_mask.shape or points.shape[-1] != 3:
            raise ValueError("points must be [B,T,J,3] and match camera time")
        ones = torch.ones_like(points[..., :1])
        homogeneous = torch.cat((points, ones), -1)
        camera = torch.einsum("btij,btkj->btki", self.world_to_camera.to(points), homogeneous)
        depth = camera[..., 2:3]
        safe_depth = depth.clamp_min(minimum_depth)
        image_homogeneous = torch.einsum(
            "btij,btkj->btki", self.intrinsics.to(points), camera[..., :3]
        )
        projected = image_homogeneous[..., :2] / safe_depth
        return projected


class StateJointProjector(nn.Module):
    """SMPL-X state-to-2D joint projection with an explicit frozen joint map."""

    def __init__(
        self,
        body_model: nn.Module,
        camera: CameraTrajectory,
        joint_indices: Tensor,
    ) -> None:
        super().__init__()
        if joint_indices.ndim != 1 or joint_indices.dtype != torch.long:
            raise ValueError("joint_indices must be long [J_observation]")
        self.body_model = body_model
        self.camera = camera.validate()
        self.register_buffer("joint_indices", joint_indices)

    def forward(self, state: TrajectoryState) -> Tensor:
        output = self.body_model(state)
        if int(self.joint_indices.max()) >= output.joints.shape[2]:
            raise ValueError("joint map exceeds SMPL-X output topology")
        return self.camera.project(output.joints[:, :, self.joint_indices])


class StateJointDepthDifference(nn.Module):
    """Camera-space depth differences for a frozen list of joint pairs."""

    def __init__(
        self, body_model: nn.Module, camera: CameraTrajectory, joint_pairs: Tensor
    ) -> None:
        super().__init__()
        if joint_pairs.ndim != 2 or joint_pairs.shape[-1] != 2 or joint_pairs.dtype != torch.long:
            raise ValueError("joint_pairs must be long [D,2]")
        self.body_model = body_model
        self.camera = camera.validate()
        self.register_buffer("joint_pairs", joint_pairs)

    def forward(self, state: TrajectoryState) -> Tensor:
        joints = self.body_model(state).joints
        if int(self.joint_pairs.max()) >= joints.shape[2]:
            raise ValueError("depth joint map exceeds SMPL-X output topology")
        ones = torch.ones_like(joints[..., :1])
        homogeneous = torch.cat((joints, ones), -1)
        camera_joints = torch.einsum(
            "btij,btkj->btki", self.camera.world_to_camera.to(joints), homogeneous
        )
        return (
            camera_joints[..., self.joint_pairs[:, 0], 2]
            - camera_joints[..., self.joint_pairs[:, 1], 2]
        )


class StatePartMaskRenderer(nn.Module):
    """Differentiable soft part masks from frozen SMPL-X vertex groups.

    This point-splat renderer is intentionally simple and deterministic. It
    provides the proposal's optional part-mask cue without introducing a
    second camera convention or an unaudited rasterizer dependency.
    """

    def __init__(
        self,
        body_model: nn.Module,
        camera: CameraTrajectory,
        vertex_groups: tuple[Tensor, ...],
        output_size_hw: tuple[int, int],
        *,
        sigma_px: float,
    ) -> None:
        super().__init__()
        if not vertex_groups or any(
            group.ndim != 1 or group.dtype != torch.long for group in vertex_groups
        ):
            raise ValueError("vertex_groups must contain non-empty long index vectors")
        if any(group.numel() == 0 for group in vertex_groups):
            raise ValueError("part-mask vertex groups cannot be empty")
        if min(output_size_hw) < 1 or sigma_px <= 0:
            raise ValueError("mask size and splat sigma must be positive")
        self.body_model = body_model
        self.camera = camera.validate()
        self.output_size_hw = output_size_hw
        self.sigma_px = sigma_px
        for index, group in enumerate(vertex_groups):
            self.register_buffer(f"vertex_group_{index}", group)
        self.group_count = len(vertex_groups)

    def forward(self, state: TrajectoryState) -> Tensor:
        vertices = self.body_model(state).vertices
        height, width = self.output_size_hw
        yy, xx = torch.meshgrid(
            torch.arange(height, device=vertices.device, dtype=vertices.dtype),
            torch.arange(width, device=vertices.device, dtype=vertices.dtype),
            indexing="ij",
        )
        masks = []
        for index in range(self.group_count):
            group = getattr(self, f"vertex_group_{index}")
            if int(group.max()) >= vertices.shape[2]:
                raise ValueError("part-mask vertex map exceeds SMPL-X output topology")
            projected = self.camera.project(vertices[:, :, group])
            image_size = self.camera.image_size_wh.to(vertices)
            x = projected[..., 0] * width / image_size[..., None, 0]
            y = projected[..., 1] * height / image_size[..., None, 1]
            distance_sq = (x[..., None, None] - xx).square() + (y[..., None, None] - yy).square()
            occupancy = torch.exp(-0.5 * distance_sq / (self.sigma_px**2))
            masks.append(1 - torch.prod(1 - occupancy.clamp(0, 1), dim=2))
        return torch.stack(masks, dim=2)
