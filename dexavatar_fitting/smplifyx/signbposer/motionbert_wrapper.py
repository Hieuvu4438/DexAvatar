"""
MotionBERT Prior Wrapper
========================
Wrapper để MotionBERTBodyPrior compatible với interface của SignBPoser.

SignBPoser interface:
    signbposer.decode(pose_embedding, output_type='aa') → body_pose
    signbposer.encode(body_pose) → distribution
    signbposer.sample_poses(N) → body_pose

MotionBERT wrapper cung cấp interface tương tự:
    wrapper.decode(body_pose_or_embedding, output_type='aa') → body_pose
    wrapper.encode(body_pose) → identity (no-op)
    wrapper.sample_poses(N) → random body poses

Trong DexAvatar pipeline:
    - Old: pose_embedding (33-dim) → signbposer.decode() → body_pose (63-dim)
    - New: body_pose (63-dim) = direct optimization variable, wrapper chỉ pass-through
"""

import torch
import torch.nn as nn
import numpy as np

# Import từ signbposer gốc để reuse rotation conversion
from signbposer import SignbPoser


class MotionBERTPriorWrapper(nn.Module):
    """
    Wrapper để MotionBERTBodyPrior compatible với DexAvatar fitting pipeline.

    Thay vì optimize latent space (33-dim), ta optimize body_pose (63-dim) trực tiếp.
    Wrapper này chỉ cung cấp interface compatible, không cần decode thực sự.
    """

    def __init__(self, motionbert_prior, guidance_weight=1.0):
        super().__init__()
        self.prior = motionbert_prior
        self.guidance_weight = guidance_weight
        self._cached_features = None
        self._cached_keypoints = None

    def decode(self, pose_input, output_type='matrot'):
        """
        Compatible interface với SignBPoser.decode().

        Args:
            pose_input: body_pose (1, 63) - direct optimization variable
            output_type: 'aa' hoặc 'matrot'

        Returns:
            body_pose: (1, 1, 21, 9) nếu matrot, (1, 1, 21, 3) nếu aa
        """
        assert output_type in ['matrot', 'aa']

        # pose_input IS body_pose (không cần decode qua VAE)
        body_pose = pose_input.view(-1, self.prior.num_joints, 3)  # (B, 21, 3)

        if output_type == 'matrot':
            # Convert axis-angle → rotation matrix
            return SignbPoser.aa2matrot(body_pose).view(-1, 1, self.prior.num_joints, 9)
        else:
            return body_pose.view(-1, 1, self.prior.num_joints, 3)

    def encode(self, body_pose):
        """
        No-op encode (compatible interface).
        Trả về body_pose trực tiếp vì không có latent space.
        """
        return body_pose

    def forward(self, body_pose, output_type='matrot'):
        """Forward pass (compatible interface)."""
        return {'pose_aa': self.decode(body_pose, output_type='aa').view(-1, 63),
                'mean': body_pose, 'std': torch.zeros_like(body_pose)}

    def sample_poses(self, num_poses, output_type='aa', seed=None):
        """Sample random body poses từ learned distribution."""
        return self.prior.sample_poses(num_poses, output_type, seed)

    def prior_loss(self, body_pose, keypoints_2d=None, smplerx_init=None):
        """
        Compute prior loss cho body pose optimization.

        Args:
            body_pose: (1, 63) - body pose cần optimize (requires_grad=True)
            keypoints_2d: (1, J, 2) - 2D keypoints (optional, cho reconstruction loss)
            smplerx_init: (1, 63) - SMPLer-X initial pose (optional, cho init loss)

        Returns:
            loss: scalar
        """
        loss = torch.tensor(0.0, device=body_pose.device)

        # 1. MotionBERT guidance: pull body_pose toward MotionBERT prediction
        if keypoints_2d is not None:
            predicted_pose = self.prior.predict_single_frame(keypoints_2d)
            recon_loss = torch.nn.functional.l1_loss(body_pose, predicted_pose.detach())
            loss = loss + self.guidance_weight * recon_loss

        # 2. L2 regularization (light, keep poses natural)
        l2_reg = 1e-4 * torch.norm(body_pose, p=2)
        loss = loss + l2_reg

        return loss

    def set_cached_features(self, keypoints_2d):
        """Cache motion features để tránh recompute mỗi iteration."""
        self._cached_keypoints = keypoints_2d
        with torch.no_grad():
            self._cached_features = self.prior.encode(keypoints_2d)

    def get_cached_features(self):
        """Lấy cached motion features."""
        return self._cached_features


class MotionBERTDirectWrapper(nn.Module):
    """
    Alternative wrapper: optimize body_pose trực tiếp,
    dùng MotionBERT features làm conditioning (không cần keypoints_2d).

    Phù hợp khi không có 2D keypoints sẵn trong pipeline.
    """

    def __init__(self, num_joints=21):
        super().__init__()
        self.num_joints = num_joints

    def decode(self, pose_input, output_type='matrot'):
        """Pass-through: body_pose → body_pose."""
        assert output_type in ['matrot', 'aa']
        body_pose = pose_input.view(-1, self.num_joints, 3)

        if output_type == 'matrot':
            return SignbPoser.aa2matrot(body_pose).view(-1, 1, self.num_joints, 9)
        else:
            return body_pose.view(-1, 1, self.num_joints, 3)

    def encode(self, body_pose):
        return body_pose

    def forward(self, body_pose, output_type='matrot'):
        return {'pose_aa': self.decode(body_pose, output_type='aa').view(-1, 63),
                'mean': body_pose, 'std': torch.zeros_like(body_pose)}

    def sample_poses(self, num_poses, output_type='aa', seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randn(num_poses, self.num_joints * 3) * 0.1

    def prior_loss(self, body_pose, smplerx_init=None):
        """
        Simple prior loss: L2 regularization + optional init loss.
        """
        loss = 1e-4 * torch.norm(body_pose, p=2)

        if smplerx_init is not None:
            init_loss = torch.nn.functional.l1_loss(body_pose, smplerx_init.detach())
            loss = loss + init_loss

        return loss
