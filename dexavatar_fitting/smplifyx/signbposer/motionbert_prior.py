"""
MotionBERT Body Pose Prior
==========================
Pretrained MotionBERT encoder + trainable SMPL regression head.
Thay thế SignBPoser: predict body_pose trực tiếp từ 2D keypoints,
không cần latent space.

Usage:
    from signbposer.motionbert_prior import MotionBERTBodyPrior
    model = MotionBERTBodyPrior(motionbert_ckpt_path='checkpoints/motionbert/pose3d_MB_ft_h36m.tar')
    body_pose = model(keypoints_2d)  # (B, 63)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding cho timestep hoặc positional encoding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class SMPLRegressionHead(nn.Module):
    """
    MLP head: motion features → SMPL body_pose (63-dim axis-angle).
    Input: motion features từ MotionBERT encoder (256-dim hoặc 512-dim)
    Output: body_pose (21 joints × 3 axis-angle) hoặc rot6d (21 × 6)
    """

    def __init__(self, input_dim=256, hidden_dim=512, num_joints=21,
                 output_type='aa', dropout=0.1):
        super().__init__()
        self.num_joints = num_joints
        self.output_type = output_type

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),    # (B, C, T) → (B, C, 1)
            nn.Flatten(),                # (B, C)
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_joints * 6),  # rot6d output
        )

        # Initialize last layer nhỏ để bắt đầu gần identity rotation
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        """
        Args:
            features: (B, T, C) hoặc (B, C, T) motion features
        Returns:
            rot6d: (B, num_joints*6) nếu output_type='rot6d'
            aa: (B, num_joints*3) nếu output_type='aa'
        """
        if features.dim() == 3 and features.shape[1] != features.shape[2]:
            # (B, T, C) → (B, C, T) cho AdaptiveAvgPool1d
            features = features.permute(0, 2, 1)

        rot6d = self.net(features)  # (B, num_joints*6)

        if self.output_type == 'aa':
            return self._rot6d_to_aa(rot6d)
        return rot6d

    @staticmethod
    def _rot6d_to_aa(rot6d):
        """Convert 6D rotation representation to axis-angle."""
        # rot6d: (B, num_joints*6) → (B*num_joints, 6)
        batch_size = rot6d.shape[0]
        rot6d_flat = rot6d.reshape(-1, 6)

        # Gram-Schmidt: 6D → rotation matrix
        b1 = torch.nn.functional.normalize(rot6d_flat[:, :3], dim=1)
        dot = torch.sum(b1 * rot6d_flat[:, 3:], dim=1, keepdim=True)
        b2 = torch.nn.functional.normalize(rot6d_flat[:, 3:] - dot * b1, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        rot_mat = torch.stack([b1, b2, b3], dim=-1)  # (N, 3, 3)

        # Rotation matrix → axis-angle
        # Sử dụng công thức Rodrigues (không cần torchgeometry)
        aa = _rotation_matrix_to_angle_axis(rot_mat)

        return aa.view(batch_size, -1)  # (B, num_joints*3)


def _rotation_matrix_to_angle_axis(rot_mat):
    """
    Convert rotation matrix to axis-angle.
    rot_mat: (N, 3, 3) → (N, 3)
    Pure PyTorch implementation, không cần torchgeometry.
    """
    # Clamp để tránh numerical issues
    trace = rot_mat[:, 0, 0] + rot_mat[:, 1, 1] + rot_mat[:, 2, 2]
    trace = torch.clamp(trace, -1.0 + 1e-7, 3.0 - 1e-7)

    angle = torch.acos((trace - 1) / 2)

    # Axis từ skew-symmetric part
    axis = torch.zeros_like(rot_mat[:, :, 0])  # (N, 3)
    axis[:, 0] = rot_mat[:, 2, 1] - rot_mat[:, 1, 2]
    axis[:, 1] = rot_mat[:, 0, 2] - rot_mat[:, 2, 0]
    axis[:, 2] = rot_mat[:, 1, 0] - rot_mat[:, 0, 1]

    # Normalize axis
    axis_norm = torch.norm(axis, dim=1, keepdim=True).clamp(min=1e-7)
    axis = axis / axis_norm

    # Handle small angle case (near identity)
    small_angle = angle.abs() < 1e-6
    axis[small_angle.squeeze()] = torch.zeros(3, device=axis.device)

    # aa = angle * axis
    aa = angle.unsqueeze(-1) * axis  # (N, 3)

    return aa


class MotionBERTBodyPrior(nn.Module):
    """
    Body pose prior sử dụng pretrained MotionBERT encoder + trainable SMPL head.

    Thay thế SignBPoser:
    - Không cần latent space (direct body_pose prediction)
    - Pretrained trên AMASS (real mocap data)
    - Finetune trên sign language data

    Args:
        motionbert_ckpt_path: path to pretrained MotionBERT checkpoint
        freeze_encoder: freeze MotionBERT encoder weights
        hidden_dim: hidden dimension cho SMPL head
        output_type: 'aa' (axis-angle) hoặc 'rot6d'
    """

    def __init__(self, motionbert_ckpt_path=None, freeze_encoder=True,
                 input_dim=256, hidden_dim=512, num_joints=21,
                 output_type='aa', dropout=0.1):
        super().__init__()

        self.freeze_encoder = freeze_encoder
        self.num_joints = num_joints

        # MotionBERT encoder (pretrained)
        self.encoder = None
        self.encoder_dim = input_dim
        if motionbert_ckpt_path and os.path.exists(motionbert_ckpt_path):
            self._load_motionbert_encoder(motionbert_ckpt_path)

        # SMPL regression head (trainable)
        self.smpl_head = SMPLRegressionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_joints=num_joints,
            output_type=output_type,
            dropout=dropout,
        )

    def _load_motionbert_encoder(self, ckpt_path):
        """Load pretrained MotionBERT encoder."""
        try:
            # Thêm MotionBERT vào path
            mb_dir = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))
            mb_path = os.path.join(mb_dir, 'MotionBERT')
            if mb_path not in sys.path:
                sys.path.insert(0, mb_path)

            from models.motion_bert import MotionBERT

            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            self.encoder = MotionBERT()
            self.encoder.load_state_dict(state_dict, strict=False)

            if self.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()

            print(f"[MotionBERTBodyPrior] Loaded pretrained encoder from {ckpt_path}")

        except Exception as e:
            print(f"[MotionBERTBodyPrior] Warning: Could not load MotionBERT: {e}")
            print("[MotionBERTBodyPrior] Using random encoder for testing")
            self.encoder = None

    def encode(self, keypoints_2d):
        """
        Extract motion features từ 2D keypoints.

        Args:
            keypoints_2d: (B, T, J, 2) hoặc (B, J, 2)

        Returns:
            features: (B, T, encoder_dim) hoặc (B, 1, encoder_dim)
        """
        if keypoints_2d.dim() == 3:
            keypoints_2d = keypoints_2d.unsqueeze(1)  # add temporal dim

        if self.encoder is not None:
            with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                features = self.encoder.get_features(keypoints_2d)
        else:
            # Fallback: random features (for testing/debugging)
            B, T = keypoints_2d.shape[:2]
            features = torch.randn(B, T, self.encoder_dim,
                                   device=keypoints_2d.device)

        return features

    def forward(self, keypoints_2d):
        """
        Predict body_pose từ 2D keypoints.

        Args:
            keypoints_2d: (B, T, J, 2) hoặc (B, J, 2)

        Returns:
            body_pose: (B, 63) axis-angle (21 joints × 3)
        """
        features = self.encode(keypoints_2d)  # (B, T, C)
        body_pose = self.smpl_head(features)  # (B, 63)
        return body_pose

    def predict_single_frame(self, keypoints_2d_single):
        """
        Wrapper cho single-frame inference (DexAvatar pipeline).

        Args:
            keypoints_2d_single: (1, J, 2) hoặc (J, 2)

        Returns:
            body_pose: (1, 63)
        """
        if keypoints_2d_single.dim() == 2:
            keypoints_2d_single = keypoints_2d_single.unsqueeze(0)
        # Add temporal dim: (1, J, 2) → (1, 1, J, 2)
        keypoints_2d = keypoints_2d_single.unsqueeze(1)
        return self.forward(keypoints_2d)

    def sample_poses(self, num_poses, output_type='aa', seed=None):
        """Sample random body poses (compatible với SignBPoser interface)."""
        if seed is not None:
            torch.manual_seed(seed)
        # Sample từ distribution learned bởi SMPL head
        dummy_features = torch.randn(num_poses, 1, self.encoder_dim)
        with torch.no_grad():
            body_pose = self.smpl_head(dummy_features)
        return body_pose
