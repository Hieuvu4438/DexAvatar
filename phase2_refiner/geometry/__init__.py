"""Rotation and SMPL-X geometry utilities."""

from .coordinates import (
    compose_transforms,
    invert_transform,
    transform_points,
    validate_transform,
)
from .palm import fingertips, palm_center, palm_normal
from .rotations import (
    axis_angle_to_matrix,
    compose_residual,
    geodesic_distance,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from .smplx_decode import decode_smplx_sequence, split_pose_matrices

__all__ = [
    "axis_angle_to_matrix",
    "compose_transforms",
    "compose_residual",
    "decode_smplx_sequence",
    "fingertips",
    "geodesic_distance",
    "invert_transform",
    "matrix_to_axis_angle",
    "matrix_to_rotation_6d",
    "palm_center",
    "palm_normal",
    "rotation_6d_to_matrix",
    "split_pose_matrices",
    "transform_points",
    "validate_transform",
]
