"""Rotation and SMPL-X geometry utilities."""

from .rotations import (
    axis_angle_to_matrix,
    compose_residual,
    geodesic_distance,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

__all__ = [
    "axis_angle_to_matrix",
    "compose_residual",
    "geodesic_distance",
    "matrix_to_axis_angle",
    "matrix_to_rotation_6d",
    "rotation_6d_to_matrix",
]
