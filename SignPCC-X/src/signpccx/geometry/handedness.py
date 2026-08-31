from __future__ import annotations

import numpy as np


LEFT_AXIS_ANGLE_MIRROR = np.asarray([1.0, -1.0, -1.0], dtype=np.float32)


def unmirror_left_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    value = np.asarray(axis_angle)
    if value.shape[-1] != 3:
        raise ValueError(value.shape)
    return value * LEFT_AXIS_ANGLE_MIRROR

