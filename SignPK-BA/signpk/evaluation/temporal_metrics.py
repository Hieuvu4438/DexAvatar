from __future__ import annotations

import numpy as np


def velocity(vertices: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    dt = np.diff(timestamps).clip(min=1e-8)
    return np.diff(vertices, axis=0) / dt[:, None, None]


def acceleration(vertices: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    speeds = velocity(vertices, timestamps)
    midpoint_times = (timestamps[1:] + timestamps[:-1]) * 0.5
    dt = np.diff(midpoint_times).clip(min=1e-8)
    return np.diff(speeds, axis=0) / dt[:, None, None]


def velocity_error(prediction: np.ndarray, target: np.ndarray, timestamps: np.ndarray) -> float:
    return float(np.linalg.norm(velocity(prediction, timestamps) - velocity(target, timestamps), axis=-1).mean())


def acceleration_error(prediction: np.ndarray, target: np.ndarray, timestamps: np.ndarray) -> float:
    return float(np.linalg.norm(acceleration(prediction, timestamps) - acceleration(target, timestamps), axis=-1).mean())

