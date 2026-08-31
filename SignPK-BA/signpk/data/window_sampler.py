from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalWindow:
    center_index: int
    indices: tuple[int, ...]
    padded: tuple[bool, ...]

    @property
    def padding_ratio(self) -> float:
        return sum(self.padded) / len(self.padded)


def _reflect(index: int, length: int) -> int:
    if length <= 1:
        return 0
    period = 2 * (length - 1)
    value = index % period
    return value if value < length else period - value


def make_window(center: int, length: int, window: int = 9, gap: int = 1, padding: str = "reflect") -> TemporalWindow:
    if window <= 0 or window % 2 == 0:
        raise ValueError("window length must be a positive odd number")
    if gap <= 0 or length <= 0 or not 0 <= center < length:
        raise ValueError("invalid temporal window parameters")
    raw = [center + (offset - window // 2) * gap for offset in range(window)]
    padded = tuple(not 0 <= index < length for index in raw)
    if padding == "reflect":
        indices = tuple(_reflect(index, length) for index in raw)
    elif padding == "replicate":
        indices = tuple(min(max(index, 0), length - 1) for index in raw)
    else:
        raise ValueError(f"unknown padding policy {padding!r}")
    return TemporalWindow(center, indices, padded)


def all_windows(length: int, window: int = 9, gap: int = 1, padding: str = "reflect") -> list[TemporalWindow]:
    return [make_window(i, length, window, gap, padding) for i in range(length)]


def take_window(array: np.ndarray, spec: TemporalWindow) -> np.ndarray:
    return np.asarray(array)[np.asarray(spec.indices)]

