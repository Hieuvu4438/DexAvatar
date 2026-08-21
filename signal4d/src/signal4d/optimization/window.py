from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Window:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def plan_windows(
    total_frames: int, length: int, stride: int, transitions: torch.Tensor | None = None
) -> list[Window]:
    if total_frames <= 0 or length <= 0 or stride <= 0 or stride > length:
        raise ValueError("invalid window arguments")
    if total_frames <= length:
        return [Window(0, total_frames)]
    starts = list(range(0, total_frames - length + 1, stride))
    if starts[-1] + length < total_frames:
        starts.append(total_frames - length)
    if transitions is not None:
        adjusted = []
        for start in starts:
            if start == 0 or start + length == total_frames:
                adjusted.append(start)
                continue
            candidates = range(max(0, start - 3), min(total_frames - length, start + 3) + 1)
            adjusted.append(
                min(
                    candidates,
                    key=lambda candidate: float(
                        transitions[candidate] + transitions[candidate + length - 1]
                    ),
                )
            )
        starts = sorted(set(adjusted))
    return [Window(start, min(total_frames, start + length)) for start in starts]


def hann_weights(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length == 1:
        return torch.ones(1, device=device, dtype=dtype)
    weights = torch.hann_window(length + 2, periodic=False, device=device, dtype=dtype)[1:-1]
    return weights.clamp_min(1e-3)
