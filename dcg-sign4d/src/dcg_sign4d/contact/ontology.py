"""Dynamic contact-event state and graph contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor


class EventState(IntEnum):
    OFF = 0
    ONSET = 1
    HOLD = 2
    RELEASE = 3


@dataclass(frozen=True)
class ContactGraphBatch:
    event_state: Tensor
    event_probability: Tensor
    edge_valid: Tensor
    uncertain_mask: Tensor
    segment_id: Tensor
    segment_duration: Tensor

    def validate(self) -> ContactGraphBatch:
        if self.event_state.ndim != 3:
            raise ValueError("event_state must be [B,T,E]")
        shape = self.event_state.shape
        if self.event_probability.shape != (*shape, 4):
            raise ValueError("event_probability must be [B,T,E,4]")
        if self.edge_valid.shape != (shape[0], shape[2]):
            raise ValueError("edge_valid must be [B,E]")
        for name in ("uncertain_mask", "segment_id", "segment_duration"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} shape mismatch")
        if self.event_state.dtype != torch.long:
            raise ValueError("event_state must be torch.long")
        if self.edge_valid.dtype != torch.bool or self.uncertain_mask.dtype != torch.bool:
            raise ValueError("validity/uncertainty masks must be bool")
        if not torch.isfinite(self.event_probability).all():
            raise ValueError("event probability contains NaN/Inf")
        if bool(((self.event_state < 0) | (self.event_state > 3)).any()):
            raise ValueError("invalid event state")
        return self


VALID_FRAME_TRANSITIONS = torch.tensor(
    [
        [True, True, False, False],
        [False, False, True, False],
        [False, False, True, True],
        [True, False, False, False],
    ],
    dtype=torch.bool,
)
