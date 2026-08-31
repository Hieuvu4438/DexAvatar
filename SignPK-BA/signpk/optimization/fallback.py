from __future__ import annotations

import math

import torch

from .smplx_layer import SMPLXOutput
from .state import SequenceState


def passes_sanity_checks(
    state: SequenceState,
    output: SMPLXOutput,
    *,
    max_body_degrees: float = 20.0,
    max_wrist_degrees: float = 30.0,
    max_finger_degrees: float = 25.0,
) -> tuple[bool, str | None]:
    tensors = [output.vertices, output.joints, state.translation, state.beta]
    if any(not torch.isfinite(value).all() for value in tensors):
        return False, "NaN/Inf"
    if output.vertices.shape != (state.num_frames, 10475, 3):
        return False, "topology"
    residuals = state.residual_radians()
    limits = {
        "root": max_body_degrees,
        "upper_body": max_body_degrees,
        "wrists": max_wrist_degrees,
        "left_hand": max_finger_degrees,
        "right_hand": max_finger_degrees,
    }
    for name, values in residuals.items():
        if values.max() > math.radians(limits[name]) * 1.5:
            return False, f"{name}_residual"
    return True, None

