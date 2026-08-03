"""Proposal-aligned burst masks for conditional diffusion training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


FINGER_CHAINS = (
    (21, 22, 23),
    (24, 25, 26),
    (27, 28, 29),
    (30, 31, 32),
    (33, 34, 35),
    (36, 37, 38),
    (39, 40, 41),
    (42, 43, 44),
    (45, 46, 47),
    (48, 49, 50),
)
MODES = (
    "full",
    "left_hand",
    "right_hand",
    "both_hands",
    "finger_chain",
    "wrist_attachment",
    "upper_body",
    "hand_swap",
    "palm_depth",
    "keypoint_dropout",
)
MODE_PROBABILITIES = (0.20, 0.12, 0.12, 0.10, 0.10, 0.10, 0.08, 0.06, 0.06, 0.06)


@dataclass(frozen=True)
class MaskResult:
    conditioning: torch.Tensor
    mode: str
    start: int
    duration: int


def sample_conditioning_mask(
    valid: torch.Tensor,
    generator: torch.Generator,
    durations: tuple[int, ...] = (4, 8, 16),
    mode: str | None = None,
) -> MaskResult:
    if valid.ndim != 2 or valid.shape[1] != 51:
        raise ValueError("valid must have shape (T,51)")
    length = valid.shape[0]
    selected_mode = (
        mode
        or MODES[
            int(
                torch.multinomial(
                    torch.tensor(MODE_PROBABILITIES), 1, generator=generator
                )
            )
        ]
    )
    if selected_mode not in MODES:
        raise ValueError(f"Unknown corruption mode: {selected_mode}")
    duration = min(
        durations[int(torch.randint(len(durations), (), generator=generator))], length
    )
    start = int(torch.randint(max(1, length - duration + 1), (), generator=generator))
    result = valid.clone()
    window = slice(start, start + duration)
    if selected_mode == "full":
        return MaskResult(result, selected_mode, start, duration)
    if selected_mode == "left_hand":
        result[window, 21:36] = False
    elif selected_mode == "right_hand":
        result[window, 36:51] = False
    elif selected_mode in ("both_hands", "hand_swap", "palm_depth"):
        result[window, 21:51] = False
    elif selected_mode == "finger_chain":
        chain = FINGER_CHAINS[
            int(torch.randint(len(FINGER_CHAINS), (), generator=generator))
        ]
        result[window, list(chain)] = False
    elif selected_mode == "wrist_attachment":
        result[window, [19, 20, 21, 36]] = False
    elif selected_mode == "upper_body":
        result[window, 2:21] = False
    elif selected_mode == "keypoint_dropout":
        selected = torch.rand((duration, 51), generator=generator) < 0.4
        result[window] &= ~selected
    return MaskResult(result, selected_mode, start, duration)
