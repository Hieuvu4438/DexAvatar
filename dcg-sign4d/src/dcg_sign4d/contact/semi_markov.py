"""Exact segmental Viterbi decoding for off/onset/hold/release events."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .ontology import VALID_FRAME_TRANSITIONS, ContactGraphBatch, EventState


@dataclass(frozen=True)
class _BackPointer:
    start: int
    previous_state: int


class SemiMarkovDecoder:
    """Exact max-sum decoder over contiguous event segments.

    `duration_logits[b, s, e, d-1]` scores a segment starting at `s` with
    duration `d`. ONSET and RELEASE are one-frame boundary events. OFF and HOLD
    can last up to `max_duration`. Any state may occur at a cropped-window
    boundary; all internal transitions obey the frozen transition graph.
    """

    def __init__(
        self,
        max_duration: int,
        transition_scores: Tensor | None = None,
        *,
        fps: float = 1.0,
    ) -> None:
        if max_duration < 1 or fps <= 0:
            raise ValueError("max_duration and fps must be positive")
        self.max_duration = max_duration
        self.fps = fps
        if transition_scores is None:
            transition_scores = torch.zeros(4, 4)
        if transition_scores.shape != (4, 4):
            raise ValueError("transition_scores must be [4,4]")
        self.transition_scores = transition_scores

    def decode(
        self,
        event_logits: Tensor,
        duration_logits: Tensor,
        edge_valid: Tensor,
        frame_valid: Tensor | None = None,
    ) -> ContactGraphBatch:
        if event_logits.ndim != 4 or event_logits.shape[-1] != 4:
            raise ValueError("event_logits must be [B,T,E,4]")
        batch, time, edges, _ = event_logits.shape
        if duration_logits.shape[:3] != (batch, time, edges):
            raise ValueError("duration_logits prefix must match [B,T,E]")
        if duration_logits.shape[-1] < self.max_duration:
            raise ValueError("duration logits do not cover max_duration")
        if edge_valid.shape != (batch, edges) or edge_valid.dtype != torch.bool:
            raise ValueError("edge_valid must be bool [B,E]")
        if frame_valid is None:
            frame_valid = torch.ones(batch, time, dtype=torch.bool, device=event_logits.device)
        if frame_valid.shape != (batch, time) or frame_valid.dtype != torch.bool:
            raise ValueError("frame_valid must be bool [B,T]")
        # Valid frames must be a prefix so padded frames cannot create segments.
        for valid in frame_valid:
            count = int(valid.sum())
            if not bool(valid[:count].all()) or bool(valid[count:].any()):
                raise ValueError("frame_valid must be a contiguous prefix")

        probabilities = event_logits.softmax(dim=-1)
        states = torch.full((batch, time, edges), int(EventState.OFF), device=event_logits.device)
        states = states.long()
        segment_ids = torch.full_like(states, -1)
        durations = torch.zeros(batch, time, edges, device=event_logits.device)
        uncertain = torch.zeros(batch, time, edges, dtype=torch.bool, device=event_logits.device)

        for batch_idx in range(batch):
            valid_time = int(frame_valid[batch_idx].sum())
            for edge_idx in range(edges):
                if not bool(edge_valid[batch_idx, edge_idx]) or valid_time == 0:
                    continue
                decoded, decoded_segments = self._decode_one(
                    event_logits[batch_idx, :valid_time, edge_idx],
                    duration_logits[batch_idx, :valid_time, edge_idx],
                )
                states[batch_idx, :valid_time, edge_idx] = decoded
                for segment_id, (start, end) in enumerate(decoded_segments):
                    segment_ids[batch_idx, start:end, edge_idx] = segment_id
                    durations[batch_idx, start:end, edge_idx] = float(end - start) / self.fps
                confidence = probabilities[batch_idx, :valid_time, edge_idx].amax(dim=-1)
                uncertain[batch_idx, :valid_time, edge_idx] = confidence < 0.5

        return ContactGraphBatch(
            event_state=states,
            event_probability=probabilities,
            edge_valid=edge_valid,
            uncertain_mask=uncertain,
            segment_id=segment_ids,
            segment_duration=durations,
        ).validate()

    def _decode_one(
        self, logits: Tensor, duration_logits: Tensor
    ) -> tuple[Tensor, list[tuple[int, int]]]:
        time = logits.shape[0]
        neg_inf = torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device)
        scores = torch.full((time + 1, 4), neg_inf, dtype=logits.dtype, device=logits.device)
        pointers: list[list[_BackPointer | None]] = [[None] * 4 for _ in range(time + 1)]
        prefix = torch.cat((torch.zeros(1, 4, device=logits.device), logits.cumsum(dim=0)))
        transitions = self.transition_scores.to(device=logits.device, dtype=logits.dtype)

        for end in range(1, time + 1):
            for state in range(4):
                state_limit = (
                    1 if state in (EventState.ONSET, EventState.RELEASE) else self.max_duration
                )
                for duration in range(1, min(end, state_limit) + 1):
                    start = end - duration
                    segment_score = prefix[end, state] - prefix[start, state]
                    segment_score = segment_score + duration_logits[start, duration - 1]
                    if start == 0:
                        candidate = segment_score
                        previous_state = -1
                    else:
                        allowed_previous = torch.where(
                            VALID_FRAME_TRANSITIONS[:, state].to(logits.device)
                        )[0]
                        # A same-state frame transition belongs inside one segment.
                        allowed_previous = allowed_previous[allowed_previous != state]
                        if allowed_previous.numel() == 0:
                            continue
                        previous_values = (
                            scores[start, allowed_previous] + transitions[allowed_previous, state]
                        )
                        best_index = int(previous_values.argmax())
                        previous_state = int(allowed_previous[best_index])
                        candidate = previous_values[best_index] + segment_score
                    if bool(candidate > scores[end, state]):
                        scores[end, state] = candidate
                        pointers[end][state] = _BackPointer(start, previous_state)

        state = int(scores[time].argmax())
        if pointers[time][state] is None:
            raise RuntimeError("no valid semi-Markov path")
        output = torch.empty(time, dtype=torch.long, device=logits.device)
        segments_reversed: list[tuple[int, int]] = []
        end = time
        while end > 0:
            pointer = pointers[end][state]
            if pointer is None:
                raise RuntimeError("broken semi-Markov backpointer")
            output[pointer.start : end] = state
            segments_reversed.append((pointer.start, end))
            end = pointer.start
            state = pointer.previous_state
        return output, list(reversed(segments_reversed))
