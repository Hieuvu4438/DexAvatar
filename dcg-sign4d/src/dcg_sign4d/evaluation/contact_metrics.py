"""Gold-subset dynamic event metrics with uncertain-label exclusion."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _active_intervals(
    state: NDArray[np.integer], valid: NDArray[np.bool_]
) -> list[tuple[int, int]]:
    active = np.isin(state, (1, 2)) & valid
    padded = np.pad(active.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True))


def _interval_iou(first: tuple[int, int], second: tuple[int, int]) -> float:
    intersection = max(0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union else 0.0


def contact_event_metrics(
    prediction: NDArray[np.integer],
    target: NDArray[np.integer],
    uncertain: NDArray[np.bool_],
    *,
    fps: float,
) -> dict[str, float | int]:
    if prediction.shape != target.shape or uncertain.shape != target.shape:
        raise ValueError("prediction/target/uncertain shape mismatch")
    if target.ndim != 2 or fps <= 0:
        raise ValueError("events must be [T,E] and fps positive")
    active = ~uncertain
    confusion = np.zeros((4, 4), dtype=np.int64)
    for expected, observed in zip(target[active], prediction[active], strict=True):
        confusion[expected, observed] += 1
    precision, recall, f1 = [], [], []
    for state in range(4):
        true_positive = confusion[state, state]
        predicted_positive = confusion[:, state].sum()
        actual_positive = confusion[state].sum()
        state_precision = true_positive / predicted_positive if predicted_positive else 0.0
        state_recall = true_positive / actual_positive if actual_positive else 0.0
        state_f1 = (
            2 * state_precision * state_recall / (state_precision + state_recall)
            if state_precision + state_recall
            else 0.0
        )
        precision.append(state_precision)
        recall.append(state_recall)
        f1.append(state_f1)
    correct = np.trace(confusion)
    support = confusion.sum()
    micro = float(correct / support) if support else float("nan")
    active_true_positive = confusion[1:3, 1:3].sum()
    active_predicted = confusion[:, 1:3].sum()
    active_actual = confusion[1:3, :].sum()
    active_precision = active_true_positive / active_predicted if active_predicted else 0.0
    active_recall = active_true_positive / active_actual if active_actual else 0.0
    active_f1 = (
        2 * active_precision * active_recall / (active_precision + active_recall)
        if active_precision + active_recall
        else 0.0
    )
    timing: dict[str, float] = {}
    for name, state in (("onset", 1), ("release", 3)):
        errors = []
        for edge in range(target.shape[1]):
            expected = np.flatnonzero((target[:, edge] == state) & active[:, edge])
            observed = np.flatnonzero((prediction[:, edge] == state) & active[:, edge])
            if len(expected) and len(observed):
                errors.extend(min(abs(frame - observed)).item() / fps for frame in expected)
        timing[f"{name}_timing_mae_sec"] = float(np.mean(errors)) if errors else float("nan")
    matched = 0
    prediction_segments = 0
    target_segments = 0
    interval_ious = []
    for edge in range(target.shape[1]):
        valid = active[:, edge]
        expected_intervals = _active_intervals(target[:, edge], valid)
        observed_intervals = _active_intervals(prediction[:, edge], valid)
        target_segments += len(expected_intervals)
        prediction_segments += len(observed_intervals)
        unused = set(range(len(observed_intervals)))
        for expected_interval in expected_intervals:
            candidates = [
                (_interval_iou(expected_interval, observed_intervals[index]), index)
                for index in unused
            ]
            if candidates:
                best_iou, best_index = max(candidates)
                interval_ious.append(best_iou)
                if best_iou >= 0.5:
                    matched += 1
                    unused.remove(best_index)
    segment_precision = matched / prediction_segments if prediction_segments else 0.0
    segment_recall = matched / target_segments if target_segments else 0.0
    segment_f1 = (
        2 * segment_precision * segment_recall / (segment_precision + segment_recall)
        if segment_precision + segment_recall
        else 0.0
    )
    return {
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "micro_accuracy": micro,
        "micro_precision": micro,
        "micro_recall": micro,
        "micro_f1": micro,
        "active_contact_precision": float(active_precision),
        "active_contact_recall": float(active_recall),
        "active_contact_f1": float(active_f1),
        "segmental_f1_iou50": float(segment_f1),
        "mean_matched_interval_iou": (
            float(np.mean(interval_ious)) if interval_ious else float("nan")
        ),
        "predicted_segments": prediction_segments,
        "target_segments": target_segments,
        "valid_support": int(support),
        "uncertain_count": int(uncertain.sum()),
        **timing,
    }
