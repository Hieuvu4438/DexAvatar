from __future__ import annotations

import torch


def binary_contact_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError("contact tensors must match")
    prediction = prediction.bool()
    target = target.bool()
    tp = int((prediction & target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive": fp,
        "false_negative": fn,
    }


def contact_events(sequence: torch.Tensor) -> list[tuple[int, int]]:
    values = sequence.bool().flatten().tolist()
    events: list[tuple[int, int]] = []
    start = None
    for index, active in enumerate(values + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            events.append((start, index))
            start = None
    return events


def event_metrics(
    prediction: torch.Tensor, target: torch.Tensor, tolerance_frames: int = 1
) -> dict[str, float]:
    if prediction.ndim == 1:
        prediction = prediction[:, None]
        target = target[:, None]
    scores = []
    onset_errors = []
    offset_errors = []
    for edge in range(prediction.shape[1]):
        pred_events = contact_events(prediction[:, edge])
        target_events = contact_events(target[:, edge])
        if not pred_events and not target_events:
            scores.append(1.0)
            continue
        matched: set[int] = set()
        true_positive = 0
        for onset, offset in pred_events:
            candidates = [
                index
                for index, (truth_on, truth_off) in enumerate(target_events)
                if index not in matched and abs(onset - truth_on) <= tolerance_frames
            ]
            if candidates:
                best = min(candidates, key=lambda index: abs(onset - target_events[index][0]))
                matched.add(best)
                true_positive += 1
                onset_errors.append(abs(onset - target_events[best][0]))
                offset_errors.append(abs(offset - target_events[best][1]))
        precision = true_positive / max(len(pred_events), 1)
        recall = true_positive / max(len(target_events), 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return {
        "event_macro_f1": sum(scores) / max(len(scores), 1),
        "onset_mae_frames": sum(onset_errors) / max(len(onset_errors), 1),
        "offset_mae_frames": sum(offset_errors) / max(len(offset_errors), 1),
    }
