from __future__ import annotations

import torch


def finite_difference(sequence: torch.Tensor, fps: float, order: int = 1) -> torch.Tensor:
    result = sequence
    for _ in range(order):
        result = (result[1:] - result[:-1]) * fps
    return result


def dynamics_errors(
    prediction: torch.Tensor, target: torch.Tensor, fps: float
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("joint trajectories must match [T,J,3]")
    output: dict[str, torch.Tensor] = {}
    for order, name in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
        if prediction.shape[0] <= order:
            output[f"{name}_error"] = prediction.new_empty((0,))
            continue
        pred_derivative = finite_difference(prediction, fps, order)
        target_derivative = finite_difference(target, fps, order)
        output[f"{name}_error"] = torch.linalg.vector_norm(
            pred_derivative - target_derivative, dim=-1
        ).mean(-1)
    return output


def boundary_f1(
    predicted: torch.Tensor, target: torch.Tensor, tolerance_frames: int = 1
) -> dict[str, float]:
    predicted_indices = torch.where(predicted.bool())[0].tolist()
    target_indices = torch.where(target.bool())[0].tolist()
    matched: set[int] = set()
    true_positive = 0
    for value in predicted_indices:
        candidates = [
            index
            for index, truth in enumerate(target_indices)
            if index not in matched and abs(value - truth) <= tolerance_frames
        ]
        if candidates:
            best = min(candidates, key=lambda index: abs(value - target_indices[index]))
            matched.add(best)
            true_positive += 1
    precision = true_positive / max(len(predicted_indices), 1)
    recall = true_positive / max(len(target_indices), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}
