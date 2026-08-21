from __future__ import annotations

import torch


def retrieval_top1_accuracy(
    prediction_embeddings: torch.Tensor, target_embeddings: torch.Tensor
) -> float:
    if prediction_embeddings.shape != target_embeddings.shape or prediction_embeddings.ndim != 2:
        raise ValueError("semantic embeddings must match [N,D]")
    prediction = torch.nn.functional.normalize(prediction_embeddings, dim=-1)
    target = torch.nn.functional.normalize(target_embeddings, dim=-1)
    nearest = (prediction @ target.transpose(0, 1)).argmax(-1)
    truth = torch.arange(prediction.shape[0], device=prediction.device)
    return float((nearest == truth).float().mean())


def semantic_noninferiority(candidate: float, baseline: float, margin: float) -> bool:
    return candidate - baseline >= -margin
