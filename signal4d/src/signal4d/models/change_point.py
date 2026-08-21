from __future__ import annotations

import torch


def robust_standardize(features: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # CUDA median-with-indices is non-deterministic; change cues are detached
    # observations, so compute the robust statistics deterministically on CPU.
    cpu_features = features.detach().cpu()
    median = cpu_features.median(dim=0).values.to(features)
    mad = (cpu_features - median.cpu()).abs().median(dim=0).values.to(features).clamp_min(eps)
    return (features - median) / (1.4826 * mad)


def rule_based_change_probability(
    features: torch.Tensor, reliability: torch.Tensor | None = None
) -> torch.Tensor:
    """Monotone multi-cue detector; a single moderate cue cannot trigger a boundary."""
    if features.ndim != 2:
        raise ValueError("change features must have shape [T,F]")
    standardized = robust_standardize(features).clamp(-8, 8)
    positive = torch.relu(standardized)
    if reliability is not None:
        if reliability.shape != features.shape:
            raise ValueError("reliability must match features")
        positive = positive * reliability.clamp(0, 1)
    strongest = positive.topk(k=min(2, positive.shape[-1]), dim=-1).values
    if strongest.shape[-1] == 1:
        evidence = strongest[..., 0] - 4.0
    else:
        evidence = strongest[..., 0] + strongest[..., 1] - 4.0
    probability = torch.sigmoid(evidence)
    probability[0] = 0
    return probability
