from __future__ import annotations

from torch import Tensor

from signpk.geometry.robustifiers import masked_mean


def heteroscedastic_nll(
    residual: Tensor,
    log_variance: Tensor,
    valid: Tensor | None = None,
    minimum: float = -6.0,
    maximum: float = 4.0,
) -> Tensor:
    log_variance = log_variance.clamp(minimum, maximum)
    squared = residual.square()
    while log_variance.ndim < squared.ndim:
        log_variance = log_variance.unsqueeze(-1)
    return masked_mean((-log_variance).exp() * squared + log_variance, valid)


def expected_calibration_error(
    residual: Tensor,
    log_variance: Tensor,
    bins: int = 10,
) -> Tensor:
    predicted_scale = (0.5 * log_variance).exp().flatten()
    error = residual.abs().flatten()
    boundaries = predicted_scale.quantile(
        predicted_scale.new_tensor([index / bins for index in range(bins + 1)])
    )
    result = predicted_scale.new_zeros(())
    for index in range(bins):
        selected = (predicted_scale >= boundaries[index]) & (
            predicted_scale <= boundaries[index + 1] if index == bins - 1 else predicted_scale < boundaries[index + 1]
        )
        if selected.any():
            result = result + selected.float().mean() * (
                predicted_scale[selected].mean() - error[selected].mean()
            ).abs()
    return result

