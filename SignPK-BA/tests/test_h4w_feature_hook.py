from pathlib import Path

import torch
from torch import nn

from signpk.observers.h4w_feature_hook import (
    H4WFeatureCapture,
    load_h4w_feature_cache,
    save_h4w_feature_cache,
)


class _WiLoR(nn.Module):
    def forward(self, value):
        output = [value for _ in range(14)]
        output[6] = value + 1
        output[13] = value + 2
        return tuple(output)


class _HandControl(nn.Module):
    def forward(self, value):
        return [value + 3]


class _BodyRegressor(nn.Module):
    def forward(self, value):
        return value


class _MockH4W(nn.Module):
    def __init__(self):
        super().__init__()
        self.wilor = _WiLoR()
        self.hand_control_net = _HandControl()
        self.body_regressor = _BodyRegressor()

    def forward(self, value):
        self.wilor(value)
        self.hand_control_net(value)
        self.body_regressor(value)


def test_h4w_feature_hook_and_cache_roundtrip(tmp_path: Path):
    model = _MockH4W()
    with H4WFeatureCapture(model) as capture:
        model(torch.ones(2, 8))
        batch = capture.pop_batch()
    assert batch["body_pose_token"].dtype == torch.float16
    path = tmp_path / "features.pt"
    save_h4w_feature_cache(path, torch.tensor([1, 3]), [batch], {"commit": "abc"})
    features, metadata = load_h4w_feature_cache(path, (1, 3))
    assert features["left_wilor_feature"].shape == (2, 8)
    assert metadata["commit"] == "abc"
