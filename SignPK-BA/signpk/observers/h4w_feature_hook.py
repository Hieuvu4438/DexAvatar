from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from signpk.data.cache_schema import CACHE_SCHEMA_VERSION


FEATURE_SCHEMA = "signpk-h4w-features-v1"


def _frozen_half(value: Tensor) -> Tensor:
    return value.detach().to(device="cpu", dtype=torch.float16).contiguous()


class H4WFeatureCapture(AbstractContextManager):
    """Capture frozen H4W++ tokens with hooks instead of patching upstream."""

    def __init__(self, model: nn.Module):
        required = ("wilor", "hand_control_net", "body_regressor")
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise AttributeError(f"H4W++ model missing hook points {missing}")
        self.model = model
        self.handles: list[Any] = []
        self.current: dict[str, Tensor] = {}

    def __enter__(self):
        def wilor_hook(_module, _inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) < 14:
                raise ValueError("unexpected WiLoR output contract")
            self.current["right_wilor_feature"] = _frozen_half(output[6])
            self.current["left_wilor_feature"] = _frozen_half(output[13])

        def hand_control_hook(_module, _inputs, output):
            value = output[-1] if isinstance(output, (tuple, list)) else output
            if not isinstance(value, Tensor):
                raise ValueError("unexpected HandControlNet output contract")
            self.current["hand_control_feature"] = _frozen_half(value)

        def body_pre_hook(_module, inputs):
            if not inputs or not isinstance(inputs[0], Tensor):
                raise ValueError("unexpected BodyRotationNet input contract")
            self.current["body_pose_token"] = _frozen_half(inputs[0])

        self.handles = [
            self.model.wilor.register_forward_hook(wilor_hook),
            self.model.hand_control_net.register_forward_hook(hand_control_hook),
            self.model.body_regressor.register_forward_pre_hook(body_pre_hook),
        ]
        return self

    def pop_batch(self) -> dict[str, Tensor]:
        required = {
            "right_wilor_feature",
            "left_wilor_feature",
            "hand_control_feature",
            "body_pose_token",
        }
        missing = required - set(self.current)
        if missing:
            raise RuntimeError(f"H4W++ forward did not trigger feature hooks {sorted(missing)}")
        result, self.current = self.current, {}
        return result

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.current = {}
        return False


def save_h4w_feature_cache(
    path: str | Path,
    frame_ids: Tensor,
    batches: list[dict[str, Tensor]],
    metadata: dict[str, Any],
) -> None:
    if not batches:
        raise ValueError("cannot save an empty H4W++ feature cache")
    keys = set(batches[0])
    if any(set(batch) != keys for batch in batches):
        raise ValueError("H4W++ feature batches have inconsistent keys")
    features = {key: torch.cat([batch[key] for batch in batches], dim=0) for key in keys}
    length = len(frame_ids)
    if any(value.shape[0] != length for value in features.values()):
        raise ValueError("H4W++ feature count does not match frame IDs")
    payload: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA,
        "observer_schema_version": CACHE_SCHEMA_VERSION,
        "frame_ids": frame_ids.detach().cpu().to(torch.int64),
        "metadata_json": json.dumps(metadata, sort_keys=True),
        **features,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_h4w_feature_cache(
    path: str | Path,
    expected_frame_ids: tuple[int, ...],
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != FEATURE_SCHEMA:
        raise ValueError("unsupported H4W++ feature cache schema")
    frame_ids = tuple(int(value) for value in payload["frame_ids"].tolist())
    if frame_ids != expected_frame_ids:
        raise ValueError("H4W++ feature frame IDs do not match the manifest")
    features = {
        key: value.float()
        for key, value in payload.items()
        if key not in {"schema_version", "observer_schema_version", "frame_ids", "metadata_json"}
    }
    return features, json.loads(payload["metadata_json"])
