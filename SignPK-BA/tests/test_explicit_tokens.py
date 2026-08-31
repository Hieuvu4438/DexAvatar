from dataclasses import fields, replace
from pathlib import Path

import torch

from signpk.data.frame_manifest import SignManifest
from signpk.models.explicit_tokens import ExplicitTokenBuilder
from signpk.observers.h4w_wrapper import load_h4w_cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
H4W_CACHE = PROJECT_ROOT.parent / "SignCAST/data/cache/v3/h4wpp"


def _slice(value, length):
    return type(value)(
        **{
            field.name: (
                None if getattr(value, field.name) is None else getattr(value, field.name)[:length]
            )
            for field in fields(value)
        }
    )


def test_explicit_tokens_use_real_h4w_geometry_and_omni_features():
    manifest = SignManifest.load(PROJECT_ROOT / "data/manifests/sgnify/Ablehnen/manifest.json")
    body, left, right, _ = load_h4w_cache(H4W_CACHE, manifest)
    length = 3
    body, left, right = _slice(body, length), _slice(left, length), _slice(right, length)
    left = replace(left, temporal_token=torch.randn(length, 1024))
    right = replace(right, temporal_token=torch.randn(length, 1024))
    tokens = ExplicitTokenBuilder().build(
        body,
        left,
        right,
        left,
        right,
        torch.zeros(length, 3),
        torch.tensor([record.timestamp_sec for record in manifest.records[:length]]),
    )
    assert tokens.body.shape == (1, length, 14, 12)
    assert tokens.left.shape == (1, length, 15, 54)
    assert tokens.left_observer_feature.shape == (1, length, 1024)
    assert torch.isfinite(tokens.body).all() and torch.isfinite(tokens.left).all()
