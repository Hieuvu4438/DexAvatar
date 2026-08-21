from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..utils.hashing import verify_sha256


def load_numpy(path: str | Path, expected_sha256: str | None = None) -> Any:
    path = Path(path)
    if expected_sha256:
        verify_sha256(path, expected_sha256)
    return np.load(path, allow_pickle=False)


def load_torch_weights(path: str | Path, expected_sha256: str) -> Any:
    verify_sha256(path, expected_sha256)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("Installed PyTorch does not support safe weights_only loading") from exc


def load_json(path: str | Path, expected_sha256: str | None = None) -> Any:
    if expected_sha256:
        verify_sha256(path, expected_sha256)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_trusted_legacy_pickle(
    path: str | Path, expected_sha256: str, trusted: bool = False
) -> Any:
    """One-time legacy converter boundary; never call without an explicit trust decision."""
    if not trusted:
        raise PermissionError("Legacy pickle loading requires trusted=True and an expected SHA-256")
    verify_sha256(path, expected_sha256)
    with Path(path).open("rb") as handle:
        return pickle.load(handle, encoding="latin1")
