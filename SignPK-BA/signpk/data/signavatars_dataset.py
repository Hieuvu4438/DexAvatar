from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from signpk.models.explicit_tokens import ExplicitTokenBatch


FORBIDDEN_BENCHMARK_MARKERS = ("evaluation_from_author", "smplx_gt", "sgnify")
WINDOW_SCHEMA = "signpk-training-window-v1"


class SignAvatarWindowDataset(Dataset):
    """Leakage-audited, cache-only PKC training dataset.

    Observer extraction and training are intentionally decoupled. Each JSONL
    row must contain ``cache_path``, ``signer_id``, ``sequence_id``, ``split``,
    ``quality_weight`` and ``source_dataset``.
    """

    def __init__(self, index_path: str | Path, split: str = "train"):
        self.index_path = Path(index_path).resolve()
        rows = [
            json.loads(line)
            for line in self.index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        signer_splits: dict[str, set[str]] = {}
        for row in rows:
            required = {
                "cache_path",
                "signer_id",
                "sequence_id",
                "split",
                "quality_weight",
                "source_dataset",
            }
            missing = required - set(row)
            if missing:
                raise KeyError(f"training index row missing {sorted(missing)}")
            signer_splits.setdefault(str(row["signer_id"]), set()).add(str(row["split"]))
            resolved = self._resolve(row["cache_path"])
            if any(marker in str(resolved).lower() for marker in FORBIDDEN_BENCHMARK_MARKERS):
                raise ValueError(f"benchmark leakage in PKC cache path: {resolved}")
            if not 0 < float(row["quality_weight"]) <= 1:
                raise ValueError("quality_weight must be in (0,1]")
        leaked_signers = sorted(name for name, splits in signer_splits.items() if len(splits) > 1)
        if leaked_signers:
            raise ValueError(f"signers occur in multiple splits: {leaked_signers[:8]}")
        self.rows = [row for row in rows if row["split"] == split]
        if not self.rows:
            raise ValueError(f"no {split!r} windows in {self.index_path}")

    def _resolve(self, path: str | Path) -> Path:
        path = Path(path).expanduser()
        return path.resolve() if path.is_absolute() else (self.index_path.parent / path).resolve()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = self._resolve(row["cache_path"])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != WINDOW_SCHEMA:
            raise ValueError(f"unsupported training cache schema at {path}")
        payload = dict(payload)
        payload["quality_weight"] = torch.tensor(float(row["quality_weight"]), dtype=torch.float32)
        payload["sequence_id"] = str(row["sequence_id"])
        return payload


def tokens_from_batch(batch: dict[str, Any], device: torch.device) -> ExplicitTokenBatch:
    token_fields = {
        "body",
        "left",
        "right",
        "relation",
        "timestamps",
        "upper_base_rotmat",
        "left_base_rotmat",
        "right_base_rotmat",
        "left_valid",
        "right_valid",
        "disagreement",
    }
    values = {name: batch[name].to(device) for name in token_fields}
    for name in (
        "left_observer_feature",
        "right_observer_feature",
        "left_h4w_feature",
        "right_h4w_feature",
        "body_observer_feature",
    ):
        if name in batch and batch[name] is not None:
            values[name] = batch[name].to(device)
    tokens = ExplicitTokenBatch(**values)
    tokens.validate()
    return tokens
