from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from signal4d.extensions.v6_uqdiff.dposer_bridge import (
    REQUIRED_RUNTIME_FILES,
    one_step_target,
    validate_runtime_registry,
)
from signal4d.extensions.v6_uqdiff.normalizer import WholeBodyAxisNormalizer, ZScoreStats
from signal4d.utils.hashing import sha256_file


def _normalizer() -> WholeBodyAxisNormalizer:
    return WholeBodyAxisNormalizer(
        {
            name: ZScoreStats(torch.arange(size, dtype=torch.float32), torch.ones(size) * 2)
            for name, size in WholeBodyAxisNormalizer.PARTS
        }
    )


def test_normalizer_round_trip_including_left_hand_reflection() -> None:
    normalizer = _normalizer()
    parts = {
        name: torch.randn(2, size) for name, size in WholeBodyAxisNormalizer.PARTS
    }
    restored = normalizer.denormalize_parts(normalizer.normalize_parts(parts))
    for name in parts:
        torch.testing.assert_close(restored[name], parts[name])


def test_one_step_target_matches_published_formula() -> None:
    perturbed = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    alpha = torch.tensor([[0.5], [0.25]])
    sigma_squared = torch.tensor([0.2, 0.4])
    score = torch.tensor([[2.0, -1.0], [1.0, 3.0]])
    expected = (perturbed + sigma_squared[:, None] * score) / alpha
    torch.testing.assert_close(one_step_target(perturbed, alpha, sigma_squared, score), expected)


def test_runtime_registry_requires_exact_file_set_and_hashes(tmp_path: Path) -> None:
    entries = []
    for relative in REQUIRED_RUNTIME_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        entries.append({"path": relative, "sha256": sha256_file(path)})
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"files": entries}), encoding="utf-8")
    validate_runtime_registry(tmp_path, registry_path)
    entries[0]["sha256"] = "0" * 64
    registry_path.write_text(json.dumps({"files": entries}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_runtime_registry(tmp_path, registry_path)

