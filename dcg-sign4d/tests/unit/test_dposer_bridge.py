import json
from pathlib import Path

import pytest
import torch

from dcg_sign4d.diffusion.dposer_bridge import REQUIRED_RUNTIME_FILES, validate_runtime_registry
from dcg_sign4d.diffusion.dposer_normalizer import DPoserXWholeBodyNormalizer, ZScoreStats
from dcg_sign4d.utils.hashing import file_sha256


def _normalizer():
    return DPoserXWholeBodyNormalizer(
        {
            name: ZScoreStats(torch.arange(size, dtype=torch.float32), torch.ones(size) * 2)
            for name, size in DPoserXWholeBodyNormalizer.PARTS
        }
    )


def test_dposer_normalizer_round_trip_and_left_reflection():
    normalizer = _normalizer()
    parts = {name: torch.randn(2, size) for name, size in DPoserXWholeBodyNormalizer.PARTS}
    restored = normalizer.denormalize_parts(normalizer.normalize_parts(parts))
    for name in parts:
        torch.testing.assert_close(restored[name], parts[name], atol=1e-5, rtol=1e-5)


def test_dposer_registry_requires_exact_hash_pinned_set(tmp_path: Path):
    entries = []
    for relative in REQUIRED_RUNTIME_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        entries.append({"path": relative, "sha256": file_sha256(path)})
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"files": entries}), encoding="utf-8")
    validate_runtime_registry(tmp_path, registry)
    entries[0]["sha256"] = "0" * 64
    registry.write_text(json.dumps({"files": entries}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_runtime_registry(tmp_path, registry)
