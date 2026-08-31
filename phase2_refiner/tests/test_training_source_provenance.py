from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import phase2_refiner.train as train_module
from phase2_refiner.train import TRAINING_SOURCE_FILES, training_source_hashes


def test_training_source_provenance_exactly_covers_declared_files() -> None:
    hashes = training_source_hashes()
    assert set(hashes) == set(TRAINING_SOURCE_FILES)
    assert all(len(value) == 64 for value in hashes.values())
    assert "phase2_refiner/train.py" in hashes
    assert "phase2_refiner/models/spatial_temporal_refiner.py" in hashes
    assert "phase2_refiner/losses/sequence.py" in hashes


def test_atomic_torch_save_publishes_readable_checkpoint(tmp_path: Path) -> None:
    destination = tmp_path / "last.pt"
    train_module._atomic_torch_save({"step": 7}, destination)
    assert torch.load(destination, weights_only=False)["step"] == 7
    assert list(tmp_path.glob(".last.pt.tmp.*")) == []


def test_atomic_torch_save_preserves_prior_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "best.pt"
    destination.write_bytes(b"prior")

    def fail_after_partial_write(payload: dict, path: Path) -> None:
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr(train_module.torch, "save", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated"):
        train_module._atomic_torch_save({"step": 8}, destination)
    assert destination.read_bytes() == b"prior"
    assert list(tmp_path.glob(".best.pt.tmp.*")) == []


def test_checkpoint_persists_early_stopping_counter() -> None:
    state = train_module._checkpoint(
        model=SimpleNamespace(state_dict=lambda: {"weight": torch.tensor(1.0)}),
        ema=SimpleNamespace(shadow={"weight": torch.tensor(1.0)}),
        optimizer=SimpleNamespace(state_dict=lambda: {"optimizer": True}),
        scheduler=SimpleNamespace(state_dict=lambda: {"scheduler": True}),
        config={"model": {}},
        provenance={"test": True},
        step=4000,
        micro_step=8000,
        best=0.25,
        validations_without_improvement=3,
        train_dataset=SimpleNamespace(rng=np.random.default_rng(42)),
        loader_generator=torch.Generator().manual_seed(42),
        batch_sampler=None,
    )
    assert state["step"] == 4000
    assert state["validations_without_improvement"] == 3
