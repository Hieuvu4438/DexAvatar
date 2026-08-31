import pytest

from cusp_sl.config import CUSPConfig, DataConfig
from cusp_sl.train_deterministic import with_training_seed


def test_runtime_restart_seed_changes_only_training_section_seed():
    config = CUSPConfig(
        output_dir="out",
        data=DataConfig(train_manifest="train", val_manifest="val"),
    )
    changed = with_training_seed(config, 44)
    assert changed.training.seed == 44
    assert config.training.seed == 42
    assert changed.data == config.data
    assert changed.flow == config.flow
    assert changed.reliability == config.reliability


def test_runtime_restart_seed_must_be_non_negative():
    config = CUSPConfig(
        output_dir="out",
        data=DataConfig(train_manifest="train", val_manifest="val"),
    )
    with pytest.raises(ValueError, match="non-negative"):
        with_training_seed(config, -1)
