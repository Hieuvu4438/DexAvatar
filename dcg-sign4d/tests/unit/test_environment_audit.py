import pathlib

import pytest

from dcg_sign4d.cli.audit_environment import audit

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_smoke_config_is_explicitly_development_only():
    result = audit(ROOT / "configs/smoke.yaml")
    assert result["development_only"] is True
    assert result["author_required_present"] is False


def test_production_config_fails_closed_on_author_required():
    with pytest.raises(ValueError, match="AUTHOR_REQUIRED"):
        audit(ROOT / "configs/inference/dcg_sign4d_v1.yaml")
