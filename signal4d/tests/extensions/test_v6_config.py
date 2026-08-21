from __future__ import annotations

import pytest

from signal4d.extensions.v6_uqdiff.config import V6Config
from signal4d.extensions.v6_uqdiff.joint_map import body_joint_indices


def _config() -> dict[str, object]:
    return {
        "method_name": "signal4d_v6_test",
        "base_method_config": "v5.yaml",
        "warm_start_root": "v5/predictions",
        "dposer": {
            "source_commit": "a" * 40,
            "checkpoint_registry": "registry.json",
        },
        "refinement": {"open_body_joints": ["left_elbow", "right_elbow"]},
    }


def test_named_joint_contract_resolves_smplx_order() -> None:
    assert body_joint_indices(["left_collar", "right_wrist"]) == (12, 20)


def test_config_exposes_resolved_open_indices() -> None:
    config = V6Config.model_validate(_config())
    assert config.open_body_indices == (17, 18)


@pytest.mark.parametrize(
    "names",
    [["left_elbow", "left_elbow"], ["not_a_joint"]],
)
def test_named_joint_contract_rejects_invalid_lists(names: list[str]) -> None:
    value = _config()
    value["refinement"] = {"open_body_joints": names}
    with pytest.raises(ValueError):
        V6Config.model_validate(value)

