import pytest

from phase2_refiner.t5_optimize import _accepted_regions, validate_t5_config


def test_t5_config_enforces_proposal_step_bound() -> None:
    with pytest.raises(ValueError, match="proposal bound"):
        validate_t5_config(
            {
                "enabled": True,
                "steps": 21,
                "learning_rate": 0.03,
                "body_max_degrees": 3.0,
                "hand_max_degrees": 5.0,
            }
        )


def test_t5_group_acceptance_is_observation_only_and_fail_closed() -> None:
    accepted = _accepted_regions(
        {"ubody": 1.0, "lhand": 2.0, "rhand": None},
        {"ubody": 0.9, "lhand": 2.01, "rhand": None},
        minimum_gain=0.01,
        worsening_tolerance=0.001,
    )
    assert accepted == {"ubody": True, "lhand": False, "rhand": False}
