from __future__ import annotations

from phase3_posterior.gates import stage_decision


def test_stage_gate_is_fail_closed() -> None:
    decision = stage_decision("g5", {})
    assert not decision["passed"]


def test_g3_go_boundary() -> None:
    decision = stage_decision(
        "g3",
        {
            "recovery": {"ubody": 0.30, "lhand": 0.31, "rhand": 0.32},
            "max_clean_regression": 0.009,
        },
    )
    assert decision["passed"]
