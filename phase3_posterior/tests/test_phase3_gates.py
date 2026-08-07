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


def test_g0_requires_complete_cache_audit_not_only_legacy_fields() -> None:
    legacy = {
        "no_leakage": True,
        "disjoint": True,
        "hashes_licenses": True,
        "manual_failure_rate": 0.0,
    }
    assert not stage_decision("g0", legacy)["passed"]
    assert stage_decision("g0", {**legacy, "cache_audit_passed": True})["passed"]


def test_g2_requires_all_declared_relation_metrics() -> None:
    complete = {
        "relation_mae_gain": 0.10,
        "contact_f1": 0.65,
        "sign_contact_f1": 0.60,
        "depth_order_accuracy": 0.80,
        "contact_slip_gain": 0.15,
        "contact_slip_comparison_available": True,
        "max_region_regression": 0.01,
        "relation_only_reconstruction_unchanged": True,
    }
    assert stage_decision("g2", complete)["passed"]
    for key in complete:
        incomplete = dict(complete)
        incomplete.pop(key)
        assert not stage_decision("g2", incomplete)["passed"]
