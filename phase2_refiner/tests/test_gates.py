from phase2_refiner.gates import g4_decision, g6_decision


def _seed(prediction=(9.0, 9.0, 9.0), baseline=(10.0, 10.0, 10.0)):
    regions = ("ubody", "lhand", "rhand")
    return {
        "frames": 1493,
        "prediction": dict(zip(regions, prediction)),
        "baseline": dict(zip(regions, baseline)),
        "paired_bootstrap": {
            region: {"ci95_high_mm": -0.1} for region in regions
        },
    }


def test_g6_requires_all_contract_checks():
    payload = {
        "expected_frames": 1493,
        "seeds": [_seed(), _seed((9.05, 9.0, 8.95)), _seed((8.95, 9.0, 9.05))],
        "diagnostics": {
            "hard_subset_relative_gain": 0.09,
            "clean_regression_fraction": {"ubody": 0.0, "lhand": 0.0, "rhand": 0.0},
            "group_frame_fallback_fraction": 0.0,
        },
    }
    assert g6_decision(payload)["passed"]
    payload["diagnostics"]["hard_subset_relative_gain"] = 0.079
    decision = g6_decision(payload)
    assert not decision["passed"]
    assert not decision["checks"]["hard_subset_gain_at_least_8pct"]


def test_g6_refuses_fewer_than_three_seeds():
    decision = g6_decision({"seeds": [_seed()]})
    assert not decision["passed"]
    assert not decision["checks"]["exactly_three_seeds"]


def test_g4_requires_external_real_residual_evidence():
    payload = {
        "prediction": {"ubody": 9.0, "lhand": 9.0, "rhand": 9.0},
        "baseline": {"ubody": 10.0, "lhand": 10.0, "rhand": 10.0},
        "frames": 100,
        "expected_frames": 100,
        "hard_subset_relative_gain": 0.1,
        "source_disjoint_verified": False,
        "real_residual_audit_passed": False,
    }
    decision = g4_decision(payload)
    assert not decision["passed"]
    payload["source_disjoint_verified"] = True
    payload["real_residual_audit_passed"] = True
    assert g4_decision(payload)["passed"]
