from copy import deepcopy

from phase2_refiner.audit_completion import audit_completion


DIGEST = "a" * 64


def _passing_documents():
    g0 = {
        "frames": 1493,
        "signs": 57,
        "manifest_sha256": DIGEST,
        "prediction": {"ubody": 30.0, "lhand": 14.0, "rhand": 13.0},
    }
    regional = {
        region: {
            "mean_delta_mm": -1.0,
            "ci95_high_mm": -0.1,
        }
        for region in ("ubody", "lhand", "rhand")
    }
    g1 = {
        "frames": 1493,
        "signs": 57,
        "manifest_sha256": DIGEST,
        "baseline": {"ubody": 30.0, "lhand": 14.0, "rhand": 13.0},
        "prediction": {"ubody": 29.0, "lhand": 13.0, "rhand": 12.0},
        "paired_bootstrap": regional,
    }
    g2 = {
        "gates": {"volume": True, "integrity": True},
        "sign_domain_training_volume": {"clips": 10_000, "frames": 320_000},
        "train": {
            "fraction_clips_at_least_16": 1.0,
            "complete_body_and_both_hand_fraction": 1.0,
        },
        "train_validation_clip_overlap": [],
        "train_validation_source_group_overlap": [],
    }
    g3 = {
        "gates": {
            "G3": True,
            "regional_recovery_at_least_30_percent": True,
            "clean_to_injected_below_2_percent": True,
        },
        "translation_centered_per_region": True,
    }
    formal = {
        "contract_version": "phase2r-formal-v1",
        "passed": True,
        "checks": {"clips": True, "sources": True, "signers": True},
    }
    g4 = {"passed": True}
    causal = {
        "feedback_intervention_improves_corrupt_reconstruction": True,
        "feedback_intervention_clean_regression_at_most_1pct": True,
    }
    g5 = {
        "gate": {"passed": True, "checks": causal},
        "group_gates": {
            group: {"passed": True}
            for group in ("body", "left_hand", "right_hand")
        },
    }
    g6 = {"passed": True, "checks": {"exactly_three_seeds": True}}
    g7 = {"decision": "GO_BY_PROJECT_SCOPE"}
    return g0, g1, g2, g3, formal, g4, g5, g6, g7


def _audit(documents, prerequisites=None, runtime=None):
    return audit_completion(
        g0=documents[0],
        g1=documents[1],
        g2=documents[2],
        g3=documents[3],
        formal=documents[4],
        g4=documents[5],
        g5=documents[6],
        g6=documents[7],
        g7=documents[8],
        prerequisites=prerequisites
        or {
            "license": True,
            "annotations": True,
            "signers": True,
            "audit": True,
        },
        runtime=runtime or {"a1r": "COMPLETE", "mesh": "COMPLETE"},
    )


def test_completion_audit_requires_every_gate() -> None:
    report = _audit(_passing_documents())

    assert report["full_go"] is True
    assert report["decision"] == "FULL_GO"
    assert all(report["gates"].values())


def test_completion_audit_rejects_proxy_formal_data_and_old_noncausal_g5() -> None:
    documents = list(deepcopy(_passing_documents()))
    documents[4]["passed"] = False
    documents[6]["gate"]["checks"].pop(
        "feedback_intervention_improves_corrupt_reconstruction"
    )

    report = _audit(tuple(documents))

    assert report["full_go"] is False
    assert report["gates"]["G2"] is False
    assert report["gates"]["G4"] is False
    assert report["gates"]["G5"] is False
    assert report["supporting_checks"]["historical_g2_volume_integrity"] is True
    assert report["supporting_checks"]["formal_a1r_3d_target_contract"] is False


def test_completion_audit_records_external_and_runtime_prerequisites() -> None:
    report = _audit(
        _passing_documents(),
        prerequisites={"license": False, "annotations": False},
        runtime={"a1r": "WAITING_FOR_GPU", "mesh": "MISSING"},
    )

    assert report["unmet_prerequisites"] == ["annotations", "license"]
    assert report["runtime_waits"] == ["a1r", "mesh"]
    # Prerequisites explain reachability, but final completion remains a gate claim.
    assert report["full_go"] is True
