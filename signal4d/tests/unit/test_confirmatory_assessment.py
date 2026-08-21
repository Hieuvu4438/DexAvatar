import json
from pathlib import Path

from signal4d.cli import assess_confirmatory, verify_tree


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_verify_tree_detects_bit_exact_and_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "clip").mkdir(parents=True)
    (second / "clip").mkdir(parents=True)
    (first / "clip" / "value.bin").write_bytes(b"same")
    (second / "clip" / "value.bin").write_bytes(b"same")
    report = verify_tree.run(str(first), str(second), str(tmp_path / "exact.json"))
    assert report["passed"] is True
    (second / "clip" / "value.bin").write_bytes(b"different")
    report = verify_tree.run(str(first), str(second), str(tmp_path / "different.json"))
    assert report["passed"] is False
    assert report["hash_mismatches"] == ["clip/value.bin"]


def test_assessment_requires_every_preregistered_gate(tmp_path: Path) -> None:
    candidate = {
        "clips": 56,
        "frames": 769,
        "coverage": 1.0,
        "velocity_error": 9.9,
        "acceleration_error": 19.9,
        "jerk_error": 29.9,
    }
    baseline = {
        "clips": 56,
        "frames": 769,
        "coverage": 1.0,
        "velocity_error": 10.0,
        "acceleration_error": 20.0,
        "jerk_error": 30.0,
    }
    _write(tmp_path / "candidate.json", candidate)
    _write(tmp_path / "baseline.json", baseline)
    root = tmp_path / "comparisons"
    values = {
        "tr_v2v_left_hand_mm": (-0.8, -0.2),
        "tr_v2v_upper_body_mm": (0.1, 0.4),
        "tr_v2v_right_hand_mm": (0.1, 0.4),
        "velocity_error": (-0.1, 0.1),
        "acceleration_error": (-0.1, 0.1),
        "jerk_error": (-0.1, 0.1),
    }
    for metric, (delta, high) in values.items():
        _write(
            root / f"{metric}.json",
            {
                "metric": metric,
                "candidate_minus_baseline": delta,
                "ci_low": -1.0,
                "ci_high": high,
            },
        )
    _write(tmp_path / "repro.json", {"passed": True})
    report = assess_confirmatory.run(
        str(tmp_path / "candidate.json"),
        str(tmp_path / "baseline.json"),
        str(root),
        str(tmp_path / "repro.json"),
        str(tmp_path / "decision.json"),
        56,
        769,
    )
    assert report["decision"] == "PASS"
    values["tr_v2v_left_hand_mm"] = (-0.4, -0.2)
    _write(
        root / "tr_v2v_left_hand_mm.json",
        {
            "metric": "tr_v2v_left_hand_mm",
            "candidate_minus_baseline": -0.4,
            "ci_low": -1.0,
            "ci_high": -0.2,
        },
    )
    report = assess_confirmatory.run(
        str(tmp_path / "candidate.json"),
        str(tmp_path / "baseline.json"),
        str(root),
        str(tmp_path / "repro.json"),
        str(tmp_path / "decision_fail.json"),
        56,
        769,
    )
    assert report["decision"] == "FAIL"
