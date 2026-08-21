import json
from pathlib import Path

from signal4d.cli import render_confirmatory_report


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_render_confirmatory_report_uses_frozen_artifacts(tmp_path: Path) -> None:
    metrics = {
        "tr_v2v_left_hand_mm": 10.0,
        "tr_v2v_upper_body_mm": 20.0,
        "tr_v2v_right_hand_mm": 11.0,
        "velocity_error": 1.0,
        "acceleration_error": 2.0,
        "jerk_error": 3.0,
    }
    baseline = {"clips": 56, "frames": 769, **metrics}
    candidate = {"clips": 56, "frames": 769, **{k: v - 0.5 for k, v in metrics.items()}}
    comparisons = {
        key: {"candidate_minus_baseline": -0.5, "ci_low": -0.8, "ci_high": -0.2}
        for key in metrics
    }
    decision = {"decision": "PASS", "checks": {"complete": True}, "comparisons": comparisons}
    gate = {
        "oof_clip_macro_delta_mm": -0.9,
        "oof_ci95_clip_bootstrap_mm": [-1.2, -0.6],
        "oof_switches": 0,
    }
    release = {"status": "frozen_before_confirmatory_test"}
    for name, value in (
        ("decision.json", decision),
        ("baseline.json", baseline),
        ("candidate.json", candidate),
        ("gate.json", gate),
        ("release.json", release),
    ):
        _write(tmp_path / name, value)
    output = tmp_path / "report.md"
    result = render_confirmatory_report.run(
        str(tmp_path / "decision.json"),
        str(tmp_path / "baseline.json"),
        str(tmp_path / "candidate.json"),
        str(tmp_path / "gate.json"),
        str(tmp_path / "release.json"),
        str(output),
    )
    assert result["decision"] == "PASS"
    text = output.read_text(encoding="utf-8")
    assert "**Decision: PASS**" in text
    assert "56 clips and 769" in text
    assert "temporally disjoint" in text
