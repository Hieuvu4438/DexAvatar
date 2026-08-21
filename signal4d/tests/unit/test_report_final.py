import csv
import json

from signal4d.cli.report_final import run


def test_report_final_is_generated_from_evaluator_summaries(tmp_path) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    summary = {
        "tr_v2v_upper_body_mm": 10.0,
        "tr_v2v_left_hand_mm": 8.0,
        "tr_v2v_right_hand_mm": 9.0,
        "velocity_error": 1.0,
        "acceleration_error": 2.0,
        "jerk_error": 3.0,
        "coverage": 1.0,
    }
    (evaluation / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with (evaluation / "per_clip.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frames", *summary])
        writer.writeheader()
        writer.writerow({"frames": 20, **summary})
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps({"delta": -1.0}), encoding="utf-8")
    slices = tmp_path / "slices.json"
    slices.write_text(
        json.dumps({"slices": {"short": {"operator": "<=", "value": 24}}}),
        encoding="utf-8",
    )

    report = run(
        [f"candidate={evaluation}"],
        [f"left={comparison}"],
        str(tmp_path / "report"),
        str(slices),
    )

    assert report["status"] == "generated_from_raw_evaluator_outputs"
    assert "candidate,10.0,8.0,9.0" in (tmp_path / "report/primary_metrics.csv").read_text()
    assert "<svg" in (tmp_path / "report/primary_geometry.svg").read_text()
    assert "candidate,short,1" in (tmp_path / "report/stress_slices.csv").read_text()
