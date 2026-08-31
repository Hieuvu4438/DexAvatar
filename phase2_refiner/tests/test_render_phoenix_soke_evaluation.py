import pytest

from scripts.render_phoenix_soke_evaluation import _write_report, render


def test_render_phoenix_soke_evaluation_reports_locked_verdict() -> None:
    payload = {
        "schema": "signal4d-phoenix-soke-pampjpe-v1",
        "mode": "final_test_evaluation",
        "clips": 642,
        "frames": 63201,
        "metrics": {
            "initializer": {"body_pa_mpjpe_mm": 30.0, "hand_pa_mpjpe_mm": 8.0},
            "transformer_always": {
                "body_pa_mpjpe_mm": 26.0,
                "hand_pa_mpjpe_mm": 7.0,
            },
            "transformer_gated": {
                "body_pa_mpjpe_mm": 25.0,
                "hand_pa_mpjpe_mm": 6.5,
            },
        },
        "soke_table3_reference": {
            "phoenix_body_pa_mpjpe_mm": 25.79,
            "phoenix_hand_pa_mpjpe_mm": 6.78,
        },
        "comparison": {
            "beats_soke_body": True,
            "beats_soke_hand": True,
            "beats_soke_both": True,
        },
        "metric_definition": {
            "body": "body contract",
            "hand": "hand contract",
            "aggregation": "frame micro",
            "decoder": "fixed decoder",
        },
        "protocol_difference": "Different training protocols.",
        "manifest": "/test.json",
        "manifest_sha256": "a" * 64,
        "checkpoint": "/best.pt",
        "checkpoint_sha256": "b" * 64,
        "calibration": "/calibration.json",
        "calibration_sha256": "c" * 64,
        "config": "/config.yaml",
        "config_sha256": "d" * 64,
    }
    report = render(payload)
    assert "BEATS SOKE ON BOTH REPORTED REGIONS" in report
    assert "| Transformer, dev-calibrated gate | 25.0000 | 6.5000" in report
    assert "642 clips / 63,201 frames" in report
    assert "metric-compatible comparison, not an identical training task" in report


def test_rendered_report_is_published_atomically(tmp_path) -> None:
    output = tmp_path / "report.md"
    _write_report(output, "complete report")
    assert output.read_text() == "complete report"
    assert list(tmp_path.glob(".report.md.tmp.*")) == []
    with pytest.raises(FileExistsError):
        _write_report(output, "overwrite")
