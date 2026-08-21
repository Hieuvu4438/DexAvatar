import json
from pathlib import Path

from signal4d.cli.evaluate import run as evaluate
from signal4d.cli.run_pipeline import run as run_pipeline
from signal4d.scripts.synthetic import create_synthetic_artifact


def test_synthetic_m0_end_to_end(tmp_path) -> None:
    manifest = create_synthetic_artifact(tmp_path / "artifact", num_clips=2, frames=12, joints=9)
    package = Path(__file__).resolve().parents[2]
    run_root = tmp_path / "run"
    run_pipeline(
        str(package / "configs/method/m0.yaml"),
        str(manifest),
        str(tmp_path / "artifact/cache"),
        str(run_root),
    )
    evaluate(str(manifest), str(run_root / "predictions"), str(run_root / "evaluation"))
    rows = json.loads((run_root / "evaluation/metrics.json").read_text())
    assert len(rows) == 2
    assert all(row["tr_v2v_body_mm"] < 30 for row in rows)
