from pathlib import Path

import numpy as np

from signpccx.evaluation.audited import paired_sign_bootstrap, translation_aligned_error


def test_audited_error_is_translation_invariant():
    target = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    assert np.max(translation_aligned_error(target + [20, -5, 7], target)) < 1e-12


def test_paired_bootstrap_is_deterministic(tmp_path: Path):
    header = "sign,tr all,tr above pelvis upper body,tr above pelvis minus face,tr left hand,tr right hand\n"
    baseline = header + "a,2,3,4,5,6\n" + "b,4,5,6,,8\n"
    candidate = header + "a,1,2,3,4,5\n" + "b,3,4,5,,7\n"
    (tmp_path / "baseline.csv").write_text(baseline)
    (tmp_path / "candidate.csv").write_text(candidate)
    first = paired_sign_bootstrap(tmp_path / "candidate.csv", tmp_path / "baseline.csv", tmp_path / "a.json", 100, 7)
    second = paired_sign_bootstrap(tmp_path / "candidate.csv", tmp_path / "baseline.csv", tmp_path / "b.json", 100, 7)
    assert first["results"] == second["results"]
    assert all(item["mean_sign_delta_mm"] == -1 for item in first["results"])
