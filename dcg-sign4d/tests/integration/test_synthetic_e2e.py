import json
from pathlib import Path

import pytest
import torch

from dcg_sign4d.geometry.patch_map import PatchMap
from dcg_sign4d.inference.artifacts import (
    validate_prediction_artifact,
    write_prediction_artifact,
)
from dcg_sign4d.initialization.trajectory_io import load_trajectory
from dcg_sign4d.synthetic import make_observations, make_state
from dcg_sign4d.synthetic_pipeline import build_smoke_reconstructor

ROOT = Path(__file__).resolve().parents[2]


def run_identity():
    return {
        "schema_version": "dcg_run_identity_v1",
        "development_only": True,
        "git_commit": "a" * 40,
        "dirty_worktree": True,
        "diff_sha256": "b" * 64,
        "source_snapshot_sha256": "c" * 64,
        "source_file_count": 1,
        "config_sha256": "d" * 64,
        "manifest_sha256": "e" * 64,
        "dependency_commits": {"fixture": "f" * 40},
        "checkpoint_sha256": {},
        "environment_lock_sha256": "1" * 64,
        "python": "fixture",
        "torch": "fixture",
        "cuda_version": None,
        "execution_device": "cpu",
        "hardware": "cpu",
        "started_at_utc": "2026-08-23T00:00:00+00:00",
        "ended_at_utc": "2026-08-23T00:00:01+00:00",
        "elapsed_seconds": 1.0,
        "frame_count": 6,
        "seconds_per_frame": 1 / 6,
        "peak_memory_bytes": 0,
        "sampler": {"diffusion_steps": 4, "rounds": 1, "num_hypotheses": 2},
    }


def test_alternating_k_hypotheses_and_artifacts(tmp_path):
    state = make_state(time=6)
    observations = make_observations(time=6)
    patch_map = PatchMap.load(ROOT / "assets/patch_maps/synthetic_smoke.yaml")
    reconstructor = build_smoke_reconstructor(
        state, patch_map, rounds=1, diffusion_steps=4, num_hypotheses=2
    )
    hypotheses = reconstructor.reconstruct(state, observations)
    assert len(hypotheses) == 2
    assert len({hypothesis.seed for hypothesis in hypotheses}) == 2
    assert not torch.equal(
        hypotheses[0].trajectory.root_translation,
        hypotheses[1].trajectory.root_translation,
    )
    artifact = write_prediction_artifact(
        tmp_path,
        "clip",
        (state, {"backend": "fixture"}),
        hypotheses,
        run_identity(),
        input_manifest={"clip_id": "clip"},
        observation_hashes={"fixture": "2" * 64},
        ranker_config={"fixture": True},
    )
    ranking = json.loads((artifact / "ranking.json").read_text())
    selected = int((artifact / "selected_hypothesis.txt").read_text())
    assert ranking["selected_hypothesis"] == selected
    restored, _ = load_trajectory(artifact / f"hypothesis_{selected:03d}")
    assert restored.root_translation.shape == state.root_translation.shape
    assert json.loads((artifact / "run_identity.json").read_text())["failure_count"] == 0
    assert json.loads((artifact / "run_identity.json").read_text())["retry_count"] == 0
    assert (artifact / "input_manifest.json").is_file()
    assert (artifact / "observation_hashes.json").is_file()
    for hypothesis in hypotheses:
        round_root = artifact / f"hypothesis_{hypothesis.identifier:03d}/rounds/round_000"
        assert (round_root / "trajectory.npz").is_file()
        assert (round_root / "contact_graph.npz").is_file()
        assert (round_root / "runtime_objective.json").is_file()
    validation = validate_prediction_artifact(artifact)
    assert validation["hypotheses"] == 2
    (artifact / "selected_hypothesis.txt").write_text("999\n")
    with pytest.raises(ValueError, match="selected hypothesis disagrees"):
        validate_prediction_artifact(artifact)
