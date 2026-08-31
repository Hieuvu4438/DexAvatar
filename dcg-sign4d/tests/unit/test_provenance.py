from copy import deepcopy

import pytest

from dcg_sign4d.inference.provenance import validate_run_identity


def identity():
    return {
        "schema_version": "dcg_run_identity_v1",
        "development_only": True,
        "git_commit": "a" * 40,
        "dirty_worktree": False,
        "diff_sha256": "b" * 64,
        "source_snapshot_sha256": "c" * 64,
        "source_file_count": 1,
        "config_sha256": "d" * 64,
        "manifest_sha256": "e" * 64,
        "dependency_commits": {"source": "f" * 40},
        "checkpoint_sha256": {},
        "environment_lock_sha256": "1" * 64,
        "python": "3.13",
        "torch": "2.11",
        "cuda_version": None,
        "execution_device": "cpu",
        "hardware": "fixture",
        "started_at_utc": "2026-08-23T00:00:00+00:00",
        "ended_at_utc": "2026-08-23T00:00:02+00:00",
        "elapsed_seconds": 2.0,
        "frame_count": 4,
        "seconds_per_frame": 0.5,
        "peak_memory_bytes": 0,
        "sampler": {"diffusion_steps": 4, "rounds": 1, "num_hypotheses": 1},
    }


def test_development_identity_validates_but_production_requires_checkpoint():
    value = identity()
    validate_run_identity(value)
    production = {**value, "development_only": False}
    with pytest.raises(ValueError, match="requires model checkpoint hashes"):
        validate_run_identity(production)
    production["checkpoint_sha256"] = {"model": "2" * 64}
    validate_run_identity(production)


def test_timing_and_sampler_inconsistency_fail_closed():
    wrong_time = deepcopy(identity())
    wrong_time["seconds_per_frame"] = 1.0
    with pytest.raises(ValueError, match="seconds_per_frame is inconsistent"):
        validate_run_identity(wrong_time)
    wrong_sampler = deepcopy(identity())
    wrong_sampler["sampler"]["rounds"] = 0
    with pytest.raises(ValueError, match="sampler.rounds must be positive"):
        validate_run_identity(wrong_sampler)
