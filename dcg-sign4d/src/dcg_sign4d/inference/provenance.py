"""Strict prediction-run provenance collection and validation."""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from dcg_sign4d.utils.hashing import canonical_hash, file_sha256

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _git(scope: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(scope), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _source_snapshot(scope: Path) -> tuple[str, list[dict[str, str]]]:
    candidates = [scope / "pyproject.toml", scope / "README.md"]
    candidates.extend(sorted((scope / "src").rglob("*.py")))
    candidates.extend(sorted((scope / "configs").rglob("*.yaml")))
    candidates.extend(sorted((scope / "configs").rglob("*.json")))
    rows = [
        {"path": str(path.relative_to(scope)), "sha256": file_sha256(path)}
        for path in candidates
        if path.is_file()
    ]
    return canonical_hash(rows), rows


def build_run_identity(
    *,
    scope_root: str | Path,
    config_path: str | Path,
    manifest_path: str | Path,
    dependency_commits: dict[str, str],
    checkpoint_sha256: dict[str, str],
    sampler: dict[str, int | float],
    started_at_utc: str,
    ended_at_utc: str,
    peak_memory_bytes: int,
    frame_count: int,
    execution_device: str,
    development_only: bool,
) -> dict[str, Any]:
    """Collect reproducible identity without embedding a potentially sensitive full diff."""

    scope = Path(scope_root).resolve()
    repository = Path(_git(scope, "rev-parse", "--show-toplevel").strip())
    relative_scope = scope.relative_to(repository)
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
        "--",
        str(relative_scope),
    )
    diff = _git(repository, "diff", "--binary", "HEAD", "--", str(relative_scope))
    snapshot_hash, snapshot_rows = _source_snapshot(scope)
    environment = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    started = datetime.fromisoformat(started_at_utc.replace("Z", "+00:00"))
    ended = datetime.fromisoformat(ended_at_utc.replace("Z", "+00:00"))
    elapsed = (ended - started).total_seconds()
    if frame_count < 1 or elapsed < 0:
        raise ValueError("invalid frame count or run timestamps")
    use_cuda = execution_device.startswith("cuda")
    identity = {
        "schema_version": "dcg_run_identity_v1",
        "development_only": development_only,
        "git_commit": _git(repository, "rev-parse", "HEAD").strip(),
        "dirty_worktree": bool(status),
        "diff_sha256": hashlib.sha256((status + "\n" + diff).encode()).hexdigest(),
        "source_snapshot_sha256": snapshot_hash,
        "source_file_count": len(snapshot_rows),
        "config_sha256": file_sha256(config_path),
        "manifest_sha256": file_sha256(manifest_path),
        "dependency_commits": dependency_commits,
        "checkpoint_sha256": checkpoint_sha256,
        "environment_lock_sha256": hashlib.sha256(environment.encode()).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "execution_device": execution_device,
        "hardware": torch.cuda.get_device_name(0) if use_cuda else platform.machine(),
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "elapsed_seconds": elapsed,
        "frame_count": frame_count,
        "seconds_per_frame": elapsed / frame_count,
        "peak_memory_bytes": peak_memory_bytes,
        "sampler": sampler,
    }
    validate_run_identity(identity)
    return identity


def validate_run_identity(identity: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "development_only",
        "git_commit",
        "dirty_worktree",
        "diff_sha256",
        "source_snapshot_sha256",
        "source_file_count",
        "config_sha256",
        "manifest_sha256",
        "dependency_commits",
        "checkpoint_sha256",
        "environment_lock_sha256",
        "python",
        "torch",
        "cuda_version",
        "execution_device",
        "hardware",
        "started_at_utc",
        "ended_at_utc",
        "elapsed_seconds",
        "frame_count",
        "seconds_per_frame",
        "peak_memory_bytes",
        "sampler",
    }
    missing = sorted(required - identity.keys())
    if missing:
        raise ValueError(f"run identity is missing required fields: {missing}")
    if identity["schema_version"] != "dcg_run_identity_v1":
        raise ValueError("unknown run identity schema")
    if not _COMMIT.fullmatch(identity["git_commit"]):
        raise ValueError("git_commit must be an exact 40-hex identity")
    for name in (
        "diff_sha256",
        "source_snapshot_sha256",
        "config_sha256",
        "manifest_sha256",
        "environment_lock_sha256",
    ):
        if not _SHA256.fullmatch(identity[name]):
            raise ValueError(f"{name} must be an exact SHA-256")
    for mapping_name, pattern in (
        ("dependency_commits", _COMMIT),
        ("checkpoint_sha256", _SHA256),
    ):
        mapping = identity[mapping_name]
        if not isinstance(mapping, dict):
            raise ValueError(f"{mapping_name} must be a mapping")
        if not all(name and pattern.fullmatch(value) for name, value in mapping.items()):
            raise ValueError(f"invalid identity in {mapping_name}")
    if not identity["development_only"] and not identity["checkpoint_sha256"]:
        raise ValueError("production run identity requires model checkpoint hashes")
    parsed_times = []
    for name in ("started_at_utc", "ended_at_utc"):
        parsed = datetime.fromisoformat(identity[name].replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware")
        parsed_times.append(parsed)
    expected_elapsed = (parsed_times[1] - parsed_times[0]).total_seconds()
    if expected_elapsed < 0 or abs(identity["elapsed_seconds"] - expected_elapsed) > 1e-9:
        raise ValueError("run elapsed time is inconsistent with timestamps")
    if identity["frame_count"] < 1:
        raise ValueError("run frame_count must be positive")
    if abs(identity["seconds_per_frame"] - expected_elapsed / identity["frame_count"]) > 1e-9:
        raise ValueError("seconds_per_frame is inconsistent")
    if identity["peak_memory_bytes"] < 0 or identity["source_file_count"] < 1:
        raise ValueError("invalid run resource/source counts")
    sampler = identity["sampler"]
    for name in ("diffusion_steps", "rounds", "num_hypotheses"):
        if name not in sampler or int(sampler[name]) < 1:
            raise ValueError(f"sampler.{name} must be positive")
