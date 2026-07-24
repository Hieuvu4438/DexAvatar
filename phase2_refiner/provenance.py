"""Reproducibility metadata shared by Phase 2 commands."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path


DEPENDENCIES = ("numpy", "torch", "scipy", "opencv-python", "PyYAML", "smplx")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(repository: str | Path = ".") -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def dependency_versions() -> dict[str, str]:
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def run_provenance(config_path: str | Path, seed: int) -> dict:
    path = Path(config_path).resolve()
    return {
        "config": str(path),
        "config_sha256": sha256_file(path),
        "git_sha": git_sha(path.parent),
        "dependencies": dependency_versions(),
        "seed": int(seed),
    }
