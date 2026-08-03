"""Hash-bound provenance and non-destructive output helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any


DEPENDENCIES = ("numpy", "torch", "PyYAML", "scipy", "smplx")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_state(repository: str | Path = ".") -> dict[str, Any]:
    root = Path(repository).resolve()
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "sha": sha.stdout.strip() if sha.returncode == 0 else "unknown",
        "dirty": bool(status.stdout.strip()),
        "status": status.stdout.splitlines(),
    }


def dependency_versions() -> dict[str, str]:
    result = {}
    for name in DEPENDENCIES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def require_new_output(path: str | Path) -> Path:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def atomic_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(target)


def run_provenance(
    config_path: str | Path,
    seed: int,
    inputs: list[str | Path] | None = None,
    repository: str | Path = ".",
) -> dict[str, Any]:
    config = Path(config_path).resolve()
    hashes = {}
    for value in inputs or []:
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[str(path)] = sha256_file(path)
    return {
        "config": str(config),
        "config_sha256": sha256_file(config),
        "git": git_state(repository),
        "dependencies": dependency_versions(),
        "seed": int(seed),
        "input_sha256": hashes,
    }
