"""Fail-closed source/checkpoint/license provenance audits."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from dcg_sign4d.utils.hashing import file_sha256


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def audit_third_party(root: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for expected in manifest["repositories"]:
        source = root / expected["name"]
        actual_commit = _git(source, "rev-parse", "HEAD")
        actual_url = _git(source, "remote", "get-url", "origin")
        license_path = source / expected["license_file"]
        actual_license_hash = file_sha256(license_path)
        clean = not bool(_git(source, "status", "--short"))
        row = {
            "name": expected["name"],
            "commit": actual_commit,
            "url": actual_url,
            "license_sha256": actual_license_hash,
            "clean": clean,
            "pass": actual_commit == expected["commit"]
            and actual_url == expected["url"]
            and actual_license_hash == expected["license_sha256"]
            and clean,
        }
        rows.append(row)
    return {
        "manifest_sha256": file_sha256(manifest_path),
        "pin_status": manifest["status"],
        "engineering_pass": all(row["pass"] for row in rows),
        "scientifically_frozen": manifest["status"] == "scientifically_frozen",
        "repositories": rows,
    }


def audit_dposer_runtime(root: str | Path, registry_path: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    registry_path = Path(registry_path).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    actual_commit = _git(root, "rev-parse", "HEAD")
    rows = []
    for expected in registry["files"]:
        path = root / expected["path"]
        actual_hash = file_sha256(path) if path.is_file() else None
        rows.append(
            {
                "path": expected["path"],
                "exists": path.is_file(),
                "sha256": actual_hash,
                "pass": actual_hash == expected["sha256"],
            }
        )
    return {
        "registry_sha256": file_sha256(registry_path),
        "source_commit": actual_commit,
        "source_commit_pass": actual_commit == registry["source_commit"],
        "checkpoint_hashes_pass": all(row["pass"] for row in rows),
        "scientifically_frozen": registry["status"] == "scientifically_frozen",
        "safe_load_status": "legacy_checkpoint_requires_explicit_trust_and_upstream_loader",
        "files": rows,
    }
