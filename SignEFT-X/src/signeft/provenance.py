from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess

import torch
import yaml

from signeft.io_utils import atomic_write_json, sha256_file, tree_sha256


def _git(path: Path) -> dict[str, object]:
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(path), "status", "--short"], text=True)
    return {"path": str(path.resolve()), "commit": commit, "dirty": bool(status), "status": status.splitlines()}


def verify_dependency_lock(lock_path: Path, required: set[str]) -> dict[str, object]:
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    base = lock_path.parent
    checked = {}
    for name in required:
        if name in lock.get("repositories", {}):
            item = lock["repositories"][name]
            path = (base / item["path"]).resolve()
            actual = _git(path)
            if actual["commit"] != item["commit"]:
                raise RuntimeError(f"repository lock mismatch: {name}")
            checked[name] = {"kind": "repository", **actual}
        elif name in lock.get("checkpoints", {}):
            item = lock["checkpoints"][name]
            path = (base / item["path"]).resolve()
            digest = sha256_file(path)
            if digest != item["sha256"]:
                raise RuntimeError(f"checkpoint lock mismatch: {name}: {digest}")
            checked[name] = {
                "kind": "checkpoint", "path": str(path), "sha256": digest,
                "bytes": path.stat().st_size,
            }
        else:
            raise KeyError(f"dependency is not locked: {name}")
    return checked


def capture_provenance(
    output: Path,
    repository: Path,
    lock_path: Path,
    config: Path,
    manifest: Path,
    evaluator: Path,
    required_dependencies: set[str],
) -> dict[str, object]:
    dependencies = verify_dependency_lock(lock_path, required_dependencies)
    packages = {}
    for name in ("torch", "torchvision", "numpy", "scipy", "smplx", "opencv-python", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "schema_version": "signeft.provenance.v1",
        "repository": _git(repository),
        "implementation_tree_sha256": tree_sha256(repository / "src" / "signeft"),
        "dependency_lock": str(lock_path.resolve()),
        "dependency_lock_sha256": sha256_file(lock_path),
        "dependencies": dependencies,
        "config": {"path": str(config.resolve()), "sha256": sha256_file(config)},
        "manifest": {"path": str(manifest.resolve()), "sha256": sha256_file(manifest)},
        "evaluator": {
            "path": str(evaluator.resolve()), "sha256": sha256_file(evaluator),
            "read_only": not bool(evaluator.stat().st_mode & 0o222),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "pid": os.getpid(),
        },
        "fit_contract": {
            "ground_truth_path_available_to_fit": False,
            "temporal_pose_term": False,
            "frame_independent": True,
        },
    }
    atomic_write_json(output, report)
    return report

