from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Iterable

import yaml

from signpccx.io import atomic_write_json, atomic_write_text, resolve_from_config, sha256_file
from signpccx.data.manifest import read_jsonl
from signpccx.model.canonicalizer import external_frame_paths
from signpccx.evaluation.official import OFFICIAL_EVALUATOR_SHA256
from signpccx.geometry.topology import load_canonical_faces, validate_faces_lock


def _run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=False, timeout=120
    )
    if result.returncode != 0:
        return f"UNAVAILABLE(returncode={result.returncode}): {result.stderr.strip()}"
    return result.stdout.strip()


def _tree_hash(paths: Iterable[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=str):
        relative = path.relative_to(base.resolve()) if path.is_relative_to(base.resolve()) else path
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _git_record(path: Path) -> dict[str, object]:
    commit = _run(["git", "rev-parse", "HEAD"], path)
    status = _run(["git", "status", "--short"], path)
    source_diff = _run(["git", "diff", "--", "*.py"], path)
    return {
        "path": str(path.resolve()),
        "commit": commit,
        "dirty": bool(status),
        "status": status.splitlines(),
        "tracked_python_diff_sha256": hashlib.sha256(source_diff.encode("utf-8")).hexdigest(),
    }


def record_provenance(config: dict) -> dict[str, object]:
    import torch

    config_path = Path(config["_config_path"]).resolve()
    package_root = next(
        parent for parent in (config_path.parent, *config_path.parents)
        if (parent / "pyproject.toml").is_file()
    )
    paths = {key: resolve_from_config(config, value) for key, value in config["paths"].items()}
    run_root = paths["run_root"]
    manifest_root = paths.get("manifest_root", run_root / "manifests")
    manifests = sorted(manifest_root.glob("*.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"No manifests under {run_root}")
    lock_path = package_root / "third_party.lock.yaml"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    repositories = {}
    for name, item in lock["repositories"].items():
        raw_path = item.get("path")
        if raw_path is None:
            repositories[name] = {**item, "installed": False}
            continue
        repository_path = (package_root / raw_path).resolve()
        repositories[name] = {
            **item,
            "installed": repository_path.exists(),
            "observed": _git_record(repository_path) if repository_path.exists() else None,
        }
    source_files = [
        path for root in (package_root / "src", package_root / "configs")
        for path in root.rglob("*") if path.is_file()
    ] + [
        package_root / "README.md", package_root / "RESULTS.md",
        package_root / "pyproject.toml", lock_path,
    ]
    model_candidates = {
        "canonical_model": paths.get("canonical_model"),
        "smplx_model": None if "smplx_model_root" not in paths else paths["smplx_model_root"] / "smplx" / "SMPLX_NEUTRAL.npz",
        "mano_smplx_vertex_ids": paths.get("mano_smplx_vertex_ids"),
    }
    if "evaluator_assets" in paths:
        for name in ("upper_body", "upper_body_minus_face", "upper_body_minus_head"):
            model_candidates[name] = (
                paths["evaluator_assets"] / "sgnify_part_segm_above_pelvis_joint" / f"{name}.npy"
            )
    model_hashes = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in model_candidates.items() if path is not None and path.is_file()
    }
    initializer_source = None
    if "external_v1_root" in paths:
        initializer_files = []
        for manifest in manifests:
            for frame in read_jsonl(manifest):
                source = external_frame_paths(paths["external_v1_root"], frame)
                initializer_files.extend((source.result_path, source.mesh_path))
        missing = [str(path) for path in initializer_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing initializer source artifact: {missing[0]}")
        initializer_source = {
            "root": str(paths["external_v1_root"].resolve()),
            "files": len(initializer_files),
            "sha256": _tree_hash(initializer_files, paths["external_v1_root"]),
        }
    pip_freeze = _run([sys.executable, "-m", "pip", "freeze"])
    conda = os.environ.get("CONDA_EXE")
    conda_explicit = "UNAVAILABLE: CONDA_EXE is not set"
    if conda:
        conda_explicit = _run([conda, "list", "--explicit"])
    environment_root = run_root / "environment"
    atomic_write_text(environment_root / "pip_freeze.txt", pip_freeze + "\n")
    atomic_write_text(environment_root / "conda_explicit.txt", conda_explicit + "\n")
    evaluator = paths["evaluator"]
    evaluator_mode = oct(evaluator.stat().st_mode & 0o777)
    record = {
        "schema_version": "signpccx.provenance.v1",
        "run_id": config["experiment"]["name"],
        "seed": config["experiment"].get("seed"),
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "resolved_sha256": hashlib.sha256(json.dumps(
                {key: value for key, value in config.items() if key != "_config_path"},
                sort_keys=True, separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
        },
        "source_tree_sha256": _tree_hash(source_files, package_root),
        "workspace_git": _git_record(package_root.parent),
        "third_party_lock_sha256": sha256_file(lock_path),
        "third_party": repositories,
        "artifacts": lock.get("artifacts", {}),
        "initializer_source_set": initializer_source,
        "data_hashes": {
            "signs": sha256_file(paths["signs_file"]),
            "segments": sha256_file(paths["segments_file"]),
            "manifest_set": _tree_hash(manifests, manifest_root),
        },
        "manifest_counts": {
            "signs": len(manifests),
            "frames": sum(1 for path in manifests for _ in path.open("r", encoding="utf-8")),
        },
        "evaluator": {
            "path": str(evaluator.resolve()),
            "sha256": sha256_file(evaluator),
            "mode": evaluator_mode,
        },
        "model_hashes": model_hashes,
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn": torch.backends.cudnn.version(),
            "gpu": _run([
                "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]).splitlines(),
        },
        "environment_files": {
            "pip_freeze": str((environment_root / "pip_freeze.txt").resolve()),
            "conda_explicit": str((environment_root / "conda_explicit.txt").resolve()),
        },
    }
    atomic_write_json(run_root / "provenance.json", record)
    return record


def doctor(config: dict) -> dict[str, object]:
    """Fail-fast environment/data audit required before a benchmark run."""
    import torch

    config_path = Path(config["_config_path"]).resolve()
    package_root = next(
        parent for parent in (config_path.parent, *config_path.parents)
        if (parent / "pyproject.toml").is_file()
    )
    paths = {key: resolve_from_config(config, value) for key, value in config["paths"].items()}
    run_root = paths["run_root"]
    manifest_root = paths.get("manifest_root", run_root / "manifests")
    manifests = sorted(manifest_root.glob("*.jsonl"))
    checks: dict[str, object] = {}
    required_paths = (
        "evaluator", "evaluator_assets", "canonical_model", "h4wpp_frame_cache",
        "smplx_model_root", "mano_smplx_vertex_ids",
    )
    missing = [name for name in required_paths if name not in paths or not paths[name].exists()]
    if missing:
        raise FileNotFoundError(f"doctor missing required paths: {missing}")
    if sha256_file(paths["evaluator"]) != OFFICIAL_EVALUATOR_SHA256:
        raise RuntimeError("doctor evaluator checksum mismatch")
    if paths["evaluator"].stat().st_mode & 0o222:
        raise PermissionError("official evaluator must be read-only")
    checks["evaluator"] = {"sha256": OFFICIAL_EVALUATOR_SHA256, "read_only": True}
    faces = load_canonical_faces(paths["canonical_model"])
    topology = config["topology"]
    validate_faces_lock(faces, topology["faces_sha256_int64"], int(topology["face_count"]))
    checks["topology"] = {
        "vertices": int(topology["vertex_count"]), "faces": int(len(faces)),
        "faces_sha256_int64": topology["faces_sha256_int64"],
    }
    if not manifests:
        raise FileNotFoundError(f"doctor found no manifests: {manifest_root}")
    records = [record for manifest in manifests for record in read_jsonl(manifest)]
    expected_signs = int(config["data"]["expected_signs"])
    expected_frames = int(config["data"]["expected_frames"])
    if (len(manifests), len(records)) != (expected_signs, expected_frames):
        raise RuntimeError(
            f"doctor manifest counts {len(manifests)}/{len(records)} != "
            f"{expected_signs}/{expected_frames}"
        )
    missing_teacher = [
        str(paths["h4wpp_frame_cache"] / "clips" / record.sign / f"{record.source_frame_id:06d}.npz")
        for record in records
        if not (paths["h4wpp_frame_cache"] / "clips" / record.sign / f"{record.source_frame_id:06d}.npz").is_file()
    ]
    if missing_teacher:
        raise FileNotFoundError(f"doctor missing H4W++ frames: {missing_teacher[:3]}")
    checks["data"] = {
        "signs": len(manifests), "frames": len(records), "teacher_frames": len(records),
        "manifest_summary_sha256": sha256_file(manifest_root / "summary.json"),
    }
    temporal = config.get("temporal", {})
    forbidden = [name for name in ("pose_smoothing", "velocity_loss", "acceleration_loss") if temporal.get(name)]
    if forbidden:
        raise RuntimeError(f"doctor temporal pose terms enabled: {forbidden}")
    checks["method"] = {"temporal_pose_terms": False, "method": config.get("method")}
    lock = yaml.safe_load((package_root / "third_party.lock.yaml").read_text(encoding="utf-8"))
    repositories = {}
    for name, item in lock["repositories"].items():
        raw_path = item.get("path")
        if raw_path is None:
            repositories[name] = {"installed": False, "optional": True}
            continue
        repository = (package_root / raw_path).resolve()
        observed = _run(["git", "rev-parse", "HEAD"], repository)
        if observed != item["commit"]:
            raise RuntimeError(f"doctor commit mismatch {name}: {observed} != {item['commit']}")
        repositories[name] = {"installed": True, "commit": observed}
    checks["repositories"] = repositories
    disk = shutil.disk_usage(run_root.parent if run_root.parent.exists() else package_root)
    checks["runtime"] = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(), "disk_free_gb": disk.free / 1024 ** 3,
    }
    report = {"schema_version": "signpccx.doctor.v1", "status": "ok", "checks": checks}
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_root / "doctor.json", report)
    return report
