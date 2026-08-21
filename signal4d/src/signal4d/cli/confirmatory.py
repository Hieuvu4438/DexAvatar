from __future__ import annotations

import json
from pathlib import Path

from ..utils.hashing import hash_path_tree, sha256_file, sha256_json
from . import evaluate_sgnify, fit_smplx


def run(
    freeze_file: str,
    config: str,
    manifest: str,
    cache_root: str,
    output_root: str,
    gt_root: str,
    gt_cache_root: str,
    model_path: str,
    upper_indices: str,
    left_indices: str,
    right_indices: str,
    device: str,
) -> dict[str, object]:
    freeze = json.loads(Path(freeze_file).read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_before_confirmatory_test":
        raise ValueError("release freeze is not valid for confirmatory evaluation")
    for frozen_path, expected_hash in freeze["configs"].items():
        if sha256_file(frozen_path) != expected_hash:
            raise ValueError(f"frozen config changed: {frozen_path}")
    for frozen_path, expected_hash in freeze["manifests"].items():
        if sha256_file(frozen_path) != expected_hash:
            raise ValueError(f"frozen manifest changed: {frozen_path}")
    for frozen_path, expected_tree in freeze["artifacts"].items():
        if hash_path_tree(frozen_path) != expected_tree:
            raise ValueError(f"frozen artifact tree changed: {frozen_path}")
    if freeze["configs"].get(config) != sha256_file(config):
        raise ValueError("config does not match release freeze")
    if freeze["manifests"].get(manifest) != sha256_file(manifest):
        raise ValueError("manifest does not match release freeze")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError("confirmatory output must be a new immutable directory")

    fit_smplx.run(config, manifest, cache_root, output_root, model_path, device)
    metrics = evaluate_sgnify.run(
        manifest_path=manifest,
        prediction_root=str(output / "predictions"),
        gt_root=gt_root,
        gt_cache_root=gt_cache_root,
        model_path=model_path,
        upper_indices=upper_indices,
        left_indices=left_indices,
        right_indices=right_indices,
        output_root=str(output / "evaluation"),
    )
    record: dict[str, object] = {
        "schema_version": "1.0",
        "status": "confirmatory_complete",
        "freeze_file": str(Path(freeze_file).resolve()),
        "freeze_sha256": sha256_file(freeze_file),
        "config_sha256": sha256_file(config),
        "manifest_sha256": sha256_file(manifest),
        "gt_cache_tree_sha256": sha256_json(hash_path_tree(gt_cache_root)),
        "release_integrity_verified": True,
        "metrics": metrics,
    }
    (output / "confirmatory.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record
