from __future__ import annotations

import json
import platform
from pathlib import Path

import torch

from ..utils.hashing import hash_path_tree, sha256_file


def run(
    output: str,
    configs: list[str],
    manifests: list[str],
    artifacts: list[str],
) -> dict[str, object]:
    if not configs or not manifests:
        raise ValueError("release freeze requires configs and manifests")
    report: dict[str, object] = {
        "schema_version": "1.0",
        "status": "frozen_before_confirmatory_test",
        "configs": {path: sha256_file(path) for path in configs},
        "manifests": {path: sha256_file(path) for path in manifests},
        "artifacts": {path: hash_path_tree(path) for path in artifacts},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["freeze_sha256"] = sha256_file(destination)
    return report
