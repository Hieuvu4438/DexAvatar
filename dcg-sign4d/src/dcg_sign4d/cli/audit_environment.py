from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import yaml

from dcg_sign4d.utils.hashing import file_sha256


def _has_author_required(value: object) -> bool:
    if isinstance(value, dict):
        return any(_has_author_required(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_author_required(item) for item in value)
    return value == "AUTHOR_REQUIRED"


def audit(config_path: str | Path) -> dict[str, object]:
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    development = bool(config.get("experiment", {}).get("development_only", False))
    placeholders = _has_author_required(config)
    if placeholders and not development:
        raise ValueError("production config contains unresolved AUTHOR_REQUIRED decisions")
    if development and path.name != "smoke.yaml":
        raise ValueError("development defaults are only allowed in configs/smoke.yaml")
    return {
        "config": str(path),
        "config_sha256": file_sha256(path),
        "development_only": development,
        "author_required_present": placeholders,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "identity_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit config and local runtime")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.config), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
