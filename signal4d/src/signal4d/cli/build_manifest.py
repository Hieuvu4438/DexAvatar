from __future__ import annotations

from pathlib import Path

from ..config import load_yaml
from ..data.manifest import build_sgnify_manifest, write_manifest


def run(config_path: str, output: str) -> None:
    cfg = load_yaml(config_path)
    if cfg.get("dataset") != "sgnify":
        raise ValueError("build-manifest currently supports the SGNify filesystem contract")
    base = Path(config_path).resolve().parents[2]

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (base / path).resolve()

    rows = build_sgnify_manifest(
        resolve(str(cfg["frames_root"])),
        resolve(str(cfg["segments"])),
        split=str(cfg.get("split", "development")),  # type: ignore[arg-type]
        fps=float(cfg.get("fps", 15.0)),
    )
    write_manifest(rows, output)
