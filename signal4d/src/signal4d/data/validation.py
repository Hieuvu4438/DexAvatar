from __future__ import annotations

from pathlib import Path

from .manifest import ClipManifest


def validate_source_images(manifest: ClipManifest, dataset_root: str | Path) -> None:
    root = Path(dataset_root).resolve()
    for relpath in manifest.image_relpaths:
        path = (root / relpath).resolve()
        if root not in path.parents:
            raise ValueError(f"image escapes dataset root: {relpath}")
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing or empty image: {path}")
