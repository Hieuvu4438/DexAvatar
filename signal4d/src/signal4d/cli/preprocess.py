from __future__ import annotations

from pathlib import Path

from ..adapters.sgnify_local import build_sgnify_observation_cache
from ..data.manifest import load_manifest


def run(
    manifest_path: str,
    output_root: str,
    body_root: str,
    wilor_root: str,
    model_path: str,
    body_subpath: str = "smplerx/smplx",
    body_source_name: str = "smplerx",
    device: str = "cpu",
    legacy_root: str | None = None,
    legacy_subpath: str = "smplifyx/results",
    legacy_source_name: str = "legacy_dexavatar",
) -> None:
    for row in load_manifest(manifest_path):
        batch, metadata = build_sgnify_observation_cache(
            row,
            body_root=body_root,
            wilor_root=wilor_root,
            model_path=model_path,
            body_subpath=body_subpath,
            body_source_name=body_source_name,
            legacy_root=legacy_root,
            legacy_subpath=legacy_subpath,
            legacy_source_name=legacy_source_name,
            device=device,
        )
        batch.save(Path(output_root) / row.clip_id, metadata)
