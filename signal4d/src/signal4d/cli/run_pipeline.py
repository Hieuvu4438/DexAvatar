from __future__ import annotations

import json
from pathlib import Path

from ..config import load_method_config
from ..data.cache import ObservationBatch
from ..data.manifest import load_manifest
from ..data.provenance import RunProvenance
from ..optimization.solver import fit_sequence
from ..utils.hashing import sha256_file
from ..utils.seed import seed_everything


def run(config_path: str, manifest_path: str, cache_root: str, output_root: str) -> None:
    config = load_method_config(config_path)
    manifest = load_manifest(manifest_path)
    seed_everything(config.seed)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    provenance = RunProvenance.start(sha256_file(manifest_path), config.sha256, config.seed)
    for row in manifest:
        batch, cache_metadata = ObservationBatch.load(Path(cache_root) / row.clip_id)
        batch.validate_against(row)
        prediction, diagnostics = fit_sequence(
            batch,
            config,
            fps=row.fps,
            log_path=str(output_root / "logs" / f"{row.clip_id}.jsonl"),
        )
        prediction.save(
            output_root / "predictions" / row.clip_id,
            {
                "schema_version": "1.0",
                "clip_id": row.clip_id,
                "coordinate_convention": cache_metadata["camera_convention"],
                "length_unit": cache_metadata["length_unit"],
                "status": "success",
                "method_name": config.method_name,
                "config_sha256": config.sha256,
                "manifest_item_sha256": row.sha256,
            },
        )
        (output_root / "predictions" / row.clip_id / "factor_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
        )
    provenance.finish()
    provenance.write(output_root / "run.json")
