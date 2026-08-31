from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from signeft.data.manifest import read_manifest
from signeft.io_utils import atomic_write_json, atomic_write_text, sha256_file


ENGINEERING_PANEL_SIGNS = (
    "Ablehnen", "Akzeptieren", "Arzt", "AufgebenResignieren",
    "AusgebenGeldVerschwenden", "Auto", "BesuchenEinmischen", "Blitz",
    "Blume", "Boese", "BroetchenAufschneiden", "Dort",
)


def build_engineering_split(manifest: Path, output_root: Path) -> dict[str, object]:
    records = read_manifest(manifest)
    panel_names = set(ENGINEERING_PANEL_SIGNS)
    available = {record.sign_id for record in records}
    if missing := panel_names - available:
        raise RuntimeError(f"engineering panel signs missing: {sorted(missing)}")
    panel = [record for record in records if record.sign_id in panel_names]
    holdout = [record for record in records if record.sign_id not in panel_names]
    if len(panel) != 298 or len(holdout) != 1195:
        raise RuntimeError(f"unexpected panel/holdout counts: {len(panel)}/{len(holdout)}")
    paths = {
        "engineering12": output_root / "engineering12.jsonl",
        "untouched45": output_root / "untouched45.jsonl",
    }
    for name, subset in (("engineering12", panel), ("untouched45", holdout)):
        content = "\n".join(json.dumps(asdict(record), sort_keys=True) for record in subset) + "\n"
        atomic_write_text(paths[name], content)
    report = {
        "schema_version": "signeft.development-split.v1",
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": sha256_file(manifest),
        "selection_warning": (
            "engineering12 was used by prior local experiments; it is not an author-sanctioned dev split"
        ),
        "publication_policy": "report engineering12 separately; untouched45 is the only local holdout",
        "engineering12": {
            "signs": list(ENGINEERING_PANEL_SIGNS), "frames": len(panel),
            "path": str(paths["engineering12"].resolve()), "sha256": sha256_file(paths["engineering12"]),
        },
        "untouched45": {
            "signs": sorted(available - panel_names), "frames": len(holdout),
            "path": str(paths["untouched45"].resolve()), "sha256": sha256_file(paths["untouched45"]),
        },
    }
    atomic_write_json(output_root / "split_lock.json", report)
    return report

