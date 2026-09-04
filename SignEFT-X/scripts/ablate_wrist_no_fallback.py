#!/usr/bin/env python3
"""Run the unrestricted-wrist candidate without protected-drift rollback.

The released and archived method sources are read-only.  This runner creates a
hash-recorded runtime copy of the archived hand ablation, changes exactly the
three non-active-region acceptance predicates, and writes all artifacts below
an isolated output root.  Expert-unavailable frames retain the same exact
baseline fallback as every wrist condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT / "_archive/research_history"
ARCHIVE_SRC = ARCHIVE / "src"
sys.path.insert(0, str(ARCHIVE_SRC))

import torch  # noqa: E402
from signeft.export import materialize, preflight  # noqa: E402
from signeft.io_utils import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    load_config,
    sha256_file,
)


SOURCE = ARCHIVE_SRC / "signeft/optim/hand.py"
NEEDLE = """                ) & (other_drift < 0.01)
                & (face_drift < 0.01) & (lower_drift < 0.01)
"""
REPLACEMENT = """                ) & torch.ones_like(available[:, side_index], dtype=torch.bool)
                # Ablation only: materialize raw candidates instead of rolling
                # them back on protected-region drift. Availability fallback
                # and every fitting objective remain unchanged.
"""


def remap_manifest(source: Path, destination: Path) -> None:
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        state = Path(item["a3f_state_path"])
        if not state.is_file():
            state = (
                ARCHIVE / "baseline_states/a3f" / item["sign_id"]
                / f"{int(item['source_frame_id']):06d}.npz"
            )
        if not state.is_file() or sha256_file(state) != item["sha256_a3f_state"]:
            raise RuntimeError(f"baseline state mismatch: {item['record_id']}")
        item["a3f_state_path"] = str(state.resolve())
        rows.append(json.dumps(item, sort_keys=True))
    atomic_write_text(destination, "\n".join(rows) + "\n")


def load_ablation_module(output_root: Path):
    original = SOURCE.read_text(encoding="utf-8")
    if original.count(NEEDLE) != 1:
        raise RuntimeError("archived acceptance predicate changed; refusing patch")
    modified = original.replace(NEEDLE, REPLACEMENT)
    runtime_source = output_root / "runtime_src/signeft/optim/hand.py"
    atomic_write_text(runtime_source, modified)
    module = types.ModuleType("signeft.optim.hand_no_isolation_fallback")
    module.__file__ = str(runtime_source)
    module.__package__ = "signeft.optim"
    sys.modules[module.__name__] = module
    exec(compile(modified, str(runtime_source), "exec"), module.__dict__)
    return module, runtime_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=ARCHIVE / "configs/paper_ablation_native_free_wrist_full57.yaml",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT / "outputs/ablation_wrist/free_no_isolation_fallback",
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (output / "refinement_summary.json").is_file():
        raise FileExistsError(f"refusing to overwrite completed ablation: {output}")
    config = load_config(args.config)
    source_manifest = Path(config["paths"]["manifest"])
    manifest = output / "inputs/trv2v_remapped.jsonl"
    remap_manifest(source_manifest, manifest)
    module, runtime_source = load_ablation_module(output)
    torch.manual_seed(int(config.get("seed", 20260903)))
    np.random.seed(int(config.get("seed", 20260903)))
    paths = {key: Path(value) for key, value in config["paths"].items()}
    summary = module.run_h1_refinement(
        manifest, output, paths["smplx_model_root"],
        paths["pose_observation_root"], paths["wilor_observation_root"],
        device=config["runtime"]["device"],
        batch_size=int(config["runtime"]["batch_size"]),
        finger_radius_deg=float(config["hands"]["finger_radius_deg"]),
        wrist_radius_deg=float(config["hands"]["wrist_radius_deg"]),
        primary_gate=str(config["hands"].get("primary_gate", "dual")),
        proposal_mode=str(config["hands"].get("proposal_mode", "canonical_fit")),
        normalize_expert_bones=bool(config["hands"].get("normalize_expert_bones", True)),
    )
    layout = materialize(manifest, output, output / "official_meshes")
    topology = preflight(
        manifest, output / "official_meshes",
        config["topology"]["faces_sha256_int64"], output / "preflight.json",
    )
    atomic_write_json(output / "ablation_protocol.json", {
        "schema_version": "signeft.raw-free-wrist-ablation.v1",
        "description": "unrestricted wrist with raw candidate materialization",
        "changed_predicates": [
            "other_hand_drift_lt_0.01_mm",
            "face_drift_lt_0.01_mm",
            "lower_body_drift_lt_0.01_mm",
        ],
        "change": "disabled for materialization only",
        "expert_unavailable_fallback_retained": True,
        "source": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "runtime_source": str(runtime_source),
        "runtime_source_sha256": sha256_file(runtime_source),
        "replacement_sha256": hashlib.sha256(REPLACEMENT.encode()).hexdigest(),
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(Path(args.config).resolve()),
        "objective_uses_ground_truth": False,
        "refinement": summary,
        "materialization_frames": layout["frames"],
        "preflight_frames": topology["frames"],
    })


if __name__ == "__main__":
    main()
