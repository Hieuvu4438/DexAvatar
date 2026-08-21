from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from ...data.manifest import load_manifest
from ...io.predictions import PredictionArtifact
from ...models.gating import merge_predictions
from ...utils.hashing import sha256_file
from .config import load_v6_config
from .safe_gate import safe_acceptance_mask


def _tensor(diagnostics: dict[str, object], key: str, frames: int) -> torch.Tensor:
    value = torch.as_tensor(diagnostics[key], dtype=torch.float32)
    if value.shape != (frames,) or not torch.isfinite(value).all():
        raise ValueError(f"diagnostic {key} must be finite [T]")
    return value


def run(
    config_path: str,
    manifest_path: str,
    candidate_root: str,
    baseline_root: str,
    output_root: str,
) -> dict[str, Any]:
    """Apply the frozen V6 rule gate using only inference-time diagnostics."""
    config = load_v6_config(config_path)
    if not config.safe_gate.enabled:
        raise ValueError("offline safe-gate config must set safe_gate.enabled=true")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V6 gated run: {output}")
    output.mkdir(parents=True)
    incomplete = output / ".run_incomplete"
    incomplete.write_text("incomplete\n", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    clips = load_manifest(manifest_path)
    for item in clips:
        candidate_dir = Path(candidate_root) / item.clip_id
        baseline_dir = Path(baseline_root) / item.clip_id
        candidate, candidate_meta = PredictionArtifact.load(candidate_dir)
        baseline, baseline_meta = PredictionArtifact.load(baseline_dir)
        expected = item.frame_ids
        if candidate.frame_ids.tolist() != expected or baseline.frame_ids.tolist() != expected:
            raise ValueError(f"manifest/prediction frame mismatch for {item.clip_id}")
        diagnostics_path = candidate_dir / "v6_diagnostics.json"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        frames = len(expected)
        selected = safe_acceptance_mask(
            _tensor(diagnostics, "base_frame_objective", frames),
            _tensor(diagnostics, "candidate_frame_objective", frames),
            _tensor(diagnostics, "rotation_delta_max_rad", frames),
            _tensor(diagnostics, "uncertainty_ratio", frames),
            require_objective_improvement=config.safe_gate.require_objective_improvement,
            minimum_objective_improvement=(
                config.safe_gate.minimum_objective_improvement
            ),
            max_rotation_delta_rad=config.safe_gate.max_rotation_delta_rad,
            max_uncertainty_ratio=config.safe_gate.max_uncertainty_ratio,
            transition_radius=config.safe_gate.transition_radius,
        ).cpu().numpy()
        merged = merge_predictions(candidate, baseline, selected)
        clip_output = output / "predictions" / item.clip_id
        merged.save(
            clip_output,
            {
                "schema_version": "1.0",
                "status": "success",
                "method_name": config.method_name,
                "clip_id": item.clip_id,
                "manifest_item_sha256": item.sha256,
                "coordinate_convention": candidate_meta.get("coordinate_convention"),
                "length_unit": candidate_meta.get("length_unit"),
                "smplx_model_sha256": candidate_meta.get("smplx_model_sha256"),
                "candidate_artifact_sha256": candidate_meta["artifact_sha256"],
                "baseline_artifact_sha256": baseline_meta["artifact_sha256"],
                "gate_config_sha256": config.sha256,
                "candidate_frames": int(selected.sum()),
                "baseline_frames": int((~selected).sum()),
                "gt_used": False,
                "gt_used_for_selection": False,
            },
        )
        gated_diagnostics = dict(diagnostics)
        gated_diagnostics.update(
            {
                "acceptance_mask": selected.tolist(),
                "accepted_frames": int(selected.sum()),
                "safe_gate": config.safe_gate.model_dump(mode="json"),
                "safe_gate_config_sha256": config.sha256,
                "gt_used": False,
            }
        )
        diagnostics_text = json.dumps(gated_diagnostics, indent=2, sort_keys=True) + "\n"
        for filename in ("v6_diagnostics.json", "factor_diagnostics.json"):
            (clip_output / filename).write_text(diagnostics_text, encoding="utf-8")
        for frame_id, choice in zip(expected, selected, strict=True):
            rows.append(
                {
                    "clip_id": item.clip_id,
                    "frame_id": frame_id,
                    "selected_candidate": int(choice),
                }
            )
    with (output / "selection.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "1.0",
        "status": "success",
        "method_name": config.method_name,
        "clips": len(clips),
        "frames": len(rows),
        "candidate_frames": sum(row["selected_candidate"] for row in rows),
        "baseline_frames": sum(1 - row["selected_candidate"] for row in rows),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": config.sha256,
        "config_file_sha256": sha256_file(config_path),
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "candidate_root": str(Path(candidate_root).resolve()),
        "baseline_root": str(Path(baseline_root).resolve()),
        "gt_used": False,
        "gt_used_for_selection": False,
    }
    (output / "gate_run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    incomplete.unlink()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="signal4d-v6-safe-gate")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    report = run(
        args.config,
        args.manifest,
        args.candidate_root,
        args.baseline_root,
        args.output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
