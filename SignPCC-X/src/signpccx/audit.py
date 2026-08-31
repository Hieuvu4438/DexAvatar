from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from signpccx.data.manifest import read_jsonl
from signpccx.evaluation.official import OFFICIAL_EVALUATOR_SHA256
from signpccx.io import atomic_write_json, sha256_file


def _validate_frame(
    run_root: Path,
    record,
    method: dict[str, bool],
) -> dict[str, object]:
    fit_root = run_root / "fit_sequences"
    stem = f"{record.source_frame_id:06d}"
    npz_path = fit_root / "frames" / record.sign / f"{stem}.npz"
    sidecar_path = npz_path.with_suffix(".json")
    log_path = fit_root / "logs" / record.sign / f"{stem}.jsonl"
    for path in (npz_path, sidecar_path, log_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing full-method frame artifact: {path}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("sha256") != sha256_file(npz_path):
        raise RuntimeError(f"frame NPZ hash mismatch: {npz_path}")
    if sidecar.get("log_sha256") != sha256_file(log_path):
        raise RuntimeError(f"frame log hash mismatch: {log_path}")
    if sidecar.get("objective_uses_ground_truth") is not False:
        raise RuntimeError(f"GT-use invariant not proven: {sidecar_path}")
    if sidecar.get("objective_uses_temporal_pose") is not False:
        raise RuntimeError(f"temporal invariant not proven: {sidecar_path}")
    with np.load(npz_path, allow_pickle=False) as archive:
        required = {
            "mesh_parametric": (1, 10475, 3), "betas": (1, 10),
            "global_orient": (1, 3), "body_pose": (1, 63),
            "left_hand_pose": (1, 45), "right_hand_pose": (1, 45),
            "transl": (1, 3), "frame_ids": (1,),
        }
        for key, shape in required.items():
            if key not in archive.files or archive[key].shape != shape:
                raise ValueError(f"{npz_path}: {key} shape contract")
            if archive[key].dtype.kind in "fc" and not np.isfinite(archive[key]).all():
                raise FloatingPointError(f"{npz_path}: non-finite {key}")
        if int(archive["frame_ids"][0]) != record.source_frame_id:
            raise RuntimeError(f"{npz_path}: frame ID mismatch")
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    stages = Counter(row["stage"] for row in rows)
    if not 50 <= stages["S0_camera_root"] <= 60:
        raise RuntimeError(f"{log_path}: S0 outside 50-60 steps")
    if not 75 <= stages["S1_upper_body"] <= 100:
        raise RuntimeError(f"{log_path}: S1 outside 75-100 steps")
    s4 = [row for row in rows if row["stage"] == "S4_lbfgs_refine"]
    decisions = [row for row in s4 if row.get("event") == "decision"]
    closure_steps = len(s4) - len(decisions)
    if len(decisions) != 1 or closure_steps < 1:
        raise RuntimeError(
            f"{log_path}: S4 requires finite closure work and one decision; "
            f"got closures={closure_steps}, decisions={len(decisions)}"
        )
    configured_max_iter = decisions[0].get("configured_max_iter")
    if configured_max_iter is not None and not 15 <= int(configured_max_iter) <= 25:
        raise RuntimeError(f"{log_path}: S4 configured max_iter outside 15-25")
    if not 20 <= stages["S5_canonical_refit"] <= 60:
        raise RuntimeError(f"{log_path}: S5 outside 20-60 steps")
    if method.get("hypotheses"):
        if stages["K0"] < 8:
            raise RuntimeError(f"{log_path}: insufficient structured K0 candidates")
        s2 = [count for stage, count in stages.items() if stage.startswith("S2_hand_hypothesis_")]
        s3 = [count for stage, count in stages.items() if stage.startswith("S3_bimanual_contact_")]
        if len(s2) != 8 or any(count != 25 for count in s2):
            raise RuntimeError(f"{log_path}: K1 must be 4 candidates/side x25: {s2}")
        if len(s3) != 4 or any(not 80 <= count <= 120 for count in s3):
            raise RuntimeError(f"{log_path}: K2 must be top2x2 with 80-120 steps: {s3}")
    else:
        if any(stage == "K0" or stage.startswith(("S2_", "S3_")) for stage in stages):
            raise RuntimeError(f"{log_path}: hypothesis stages leaked into ablation")
    final = sidecar.get("final_raw_losses", {})
    if method.get("contact"):
        for key in ("contact", "penetration", "penetration_depth", "penetration_count"):
            if key not in final:
                raise RuntimeError(f"{sidecar_path}: missing M4 diagnostic {key}")
    elif any(key in final for key in ("contact", "penetration")):
        raise RuntimeError(f"{sidecar_path}: M4 leaked into contact-off ablation")
    return {
        "sign": record.sign, "frame_id": record.source_frame_id,
        "npz_sha256": sidecar["sha256"], "log_sha256": sidecar["log_sha256"],
        "lbfgs_accepted": bool(sidecar.get("lbfgs_accepted")),
        "contact_proposals": len(sidecar.get("contact_proposals", [])),
    }


def audit_full_run(
    run_root: Path,
    manifest_root: Path,
    method: dict[str, bool],
    evaluator: Path,
    *,
    signs: set[str] | None = None,
    require_evaluation: bool = False,
    metrics_name: str = "metrics",
) -> dict[str, object]:
    manifests = sorted(manifest_root.glob("*.jsonl"))
    if signs is not None:
        unknown = signs - {path.stem for path in manifests}
        if unknown:
            raise ValueError(f"audit unknown signs: {sorted(unknown)}")
        manifests = [path for path in manifests if path.stem in signs]
    if not manifests:
        raise RuntimeError("audit has no selected manifests")
    records = [record for path in manifests for record in read_jsonl(path)]
    frames = [_validate_frame(run_root, record, method) for record in records]
    for manifest in manifests:
        sequence = run_root / "fit_sequences" / "clips" / manifest.stem / "mesh_parametric_final.npz"
        sidecar = sequence.with_suffix(".json")
        if not sequence.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"missing completed sign sequence: {sequence}")
        report = json.loads(sidecar.read_text(encoding="utf-8"))
        if report.get("sha256") != sha256_file(sequence):
            raise RuntimeError(f"sequence hash mismatch: {sequence}")
        with np.load(sequence, allow_pickle=False) as archive:
            expected = np.asarray([record.source_frame_id for record in read_jsonl(manifest)])
            if not np.array_equal(archive["frame_ids"], expected):
                raise RuntimeError(f"sequence frame IDs mismatch: {sequence}")
            if archive["mesh_parametric"].shape != (len(expected), 10475, 3):
                raise ValueError(f"sequence mesh shape mismatch: {sequence}")
    evaluation = None
    if require_evaluation:
        preflight = json.loads((run_root / "preflight.json").read_text(encoding="utf-8"))
        if preflight.get("status") != "ok" or preflight.get("signs") != len(manifests) or preflight.get("frames") != len(records):
            raise RuntimeError("run preflight does not prove selected scope")
        official_path = run_root / metrics_name / "official_result.json"
        audited_path = run_root / metrics_name / "audited" / "summary.json"
        official = json.loads(official_path.read_text(encoding="utf-8"))
        audited = json.loads(audited_path.read_text(encoding="utf-8"))
        if official.get("evaluator_sha256") != OFFICIAL_EVALUATOR_SHA256:
            raise RuntimeError("official evaluator hash missing/mismatched")
        if audited.get("official_rounded_parity") is not True:
            raise RuntimeError("audited/official parity not proven")
        evaluation = {"official": str(official_path.resolve()), "audited": str(audited_path.resolve())}
    if sha256_file(evaluator) != OFFICIAL_EVALUATOR_SHA256:
        raise RuntimeError("current evaluator no longer matches lock")
    report = {
        "schema_version": "signpccx.full-run-audit.v1", "status": "ok",
        "run_root": str(run_root.resolve()), "method": method,
        "signs": len(manifests), "frames": len(records),
        "lbfgs_accepted": sum(item["lbfgs_accepted"] for item in frames),
        "contact_proposals": sum(item["contact_proposals"] for item in frames),
        "evaluation": evaluation, "frame_evidence": frames,
    }
    output = run_root / ("full_run_audit.json" if signs is None else "panel_run_audit.json")
    atomic_write_json(output, report)
    return report
