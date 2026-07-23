#!/usr/bin/env python3
"""Phase 0 read-only asset and protocol audit for DexAvatar diagnostics."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import pickle
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PREDICTION_FIELDS = (
    "body_pose",
    "left_hand_pose",
    "right_hand_pose",
    "betas",
    "global_orient",
    "transl",
    "expression",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return json_safe(value.detach().cpu().numpy())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__}


def numeric_suffix(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if not match:
        raise ValueError(f"No numeric suffix in {path.name}")
    return int(match.group(1))


def discover_outputs(repo: Path) -> dict[str, Any]:
    outputs_root = repo / "outputs"
    if not outputs_root.is_dir():
        return {"root": str(outputs_root), "exists": False, "methods": []}

    methods = []
    for method_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        result_files = sorted(method_dir.glob("*/smplifyx/results/*.pkl"))
        mesh_files = sorted(method_dir.glob("*/smplifyx/meshes/*.obj"))
        sample = None
        load_error = None
        if result_files:
            try:
                with result_files[0].open("rb") as handle:
                    record = pickle.load(handle)
                sample = {
                    "path": str(result_files[0].relative_to(repo)),
                    "fields": {str(key): json_safe(value) for key, value in record.items()},
                    "required_fields": {
                        key: json_safe(record[key]) if key in record else None
                        for key in PREDICTION_FIELDS
                    },
                }
            except Exception as error:  # audit must report rather than hide malformed assets
                load_error = f"{type(error).__name__}: {error}"
        methods.append(
            {
                "name": method_dir.name,
                "result_pickle_count": len(result_files),
                "mesh_count": len(mesh_files),
                "signs_with_results": len({path.parents[2].name for path in result_files}),
                "signs_with_meshes": len({path.parents[2].name for path in mesh_files}),
                "sample": sample,
                "sample_load_error": load_error,
            }
        )
    return {"root": str(outputs_root), "exists": True, "methods": methods}


def audit_gt(repo: Path) -> dict[str, Any]:
    root = repo / "data" / "smplx_gt"
    suffix_counts = Counter(path.suffix.lower() for path in root.rglob("*") if path.is_file()) if root.is_dir() else Counter()
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "sign_count": len([path for path in root.iterdir() if path.is_dir()]) if root.is_dir() else 0,
        "file_types": dict(sorted(suffix_counts.items())),
        "has_parameter_files": any(suffix in suffix_counts for suffix in (".pkl", ".npz", ".npy", ".pt", ".pth")),
        "representation": "meshes-only" if suffix_counts and set(suffix_counts) == {".obj"} else "mixed-or-missing",
    }


def audit_models(repo: Path) -> dict[str, Any]:
    candidates = sorted(
        set(repo.glob("**/SMPLX_NEUTRAL.npz"))
        | set(repo.glob("**/SMPLX_NEUTRAL.pkl"))
        | set(repo.glob("**/SMPLX_MALE.npz"))
        | set(repo.glob("**/SMPLX_FEMALE.npz"))
    )
    records = []
    for path in candidates:
        record: dict[str, Any] = {
            "path": str(path.relative_to(repo)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        try:
            if path.suffix == ".npz":
                with np.load(path, allow_pickle=True) as model:
                    record["loadable"] = True
                    record["arrays"] = {
                        key: {"shape": list(model[key].shape), "dtype": str(model[key].dtype)}
                        for key in model.files
                        if key in {"v_template", "J_regressor", "weights", "posedirs", "shapedirs", "hands_componentsl", "hands_componentsr"}
                    }
            else:
                with path.open("rb") as handle:
                    model = pickle.load(handle, encoding="latin1")
                record["loadable"] = True
                record["keys"] = sorted(str(key) for key in model.keys()) if hasattr(model, "keys") else []
        except Exception as error:
            record["loadable"] = False
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)

    paper_config = repo / "dexavatar_fitting" / "cfg_files" / "fit_smplx_vposer_x_paper.yaml"
    config_lines = []
    if paper_config.is_file():
        for line in paper_config.read_text().splitlines():
            if re.match(r"^(gender|use_pca|num_pca_comps|flat_hand_mean):", line):
                config_lines.append(line)

    return {
        "models": records,
        "paper_config_relevant_lines": config_lines,
        "evaluator_behavior": "loads neutral NPZ J_regressor directly; does not instantiate SMPL-X",
    }


def build_manifest(repo: Path, prediction_method: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_root = repo / "data" / "evaluation_from_author"
    gt_root = repo / "data" / "smplx_gt"
    prediction_root = repo / "outputs" / prediction_method

    segment_candidates = [
        eval_root / "segment.json",
        eval_root / "data" / "data" / "segment.json",
    ]
    signs_candidates = [
        eval_root / "signs.txt",
        eval_root / "data" / "data" / "signs.txt",
    ]
    segment_path = next((path for path in segment_candidates if path.is_file()), None)
    signs_path = next((path for path in signs_candidates if path.is_file()), None)
    if segment_path is None or signs_path is None:
        return (
            {
                "available": False,
                "error": "Missing segment.json or signs.txt",
                "segment_candidates": [str(path) for path in segment_candidates],
                "sign_candidates": [str(path) for path in signs_candidates],
            },
            [],
        )

    segments = json.loads(segment_path.read_text())
    classes: dict[str, str] = {}
    for line in signs_path.read_text().splitlines():
        tokens = line.split()
        if len(tokens) >= 2:
            classes[tokens[0]] = tokens[1]

    rows = []
    sign_summaries = []
    for sign in sorted(segments):
        gt_dir = gt_root / sign
        pred_mesh_dir = prediction_root / sign / "smplifyx" / "meshes"
        gt_by_number = {
            numeric_suffix(path): path
            for path in gt_dir.glob("*.obj")
            if re.search(r"\d+$", path.stem)
        } if gt_dir.is_dir() else {}
        start, end = segments[sign]
        selected_gt = [
            gt_by_number[frame]
            for frame in range(int(start) * 2, int(end) * 2 + 1)
            if frame in gt_by_number
        ]
        predictions = sorted(
            pred_mesh_dir.glob("*.obj"),
            key=lambda path: numeric_suffix(path),
        ) if pred_mesh_dir.is_dir() else []
        paired = min(len(selected_gt), len(predictions))
        sign_class = classes.get(sign)
        for ordinal in range(paired):
            rows.append(
                {
                    "sign": sign,
                    "class": sign_class,
                    "ordinal": ordinal,
                    "gt_path": str(selected_gt[ordinal].relative_to(repo)),
                    "prediction_path": str(predictions[ordinal].relative_to(repo)),
                    "left_evaluated": sign_class != "0",
                    "right_evaluated": True,
                }
            )
        sign_summaries.append(
            {
                "sign": sign,
                "class": sign_class,
                "segment_start": int(start),
                "segment_end": int(end),
                "selected_gt": len(selected_gt),
                "prediction_meshes": len(predictions),
                "ordinal_pairs": paired,
                "missing_prediction_pairs": max(0, len(selected_gt) - len(predictions)),
                "extra_prediction_meshes": max(0, len(predictions) - len(selected_gt)),
            }
        )

    class_zero = [item for item in sign_summaries if item["class"] == "0"]
    class_other = [item for item in sign_summaries if item["class"] != "0"]
    summary = {
        "available": True,
        "prediction_method": prediction_method,
        "prediction_root": str(prediction_root),
        "prediction_root_exists": prediction_root.is_dir(),
        "segment_path": str(segment_path.relative_to(repo)),
        "signs_path": str(signs_path.relative_to(repo)),
        "pairing": "GT frame numbers in inclusive [2*start, 2*end], paired by ordinal index with independently numerically sorted prediction meshes",
        "sign_count": len(sign_summaries),
        "selected_gt_total": sum(item["selected_gt"] for item in sign_summaries),
        "prediction_mesh_total": sum(item["prediction_meshes"] for item in sign_summaries),
        "ordinal_pair_total": len(rows),
        "equals_2872": len(rows) == 2872,
        "class_zero_sign_count": len(class_zero),
        "two_hand_sign_count": len(class_other),
        "class_zero_pair_count": sum(item["ordinal_pairs"] for item in class_zero),
        "left_hand_sign_count": len(class_other),
        "left_hand_pair_count": sum(item["ordinal_pairs"] for item in class_other),
        "right_hand_sign_count": len(sign_summaries),
        "right_hand_pair_count": len(rows),
        "per_sign": sign_summaries,
    }
    return summary, rows


def audit_masks(repo: Path) -> dict[str, Any]:
    root = repo / "data" / "evaluation_from_author" / "data" / "data"
    mano_path = root / "MANO_SMPLX_vertex_ids.pkl"
    segmentation = root / "sgnify_part_segm_above_pelvis_joint" / "upper_body_minus_face.npy"
    result: dict[str, Any] = {
        "root": str(root),
        "mano_path": str(mano_path.relative_to(repo)) if mano_path.is_file() else str(mano_path),
        "ubody_minus_face_path": str(segmentation.relative_to(repo)) if segmentation.is_file() else str(segmentation),
    }
    if mano_path.is_file():
        with mano_path.open("rb") as handle:
            mano = pickle.load(handle)
        for key, label in (("left_hand", "LHand"), ("right_hand", "RHand")):
            values = np.asarray(mano[key])
            result[label] = {
                "size": int(values.size),
                "min": int(values.min()),
                "max": int(values.max()),
                "sha256": sha256(mano_path),
            }
    if segmentation.is_file():
        values = np.load(segmentation)
        result["UBody(-F)"] = {
            "size": int(values.size),
            "min": int(values.min()),
            "max": int(values.max()),
            "sha256": sha256(segmentation),
        }
    return result


def audit_baselines(repo: Path) -> dict[str, Any]:
    roots = [repo / "outputs", repo / "SGNify"]
    names = []
    for root in roots:
        if root.is_dir():
            names.extend(str(path.relative_to(repo)) for path in root.iterdir() if path.is_dir())
    lowered = {name.lower(): name for name in names}
    return {
        "candidate_directories": sorted(names),
        "eva_candidates": sorted(name for name in names if "eva" in name.lower()),
        "sgnify_output_candidates": sorted(name for name in names if "sgnify" in name.lower() and name.startswith("outputs/")),
        "dexavatar_prediction_candidates": sorted(name for name in names if "method_" in name.lower() or "dexavatar" in name.lower()),
        "paired_eva_available": any("eva" in name for name in lowered),
        "paired_sgnify_available": any("sgnify" in name and name.startswith("outputs/") for name in lowered),
    }


def audit_rectifier(repo: Path) -> dict[str, Any]:
    terms = re.compile(r"hand.{0,40}(rectif|biomech|bend|splay|twist)|(rectif|biomech|bend|splay|twist).{0,40}hand", re.IGNORECASE)
    excluded_roots = {".git", "probes", "outputs", "docs", "nlf", "sapiens", "LHM-plusplus"}
    candidates = []
    for path in repo.rglob("*.py"):
        if any(part in excluded_roots for part in path.parts):
            continue
        try:
            matches = [
                {"line": index, "text": line.strip()[:240]}
                for index, line in enumerate(path.read_text(errors="ignore").splitlines(), 1)
                if terms.search(line)
            ]
        except OSError:
            continue
        if matches:
            candidates.append(
                {
                    "path": str(path.relative_to(repo)),
                    "matches": matches[:20],
                }
            )
    explicit_rectifier = [
        item for item in candidates
        if any(re.search(r"hand.{0,20}rectif|rectif.{0,20}hand", match["text"], re.IGNORECASE) for match in item["matches"])
    ]
    return {
        "reusable_rectifier_found": bool(explicit_rectifier),
        "reusable_candidates": explicit_rectifier,
        "related_code": candidates[:50],
        "assessment": "A reusable GT hand-pose rectifier requires an implementation that transforms 15-joint poses with per-joint bend/splay/twist limits and MANO/SMPL-X axis alignment. Generic camera rectification and scalar hand-pose penalties do not qualify.",
    }


def audit_runtime_feasibility(repo: Path) -> dict[str, Any]:
    launcher = repo / "scripts" / "exp1_paper_nlf_fit.sh"
    required = {
        "launcher": launcher,
        "frames": repo / "data" / "frames",
        "inputs": repo / "outputs" / "method_nlf_wilor",
        "models": repo / "SMPLer-X" / "common" / "utils" / "human_model_files",
        "config": repo / "dexavatar_fitting" / "cfg_files" / "fit_smplx_vposer_x_paper.yaml",
    }
    availability = {key: path.exists() for key, path in required.items()}
    return {
        "measured": False,
        "reason": "Phase 0 does not run an upstream launcher that writes outside probes/. A probe-owned output redirect is not part of the audit-only phase.",
        "canonical_launcher": str(launcher.relative_to(repo)) if launcher.exists() else str(launcher),
        "required_assets": {key: {"path": str(path), "exists": availability[key]} for key, path in required.items()},
        "all_required_assets_present": all(availability.values()),
    }


def source_facts(repo: Path) -> dict[str, Any]:
    evaluator = repo / "data" / "evaluation_from_author" / "evaluate_new_fitting.py"
    facts: dict[str, Any] = {"path": str(evaluator.relative_to(repo)), "exists": evaluator.is_file()}
    if evaluator.is_file():
        text = evaluator.read_text()
        facts.update(
            {
                "sha256": sha256(evaluator),
                "hardcoded_author_paths": sorted(set(re.findall(r"/home/kaustubh/[^'\"]+", text))),
                "central_argument_referenced_after_parse": text.count("central") > 2,
                "contains_ordinal_pairing": "mocap_objs[idx][inter_idx]" in text,
                "contains_class_zero_left_skip": "key == 'left hand' and class_sign[soma_key]== '0'" in text,
            }
        )
    return facts


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--prediction-method", default="method_hamer")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output_dir = (args.output_dir or repo / "probes" / "results" / "phase0").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc).isoformat()
    manifest_summary, manifest_rows = build_manifest(repo, args.prediction_method)
    audit = {
        "schema_version": 1,
        "phase": "Phase 0 asset audit",
        "started_utc": started,
        "repo": str(repo),
        "git": {
            "branch": command_output(["git", "branch", "--show-current"], repo),
            "head": command_output(["git", "rev-parse", "HEAD"], repo),
            "origin": command_output(["git", "remote", "get-url", "origin"], repo),
            "status_porcelain": command_output(["git", "status", "--porcelain"], repo).splitlines(),
        },
        "command": " ".join(sys.argv),
        "predictions": discover_outputs(repo),
        "ground_truth": audit_gt(repo),
        "smplx_models": audit_models(repo),
        "evaluator_source": source_facts(repo),
        "frame_manifest": manifest_summary,
        "region_masks": audit_masks(repo),
        "baselines": audit_baselines(repo),
        "rectifier": audit_rectifier(repo),
        "runtime": audit_runtime_feasibility(repo),
    }

    blockers = []
    if not audit["ground_truth"]["has_parameter_files"]:
        blockers.append("E1 blocked: SGNify GT SMPL-X parameters are unavailable; GT is meshes-only.")
    if not audit["rectifier"]["reusable_rectifier_found"] or not audit["ground_truth"]["has_parameter_files"]:
        blockers.append("E3 blocked: GT hand parameters and/or a reusable per-joint hand rectifier are unavailable.")
    if not manifest_summary.get("equals_2872", False):
        blockers.append(
            f"Protocol discrepancy: author-style ordinal manifest has {manifest_summary.get('ordinal_pair_total', 0)} pairs, not 2,872."
        )
    if not manifest_summary.get("prediction_root_exists", False):
        blockers.append(f"Parity blocked: prediction method outputs/{args.prediction_method} are absent.")
    if not audit["baselines"]["paired_eva_available"]:
        blockers.append("Paired EVA* comparison blocked: no EVA* per-frame outputs found.")
    audit["blockers"] = blockers
    audit["completed_utc"] = datetime.now(timezone.utc).isoformat()

    json_path = output_dir / "audit.json"
    manifest_path = output_dir / "frame_manifest.csv"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_csv(manifest_path, manifest_rows)

    summary = {
        "audit_json": str(json_path),
        "manifest_csv": str(manifest_path),
        "head": audit["git"]["head"],
        "prediction_method": args.prediction_method,
        "prediction_meshes": manifest_summary.get("prediction_mesh_total", 0),
        "selected_gt": manifest_summary.get("selected_gt_total", 0),
        "ordinal_pairs": manifest_summary.get("ordinal_pair_total", 0),
        "equals_2872": manifest_summary.get("equals_2872", False),
        "left_hand_signs": manifest_summary.get("left_hand_sign_count", 0),
        "left_hand_pairs": manifest_summary.get("left_hand_pair_count", 0),
        "right_hand_signs": manifest_summary.get("right_hand_sign_count", 0),
        "right_hand_pairs": manifest_summary.get("right_hand_pair_count", 0),
        "blockers": blockers,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
