"""Target-free audit of geometry selection against the exact base candidate.

The evaluator consumes only frozen candidate energy terms and targetless cache
metadata.  It never decodes pose targets.  Candidate zero is the architectural
identity/base path; the selected candidate is read from the append-only
reselection artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from cusp_sl.evaluate_frontend_evidence import cluster_delta_interval
from phase2_refiner.data.cache_schema import load_cache_clip


TERM_NAMES = ("observation", "motion", "physical", "form")
CANDIDATE_ARRAYS = (
    "candidate_rotation",
    "candidate_residual",
    "candidate_valid",
    "energy_terms",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def target_reads_prohibited(manifest: dict) -> bool:
    """Accept the two versioned JSON encodings used by targetless manifests."""
    if "target_reads_permitted" not in manifest:
        return False
    value = manifest["target_reads_permitted"]
    return value is False or (type(value) is int and value == 0)


def compare_payloads(
    source: dict[str, np.ndarray], selected: dict[str, np.ndarray]
) -> tuple[int, dict[str, float]]:
    """Check frozen-candidate invariants and return selected-minus-base terms."""
    missing = set(CANDIDATE_ARRAYS).difference(source) | set(
        CANDIDATE_ARRAYS
    ).difference(selected)
    if missing:
        raise ValueError(f"Candidate payload omits arrays: {sorted(missing)}")
    for name in CANDIDATE_ARRAYS:
        if not np.array_equal(source[name], selected[name]):
            raise ValueError(f"Re-selection modified frozen array: {name}")
    terms = selected["energy_terms"]
    valid = selected["candidate_valid"]
    index = int(selected["selected_index"])
    if terms.ndim != 2 or terms.shape[1] != len(TERM_NAMES):
        raise ValueError(f"Invalid energy term shape: {terms.shape}")
    if valid.shape != (len(terms),) or not bool(valid[0]):
        raise ValueError("Candidate validity does not preserve a valid base")
    if index < 0 or index >= len(terms) or not bool(valid[index]):
        raise ValueError(f"Selected invalid candidate index: {index}")
    if "energy" not in selected or selected["energy"].shape != (len(terms),):
        raise ValueError("Selected payload omits normalized composite energy")
    result = {
        f"base_{name}": float(terms[0, term_index])
        for term_index, name in enumerate(TERM_NAMES)
    }
    result.update(
        {
            f"selected_{name}": float(terms[index, term_index])
            for term_index, name in enumerate(TERM_NAMES)
        }
    )
    result["base_composite"] = float(selected["energy"][0])
    result["selected_composite"] = float(selected["energy"][index])
    return index, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Append-only output exists: {args.output}")
    if args.bootstrap_replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")

    input_manifest = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    source_manifest_path = args.candidate_root / "manifest.json"
    selected_manifest_path = args.selected_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    selected_manifest = json.loads(selected_manifest_path.read_text(encoding="utf-8"))
    if input_manifest.get("role") != "development_targetless_inference":
        raise ValueError("Selection evidence requires a targetless development input")
    if not target_reads_prohibited(input_manifest):
        raise ValueError("Input manifest does not explicitly prohibit target reads")
    input_hash = sha256(args.input_manifest)
    for manifest in (source_manifest, selected_manifest):
        if manifest.get("protocol_role") != "development_validation":
            raise ValueError("Selection evidence is development-only")
        if manifest.get("variant") != "a7_geometry":
            raise ValueError("Selection evidence requires A7 geometry artifacts")
        if manifest.get("input_manifest_sha256") != input_hash:
            raise ValueError("Candidate/input manifest hash mismatch")
    if selected_manifest.get("selection_stage") != "frozen_candidate_cpu_reselection":
        raise ValueError("Selected artifact was not frozen-candidate re-selection")
    if selected_manifest.get("source_candidate_manifest_sha256") != sha256(
        source_manifest_path
    ):
        raise ValueError("Selected/source candidate manifest hash mismatch")

    input_entries = input_manifest.get("clips", [])
    input_summaries = {
        str(item["clip_id"]): item for item in input_manifest.get("summaries", [])
    }
    source_summaries = {
        str(item["clip_id"]): item for item in source_manifest.get("summaries", [])
    }
    selected_summaries = {
        str(item["clip_id"]): item for item in selected_manifest.get("summaries", [])
    }
    if not (
        len(input_entries)
        == len(input_summaries)
        == len(source_summaries)
        == len(selected_summaries)
    ):
        raise ValueError("Input/source/selected coverage differs")

    records: list[dict[str, object]] = []
    candidate_indices: Counter[int] = Counter()
    for entry in input_entries:
        cache_path = _resolve(args.input_manifest.parent, str(entry))
        clip = load_cache_clip(cache_path)
        clip_id = str(clip.clip_id)
        input_item = input_summaries.get(clip_id)
        source_item = source_summaries.get(clip_id)
        selected_item = selected_summaries.get(clip_id)
        if input_item is None or source_item is None or selected_item is None:
            raise ValueError(f"Missing declared clip: {clip_id}")
        if sha256(cache_path) != input_item.get("targetless_cache_sha256"):
            raise ValueError(f"Targetless cache hash mismatch: {cache_path}")
        metadata = json.loads(clip.metadata_json)
        source_group = str(metadata.get("source_group", ""))
        if not source_group:
            raise ValueError(f"Targetless cache lacks source_group: {cache_path}")
        source_path = args.candidate_root / "clips" / f"{clip_id}.npz"
        selected_path = args.selected_root / "clips" / f"{clip_id}.npz"
        if sha256(source_path) != source_item.get("prediction_sha256"):
            raise ValueError(f"Source prediction hash mismatch: {source_path}")
        if sha256(selected_path) != selected_item.get("prediction_sha256"):
            raise ValueError(f"Selected prediction hash mismatch: {selected_path}")
        if selected_item.get("source_prediction_sha256") != source_item.get(
            "prediction_sha256"
        ):
            raise ValueError(f"Per-clip source provenance mismatch: {clip_id}")
        with np.load(source_path, allow_pickle=False) as payload:
            source = {name: payload[name] for name in payload.files}
        with np.load(selected_path, allow_pickle=False) as payload:
            selected = {name: payload[name] for name in payload.files}
        index, evidence = compare_payloads(source, selected)
        if index != int(selected_item.get("selected_index", -1)):
            raise ValueError(f"Manifest/payload selected index mismatch: {clip_id}")
        candidate_indices[index] += 1
        records.append(
            {
                "clip_id": clip_id,
                "source_group": source_group,
                "frames": len(clip.frame_names),
                "selected_index": index,
                **evidence,
            }
        )
    if set(source_summaries) != {str(row["clip_id"]) for row in records}:
        raise ValueError("Candidate clip set differs from targetless input")

    args.output.mkdir(parents=True)
    with (args.output / "per_clip.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    weights = np.asarray([row["frames"] for row in records], dtype=np.float64)

    def aggregate(name: str) -> float:
        return float(
            np.average([float(row[name]) for row in records], weights=weights)
        )

    summary: dict[str, object] = {
        "role": "target_free_frozen_candidate_selection_evidence_audit",
        "target_reads": 0,
        "clips": len(records),
        "frames": int(weights.sum()),
        "source_groups": len({str(row["source_group"]) for row in records}),
        "input_manifest_sha256": input_hash,
        "source_candidate_manifest_sha256": sha256(source_manifest_path),
        "selected_manifest_sha256": sha256(selected_manifest_path),
        "energy_statistics_sha256": selected_manifest["energy_statistics_sha256"],
        "selected_index_counts": {
            str(key): value for key, value in sorted(candidate_indices.items())
        },
        "candidate_arrays_bit_identical": True,
    }
    for term in (*TERM_NAMES, "composite"):
        base = f"base_{term}"
        method = f"selected_{term}"
        summary[base] = aggregate(base)
        summary[method] = aggregate(method)
        summary[f"clustered_selected_minus_base_{term}"] = cluster_delta_interval(
            records,
            method,
            base,
            "frames",
            replicates=args.bootstrap_replicates,
            seed=args.seed + len(summary),
        )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
