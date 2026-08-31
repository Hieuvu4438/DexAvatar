"""Immutable prediction artifact writer following the method-freeze layout."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dcg_sign4d.contact.ontology import ContactGraphBatch
from dcg_sign4d.inference.hypothesis import Hypothesis
from dcg_sign4d.initialization.artifact import (
    load_initialization_artifact,
    save_initialization_artifact,
)
from dcg_sign4d.initialization.trajectory_io import load_trajectory, save_trajectory
from dcg_sign4d.utils.hashing import canonical_hash

from .provenance import validate_run_identity

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _save_graph(path: Path, hypothesis_graph: Any) -> None:
    np.savez_compressed(
        path,
        event_state=hypothesis_graph.event_state.cpu().numpy(),
        event_probability=hypothesis_graph.event_probability.detach().cpu().numpy(),
        edge_valid=hypothesis_graph.edge_valid.cpu().numpy(),
        uncertain_mask=hypothesis_graph.uncertain_mask.cpu().numpy(),
        segment_id=hypothesis_graph.segment_id.cpu().numpy(),
        segment_duration=hypothesis_graph.segment_duration.cpu().numpy(),
    )


def _load_graph(path: Path) -> ContactGraphBatch:
    with np.load(path, allow_pickle=False) as arrays:
        required = {
            "event_state",
            "event_probability",
            "edge_valid",
            "uncertain_mask",
            "segment_id",
            "segment_duration",
        }
        if set(arrays.files) != required:
            raise ValueError(f"contact graph arrays mismatch: {path}")
        graph = ContactGraphBatch(
            event_state=torch.from_numpy(arrays["event_state"]).long(),
            event_probability=torch.from_numpy(arrays["event_probability"]),
            edge_valid=torch.from_numpy(arrays["edge_valid"]).bool(),
            uncertain_mask=torch.from_numpy(arrays["uncertain_mask"]).bool(),
            segment_id=torch.from_numpy(arrays["segment_id"]).long(),
            segment_duration=torch.from_numpy(arrays["segment_duration"]),
        ).validate()
    sums = graph.event_probability.sum(-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        raise ValueError(f"contact probabilities do not sum to one: {path}")
    return graph


def validate_prediction_artifact(
    root: str | Path, *, require_complete: bool = True
) -> dict[str, Any]:
    """Deep validation of identities, trajectories, graphs, ranking and selection."""

    root = Path(root)
    if require_complete and not (root / "ARTIFACT_COMPLETE").is_file():
        raise ValueError("prediction artifact has no completion marker")
    for name in (
        "input_manifest.json",
        "observation_hashes.json",
        "ranker_config.json",
        "ranking.json",
        "selected_hypothesis.txt",
        "run_identity.json",
    ):
        if not (root / name).is_file():
            raise ValueError(f"prediction artifact is missing {name}")
    run_identity = json.loads((root / "run_identity.json").read_text(encoding="utf-8"))
    stored_identity_hash = run_identity.pop("run_identity_hash", None)
    if stored_identity_hash != canonical_hash(run_identity):
        raise ValueError("run identity hash mismatch")
    validate_run_identity(run_identity)
    observation_hashes = json.loads((root / "observation_hashes.json").read_text(encoding="utf-8"))
    if not all(_SHA256.fullmatch(value) for value in observation_hashes.values()):
        raise ValueError("invalid observation hash in artifact")
    ranker_config = json.loads((root / "ranker_config.json").read_text(encoding="utf-8"))
    ranking = json.loads((root / "ranking.json").read_text(encoding="utf-8"))
    if ranking.get("ranker_config_hash") != canonical_hash(ranker_config):
        raise ValueError("ranker config hash mismatch")
    selected = int((root / "selected_hypothesis.txt").read_text(encoding="utf-8"))
    if ranking.get("selected_hypothesis") != selected:
        raise ValueError("selected hypothesis disagrees with ranking")
    rows = ranking.get("hypotheses", [])
    identifiers = [int(row["id"]) for row in rows]
    if not rows or len(identifiers) != len(set(identifiers)) or selected not in identifiers:
        raise ValueError("invalid hypothesis table")
    if len(rows) != int(run_identity["sampler"]["num_hypotheses"]):
        raise ValueError("hypothesis count disagrees with sampler config")
    expected_order = sorted(rows, key=lambda row: (-float(row["total"]), int(row["id"])))
    if rows != expected_order or selected != int(rows[0]["id"]):
        raise ValueError("ranking order or top-1 selection is invalid")
    if run_identity["hypothesis_seeds"] != [int(row["seed"]) for row in rows]:
        raise ValueError("hypothesis seeds disagree with run identity")
    if len(set(run_identity["hypothesis_seeds"])) != len(rows):
        raise ValueError("hypothesis seeds are not independent")
    fallback_count = sum(row["status"] == "fallback_initialization" for row in rows)
    retry_count = sum(int(row["retry_count"]) for row in rows)
    if run_identity["failure_count"] != fallback_count:
        raise ValueError("failure count disagrees with hypothesis table")
    if run_identity["retry_count"] != retry_count:
        raise ValueError("retry count disagrees with hypothesis table")
    initialization_root = root / "initialization"
    if (initialization_root / "INITIALIZATION_COMPLETE").is_file():
        initial, _, _ = load_initialization_artifact(initialization_root)
    else:
        if not run_identity["development_only"]:
            raise ValueError("production prediction requires complete camera initialization")
        initial, _ = load_trajectory(initialization_root)
    for row in rows:
        if not all(
            math.isfinite(float(row[name]))
            for name in ("total", "observation", "contact", "event", "motion")
        ):
            raise ValueError("ranking contains NaN/Inf")
        hypothesis_root = root / f"hypothesis_{int(row['id']):03d}"
        trajectory, _ = load_trajectory(hypothesis_root)
        graph = _load_graph(hypothesis_root / "contact_graph.npz")
        if graph.event_state.shape[:2] != trajectory.valid_mask.shape:
            raise ValueError("trajectory/contact graph batch-time mismatch")
        if row["status"] == "fallback_initialization":
            for name in initial.__dataclass_fields__:
                expected = getattr(initial, name)
                observed = getattr(trajectory, name)
                if isinstance(expected, torch.Tensor) and not torch.equal(expected, observed):
                    raise ValueError("fallback trajectory differs from initializer")
        terms = json.loads((hypothesis_root / "ranking_terms.json").read_text(encoding="utf-8"))
        if any(abs(float(terms[name]) - float(row[name])) > 1e-10 for name in terms):
            raise ValueError("per-hypothesis ranking terms disagree with ranking table")
        round_root = hypothesis_root / "rounds"
        round_directories = sorted(round_root.glob("round_*")) if round_root.is_dir() else []
        if row["status"] == "ok" and len(round_directories) != int(
            run_identity["sampler"]["rounds"]
        ):
            raise ValueError("successful hypothesis has incomplete round artifacts")
        for round_index, round_directory in enumerate(round_directories):
            if round_directory.name != f"round_{round_index:03d}":
                raise ValueError("round artifact indices are not contiguous")
            round_trajectory, _ = load_trajectory(round_directory)
            round_graph = _load_graph(round_directory / "contact_graph.npz")
            if round_graph.event_state.shape[:2] != round_trajectory.valid_mask.shape:
                raise ValueError("round trajectory/contact graph mismatch")
            objective = json.loads(
                (round_directory / "runtime_objective.json").read_text(encoding="utf-8")
            )
            objective_finite = all(math.isfinite(float(value)) for value in objective.values())
            if not objective or not objective_finite:
                raise ValueError("round runtime objective is empty or non-finite")
    return {
        "schema_version": "dcg_prediction_artifact_validation_v1",
        "clip_id": ranking["clip_id"],
        "hypotheses": len(rows),
        "selected_hypothesis": selected,
        "failure_count": fallback_count,
        "retry_count": retry_count,
        "run_identity_hash": stored_identity_hash,
    }


def write_prediction_artifact(
    output: str | Path,
    clip_id: str,
    initialization: tuple[Any, dict[str, Any]],
    hypotheses: list[Hypothesis],
    run_identity: dict[str, Any],
    *,
    input_manifest: dict[str, Any],
    observation_hashes: dict[str, str],
    ranker_config: dict[str, Any],
) -> Path:
    validate_run_identity(run_identity)
    if not hypotheses:
        raise ValueError("at least one hypothesis is required")
    if input_manifest.get("clip_id") not in {None, clip_id}:
        raise ValueError("input manifest clip does not match artifact clip")
    if not all(_SHA256.fullmatch(value) for value in observation_hashes.values()):
        raise ValueError("observation identities must be exact SHA-256 values")
    if not run_identity["development_only"]:
        if not input_manifest or not observation_hashes or not ranker_config:
            raise ValueError(
                "production artifacts require manifest, observations and ranker config"
            )
        if input_manifest.get("manifest_sha256") != run_identity["manifest_sha256"]:
            raise ValueError("input manifest identity mismatch")
        if ranker_config.get("fit_split") != "validation":
            raise ValueError("production ranker must be fit on validation")
        if ranker_config.get("uses_ground_truth") is not False:
            raise ValueError("production ranker must explicitly exclude ground truth")
    output = Path(output)
    root = output / clip_id
    if root.exists():
        raise FileExistsError(f"immutable prediction already exists: {root}")
    output.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{clip_id}.", dir=output))
    root = temporary
    (root / "input_manifest.json").write_text(
        json.dumps(input_manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (root / "observation_hashes.json").write_text(
        json.dumps(observation_hashes, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (root / "ranker_config.json").write_text(
        json.dumps(ranker_config, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    if len(initialization) == 2:
        if not run_identity["development_only"]:
            raise ValueError("production prediction requires camera initialization")
        initial_state, initial_metadata = initialization
        save_trajectory(initial_state, root / "initialization", initial_metadata)
    elif len(initialization) == 3:
        initial_state, initial_camera, initial_metadata = initialization
        initial_metadata = dict(initial_metadata)
        source_hashes = initial_metadata.pop("source_hashes", None)
        # Loaded initialization metadata carries the identity of the source artifact.
        # The embedded copy gets new trajectory/camera/source file hashes and must have
        # its identity recomputed by save_initialization_artifact.
        initial_metadata.pop("metadata_identity_sha256", None)
        if source_hashes is None:
            raise ValueError("complete initialization requires source hashes")
        save_initialization_artifact(
            root / "initialization",
            initial_state,
            initial_camera,
            metadata=initial_metadata,
            source_hashes=source_hashes,
        )
    else:
        raise ValueError("initialization must contain state/metadata or state/camera/metadata")
    ranking = {"clip_id": clip_id, "hypotheses": []}
    for hypothesis in hypotheses:
        hypothesis_root = root / f"hypothesis_{hypothesis.identifier:03d}"
        save_trajectory(
            hypothesis.trajectory,
            hypothesis_root,
            {"seed": hypothesis.seed, "status": hypothesis.status},
        )
        graph = hypothesis.graph
        _save_graph(hypothesis_root / "contact_graph.npz", graph)
        (hypothesis_root / "ranking_terms.json").write_text(
            json.dumps(hypothesis.ranking_terms, sort_keys=True, indent=2), encoding="utf-8"
        )
        (hypothesis_root / "diagnostics.json").write_text(
            json.dumps(hypothesis.diagnostics, sort_keys=True, indent=2), encoding="utf-8"
        )
        for round_result in hypothesis.rounds:
            round_root = hypothesis_root / "rounds" / f"round_{round_result.round_index:03d}"
            save_trajectory(
                round_result.trajectory,
                round_root,
                {"round": round_result.round_index, "hypothesis_seed": hypothesis.seed},
            )
            _save_graph(round_root / "contact_graph.npz", round_result.graph)
            (round_root / "diagnostics.json").write_text(
                json.dumps(round_result.diagnostics, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            (round_root / "runtime_objective.json").write_text(
                json.dumps(round_result.runtime_objective, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
        ranking["hypotheses"].append(
            {
                "id": hypothesis.identifier,
                "seed": hypothesis.seed,
                "total": hypothesis.score,
                "status": hypothesis.status,
                "retry_count": hypothesis.retry_count,
                **hypothesis.ranking_terms,
            }
        )
    selected = hypotheses[0].identifier
    ranking["selected_hypothesis"] = selected
    ranking["ranker_config_hash"] = canonical_hash(ranker_config)
    (root / "ranking.json").write_text(
        json.dumps(ranking, sort_keys=True, indent=2), encoding="utf-8"
    )
    (root / "selected_hypothesis.txt").write_text(f"{selected}\n", encoding="utf-8")
    identity = dict(run_identity)
    identity["hypothesis_seeds"] = [hypothesis.seed for hypothesis in hypotheses]
    identity["failure_count"] = sum(
        hypothesis.status == "fallback_initialization" for hypothesis in hypotheses
    )
    identity["retry_count"] = sum(hypothesis.retry_count for hypothesis in hypotheses)
    identity["run_identity_hash"] = canonical_hash(identity)
    (root / "run_identity.json").write_text(
        json.dumps(identity, sort_keys=True, indent=2), encoding="utf-8"
    )
    try:
        validate_prediction_artifact(root, require_complete=False)
        (root / "ARTIFACT_COMPLETE").write_text("complete\n", encoding="utf-8")
        os.replace(root, output / clip_id)
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    return output / clip_id
