"""Validation-only pairwise fitting for the GT-free deployment ranker."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

from dcg_sign4d.utils.hashing import canonical_hash, file_sha256

from .ranking import RankingWeights

FEATURES = ("observation", "contact", "event", "motion")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"clip_id", "hypothesis_id", "split", "validation_error", *FEATURES}
    if not rows or not required <= rows[0].keys():
        raise ValueError("ranker table lacks required columns")
    if any(row["split"] != "validation" for row in rows):
        raise ValueError("ranker fitting accepts validation rows only")
    identities = [(row["clip_id"], row["hypothesis_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate clip/hypothesis row in ranker table")
    return rows


def fit_ranker(
    input_csv: str | Path,
    output: str | Path,
    *,
    steps: int,
    learning_rate: float,
    l2_weight: float,
    minimum_pair_accuracy: float,
    seed: int,
    development_only: bool,
    config_sha256: str,
) -> dict[str, Any]:
    if steps < 1 or learning_rate <= 0 or l2_weight < 0:
        raise ValueError("invalid ranker optimization settings")
    if not 0 <= minimum_pair_accuracy <= 1:
        raise ValueError("minimum pair accuracy must lie in [0,1]")
    source = Path(input_csv)
    rows = _rows(source)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["clip_id"]].append(row)
    differences = []
    for clip_rows in grouped.values():
        for first_index, first in enumerate(clip_rows):
            for second in clip_rows[first_index + 1 :]:
                first_error = float(first["validation_error"])
                second_error = float(second["validation_error"])
                if first_error == second_error:
                    continue
                better, worse = (first, second) if first_error < second_error else (second, first)
                differences.append([float(better[name]) - float(worse[name]) for name in FEATURES])
    if not differences:
        raise ValueError("ranker fitting requires at least one unequal within-clip pair")
    torch.manual_seed(seed)
    delta = torch.tensor(differences, dtype=torch.float32)
    raw_weights = torch.nn.Parameter(torch.zeros(len(FEATURES)))
    optimizer = torch.optim.Adam([raw_weights], lr=learning_rate)
    history = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.softplus(-(delta @ raw_weights)).mean()
        loss = loss + l2_weight * raw_weights.square().sum()
        loss.backward()
        optimizer.step()
        if step in {1, steps}:
            history.append({"step": step, "loss": float(loss.detach())})
    norm = raw_weights.detach().abs().sum().clamp_min(1e-8)
    weights = raw_weights.detach() / norm
    margins = delta @ weights
    accuracy = float((margins > 0).float().mean())
    ties = float((margins == 0).float().mean())
    gate_pass = accuracy >= minimum_pair_accuracy
    weight_payload = dict(zip(FEATURES, weights.tolist(), strict=True))
    # Exercise the deployment contract before serializing it.
    RankingWeights(**weight_payload)
    payload: dict[str, Any] = {
        "schema_version": "dcg_ranker_v1",
        "development_only": development_only,
        "fit_split": "validation",
        "fit_uses_validation_error": True,
        "use_ground_truth": False,
        "weights": weight_payload,
        "pairs": len(differences),
        "clips": len(grouped),
        "pairwise_accuracy": accuracy,
        "tie_rate": ties,
        "minimum_pair_accuracy": minimum_pair_accuracy,
        "gate_status": "PASS" if gate_pass else "FAIL",
        "optimization": {
            "steps": steps,
            "learning_rate": learning_rate,
            "l2_weight": l2_weight,
            "seed": seed,
            "history": history,
        },
        "source_sha256": file_sha256(source),
        "config_sha256": config_sha256,
    }
    payload["artifact_identity_sha256"] = canonical_hash(payload)
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"immutable ranker artifact exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.incomplete")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", "utf-8")
    os.replace(temporary, destination)
    if not gate_pass:
        raise RuntimeError("validation-fitted ranker failed its preregistered accuracy gate")
    return payload


def load_frozen_ranker(
    path: str | Path, *, allow_development: bool = False
) -> tuple[RankingWeights, dict[str, Any]]:
    payload = json.loads(Path(path).read_text("utf-8"))
    identity = payload.pop("artifact_identity_sha256", None)
    if identity != canonical_hash(payload):
        raise ValueError("ranker artifact identity mismatch")
    payload["artifact_identity_sha256"] = identity
    if payload.get("schema_version") != "dcg_ranker_v1":
        raise ValueError("unknown ranker artifact schema")
    if payload.get("gate_status") != "PASS":
        raise PermissionError("ranker artifact did not pass validation gate")
    if payload.get("development_only") and not allow_development:
        raise PermissionError("development ranker cannot enter production inference")
    if payload.get("fit_split") != "validation" or payload.get("use_ground_truth") is not False:
        raise ValueError("ranker artifact violates deployment fitting contract")
    return RankingWeights(**payload["weights"]), payload
