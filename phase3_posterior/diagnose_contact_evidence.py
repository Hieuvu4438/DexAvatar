"""Probe whether 2D observation relations explain unseen-signer contact labels.

This diagnostic never trains a Phase 3 checkpoint and never reads Lane-L.  It
uses signer 08 only to choose a classifier threshold, then reports untouched
signers 01/02.  A recovery model is authorized only if observation evidence
materially improves the held-out F1 over relation-only evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.dataset import _keypoints_in_model_coordinates
from phase3_posterior.data.cache_schema import load_index, load_relation_sidecar
from phase3_posterior.geometry.relation_anchors import (
    build_observation_edge_features,
)
from phase3_posterior.provenance import atomic_json, sha256_file
from phase3_posterior.relation_evaluation import relation_edge_masks


def _counts(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    tp = int(np.count_nonzero(target & prediction))
    fp = int(np.count_nonzero(~target & prediction))
    fn = int(np.count_nonzero(target & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _signer(entry) -> int:
    return int(entry.signer.rsplit("_", 1)[-1])


def _clip_rows(entry) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clip = load_cache_clip(entry.clip_path)
    relation = load_relation_sidecar(entry.relation_path)
    keypoints = torch.from_numpy(
        _keypoints_in_model_coordinates(clip, slice(0, len(clip.frame_names)))
    ).float()
    observation, observation_valid = build_observation_edge_features(
        keypoints,
        torch.from_numpy(clip.keypoint_valid).bool(),
        torch.from_numpy(clip.u0_reliability).float(),
        torch.from_numpy(relation.edge_index).long(),
        torch.from_numpy(clip.reprojection_residual_2d).float() * 10.0,
    )
    edge_index = torch.from_numpy(relation.edge_index).long()
    hand_body = relation_edge_masks(edge_index)["hand_body"].numpy()
    valid = relation.contact_valid & observation_valid.numpy() & hand_body[None]
    frame, edge = np.nonzero(valid)
    edge_identity = edge.astype(np.float32)[:, None] / max(
        relation.edge_index.shape[1] - 1, 1
    )
    time = frame.astype(np.float32)[:, None] / max(len(clip.frame_names) - 1, 1)
    relation_features = np.concatenate(
        (relation.edge_features[valid], edge_identity, time), axis=1
    )
    observation_features = np.concatenate(
        (relation_features, observation.numpy()[valid]), axis=1
    )
    return relation_features, observation_features, relation.contact_target[valid]


def _collect(
    entries,
    signers: set[int],
    *,
    sample_negatives: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    relation_rows, observation_rows, labels = [], [], []
    clips = positives = negatives = 0
    for entry in entries:
        if entry.source != "how2sign" or _signer(entry) not in signers:
            continue
        relation, observation, target = _clip_rows(entry)
        if sample_negatives:
            positive_index = np.flatnonzero(target)
            negative_index = np.flatnonzero(~target)
            # Preserve all rare positives. Mix nearest-surface hard negatives
            # with random negatives so the probe cannot win on easy imbalance.
            hard_count = min(32, len(negative_index))
            hard = negative_index[
                np.argsort(relation[negative_index, 11])[:hard_count]
            ]
            remaining = np.setdiff1d(negative_index, hard, assume_unique=False)
            random_count = min(32, len(remaining))
            random = (
                rng.choice(remaining, size=random_count, replace=False)
                if random_count
                else np.empty(0, dtype=np.int64)
            )
            keep = np.concatenate((positive_index, hard, random))
            relation, observation, target = (
                relation[keep],
                observation[keep],
                target[keep],
            )
        relation_rows.append(relation)
        observation_rows.append(observation)
        labels.append(target)
        clips += 1
        positives += int(target.sum())
        negatives += int((~target).sum())
        if clips % 500 == 0:
            print(
                json.dumps(
                    {
                        "event": "contact_probe_loading",
                        "clips": clips,
                        "signers": sorted(signers),
                    }
                ),
                flush=True,
            )
    if not relation_rows:
        raise ValueError(f"No How2Sign rows for signers {sorted(signers)}")
    return (
        np.concatenate(relation_rows),
        np.concatenate(observation_rows),
        np.concatenate(labels).astype(bool),
        {"clips": clips, "positives": positives, "negatives": negatives},
    )


def _fit_and_score(
    train: np.ndarray,
    train_target: np.ndarray,
    calibration: np.ndarray,
    calibration_target: np.ndarray,
    evaluation: np.ndarray,
    evaluation_target: np.ndarray,
    seed: int,
) -> dict:
    positives = max(int(train_target.sum()), 1)
    negatives = max(int((~train_target).sum()), 1)
    weights = np.where(train_target, len(train_target) / (2 * positives), len(train_target) / (2 * negatives))
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(train, train_target, sample_weight=weights)
    calibration_score = model.predict_proba(calibration)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 181)
    calibration_results = [
        _counts(calibration_target, calibration_score >= threshold)
        for threshold in thresholds
    ]
    selected_index = int(np.argmax([item["f1"] for item in calibration_results]))
    threshold = float(thresholds[selected_index])
    evaluation_score = model.predict_proba(evaluation)[:, 1]
    return {
        "threshold_selected_on_signer_08": threshold,
        "calibration_signer_08": calibration_results[selected_index],
        "untouched_signers_01_02": _counts(
            evaluation_target, evaluation_score >= threshold
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_entries = load_index(args.train_index)
    val_entries = load_index(args.val_index)
    train_relation, train_observation, train_target, train_counts = _collect(
        train_entries, {3, 5}, sample_negatives=True, seed=args.seed
    )
    cal_relation, cal_observation, cal_target, cal_counts = _collect(
        train_entries, {8}, sample_negatives=False, seed=args.seed
    )
    eval_relation, eval_observation, eval_target, eval_counts = _collect(
        val_entries, {1, 2}, sample_negatives=False, seed=args.seed
    )
    relation_only = _fit_and_score(
        train_relation,
        train_target,
        cal_relation,
        cal_target,
        eval_relation,
        eval_target,
        args.seed,
    )
    observation_aware = _fit_and_score(
        train_observation,
        train_target,
        cal_observation,
        cal_target,
        eval_observation,
        eval_target,
        args.seed,
    )
    old_f1 = float(relation_only["untouched_signers_01_02"]["f1"])
    new_f1 = float(observation_aware["untouched_signers_01_02"]["f1"])
    report = {
        "schema": "phase3-contact-evidence-probe-v1",
        "lane_l_reads": 0,
        "split_protocol": {
            "fit_signers": [3, 5],
            "threshold_signers": [8],
            "untouched_evaluation_signers": [1, 2],
        },
        "counts": {
            "fit": train_counts,
            "threshold": cal_counts,
            "evaluation": eval_counts,
        },
        "relation_only": relation_only,
        "observation_aware": observation_aware,
        "observation_absolute_f1_gain": new_f1 - old_f1,
        "recovery_training_authorized": new_f1 >= 0.60 and new_f1 > old_f1,
        "train_index_sha256": sha256_file(args.train_index),
        "val_index_sha256": sha256_file(args.val_index),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
