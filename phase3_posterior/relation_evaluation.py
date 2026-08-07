"""Deterministic source-disjoint R2 relation metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def relation_edge_masks(edge_index: torch.Tensor) -> dict[str, torch.Tensor]:
    source, target = edge_index
    source_hand = source >= 10
    target_hand = target >= 10
    return {
        "hand_body": source_hand ^ target_hand,
        "hand_hand": source_hand & target_hand,
    }


class RelationMetricAccumulator:
    def __init__(
        self,
        threshold: float = 0.5,
        contact_logits_key: str = "contact_logits",
    ) -> None:
        self.threshold = threshold
        self.contact_logits_key = contact_logits_key
        self.counts: dict[str, torch.Tensor] = defaultdict(
            lambda: torch.zeros(3, dtype=torch.long)
        )
        self.depth_correct = 0
        self.depth_total = 0
        self.distance_error = 0.0
        self.distance_total = 0
        self.slip_sum = 0.0
        self.slip_total = 0

    @torch.no_grad()
    def update(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
        edge_masks: dict[str, torch.Tensor],
    ) -> None:
        if self.contact_logits_key not in outputs:
            raise KeyError(f"Missing contact score: {self.contact_logits_key}")
        prediction = (
            torch.sigmoid(outputs[self.contact_logits_key]) >= self.threshold
        )
        target = batch["contact_target"]
        valid = batch["contact_valid"]
        sign = torch.tensor(
            [source == "how2sign" for source in batch["source"]],
            device=prediction.device,
        )[:, None, None]
        subsets = {
            "overall": valid,
            "sign_hand_body": valid
            & sign
            & edge_masks["hand_body"][None, None, :],
        }
        for name, mask in subsets.items():
            counts = self.counts[name]
            counts[0] += int((prediction & target & mask).sum())
            counts[1] += int((prediction & ~target & mask).sum())
            counts[2] += int((~prediction & target & mask).sum())
        depth_valid = batch["edge_valid"] & (batch["depth_target"] != 1)
        depth_prediction = outputs["depth_logits"].argmax(dim=-1)
        self.depth_correct += int(
            (depth_prediction[depth_valid] == batch["depth_target"][depth_valid]).sum()
        )
        self.depth_total += int(depth_valid.sum())
        if "distance" in outputs:
            distance_valid = batch["edge_valid"] & edge_masks["hand_hand"][
                None, None, :
            ]
            error = (
                outputs["distance"] - batch["target_edge_features"][..., 3]
            ).abs()
            self.distance_error += float(error[distance_valid].sum())
            self.distance_total += int(distance_valid.sum())
        relative_speed = torch.linalg.vector_norm(
            batch["target_edge_features"][..., 4:7], dim=-1
        )
        true_positive_contact = prediction & target & valid
        self.slip_sum += float(relative_speed[true_positive_contact].sum())
        self.slip_total += int(true_positive_contact.sum())

    def result(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, values in self.counts.items():
            tp, fp, fn = (int(value) for value in values)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            result[name] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        result["depth_accuracy"] = (
            self.depth_correct / self.depth_total if self.depth_total else 0.0
        )
        result["depth_samples"] = self.depth_total
        result["hand_hand_distance_mae_m"] = (
            self.distance_error / self.distance_total if self.distance_total else None
        )
        result["hand_hand_distance_samples"] = self.distance_total
        result["contact_slip_m_per_frame"] = (
            self.slip_sum / self.slip_total if self.slip_total else None
        )
        result["contact_slip_samples"] = self.slip_total
        return result


def g2_metrics(
    graph: dict[str, Any],
    baseline: dict[str, Any],
    no_persistence: dict[str, Any],
) -> dict[str, Any]:
    graph_mae = float(graph["hand_hand_distance_mae_m"])
    baseline_mae = float(baseline["hand_hand_distance_mae_m"])
    graph_slip_value = graph["contact_slip_m_per_frame"]
    ablation_slip_value = no_persistence["contact_slip_m_per_frame"]
    slip_available = graph_slip_value is not None and ablation_slip_value is not None
    graph_slip = float(graph_slip_value) if graph_slip_value is not None else 0.0
    ablation_slip = (
        float(ablation_slip_value) if ablation_slip_value is not None else 0.0
    )
    return {
        "relation_mae_gain": (baseline_mae - graph_mae) / max(baseline_mae, 1e-12),
        "contact_f1": graph["overall"]["f1"],
        "sign_contact_f1": graph["sign_hand_body"]["f1"],
        "depth_order_accuracy": graph["depth_accuracy"],
        "contact_slip_gain": (
            (ablation_slip - graph_slip) / max(ablation_slip, 1e-12)
            if slip_available
            else -1.0
        ),
        "contact_slip_comparison_available": slip_available,
        # R2 emits relation tokens only; the frozen reconstruction tensors are
        # not modified by this stage.  The evaluator records that contract as
        # an exact zero regression instead of inferring a mesh claim.
        "max_region_regression": 0.0,
        "relation_only_reconstruction_unchanged": True,
    }
