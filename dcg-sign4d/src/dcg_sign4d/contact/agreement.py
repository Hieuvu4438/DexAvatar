"""Independent double-annotation agreement audit for contact events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .labels import ContactAnnotation


@dataclass(frozen=True)
class AgreementThresholds:
    minimum_events: int
    minimum_edge_agreement: float
    minimum_mean_interval_iou: float
    maximum_mean_boundary_error_frames: float

    def validate(self) -> AgreementThresholds:
        if self.minimum_events < 1:
            raise ValueError("minimum_events must be positive")
        if not 0 <= self.minimum_edge_agreement <= 1:
            raise ValueError("edge agreement threshold must lie in [0,1]")
        if not 0 <= self.minimum_mean_interval_iou <= 1:
            raise ValueError("interval IoU threshold must lie in [0,1]")
        if self.maximum_mean_boundary_error_frames < 0:
            raise ValueError("boundary error threshold cannot be negative")
        return self


def _interval_iou(first: ContactAnnotation, second: ContactAnnotation) -> float:
    intersection = max(
        0,
        min(first.release_frame, second.release_frame)
        - max(first.onset_frame, second.onset_frame)
        + 1,
    )
    union = (
        max(first.release_frame, second.release_frame)
        - min(first.onset_frame, second.onset_frame)
        + 1
    )
    return intersection / union


def audit_double_annotations(
    records: list[dict[str, Any]], thresholds: AgreementThresholds
) -> dict[str, Any]:
    """Audit exactly two pre-adjudication annotations for each clip/event ID."""

    thresholds.validate()
    grouped: dict[tuple[str, str], list[ContactAnnotation]] = {}
    for index, record in enumerate(records, start=1):
        event_id = str(record.get("event_id", "")).strip()
        if not event_id:
            raise ValueError(f"annotation row {index} has no event_id")
        annotation = ContactAnnotation(
            clip_id=record["clip_id"],
            edge=tuple(record["edge"]),
            onset_frame=int(record["onset_frame"]),
            hold_start=int(record["hold_start"]),
            hold_end=int(record["hold_end"]),
            release_frame=int(record["release_frame"]),
            uncertain=bool(record["uncertain"]),
            annotator_id=record["annotator_id"],
            confidence=float(record["confidence"]),
            adjudication_result=record.get("adjudication_result"),
        ).validate()
        if annotation.adjudication_result is not None:
            raise ValueError("agreement audit requires independent pre-adjudication rows")
        grouped.setdefault((annotation.clip_id, event_id), []).append(annotation)
    if not grouped:
        raise ValueError("empty contact annotation set")

    edge_matches: list[float] = []
    uncertain_matches: list[float] = []
    interval_ious: list[float] = []
    boundary_errors: list[float] = []
    per_event: list[dict[str, Any]] = []
    annotators: set[str] = set()
    for (clip_id, event_id), pair in sorted(grouped.items()):
        if len(pair) != 2 or pair[0].annotator_id == pair[1].annotator_id:
            raise ValueError(f"{clip_id}/{event_id} requires exactly two distinct annotators")
        pair = sorted(pair, key=lambda item: item.annotator_id)
        annotators.update(item.annotator_id for item in pair)
        edge_match = pair[0].edge == pair[1].edge
        uncertain_match = pair[0].uncertain == pair[1].uncertain
        edge_matches.append(float(edge_match))
        uncertain_matches.append(float(uncertain_match))
        event_report = {
            "clip_id": clip_id,
            "event_id": event_id,
            "annotators": [item.annotator_id for item in pair],
            "edge_match": edge_match,
            "uncertain_match": uncertain_match,
            "edges": [list(item.edge) for item in pair],
        }
        if edge_match:
            iou = _interval_iou(pair[0], pair[1])
            onset_error = abs(pair[0].onset_frame - pair[1].onset_frame)
            release_error = abs(pair[0].release_frame - pair[1].release_frame)
            interval_ious.append(iou)
            boundary_errors.extend((onset_error, release_error))
            event_report.update(
                {
                    "interval_iou": iou,
                    "onset_error_frames": onset_error,
                    "release_error_frames": release_error,
                }
            )
        per_event.append(event_report)

    edge_agreement = float(np.mean(edge_matches))
    uncertain_agreement = float(np.mean(uncertain_matches))
    mean_iou = float(np.mean(interval_ious)) if interval_ious else None
    mean_boundary = float(np.mean(boundary_errors)) if boundary_errors else None
    checks = {
        "minimum_events": len(grouped) >= thresholds.minimum_events,
        "edge_agreement": edge_agreement >= thresholds.minimum_edge_agreement,
        "interval_iou": mean_iou is not None and mean_iou >= thresholds.minimum_mean_interval_iou,
        "boundary_error": mean_boundary is not None
        and mean_boundary <= thresholds.maximum_mean_boundary_error_frames,
    }
    return {
        "schema_version": "dcg_contact_annotation_agreement_v1",
        "events": len(grouped),
        "annotators": sorted(annotators),
        "edge_agreement": edge_agreement,
        "uncertain_agreement": uncertain_agreement,
        "mean_interval_iou": mean_iou,
        "mean_boundary_error_frames": mean_boundary,
        "matched_edge_events": len(interval_ious),
        "checks": checks,
        "agreement_gate_pass": all(checks.values()),
        "thresholds": thresholds.__dict__,
        "per_event": per_event,
    }
