from __future__ import annotations

from typing import Any

import torch

from ..data.manifest import ClipManifest
from ..io.predictions import PredictionArtifact
from .contact import binary_contact_metrics, event_metrics
from .dynamics import dynamics_errors
from .geometric import region_metrics
from .uncertainty import risk_coverage_curve, spearman_risk_error


def completeness_gate(prediction: PredictionArtifact, manifest: ClipManifest) -> None:
    prediction.validate()
    if prediction.frame_ids.tolist() != manifest.frame_ids:
        raise ValueError(f"prediction is incomplete or reordered for {manifest.clip_id}")


def evaluate_prediction(
    prediction: PredictionArtifact,
    manifest: ClipManifest,
    ground_truth: dict[str, torch.Tensor],
    regions: dict[str, torch.Tensor],
) -> dict[str, Any]:
    completeness_gate(prediction, manifest)
    target_joints = ground_truth["joints_3d"]
    if target_joints.shape != prediction.joints_3d.shape:
        raise ValueError("ground-truth joints do not match prediction")
    output: dict[str, Any] = {"clip_id": manifest.clip_id, "frames": len(manifest.frame_ids)}
    geometry = region_metrics(prediction.joints_3d, target_joints, regions)
    output.update({key: float(value.mean()) for key, value in geometry.items()})
    output.update(
        {
            key: float(value.mean()) if value.numel() else None
            for key, value in dynamics_errors(
                prediction.joints_3d, target_joints, manifest.fps
            ).items()
        }
    )
    point_error = torch.linalg.vector_norm(prediction.joints_3d - target_joints, dim=-1)
    risk_joint = prediction.uncertainty
    output["risk_error_spearman"] = spearman_risk_error(point_error, risk_joint)
    output["aurc"] = risk_coverage_curve(point_error, risk_joint)["aurc"]
    if prediction.contacts is not None and "contacts" in ground_truth:
        output.update(
            {
                f"contact_{key}": value
                for key, value in binary_contact_metrics(
                    prediction.contacts, ground_truth["contacts"]
                ).items()
            }
        )
        output.update(
            {
                f"contact_{key}": value
                for key, value in event_metrics(
                    prediction.contacts, ground_truth["contacts"]
                ).items()
            }
        )
    return output
