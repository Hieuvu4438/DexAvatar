"""Decode selected rotations to evaluator-compatible SMPL-X meshes and PKLs."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from cusp_sl.config import load_config


def use_bit_exact_base_copy(
    selected_index: int, prediction_manifest: dict
) -> bool:
    """Copy legacy base only when candidate zero really is the legacy base."""
    return (
        selected_index == 0
        and prediction_manifest.get("input_manifest_role")
        != "frozen_strong_a1_derived_cache"
    )


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    value = np.asarray(vertices, dtype=np.float32)
    if value.shape != (10475, 3):
        raise ValueError(f"{path}: expected 10475 vertices, got {value.shape}")
    return value


def write_obj_from_template(template: Path, output: Path, vertices: np.ndarray) -> None:
    lines = template.read_text(encoding="utf-8").splitlines()
    result, index = [], 0
    for line in lines:
        if line.startswith("v "):
            x, y, z = vertices[index]
            result.append(f"v {x:.10f} {y:.10f} {z:.10f}")
            index += 1
        else:
            result.append(line)
    if index != 10475:
        raise ValueError(f"Template {template} contains {index} vertices")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(result) + "\n", encoding="utf-8")


def load_model(asset: Path, device: torch.device):
    fitting = Path(__file__).resolve().parents[1] / "dexavatar_fitting"
    sys.path.insert(0, str(fitting))
    from rewrite_body_model import SMPLX
    model = SMPLX(
        model_path=str(asset), ext="pkl", use_face_contour=True,
        flat_hand_mean=True, use_pca=False, num_betas=10,
        num_expression_coeffs=10, create_body_pose=True,
    )
    return model.to(device).eval()


def model_kwargs(records: list[dict], pose: np.ndarray, device) -> dict[str, torch.Tensor]:
    keys = ("betas", "global_orient", "jaw_pose", "leye_pose", "reye_pose", "expression", "transl")
    output = {
        key: torch.from_numpy(np.concatenate([np.asarray(record[key], dtype=np.float32).reshape(1, -1) for record in records], axis=0)).to(device)
        for key in keys
    }
    output["body_pose"] = torch.from_numpy(pose[:, :21].reshape(len(records), 63)).to(device)
    output["left_hand_pose"] = torch.from_numpy(pose[:, 21:36].reshape(len(records), 45)).to(device)
    output["right_hand_pose"] = torch.from_numpy(pose[:, 36:51].reshape(len(records), 45)).to(device)
    return output


@torch.no_grad()
def decode(model, records: list[dict], pose: np.ndarray, device, batch_size: int) -> np.ndarray:
    values = []
    for start in range(0, len(records), batch_size):
        stop = min(start + batch_size, len(records))
        kwargs = model_kwargs(records[start:stop], pose[start:stop], device)
        vertices = model(**kwargs).vertices.detach().float().cpu().numpy()
        # Preserve the 180-degree x rotation used by the released fitting code.
        vertices[..., 1:] *= -1.0
        values.append(vertices)
    return np.concatenate(values, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose a new empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    prediction_manifest_path = args.predictions / "manifest.json"
    prediction_manifest = json.loads(
        prediction_manifest_path.read_text(encoding="utf-8")
    )
    declared_entries = prediction_manifest.get("summaries")
    if declared_entries is None and isinstance(
        prediction_manifest.get("clips"), list
    ):
        declared_entries = prediction_manifest["clips"]
    declared = {
        str(item["clip_id"]): item
        for item in (declared_entries or [])
    }
    prediction_paths = sorted((args.predictions / "clips").glob("*.npz"))
    if not declared:
        raise ValueError("Prediction manifest has no declared clip summaries")
    if {path.stem for path in prediction_paths} != set(declared):
        raise ValueError("Prediction files and manifest clip sets differ")
    model = None
    total = 0
    summaries = []
    for prediction_path in prediction_paths:
        prediction_hash = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
        declared_hash = declared[prediction_path.stem].get(
            "prediction_sha256"
        ) or declared[prediction_path.stem].get("filtered_prediction_sha256")
        if prediction_hash != declared_hash:
            raise ValueError(f"Prediction hash mismatch: {prediction_path}")
        with np.load(prediction_path, allow_pickle=False) as payload:
            sign = str(payload["clip_id"].item())
            names = payload["frame_names"].astype(str).tolist()
            selected_pose = payload["selected_axis_angle"].astype(np.float32)
            selected_index = int(payload["selected_index"].item())
            candidate_valid = payload["candidate_valid"].astype(bool)
        if sign != prediction_path.stem or sign not in declared:
            raise ValueError(f"Prediction clip ID mismatch: {prediction_path}")
        if len(names) != int(declared[sign]["frames"]):
            raise ValueError(f"Prediction frame count mismatch: {prediction_path}")
        if (
            selected_index < 0
            or selected_index >= len(candidate_valid)
            or not bool(candidate_valid[selected_index])
        ):
            raise ValueError(f"Prediction selected invalid candidate: {prediction_path}")
        if selected_pose.shape != (len(names), 51, 3) or not np.isfinite(
            selected_pose
        ).all():
            raise ValueError(f"Prediction has invalid selected pose: {prediction_path}")
        source_root = Path(config.protocol.baseline_root) / sign / "smplifyx"
        result_paths = [source_root / "results" / f"{name}.pkl" for name in names]
        mesh_paths = [source_root / "meshes" / f"{name}.obj" for name in names]
        records = []
        for path in result_paths:
            with path.open("rb") as handle:
                records.append(pickle.load(handle, encoding="latin1"))
        destination = args.output / sign / "smplifyx"
        if use_bit_exact_base_copy(selected_index, prediction_manifest):
            for source_pkl, source_mesh in zip(result_paths, mesh_paths):
                (destination / "results").mkdir(parents=True, exist_ok=True)
                (destination / "meshes").mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_pkl, destination / "results" / source_pkl.name)
                shutil.copy2(source_mesh, destination / "meshes" / source_mesh.name)
            mode = "bit_exact_base_copy"
        else:
            if model is None:
                asset = Path(config.protocol.smplx_model_folder) / "smplx" / "SMPLX_NEUTRAL.pkl"
                model = load_model(asset, device)
            base_pose = np.concatenate(
                [
                    np.concatenate((record["body_pose"].reshape(21, 3), record["left_hand_pose"].reshape(15, 3), record["right_hand_pose"].reshape(15, 3)), axis=0)[None]
                    for record in records
                ], axis=0,
            ).astype(np.float32)
            decoded_base = decode(model, records, base_pose, device, args.batch_size)
            decoded_edit = decode(model, records, selected_pose, device, args.batch_size)
            for index, (name, record, template) in enumerate(zip(names, records, mesh_paths)):
                # The released mesh renderer did not always serialize the same
                # orientation candidate as its PKL.  Transfer only the SMPL-X
                # deformation delta, keeping the released base mesh as identity.
                original = load_obj_vertices(template)
                vertices = original + (decoded_edit[index] - decoded_base[index])
                record = dict(record)
                record["body_pose"] = selected_pose[index, :21].reshape(1, 63)
                record["body_pose_fore"] = record["body_pose"][:, :45].copy()
                record["body_pose_op"] = record["body_pose"][:, 45:].copy()
                record["left_hand_pose"] = selected_pose[index, 21:36].reshape(1, 45)
                record["right_hand_pose"] = selected_pose[index, 36:51].reshape(1, 45)
                pkl_output = destination / "results" / f"{name}.pkl"
                pkl_output.parent.mkdir(parents=True, exist_ok=True)
                with pkl_output.open("wb") as handle:
                    pickle.dump(record, handle, protocol=2)
                write_obj_from_template(template, destination / "meshes" / f"{name}.obj", vertices)
            mode = "smplx_deformation_delta_transfer"
        total += len(names)
        summaries.append({
            "sign": sign,
            "frames": len(names),
            "selected_index": selected_index,
            "prediction_sha256": prediction_hash,
            "render_mode": mode,
        })
        print(f"[render] {sign}: {len(names)} frames ({mode})")
    if total != int(prediction_manifest.get("frames", -1)):
        raise ValueError("Rendered coverage differs from prediction manifest")
    declared_clip_count = (
        len(declared_entries)
        if isinstance(declared_entries, list)
        else int(prediction_manifest.get("clips", -1))
    )
    if len(summaries) != declared_clip_count:
        raise ValueError("Rendered clip count differs from prediction manifest")
    if total != config.protocol.expected_frames:
        raise ValueError(f"Rendered {total} != expected {config.protocol.expected_frames}")
    report = {
        "frames": total,
        "signs": len(summaries),
        "source_predictions": str(args.predictions.resolve()),
        "source_predictions_manifest_sha256": hashlib.sha256(
            prediction_manifest_path.read_bytes()
        ).hexdigest(),
        "input_manifest_role": prediction_manifest.get(
            "input_manifest_role", "legacy_unspecified"
        ),
        "clips": summaries,
    }
    (args.output / "render_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"frames": total, "signs": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
