from __future__ import annotations

from pathlib import Path

import numpy as np

from signpccx.data.manifest import read_jsonl
from signpccx.io import atomic_write_json, sha256_file


def forward_shared_beta_sequences(
    frame_cache: Path,
    manifest_root: Path,
    identity_npz: Path,
    model_root: Path,
    output_root: Path,
    device: str = "cpu",
) -> dict[str, object]:
    import smplx
    import torch

    if output_root.exists() and any(output_root.rglob("*.npz")):
        raise FileExistsError(f"Refusing to overwrite fitted sequences: {output_root}")
    with np.load(identity_npz, allow_pickle=False) as archive:
        beta = np.asarray(archive["beta"], dtype=np.float32).reshape(1, 10)
    model = smplx.create(
        str(model_root), model_type="smplx", gender="neutral", num_betas=10,
        use_pca=False, use_face_contour=True,
    ).to(device)
    items = []
    for manifest_path in sorted(manifest_root.glob("*.jsonl")):
        records = read_jsonl(manifest_path)
        arrays: dict[str, list[np.ndarray]] = {
            key: [] for key in (
                "smplx_root_pose_aa", "smplx_body_pose_aa", "smplx_left_hand_pose_aa",
                "smplx_right_hand_pose_aa", "smplx_jaw_pose_aa", "smplx_expression",
                "smplx_translation",
            )
        }
        for record in records:
            cache = frame_cache / "clips" / record.sign / f"{record.source_frame_id:06d}.npz"
            with np.load(cache, allow_pickle=False) as archive:
                for key in arrays:
                    arrays[key].append(np.asarray(archive[key], dtype=np.float32))
        tensor = lambda key: torch.from_numpy(np.stack(arrays[key])).float().to(device)
        batch = len(records)
        zeros = torch.zeros((batch, 3), dtype=torch.float32, device=device)
        with torch.inference_mode():
            result = model(
                global_orient=tensor("smplx_root_pose_aa").reshape(batch, 3),
                body_pose=tensor("smplx_body_pose_aa").reshape(batch, 63),
                left_hand_pose=tensor("smplx_left_hand_pose_aa").reshape(batch, 45),
                right_hand_pose=tensor("smplx_right_hand_pose_aa").reshape(batch, 45),
                jaw_pose=tensor("smplx_jaw_pose_aa").reshape(batch, 3),
                leye_pose=zeros,
                reye_pose=zeros,
                expression=tensor("smplx_expression").reshape(batch, 10),
                betas=torch.from_numpy(beta).float().to(device).expand(batch, -1),
                transl=tensor("smplx_translation").reshape(batch, 3),
                return_verts=True,
            )
        vertices = result.vertices.detach().cpu().numpy().astype(np.float32)
        if vertices.shape != (batch, 10475, 3) or not np.isfinite(vertices).all():
            raise ValueError(f"{manifest_path.stem}: invalid shared-beta forward {vertices.shape}")
        destination = output_root / "clips" / manifest_path.stem / "mesh_parametric_final.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            mesh_parametric=vertices,
            faces=np.asarray(model.faces, dtype=np.int64),
            frame_ids=np.asarray([record.source_frame_id for record in records], dtype=np.int64),
            shared_beta=beta[0],
        )
        items.append({"sign": manifest_path.stem, "frames": batch, "sha256": sha256_file(destination)})
    report = {
        "schema_version": "signpccx.shared-beta-forward.v1",
        "device": device,
        "identity": str(identity_npz.resolve()),
        "model": str((model_root / "smplx" / "SMPLX_NEUTRAL.npz").resolve()),
        "signs": len(items),
        "frames": sum(item["frames"] for item in items),
        "items": items,
    }
    atomic_write_json(output_root / "run_manifest.json", report)
    return report

