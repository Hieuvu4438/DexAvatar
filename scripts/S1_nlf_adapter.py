#!/usr/bin/env python3
"""NLF -> SMPLer-X schema adapter.

Runs NLF (Neural Localizer Fields) inference on a folder of images and writes
one pkl per image to ``${out_folder}/nlf/smplx/{base_name}.pkl`` in the exact
schema that ``SMPLer-X/main/inference.py`` produces.

That schema is consumed at ``dexavatar_fitting/smplifyx/data_parser.py:428-432``
via ``--smplx_init_dir nlf/smplx``. By writing byte-compatible pkls, every
downstream M4 variant, YAML config, and biomech/ensemble/temporal method keeps
working without modification.

Usage:
    python S1_nlf_adapter.py --img_folder <DIR> --out_folder <DIR> \
        --model_path <NLF_TORCHSCRIPT>

Required env vars (set by nlf/activate_nlf.sh):
    DATA_ROOT : default lookup path for the NLF TorchScript model.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
import torchvision


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_folder", required=True, help="Directory of input images")
    p.add_argument("--out_folder", required=True, help="Top-level output dir; pkls go to <out>/nlf/smplx/")
    p.add_argument(
        "--model_path",
        default=os.path.join(
            os.environ.get("DATA_ROOT", os.path.expanduser("~/nlf_data")),
            "models", "nlf_l_multi3.torchscript",
        ),
        help="Path to NLF TorchScript bundle (default: $DATA_ROOT/models/nlf_l_multi3.torchscript)",
    )
    p.add_argument("--batch_size", type=int, default=1, help="NLF batch size (1 is safest)")
    p.add_argument("--device", default="cuda:0", help="Torch device")
    p.add_argument(
        "--default_focal",
        type=float,
        default=5000.0,
        help="Focal length written into every pkl (matches all YAMLs in dexavatar_fitting/cfg_files)",
    )
    p.add_argument(
        "--detector_threshold",
        type=float,
        default=0.3,
        help="NLF internal YOLO detector score threshold (passed to detect_smpl_batched)",
    )
    p.add_argument(
        "--smplerx_shared_dir",
        default="",
        help="Path to SMPLer-X shared dir (e.g. outputs/shared/SIGN/smplerx/smplx). "
             "If provided, camera params (transl, focal, princpt) are copied from "
             "here instead of being computed heuristically.",
    )
    p.add_argument(
        "--wilor_pkl",
        default="",
        help="Path to WiLoR wilor.pkl for this sign. If provided, hand poses "
             "are copied from WiLoR instead of NLF (which has poor hand predictions).",
    )
    return p.parse_args()


def list_images(folder):
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(IMG_EXTS))
    return [os.path.join(folder, n) for n in names]


def pick_largest_detection(per_image_dict):
    """Pick the largest of N detections by vertex-spread proxy.

    NLF's `detect_smpl_batched` returns a list of detected persons. When more
    than one person is in frame, we want the signer (usually the largest).
    `vertices3d` is in NLF units; we use its per-axis std as a stable
    size proxy that is rotation-invariant.
    """
    poses = per_image_dict["pose"]
    if len(poses) == 0:
        return None
    if len(poses) == 1:
        idx = 0
    else:
        verts = per_image_dict["vertices3d"]  # (n_persons, V, 3)
        spreads = verts.std(dim=1).sum(dim=1)  # (n_persons,)
        idx = int(torch.argmax(spreads).item())
    return {
        "pose":  poses[idx].detach(),
        "betas": per_image_dict["betas"][idx].detach(),
        "trans": per_image_dict["trans"][idx].detach(),
        "joints3d": per_image_dict["joints3d"][idx].detach(),
        "joints2d": per_image_dict["joints2d"][idx].detach(),
    }


def compute_transl_from_2d3d(joints3d, joints2d, W, H, focal_length):
    """DEPRECATED: Use SMPLer-X camera params from shared dir instead.
    Kept for standalone usage without shared data."""
    # Use only torso/head joints for robust height estimate
    torso_idxs = [0, 3, 6, 9, 12, 15]
    torso_3d = joints3d[torso_idxs, :]
    torso_2d = joints2d[torso_idxs, :]

    body_height_3d = float(torso_3d[:, 1].max() - torso_3d[:, 1].min())
    body_height_2d = float(torso_2d[:, 1].max() - torso_2d[:, 1].min())
    body_height_2d = max(body_height_2d, 1.0)

    # Default upper-body height for seated signer
    UPPER_BODY_M = 1.0
    transl_z = focal_length * UPPER_BODY_M / body_height_2d

    princpt = np.array([W / 2.0, H / 2.0], dtype=np.float64)
    pelvis_2d = joints2d[0, :2]
    transl_xy = (pelvis_2d - princpt) * transl_z / focal_length

    return np.array([float(transl_xy[0]), float(transl_xy[1]), float(transl_z)], dtype=np.float32)


def load_smplerx_camera_params(img_name, shared_smplerx_dir):
    """Look up SMPLer-X camera parameters for a given frame.

    Returns (transl, focal, princpt) or (None, None, None) if not found.
    """
    pkl_path = os.path.join(shared_smplerx_dir, f"{img_name}.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        return d.get("transl"), d.get("focal"), d.get("princpt")
    return None, None, None


def nlf_to_smplerx_pkl(det, joints3d, joints2d, img_name, W, H,
                       default_focal, shared_smplerx_dir=None,
                       wilor_data=None):
    """Convert NLF detection to SMPLer-X pkl schema.

    - Body pose + global_orient + jaw/eyes from NLF
    - Hand poses from WiLoR (much more accurate than NLF hands)
    - Betas from NLF, clipped to [-2, +2]
    - Camera params from SMPLer-X shared dir (fallback: computed)
    """
    pose = det["pose"]
    if pose.dim() == 2 and pose.shape[0] == 1:
        pose = pose[0]
    elif pose.dim() == 2 and pose.shape[1] == 165:
        if pose.shape[1] == 3:
            pose = pose[:, 0]

    # Camera parameters: prefer SMPLer-X (known-working), fall back to computed
    transl, focal, princpt = None, None, None
    if shared_smplerx_dir:
        transl, focal, princpt = load_smplerx_camera_params(img_name, shared_smplerx_dir)

    if transl is None:
        j3d = joints3d
        j2d = joints2d
        if hasattr(j3d, 'dim') and j3d.dim() == 2 and j3d.shape[0] == 1:
            j3d = j3d[0]
        if hasattr(j2d, 'dim') and j2d.dim() == 2 and j2d.shape[0] == 1:
            j2d = j2d[0]
        transl = compute_transl_from_2d3d(
            j3d.cpu().numpy() if hasattr(j3d, 'cpu') else np.asarray(j3d),
            j2d.cpu().numpy() if hasattr(j2d, 'cpu') else np.asarray(j2d),
            W, H, default_focal,
        )
        focal = np.array([default_focal, default_focal], dtype=np.float32)
        princpt = np.array([W / 2.0, H / 2.0], dtype=np.float32)

    # Hand poses: prefer WiLoR (high quality), fall back to NLF
    left_hand_pose = pose[25 * 3:40 * 3].cpu().numpy()   # NLF default
    right_hand_pose = pose[40 * 3:55 * 3].cpu().numpy()  # NLF default
    if wilor_data is not None:
        img_key = img_name + ".png"
        if img_key in wilor_data:
            hands = wilor_data[img_key].get("hands", [])
            for h in hands:
                aa = h.get("pred_mano_pose_axis_angle")
                if aa is not None:
                    aa_flat = np.asarray(aa, dtype=np.float32).reshape(-1)
                    if aa_flat.shape[0] == 45:
                        if h.get("is_right", 0) == 1.0:
                            right_hand_pose = aa_flat
                        else:
                            left_hand_pose = aa_flat

    # Clip betas to prevent extreme body shapes
    betas = np.clip(det["betas"].cpu().numpy(), -2.0, 2.0)

    # NLF body_pose: merge SMPLer-X values for joints NLF predicts as zero
    # (spine3=j9, L_foot=j10 are always zero → identity rotation → distorts mesh)
    body_pose = pose[3:3 + 21 * 3].cpu().numpy().copy()
    if shared_smplerx_dir:
        smp_transl, smp_focal, smp_princpt = load_smplerx_camera_params(
            img_name, shared_smplerx_dir)
        # Also load the full SMPLer-X pkl for body_pose
        smp_pkl = os.path.join(shared_smplerx_dir, f"{img_name}.pkl")
        if os.path.exists(smp_pkl):
            with open(smp_pkl, "rb") as f:
                smp_data = pickle.load(f)
            smp_bp = smp_data.get("body_pose")
            if smp_bp is not None:
                # Replace NLF zero joints with SMPLer-X values
                ZERO_JOINTS = [9, 10]  # spine3, L_foot
                for j in ZERO_JOINTS:
                    i = j * 3
                    if np.linalg.norm(body_pose[i:i+3]) < 0.001:
                        body_pose[i:i+3] = smp_bp[i:i+3]

    smplx = {
        "global_orient":   pose[0:3].cpu().numpy(),
        "body_pose":       body_pose,
        "left_hand_pose":  left_hand_pose,
        "right_hand_pose": right_hand_pose,
        "jaw_pose":        pose[22 * 3:23 * 3].cpu().numpy(),
        "leye_pose":       pose[23 * 3:24 * 3].cpu().numpy(),
        "reye_pose":       pose[24 * 3:25 * 3].cpu().numpy(),
        "betas":           betas,
        "expression":      np.zeros(10, dtype=np.float32),
        "transl":          transl.astype(np.float32),
        "focal":           np.asarray(focal, dtype=np.float32).flatten()[:2],
        "princpt":         np.asarray(princpt, dtype=np.float32).flatten()[:2],
    }
    return smplx


def main():
    args = parse_args()

    if not os.path.isdir(args.img_folder):
        print(f"[NLF] ERROR: img_folder does not exist: {args.img_folder}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.model_path):
        print(f"[NLF] ERROR: model_path does not exist: {args.model_path}", file=sys.stderr)
        print(f"[NLF]   Set NLF_MODEL_PATH or download the TorchScript bundle to this path.", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.join(args.out_folder, "nlf", "smplx")
    os.makedirs(out_dir, exist_ok=True)

    image_paths = list_images(args.img_folder)
    if not image_paths:
        print(f"[NLF] ERROR: no images found in {args.img_folder}", file=sys.stderr)
        sys.exit(1)
    print(f"[NLF] Found {len(image_paths)} images. Loading model from {args.model_path} ...")

    # IMPORTANT: torchvision must be imported before torch.jit.load for the
    # NLF TorchScript bundle to deserialize successfully (matches demo.ipynb).
    _ = torchvision.io

    model = torch.jit.load(args.model_path, map_location=args.device).eval()

    # Load WiLoR hand data if available
    wilor_images = None
    if args.wilor_pkl and os.path.isfile(args.wilor_pkl):
        with open(args.wilor_pkl, "rb") as f:
            wilor_data = pickle.load(f)
        wilor_images = wilor_data.get("images", {})
        print(f"[NLF] Loaded WiLoR hand poses for {len(wilor_images)} frames")
    elif args.wilor_pkl:
        print(f"[NLF] WiLoR pkl not found: {args.wilor_pkl}, using NLF hand poses")

    n_processed = 0
    n_skipped = 0
    t0 = time.time()

    bs = max(1, args.batch_size)
    for batch_start in range(0, len(image_paths), bs):
        batch_paths = image_paths[batch_start:batch_start + bs]
        frames = []
        sizes = []  # (W, H) per frame
        for p in batch_paths:
            img = torchvision.io.read_image(p)  # (3, H, W) uint8
            frames.append(img.to(args.device))
            H, W = img.shape[1], img.shape[2]
            sizes.append((W, H))

        with torch.inference_mode():
            out = model.detect_smpl_batched(
                torch.stack(frames, dim=0),
                model_name="smplx",
                detector_threshold=args.detector_threshold,
            )

        for path, (W, H), poses_per_image, betas_per_image, trans_per_image, verts_per_image, \
            joints3d_per_image, joints2d_per_image in zip(
            batch_paths,
            sizes,
            out["pose"],
            out["betas"],
            out["trans"],
            out["vertices3d"],
            out["joints3d"],
            out["joints2d"],
        ):
            per_image = {
                "pose": poses_per_image,
                "betas": betas_per_image,
                "trans": trans_per_image,
                "vertices3d": verts_per_image,
                "joints3d": joints3d_per_image,
                "joints2d": joints2d_per_image,
            }
            det = pick_largest_detection(per_image)
            base = os.path.splitext(os.path.basename(path))[0]
            pkl_path = os.path.join(out_dir, f"{base}.pkl")

            if det is None:
                n_skipped += 1
                print(f"[NLF]   skip (no detection): {path}", file=sys.stderr)
                continue

            smplx = nlf_to_smplerx_pkl(
                det, det["joints3d"], det["joints2d"],
                img_name=base, W=W, H=H,
                default_focal=args.default_focal,
                shared_smplerx_dir=args.smplerx_shared_dir,
                wilor_data=wilor_images,
            )
            with open(pkl_path, "wb") as f:
                pickle.dump(smplx, f)
            n_processed += 1

        if (batch_start // bs) % 10 == 0:
            elapsed = time.time() - t0
            done = batch_start + len(batch_paths)
            print(f"[NLF]   {done}/{len(image_paths)}  ({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(
        f"[NLF] Done. processed={n_processed} skipped={n_skipped} "
        f"out={out_dir} elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
