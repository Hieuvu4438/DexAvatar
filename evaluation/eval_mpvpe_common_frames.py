#!/usr/bin/env python3
"""Evaluate MPVPE/TR-V2V on COMMON frames only (intersection across all methods).

This ensures a fair comparison by evaluating every method on exactly the same frames.

Usage:
    python eval_mpvpe_common_frames.py \
        --methods method_biomech method_hand2d method_hamer output_wilor \
        --method_names "DexAvatar-Biomech" "DexAvatar-Hand2D" "DexAvatar-HaMeR" "DexAvatar-WiLoR" \
        --output_csv outputs/mpvpe_common_frames.csv
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
SMPLX_MODEL = PROJECT / "SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.pkl"
SIGNS_TXT = PROJECT / "data/signs.txt"
GT_ROOT = PROJECT / "data/smplx_gt"
OUTPUT_BASE = PROJECT / "outputs"
UBODY_IDX = PROJECT / "dexavatar_fitting/assets/smplx_upper_body_minus_face_vidx.npy"
LHAND_IDX = PROJECT / "dexavatar_fitting/assets/smplx_left_hand_vidx.npy"
RHAND_IDX = PROJECT / "dexavatar_fitting/assets/smplx_right_hand_vidx.npy"


def load_obj_vertices(path: Path) -> np.ndarray:
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float32)


def load_smplx_joints(model_path: Path) -> dict:
    import pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f, encoding="latin1")
    J_raw = model["J_regressor"]
    if hasattr(J_raw, "todense"):
        J_raw = J_raw.todense()
    J_regressor = np.array(J_raw, dtype=np.float32)
    joint_names = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    ]
    name2idx = {name: i for i, name in enumerate(joint_names)}
    return {"J_regressor": J_regressor, "name2idx": name2idx}


def get_joint_positions(verts, J_regressor, joint_idx):
    return J_regressor[joint_idx] @ verts


def mpvpe(pred_v, gt_v, idxs, align_pred, align_gt):
    p = pred_v[idxs] - align_pred
    g = gt_v[idxs] - align_gt
    return float(np.linalg.norm(p - g, axis=-1).mean() * 1000.0)


def tr_v2v(pred_v, gt_v, idxs):
    """TR-V2V (mm) — per-region centroid alignment (SGNify paper Table 3)."""
    p = pred_v[idxs]; g = gt_v[idxs]
    p = p - p.mean(axis=0, keepdims=True)
    g = g - g.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(p - g, axis=-1).mean() * 1000.0)


def get_frame_stems(pred_sign_dir: Path) -> set:
    """Get set of frame stems (e.g. 'low_131') from a sign directory."""
    stems = set()
    meshes_dir = pred_sign_dir / "smplifyx" / "meshes"
    if not meshes_dir.exists():
        return stems
    for f in meshes_dir.iterdir():
        if f.suffix in ('.obj', '.npy', '.npz', '.pkl') and f.stem.startswith('low_'):
            stems.add(f.stem)
    return stems


def stem_to_gt_path(stem: str, gt_sign_dir: Path) -> Path:
    m = re.search(r'low_(\d+)', stem)
    if not m:
        return None
    gt_idx = int(m.group(1)) * 2
    return gt_sign_dir / f"{gt_idx:05d}.obj"


def select_central_stems(stems: set) -> set:
    """Filter stems to central frames only: 0.5×T/8 < t < 7×T/8 (SGNify paper, Appendix C)."""
    if not stems:
        return stems
    gt_indices = {}
    for stem in stems:
        m = re.search(r'low_(\d+)', stem)
        if m:
            gt_indices[stem] = int(m.group(1)) * 2
    if not gt_indices:
        return stems
    valid = list(gt_indices.values())
    T = max(valid) - min(valid) + 1
    t_min = min(valid)
    core_lo = t_min + 0.5 * T / 8.0
    core_hi = t_min + 7.0 * T / 8.0
    return {stem for stem, gt_idx in gt_indices.items() if core_lo < gt_idx < core_hi}


def evaluate_method_common(method_name, pred_root, joints, ubody_idx, lhand_idx, rhand_idx,
                           signs, common_frames_per_sign, central_frames=False):
    J_reg = joints["J_regressor"]
    pelvis_i = joints["name2idx"]["pelvis"]
    lwrist_i = joints["name2idx"]["left_wrist"]
    rwrist_i = joints["name2idx"]["right_wrist"]

    frame_rows = []
    sign_stats = {}
    central_skipped = 0

    for sign in signs:
        common_stems = common_frames_per_sign.get(sign, set())
        if central_frames and common_stems:
            original_count = len(common_stems)
            common_stems = select_central_stems(common_stems)
            central_skipped += original_count - len(common_stems)
        if not common_stems:
            continue

        pred_sign = pred_root / sign
        gt_sign = GT_ROOT / sign
        rows = []

        for stem in sorted(common_stems):
            pred_file = pred_sign / "smplifyx" / "meshes" / f"{stem}.obj"
            gt_file = stem_to_gt_path(stem, gt_sign)
            if gt_file is None or not pred_file.exists() or not gt_file.exists():
                continue

            pv = load_obj_vertices(pred_file)
            gv = load_obj_vertices(gt_file)

            pelvis_pred = get_joint_positions(pv, J_reg, pelvis_i)
            pelvis_gt = get_joint_positions(gv, J_reg, pelvis_i)
            lwrist_pred = get_joint_positions(pv, J_reg, lwrist_i)
            lwrist_gt = get_joint_positions(gv, J_reg, lwrist_i)
            rwrist_pred = get_joint_positions(pv, J_reg, rwrist_i)
            rwrist_gt = get_joint_positions(gv, J_reg, rwrist_i)

            m_ubody = mpvpe(pv, gv, ubody_idx, pelvis_pred, pelvis_gt)
            m_lhand = mpvpe(pv, gv, lhand_idx, lwrist_pred, lwrist_gt)
            m_rhand = mpvpe(pv, gv, rhand_idx, rwrist_pred, rwrist_gt)
            t_ubody = tr_v2v(pv, gv, ubody_idx)
            t_lhand = tr_v2v(pv, gv, lhand_idx)
            t_rhand = tr_v2v(pv, gv, rhand_idx)

            metrics = [m_ubody, m_lhand, m_rhand, t_ubody, t_lhand, t_rhand]
            rows.append(metrics)
            frame_rows.append((sign, stem, *metrics))

        if rows:
            arr = np.array(rows, dtype=np.float32)
            sign_stats[sign] = arr

    if not frame_rows:
        return {}

    all_arr = np.vstack([sign_stats[s] for s in sign_stats])
    mean = all_arr.mean(axis=0)

    return {
        "method": method_name,
        "mpvpe_ubody": mean[0], "mpvpe_lhand": mean[1], "mpvpe_rhand": mean[2],
        "trv2v_ubody": mean[3], "trv2v_lhand": mean[4], "trv2v_rhand": mean[5],
        "frames": len(frame_rows),
        "signs": len(sign_stats),
        "total_signs": len(signs),
        "frame_rows": frame_rows,
        "sign_stats": sign_stats,
        "central_skipped": central_skipped,
    }


def print_table(title, metric, results):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")
    print(f"{'Method':<25} {'UBody(-F)':>10} {'LHand':>10} {'RHand':>10} {'Frames':>8}")
    print("-" * 65)
    for r in results:
        print(f"{r['method']:<25} {r[f'{metric}_ubody']:>10.2f} {r[f'{metric}_lhand']:>10.2f} "
              f"{r[f'{metric}_rhand']:>10.2f} {r['frames']:>8}")
    print("-" * 65)
    best_u = min(r[f'{metric}_ubody'] for r in results)
    best_l = min(r[f'{metric}_lhand'] for r in results)
    best_r = min(r[f'{metric}_rhand'] for r in results)
    print(f"{'Best':<25} {best_u:>10.2f} {best_l:>10.2f} {best_r:>10.2f}")


def main():
    ap = argparse.ArgumentParser(description="MPVPE/TR-V2V on common frames only.")
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--method_names", nargs="+", required=True)
    ap.add_argument("--output_csv", default="")
    ap.add_argument("--central_frames", action="store_true",
                    help="Filter to central frames only (0.5×T/8 < t < 7×T/8, per SGNify paper)")
    args = ap.parse_args()

    print("Loading SMPL-X joint regressor...")
    joints = load_smplx_joints(SMPLX_MODEL)
    ubody_idx = np.load(UBODY_IDX)
    lhand_idx = np.load(LHAND_IDX)
    rhand_idx = np.load(RHAND_IDX)
    signs = []
    with open(SIGNS_TXT) as f:
        for line in f:
            line = line.strip()
            if line:
                signs.append(line.split()[0])
    print(f"Loaded {len(signs)} signs\n")

    # Step 1: Find common frames across ALL methods for each sign
    print("Finding common frames across all methods...")
    common_frames_per_sign = {}
    total_common = 0
    for sign in signs:
        all_stems = None
        for method_dir in args.methods:
            pred_sign = OUTPUT_BASE / method_dir / sign
            stems = get_frame_stems(pred_sign)
            if all_stems is None:
                all_stems = stems
            else:
                all_stems = all_stems.intersection(stems)
        common_frames_per_sign[sign] = all_stems or set()
        total_common += len(common_frames_per_sign[sign])

    print(f"Total common frames: {total_common}")

    # Show per-sign differences
    for sign in signs:
        common = common_frames_per_sign[sign]
        per_method = {}
        for method_dir in args.methods:
            per_method[method_dir] = len(get_frame_stems(OUTPUT_BASE / method_dir / sign))
        if any(v != len(common) for v in per_method.values()):
            detail = ", ".join(f"{m}={n}" for m, n in per_method.items())
            print(f"  {sign}: common={len(common)} ({detail})")

    # Step 2: Evaluate each method on common frames only
    results = []
    for method_dir, method_name in zip(args.methods, args.method_names):
        pred_root = OUTPUT_BASE / method_dir
        if not pred_root.exists():
            print(f"[SKIP] {pred_root}")
            continue
        print(f"\nEvaluating {method_name} on common frames...")
        r = evaluate_method_common(method_name, pred_root, joints, ubody_idx, lhand_idx, rhand_idx,
                                   signs, common_frames_per_sign, central_frames=args.central_frames)
        if r:
            results.append(r)
            print(f"  → {r['frames']} frames, {r['signs']}/{r['total_signs']} signs")

    if not results:
        print("No results.")
        return

    # Print tables
    frame_desc = "Central Frames" if args.central_frames else "Common Frames"
    print(f"\nFrame filter: {frame_desc}")
    print_table(f"MPVPE (mm) — {frame_desc} — Pelvis/Wrist Aligned", "mpvpe", results)
    print_table(f"TR-V2V (mm) — {frame_desc} — Pelvis Aligned (SGNify)", "trv2v", results)

    # Write CSV
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write("method,sign,frame,mpvpe_ubody,mpvpe_lhand,mpvpe_rhand,trv2v_ubody,trv2v_lhand,trv2v_rhand\n")
            for r in results:
                for row in r["frame_rows"]:
                    f.write(f"{r['method']},{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},"
                            f"{row[5]:.6f},{row[6]:.6f},{row[7]:.6f}\n")
        print(f"\nCSV written to {out}")

        summary_path = out.with_name("mpvpe_common_summary.csv")
        with open(summary_path, "w") as f:
            f.write("method,mpvpe_ubody,mpvpe_lhand,mpvpe_rhand,trv2v_ubody,trv2v_lhand,trv2v_rhand,frames,signs\n")
            for r in results:
                f.write(f"{r['method']},{r['mpvpe_ubody']:.6f},{r['mpvpe_lhand']:.6f},{r['mpvpe_rhand']:.6f},"
                        f"{r['trv2v_ubody']:.6f},{r['trv2v_lhand']:.6f},{r['trv2v_rhand']:.6f},"
                        f"{r['frames']},{r['signs']}\n")
        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
