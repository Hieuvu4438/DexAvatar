#!/usr/bin/env python3
"""Evaluate MPVPE on UBody(-F), LHand, RHand regions for multiple methods.

MPVPE alignment (matching OSX):
  - UBody(-F): pelvis-aligned (subtract pelvis joint from both pred & GT)
  - LHand:     left-wrist-aligned
  - RHand:     right-wrist-aligned

Usage:
    python eval_mpvpe_regions.py
    python eval_mpvpe_regions.py --methods biomech hand2d --output_csv results/mpvpe_comparison.csv
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
SMPLX_MODEL = PROJECT / "SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.pkl"
SIGNS_TXT = PROJECT / "data/signs.txt"
GT_ROOT = PROJECT / "data/smplx_gt"
OUTPUT_BASE = PROJECT / "outputs"

# Region index files (same as TR-V2V evaluator)
UBODY_IDX = PROJECT / "dexavatar_fitting/assets/smplx_upper_body_minus_face_vidx.npy"
LHAND_IDX = PROJECT / "dexavatar_fitting/assets/smplx_left_hand_vidx.npy"
RHAND_IDX = PROJECT / "dexavatar_fitting/assets/smplx_right_hand_vidx.npy"
FRAMES_ROOT = PROJECT / "data/frames"


# ── helpers ──────────────────────────────────────────────────────────────────
def load_obj_vertices(path: Path) -> np.ndarray:
    """Load vertex positions from .obj file → (V, 3)."""
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(verts, dtype=np.float32)


def load_smplx_joints(model_path: Path) -> dict:
    """Load SMPL-X model and return J_regressor + joint names.

    Returns dict with:
      - 'J_regressor': np.ndarray (J, V)  – sparse matrix, row per joint
      - 'joint_names': list[str]          – joint name per row
    """
    import pickle

    with open(model_path, "rb") as f:
        model = pickle.load(f, encoding="latin1")

    J_raw = model["J_regressor"]
    if hasattr(J_raw, "todense"):
        J_raw = J_raw.todense()
    J_regressor = np.array(J_raw, dtype=np.float32)  # (55, 10475)
    # Joint names from SMPL-X specification (first 55 joints)
    joint_names = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        # hand joints follow (20 per hand)
    ]
    # Build name → index map
    name2idx = {name: i for i, name in enumerate(joint_names)}
    return {"J_regressor": J_regressor, "name2idx": name2idx}


def get_joint_positions(verts: np.ndarray, J_regressor: np.ndarray, joint_idx: int) -> np.ndarray:
    """Get joint position via linear blend of vertices → (3,)."""
    return J_regressor[joint_idx] @ verts  # (V,) @ (V,3) → (3,)


def mpvpe(pred_v: np.ndarray, gt_v: np.ndarray, idxs: np.ndarray,
          align_pred: np.ndarray, align_gt: np.ndarray) -> float:
    """Mean Per-Vertex Position Error (mm) with per-region alignment.

    align_pred / align_gt: (3,) vectors to subtract from pred / gt respectively.
    """
    p = pred_v[idxs] - align_pred
    g = gt_v[idxs] - align_gt
    err = np.linalg.norm(p - g, axis=-1)
    return float(err.mean() * 1000.0)


def tr_v2v(pred_v: np.ndarray, gt_v: np.ndarray, idxs: np.ndarray) -> float:
    """TR-V2V (mm) — per-region centroid alignment.

    Matches SGNify paper Table 3: remove translation per region, then compute V2V.
    """
    p = pred_v[idxs]
    g = gt_v[idxs]
    p = p - p.mean(axis=0, keepdims=True)
    g = g - g.mean(axis=0, keepdims=True)
    err = np.linalg.norm(p - g, axis=-1)
    return float(err.mean() * 1000.0)


def collect_pairs(pred_sign_dir: Path, gt_sign_dir: Path) -> list:
    """Match prediction meshes to GT meshes by frame index (2× mapping)."""
    if not pred_sign_dir.exists() or not gt_sign_dir.exists():
        return []

    # Only smplifyx/meshes/ — smplerx/mesh/ is in camera space, not comparable to GT
    pred_files = sorted(
        list(pred_sign_dir.glob("**/meshes/low_*.obj")) +
        list(pred_sign_dir.glob("**/meshes/low_*.npy")) +
        list(pred_sign_dir.glob("**/meshes/low_*.npz")) +
        list(pred_sign_dir.glob("**/meshes/low_*.pkl"))
    )
    pairs = []
    for pf in pred_files:
        m = re.search(r"low_(\d+)", pf.stem)
        if not m:
            continue
        pred_idx = int(m.group(1))
        gt_idx = pred_idx * 2
        gf = gt_sign_dir / f"{gt_idx:05d}.obj"
        if gf.exists():
            pairs.append((pf, gf))
    return pairs


def select_central_frames(pairs):
    """Filter pairs to central frames only: 0.5×T/8 < t < 7×T/8 (SGNify paper, Appendix C)."""
    if not pairs:
        return pairs
    gt_indices = []
    for pp, gp in pairs:
        m = re.search(r'(\d{5})', gp.stem)
        gt_indices.append(int(m.group(1)) if m else None)
    valid = [i for i in gt_indices if i is not None]
    if not valid:
        return pairs
    T = max(valid) - min(valid) + 1
    t_min = min(valid)
    core_lo = t_min + 0.5 * T / 8.0
    core_hi = t_min + 7.0 * T / 8.0
    return [(pp, gp) for (pp, gp), gt_idx in zip(pairs, gt_indices)
            if gt_idx is not None and core_lo < gt_idx < core_hi]


def load_sign_frames(frames_root: Path, sign_name: str):
    """Load [start, end] from start_end_central.txt for a sign.

    Returns (start, end) as 30fps PNG frame indices, or None if file not found.
    """
    txt_path = frames_root / sign_name / "start_end_central.txt"
    if not txt_path.exists():
        return None
    with open(txt_path, 'r') as f:
        content = f.read().strip().strip('[]')
    parts = [x.strip() for x in content.split(',')]
    return int(parts[0]), int(parts[1])


def filter_by_sign_frames(pairs, start, end):
    """Filter prediction frames to only those within [start, end] range.

    Prediction files are low_{idx}.obj where idx is 30fps PNG frame index.
    """
    filtered = []
    for pp, gp in pairs:
        m = re.search(r'low_(\d+)', pp.stem)
        if not m:
            continue
        pred_idx = int(m.group(1))
        if start <= pred_idx <= end:
            filtered.append((pp, gp))
    return filtered


def get_input_frame_range(frames_root: Path, sign_name: str):
    """Return (min_idx, max_idx) from data/frames/{sign}/low_*.png, or None.

    T = max_idx - min_idx + 1 defines the input video length per the SGNify paper.
    Mirrors data_parser.py filename_to_int (lines 154-158).
    """
    ids = []
    for p in (frames_root / sign_name).glob("low_*.png"):
        try:
            ids.append(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    if not ids:
        return None
    return min(ids), max(ids)


def paper_central_range(min_idx: int, max_idx: int):
    """Return (new_start, new_end) per the SGNify paper, applied to the input video.

    new_start = min + round(0.5*T/8),  new_end = min + round(7*T/8),
    where T = max - min + 1. Caller should apply strict < on both ends.
    """
    T = max_idx - min_idx + 1
    new_start = min_idx + round(0.5 * T / 8.0)
    new_end   = min_idx + round(7.0 * T / 8.0)
    return new_start, new_end


def filter_by_paper_central(pairs, new_start: int, new_end: int):
    """Filter pairs to those with pred_idx strictly inside (new_start, new_end)."""
    out = []
    for pp, gp in pairs:
        m = re.search(r'low_(\d+)', pp.stem)
        if not m:
            continue
        if new_start < int(m.group(1)) < new_end:
            out.append((pp, gp))
    return out


# ── main ─────────────────────────────────────────────────────────────────────
def evaluate_method(method_name: str, pred_root: Path, joints: dict,
                    ubody_idx, lhand_idx, rhand_idx, signs: list,
                    central_frames: bool = False,
                    sign_frames: bool = False,
                    paper_central_ranges: dict | None = None) -> dict:
    """Evaluate one method, return per-frame and summary results."""
    J_reg = joints["J_regressor"]
    pelvis_i = joints["name2idx"]["pelvis"]
    lwrist_i = joints["name2idx"]["left_wrist"]
    rwrist_i = joints["name2idx"]["right_wrist"]

    frame_rows = []
    sign_stats = {}
    missing = []
    central_skipped = 0

    for sign in signs:
        pairs = collect_pairs(pred_root / sign, GT_ROOT / sign)
        if sign_frames:
            # Use start/end from start_end_central.txt
            bounds = load_sign_frames(FRAMES_ROOT, sign)
            if bounds:
                original_count = len(pairs)
                pairs = filter_by_sign_frames(pairs, bounds[0], bounds[1])
                central_skipped += original_count - len(pairs)
        elif paper_central_ranges is not None and sign in paper_central_ranges:
            # Use paper's 0.5T/8 < t < 7T/8 formula on input video T
            new_start, new_end = paper_central_ranges[sign]
            original_count = len(pairs)
            pairs = filter_by_paper_central(pairs, new_start, new_end)
            central_skipped += original_count - len(pairs)
        elif central_frames:
            # Use formula 0.5×T/8 < t < 7×T/8 (on GT indices)
            original_count = len(pairs)
            pairs = select_central_frames(pairs)
            central_skipped += original_count - len(pairs)
        if not pairs:
            missing.append(sign)
            continue
        rows = []
        for pf, gf in pairs:
            pv = load_obj_vertices(pf)
            gv = load_obj_vertices(gf)

            # Joint positions for alignment
            pelvis_pred = get_joint_positions(pv, J_reg, pelvis_i)
            pelvis_gt   = get_joint_positions(gv, J_reg, pelvis_i)
            lwrist_pred = get_joint_positions(pv, J_reg, lwrist_i)
            lwrist_gt   = get_joint_positions(gv, J_reg, lwrist_i)
            rwrist_pred = get_joint_positions(pv, J_reg, rwrist_i)
            rwrist_gt   = get_joint_positions(gv, J_reg, rwrist_i)

            # MPVPE (pelvis/wrist aligned)
            m_ubody = mpvpe(pv, gv, ubody_idx, pelvis_pred, pelvis_gt)
            m_lhand = mpvpe(pv, gv, lhand_idx, lwrist_pred, lwrist_gt)
            m_rhand = mpvpe(pv, gv, rhand_idx, rwrist_pred, rwrist_gt)

            # TR-V2V (per-region centroid aligned, matching SGNify paper Table 3)
            t_ubody = tr_v2v(pv, gv, ubody_idx)
            t_lhand = tr_v2v(pv, gv, lhand_idx)
            t_rhand = tr_v2v(pv, gv, rhand_idx)

            metrics = [m_ubody, m_lhand, m_rhand, t_ubody, t_lhand, t_rhand]
            rows.append(metrics)
            frame_rows.append((sign, pf.stem, *metrics))

        arr = np.array(rows, dtype=np.float32)
        sign_stats[sign] = arr

    if not frame_rows:
        print(f"[WARN] {method_name}: no matched pairs found!", file=sys.stderr)
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
        "missing": missing,
        "frame_rows": frame_rows,
        "sign_stats": sign_stats,
        "central_skipped": central_skipped,
    }


def print_table(title: str, metric: str, results: list):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    header = f"{'Method':<25} {'UBody(-F)':>10} {'LHand':>10} {'RHand':>10}"
    print(header)
    print("-" * 60)
    for r in results:
        print(f"{r['method']:<25} {r[f'{metric}_ubody']:>10.2f} {r[f'{metric}_lhand']:>10.2f} {r[f'{metric}_rhand']:>10.2f}")
    print("-" * 60)
    best_u = min(r[f'{metric}_ubody'] for r in results)
    best_l = min(r[f'{metric}_lhand'] for r in results)
    best_r = min(r[f'{metric}_rhand'] for r in results)
    print(f"{'Best':<25} {best_u:>10.2f} {best_l:>10.2f} {best_r:>10.2f}")


def main():
    ap = argparse.ArgumentParser(description="MPVPE evaluation with UBody/LHand/RHand regions.")
    ap.add_argument("--methods", nargs="+", default=["method_biomech", "method_hand2d"],
                    help="Method names (subdirs under outputs/)")
    ap.add_argument("--method_names", nargs="+", default=None,
                    help="Display names for methods (same order as --methods)")
    ap.add_argument("--output_csv", default="", help="Optional CSV output path")
    ap.add_argument("--central_frames", action="store_true",
                    help="Filter to central frames only (0.5×T/8 < t < 7×T/8, per SGNify paper, on GT indices)")
    ap.add_argument("--sign_frames", action="store_true",
                    help="Filter to [start,end] from data/frames/*/start_end_central.txt (actual sign boundaries)")
    ap.add_argument("--paper_central", action="store_true",
                    help="Filter to 0.5*T/8 < t < 7*T/8 with T = input video length "
                         "from data/frames/{sign}/low_*.png (SGNify paper definition)")
    args = ap.parse_args()

    if args.method_names is None:
        args.method_names = [f"DexAvatar-{m.replace('method_', '').capitalize()}" for m in args.methods]
    assert len(args.methods) == len(args.method_names), "--methods and --method_names must have same length"
    n_filters = sum([args.central_frames, args.sign_frames, args.paper_central])
    assert n_filters <= 1, "--central_frames, --sign_frames, and --paper_central are mutually exclusive"

    # Load shared resources
    print("Loading SMPL-X joint regressor...")
    joints = load_smplx_joints(SMPLX_MODEL)
    ubody_idx = np.load(UBODY_IDX)
    lhand_idx = np.load(LHAND_IDX)
    rhand_idx = np.load(RHAND_IDX)
    signs = []
    with open(SIGNS_TXT, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                signs.append(line.split()[0])
    print(f"Loaded {len(signs)} signs, {len(ubody_idx)} UBody verts, {len(lhand_idx)} LHand verts, {len(rhand_idx)} RHand verts\n")

    # Build per-sign paper-central range map (T = input video length)
    paper_central_ranges = None
    if args.paper_central:
        paper_central_ranges = {}
        for sign in signs:
            r = get_input_frame_range(FRAMES_ROOT, sign)
            if r is None:
                print(f"[WARN] {sign}: no input frames in {FRAMES_ROOT/sign}; skipping sign")
                continue
            ns, ne = paper_central_range(r[0], r[1])
            paper_central_ranges[sign] = (ns, ne)
            print(f"  paper_central {sign}: input T={r[1]-r[0]+1} (min={r[0]}, max={r[1]}) "
                  f"→ keep pred_idx in ({ns}, {ne})")

    # Evaluate each method
    results = []
    for method_dir, method_name in zip(args.methods, args.method_names):
        pred_root = OUTPUT_BASE / method_dir
        if not pred_root.exists():
            print(f"[SKIP] {pred_root} does not exist")
            continue
        print(f"Evaluating {method_name} from {pred_root} ...")
        r = evaluate_method(method_name, pred_root, joints, ubody_idx, lhand_idx, rhand_idx, signs,
                            central_frames=args.central_frames,
                            sign_frames=args.sign_frames,
                            paper_central_ranges=paper_central_ranges)
        if r:
            results.append(r)
            print(f"  → {r['frames']} frames, {r['signs']}/{r['total_signs']} signs")

    if not results:
        print("No results to show.")
        return

    # Print tables
    if args.paper_central:
        frame_desc = "Central Frames (paper, on input video T)"
    elif args.sign_frames:
        frame_desc = "Sign Boundaries (start_end_central.txt)"
    elif args.central_frames:
        frame_desc = "Central Frames (formula)"
    else:
        frame_desc = "All Frames"
    print(f"\nFrame filter: {frame_desc}")
    print_table(f"MPVPE (mm) — {frame_desc} — Pelvis/Wrist Aligned", "mpvpe", results)
    print_table(f"TR-V2V (mm) — {frame_desc} — Region-Centroid Aligned (SGNify)", "trv2v", results)

    # Per-sign detail for each method
    for r in results:
        print(f"\n{'─' * 60}")
        print(f"  Per-sign detail: {r['method']}")
        print(f"{'─' * 60}")
        print(f"{'Sign':<30} {'Frames':>6} {'UBody':>8} {'LHand':>8} {'RHand':>8}")
        print("-" * 60)
        for sign, arr in sorted(r["sign_stats"].items()):
            m = arr.mean(axis=0)
            print(f"{sign:<30} {len(arr):>6} {m[0]:>8.2f} {m[1]:>8.2f} {m[2]:>8.2f}")
        if r["missing"]:
            print(f"\n  Missing signs ({len(r['missing'])}): {' '.join(r['missing'])}")

    # Write CSV
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            f.write("method,sign,frame,mpvpe_ubody,mpvpe_lhand,mpvpe_rhand,trv2v_ubody,trv2v_lhand,trv2v_rhand\n")
            for r in results:
                for row in r["frame_rows"]:
                    f.write(f"{r['method']},{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f},{row[6]:.6f},{row[7]:.6f}\n")
        print(f"\nCSV written to {out}")

    # Also write summary CSV alongside
    summary_path = Path(args.output_csv).with_name("mpvpe_summary.csv") if args.output_csv else None
    if summary_path:
        with open(summary_path, "w") as f:
            f.write("method,mpvpe_ubody,mpvpe_lhand,mpvpe_rhand,trv2v_ubody,trv2v_lhand,trv2v_rhand,frames,signs\n")
            for r in results:
                f.write(f"{r['method']},{r['mpvpe_ubody']:.6f},{r['mpvpe_lhand']:.6f},{r['mpvpe_rhand']:.6f},"
                        f"{r['trv2v_ubody']:.6f},{r['trv2v_lhand']:.6f},{r['trv2v_rhand']:.6f},"
                        f"{r['frames']},{r['signs']}\n")
        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
