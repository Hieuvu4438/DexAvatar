"""SGNify / DexAvatar standard benchmark evaluation.

Evaluates a body-fitting method against the SGNify ground-truth SMPLX meshes and
reports the standard vertex-to-vertex (V2V) metrics, in millimetres, for the
three canonical body parts used by the SGNify and DexAvatar papers:

    - Upper Body  (all vertices above the pelvis joint)
    - Left Hand   (778 MANO vertices)
    - Right Hand  (778 MANO vertices)

Two alignment conventions are reported for every part:

    - TR-V2V : Translation-Removed V2V. Each part is centred on its own
               centroid before the vertex error is measured. This is the metric
               computed by ``evaluate_new_fitting*.py`` and is the repo's metric
               of record.
    - PA-V2V : Procrustes-Aligned V2V (rigid rotation + translation, no scale).
               This is the canonical aligned-V2V reported in the SGNify/DexAvatar
               papers (a.k.a. aligned MPVPE).

Raw (un-aligned) error is omitted because the method output and the GT meshes
live in different world frames (a ~17 m global offset), which would dominate and
render the raw number meaningless.

Frame correspondence
---------------------
The method outputs one SMPLX mesh per evaluated frame as
``<method_folder>/<sign>/smplifyx/meshes/low_<frame>.obj``. The GT SMPLX meshes
live in ``<gt_folder>/<sign>/<frame:05d>.obj`` and are indexed at 2x the method
frame rate, so method frame ``f`` corresponds to GT frame ``2*f``. This 1:1 / 2x
mapping is verified to hold with zero mismatches across the whole corpus.

One-handed signs (class ``0`` in ``signs.txt``) only use the right hand: their
left hand is dropped from evaluation and the left-hand vertices are removed from
the upper-body mask, matching the original SGNify protocol.

Usage
-----
    python evaluate_method_sgnify.py \
        --method_folder /home/haipd/DexAvatar/outputs/method_nlf_wilor \
        --gt_folder     /home/haipd/DexAvatar/data/smplx_gt \
        --method_name   DexAvatar-NLF-WiLoR

All paths default to the nlf_wilor run on this machine, so a bare
``python evaluate_method_sgnify.py`` reproduces the benchmark.
"""
from __future__ import annotations

import argparse
import csv
import os
import os.path as osp
import pickle
import re

import numpy as np
from loguru import logger
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Defaults (this machine)
# --------------------------------------------------------------------------- #
DEFAULT_METHOD_FOLDER = "/home/haipd/DexAvatar/outputs/method_nlf_wilor"
DEFAULT_GT_FOLDER = "/home/haipd/DexAvatar/data/smplx_gt"
DEFAULT_SIGNS_FILE = "/home/haipd/DexAvatar/data/evaluation_from_author/signs.txt"
DEFAULT_DATA_ROOT = "/home/haipd/DexAvatar/data/evaluation_from_author/data/data"

MM = 1000.0  # mesh vertices are in metres -> report in millimetres
PARTS = ("upper body", "left hand", "right hand")
METRICS = ("TR-V2V", "PA-V2V")


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_vertices(obj_path: str) -> np.ndarray:
    """Fast loader returning only the vertex array (N, 3) of an SMPLX .obj mesh."""
    pts = []
    with open(obj_path, "r") as f:
        for line in f:
            if line[0] == "v" and line[1] == " ":
                pts.append(line[2:].rstrip())
    return np.fromstring(" ".join(pts), sep=" ", dtype=np.float64).reshape(-1, 3)


def load_sign_classes(signs_file: str) -> dict[str, str]:
    """Return {sign_name: class_label}. Class '0' == one-handed, '~0' == two-handed."""
    classes: dict[str, str] = {}
    with open(signs_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            tokens = line.strip().split(" ")
            classes[tokens[0]] = tokens[1] if len(tokens) > 1 else "~0"
    return classes


def list_method_frames(sign_folder: str) -> list[tuple[int, str]]:
    """(frame_id, path) for every low_<N>.obj under <sign>/smplifyx/meshes, sorted."""
    meshes_dir = osp.join(sign_folder, "smplifyx", "meshes")
    if not osp.isdir(meshes_dir):
        return []
    out = []
    for name in os.listdir(meshes_dir):
        if not name.endswith(".obj"):
            continue
        m = re.search(r"\d+", name)
        if m is not None:
            out.append((int(m.group()), osp.join(meshes_dir, name)))
    out.sort(key=lambda x: x[0])
    return out


def list_gt_frames(sign_folder: str) -> dict[int, str]:
    """{frame_id: path} for every <N>.obj GT mesh under <sign>/."""
    out: dict[int, str] = {}
    if not osp.isdir(sign_folder):
        return out
    for name in os.listdir(sign_folder):
        if name.endswith(".obj") and name[:-4].isdigit():
            out[int(name[:-4])] = osp.join(sign_folder, name)
    return out


def build_part_masks(data_root: str) -> dict[str, np.ndarray]:
    """Vertex-index masks for each body part."""
    with open(osp.join(data_root, "MANO_SMPLX_vertex_ids.pkl"), "rb") as f:
        mano = pickle.load(f)
    segm_dir = osp.join(data_root, "sgnify_part_segm_above_pelvis_joint")
    return {
        "upper body": np.load(osp.join(segm_dir, "upper_body.npy")),
        "left hand": mano["left_hand"],
        "right hand": mano["right_hand"],
    }


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def tr_v2v(pred: np.ndarray, gt: np.ndarray) -> float:
    """Translation-removed V2V (mm): centre each cloud on its own centroid."""
    p = pred - pred.mean(axis=0, keepdims=True)
    g = gt - gt.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(p - g, axis=1).mean() * MM)


def rigid_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Rigid Procrustes (rotation + translation, no scale) of pred onto gt."""
    mu_p = pred.mean(axis=0)
    mu_g = gt.mean(axis=0)
    H = (pred - mu_p).T @ (gt - mu_g)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (R @ pred.T).T + (mu_g - R @ mu_p)


def pa_v2v(pred: np.ndarray, gt: np.ndarray) -> float:
    """Procrustes-aligned V2V (mm), rigid, no scale."""
    aligned = rigid_align(pred, gt)
    return float(np.linalg.norm(aligned - gt, axis=1).mean() * MM)


METRIC_FUNCS = {"TR-V2V": tr_v2v, "PA-V2V": pa_v2v}


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #
def evaluate(method_folder, gt_folder, signs_file, data_root, method_name):
    """Run the benchmark. Returns (summary, per_frame_rows, per_sign_rows)."""
    classes = load_sign_classes(signs_file)
    base_masks = build_part_masks(data_root)
    signs = sorted(s for s in classes if osp.isdir(osp.join(method_folder, s)))

    def new_acc():
        return {p: {m: [] for m in METRICS} for p in PARTS}

    accum, per_frame_rows, per_sign_rows = new_acc(), [], []
    frames_evaluated = skipped_frames = 0

    for sign in tqdm(signs, desc="Signs"):
        one_handed = classes.get(sign, "~0") == "0"
        masks = {
            "upper body": (
                np.setdiff1d(base_masks["upper body"], base_masks["left hand"])
                if one_handed else base_masks["upper body"]
            ),
            "left hand": base_masks["left hand"],
            "right hand": base_masks["right hand"],
        }

        m_frames = list_method_frames(osp.join(method_folder, sign))
        g_frames = list_gt_frames(osp.join(gt_folder, sign))
        # Method frame f <-> GT frame 2*f.
        pairs = [(f, p, g_frames[2 * f]) for f, p in m_frames if (2 * f) in g_frames]
        if not pairs:
            logger.warning(f"[{sign}] no frame pairs (method frames have no GT).")
            continue

        sign_acc = new_acc()
        for f, m_path, g_path in pairs:
            pred = load_vertices(m_path)
            gt = load_vertices(g_path)
            if pred.shape != gt.shape or not np.isfinite(pred).all():
                logger.warning(f"[{sign}] frame {f}: bad meshes, skipping.")
                skipped_frames += 1
                continue

            frames_evaluated += 1
            row = {"sign": sign, "frame": f, "one_handed": one_handed}
            for part in PARTS:
                if part == "left hand" and one_handed:
                    continue
                idx = masks[part]
                p_pts, g_pts = pred[idx], gt[idx]
                for metric in METRICS:
                    val = METRIC_FUNCS[metric](p_pts, g_pts)
                    row[f"{metric}_{part}"] = val
                    sign_acc[part][metric].append(val)
                    accum[part][metric].append(val)
            per_frame_rows.append(row)

        srow = {"sign": sign, "frames": len(pairs), "one_handed": one_handed}
        for part in PARTS:
            if part == "left hand" and one_handed:
                continue
            for metric in METRICS:
                vals = np.array(sign_acc[part][metric])
                srow[f"{metric}_{part}"] = float(vals.mean()) if vals.size else float("nan")
        per_sign_rows.append(srow)

    logger.info(
        f"Evaluated {frames_evaluated} frames across {len(per_sign_rows)} signs "
        f"({skipped_frames} skipped)."
    )

    summary = {"_meta": {
        "method": method_name,
        "frames": frames_evaluated,
        "signs": len(per_sign_rows),
        "total_signs": len(signs),
    }}
    for part in PARTS:
        summary[part] = {}
        for metric in METRICS:
            vals = np.array(accum[part][metric])
            summary[part][metric] = float(vals.mean()) if vals.size else float("nan")
    return summary, per_frame_rows, per_sign_rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_table(summary: dict) -> None:
    meta = summary["_meta"]
    logger.info("=" * 58)
    logger.info(f"Benchmark: {meta['method']}  (SGNify / DexAvatar protocol)")
    logger.info(f"Frames={meta['frames']}  Signs={meta['signs']}/{meta['total_signs']}")
    logger.info("=" * 58)
    header = f"{'Part':<12}{'TR-V2V (mm)':>16}{'PA-V2V (mm)':>16}"
    logger.info(header)
    logger.info("-" * len(header))
    for part in PARTS:
        logger.info(
            f"{part:<12}"
            f"{summary[part]['TR-V2V']:>16.2f}"
            f"{summary[part]['PA-V2V']:>16.2f}"
        )
    logger.info("-" * len(header))
    logger.info("TR-V2V = translation-removed (per-part centroid)  [repo metric]")
    logger.info("PA-V2V = Procrustes rigid (rot+trans, no scale)   [paper metric]")


def write_summary_csv(summary: dict, path: str) -> None:
    meta = summary["_meta"]
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    cols = (
        ["method", "frames", "signs", "total_signs"]
        + [f"trv2v_{p.split()[0]}" for p in ("upper body", "left hand", "right hand")]
        + [f"pav2v_{p.split()[0]}" for p in ("upper body", "left hand", "right hand")]
    )
    # map part label -> csv short key
    short = {"upper body": "ubody", "left hand": "lhand", "right hand": "rhand"}
    row = [meta["method"], meta["frames"], meta["signs"], meta["total_signs"]]
    for metric in METRICS:
        tag = "trv2v" if metric == "TR-V2V" else "pav2v"
        for part in PARTS:
            v = summary[part][metric]
            row.append(f"{v:.4f}" if not np.isnan(v) else "")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow(row)
    logger.info(f"Summary CSV -> {path}")


def _write_rows_csv(rows: list[dict], path: str, label: str) -> None:
    if not rows:
        return
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info(f"{label} -> {path}  ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method_folder", default=DEFAULT_METHOD_FOLDER)
    parser.add_argument("--gt_folder", default=DEFAULT_GT_FOLDER)
    parser.add_argument("--signs_file", default=DEFAULT_SIGNS_FILE)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--method_name", default="DexAvatar-NLF-WiLoR")
    parser.add_argument("--out_dir", default=None,
                        help="Where to write CSVs (default: <method_folder>).")
    args = parser.parse_args()

    out_dir = args.out_dir or args.method_folder
    summary, per_frame, per_sign = evaluate(
        method_folder=args.method_folder,
        gt_folder=args.gt_folder,
        signs_file=args.signs_file,
        data_root=args.data_root,
        method_name=args.method_name,
    )
    print_table(summary)
    write_summary_csv(summary, osp.join(out_dir, "benchmark_summary.csv"))
    _write_rows_csv(per_frame, osp.join(out_dir, "benchmark_per_frame.csv"), "Per-frame CSV")
    _write_rows_csv(per_sign, osp.join(out_dir, "benchmark_per_sign.csv"), "Per-sign CSV")


if __name__ == "__main__":
    main()
