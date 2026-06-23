#!/usr/bin/env python3
"""Temporal smoothing / outlier filtering for NLF init pkls.

Scans the ``nlf/smplx/*.pkl`` files in frame order, detects sudden non-physical
jumps in ``transl`` and ``global_orient``, and replaces outliers with linearly
(transl) or spherically (global_orient) interpolated values from neighboring
healthy frames.

Run automatically after NLF extraction or standalone:
    python scripts/smooth_nlf_init.py --pkl_dir outputs/method_nlf_wilor/Ablehnen/nlf/smplx
"""
import argparse
import os
import pickle
import re
import glob

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def load_pkls(pkl_dir):
    """Load all pkls sorted by frame number."""
    files = sorted(
        glob.glob(os.path.join(pkl_dir, "*.pkl")),
        key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group()),
    )
    data = []
    for f in files:
        with open(f, "rb") as fh:
            data.append((f, pickle.load(fh)))
    return data


def save_pkls(data):
    """Write modified pkls back."""
    for path, pkl in data:
        with open(path, "wb") as f:
            pickle.dump(pkl, f)


def mad_outlier_mask(series, threshold=3.5, window=5):
    """Return bool mask: True = outlier, using rolling MAD in a window.

    For each element, compute the median of the window around it (excluding self),
    then the MAD of that window. Flag as outlier if |x - median| > threshold * MAD.
    """
    n = len(series)
    mask = np.zeros(n, dtype=bool)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neighbors = [series[j] for j in range(lo, hi) if j != i]
        if len(neighbors) < 2:
            continue
        med = np.median(neighbors)
        mad = np.median(np.abs(np.array(neighbors) - med))
        if mad < 1e-10:
            continue
        z = 0.6745 * abs(series[i] - med) / mad
        if z > threshold:
            mask[i] = True
    return mask


def interpolate_outliers(data, outlier_mask, key, mode="linear"):
    """Replace outlier frames with interpolation from nearest good neighbors.

    Args:
        data: list of (path, pkl_dict) tuples
        outlier_mask: bool array, True = outlier
        key: dict key to fix ('transl' or 'global_orient')
        mode: 'linear' for transl, 'slerp' for global_orient
    """
    indices = np.arange(len(data))
    good_idx = indices[~outlier_mask]
    bad_idx = indices[outlier_mask]
    if len(bad_idx) == 0 or len(good_idx) < 2:
        return 0

    n_fixed = 0
    for bi in bad_idx:
        # Find nearest good neighbors on each side
        left = good_idx[good_idx < bi]
        right = good_idx[good_idx > bi]
        if len(left) > 0 and len(right) > 0:
            i0, i1 = left[-1], right[0]
        elif len(left) > 0:
            # Only left neighbor — copy
            data[bi][1][key] = data[left[-1]][1][key].copy()
            n_fixed += 1
            continue
        elif len(right) > 0:
            # Only right neighbor — copy
            data[bi][1][key] = data[right[0]][1][key].copy()
            n_fixed += 1
            continue
        else:
            continue

        # Interpolate
        alpha = (bi - i0) / (i1 - i0)  # position between neighbors
        v0 = data[i0][1][key]
        v1 = data[i1][1][key]

        if mode == "linear":
            data[bi][1][key] = v0 + alpha * (v1 - v0)
        elif mode == "slerp":
            # global_orient is axis-angle (3,). Convert to rotation vectors.
            r0 = R.from_rotvec(v0.reshape(1, 3))
            r1 = R.from_rotvec(v1.reshape(1, 3))
            # Slerp requires times in [0, 1]
            slerp = Slerp([0, 1], R.concatenate([r0, r1]))
            r_interp = slerp(alpha)
            data[bi][1][key] = r_interp.as_rotvec().reshape(3).astype(np.float32)
        n_fixed += 1
    return n_fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_dir", required=True, help="Path to nlf/smplx/ directory")
    parser.add_argument("--threshold", type=float, default=3.5,
                        help="MAD z-score threshold for outlier detection")
    parser.add_argument("--window", type=int, default=5,
                        help="Rolling window size for median/MAD")
    parser.add_argument("--dry_run", action="store_true",
                        help="Detect but don't modify")
    args = parser.parse_args()

    data = load_pkls(args.pkl_dir)
    if len(data) < 3:
        print(f"Too few frames ({len(data)}), skipping smoothing.")
        return

    # Extract series
    transl_y = np.array([d[1]["transl"][1] for d in data])
    transl_z = np.array([d[1]["transl"][2] for d in data])
    orient_vecs = np.array([d[1]["global_orient"] for d in data])

    total_fixed = 0

    # Detect and fix transl_y outliers
    mask_y = mad_outlier_mask(transl_y, args.threshold, args.window)
    if mask_y.any():
        fixed = interpolate_outliers(data, mask_y, "transl", "linear")
        total_fixed += fixed
        bad_names = [os.path.basename(data[i][0]) for i in np.where(mask_y)[0]]
        print(f"Fixed {fixed} transl outliers: {bad_names}")

    # Detect and fix transl_z outliers
    mask_z = mad_outlier_mask(transl_z, args.threshold, args.window)
    if mask_z.any():
        # Build composite transl mask
        mask_t = mask_y | mask_z
        fixed = interpolate_outliers(data, mask_t, "transl", "linear")
        # Don't double-count
        new_fixed = max(0, fixed - total_fixed)
        total_fixed += new_fixed
        bad_names = [os.path.basename(data[i][0]) for i in np.where(mask_t)[0]]
        print(f"Fixed transl_z outliers: {bad_names}")

    # Detect and fix global_orient outliers — use magnitude change
    orient_mag = np.linalg.norm(orient_vecs, axis=1)
    mask_o_mag = mad_outlier_mask(orient_mag, args.threshold, args.window)

    # Also check angular distance from neighbors (more sensitive)
    mask_o_ang = np.zeros(len(data), dtype=bool)
    for i in range(1, len(data) - 1):
        r_prev = R.from_rotvec(orient_vecs[i-1].reshape(1, 3))
        r_curr = R.from_rotvec(orient_vecs[i].reshape(1, 3))
        r_next = R.from_rotvec(orient_vecs[i+1].reshape(1, 3))
        # Magnitude of rotation difference
        ang_prev = np.linalg.norm(r_prev.as_rotvec() - r_curr.as_rotvec())
        ang_next = np.linalg.norm(r_next.as_rotvec() - r_curr.as_rotvec())
        if ang_prev > 1.0 and ang_next > 1.0:  # > ~57 deg from both neighbors
            mask_o_ang[i] = True

    mask_o = mask_o_mag | mask_o_ang
    if mask_o.any():
        fixed = interpolate_outliers(data, mask_o, "global_orient", "slerp")
        total_fixed += fixed
        bad_names = [os.path.basename(data[i][0]) for i in np.where(mask_o)[0]]
        print(f"Fixed {fixed} global_orient outliers: {bad_names}")

    if total_fixed == 0:
        print("No outliers detected.")
    elif not args.dry_run:
        save_pkls(data)
        print(f"Saved {total_fixed} fixes to {args.pkl_dir}")
    else:
        print(f"[DRY RUN] Would fix {total_fixed} frames.")


if __name__ == "__main__":
    main()
