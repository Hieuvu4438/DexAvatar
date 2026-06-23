#!/usr/bin/env python3
"""
Fit the DPoser-X body normalizer (min/max, "axis_normalize1.pt") on the merged
SIGN training poses and place it where BOTH consumers expect it:

  1. {data_root}/{version}/train/axis_normalize1.pt   <- DPoser-X TRAINER loads
       the normalizer from the train-split dir (diffusion.py main(): data_path =
       {data_root}/{version}/train, Posenormalizer reads {...}/axis_normalize1.pt).
  2. {fitting_normalizer_dir}/axis_normalize1.pt      <- DPoser-X FITTING integration
       (dposerx_body.py hardcodes min_max=True -> axis_normalize1.pt); pointed to
       by dposerx_normalizer_dir in fit_smplx_vposer_x_dposerx_sign.yaml.

min/max (not mean/std) is required so train/fit normalization stay consistent:
dposerx_body.py uses min_max=True, so we train with data.min_max=True too (doc §4.3
Option B, §6.2 consistency trap).

Usage:
    python scripts/fit_sign_normalizer.py
"""
import os
import argparse
import numpy as np
import torch

TRAIN_DEFAULT = "/home/haipd/DexAvatar/data/signbposer_data_sign/train/body_poses.npy"
VALID_DEFAULT = "/home/haipd/DexAvatar/data/signbposer_data_sign/valid/body_poses.npy"
TRAINER_DIR_DEFAULT = "/home/haipd/DexAvatar/data/body_data/sign_v1/train"
FITTING_DIR_DEFAULT = "/home/haipd/DexAvatar/checkpoints/dposerx_body_sign/body_normalizer"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train_npy", default=TRAIN_DEFAULT)
    ap.add_argument("--valid_npy", default=VALID_DEFAULT,
                    help="Optional; included in the stats if present.")
    ap.add_argument("--trainer_dir", default=TRAINER_DIR_DEFAULT,
                    help="DPoser-X train-split dir (trainer consumer).")
    ap.add_argument("--fitting_dir", default=FITTING_DIR_DEFAULT,
                    help="Directory used by dposerx_normalizer_dir (fitting consumer).")
    args = ap.parse_args()

    arrays = []
    for p in [args.train_npy, args.valid_npy]:
        if p and os.path.exists(p):
            a = np.load(p).astype(np.float32)
            print(f"  loaded {p}: {a.shape}")
            arrays.append(a)
    if not arrays:
        raise SystemExit("No body_poses.npy found. Run merge_h2s_phoenix_for_dposerx.py first.")
    poses = np.concatenate(arrays, axis=0)
    print(f"  total: {poses.shape}")

    min_poses = torch.tensor(poses.min(axis=0), dtype=torch.float32)
    max_poses = torch.tensor(poses.max(axis=0), dtype=torch.float32)
    blob = {"min_poses": min_poses, "max_poses": max_poses}

    for out_dir in (args.trainer_dir, args.fitting_dir):
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "axis_normalize1.pt")
        torch.save(blob, out_path)
        print(f"  saved {out_path}")

    print("\nDone. (min/max scheme, consistent with dposerx_body.py min_max=True)")


if __name__ == "__main__":
    main()
