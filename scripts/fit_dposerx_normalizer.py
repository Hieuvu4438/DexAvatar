#!/usr/bin/env python3
"""
Fit DPoser-X body normalizer from sign-language data.

Reads existing SMPLer-X body-pose extractions and computes the per-dim
min/max statistics required by DPoser-X's `Posenormalizer`. Writes
`axis_normalize1.pt` into the given output dir (default
`checkpoints/dposerx_body/body_normalizer/`).

Source datasets (read-only):
    /home/haipd/DexAvatar/data/signbposer_data/train/body_poses.npy
    /home/haipd/DexAvatar/data/signbposer_data/val/body_poses.npy
    (falls back to train if val missing)

Output (created, never overwrites source):
    /home/haipd/DexAvatar/checkpoints/dposerx_body/body_normalizer/axis_normalize1.pt
"""
import os
import sys
import argparse
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_npy",
                        default="/home/haipd/DexAvatar/data/signbposer_data/train/body_poses.npy")
    parser.add_argument("--val_npy",
                        default="/home/haipd/DexAvatar/data/signbposer_data/val/body_poses.npy")
    parser.add_argument("--output_dir",
                        default="/home/haipd/DexAvatar/checkpoints/dposerx_body/body_normalizer")
    args = parser.parse_args()

    npy_paths = [args.train_npy]
    if os.path.exists(args.val_npy):
        npy_paths.append(args.val_npy)

    arrays = []
    for p in npy_paths:
        if not os.path.exists(p):
            print(f"  WARN: missing {p}, skipping")
            continue
        a = np.load(p).astype(np.float32)
        print(f"  loaded {p}: shape={a.shape}")
        arrays.append(a)

    if not arrays:
        raise SystemExit("No body_poses.npy files found; run the SMPLer-X extraction first.")

    poses = np.concatenate(arrays, axis=0)  # (N, 63)
    print(f"  total: {poses.shape}")

    # Min/max per dim — same scheme as DPoser-X's Posenormalizer (min_max=True).
    min_poses = torch.tensor(poses.min(axis=0), dtype=torch.float32)
    max_poses = torch.tensor(poses.max(axis=0), dtype=torch.float32)
    print(f"  min range: [{min_poses.min().item():.3f}, {min_poses.max().item():.3f}]")
    print(f"  max range: [{max_poses.min().item():.3f}, {max_poses.max().item():.3f}]")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "axis_normalize1.pt")
    torch.save({"min_poses": min_poses, "max_poses": max_poses}, out_path)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
