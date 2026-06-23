#!/usr/bin/env python3
"""
Convert merged sign body poses (How2Sign+PHOENIX) into the on-disk layout that
DPoser-X's AMASSDataset expects:

    {dst}/{version}/{train|valid|test}/pose_body.pt   (N, 63) float32
    {dst}/{version}/{train|valid|test}/root_orient.pt (N, 3)  float32

`pose_body` is the 21-joint SMPL body axis-angle (flattened 63-dim).
`root_orient` is set to zeros: a body-POSE prior is root-invariant and the sign
data carries no global orientation; DPoser-X normalizes global orient out anyway.

Input (default from merge_h2s_phoenix_for_dposerx.py):
    data/signbposer_data_sign/{train,valid,test}/body_poses.npy

Usage:
    python scripts/convert_sign_to_dposerx_layout.py
"""
import os
import argparse
import numpy as np
import torch

SRC_DEFAULT = "/home/haipd/DexAvatar/data/signbposer_data_sign"
DST_DEFAULT = "/home/haipd/DexAvatar/data/body_data"
VERSION_DEFAULT = "sign_v1"

SPLITS = [("train", "train"), ("valid", "valid"), ("test", "test")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--dst", default=DST_DEFAULT)
    ap.add_argument("--version", default=VERSION_DEFAULT)
    args = ap.parse_args()

    for src_split, dst_split in SPLITS:
        npy = os.path.join(args.src, src_split, "body_poses.npy")
        if not os.path.exists(npy):
            print(f"[{dst_split}] missing {npy} -- skip")
            continue
        poses = np.load(npy).astype(np.float32)
        if poses.ndim != 2 or poses.shape[1] != 63:
            raise ValueError(f"{npy}: expected (N,63), got {poses.shape}")

        out_dir = os.path.join(args.dst, args.version, dst_split)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(torch.from_numpy(poses),
                   os.path.join(out_dir, "pose_body.pt"))
        torch.save(torch.zeros(poses.shape[0], 3, dtype=torch.float32),
                   os.path.join(out_dir, "root_orient.pt"))
        print(f"[{dst_split}] N={poses.shape[0]} -> {out_dir}/{{pose_body,root_orient}}.pt")

    print("\nDone. DPoser-X layout under:", os.path.join(args.dst, args.version))


if __name__ == "__main__":
    main()
