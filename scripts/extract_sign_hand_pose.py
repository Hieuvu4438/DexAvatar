#!/usr/bin/env python3
"""
Extract per-frame hand poses (left + right, 45-dim axis-angle each) from
existing SMPLer-X extractions under
    /home/haipd/DexAvatar/data/signbposer_data/raw/how2sign/
    /home/haipd/DexAvatar/data/signbposer_data/raw/phoenix/
and write them to
    /home/haipd/DexAvatar/data/vqvae_hand_data/{train,val,test}/{lhand,rhand}_poses.npy

Source pkls are NOT moved. Outputs are placed in a new data/ sub-folder.
"""
import os
import sys
import glob
import pickle
import argparse
import numpy as np


def collect_pkls(raw_dir):
    """Yield (source, pkl_path) pairs under `raw_dir`/<video>/smplx/*.pkl."""
    if not os.path.isdir(raw_dir):
        return
    for pkl in sorted(glob.glob(os.path.join(raw_dir, "*", "smplx", "*.pkl"))):
        yield pkl


def extract_from_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict):
        return None, None
    lhand = d.get("left_hand_pose")
    rhand = d.get("right_hand_pose")
    if lhand is None or rhand is None:
        return None, None
    lhand = np.asarray(lhand, dtype=np.float32).flatten()
    rhand = np.asarray(rhand, dtype=np.float32).flatten()
    if lhand.size != 45 or rhand.size != 45:
        return None, None
    return lhand, rhand


def process_split(raw_dir, out_split_dir, split_name, max_files=0):
    lhand_list, rhand_list = [], []
    pkls = list(collect_pkls(raw_dir))
    if max_files > 0:
        pkls = pkls[:max_files]
    print(f"  [{split_name}] scanning {raw_dir}: {len(pkls)} pkls")
    for pkl in pkls:
        l, r = extract_from_pkl(pkl)
        if l is None:
            continue
        lhand_list.append(l)
        rhand_list.append(r)
    if not lhand_list:
        print(f"  [{split_name}] no valid hand poses found")
        return False

    os.makedirs(out_split_dir, exist_ok=True)
    lhand_arr = np.stack(lhand_list, axis=0)
    rhand_arr = np.stack(rhand_list, axis=0)
    np.save(os.path.join(out_split_dir, "lhand_poses.npy"), lhand_arr)
    np.save(os.path.join(out_split_dir, "rhand_poses.npy"), rhand_arr)
    print(f"  [{split_name}] saved {lhand_arr.shape} lhand + {rhand_arr.shape} rhand "
          f"to {out_split_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root",
                        default="/home/haipd/DexAvatar/data/signbposer_data/raw")
    parser.add_argument("--output_root",
                        default="/home/haipd/DexAvatar/data/vqvae_hand_data")
    parser.add_argument("--max_files", type=int, default=0,
                        help="Cap on pkls per source dir (0 = no cap)")
    args = parser.parse_args()

    # Mapping from existing data dir to split name.
    # The existing body_poses.npy uses (How2Sign → train, Phoenix → train) and (Phoenix → val/test).
    # We mirror that: put How2Sign in train, Phoenix in val, and the test split (if any) in test.
    sources = [
        ("how2sign", "train"),
        ("phoenix", "val"),
    ]
    test_dir = os.path.join(args.data_root, "phoenix", "extracted")
    if os.path.isdir(test_dir):
        sources.append(("phoenix", "test"))

    for sub, split in sources:
        raw_dir = os.path.join(args.data_root, sub)
        out_dir = os.path.join(args.output_root, split)
        if not os.path.isdir(raw_dir):
            print(f"  skip: missing {raw_dir}")
            continue
        process_split(raw_dir, out_dir, split, max_files=args.max_files)

    print("Done.")


if __name__ == "__main__":
    main()
