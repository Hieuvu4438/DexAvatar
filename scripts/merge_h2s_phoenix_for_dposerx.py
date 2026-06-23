#!/usr/bin/env python3
"""
Merge How2Sign + PHOENIX-2014-T body poses into the per-split arrays consumed
by the DPoser-X sign body-prior pipeline.

Inputs (all under /home/haipd/DexAvatar/data):
  How2Sign (already prepared):  data/signbposer_data/{train,val,test}/body_poses.npy
  PHOENIX (from extract_phoenix_sign.py):
                                data/signbposer_data/raw/phoenix_sign/{train,dev,test}/body_poses.npy

Split mapping -> DPoser-X convention (train / valid / test):
  train : h2s train  + phx train
  valid : h2s val    + phx dev
  test  : h2s test   + phx test

Outputs:
  data/signbposer_data_sign/{train,valid,test}/body_poses.npy   (N, 63) float32
  data/signbposer_data_sign/{train,valid,test}/metadata.pkl     list of dicts w/ 'source'
  data/signbposer_data_sign/{train,valid,test}/sample_weights.npy  (N,) inverse-source-freq

Class imbalance: PHOENIX (subset ~21k) >> How2Sign (~1.4k). We CAP PHOENIX per
split to <= phx_cap_mult * (h2s count) by random subsampling so the blend stays
<= ~phx_cap_mult:1 (doc §3.4). sample_weights is also written (inverse source
frequency) for the record / a future weighted sampler.

Usage:
    python scripts/merge_h2s_phoenix_for_dposerx.py --phx_cap_mult 10
    # PHX-only-present check: missing PHOENIX splits are skipped (How2Sign still emitted).
"""
import os
import argparse
import pickle
import numpy as np

H2S = "/home/haipd/DexAvatar/data/signbposer_data"
PHX = "/home/haipd/DexAvatar/data/signbposer_data/raw/phoenix_sign"
OUT = "/home/haipd/DexAvatar/data/signbposer_data_sign"

# (out_split, h2s_split, phx_split)
SPLIT_MAP = [("train", "train", "train"),
             ("valid", "val", "dev"),
             ("test", "test", "test")]


def load_npy(path):
    if not os.path.exists(path):
        return None
    a = np.load(path).astype(np.float32)
    if a.ndim == 2 and a.shape[1] == 63:
        return a
    if a.ndim == 1 and a.shape[0] == 63:
        return a[None, :]
    if a.size % 63 == 0:  # try rescue flat arrays
        return a.reshape(-1, 63)
    raise ValueError(f"Unexpected shape {a.shape} in {path} (expected (*,63))")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h2s_dir", default=H2S)
    ap.add_argument("--phx_dir", default=PHX)
    ap.add_argument("--out_dir", default=OUT)
    ap.add_argument("--phx_cap_mult", type=float, default=10.0,
                    help="Cap PHOENIX per split to phx_cap_mult * h2s_count (0 = no cap).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    for out_split, h2s_split, phx_split in SPLIT_MAP:
        h2s = load_npy(os.path.join(args.h2s_dir, h2s_split, "body_poses.npy"))
        phx = load_npy(os.path.join(args.phx_dir, phx_split, "body_poses.npy"))

        parts = []  # (array, source_tag)
        if h2s is not None and len(h2s):
            parts.append((h2s, "h2s"))
        if phx is not None and len(phx):
            if args.phx_cap_mult > 0 and h2s is not None and len(h2s):
                cap = int(args.phx_cap_mult * len(h2s))
                if len(phx) > cap:
                    idx = rng.choice(len(phx), size=cap, replace=False)
                    phx = phx[idx]
                    print(f"  [{out_split}] capped PHOENIX {len(phx) if False else ''}-> {cap}")
            parts.append((phx, "phx"))

        if not parts:
            print(f"[{out_split}] no data (h2s={h2s is not None}, phx={phx is not None}); skip.")
            continue

        arrays = [p[0] for p in parts]
        sources = [p[1] for p in parts]
        merged = np.concatenate(arrays, axis=0).astype(np.float32)

        src_tags = []
        for arr, tag in parts:
            src_tags.extend([tag] * len(arr))
        meta = [{"source": t} for t in src_tags]

        # inverse-source-frequency sample weights (normalized to sum=N)
        counts = {t: src_tags.count(t) for t in set(src_tags)}
        weights = np.array([1.0 / counts[t] for t in src_tags], dtype=np.float32)
        weights *= len(weights) / weights.sum()

        out_split_dir = os.path.join(args.out_dir, out_split)
        os.makedirs(out_split_dir, exist_ok=True)
        np.save(os.path.join(out_split_dir, "body_poses.npy"), merged)
        with open(os.path.join(out_split_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(meta, f)
        np.save(os.path.join(out_split_dir, "sample_weights.npy"), weights)

        src_summary = ", ".join(f"{t}={counts[t]}" for t in sorted(counts))
        print(f"[{out_split}] N={len(merged)} ({src_summary}) -> {out_split_dir}")

    print("\nDone. Merged data under:", args.out_dir)


if __name__ == "__main__":
    main()
