"""Compare methods on a common sign subset with active-hand TR-V2V.

Reads the per-frame CSV from eval_mpvpe_regions.py and prints, for each method,
the mean TR-V2V for UBody(-F) / LHand / RHand, applying the SGNify active-hand
protocol (LHand averaged over two-handed signs only; one-handed signs' LHand
dropped). Restricted to the signs present in --ref_method so the comparison is
apples-to-apples.
"""
from __future__ import annotations
import argparse, csv
import numpy as np

def load_sign_classes(path):
    cls = {}
    with open(path) as f:
        for line in f:
            t = line.strip().split()
            if t:
                cls[t[0]] = t[1] if len(t) > 1 else "~0"
    return cls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="per-frame CSV from eval_mpvpe_regions.py")
    ap.add_argument("--signs_txt", default="/home/haipd/DexAvatar/data/signs.txt")
    ap.add_argument("--ref_method", default="exp1_paper_nlf",
                    help="restrict comparison to signs present in this method")
    args = ap.parse_args()

    cls = load_sign_classes(args.signs_txt)
    # rows[method][sign] = {ub:[], lh:[], rh:[]}
    rows = {}
    ref_signs = set()
    with open(args.csv) as f:
        r = csv.DictReader(f)
        for row in r:
            m = row["method"]; s = row["sign"]
            d = rows.setdefault(m, {}).setdefault(s, {"ub": [], "lh": [], "rh": []})
            d["ub"].append(float(row["trv2v_ubody"]))
            d["lh"].append(float(row["trv2v_lhand"]))
            d["rh"].append(float(row["trv2v_rhand"]))
            if m == args.ref_method or args.ref_method == "ALL":
                ref_signs.add(s)

    methods = list(rows.keys())
    print(f"Signs in comparison (from {args.ref_method}): {sorted(ref_signs)}\n")
    print(f"{'method':<28}{'UBody-F':>9}{'LHand':>9}{'RHand':>9}   (active-hand, TR-V2V mm)")
    print("-" * 70)
    for m in methods:
        ub, lh, rh = [], [], []
        for s in ref_signs:
            if s not in rows[m]:
                continue
            d = rows[m][s]
            ub += d["ub"]; rh += d["rh"]
            if cls.get(s, "~0") != "0":   # two-handed only for LHand
                lh += d["lh"]
        f = lambda x: f"{np.mean(x):7.2f}" if x else "   n/a"
        print(f"{m:<28}{f(ub):>9}{f(lh):>9}{f(rh):>9}   "
              f"(n_frames UBody={len(ub)}, LHand={len(lh)}, RHand={len(rh)})")

if __name__ == "__main__":
    main()
