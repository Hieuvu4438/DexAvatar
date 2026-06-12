#!/usr/bin/env python3
"""
Evaluate active hands only (SGNify paper protocol).
For one-handed signs (Class 0), the left hand is inactive and not evaluated.
Therefore, LHand is only averaged over two-handed signs, while RHand and UBody are averaged over all signs.

Usage:
    python evaluation/evaluate_active_hands.py --csv <path_to_frames_csv> --signs_txt data/signs.txt
"""

import argparse
import csv
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Compute active-hand TR-V2V metrics matching the paper.")
    parser.add_argument("--csv", required=True, help="Path to per-frame CSV from evaluation script.")
    parser.add_argument("--signs_txt", default="data/signs.txt", help="Path to signs.txt containing sign classes.")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found at {args.csv}")
        return

    one_handed_signs = set()
    two_handed_signs = set()

    with open(args.signs_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            sign = parts[0]
            cls = parts[1] if len(parts) > 1 else "~0"
            if cls == "0":
                one_handed_signs.add(sign)
            else:
                two_handed_signs.add(sign)

    rows = []
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Detect header naming style
    fields = reader.fieldnames
    ub_key = next((f for f in fields if 'UBody' in f or 'ubody' in f), None)
    lh_key = next((f for f in fields if 'LHand' in f or 'lhand' in f), None)
    rh_key = next((f for f in fields if 'RHand' in f or 'rhand' in f), None)

    if not all([ub_key, lh_key, rh_key]):
        print(f"Error: Could not identify metrics columns. Found fields: {fields}")
        return

    ubody_all = []
    lhand_all = []
    rhand_all = []
    lhand_two_handed = []

    for r in rows:
        sign = r['sign']
        ub = float(r[ub_key])
        lh = float(r[lh_key])
        rh = float(r[rh_key])
        
        ubody_all.append(ub)
        lhand_all.append(lh)
        rhand_all.append(rh)
        
        if sign in two_handed_signs:
            lhand_two_handed.append(lh)

    print("\n" + "="*50)
    print(f"  TR-V2V Evaluation Report ({os.path.basename(args.csv)})")
    print("="*50)
    print("1) Standard Average (All 57 signs):")
    print(f"   - UBody(-F) : {np.mean(ubody_all):.2f} mm")
    print(f"   - LHand     : {np.mean(lhand_all):.2f} mm")
    print(f"   - RHand     : {np.mean(rhand_all):.2f} mm")
    print("-"*50)
    print("2) SGNify Active-Hand Protocol (Dominant/Two-Handed Only):")
    print(f"   - UBody(-F) : {np.mean(ubody_all):.2f} mm (All 57 signs)")
    print(f"   - LHand     : {np.mean(lhand_two_handed):.2f} mm (Only 42 two-handed signs)")
    print(f"   - RHand     : {np.mean(rhand_all):.2f} mm (All 57 signs)")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
