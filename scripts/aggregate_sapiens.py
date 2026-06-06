#!/usr/bin/env python3
"""Aggregate Sapiens per-frame JSON outputs into a single sapiens.pkl file.

Usage:
    python aggregate_sapiens.py --sapiens_dir OUTPUT_PATH/sapiens_1b --output_path OUTPUT_PATH [--subfolder SUBFOLDER]

The script reads all JSON files from sapiens_dir, extracts keypoints + confidence scores,
and creates a pickle file at output_path/sapiens.pkl with format:
    dict[subfolder/filename.png] = [keypoints_array[1,133,2], confidence_array[1,133]]
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sapiens_dir', required=True, help='Path to sapiens_1b directory with JSON outputs')
    parser.add_argument('--output_path', required=True, help='Output path where sapiens.pkl will be saved')
    parser.add_argument('--subfolder', default=None, help='Subfolder name for dictionary keys (default: auto-detect)')
    args = parser.parse_args()

    sapiens_dir = args.sapiens_dir
    if not os.path.isdir(sapiens_dir):
        print(f"ERROR: sapiens_dir not found: {sapiens_dir}")
        sys.exit(1)

    json_files = sorted([f for f in os.listdir(sapiens_dir) if f.endswith('.json')])
    if not json_files:
        print(f"ERROR: No JSON files found in {sapiens_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files in {sapiens_dir}")

    if args.subfolder:
        subfolder = args.subfolder
    else:
        subfolder = os.path.basename(os.path.normpath(args.output_path))

    sapiens_dict = {}
    for jf in json_files:
        json_path = os.path.join(sapiens_dir, jf)
        with open(json_path, 'r') as f:
            data = json.load(f)

        instances = data.get('instance_info', [])
        if not instances:
            print(f"  WARNING: No instances in {jf}, skipping")
            continue

        inst = instances[0]
        kps = np.array(inst['keypoints'], dtype=np.float32)           # [133, 2]
        scores = np.array(inst['keypoint_scores'], dtype=np.float32)  # [133]

        kps = kps[np.newaxis, ...]       # [1, 133, 2]
        scores = scores[np.newaxis, ...]  # [1, 133]

        img_name = jf.replace('.json', '.png')
        key = os.path.join(subfolder, img_name)
        sapiens_dict[key] = [kps, scores]

    output_file = os.path.join(args.output_path, 'sapiens.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(sapiens_dict, f)

    print(f"Saved {len(sapiens_dict)} entries to {output_file}")


if __name__ == '__main__':
    main()
