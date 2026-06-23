#!/usr/bin/env python3
"""Runner: NLF + WiLoR + Original DexAvatar (no VQVAE/DPoserX) for all signs.

Reuses shared pre-extracted Sapiens/WiLoR stages.
Launches sequentially. Skip signs already complete (meshes match GT segment count).

Usage:
    python scripts/run_nlf_wilor_all.py \
        --input_img_folder data/frames \
        --output_path outputs/method_nlf_wilor \
        --fitting_experiment /home/haipd/DexAvatar/dexavatar_fitting
"""
import os, sys, json, glob, re, argparse
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--input_img_folder', type=str, default='data/frames')
parser.add_argument('--output_path', type=str, default='outputs/method_nlf_wilor')
parser.add_argument('--fitting_experiment', type=str, default='/home/haipd/DexAvatar/dexavatar_fitting')
args = parser.parse_args()

inp_img_folder = os.path.join(PROJECT_DIR, args.input_img_folder)
base_output_dir = os.path.join(PROJECT_DIR, args.output_path)

# Load GT segment to compute expected frame count per sign
with open(os.path.join(PROJECT_DIR, 'data', 'segment.json')) as f:
    frame_seg = json.load(f)

sub_folder_list = sorted(os.listdir(inp_img_folder))

total = 0
skipped = 0
for sub_folder in sub_folder_list:
    input_folder = os.path.abspath(os.path.join(inp_img_folder, sub_folder))
    if not os.path.isdir(input_folder):
        continue
    total += 1

    out_folder = os.path.abspath(os.path.join(base_output_dir, sub_folder))
    os.makedirs(out_folder, exist_ok=True)

    # Check if already complete: mesh count matches expected GT segment frames
    seg = frame_seg.get(sub_folder, [0, 0])
    expected_frames = (seg[1] - seg[0]) // 2 + 1  # frames are named low_{idx}.png with step 2

    meshes_dir = os.path.join(out_folder, 'smplifyx', 'meshes')
    if os.path.isdir(meshes_dir):
        existing_meshes = glob.glob(os.path.join(meshes_dir, '*.obj'))
        mesh_count = len(existing_meshes)
    else:
        mesh_count = 0

    if mesh_count == expected_frames:
        print(f"[{skipped+1}/{total}] SKIP {sub_folder}: already complete ({mesh_count} meshes)")
        skipped += 1
        continue

    print(f"[{skipped+1}/{total}] RUN {sub_folder}: {mesh_count}/{expected_frames} meshes, need {expected_frames}")
    skipped += 1

    cmd = (
        f"cd {PROJECT_DIR} && "
        f"ROOT_PATH={input_folder} "
        f"OUTPUT_PATH={out_folder} "
        f"FITTING_EXPERIMENT={args.fitting_experiment} "
        f"bash -c 'unset LD_LIBRARY_PATH && bash {SCRIPT_DIR}/pipeline_nlf_wilor.sh'"
    )
    print(f"  CMD: {cmd[:200]}...")
    ret = os.system(cmd)
    if ret != 0:
        print(f"  ERROR: {sub_folder} failed with code {ret}")
    else:
        print(f"  DONE: {sub_folder}")

print(f"\nAll done. Processed {total} signs.")
