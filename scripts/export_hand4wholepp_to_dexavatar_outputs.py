#!/usr/bin/env python3
"""
Export Hand4Whole++ cached predictions (57 signs, 1493 frames)
to DexAvatar outputs format (similar to outputs/method_hamer).
Structure:
  outputs/method_hand4wholepp/<sign>/smplifyx/meshes/low_<frame>.obj
  outputs/method_hand4wholepp/<sign>/smplifyx/results/low_<frame>.pkl
"""

import os
import json
import pickle
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor

CACHE_ROOT = Path('/home/haipd/DexAvatar/SignCAST/data/cache/v3/h4wpp')
OUTPUT_ROOT = Path('/home/haipd/DexAvatar/outputs/method_hand4wholepp')
SMPLX_NEUTRAL_PATH = Path('/home/haipd/DexAvatar/data/evaluation_from_author/data/data/SMPLX_NEUTRAL.npz')
SIGNS_FILE = Path('/home/haipd/DexAvatar/data/evaluation_from_author/data/data/signs.txt')

def get_face_lines():
    smplx_neutral = np.load(SMPLX_NEUTRAL_PATH, allow_pickle=True)
    faces = smplx_neutral['f'] + 1  # 1-based index for OBJ
    return ''.join([f'f {f[0]} {f[1]} {f[2]}\n' for f in faces])

def process_sign(sign_name: str, face_lines: str):
    sign_cache_dir = CACHE_ROOT / sign_name
    manifest_path = sign_cache_dir / 'manifest.json'
    if not manifest_path.is_file():
        print(f"[WARN] No manifest found for {sign_name}")
        return sign_name, 0

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    
    meshes_dir = OUTPUT_ROOT / sign_name / 'smplifyx' / 'meshes'
    results_dir = OUTPUT_ROOT / sign_name / 'smplifyx' / 'results'
    meshes_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for frame_info in manifest['frames']:
        fid = frame_info['frame_id']
        npz_file = sign_cache_dir / frame_info['cache']
        data = np.load(npz_file)

        # 1. Write OBJ mesh
        verts = data['smplx_vertices']  # (10475, 3)
        obj_file = meshes_dir / f"low_{fid:03d}.obj"
        
        lines = []
        for v in verts:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        lines.append(face_lines)
        
        with open(obj_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        # 2. Write PKL results
        focal = data['camera_focal']
        princpt = data['camera_principal_point']
        K = np.array([
            [focal[0], 0.0, princpt[0]],
            [0.0, focal[1], princpt[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        result_dict = {
            'betas': data['smplx_shape'].reshape(1, -1),
            'global_orient': data['smplx_root_pose_aa'].reshape(1, -1),
            'body_pose': data['smplx_body_pose_aa'].reshape(1, -1),
            'transl': data['smplx_trans'].reshape(1, -1),
            'left_hand_pose': data['smplx_lhand_pose_aa'].reshape(1, -1),
            'right_hand_pose': data['smplx_rhand_pose_aa'].reshape(1, -1),
            'expression': data['smplx_expression'].reshape(1, -1),
            'K': K,
        }
        pkl_file = results_dir / f"low_{fid:03d}.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(result_dict, f)

        count += 1

    return sign_name, count

def main():
    face_lines = get_face_lines()
    with open(SIGNS_FILE, 'r', encoding='utf-8') as f:
        signs = [line.strip().split()[0] for line in f if line.strip()]
    signs = sorted(signs)

    print(f"Exporting Hand4Whole++ predictions for {len(signs)} signs to {OUTPUT_ROOT}...")
    total_frames = 0
    for s in signs:
        _, count = process_sign(s, face_lines)
        total_frames += count
        print(f"  [DONE] {s}: {count} frames")

    print(f"\nExport completed: {len(signs)} signs, {total_frames} frames exported to {OUTPUT_ROOT}")

if __name__ == '__main__':
    main()
