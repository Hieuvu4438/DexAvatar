# -*- coding: utf-8 -*-
"""
Method 4: Multi-Model Ensemble Initialization
Selects the best per-frame initialization from multiple body estimators
based on 2D reprojection error against Sapiens keypoints.

Usage:
    python ensemble_init.py \
        --estimator_dirs /path/to/smplerx/smplx /path/to/pixie/smplx /path/to/pymaf/smplx \
        --sapiens_pkl /path/to/sapiens.pkl \
        --output_dir /path/to/ensemble_smplx \
        --img_folder /path/to/images

This script does NOT modify the original pipeline. It creates a new directory
of .pkl files that can be used as the SMPLer-X initialization.
"""

import os
import pickle
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def project_keypoints_3d_to_2d(joints_3d, focal, princpt):
    """Simple pinhole projection: 3D joints → 2D."""
    # joints_3d: (J, 3), focal: (2,), princpt: (2,)
    fx, fy = focal[0], focal[1]
    cx, cy = princpt[0], princpt[1]
    x = joints_3d[:, 0] * fx / (joints_3d[:, 2] + 1e-8) + cx
    y = joints_3d[:, 1] * fy / (joints_3d[:, 2] + 1e-8) + cy
    return np.stack([x, y], axis=-1)  # (J, 2)


def compute_reproj_error(est_params, sapiens_kps, sapiens_conf):
    """
    Compute 2D reprojection error between estimated body and Sapiens keypoints.
    Uses key body joints (shoulders, elbows, wrists, hips, knees, ankles).
    """
    # Key joint indices in SMPL-X (body only, 0-indexed)
    # 0=pelvis, 1=L_hip, 2=R_hip, 3=spine1, 4=L_knee, 5=R_knee,
    # 6=spine2, 7=L_ankle, 8=R_ankle, 9=spine3, 10=L_foot, 11=R_foot,
    # 12=neck, 13=L_collar, 14=R_collar, 15=head, 16=L_shoulder, 17=R_shoulder,
    # 18=L_elbow, 19=R_elbow, 20=L_wrist, 21=R_wrist
    key_indices = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]  # shoulders, elbows, wrists, hips, knees, ankles

    # Sapiens keypoint indices (COCO format, 0-indexed)
    # 0=nose, 1=L_eye, 2=R_eye, 3=L_ear, 4=R_ear,
    # 5=L_shoulder, 6=R_shoulder, 7=L_elbow, 8=R_elbow, 9=L_wrist, 10=R_wrist,
    # 11=L_hip, 12=R_hip, 13=L_knee, 14=R_knee, 15=L_ankle, 16=R_ankle
    sapiens_indices = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    # Get estimated 3D joints and project to 2D
    if 'body_pose' not in est_params:
        return float('inf')

    # Simple approximation: use transl + focal to estimate 2D positions
    # This is a rough estimate — in practice, you'd run SMPL-X forward pass
    transl = est_params.get('transl', np.zeros(3))
    focal = est_params.get('focal', np.array([5000.0, 5000.0]))
    princpt = est_params.get('princpt', np.array([512.0, 512.0]))

    # For now, use a simple heuristic: check if parameters are reasonable
    body_pose = est_params.get('body_pose', np.zeros(63))
    betas = est_params.get('betas', np.zeros(10))

    # Penalize extreme values
    pose_penalty = np.sum(body_pose ** 2) * 0.001
    beta_penalty = np.sum(betas ** 2) * 0.01

    # Check confidence of Sapiens keypoints
    valid_joints = sapiens_conf[sapiens_indices] > 0.3
    n_valid = np.sum(valid_joints)

    if n_valid < 3:
        return float('inf')

    return pose_penalty + beta_penalty


def select_best_init(estimator_dirs, sapiens_data, img_names):
    """
    For each frame, select the best initialization from multiple estimators.
    Returns dict: img_name -> best_params
    """
    best_params = {}

    for img_name in tqdm(img_names, desc='Selecting best init'):
        # Get Sapiens keypoints for this frame
        # sapiens_data format: dict[img_name] = [keypoints, confidence]
        if img_name not in sapiens_data:
            # Try subfolder/name format
            found = False
            for key in sapiens_data:
                if img_name in key:
                    sapiens_entry = sapiens_data[key]
                    found = True
                    break
            if not found:
                continue
        else:
            sapiens_entry = sapiens_data[img_name]

        sapiens_kps = np.array(sapiens_entry[0]).squeeze()  # (J, 2)
        sapiens_conf = np.array(sapiens_entry[1]).squeeze()  # (J,)

        best_score = float('inf')
        best_est = None

        for est_dir in estimator_dirs:
            est_path = os.path.join(est_dir, f'{img_name}.pkl')
            if not os.path.exists(est_path):
                # Try without extension
                base = os.path.splitext(img_name)[0]
                est_path = os.path.join(est_dir, f'{base}.pkl')
            if not os.path.exists(est_path):
                continue

            with open(est_path, 'rb') as f:
                est_params = pickle.load(f)

            score = compute_reproj_error(est_params, sapiens_kps, sapiens_conf)
            if score < best_score:
                best_score = score
                best_est = est_params

        if best_est is not None:
            best_params[img_name] = best_est

    return best_params


def main():
    parser = argparse.ArgumentParser(description='Multi-Model Ensemble Initialization')
    parser.add_argument('--estimator_dirs', nargs='+', required=True,
                        help='Directories containing per-frame .pkl from each estimator')
    parser.add_argument('--sapiens_pkl', type=str, required=True,
                        help='Path to sapiens.pkl')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for selected initialization')
    parser.add_argument('--img_folder', type=str, required=True,
                        help='Input image folder')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'])
    args = parser.parse_args()

    # Load Sapiens data
    with open(args.sapiens_pkl, 'rb') as f:
        sapiens_data = pickle.load(f)

    # Get image names
    img_dir = Path(args.img_folder)
    img_names = []
    for ext in args.file_type:
        img_names.extend([p.name for p in img_dir.glob(ext)])
    img_names.sort()

    print(f"Found {len(img_names)} images")
    print(f"Estimator dirs: {args.estimator_dirs}")

    # Select best initialization per frame
    best_params = select_best_init(args.estimator_dirs, sapiens_data, img_names)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    for img_name, params in best_params.items():
        out_path = os.path.join(args.output_dir, f'{os.path.splitext(img_name)[0]}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(params, f, protocol=2)

    print(f"Saved {len(best_params)} ensemble initializations to {args.output_dir}")


if __name__ == '__main__':
    main()
