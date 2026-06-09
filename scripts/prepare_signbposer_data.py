#!/usr/bin/env python3
"""
Prepare unified training data cho SignBPoser replacement.
Tạo training data từ SignAvatars và PHOENIX14T SMPL-X extractions.

Output:
    data/signbposer_data/
    ├── train/
    │   ├── body_poses.npy       # (N_train, 63)
    │   ├── metadata.pkl         # source, signer_id, sign_class
    │   └── sample_weights.npy   # (N_train,)
    ├── val/
    │   ├── body_poses.npy
    │   └── metadata.pkl
    └── test/
        ├── body_poses.npy
        └── metadata.pkl

Usage:
    python scripts/prepare_signbposer_data.py \
        --signavatars_dir /path/to/SignAvatars \
        --phoenix_dir /path/to/PHOENIX14T/smplx \
        --output_dir data/signbposer_data
"""

import os
import sys
import argparse
import pickle
import glob
import numpy as np
from collections import defaultdict

# Thêm DexAvatar root vào path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_signavatars_data(signavatars_dir):
    """
    Load body_pose từ SignAvatars dataset.

    SignAvatars có SMPL-X annotations sẵn, format tùy thuộc vào download.
    Hàm này handle nhiều format khác nhau.

    Returns:
        body_poses: (N, 63) axis-angle
        metadata: list of dicts
    """
    body_poses = []
    metadata = []

    # Thử load từ pkl files
    pkl_files = sorted(glob.glob(os.path.join(signavatars_dir, '**', '*.pkl'), recursive=True))

    if pkl_files:
        print(f"[SignAvatars] Found {len(pkl_files)} pkl files")
        for pkl_path in pkl_files:
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)

                # Handle different formats
                if isinstance(data, dict):
                    if 'body_pose' in data:
                        bp = np.array(data['body_pose']).flatten()
                        if len(bp) == 63:
                            body_poses.append(bp)
                            metadata.append({
                                'source': 'signavatars',
                                'file': os.path.basename(pkl_path),
                                'signer_id': data.get('signer_id', 'unknown'),
                                'sign_class': data.get('sign_class', 'unknown'),
                            })
                    elif 'smplx_params' in data:
                        smplx = data['smplx_params']
                        if 'body_pose' in smplx:
                            bp = np.array(smplx['body_pose']).flatten()
                            if len(bp) == 63:
                                body_poses.append(bp)
                                metadata.append({
                                    'source': 'signavatars',
                                    'file': os.path.basename(pkl_path),
                                })
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'body_pose' in item:
                            bp = np.array(item['body_pose']).flatten()
                            if len(bp) == 63:
                                body_poses.append(bp)
                                metadata.append({'source': 'signavatars'})
            except Exception as e:
                continue

    # Thử load từ npy files
    if not body_poses:
        npy_files = sorted(glob.glob(os.path.join(signavatars_dir, '**', '*.npy'), recursive=True))
        print(f"[SignAvatars] Found {len(npy_files)} npy files")
        for npy_path in npy_files:
            try:
                data = np.load(npy_path)
                if data.shape == (63,) or (data.ndim == 2 and data.shape[1] == 63):
                    if data.ndim == 1:
                        body_poses.append(data)
                        metadata.append({'source': 'signavatars', 'file': os.path.basename(npy_path)})
                    else:
                        for i in range(data.shape[0]):
                            body_poses.append(data[i])
                            metadata.append({'source': 'signavatars'})
            except Exception as e:
                continue

    # Thử load từ directory structure (frame-level)
    if not body_poses:
        for subdir in sorted(os.listdir(signavatars_dir)):
            subdir_path = os.path.join(signavatars_dir, subdir)
            if os.path.isdir(subdir_path):
                # Look for body_pose files
                for fname in sorted(os.listdir(subdir_path)):
                    fpath = os.path.join(subdir_path, fname)
                    try:
                        if fname.endswith('.pkl'):
                            with open(fpath, 'rb') as f:
                                data = pickle.load(f)
                            if isinstance(data, dict) and 'body_pose' in data:
                                bp = np.array(data['body_pose']).flatten()
                                if len(bp) == 63:
                                    body_poses.append(bp)
                                    metadata.append({
                                        'source': 'signavatars',
                                        'signer_id': subdir,
                                        'file': fname,
                                    })
                        elif fname.endswith('.npy'):
                            data = np.load(fpath)
                            if data.shape == (63,):
                                body_poses.append(data)
                                metadata.append({
                                    'source': 'signavatars',
                                    'signer_id': subdir,
                                    'file': fname,
                                })
                    except Exception:
                        continue

    if body_poses:
        body_poses = np.array(body_poses, dtype=np.float32)
        print(f"[SignAvatars] Loaded {len(body_poses)} body poses, shape: {body_poses.shape}")
    else:
        print("[SignAvatars] WARNING: No body poses found!")

    return body_poses, metadata


def load_phoenix_data(phoenix_dir):
    """
    Load body_pose từ PHOENIX14T SMPL-X extractions.

    Expected format: pkl files từ SMPLer-X output
    Each pkl contains: global_orient, body_pose, left_hand_pose, etc.

    Returns:
        body_poses: (N, 63) axis-angle
        metadata: list of dicts
    """
    body_poses = []
    metadata = []

    # SMPLer-X output format: smplx/*.pkl
    pkl_files = sorted(glob.glob(os.path.join(phoenix_dir, '**', '*.pkl'), recursive=True))

    if not pkl_files:
        # Thử trực tiếp trong phoenix_dir
        pkl_files = sorted(glob.glob(os.path.join(phoenix_dir, '*.pkl')))

    print(f"[PHOENIX14T] Found {len(pkl_files)} pkl files")

    for pkl_path in pkl_files:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict) and 'body_pose' in data:
                bp = np.array(data['body_pose']).flatten()
                if len(bp) == 63:
                    # Filter outlier
                    if np.linalg.norm(bp) < 10.0:  # reasonable bound
                        body_poses.append(bp)
                        metadata.append({
                            'source': 'phoenix14t',
                            'file': os.path.basename(pkl_path),
                            'split': _get_phoenix_split(pkl_path),
                        })
        except Exception as e:
            continue

    if body_poses:
        body_poses = np.array(body_poses, dtype=np.float32)
        print(f"[PHOENIX14T] Loaded {len(body_poses)} body poses, shape: {body_poses.shape}")
    else:
        print("[PHOENIX14T] WARNING: No body poses found!")

    return body_poses, metadata


def _get_phoenix_split(filepath):
    """Determine PHOENIX14T split from file path."""
    path_lower = filepath.lower()
    if 'train' in path_lower:
        return 'train'
    elif 'dev' in path_lower or 'val' in path_lower:
        return 'val'
    elif 'test' in path_lower:
        return 'test'
    return 'unknown'


def normalize_body_pose(body_poses):
    """
    Normalize body poses:
    1. Clamp values to [-π, π]
    2. Remove outliers (||pose|| > mean + 2*std)
    """
    # Clamp
    body_poses = np.clip(body_poses, -np.pi, np.pi)

    # Remove outliers
    norms = np.linalg.norm(body_poses, axis=1)
    mean_norm = np.mean(norms)
    std_norm = np.std(norms)
    threshold = mean_norm + 2 * std_norm

    mask = norms < threshold
    removed = np.sum(~mask)
    if removed > 0:
        print(f"[Normalize] Removed {removed} outlier poses (threshold={threshold:.2f})")

    return body_poses[mask], mask


def split_by_signer(body_poses, metadata, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split data theo signer_id (không phải random frame-level).
    Đảm bảo không có signer nào xuất hiện ở cả train lẫn test.
    """
    # Group by signer
    signer_to_indices = defaultdict(list)
    for i, meta in enumerate(metadata):
        signer_id = meta.get('signer_id', f'unknown_{i}')
        signer_to_indices[signer_id].append(i)

    signers = list(signer_to_indices.keys())
    np.random.seed(seed)
    np.random.shuffle(signers)

    n_signers = len(signers)
    n_test = max(1, int(n_signers * test_ratio))
    n_val = max(1, int(n_signers * val_ratio))

    test_signers = signers[:n_test]
    val_signers = signers[n_test:n_test + n_val]
    train_signers = signers[n_test + n_val:]

    splits = {
        'train': [],
        'val': [],
        'test': [],
    }

    for signer in train_signers:
        splits['train'].extend(signer_to_indices[signer])
    for signer in val_signers:
        splits['val'].extend(signer_to_indices[signer])
    for signer in test_signers:
        splits['test'].extend(signer_to_indices[signer])

    print(f"[Split] Train: {len(splits['train'])} samples ({len(train_signers)} signers)")
    print(f"[Split] Val:   {len(splits['val'])} samples ({len(val_signers)} signers)")
    print(f"[Split] Test:  {len(splits['test'])} samples ({len(test_signers)} signers)")

    return splits


def compute_sample_weights(metadata):
    """
    Compute sample weights:
    - SignAvatars (pseudo-GT): weight = 0.5
    - PHOENIX14T (DGS domain): weight = 1.2
    """
    weights = []
    for meta in metadata:
        source = meta.get('source', 'unknown')
        if source == 'signavatars':
            weights.append(0.5)  # pseudo-GT, downweight
        elif source == 'phoenix14t':
            weights.append(1.2)  # DGS domain, upweight
        else:
            weights.append(1.0)

    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()  # normalize
    return weights


def main():
    parser = argparse.ArgumentParser(description='Prepare SignBPoser training data')
    parser.add_argument('--signavatars_dir', type=str, default='',
                        help='Path to SignAvatars dataset')
    parser.add_argument('--phoenix_dir', type=str, default='',
                        help='Path to PHOENIX14T SMPL-X extractions')
    parser.add_argument('--output_dir', type=str, default='data/signbposer_data',
                        help='Output directory')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    all_poses = []
    all_metadata = []

    if args.signavatars_dir and os.path.exists(args.signavatars_dir):
        poses, meta = load_signavatars_data(args.signavatars_dir)
        if len(poses) > 0:
            all_poses.append(poses)
            all_metadata.extend(meta)

    if args.phoenix_dir and os.path.exists(args.phoenix_dir):
        poses, meta = load_phoenix_data(args.phoenix_dir)
        if len(poses) > 0:
            all_poses.append(poses)
            all_metadata.extend(meta)

    if not all_poses:
        print("ERROR: No data loaded! Check input directories.")
        return

    # Merge
    all_poses = np.concatenate(all_poses, axis=0)
    print(f"\n[Total] {len(all_poses)} body poses loaded")
    print(f"[Total] Shape: {all_poses.shape}")

    # Normalize
    all_poses, mask = normalize_body_pose(all_poses)
    all_metadata = [m for m, keep in zip(all_metadata, mask) if keep]

    # Split
    splits = split_by_signer(all_poses, all_metadata,
                             args.val_ratio, args.test_ratio, args.seed)

    # Compute weights
    weights = compute_sample_weights(all_metadata)

    # Save
    for split_name, indices in splits.items():
        if not indices:
            print(f"[Save] Skipping {split_name} (empty)")
            continue

        split_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        split_poses = all_poses[indices]
        split_meta = [all_metadata[i] for i in indices]
        split_weights = weights[indices]

        np.save(os.path.join(split_dir, 'body_poses.npy'), split_poses)
        with open(os.path.join(split_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(split_meta, f)
        np.save(os.path.join(split_dir, 'sample_weights.npy'), split_weights)

        print(f"[Save] {split_name}: {len(split_poses)} poses saved to {split_dir}")

    print(f"\nDone! Data saved to {args.output_dir}")


if __name__ == '__main__':
    main()
