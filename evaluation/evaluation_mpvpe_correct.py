#!/usr/bin/env python3
import argparse
import json
import pickle
from pathlib import Path
import numpy as np

# Load SMPL-X J_regressor
SMPLX_PKL_PATH = '/home/haipd/DexAvatar/SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.pkl'
print(f"Loading J_regressor from {SMPLX_PKL_PATH}...")
with open(SMPLX_PKL_PATH, 'rb') as f:
    smplx_data = pickle.load(f, encoding='latin1')
J_regressor = smplx_data['J_regressor']

def load_vertices(path: Path):
    if path.suffix == '.obj':
        verts = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        arr = np.array(verts, dtype=np.float32)
    elif path.suffix == '.npy':
        arr = np.load(path)
    elif path.suffix == '.npz':
        data = np.load(path)
        for key in ('vertices', 'verts', 'v'):
            if key in data:
                arr = data[key]
                break
        else:
            raise KeyError(f'No vertices/verts/v key in {path}')
    elif path.suffix == '.pkl':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for key in ('vertices', 'verts', 'v'):
            if isinstance(data, dict) and key in data:
                arr = data[key]
                break
        else:
            raise KeyError(f'No vertices/verts/v key in {path}')
    else:
        raise ValueError(f'Unsupported extension: {path.suffix}')
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f'Expected vertices shape (V,3), got {arr.shape} from {path}')
    return arr

def load_indices(path):
    p = Path(path)
    if p.suffix == '.npy':
        idx = np.load(p)
    elif p.suffix == '.json':
        with open(p, 'r', encoding='utf-8') as f:
            idx = np.asarray(json.load(f), dtype=np.int64)
    elif p.suffix == '.txt':
        idx = np.loadtxt(p, dtype=np.int64)
    else:
        raise ValueError(f'Unsupported index file: {p}')
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        raise ValueError(f'Empty indices in {path}')
    return idx

def collect_signs(signs_txt):
    signs = []
    with open(signs_txt, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            signs.append(parts[0])
    return signs

def collect_pairs(pred_root: Path, gt_root: Path, sign_name: str):
    pred_sign = pred_root / sign_name
    gt_sign = gt_root / sign_name
    if not pred_sign.exists() or not gt_sign.exists():
        return []

    # Find prediction files: smplifyx/meshes/ (smplerx/mesh/ is in camera space, not comparable to GT)
    pred_files = sorted(
        list(pred_sign.glob('**/meshes/low_*.obj')) +
        list(pred_sign.glob('**/meshes/low_*.pkl')) +
        list(pred_sign.glob('**/meshes/low_*.npz')) +
        list(pred_sign.glob('**/meshes/low_*.npy'))
    )
    pairs = []
    for pf in pred_files:
        stem = pf.stem
        # Extract frame index from low_XXX
        import re
        m = re.search(r'low_(\d+)', stem)
        if not m:
            continue
        pred_idx = int(m.group(1))
        gt_idx = pred_idx * 2
        
        # Check if corresponding GT file exists in gt_sign
        gt_file = gt_sign / f"{gt_idx:05d}.obj"
        if gt_file.exists():
            pairs.append((pf, gt_file))
    return pairs

def compute_mpvpe_part(pred_v, gt_v, idxs, regressor, joint_idx):
    if pred_v.shape[0] != 10475 or gt_v.shape[0] != 10475:
         raise ValueError(f"Expected 10475 vertices for SMPL-X mesh, got {pred_v.shape[0]} and {gt_v.shape[0]}")
    # Get joint coordinates
    pred_joint = np.dot(regressor[joint_idx], pred_v) # shape (3,)
    gt_joint = np.dot(regressor[joint_idx], gt_v) # shape (3,)
    
    # Align by subtracting joint coordinates
    pred_aligned = pred_v - pred_joint
    gt_aligned = gt_v - gt_joint
    
    # Filter by indices
    pred_part = pred_aligned[idxs]
    gt_part = gt_aligned[idxs]
    
    # Calculate Euclidean distance (L2 norm) in mm
    err = np.linalg.norm(pred_part - gt_part, axis=-1)
    return float(err.mean() * 1000.0)

def tr_v2v(pred_v, gt_v, idxs):
    """TR-V2V (mm) — per-region centroid alignment (SGNify paper Table 3)."""
    p = pred_v[idxs]
    g = gt_v[idxs]
    p = p - p.mean(axis=0, keepdims=True)
    g = g - g.mean(axis=0, keepdims=True)
    err = np.linalg.norm(p - g, axis=-1)
    return float(err.mean() * 1000.0)

def select_central_frames(pairs):
    """Filter pairs to central frames only: 0.5×T/8 < t < 7×T/8 (SGNify paper, Appendix C)."""
    import re
    if not pairs:
        return pairs
    gt_indices = []
    for pp, gp in pairs:
        m = re.search(r'(\d{5})', gp.stem)
        gt_indices.append(int(m.group(1)) if m else None)
    valid = [i for i in gt_indices if i is not None]
    if not valid:
        return pairs
    T = max(valid) - min(valid) + 1
    t_min = min(valid)
    core_lo = t_min + 0.5 * T / 8.0
    core_hi = t_min + 7.0 * T / 8.0
    return [(pp, gp) for (pp, gp), gt_idx in zip(pairs, gt_indices)
            if gt_idx is not None and core_lo < gt_idx < core_hi]

def main():
    ap = argparse.ArgumentParser(description='MPVPE evaluation on body parts (Pelvis/Wrist aligned).')
    ap.add_argument('--pred_root', required=True)
    ap.add_argument('--gt_root', required=True)
    ap.add_argument('--signs_txt', default='data/signs.txt')
    ap.add_argument('--segment_json', default='data/segment.json')
    ap.add_argument('--ubody_indices', required=True)
    ap.add_argument('--lhand_indices', required=True)
    ap.add_argument('--rhand_indices', required=True)
    ap.add_argument('--method_name', default='DexAvatar')
    ap.add_argument('--central_frames', action='store_true',
                    help='Filter to central frames only (0.5×T/8 < t < 7×T/8, per SGNify paper)')
    ap.add_argument('--output_csv', default='')
    ap.add_argument('--output_summary', default='')
    args = ap.parse_args()

    pred_root = Path(args.pred_root)
    gt_root = Path(args.gt_root)

    ubody_idx = load_indices(args.ubody_indices)
    lhand_idx = load_indices(args.lhand_indices)
    rhand_idx = load_indices(args.rhand_indices)

    signs = collect_signs(args.signs_txt)
    all_rows = []
    frame_rows = []
    sign_rows = []
    missing_signs = []
    central_skipped = 0

    for sign in signs:
        pairs = collect_pairs(pred_root, gt_root, sign)
        if args.central_frames:
            original_count = len(pairs)
            pairs = select_central_frames(pairs)
            central_skipped += original_count - len(pairs)
        if not pairs:
            missing_signs.append(sign)
            continue
        rows = []
        for pp, gp in pairs:
            pred_v = load_vertices(pp)
            gt_v = load_vertices(gp)

            # MPVPE (pelvis/wrist aligned)
            err_ubody = compute_mpvpe_part(pred_v, gt_v, ubody_idx, J_regressor, 0)
            err_lhand = compute_mpvpe_part(pred_v, gt_v, lhand_idx, J_regressor, 20)
            err_rhand = compute_mpvpe_part(pred_v, gt_v, rhand_idx, J_regressor, 21)

            # TR-V2V (pelvis-aligned, matching SGNify paper)
            t_ubody = tr_v2v(pred_v, gt_v, ubody_idx)
            t_lhand = tr_v2v(pred_v, gt_v, lhand_idx)
            t_rhand = tr_v2v(pred_v, gt_v, rhand_idx)

            metrics = [err_ubody, err_lhand, err_rhand, t_ubody, t_lhand, t_rhand]
            rows.append(metrics)
            frame_rows.append((sign, pp.stem, *metrics, str(pp), str(gp)))

        arr = np.asarray(rows, dtype=np.float32)
        mean = arr.mean(axis=0)
        sign_rows.append((sign, len(rows), mean))
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError('No matched prediction/GT pairs found across signs.')

    arr = np.asarray(all_rows, dtype=np.float32)
    mean = arr.mean(axis=0)

    frame_desc = "Central Frames" if args.central_frames else "All Frames"
    if args.central_frames:
        print(f'Frame filter: CENTRAL (0.5×T/8 < t < 7×T/8), {central_skipped} frames excluded')
    print()
    print('MPVPE (mm) — Pelvis/Wrist Aligned')
    print('Method,UBody(-F),LHand,RHand')
    print(f"{args.method_name},{mean[0]:.2f},{mean[1]:.2f},{mean[2]:.2f}")
    print()
    print('TR-V2V (mm) — Pelvis Aligned (SGNify)')
    print('Method,UBody(-F),LHand,RHand')
    print(f"{args.method_name},{mean[3]:.2f},{mean[4]:.2f},{mean[5]:.2f}")
    print(f'\nFrames,{len(all_rows)}')
    print(f'SignsCovered,{len(sign_rows)}/{len(signs)}')
    if missing_signs:
        print(f"MissingSigns,{len(missing_signs)},{' '.join(missing_signs)}")

    print('\nPer-sign summary:')
    print('Sign,Frames,MPVPE_UBody,MPVPE_LHand,MPVPE_RHand,TRV2V_UBody,TRV2V_LHand,TRV2V_RHand')
    for sign, n, m in sign_rows:
        print(f'{sign},{n},{m[0]:.2f},{m[1]:.2f},{m[2]:.2f},{m[3]:.2f},{m[4]:.2f},{m[5]:.2f}')

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            f.write('sign,frame,mpvpe_ubody,mpvpe_lhand,mpvpe_rhand,trv2v_ubody,trv2v_lhand,trv2v_rhand,pred_path,gt_path\n')
            for row in frame_rows:
                f.write(f'{row[0]},{row[1]},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f},{row[6]:.6f},{row[7]:.6f},{row[8]},{row[9]}\n')

    if args.output_summary:
        out = Path(args.output_summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            f.write('method,mpvpe_ubody,mpvpe_lhand,mpvpe_rhand,trv2v_ubody,trv2v_lhand,trv2v_rhand,frames,signs_covered,total_signs\n')
            f.write(f'{args.method_name},{mean[0]:.6f},{mean[1]:.6f},{mean[2]:.6f},{mean[3]:.6f},{mean[4]:.6f},{mean[5]:.6f},{len(all_rows)},{len(sign_rows)},{len(signs)}\n')

if __name__ == '__main__':
    main()
