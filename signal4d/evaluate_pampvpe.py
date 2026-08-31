#!/usr/bin/env python3
"""
PA-MPVPE Evaluation Script for SIGNAL-4D / SMPL-X Meshes.
Standardized Procrustes-Aligned Mean Per-Vertex Position Error (PA-MPVPE) and MPVPE benchmark.

Evaluates:
1. Standard Conference Metrics:
   - PA-MPVPE (All): All 10,475 vertices
   - PA-MPVPE (Body + Hands, No Face): 5,452 vertices (Excluding FLAME face)
   - PA-MPVPE (Body Only): 3,896 vertices (Excluding Face & Hands)
   - PA-MPVPE (Right Hand): 778 MANO vertices
   - PA-MPVPE (Left Hand): 778 MANO vertices
   - PA-MPVPE (Hands): Average of Left & Right Hand
   - PA-MPVPE (Face): 5,023 FLAME vertices
   - PA-MPVPE (Upper Body): 8,888 vertices
   - PA-MPVPE (Upper Body Minus Face): 7,279 vertices
   - PA-MPVPE (Upper Body Minus Head): 3,865 vertices
2. Joint-aligned MPVPE (Pelvis/Wrist/Neck root-aligned)
3. Author Centroid-Aligned (TR) metrics
"""

import os
import os.path as osp
import sys
import json
import pickle
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from tqdm import tqdm


# ==============================================================================
# 1. Procrustes Alignment & Metric Primitives
# ==============================================================================

def rigid_transform_3D(A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama's SVD-based Procrustes Analysis:
    Minimizes || c * R @ A.T + t.reshape(3,1) - B.T ||_F^2
    """
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    AA = A - centroid_A
    BB = B - centroid_B
    
    H = np.dot(AA.T, BB) / n
    U, s, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        Vt[2, :] = -Vt[2, :]
        R = np.dot(Vt.T, U.T)
        
    varP = np.var(A, axis=0).sum()
    if varP < 1e-12:
        c = 1.0
    else:
        c = 1.0 / varP * np.sum(s)
        
    t = -np.dot(c * R, centroid_A) + centroid_B
    return c, R, t


def rigid_align(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    c, R, t = rigid_transform_3D(A, B)
    A_aligned = np.dot(A, (c * R).T) + t
    return A_aligned


def compute_pampvpe(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    pred_aligned = rigid_align(pred_pts, gt_pts)
    error_mm = np.sqrt(np.sum((pred_aligned - gt_pts) ** 2, axis=-1)).mean() * 1000.0
    return float(error_mm)


def compute_mpvpe_aligned(pred_pts: np.ndarray, gt_pts: np.ndarray, 
                          pred_root: np.ndarray, gt_root: np.ndarray) -> float:
    pred_aligned = pred_pts - pred_root.reshape(1, 3) + gt_root.reshape(1, 3)
    error_mm = np.sqrt(np.sum((pred_aligned - gt_pts) ** 2, axis=-1)).mean() * 1000.0
    return float(error_mm)


def compute_transl_error(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    pred_centered = pred_pts - np.mean(pred_pts, axis=0, keepdims=True)
    gt_centered = gt_pts - np.mean(gt_pts, axis=0, keepdims=True)
    error_mm = np.sqrt(np.sum((pred_centered - gt_centered) ** 2, axis=-1)).mean() * 1000.0
    return float(error_mm)


# ==============================================================================
# 2. Fast Mesh Loading
# ==============================================================================

def load_obj_verts(obj_filename: str) -> np.ndarray:
    verts = []
    with open(obj_filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('v '):
                tokens = line.strip().split()
                verts.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
    if not verts:
        raise ValueError(f"No vertices found in {obj_filename}")
    return np.array(verts, dtype=np.float32)


# ==============================================================================
# 3. Assets & Metadata Loader
# ==============================================================================

class SMPLXAssets:
    def __init__(self, data_base_dir: str = '/home/haipd/DexAvatar/data/evaluation_from_author/data/data'):
        # MANO hand vertex indices
        mano_path = osp.join(data_base_dir, 'MANO_SMPLX_vertex_ids.pkl')
        if not osp.exists(mano_path):
            alt_mano = '/home/haipd/DexAvatar/SMPLer-X/common/utils/human_model_files/smplx/MANO_SMPLX_vertex_ids.pkl'
            mano_path = alt_mano if osp.exists(alt_mano) else mano_path
            
        with open(mano_path, 'rb') as f:
            mano_data = pickle.load(f)
        self.left_hand_idx = np.array(mano_data['left_hand'], dtype=np.int64)
        self.right_hand_idx = np.array(mano_data['right_hand'], dtype=np.int64)
        
        # Face vertex indices (FLAME mapping on SMPL-X: 5,023 vertices)
        face_path = osp.join(data_base_dir, 'SMPL-X__FLAME_vertex_ids.npy')
        if not osp.exists(face_path):
            face_path = '/home/haipd/DexAvatar/SMPLer-X/common/utils/human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy'
        self.face_idx = np.load(face_path).astype(np.int64)
        
        # Non-face / Body-only vertex sets
        all_idx = np.arange(10475, dtype=np.int64)
        self.non_face_idx = np.setdiff1d(all_idx, self.face_idx)
        self.body_only_idx = np.setdiff1d(self.non_face_idx, np.concatenate([self.left_hand_idx, self.right_hand_idx]))
        
        # Upper body region indices
        ub_dir = osp.join(data_base_dir, 'sgnify_part_segm_above_pelvis_joint')
        self.upper_body_idx = np.load(osp.join(ub_dir, 'upper_body.npy')).astype(np.int64) if osp.exists(osp.join(ub_dir, 'upper_body.npy')) else None
        self.upper_body_minus_face_idx = np.load(osp.join(ub_dir, 'upper_body_minus_face.npy')).astype(np.int64) if osp.exists(osp.join(ub_dir, 'upper_body_minus_face.npy')) else None
        self.upper_body_minus_head_idx = np.load(osp.join(ub_dir, 'upper_body_minus_head.npy')).astype(np.int64) if osp.exists(osp.join(ub_dir, 'upper_body_minus_head.npy')) else None
            
        # Joint Regressor (from SMPLX_NEUTRAL.npz)
        smplx_path = osp.join(data_base_dir, 'SMPLX_NEUTRAL.npz')
        if osp.exists(smplx_path):
            smplx_model_data = np.load(smplx_path, allow_pickle=True)
            self.J_regressor = smplx_model_data['J_regressor']
        else:
            self.J_regressor = None
            
        self.joint_idx = {'pelvis': 0, 'neck': 12, 'lwrist': 20, 'rwrist': 21}


# ==============================================================================
# 4. Evaluation Engine
# ==============================================================================

def find_sign_reconstruction_meshes(evaluate_folder: str, sign_name: str) -> List[Tuple[int, str]]:
    possible_dirs = [
        osp.join(evaluate_folder, sign_name, 'smplifyx', 'meshes'),
        osp.join(evaluate_folder, sign_name, 'meshes'),
        osp.join(evaluate_folder, sign_name),
    ]
    
    mesh_dir = None
    for p in possible_dirs:
        if osp.exists(p) and any(f.endswith('.obj') for f in os.listdir(p)):
            mesh_dir = p
            break
            
    if mesh_dir is None:
        return []
        
    obj_files = [f for f in os.listdir(mesh_dir) if f.endswith('.obj')]
    
    parsed = []
    for fname in obj_files:
        nums = re.findall(r'\d+', fname)
        if nums:
            frame_idx = int(nums[-1])
            parsed.append((frame_idx, osp.join(mesh_dir, fname)))
            
    parsed.sort(key=lambda x: x[0])
    return parsed


def find_gt_mesh_for_frame(gt_folder: str, sign_name: str, pred_frame_num: int) -> Optional[str]:
    gt_dir = osp.join(gt_folder, sign_name)
    if not osp.exists(gt_dir):
        return None
        
    candidates = [
        osp.join(gt_dir, f"{pred_frame_num * 2:05d}.obj"),
        osp.join(gt_dir, f"{pred_frame_num * 2}.obj"),
        osp.join(gt_dir, f"{pred_frame_num:05d}.obj"),
        osp.join(gt_dir, f"{pred_frame_num}.obj"),
    ]
    
    for c in candidates:
        if osp.exists(c):
            return c
            
    return None


def evaluate_dataset(
    evaluate_folder: str,
    gt_folder: str,
    sign_file: str,
    sign_seg_file: str,
    assets: SMPLXAssets,
    verbose: bool = True
) -> Dict[str, Any]:
    
    class_sign = {}
    with open(sign_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                class_sign[tokens[0]] = tokens[1]
                
    sign_list = sorted(list(class_sign.keys()))
    
    frame_segment = {}
    if osp.exists(sign_seg_file):
        with open(sign_seg_file, 'r', encoding='utf-8') as f:
            frame_segment = json.load(f)

    metric_keys = [
        'pa_mpvpe_all',
        'pa_mpvpe_no_face',
        'pa_mpvpe_body_only',
        'pa_mpvpe_rhand',
        'pa_mpvpe_lhand',
        'pa_mpvpe_hand',
        'pa_mpvpe_hand_active',
        'pa_mpvpe_face',
        'pa_mpvpe_upper_body',
        'pa_mpvpe_ub_minus_face',
        'pa_mpvpe_ub_minus_head',
        'mpvpe_all',
        'mpvpe_rhand',
        'mpvpe_lhand',
        'mpvpe_hand',
        'mpvpe_face',
        'tr_all',
        'tr_rhand',
        'tr_lhand',
        'tr_upper_body',
        'tr_ub_minus_face',
        'tr_ub_minus_head',
    ]
    
    overall_frame_errors: Dict[str, List[float]] = {k: [] for k in metric_keys}
    per_sign_results: Dict[str, Dict[str, Any]] = {}
    
    total_evaluated_frames = 0
    total_evaluated_signs = 0
    
    sign_pbar = tqdm(sign_list, desc="Evaluating Signs", disable=not verbose)
    
    for sign_name in sign_pbar:
        sign_cls = class_sign.get(sign_name, '~0')
        is_one_handed = (sign_cls == '0')
        
        pred_mesh_items = find_sign_reconstruction_meshes(evaluate_folder, sign_name)
        if not pred_mesh_items:
            tqdm.write(f"[WARNING] No predicted meshes found for sign: {sign_name}")
            continue
            
        sign_frame_errors: Dict[str, List[float]] = {k: [] for k in metric_keys}
        sign_matched_frames = 0
        
        for pred_frame_idx, pred_mesh_path in pred_mesh_items:
            gt_mesh_path = find_gt_mesh_for_frame(gt_folder, sign_name, pred_frame_idx)
            if gt_mesh_path is None or not osp.exists(gt_mesh_path):
                continue
                
            pred_verts = load_obj_verts(pred_mesh_path)
            gt_verts = load_obj_verts(gt_mesh_path)
            
            if pred_verts.shape != gt_verts.shape:
                tqdm.write(f"[ERROR] Shape mismatch for {sign_name} frame {pred_frame_idx}: {pred_verts.shape} vs {gt_verts.shape}")
                continue
                
            # 1. PA-MPVPE (All: 10475 verts)
            sign_frame_errors['pa_mpvpe_all'].append(compute_pampvpe(pred_verts, gt_verts))
            
            # 2. PA-MPVPE (No Face: Body + Hands = 5452 verts)
            pred_noface = pred_verts[assets.non_face_idx]
            gt_noface = gt_verts[assets.non_face_idx]
            sign_frame_errors['pa_mpvpe_no_face'].append(compute_pampvpe(pred_noface, gt_noface))
            
            # 3. PA-MPVPE (Body Only: No Face, No Hands = 3896 verts)
            pred_body = pred_verts[assets.body_only_idx]
            gt_body = gt_verts[assets.body_only_idx]
            sign_frame_errors['pa_mpvpe_body_only'].append(compute_pampvpe(pred_body, gt_body))
            
            # 4. PA-MPVPE Hands
            pred_rhand = pred_verts[assets.right_hand_idx]
            gt_rhand = gt_verts[assets.right_hand_idx]
            pa_rhand = compute_pampvpe(pred_rhand, gt_rhand)
            sign_frame_errors['pa_mpvpe_rhand'].append(pa_rhand)
            
            pred_lhand = pred_verts[assets.left_hand_idx]
            gt_lhand = gt_verts[assets.left_hand_idx]
            pa_lhand = compute_pampvpe(pred_lhand, gt_lhand)
            sign_frame_errors['pa_mpvpe_lhand'].append(pa_lhand)
            
            pa_hand = (pa_rhand + pa_lhand) / 2.0
            sign_frame_errors['pa_mpvpe_hand'].append(pa_hand)
            sign_frame_errors['pa_mpvpe_hand_active'].append(pa_rhand if is_one_handed else pa_hand)
            
            # 5. PA-MPVPE Face
            pred_face = pred_verts[assets.face_idx]
            gt_face = gt_verts[assets.face_idx]
            sign_frame_errors['pa_mpvpe_face'].append(compute_pampvpe(pred_face, gt_face))
            
            # 6. Upper Body variants
            if assets.upper_body_idx is not None:
                sign_frame_errors['pa_mpvpe_upper_body'].append(compute_pampvpe(pred_verts[assets.upper_body_idx], gt_verts[assets.upper_body_idx]))
            if assets.upper_body_minus_face_idx is not None:
                sign_frame_errors['pa_mpvpe_ub_minus_face'].append(compute_pampvpe(pred_verts[assets.upper_body_minus_face_idx], gt_verts[assets.upper_body_minus_face_idx]))
            if assets.upper_body_minus_head_idx is not None:
                sign_frame_errors['pa_mpvpe_ub_minus_head'].append(compute_pampvpe(pred_verts[assets.upper_body_minus_head_idx], gt_verts[assets.upper_body_minus_head_idx]))
            
            # 7. Joint-aligned MPVPE
            if assets.J_regressor is not None:
                pred_joints = np.dot(assets.J_regressor, pred_verts)
                gt_joints = np.dot(assets.J_regressor, gt_verts)
                
                sign_frame_errors['mpvpe_all'].append(compute_mpvpe_aligned(pred_verts, gt_verts, 
                                                                            pred_joints[assets.joint_idx['pelvis']], 
                                                                            gt_joints[assets.joint_idx['pelvis']]))
                mp_rhand = compute_mpvpe_aligned(pred_rhand, gt_rhand,
                                                 pred_joints[assets.joint_idx['rwrist']],
                                                 gt_joints[assets.joint_idx['rwrist']])
                mp_lhand = compute_mpvpe_aligned(pred_lhand, gt_lhand,
                                                 pred_joints[assets.joint_idx['lwrist']],
                                                 gt_joints[assets.joint_idx['lwrist']])
                sign_frame_errors['mpvpe_rhand'].append(mp_rhand)
                sign_frame_errors['mpvpe_lhand'].append(mp_lhand)
                sign_frame_errors['mpvpe_hand'].append((mp_rhand + mp_lhand) / 2.0)
                sign_frame_errors['mpvpe_face'].append(compute_mpvpe_aligned(pred_face, gt_face,
                                                                             pred_joints[assets.joint_idx['neck']],
                                                                             gt_joints[assets.joint_idx['neck']]))
                
            # 8. Author Centroid-Aligned (TR)
            sign_frame_errors['tr_all'].append(compute_transl_error(pred_verts, gt_verts))
            sign_frame_errors['tr_rhand'].append(compute_transl_error(pred_rhand, gt_rhand))
            sign_frame_errors['tr_lhand'].append(compute_transl_error(pred_lhand, gt_lhand))
            if assets.upper_body_idx is not None:
                sign_frame_errors['tr_upper_body'].append(compute_transl_error(pred_verts[assets.upper_body_idx], gt_verts[assets.upper_body_idx]))
            if assets.upper_body_minus_face_idx is not None:
                sign_frame_errors['tr_ub_minus_face'].append(compute_transl_error(pred_verts[assets.upper_body_minus_face_idx], gt_verts[assets.upper_body_minus_face_idx]))
            if assets.upper_body_minus_head_idx is not None:
                sign_frame_errors['tr_ub_minus_head'].append(compute_transl_error(pred_verts[assets.upper_body_minus_head_idx], gt_verts[assets.upper_body_minus_head_idx]))
                
            sign_matched_frames += 1
            
        if sign_matched_frames == 0:
            continue
            
        total_evaluated_signs += 1
        total_evaluated_frames += sign_matched_frames
        
        per_sign_mean = {}
        for k in metric_keys:
            if sign_frame_errors[k]:
                mean_val = float(np.mean(sign_frame_errors[k]))
                per_sign_mean[k] = mean_val
                overall_frame_errors[k].extend(sign_frame_errors[k])
                
        per_sign_results[sign_name] = {
            'sign_class': sign_cls,
            'is_one_handed': is_one_handed,
            'num_frames': sign_matched_frames,
            'metrics': per_sign_mean
        }
        
    micro_avg = {k: float(np.mean(overall_frame_errors[k])) for k in metric_keys if overall_frame_errors[k]}
    macro_avg = {}
    for k in metric_keys:
        vals = [per_sign_results[s]['metrics'][k] for s in per_sign_results if k in per_sign_results[s]['metrics']]
        if vals:
            macro_avg[k] = float(np.mean(vals))
            
    return {
        'evaluate_folder': evaluate_folder,
        'gt_folder': gt_folder,
        'total_signs_evaluated': total_evaluated_signs,
        'total_frames_evaluated': total_evaluated_frames,
        'micro_average_mm': micro_avg,
        'macro_average_mm': macro_avg,
        'per_sign': per_sign_results
    }


def print_summary_table(summary: Dict[str, Any]) -> None:
    micro = summary['micro_average_mm']
    macro = summary['macro_average_mm']
    
    print("\n" + "=" * 82)
    print(f"               PA-MPVPE EVALUATION REPORT (SIGNAL-4D)")
    print("=" * 82)
    print(f" Folder      : {summary['evaluate_folder']}")
    print(f" Signs/Frames: {summary['total_signs_evaluated']} signs / {summary['total_frames_evaluated']} frames")
    print("-" * 82)
    print(f" {'Metric Name':<38} | {'Micro-Avg (mm)':<18} | {'Macro-Avg (mm)':<18}")
    print("-" * 82)
    
    metric_display_names = [
        ('pa_mpvpe_all', 'PA-MPVPE (All - 10,475 v)'),
        ('pa_mpvpe_no_face', 'PA-MPVPE (Body+Hands, No Face - 5,452 v)'),
        ('pa_mpvpe_body_only', 'PA-MPVPE (Body Only - 3,896 v)'),
        ('pa_mpvpe_rhand', 'PA-MPVPE (Right Hand - 778 v)'),
        ('pa_mpvpe_lhand', 'PA-MPVPE (Left Hand - 778 v)'),
        ('pa_mpvpe_hand', 'PA-MPVPE (Hands Avg)'),
        ('pa_mpvpe_face', 'PA-MPVPE (Face - 5,023 v)'),
        ('pa_mpvpe_upper_body', 'PA-MPVPE (Upper Body - 8,888 v)'),
        ('pa_mpvpe_ub_minus_face', 'PA-MPVPE (Upper Body No Face - 7,279 v)'),
        ('pa_mpvpe_ub_minus_head', 'PA-MPVPE (Upper Body No Head - 3,865 v)'),
        ('mpvpe_all', 'MPVPE (All - Pelvis Aligned)'),
        ('mpvpe_hand', 'MPVPE (Hands - Wrist Aligned)'),
        ('tr_all', 'TR All (Author Centroid)'),
        ('tr_rhand', 'TR Right Hand (Author Centroid)'),
        ('tr_lhand', 'TR Left Hand (Author Centroid)'),
        ('tr_upper_body', 'TR Upper Body (Author Centroid)'),
        ('tr_ub_minus_face', 'TR Upper Body No Face (Author Centroid)'),
        ('tr_ub_minus_head', 'TR Upper Body No Head (Author Centroid)'),
    ]
    
    for key, disp_name in metric_display_names:
        if key in micro:
            m_val = f"{micro[key]:.2f} mm"
            ma_val = f"{macro.get(key, 0.0):.2f} mm"
            print(f" {disp_name:<38} | {m_val:<18} | {ma_val:<18}")
            
    print("=" * 82 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PA-MPVPE on 3D SMPL-X mesh reconstructions")
    parser.add_argument('--evaluate_folder', type=str, required=True,
                        help='Path to reconstruction directory containing sign folders')
    parser.add_argument('--gt_folder', type=str,
                        default='/home/haipd/DexAvatar/data/smplx_gt',
                        help='Path to ground-truth SMPL-X meshes')
    parser.add_argument('--sign_file', type=str,
                        default='/home/haipd/DexAvatar/data/evaluation_from_author/signs.txt',
                        help='Path to signs.txt')
    parser.add_argument('--sign_seg', type=str,
                        default='/home/haipd/DexAvatar/data/evaluation_from_author/segment.json',
                        help='Path to segment.json')
    parser.add_argument('--data_base_dir', type=str,
                        default='/home/haipd/DexAvatar/data/evaluation_from_author/data/data',
                        help='Path to SMPL-X / MANO data directory')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Path to save JSON evaluation report')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Path to save CSV per-sign evaluation report')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress bar and print summary only')
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not osp.exists(args.evaluate_folder):
        print(f"Error: evaluate_folder '{args.evaluate_folder}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    assets = SMPLXAssets(args.data_base_dir)
    
    summary = evaluate_dataset(
        evaluate_folder=args.evaluate_folder,
        gt_folder=args.gt_folder,
        sign_file=args.sign_file,
        sign_seg_file=args.sign_seg,
        assets=assets,
        verbose=not args.quiet
    )
    
    print_summary_table(summary)
    
    if args.output_json:
        os.makedirs(osp.dirname(osp.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON report to: {args.output_json}")
        
    if args.output_csv:
        os.makedirs(osp.dirname(osp.abspath(args.output_csv)), exist_ok=True)
        import csv
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['Sign', 'Class', 'Frames', 'PA-MPVPE All', 'PA-MPVPE NoFace', 'PA-MPVPE BodyOnly', 
                      'PA-MPVPE RHand', 'PA-MPVPE LHand', 'PA-MPVPE Hands', 'PA-MPVPE Face', 
                      'PA-MPVPE UpperBody', 'PA-MPVPE UB-NoFace', 'MPVPE All', 'TR All', 'TR UB-NoFace']
            writer.writerow(header)
            for sign, sdata in summary['per_sign'].items():
                m = sdata['metrics']
                row = [
                    sign,
                    sdata['sign_class'],
                    sdata['num_frames'],
                    f"{m.get('pa_mpvpe_all', 0.0):.4f}",
                    f"{m.get('pa_mpvpe_no_face', 0.0):.4f}",
                    f"{m.get('pa_mpvpe_body_only', 0.0):.4f}",
                    f"{m.get('pa_mpvpe_rhand', 0.0):.4f}",
                    f"{m.get('pa_mpvpe_lhand', 0.0):.4f}",
                    f"{m.get('pa_mpvpe_hand', 0.0):.4f}",
                    f"{m.get('pa_mpvpe_face', 0.0):.4f}",
                    f"{m.get('pa_mpvpe_upper_body', 0.0):.4f}",
                    f"{m.get('pa_mpvpe_ub_minus_face', 0.0):.4f}",
                    f"{m.get('mpvpe_all', 0.0):.4f}",
                    f"{m.get('tr_all', 0.0):.4f}",
                    f"{m.get('tr_ub_minus_face', 0.0):.4f}"
                ]
                writer.writerow(row)
        print(f"Saved CSV report to: {args.output_csv}")


if __name__ == '__main__':
    main()
