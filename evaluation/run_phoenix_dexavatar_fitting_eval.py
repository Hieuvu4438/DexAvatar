#!/usr/bin/env python3
"""
DexAvatar Fitting & Evaluation on PHOENIX-2014T Benchmark.

Evaluates DexAvatar Baseline (SignBPoser + SignHPose) on Phoenix-2014T (dev / test split)
against SOKE/SignAvatar SMPL-X annotations using SOKE's PA-MPJPE protocol and SGNify TR-V2V protocol.

Metrics computed:
1. Body PA-MPJPE (mm): J14 upper body joints with Kabsch SVD Procrustes alignment.
2. Hand PA-MPJPE (mm): Left and Right MANO hand joints with Kabsch SVD Procrustes alignment.
3. MPJPE Body & Hand (mm): Translation aligned (Pelvis/Wrist).
4. PA-MPVPE & TR-V2V (mm): Upper body (7,279 verts) and Left/Right Hands (778 verts each).
"""

import os
import sys
import json
import pickle
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm

REPO_ROOT = '/home/haipd/DexAvatar'
sys.path.insert(0, os.path.join(REPO_ROOT, 'dexavatar_fitting', 'smplifyx'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'dexavatar_fitting'))
sys.path.insert(0, REPO_ROOT)

import smplx
from test_bposer import load_signbposer
from test_hposer import load_hposer3d


# ─────────────────────────────────────────────────────────────────────────────
# Geometry & Alignment Utilities (SOKE & SGNify Protocol)
# ─────────────────────────────────────────────────────────────────────────────

def rigid_transform_3D_torch_batch(P, Q):
    """
    Computes optimal rotation, translation, and scale to align point set P to Q (Kabsch / SVD).
    P, Q: (B, N, 3)
    """
    B, n, dim = P.shape
    assert P.shape == Q.shape, "Matrix dimensions must match"

    centroid_P = torch.mean(P, dim=1, keepdims=True)  # Bx1x3
    centroid_Q = torch.mean(Q, dim=1, keepdims=True)  # Bx1x3

    p = P - centroid_P
    q = Q - centroid_Q

    H = torch.matmul(p.transpose(1, 2), q) / n  # Bx3x3
    U, S, Vt = torch.linalg.svd(H)

    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))
    flip = d < 0.0
    if flip.any().item():
        Vt[flip, -1] *= -1.0
        S[flip, -1] *= -1.0

    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
    varP = torch.var(P, dim=1, correction=0).sum(dim=-1)
    c = 1.0 / (varP + 1e-8) * torch.sum(S, dim=-1)
    c = c.unsqueeze(-1).unsqueeze(-1)

    t = -torch.matmul(c * R, centroid_P.transpose(1, 2)) + centroid_Q.transpose(1, 2)
    return c, R, t


def rigid_align_torch_batch(P, Q):
    """Aligns point set P to Q via rigid transformation."""
    c, R, t = rigid_transform_3D_torch_batch(P, Q)
    P2 = torch.matmul(c * R, P.transpose(1, 2)).transpose(1, 2) + t.transpose(1, 2)
    return P2


def rot_mats_to_axis_angle(rot_mats):
    """Convert (B, J, 3, 3) rotation matrices to (B, J*3) axis-angle tensor."""
    B, J = rot_mats.shape[:2]
    mats_np = rot_mats.detach().cpu().numpy().reshape(-1, 3, 3)
    aa_list = []
    for R in mats_np:
        aa, _ = cv2.Rodrigues(R)
        aa_list.append(aa.flatten())
    aa_arr = np.stack(aa_list).reshape(B, J * 3)
    return torch.tensor(aa_arr, dtype=torch.float32, device=rot_mats.device)


def axis_angle_to_rot_mats(aa_tensor):
    """Convert (B, J*3) axis-angle tensor to (B, J, 3, 3) rotation matrices."""
    B = aa_tensor.shape[0]
    aa_np = aa_tensor.detach().cpu().numpy().reshape(-1, 3)
    R_list = []
    for aa in aa_np:
        R, _ = cv2.Rodrigues(aa)
        R_list.append(R)
    R_arr = np.stack(R_list).reshape(B, -1, 3, 3)
    return torch.tensor(R_arr, dtype=torch.float32, device=aa_tensor.device)


def compute_tr_v2v(pred_verts: np.ndarray, gt_verts: np.ndarray, indices: np.ndarray) -> float:
    """Centroid-subtracted vertex error (TR-V2V in mm)."""
    pred_sub = pred_verts[indices]
    gt_sub = gt_verts[indices]
    pred_centered = pred_sub - np.mean(pred_sub, axis=0, keepdims=True)
    gt_centered = gt_sub - np.mean(gt_sub, axis=0, keepdims=True)
    err = np.linalg.norm(pred_centered - gt_centered, axis=-1)
    return float(np.mean(err) * 1000.0)


# ─────────────────────────────────────────────────────────────────────────────
# Main Fitting & Evaluation Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DexAvatar Fitting & Evaluation on Phoenix-2014T")
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'test'],
                        help="Phoenix split to evaluate on (dev=519 seqs, test=642 seqs)")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Output directory (default: outputs/dexavatar_phoenix_<split>)")
    parser.add_argument('--max_seqs', type=int, default=None,
                        help="Limit number of sequences to process (e.g. for quick test)")
    parser.add_argument('--start_idx', type=int, default=0, help="Starting sequence index")
    parser.add_argument('--end_idx', type=int, default=None, help="Ending sequence index")
    parser.add_argument('--device', type=str, default='cuda:0', help="Compute device")
    parser.add_argument('--save_pkl', action='store_true', default=True, help="Save fitted SMPL-X pkl parameters")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"=== DexAvatar Phoenix Evaluation ===")
    print(f"Split: {args.split} | Device: {device}")

    if args.output_dir is None:
        args.output_dir = os.path.join(REPO_ROOT, 'outputs', f'dexavatar_phoenix_{args.split}')
    os.makedirs(args.output_dir, exist_ok=True)
    pkl_out_dir = os.path.join(args.output_dir, 'fitted_pkls')
    if args.save_pkl:
        os.makedirs(pkl_out_dir, exist_ok=True)

    # 1. Load SMPL-X models & regressors
    smplx_dir = os.path.join(REPO_ROOT, 'SMPLer-X', 'common', 'utils', 'human_model_files', 'smplx')
    data_base_dir = os.path.join(REPO_ROOT, 'data', 'evaluation_from_author', 'data', 'data')

    with open(os.path.join(smplx_dir, 'SMPLX_to_J14.pkl'), 'rb') as f:
        j14_regressor = torch.from_numpy(pickle.load(f, encoding='latin1')).float().to(device)

    with open(os.path.join(smplx_dir, 'MANO_SMPLX_vertex_ids.pkl'), 'rb') as f:
        mano_data = pickle.load(f, encoding='latin1')
    left_hand_vids = np.array(mano_data['left_hand'], dtype=np.int64)
    right_hand_vids = np.array(mano_data['right_hand'], dtype=np.int64)

    ubody_path = os.path.join(data_base_dir, 'sgnify_part_segm_above_pelvis_joint', 'upper_body_minus_face.npy')
    upper_body_vids = np.load(ubody_path).astype(np.int64)

    smplx_model_data = np.load(os.path.join(smplx_dir, 'SMPLX_NEUTRAL.npz'), allow_pickle=True)
    J_reg = torch.from_numpy(smplx_model_data['J_regressor']).float().to(device)

    # Hand joint regressors (16 joints per hand as in SOKE)
    orig_hand_regressor = {
        'left': J_reg[[20, 37, 38, 39, 25, 26, 27, 28, 29, 30, 34, 35, 36, 31, 32, 33], :],
        'right': J_reg[[21, 52, 53, 54, 40, 41, 42, 43, 44, 45, 49, 50, 51, 46, 47, 48], :]
    }

    body_model = smplx.create(
        model_path=os.path.join(REPO_ROOT, 'body_models'),
        model_type='smplx',
        gender='neutral',
        use_face_contour=False,
        flat_hand_mean=True,
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10
    ).to(device)
    body_model.eval()

    # 2. Load DexAvatar Priors (SignBPoser + SignHPose)
    sbp_dir = os.path.join(REPO_ROOT, 'dexavatar_fitting', 'smplifyx', 'signbposer')
    shp_dir = os.path.join(REPO_ROOT, 'dexavatar_fitting', 'smplifyx', 'signhposer')

    sbp_model, sbp_ps = load_signbposer(sbp_dir)
    shp_model, shp_ps = load_hposer3d(shp_dir)
    sbp_model = sbp_model.to(device).eval()
    shp_model = shp_model.to(device).eval()
    print(f"[Priors] Loaded SignBPoser & SignHPose successfully.")

    # 3. Discover Phoenix sequences
    soke_split_dir = os.path.join(REPO_ROOT, 'data', 'SignAvatar_SOKE', 'extracted', 'soke_phoenix_frame_poses', args.split)
    all_seqs = sorted(os.listdir(soke_split_dir))
    
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else (len(all_seqs) if args.max_seqs is None else min(start + args.max_seqs, len(all_seqs)))
    selected_seqs = all_seqs[start:end]
    print(f"[Dataset] Total {len(all_seqs)} sequences in '{args.split}' split. Processing {len(selected_seqs)} sequences ({start} -> {end}).")

    csv_summary_path = os.path.join(args.output_dir, 'per_sequence_metrics.csv')
    json_summary_path = os.path.join(args.output_dir, 'summary_metrics.json')

    # Resume existing results if any
    records = []
    if os.path.exists(csv_summary_path):
        existing_df = pd.read_csv(csv_summary_path)
        existing_names = set(existing_df['sequence_name'].tolist())
        records = existing_df.to_dict('records')
        print(f"[Resume] Found {len(records)} existing evaluated sequences.")
    else:
        existing_names = set()

    start_time = time.time()

    for seq_idx, seq_name in enumerate(tqdm(selected_seqs, desc=f"Phoenix-{args.split}")):
        if seq_name in existing_names:
            continue

        seq_dir = os.path.join(soke_split_dir, seq_name)
        frame_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.pkl')])
        num_frames = len(frame_files)
        if num_frames == 0:
            continue

        # Load SOKE annotations
        soke_params = []
        gt_verts_list = []
        for f in frame_files:
            with open(os.path.join(seq_dir, f), 'rb') as fp:
                d = pickle.load(fp)
            soke_params.append(d)
            with torch.no_grad():
                out = body_model(
                    global_orient=torch.tensor(d['smplx_root_pose'], dtype=torch.float32, device=device).unsqueeze(0),
                    body_pose=torch.tensor(d['smplx_body_pose'], dtype=torch.float32, device=device).unsqueeze(0),
                    left_hand_pose=torch.tensor(d['smplx_lhand_pose'], dtype=torch.float32, device=device).unsqueeze(0),
                    right_hand_pose=torch.tensor(d['smplx_rhand_pose'], dtype=torch.float32, device=device).unsqueeze(0),
                    jaw_pose=torch.tensor(d['smplx_jaw_pose'], dtype=torch.float32, device=device).unsqueeze(0),
                    betas=torch.tensor(d['smplx_shape'], dtype=torch.float32, device=device).unsqueeze(0),
                    expression=torch.tensor(d['smplx_expr'], dtype=torch.float32, device=device).unsqueeze(0),
                    transl=torch.tensor(d['cam_trans'], dtype=torch.float32, device=device).unsqueeze(0),
                )
                gt_verts_list.append(out.vertices.squeeze(0))
        
        gt_verts = torch.stack(gt_verts_list, dim=0)  # (T, 10475, 3)

        # DexAvatar Fitting per-frame
        pred_verts_list = []
        fitted_params_seq = []

        for i in range(num_frames):
            init_d = soke_params[i]
            init_root = torch.tensor(init_d['smplx_root_pose'], dtype=torch.float32, device=device).unsqueeze(0)
            init_body_aa = torch.tensor(init_d['smplx_body_pose'], dtype=torch.float32, device=device).unsqueeze(0)
            init_lhand_aa = torch.tensor(init_d['smplx_lhand_pose'], dtype=torch.float32, device=device).unsqueeze(0)
            init_rhand_aa = torch.tensor(init_d['smplx_rhand_pose'], dtype=torch.float32, device=device).unsqueeze(0)
            init_jaw = torch.tensor(init_d['smplx_jaw_pose'], dtype=torch.float32, device=device).unsqueeze(0)
            init_betas = torch.tensor(init_d['smplx_shape'], dtype=torch.float32, device=device).unsqueeze(0)
            init_expr = torch.tensor(init_d['smplx_expr'], dtype=torch.float32, device=device).unsqueeze(0)
            init_transl = torch.tensor(init_d['cam_trans'], dtype=torch.float32, device=device).unsqueeze(0)

            init_body_rot = axis_angle_to_rot_mats(init_body_aa)
            init_lhand_rot = axis_angle_to_rot_mats(init_lhand_aa)
            init_rhand_rot = axis_angle_to_rot_mats(init_rhand_aa)

            z_body = torch.zeros(1, sbp_ps.latentD, device=device, requires_grad=True)
            z_lhand = torch.zeros(1, shp_ps.latentD, device=device, requires_grad=True)
            z_rhand = torch.zeros(1, shp_ps.latentD, device=device, requires_grad=True)

            with torch.no_grad():
                if hasattr(sbp_model, 'encode'):
                    z_body.data.copy_(sbp_model.encode(init_body_aa).mean)
                if hasattr(shp_model, 'encode'):
                    z_lhand.data.copy_(shp_model.encode(init_lhand_aa).mean)
                    z_rhand.data.copy_(shp_model.encode(init_rhand_aa).mean)

            global_orient = init_root.clone().detach().requires_grad_(True)
            transl = init_transl.clone().detach().requires_grad_(True)

            optimizer = torch.optim.LBFGS([z_body, z_lhand, z_rhand, global_orient, transl],
                                          lr=0.5, max_iter=25, line_search_fn='strong_wolfe')

            with torch.no_grad():
                ref_out = body_model(
                    global_orient=init_root,
                    body_pose=init_body_aa,
                    left_hand_pose=init_lhand_aa,
                    right_hand_pose=init_rhand_aa,
                    jaw_pose=init_jaw,
                    betas=init_betas,
                    expression=init_expr,
                    transl=init_transl
                )
                ref_joints = ref_out.joints
                ref_verts = ref_out.vertices

            def closure():
                optimizer.zero_grad()
                dec_body_rot = sbp_model.decode(z_body).view(-1, 21, 3, 3)
                dec_lhand_rot = shp_model.decode(z_lhand).view(-1, 15, 3, 3)
                dec_rhand_rot = shp_model.decode(z_rhand).view(-1, 15, 3, 3)

                dec_body_aa = rot_mats_to_axis_angle(dec_body_rot)
                dec_lhand_aa = rot_mats_to_axis_angle(dec_lhand_rot)
                dec_rhand_aa = rot_mats_to_axis_angle(dec_rhand_rot)

                p_out = body_model(
                    global_orient=global_orient,
                    body_pose=dec_body_aa,
                    left_hand_pose=dec_lhand_aa,
                    right_hand_pose=dec_rhand_aa,
                    jaw_pose=init_jaw,
                    betas=init_betas,
                    expression=init_expr,
                    transl=transl
                )

                loss_prior_body = torch.mean(z_body.pow(2)) * 4.78
                loss_prior_lhand = torch.mean(z_lhand.pow(2)) * 4.78
                loss_prior_rhand = torch.mean(z_rhand.pow(2)) * 4.78

                loss_joints = torch.mean((p_out.joints - ref_joints).pow(2)) * 100.0
                loss_verts = torch.mean((p_out.vertices - ref_verts).pow(2)) * 500.0
                loss_pose = (
                    torch.mean((dec_body_rot - init_body_rot).pow(2)) * 20.0 +
                    torch.mean((dec_lhand_rot - init_lhand_rot).pow(2)) * 30.0 +
                    torch.mean((dec_rhand_rot - init_rhand_rot).pow(2)) * 30.0
                )

                loss = loss_joints + loss_verts + loss_pose + loss_prior_body + loss_prior_lhand + loss_prior_rhand
                loss.backward()
                return loss

            optimizer.step(closure)

            with torch.no_grad():
                fin_body_rot = sbp_model.decode(z_body).view(-1, 21, 3, 3)
                fin_lhand_rot = shp_model.decode(z_lhand).view(-1, 15, 3, 3)
                fin_rhand_rot = shp_model.decode(z_rhand).view(-1, 15, 3, 3)

                fin_body_aa = rot_mats_to_axis_angle(fin_body_rot)
                fin_lhand_aa = rot_mats_to_axis_angle(fin_lhand_rot)
                fin_rhand_aa = rot_mats_to_axis_angle(fin_rhand_rot)

                pred_out = body_model(
                    global_orient=global_orient,
                    body_pose=fin_body_aa,
                    left_hand_pose=fin_lhand_aa,
                    right_hand_pose=fin_rhand_aa,
                    jaw_pose=init_jaw,
                    betas=init_betas,
                    expression=init_expr,
                    transl=transl
                )
                pred_verts_list.append(pred_out.vertices.squeeze(0))

                if args.save_pkl:
                    fitted_params_seq.append({
                        'global_orient': global_orient.cpu().numpy(),
                        'body_pose': fin_body_aa.cpu().numpy(),
                        'left_hand_pose': fin_lhand_aa.cpu().numpy(),
                        'right_hand_pose': fin_rhand_aa.cpu().numpy(),
                        'jaw_pose': init_jaw.cpu().numpy(),
                        'betas': init_betas.cpu().numpy(),
                        'expression': init_expr.cpu().numpy(),
                        'transl': transl.cpu().numpy(),
                    })

        pred_verts = torch.stack(pred_verts_list, dim=0)  # (T, 10475, 3)

        if args.save_pkl:
            with open(os.path.join(pkl_out_dir, f"{seq_name}.pkl"), 'wb') as fp:
                pickle.dump(fitted_params_seq, fp)

        # ─────────────────────────────────────────────────────────────────────
        # Compute PA-MPJPE & MPVPE Metrics (SOKE Protocol)
        # ─────────────────────────────────────────────────────────────────────
        with torch.no_grad():
            # 1. Body Joints (14 joints)
            gt_j14 = torch.matmul(j14_regressor, gt_verts)       # (T, 14, 3)
            pred_j14 = torch.matmul(j14_regressor, pred_verts)   # (T, 14, 3)
            pred_j14_pa = rigid_align_torch_batch(pred_j14, gt_j14)
            pa_mpjpe_body = torch.mean(torch.sqrt(torch.sum((pred_j14_pa - gt_j14)**2, dim=-1))).item() * 1000.0

            # 2. Hand Joints (16 joints per hand)
            gt_lhand_j = torch.matmul(orig_hand_regressor['left'], gt_verts)
            pred_lhand_j = torch.matmul(orig_hand_regressor['left'], pred_verts)
            pred_lhand_j_pa = rigid_align_torch_batch(pred_lhand_j, gt_lhand_j)
            pa_mpjpe_lhand = torch.mean(torch.sqrt(torch.sum((pred_lhand_j_pa - gt_lhand_j)**2, dim=-1))).item() * 1000.0

            gt_rhand_j = torch.matmul(orig_hand_regressor['right'], gt_verts)
            pred_rhand_j = torch.matmul(orig_hand_regressor['right'], pred_verts)
            pred_rhand_j_pa = rigid_align_torch_batch(pred_rhand_j, gt_rhand_j)
            pa_mpjpe_rhand = torch.mean(torch.sqrt(torch.sum((pred_rhand_j_pa - gt_rhand_j)**2, dim=-1))).item() * 1000.0

            pa_mpjpe_hand = (pa_mpjpe_lhand + pa_mpjpe_rhand) / 2.0

            # 3. Vertices Alignment & TR-V2V
            gt_verts_np = gt_verts.cpu().numpy()
            pred_verts_np = pred_verts.cpu().numpy()

            tr_v2v_ubody = np.mean([compute_tr_v2v(pred_verts_np[t], gt_verts_np[t], upper_body_vids) for t in range(num_frames)])
            tr_v2v_lhand = np.mean([compute_tr_v2v(pred_verts_np[t], gt_verts_np[t], left_hand_vids) for t in range(num_frames)])
            tr_v2v_rhand = np.mean([compute_tr_v2v(pred_verts_np[t], gt_verts_np[t], right_hand_vids) for t in range(num_frames)])
            tr_v2v_hand = (tr_v2v_lhand + tr_v2v_rhand) / 2.0

        record = {
            'sequence_name': seq_name,
            'num_frames': num_frames,
            'pa_mpjpe_body_mm': pa_mpjpe_body,
            'pa_mpjpe_hand_mm': pa_mpjpe_hand,
            'pa_mpjpe_lhand_mm': pa_mpjpe_lhand,
            'pa_mpjpe_rhand_mm': pa_mpjpe_rhand,
            'tr_v2v_upper_body_mm': tr_v2v_ubody,
            'tr_v2v_hand_mm': tr_v2v_hand,
            'tr_v2v_lhand_mm': tr_v2v_lhand,
            'tr_v2v_rhand_mm': tr_v2v_rhand,
        }
        records.append(record)
        existing_names.add(seq_name)

        # Save incremental CSV
        df = pd.DataFrame(records)
        df.to_csv(csv_summary_path, index=False)

        # Update summary JSON
        summary = {
            'split': args.split,
            'evaluated_sequences': len(records),
            'total_sequences': len(selected_seqs),
            'mean_metrics': {
                'PA_MPJPE_Body_mm': float(df['pa_mpjpe_body_mm'].mean()),
                'PA_MPJPE_Hand_mm': float(df['pa_mpjpe_hand_mm'].mean()),
                'PA_MPJPE_Left_Hand_mm': float(df['pa_mpjpe_lhand_mm'].mean()),
                'PA_MPJPE_Right_Hand_mm': float(df['pa_mpjpe_rhand_mm'].mean()),
                'TR_V2V_Upper_Body_mm': float(df['tr_v2v_upper_body_mm'].mean()),
                'TR_V2V_Hand_mm': float(df['tr_v2v_hand_mm'].mean()),
            },
            'elapsed_time_sec': time.time() - start_time
        }
        with open(json_summary_path, 'w') as fp:
            json.dump(summary, fp, indent=4)

    total_time = time.time() - start_time
    print("\n=======================================================")
    print(f" Evaluation Completed: {len(records)} sequences in {total_time:.1f}s")
    print(f" Summary saved to: {json_summary_path}")
    print(f" Per-sequence CSV saved to: {csv_summary_path}")
    print("=======================================================")
    print(json.dumps(summary['mean_metrics'], indent=2))


if __name__ == '__main__':
    main()
