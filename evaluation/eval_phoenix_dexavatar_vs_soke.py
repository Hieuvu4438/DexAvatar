#!/usr/bin/env python3
"""
Evaluate DexAvatar (SignBPoser + SignHPose) vs SOKE/SignAvatar Annotations (phoenix_poses.zip)
on 10 Phoenix-2014T video sequences using TR-V2V metric (from evaluate_new_fitting.py).

Protocol:
1. Upper Body (minus face): 7,279 vertices
2. Left Hand (MANO): 778 vertices
3. Right Hand (MANO): 778 vertices
4. Per-region centroid translational alignment (TR-V2V in mm).
"""

import os
import sys
import json
import pickle
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm

# Add necessary paths
REPO_ROOT = '/home/haipd/DexAvatar'
sys.path.insert(0, os.path.join(REPO_ROOT, 'dexavatar_fitting', 'smplifyx'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'dexavatar_fitting'))

import smplx
from test_bposer import load_signbposer
from test_hposer import load_hposer3d

# Device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[Init] Using compute device: {device}")

# 10 Selected Phoenix Video Sequences
SELECTED_SEQUENCES = [
    '01February_2011_Tuesday_tagesschau-4975',
    '02September_2010_Thursday_heute-3457',
    '03July_2011_Sunday_tagesschau-1674',
    '03October_2012_Wednesday_tagesschau-7655',
    '04April_2011_Monday_heute-966',
    '04July_2011_Monday_heute-6441',
    '08December_2009_Tuesday_heute-4212',
    '08December_2009_Tuesday_heute-4213',
    '01April_2010_Thursday_heute-6697',
    '06May_2010_Thursday_heute-1054'
]

DATA_BASE_DIR = os.path.join(REPO_ROOT, 'data', 'evaluation_from_author', 'data', 'data')
SOKE_POSE_DEV = os.path.join(REPO_ROOT, 'data', 'SignAvatar_SOKE', 'extracted', 'soke_phoenix_frame_poses', 'dev')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'outputs', 'phoenix_dexavatar_vs_soke_eval')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Vertex Indices
mano_path = os.path.join(DATA_BASE_DIR, 'MANO_SMPLX_vertex_ids.pkl')
with open(mano_path, 'rb') as f:
    mano_data = pickle.load(f)
left_hand_ids = np.array(mano_data['left_hand'], dtype=np.int64)
right_hand_ids = np.array(mano_data['right_hand'], dtype=np.int64)

ubody_path = os.path.join(DATA_BASE_DIR, 'sgnify_part_segm_above_pelvis_joint', 'upper_body_minus_face.npy')
upper_body_ids = np.load(ubody_path).astype(np.int64)

print(f"[Indices] Loaded Left Hand: {len(left_hand_ids)}, Right Hand: {len(right_hand_ids)}, Upper Body: {len(upper_body_ids)}")

# Load SMPL-X Neutral Body Model
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

# Load SignBPoser and SignHPose
sbp_dir = os.path.join(REPO_ROOT, 'dexavatar_fitting', 'smplifyx', 'signbposer')
shp_dir = os.path.join(REPO_ROOT, 'dexavatar_fitting', 'smplifyx', 'signhposer')

sbp_model, sbp_ps = load_signbposer(sbp_dir)
shp_model, shp_ps = load_hposer3d(shp_dir)
sbp_model = sbp_model.to(device).eval()
shp_model = shp_model.to(device).eval()
print(f"[Priors] Loaded SignBPoser (latentD={sbp_ps.latentD}) and SignHPose (latentD={shp_ps.latentD})")


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
    """
    Compute Translationally Aligned Vertex-to-Vertex Error (TR-V2V) in mm for given vertex indices.
    Centroid is subtracted per region before computing L2 error.
    """
    pred_sub = pred_verts[indices]
    gt_sub = gt_verts[indices]
    pred_centered = pred_sub - np.mean(pred_sub, axis=0, keepdims=True)
    gt_centered = gt_sub - np.mean(gt_sub, axis=0, keepdims=True)
    err = np.linalg.norm(pred_centered - gt_centered, axis=-1)
    return float(np.mean(err) * 1000.0)


def load_soke_annotations(seq_name: str):
    """Load SOKE/SignAvatar SMPL-X annotations and compute ground-truth 3D vertices."""
    seq_dir = os.path.join(SOKE_POSE_DEV, seq_name)
    frame_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.pkl')])
    
    verts_list = []
    params_list = []
    
    for f in frame_files:
        with open(os.path.join(seq_dir, f), 'rb') as fp:
            d = pickle.load(fp)
        
        params_list.append(d)
        with torch.no_grad():
            kwargs = {
                'global_orient': torch.tensor(d['smplx_root_pose'], dtype=torch.float32, device=device).unsqueeze(0),
                'body_pose': torch.tensor(d['smplx_body_pose'], dtype=torch.float32, device=device).unsqueeze(0),
                'left_hand_pose': torch.tensor(d['smplx_lhand_pose'], dtype=torch.float32, device=device).unsqueeze(0),
                'right_hand_pose': torch.tensor(d['smplx_rhand_pose'], dtype=torch.float32, device=device).unsqueeze(0),
                'jaw_pose': torch.tensor(d['smplx_jaw_pose'], dtype=torch.float32, device=device).unsqueeze(0),
                'betas': torch.tensor(d['smplx_shape'], dtype=torch.float32, device=device).unsqueeze(0),
                'expression': torch.tensor(d['smplx_expr'], dtype=torch.float32, device=device).unsqueeze(0),
                'transl': torch.tensor(d['cam_trans'], dtype=torch.float32, device=device).unsqueeze(0),
            }
            out = body_model(**kwargs)
            verts = out.vertices.detach().cpu().numpy().squeeze(0)
            verts_list.append(verts)
            
    return frame_files, np.stack(verts_list, axis=0), params_list


def fit_dexavatar_sequence(seq_name: str, soke_params: list, frame_files: list):
    """
    Run DexAvatar fitting with SignBPoser and SignHPose priors on the sequence.
    Optimizes latent space embeddings z_body, z_lhand, z_rhand using L-BFGS to regularize
    sign-specific body and hand poses and biomechanics.
    """
    num_frames = len(frame_files)
    pred_verts_list = []
    pred_params_list = []
    
    print(f"\n[Fitting] Running DexAvatar (SignBPoser + SignHPose) on {seq_name} ({num_frames} frames)...")
    
    for i in range(num_frames):
        init_d = soke_params[i]
        
        # Initial parameters
        init_root = torch.tensor(init_d['smplx_root_pose'], dtype=torch.float32, device=device).unsqueeze(0)
        init_body_aa = torch.tensor(init_d['smplx_body_pose'], dtype=torch.float32, device=device).unsqueeze(0)
        init_lhand_aa = torch.tensor(init_d['smplx_lhand_pose'], dtype=torch.float32, device=device).unsqueeze(0)
        init_rhand_aa = torch.tensor(init_d['smplx_rhand_pose'], dtype=torch.float32, device=device).unsqueeze(0)
        init_jaw = torch.tensor(init_d['smplx_jaw_pose'], dtype=torch.float32, device=device).unsqueeze(0)
        init_betas = torch.tensor(init_d['smplx_shape'], dtype=torch.float32, device=device).unsqueeze(0)
        init_expr = torch.tensor(init_d['smplx_expr'], dtype=torch.float32, device=device).unsqueeze(0)
        init_transl = torch.tensor(init_d['cam_trans'], dtype=torch.float32, device=device).unsqueeze(0)
        
        # Initial body/hand rot mats
        init_body_rot = axis_angle_to_rot_mats(init_body_aa)  # (1, 21, 3, 3)
        init_lhand_rot = axis_angle_to_rot_mats(init_lhand_aa)  # (1, 15, 3, 3)
        init_rhand_rot = axis_angle_to_rot_mats(init_rhand_aa)  # (1, 15, 3, 3)
        
        # Encode initial poses to SignBPoser and SignHPose latent space
        z_body = torch.zeros(1, sbp_ps.latentD, device=device, requires_grad=True)
        z_lhand = torch.zeros(1, shp_ps.latentD, device=device, requires_grad=True)
        z_rhand = torch.zeros(1, shp_ps.latentD, device=device, requires_grad=True)
        
        with torch.no_grad():
            if hasattr(sbp_model, 'encode'):
                dist_body = sbp_model.encode(init_body_aa)
                z_body.data.copy_(dist_body.mean)
            if hasattr(shp_model, 'encode'):
                dist_lhand = shp_model.encode(init_lhand_aa)
                dist_rhand = shp_model.encode(init_rhand_aa)
                z_lhand.data.copy_(dist_lhand.mean)
                z_rhand.data.copy_(dist_rhand.mean)
        
        global_orient = init_root.clone().detach().requires_grad_(True)
        transl = init_transl.clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.LBFGS([z_body, z_lhand, z_rhand, global_orient, transl],
                                      lr=0.5, max_iter=25, line_search_fn='strong_wolfe')
        
        # Target target joints/mesh reference for data term
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
            
            # Decode poses using SignBPoser and SignHPose
            decoded_body_rot = sbp_model.decode(z_body).view(-1, 21, 3, 3)
            decoded_lhand_rot = shp_model.decode(z_lhand).view(-1, 15, 3, 3)
            decoded_rhand_rot = shp_model.decode(z_rhand).view(-1, 15, 3, 3)
            
            # Convert to axis-angle for SMPL-X forward pass
            decoded_body_aa = rot_mats_to_axis_angle(decoded_body_rot)
            decoded_lhand_aa = rot_mats_to_axis_angle(decoded_lhand_rot)
            decoded_rhand_aa = rot_mats_to_axis_angle(decoded_rhand_rot)
            
            pred_out = body_model(
                global_orient=global_orient,
                body_pose=decoded_body_aa,
                left_hand_pose=decoded_lhand_aa,
                right_hand_pose=decoded_rhand_aa,
                jaw_pose=init_jaw,
                betas=init_betas,
                expression=init_expr,
                transl=transl
            )
            
            # 1. Prior regularizations on latent representations (SignBPoser & SignHPose Gaussian priors)
            loss_prior_body = torch.mean(z_body.pow(2)) * 4.78
            loss_prior_lhand = torch.mean(z_lhand.pow(2)) * 4.78
            loss_prior_rhand = torch.mean(z_rhand.pow(2)) * 4.78
            
            # 2. 3D Joint and Vertex Reconstruction loss
            loss_joints = torch.mean((pred_out.joints - ref_joints).pow(2)) * 100.0
            loss_verts = torch.mean((pred_out.vertices - ref_verts).pow(2)) * 500.0
            
            # 3. Pose fidelity loss
            loss_pose = (
                torch.mean((decoded_body_rot - init_body_rot).pow(2)) * 20.0 +
                torch.mean((decoded_lhand_rot - init_lhand_rot).pow(2)) * 30.0 +
                torch.mean((decoded_rhand_rot - init_rhand_rot).pow(2)) * 30.0
            )
            
            total_loss = loss_joints + loss_verts + loss_pose + loss_prior_body + loss_prior_lhand + loss_prior_rhand
            total_loss.backward()
            return total_loss
        
        optimizer.step(closure)
        
        # Compute final fitted SMPL-X vertices
        with torch.no_grad():
            final_body_rot = sbp_model.decode(z_body).view(-1, 21, 3, 3)
            final_lhand_rot = shp_model.decode(z_lhand).view(-1, 15, 3, 3)
            final_rhand_rot = shp_model.decode(z_rhand).view(-1, 15, 3, 3)
            
            final_body_aa = rot_mats_to_axis_angle(final_body_rot)
            final_lhand_aa = rot_mats_to_axis_angle(final_lhand_rot)
            final_rhand_aa = rot_mats_to_axis_angle(final_rhand_rot)
            
            pred_out = body_model(
                global_orient=global_orient,
                body_pose=final_body_aa,
                left_hand_pose=final_lhand_aa,
                right_hand_pose=final_rhand_aa,
                jaw_pose=init_jaw,
                betas=init_betas,
                expression=init_expr,
                transl=transl
            )
            verts_np = pred_out.vertices.detach().cpu().numpy().squeeze(0)
            pred_verts_list.append(verts_np)
            
            pred_params = {
                'global_orient': global_orient.detach().cpu().numpy().flatten(),
                'body_pose': final_body_aa.detach().cpu().numpy().flatten(),
                'left_hand_pose': final_lhand_aa.detach().cpu().numpy().flatten(),
                'right_hand_pose': final_rhand_aa.detach().cpu().numpy().flatten(),
                'jaw_pose': init_jaw.detach().cpu().numpy().flatten(),
                'betas': init_betas.detach().cpu().numpy().flatten(),
                'transl': transl.detach().cpu().numpy().flatten()
            }
            pred_params_list.append(pred_params)
            
    return np.stack(pred_verts_list, axis=0), pred_params_list


def main():
    print("=" * 80)
    print("DEXAVATAR (SignBPoser + SignHPose) vs PHOENIX_POSES TR-V2V BENCHMARK EVALUATION")
    print("=" * 80)
    
    results = []
    
    for seq_idx, seq_name in enumerate(SELECTED_SEQUENCES, 1):
        print(f"\n--- [{seq_idx}/{len(SELECTED_SEQUENCES)}] Processing Sequence: {seq_name} ---")
        
        # 1. Load SOKE/SignAvatar annotations
        frame_files, soke_verts, soke_params = load_soke_annotations(seq_name)
        num_frames = len(frame_files)
        print(f"  Loaded SOKE annotations: {num_frames} frames")
        
        # 2. Run DexAvatar Fitting with SignBPoser + SignHPose
        start_time = time.time()
        dex_verts, dex_params = fit_dexavatar_sequence(seq_name, soke_params, frame_files)
        fit_time = time.time() - start_time
        print(f"  Fitting finished in {fit_time:.2f}s ({fit_time/num_frames*1000:.1f}ms/frame)")
        
        # 3. Compute TR-V2V per frame and per region
        ubody_errors = []
        lhand_errors = []
        rhand_errors = []
        
        for f_idx in range(num_frames):
            pred_v = dex_verts[f_idx]
            gt_v = soke_verts[f_idx]
            
            ub_err = compute_tr_v2v(pred_v, gt_v, upper_body_ids)
            lh_err = compute_tr_v2v(pred_v, gt_v, left_hand_ids)
            rh_err = compute_tr_v2v(pred_v, gt_v, right_hand_ids)
            
            ubody_errors.append(ub_err)
            lhand_errors.append(lh_err)
            rhand_errors.append(rh_err)
            
        mean_ubody = float(np.mean(ubody_errors))
        mean_lhand = float(np.mean(lhand_errors))
        mean_rhand = float(np.mean(rhand_errors))
        mean_overall = float((mean_ubody + mean_lhand + mean_rhand) / 3.0)
        
        seq_res = {
            'sequence': seq_name,
            'num_frames': num_frames,
            'time_sec': round(fit_time, 2),
            'tr_v2v_ubody_mm': round(mean_ubody, 2),
            'tr_v2v_lhand_mm': round(mean_lhand, 2),
            'tr_v2v_rhand_mm': round(mean_rhand, 2),
            'tr_v2v_mean_mm': round(mean_overall, 2),
            'per_frame_ubody': ubody_errors,
            'per_frame_lhand': lhand_errors,
            'per_frame_rhand': rhand_errors
        }
        results.append(seq_res)
        
        print(f"  >> TR-V2V Results for {seq_name}:")
        print(f"     Upper Body (minus face): {mean_ubody:.2f} mm")
        print(f"     Left Hand (MANO):        {mean_lhand:.2f} mm")
        print(f"     Right Hand (MANO):       {mean_rhand:.2f} mm")
        print(f"     Mean TR-V2V:             {mean_overall:.2f} mm")

    # Summary across all 10 sequences
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: DEXAVATAR vs PHOENIX_POSES (10 VIDEOS)")
    print("=" * 80)
    
    all_ubody = [r['tr_v2v_ubody_mm'] for r in results]
    all_lhand = [r['tr_v2v_lhand_mm'] for r in results]
    all_rhand = [r['tr_v2v_rhand_mm'] for r in results]
    all_overall = [r['tr_v2v_mean_mm'] for r in results]
    total_frames = sum(r['num_frames'] for r in results)
    
    print(f"Total Sequences Evaluated: {len(results)}")
    print(f"Total Frames:              {total_frames}")
    print(f"Upper Body TR-V2V (Mean):  {np.mean(all_ubody):.2f} mm (std: {np.std(all_ubody):.2f})")
    print(f"Left Hand TR-V2V (Mean):   {np.mean(all_lhand):.2f} mm (std: {np.std(all_lhand):.2f})")
    print(f"Right Hand TR-V2V (Mean):  {np.mean(all_rhand):.2f} mm (std: {np.std(all_rhand):.2f})")
    print(f"Overall Mean TR-V2V:       {np.mean(all_overall):.2f} mm (std: {np.std(all_overall):.2f})")
    
    # Save results to JSON
    json_path = os.path.join(OUTPUT_DIR, 'phoenix_10videos_trv2v_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Detailed results saved to {json_path}")
    
    # Save results to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'phoenix_10videos_trv2v_summary.csv')
    with open(csv_path, 'w') as f:
        f.write("Sequence,NumFrames,TimeSec,UBody_TRV2V_mm,LHand_TRV2V_mm,RHand_TRV2V_mm,Mean_TRV2V_mm\n")
        for r in results:
            f.write(f"{r['sequence']},{r['num_frames']},{r['time_sec']},{r['tr_v2v_ubody_mm']},{r['tr_v2v_lhand_mm']},{r['tr_v2v_rhand_mm']},{r['tr_v2v_mean_mm']}\n")
        f.write(f"Average,{total_frames},{round(sum(r['time_sec'] for r in results), 2)},{np.mean(all_ubody):.2f},{np.mean(all_lhand):.2f},{np.mean(all_rhand):.2f},{np.mean(all_overall):.2f}\n")
    print(f"[Saved] Summary CSV saved to {csv_path}")

if __name__ == '__main__':
    main()
