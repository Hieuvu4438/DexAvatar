# -*- coding: utf-8 -*-
"""
Method 1: Temporal Sliding Window Optimization
Processes K frames jointly with temporal smoothness losses.
Standalone script — does NOT modify main.py or fit_single_frame.py.
"""

import time
import pickle
import sys
import os
import os.path as osp
import json
import math

import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from test_bposer import load_signbposer
from test_hposer import load_hposer3d
from optimizers import optim_factory
import fitting
from data_parser import create_dataset
from camera import create_camera
from prior import create_prior
from utils import JointMapper
from rewrite_body_model import SMPLX
import smplx as smplx_lib


def temporal_smoothness_loss(pose_seq):
    """
    Compute temporal smoothness losses from a sequence of pose embeddings.
    pose_seq: list of tensors, each (1, D) — decoded poses per frame
    Returns: velocity_loss, acceleration_loss, jerk_loss
    """
    if len(pose_seq) < 2:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    stacked = torch.cat(pose_seq, dim=0)  # (K, D)

    # Velocity: ||pose_t - pose_{t-1}||^2
    vel = stacked[1:] - stacked[:-1]  # (K-1, D)
    vel_loss = torch.sum(vel ** 2)

    if len(pose_seq) < 3:
        return vel_loss, torch.tensor(0.0), torch.tensor(0.0)

    # Acceleration: ||(pose_t - pose_{t-1}) - (pose_{t-1} - pose_{t-2})||^2
    acc = vel[1:] - vel[:-1]  # (K-2, D)
    acc_loss = torch.sum(acc ** 2)

    if len(pose_seq) < 4:
        return vel_loss, acc_loss, torch.tensor(0.0)

    # Jerk: ||delta_acc||^2
    jerk = acc[1:] - acc[:-1]  # (K-3, D)
    jerk_loss = torch.sum(jerk ** 2)

    return vel_loss, acc_loss, jerk_loss


def fit_temporal_window_main(**args):
    output_folder = args.pop('output_folder')
    output_folder = osp.expandvars(output_folder)
    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    result_folder = args.pop('result_folder', 'results')
    result_folder = osp.join(output_folder, result_folder)
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    mesh_folder = args.pop('mesh_folder', 'meshes')
    mesh_folder = osp.join(output_folder, mesh_folder)
    if not osp.exists(mesh_folder):
        os.makedirs(mesh_folder)

    out_img_folder = osp.join(output_folder, 'images')
    if not osp.exists(out_img_folder):
        os.makedirs(out_img_folder)

    # Temporal parameters
    window_size = args.get('temporal_window_size', 15)
    vel_weight = args.get('temporal_velocity_weight', 100.0)
    acc_weight = args.get('temporal_acceleration_weight', 50.0)
    jerk_weight = args.get('temporal_jerk_weight', 10.0)

    float_dtype = args.get('float_dtype', 'float32')
    dtype = torch.float64 if float_dtype == 'float64' else torch.float32

    use_cuda = args.get('use_cuda', True)
    device = torch.device('cuda') if use_cuda and torch.cuda.is_available() else torch.device('cpu')

    # Load sign class mapping
    mapping_path = args['sign_class']
    class_sign = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            tokens = line.strip().split(' ')
            class_sign[tokens[0]] = tokens[1]
    class_sign = dict(sorted(class_sign.items(), key=lambda x: x[0]))

    folder_name = args.get('img_folder').split('/')[-1]
    img_path = args.get('img_folder')
    indp_sign_class = class_sign[folder_name]

    with open(args.get('sign_segment'), 'r', encoding='utf-8') as f:
        segment_data = json.load(f)

    dataset_obj = create_dataset(
        indp_sign_segment=segment_data[folder_name],
        img_path=img_path,
        indp_sign_class=indp_sign_class,
        **args
    )

    # Create body model
    joint_mapper = JointMapper(dataset_obj.get_model2data())
    model_params = dict(
        model_path=args.get('model_folder'),
        joint_mapper=joint_mapper,
        create_global_orient=True,
        create_body_pose=not args.get('use_vposer'),
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=False,
        dtype=dtype,
        **args
    )

    neutral_model = SMPLX(
        model_path='../SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.pkl',
        ext='pkl', use_face_contour=True, flat_hand_mean=True,
        use_pca=False, num_betas=10, num_expression_coeffs=10
    )
    neutral_model = neutral_model.to(device=device)

    # Camera
    focal_length = args.get('focal_length')
    camera = create_camera(focal_length_x=focal_length, focal_length_y=focal_length, dtype=dtype, **args)
    if hasattr(camera, 'rotation'):
        camera.rotation.requires_grad = False
    camera = camera.to(device=device)

    # Priors
    body_pose_prior = create_prior(prior_type=args.get('body_prior_type'), dtype=dtype, **args).to(device)
    angle_prior = create_prior(prior_type='angle', dtype=dtype).to(device)
    shape_prior = create_prior(prior_type=args.get('shape_prior_type', 'l2'), dtype=dtype, **args).to(device)

    left_hand_prior = right_hand_prior = None
    if args.get('use_hands', True):
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args.get('num_pca_comps')
        left_hand_prior = create_prior(prior_type=args.get('left_hand_prior_type'), dtype=dtype, use_left_hand=True, **lhand_args).to(device)
        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args.get('num_pca_comps')
        right_hand_prior = create_prior(prior_type=args.get('right_hand_prior_type'), dtype=dtype, use_right_hand=True, **rhand_args).to(device)

    jaw_prior = expr_prior = None
    if args.get('use_face', True):
        jaw_prior = create_prior(prior_type=args.get('jaw_prior_type'), dtype=dtype, **args).to(device)
        expr_prior = create_prior(prior_type=args.get('expr_prior_type', 'l2'), dtype=dtype, **args).to(device)

    # Load VAE models
    signbposer, _ = load_signbposer(args.get('signbposer_dir'))
    signbposer = signbposer.to(device).eval()

    hposer3d, _ = load_hposer3d(args.get('signhposer_dir'))
    hposer3d = hposer3d.to(device).eval()

    joint_weights = dataset_obj.get_joint_weights().to(device=device, dtype=dtype).unsqueeze_(dim=0)

    # Pre-load all frame data
    all_frames = []
    for idx in range(len(dataset_obj)):
        data = dataset_obj[idx]
        all_frames.append(data)

    print(f"Loaded {len(all_frames)} frames. Window size = {window_size}")

    # Process windows
    stride = max(1, window_size // 2)
    num_windows = max(1, (len(all_frames) - window_size) // stride + 1)

    # Storage for results (average overlapping windows)
    result_accumulator = {}
    result_counts = {}

    for win_start in tqdm(range(0, len(all_frames) - window_size + 1, stride), desc='Windows'):
        win_end = min(win_start + window_size, len(all_frames))
        window_frames = all_frames[win_start:win_end]
        K = len(window_frames)

        # Create per-frame VAE embeddings
        pose_embeddings = [
            torch.zeros(1, 33, dtype=dtype, device=device, requires_grad=True)
            for _ in range(K)
        ]
        lhand_embeddings = [
            torch.zeros(1, 23, dtype=dtype, device=device, requires_grad=True)
            for _ in range(K)
        ]
        rhand_embeddings = [
            torch.zeros(1, 23, dtype=dtype, device=device, requires_grad=True)
            for _ in range(K)
        ]

        # Build loss function (reuse existing SMPLifyLoss)
        loss_kwargs = {k: v for k, v in args.items() if k not in [
            'output_folder', 'result_folder', 'mesh_folder', 'img_folder',
            'sign_class', 'sign_segment', 'temporal_window_size',
            'temporal_velocity_weight', 'temporal_acceleration_weight',
            'temporal_jerk_weight', 'use_temporal_window'
        ]}
        loss = fitting.create_loss(
            loss_type='smplify',
            search_tree=None, pen_distance=None, tri_filtering_module=None,
            body_pose_prior=body_pose_prior, shape_prior=shape_prior,
            expr_prior=expr_prior, angle_prior=angle_prior,
            jaw_prior=jaw_prior, left_hand_prior=left_hand_prior,
            right_hand_prior=right_hand_prior,
            interpenetration=False,
            dtype=dtype,
            **loss_kwargs
        ).to(device)

        # Optimizer: all embeddings jointly
        all_params = pose_embeddings + lhand_embeddings + rhand_embeddings
        optimizer, _ = optim_factory.create_optimizer(all_params, **args)

        # Load per-frame data to device
        frame_data = []
        for fd in window_frames:
            cam_param = torch.from_numpy(fd['cam_param']).to(device=device)
            gt_joints = torch.tensor(fd['keypoints'][:, :, :2], dtype=dtype, device=device)
            joints_conf = torch.tensor(fd['keypoints'][:, :, 2], dtype=dtype, device=device).reshape(1, -1)
            p3DGT_hand = fd['p3DGT_hand'].to(device=device, dtype=dtype) if torch.is_tensor(fd['p3DGT_hand']) else torch.from_numpy(fd['p3DGT_hand']).to(device=device, dtype=dtype)
            smplx_param = fd['smplx_param']
            hand_label = fd['label']
            frame_data.append({
                'cam_param': cam_param, 'gt_joints': gt_joints,
                'joints_conf': joints_conf, 'p3DGT_hand': p3DGT_hand,
                'smplx_param': smplx_param, 'hand_label': hand_label,
                'fn': fd['fn'],
            })

        # Optimization loop
        data_weight = 1000.0 / 720.0  # normalize by typical image height
        for opt_idx in range(args.get('maxiters', 30)):
            optimizer.zero_grad()

            total_loss = torch.tensor(0.0, device=device, dtype=dtype)
            body_pose_seq = []  # for temporal loss

            for fi, fd in enumerate(frame_data):
                # Decode poses from embeddings
                body_pose = signbposer.decode(pose_embeddings[fi], output_type='aa').view(1, -1)

                lhand_pose = hposer3d.decode(lhand_embeddings[fi], output_type='aa').view(1, -1)
                rhand_pose = hposer3d.decode(rhand_embeddings[fi], output_type='aa').view(1, -1)

                body_pose_seq.append(body_pose)

                # Run body model
                model_output = neutral_model(
                    return_verts=False, body_pose=body_pose,
                    left_hand_pose=lhand_pose, right_hand_pose=rhand_pose,
                    return_full_pose=True
                )

                # Per-frame loss (simplified: 2D reprojection + pose prior)
                from assets.mapping_func import get_2dkps_float, get_mapping
                src2inter, dst2inter, _ = get_mapping('coco_wholebody', 'smplx')

                projected = get_2dkps_float(model_output.joints.float()[0], fd['cam_param'].float())[None]
                pred_j = model_output.joints.float()[:, src2inter]
                proj_j = projected[:, src2inter]
                gt_j = fd['gt_joints'][:, dst2inter]
                conf_j = fd['joints_conf'][:, dst2inter]

                import utils as utils_module
                robustifier = utils_module.GMoF(rho=args.get('rho', 100))
                joint_diff = robustifier(gt_j - proj_j)
                weights = (joint_weights[:, dst2inter] * conf_j).unsqueeze(-1)
                frame_loss = torch.sum(weights ** 2 * joint_diff) * data_weight ** 2

                # Pose prior
                pprior = pose_embeddings[fi].pow(2).sum() * args.get('body_pose_prior_weights', [4.78])[0] ** 2

                # Shape prior (use avg shape)
                avg_shape = torch.from_numpy(fd['smplx_param']['betas']).to(device=device, dtype=dtype).unsqueeze(0)

                total_loss = total_loss + frame_loss + pprior

            # Temporal smoothness losses
            vel_loss, acc_loss, jerk_loss = temporal_smoothness_loss(body_pose_seq)
            temporal_loss = vel_weight * vel_loss + acc_weight * acc_loss + jerk_weight * jerk_loss

            total_loss = total_loss + temporal_loss

            total_loss.backward()
            optimizer.step()

            if opt_idx % 10 == 0:
                tqdm.write(f"  Window [{win_start}:{win_end}] Iter {opt_idx}: "
                          f"loss={total_loss.item():.4f}, vel={vel_loss.item():.4f}, "
                          f"acc={acc_loss.item():.4f}, jerk={jerk_loss.item():.4f}")

        # Save results for this window
        for fi, fd in enumerate(frame_data):
            fn = fd['fn']
            body_pose = signbposer.decode(pose_embeddings[fi], output_type='aa').view(1, -1).detach().cpu().numpy()
            lhand_pose = hposer3d.decode(lhand_embeddings[fi], output_type='aa').view(1, -1).detach().cpu().numpy()
            rhand_pose = hposer3d.decode(rhand_embeddings[fi], output_type='aa').view(1, -1).detach().cpu().numpy()

            result = {
                'body_pose': body_pose,
                'left_hand_pose': lhand_pose,
                'right_hand_pose': rhand_pose,
                'betas': fd['smplx_param']['betas'],
                'global_orient': fd['smplx_param']['global_orient'],
                'transl': fd['smplx_param'].get('transl', np.zeros(3)),
                'K': fd['cam_param'].cpu().numpy(),
            }

            if fn not in result_accumulator:
                result_accumulator[fn] = result
                result_counts[fn] = 1
            else:
                # Average with previous window results
                for key in ['body_pose', 'left_hand_pose', 'right_hand_pose']:
                    result_accumulator[fn][key] = (
                        result_accumulator[fn][key] * result_counts[fn] + result[key]
                    ) / (result_counts[fn] + 1)
                result_counts[fn] += 1

    # Save all results
    for fn, result in tqdm(result_accumulator.items(), desc='Saving'):
        result_fn = osp.join(result_folder, f'{fn}.pkl')
        with open(result_fn, 'wb') as f:
            pickle.dump(result, f, protocol=2)

    print(f"Saved {len(result_accumulator)} results to {result_folder}")


if __name__ == "__main__":
    from cmd_parser import parse_config
    args = parse_config()
    fit_temporal_window_main(**args)
