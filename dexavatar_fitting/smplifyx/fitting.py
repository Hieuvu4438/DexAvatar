# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import time

import numpy as np

import torch
import torch.nn as nn
import body_constants as body_constants
from mesh_viewer import MeshViewer
import utils
from assets.mapping_func import get_2dkps_float, get_mapping

src2inter, dst2inter, inter_name = get_mapping('coco_wholebody', 'smplx')

import pickle

regressor_mat = pickle.load(
    open('../SMPLer-X/common/utils/human_model_files/smplx/SMPLX_to_J14.pkl',
         'rb'), encoding='latin1')


@torch.no_grad()
def guess_init(model,
               joints_2d,
               edge_idxs,
               focal_length=5000,
               pose_embedding=None,
               signbposer=None,
               use_signbposer=True,
               dtype=torch.float32,
               model_type='smpl',
               **kwargs):
    ''' Initializes the camera translation vector

        Parameters
        ----------
        model: nn.Module
            The PyTorch module of the body
        joints_2d: torch.tensor 1xJx2
            The 2D tensor of the joints
        edge_idxs: list of lists
            A list of pairs, each of which represents a limb used to estimate
            the camera translation
        focal_length: float, optional (default = 5000)
            The focal length of the camera
        pose_embedding: torch.tensor 1x32
            The tensor that contains the embedding of V-Poser that is used to
            generate the pose of the model
        dtype: torch.dtype, optional (torch.float32)
            The floating point type used
        vposer: nn.Module, optional (None)
            The PyTorch module that implements the V-Poser decoder
        Returns
        -------
        init_t: torch.tensor 1x3, dtype = torch.float32
            The vector with the estimated camera location

    '''

    body_pose = signbposer.decode(
        pose_embedding, output_type='aa').view(1, -1) if use_signbposer else None
    if use_signbposer and model_type == 'smpl':
        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                 dtype=body_pose.dtype,
                                 device=body_pose.device)
        body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    output = model(body_pose=body_pose, return_verts=False,
                   return_full_pose=False)
    joints_3d = output.joints
    joints_2d = joints_2d.to(device=joints_3d.device)

    diff3d = []
    diff2d = []
    for edge in edge_idxs:
        diff3d.append(joints_3d[:, edge[0]] - joints_3d[:, edge[1]])
        diff2d.append(joints_2d[:, edge[0]] - joints_2d[:, edge[1]])

    diff3d = torch.stack(diff3d, dim=1)
    diff2d = torch.stack(diff2d, dim=1)

    length_2d = diff2d.pow(2).sum(dim=-1).sqrt()
    length_3d = diff3d.pow(2).sum(dim=-1).sqrt()

    height2d = length_2d.mean(dim=1)
    height3d = length_3d.mean(dim=1)

    est_d = focal_length * (height3d / height2d)

    # just set the z value
    batch_size = joints_3d.shape[0]
    x_coord = torch.zeros([batch_size], device=joints_3d.device,
                          dtype=dtype)
    y_coord = x_coord.clone()
    init_t = torch.stack([x_coord, y_coord, est_d], dim=1)
    return init_t


class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_signbposer=True, pose_embedding=None, signbposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_signbposer
        prev_loss = None
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    print('ftol break')
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                print('gtol break')
                break

            if self.visualize and n % self.summary_steps == 0:
                body_pose = signbposer.decode(
                    pose_embedding, output_type='aa').view(
                        1, -1) if use_signbposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                model_output = body_model(
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()

        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_joints=None, loss=None, p3DGT_hand=None, psmplx_bodyGT=None, psmplx_lhandGT=None, psmplx_rhandGT=None,
                               joints_conf=None, indp_sign_class=None, hand_label=None, joints_temp=None,
                               joint_weights=None,
                               hposer3d=None,
                               lhand_embedding3d=None,
                               rhand_embedding3d=None,
                               use_hposer3d=None,
                               return_verts=True, return_full_pose=False,
                               use_signbposer=False, signbposer=None,
                               pose_embedding=None,
                               create_graph=False,
                               use_motionbert_prior=False, motionbert_prior=None,
                               use_phd_prior=False, phd_prior=None,
                               use_dposerx_body=False, dposerx_body_prior=None,
                               use_vqvae_hand=False, hposer3d_vqvae=None,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_signbposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            # Decode body pose: handle all prior types
            if use_signbposer:
                body_pose = signbposer.decode(
                    pose_embedding, output_type='aa').view(1, -1)
            elif use_motionbert_prior or use_phd_prior or use_dposerx_body:
                # Direct optimization: pose_embedding IS body_pose
                body_pose = pose_embedding.view(1, -1)
            else:
                body_pose = None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            if use_hposer3d:

                if indp_sign_class != "0":
                    lhand_pose = hposer3d.decode(lhand_embedding3d, output_type='aa').view(1, -1)
                    rhand_pose = hposer3d.decode(rhand_embedding3d, output_type='aa').view(1, -1)
                    body_model_output = body_model(return_verts=return_verts,
                                                body_pose=body_pose, right_hand_pose=rhand_pose, left_hand_pose=lhand_pose,
                                                return_full_pose=return_full_pose)
                else:
                    if hand_label == 'right_hand':
                        rhand_pose = hposer3d.decode(rhand_embedding3d, output_type='aa').view(1, -1)
                        body_model_output = body_model(return_verts=return_verts,
                                                    body_pose=body_pose, right_hand_pose=rhand_pose,
                                                    return_full_pose=return_full_pose)
                    
                    elif hand_label == 'left_hand':
                        lhand_pose = hposer3d.decode(lhand_embedding3d, output_type='aa').view(1, -1)
                        body_model_output = body_model(return_verts=return_verts,
                                                    body_pose=body_pose, left_hand_pose=lhand_pose,
                                                    return_full_pose=return_full_pose)   
            
            else:
            
                body_model_output = body_model(return_verts=return_verts,
                                            body_pose=body_pose,
                                            return_full_pose=return_full_pose)
                
            total_loss = loss(body_model_output, camera=camera,
                              gt_joints=gt_joints, p3DGT_hand=p3DGT_hand, psmplx_bodyGT=psmplx_bodyGT, psmplx_lhandGT=psmplx_lhandGT, psmplx_rhandGT=psmplx_rhandGT,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf, indp_sign_class=indp_sign_class, hand_label=hand_label, joints_temp=joints_temp,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_hposer3d=use_hposer3d,
                              hposer3d=hposer3d,
                              lhand_embedding3d=lhand_embedding3d,
                              rhand_embedding3d=rhand_embedding3d,
                              use_signbposer=use_signbposer,
                              signbposer=signbposer,
                              use_motionbert_prior=use_motionbert_prior,
                              motionbert_prior=motionbert_prior,
                              use_phd_prior=use_phd_prior,
                              phd_prior=phd_prior,
                              use_dposerx_body=use_dposerx_body,
                              dposerx_body_prior=dposerx_body_prior,
                              use_vqvae_hand=use_vqvae_hand,
                              hposer3d_vqvae=hposer3d_vqvae,
                              **kwargs)

            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            return total_loss

        return fitting_func


def create_loss(loss_type='smplify', **kwargs):
    if loss_type == 'smplify':
        return SMPLifyLoss(**kwargs)
    elif loss_type == 'camera_init':
        return SMPLifyCameraInitLoss(**kwargs)
    else:
        raise ValueError('Unknown loss type: {}'.format(loss_type))


class SMPLifyLoss(nn.Module):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 expr_prior=None,
                 angle_prior=None,
                 jaw_prior=None,
                 use_joints_conf=True,
                 use_face=True, use_hands=True,
                 left_hand_prior=None, right_hand_prior=None,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 data_3d_weight=0.0,
                 data_init_core_weight=0.0,
                 data_init_noncore_weight=0.0,
                 data_init_lhand_weight=0.0,
                 data_init_rhand_weight=0.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 body_biomechanics_loss_weights=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0, jaw_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 hand_2d_loss_weights=0.0,
                 hand2d_rho=50,
                 # Approach D: Absolute depth + wrist alignment
                 absolute_depth_weight=0.0,
                 wrist_alignment_weight=0.0,
                 # Approach E: Uncertainty hand
                 uncertainty_min_weight=0.1,
                 reduction='sum',
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance

        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior

        self.min_constraints = torch.tensor(body_constants.BOF_body[0,:], dtype=dtype).reshape(6,3).to('cuda')
        self.max_constraints = torch.tensor(body_constants.BOF_body[1,:], dtype=dtype).reshape(6,3).to('cuda')

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('data_3d_weight',
                             torch.tensor(data_3d_weight, dtype=dtype))
        self.register_buffer('data_init_core_weight',
                             torch.tensor(data_init_core_weight, dtype=dtype))
        self.register_buffer('data_init_noncore_weight',
                             torch.tensor(data_init_noncore_weight, dtype=dtype))
        self.register_buffer('data_init_lhand_weight',
                             torch.tensor(data_init_lhand_weight, dtype=dtype))
        self.register_buffer('data_init_rhand_weight',
                             torch.tensor(data_init_rhand_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        self.register_buffer('body_biomechanics_loss_weights',
                             torch.tensor(body_biomechanics_loss_weights, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                                 torch.tensor(hand_prior_weight, dtype=dtype))
        if self.use_face:
            self.register_buffer('expr_prior_weight',
                                 torch.tensor(expr_prior_weight, dtype=dtype))
            self.register_buffer('jaw_prior_weight',
                                 torch.tensor(jaw_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))

        # Method 2: 2D hand keypoint supervision
        self.register_buffer('hand_2d_loss_weights',
                             torch.tensor(hand_2d_loss_weights, dtype=dtype))
        self.hand2d_robustifier = utils.GMoF(rho=hand2d_rho)

        # Approach D: Absolute depth + wrist alignment
        self.register_buffer('absolute_depth_weight',
                             torch.tensor(absolute_depth_weight, dtype=dtype))
        self.register_buffer('wrist_alignment_weight',
                             torch.tensor(wrist_alignment_weight, dtype=dtype))
        self.absolute_depth_robustifier = utils.GMoF(rho=100)

        # Approach E: Uncertainty-aware hand fusion
        self.uncertainty_min_weight = uncertainty_min_weight

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints, p3DGT_hand, psmplx_bodyGT, joints_conf, psmplx_lhandGT, psmplx_rhandGT,
                body_model_faces, joint_weights, lhand_embedding3d=None, rhand_embedding3d=None, use_hposer3d=False, hposer3d=None,
                use_signbposer=False, signbposer=None, pose_embedding=None, indp_sign_class=None, hand_label=None, joints_temp=None,
                use_motionbert_prior=False, motionbert_prior=None,
                use_phd_prior=False, phd_prior=None,
                use_dposerx_body=False, dposerx_body_prior=None,
                use_vqvae_hand=False, hposer3d_vqvae=None,
                **kwargs):
        pred_vert = body_model_output.vertices.float() # take out the vertices from the body model output
        new_replace_vert = torch.matmul(torch.from_numpy(regressor_mat).cuda().float(), pred_vert[0]) # converting vertices to joints
        pred_joints = body_model_output.joints.float() # take out the joints from the body model output
        index_list = [10] # index of the wrist joint in the SMPLX model
        pred_joints[0][18] = (pred_joints[0][18] + new_replace_vert[index_list]) / 2. # replace the wrist joint with the new joint calculated from the vertices
        projected_joints = get_2dkps_float(body_model_output.joints.float()[0], camera.float()) # project the 3D joints to 2D using the camera parameters from SMPLer_X
        projected_joints = projected_joints[None] # add a batch dimension
        joints_to_temp = body_model_output.body_pose.reshape(1,21,3)

        p3dgt_joints = torch.zeros((gt_joints.shape[0], gt_joints.shape[1], 3)).cuda() # create a tensor to hold the ground truth 3D joints
        if indp_sign_class != "0":
            p3dgt_joints[0][91:] = p3DGT_hand # fill the tensor with the ground truth 3D hand joints
        else:
            if hand_label == "left_hand":
                p3dgt_joints[0][91:112] = p3DGT_hand
            elif hand_label == "right_hand":
                p3dgt_joints[0][112:] = p3DGT_hand

        pred_joints = pred_joints[:, src2inter] # selecting only the SMPL-X joints that have a name in common with COCO-WholeBody 
        projected_joints = projected_joints[:, src2inter] # selecting only the SMPL-X joints that have a name in common with COCO-WholeBody
        gt_joints = gt_joints[:, dst2inter] # selecting only the COCO detections that match those same SMPL-X joint names
        p3dgt_joints = p3dgt_joints[:, dst2inter] # selecting only the COCO detections that match those same SMPL-X joint names

        def normalize_points_torch(points):
                mean = torch.mean(points, dim=0)
                std = torch.std(points, dim=0)
                normalized_points = (points - mean) / std
                return normalized_points
        

        if indp_sign_class != "0":

            hand_index = [i for i in range(12, 42)] + [i for i in range(53, 63)] # the indices 12–41 (inclusive) correspond to all of your left-hand keypoints, and indices 53–62 to your right-hand keypoints.

            pred_hand_3d = pred_joints[:, hand_index, 2:3] # take out the depth of 3D hand joints from the predicted joints
            gt_hand_3d = p3dgt_joints[:, hand_index, 2:3] # take out the depth of 3D hand joints from the ground truth joints

            relative_pred_hand_3d = (pred_hand_3d - pred_hand_3d[:, 0:1]) # calculate wrist-relative 3D hands for predicted hands
            relative_gt_hand_3d = (gt_hand_3d - gt_hand_3d[:, 0:1]) # calculate wrist-relative 3D hands for ground truth hands
            relative_pred_hand_3d = normalize_points_torch(relative_pred_hand_3d[0])[None] # normalize the wrist-relative 3D hands for predicted hands
            relative_gt_hand_3d = normalize_points_torch(relative_gt_hand_3d[0])[None] # normalize the wrist-relative 3D hands for ground truth hands
        
        else:
            if hand_label == 'right_hand':
                hand_index = [i for i in range(53, 63)]

                pred_hand_3d = pred_joints[:, hand_index, 2:3]
                gt_hand_3d = p3dgt_joints[:, hand_index, 2:3]

                relative_pred_hand_3d = (pred_hand_3d - pred_hand_3d[:, 0:1]) # calculate wrist-relative 3D hands for predicted hands
                relative_gt_hand_3d = (gt_hand_3d - gt_hand_3d[:, 0:1]) # calculate wrist-relative 3D hands for ground truth hands
                relative_pred_hand_3d = normalize_points_torch(relative_pred_hand_3d[0])[None] # normalize the wrist-relative 3D hands for predicted hands
                relative_gt_hand_3d = normalize_points_torch(relative_gt_hand_3d[0])[None]

            elif hand_label == "left_hand":
                hand_index = [i for i in range(12, 42)]
                pred_hand_3d = pred_joints[:, hand_index, 2:3] # take out the depth of 3D hand joints from the predicted joints
                gt_hand_3d = p3dgt_joints[:, hand_index, 2:3] # take out the depth of 3D hand joints from the ground truth joints

                relative_pred_hand_3d = (pred_hand_3d - pred_hand_3d[:, 0:1]) # calculate wrist-relative 3D hands for predicted hands
                relative_gt_hand_3d = (gt_hand_3d - gt_hand_3d[:, 0:1]) # calculate wrist-relative 3D hands for ground truth hands
                relative_pred_hand_3d = normalize_points_torch(relative_pred_hand_3d[0])[None] # normalize the wrist-relative 3D hands for predicted hands
                relative_gt_hand_3d = normalize_points_torch(relative_gt_hand_3d[0])[None]


        temp_loss = torch.sum(self.robustifier(joints_to_temp - joints_temp)) * 2000

        joint_hand_3d_diff = self.robustifier(((relative_pred_hand_3d) - (relative_gt_hand_3d))) # Loss between hand 3D hand joints generated by fitting and from Hamer

        # ===== METHOD 2: 2D Hand Keypoint Supervision =====
        hand_2d_keypoints = kwargs.get('hand_2d_keypoints', None)
        use_hand2d = kwargs.get('use_hand2d_supervision', False)
        hand_2d_loss = torch.tensor(0.0, device=pred_joints.device)

        if use_hand2d and hand_2d_keypoints is not None and self.hand_2d_loss_weights.item() > 0:
            # GT 2D hand keypoints: (42, 2) = [left_21, right_21] in MANO order
            # Need to reorder to match intersection space (COCO-WholeBody order)
            # MANO: wrist, thumb(3), tip, index(3), tip, middle(3), tip, ring(3), tip, pinky(3), tip
            # COCO: root, thumb(3), tip, index(3), tip, middle(3), tip, ring(3), tip, pinky(3), tip
            gt_h2d_raw = hand_2d_keypoints  # (42, 2)
            gt_h2d_reordered = torch.zeros_like(gt_h2d_raw)
            # Left hand (first 21): MANO idx 0-20 → COCO idx 0-20
            #   MANO[0]=wrist → COCO[0]=root ✓
            #   MANO[1-4]=thumb → COCO[1-4]=thumb ✓
            #   MANO[5-8]=index → COCO[5-8]=index ✓
            #   ... (same order, no reordering needed for left hand)
            gt_h2d_reordered[0:21] = gt_h2d_raw[0:21]
            # Right hand (next 21): MANO idx 22-41 → COCO idx 21-41
            #   MANO[22]=wrist → COCO[21]=root ✓
            #   MANO[23-25]=thumb1-3 → COCO[22-24]=thumb1-3 ✓
            #   MANO[26]=thumb_tip → COCO[25]=thumb_tip ✓
            #   MANO[27-29]=index1-3 → COCO[26-28]=index1-3 ✓
            #   ... (same order)
            gt_h2d_reordered[21:42] = gt_h2d_raw[21:42]

            # Get projected hand joints at intersection indices
            pred_hand_2d = projected_joints[:, hand_index, :2]  # (1, N_hand, 2)
            gt_hand_2d = gt_h2d_reordered.unsqueeze(0).to(pred_hand_2d.device)  # (1, 42, 2)

            # Trim to match sizes (both should be 42 joints)
            n_hand = min(pred_hand_2d.shape[1], gt_hand_2d.shape[1])
            hand_2d_diff = self.hand2d_robustifier(
                pred_hand_2d[:, :n_hand, :] - gt_hand_2d[:, :n_hand, :]
            )
            hand_2d_loss = torch.sum(hand_2d_diff) * self.hand_2d_loss_weights

        # Calculate the weights for each joints
        joint_weights[:, 11:23] = 0

        if self.use_joints_conf:
            weights = (joint_weights[:, dst2inter] * joints_conf[:, dst2inter]).unsqueeze(dim=-1)
        else:
            weights = torch.ones_like(gt_joints)[:,:,0].cuda().unsqueeze(dim=-1)

        
        loss_bio = 0.0

        core_body = utils.axis_angle_to_euler_XYZ(joints_to_temp[:,15:,:])

        loss_bio += torch.mean(torch.clamp(core_body - self.max_constraints, min=0))
        loss_bio += torch.mean(torch.clamp(self.min_constraints - core_body, min=0))


        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        joint_diff = self.robustifier(gt_joints - projected_joints)
        # Mutltiplying the joint_diff with the data_weight to bring it from pixel space to the normalized space
        joint_loss = (torch.sum(weights ** 2 * joint_diff) *
                      self.data_weight ** 2) + (torch.sum(joint_hand_3d_diff) * 1 ** 2) * self.data_3d_weight
        # Calculate the loss from the Pose prior
        if use_signbposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)

            pprior_loss += self.data_init_core_weight * torch.abs(
                signbposer.decode(pose_embedding, output_type='aa').view(1, -1)[:, 0:11*3] - psmplx_bodyGT[:,0:11*3]).sum() + \
                self.data_init_noncore_weight * torch.abs(signbposer.decode(pose_embedding, output_type='aa').view(1, -1)[:, 11*3:] - psmplx_bodyGT[:, 11*3:]).sum()

        # ---- MotionBERT Prior (NEW) ----
        elif use_motionbert_prior and motionbert_prior is not None:
            # Direct body pose optimization: pose_embedding IS body_pose (63-dim)
            body_pose_direct = pose_embedding  # (1, 63)

            # MotionBERT prior loss
            pprior_loss = motionbert_prior.prior_loss(body_pose_direct, smplerx_init=psmplx_bodyGT)

            # Init prior: L1 vs SMPLer-X init (same structure as SignBPoser)
            pprior_loss += self.data_init_core_weight * torch.abs(
                body_pose_direct[:, 0:11*3] - psmplx_bodyGT[:, 0:11*3]).sum()
            pprior_loss += self.data_init_noncore_weight * torch.abs(
                body_pose_direct[:, 11*3:] - psmplx_bodyGT[:, 11*3:]).sum()

        # ---- PHD Diffusion Prior (NEW) ----
        elif use_phd_prior and phd_prior is not None:
            # Direct body pose optimization: pose_embedding IS body_pose (63-dim)
            body_pose_direct = pose_embedding  # (1, 63)

            # PHD score-based prior loss
            pprior_loss = phd_prior.prior_loss(body_pose_direct)

            # Init prior: L1 vs SMPLer-X init
            pprior_loss += self.data_init_core_weight * torch.abs(
                body_pose_direct[:, 0:11*3] - psmplx_bodyGT[:, 0:11*3]).sum()
            pprior_loss += self.data_init_noncore_weight * torch.abs(
                body_pose_direct[:, 11*3:] - psmplx_bodyGT[:, 11*3:]).sum()

        # ---- DPoser-X Body Prior (NEW, additive) ----
        elif use_dposerx_body and dposerx_body_prior is not None:
            # Direct body pose optimization: pose_embedding IS body_pose (63-dim)
            body_pose_direct = pose_embedding  # (1, 63)

            # DPoser-X score-based prior loss
            dposerx_t = None
            if kwargs.get('dposerx_timestep_strategy', 'random') == 'fixed':
                dposerx_t = torch.full((batch_size,),
                                        float(kwargs.get('dposerx_fixed_timestep', 50))
                                        / max(dposerx_body_prior.sde.N - 1, 1),
                                        device=device)
            pprior_loss = dposerx_body_prior.prior_loss(
                body_pose_direct, condition=None, t=dposerx_t)

            # Init prior: L1 vs SMPLer-X init (same structure as PHD branch)
            pprior_loss += self.data_init_core_weight * torch.abs(
                body_pose_direct[:, 0:11*3] - psmplx_bodyGT[:, 0:11*3]).sum()
            pprior_loss += self.data_init_noncore_weight * torch.abs(
                body_pose_direct[:, 11*3:] - psmplx_bodyGT[:, 11*3:]).sum()

        # pose embedding loss + core 11 joints rotation loss + Left over rotation loss (this only happens for the first stage of fitting because the \
        # rest of the data_init_prior_weight is set to 0)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2
            


        left_hand_prior_loss, right_hand_prior_loss, hand_prior_3dloss = 0.0, 0.0, 0.0
        
        if use_hposer3d:
        
            if indp_sign_class != "0":
                lhand_prior_3dloss = (lhand_embedding3d.pow(2).sum() * self.hand_prior_weight ** 2) + \
                    torch.abs(hposer3d.decode(lhand_embedding3d, output_type='aa').view(1, -1) - psmplx_lhandGT).sum() * self.data_init_lhand_weight
                rhand_prior_3dloss = (rhand_embedding3d.pow(2).sum() * self.hand_prior_weight ** 2) + \
                    torch.abs(hposer3d.decode(rhand_embedding3d, output_type='aa').view(1, -1) - psmplx_rhandGT).sum() * self.data_init_rhand_weight
                
                lhand_prior_smooth = torch.sum(self.robustifier(hposer3d.decode(lhand_embedding3d, output_type='aa').view(1, -1) - psmplx_lhandGT)) * self.data_init_lhand_weight
                rhand_prior_smooth = torch.sum(self.robustifier(hposer3d.decode(rhand_embedding3d, output_type='aa').view(1, -1) - psmplx_rhandGT)) * self.data_init_rhand_weight
                
                hand_prior_3dloss = lhand_prior_3dloss + rhand_prior_3dloss + lhand_prior_smooth + rhand_prior_smooth
                #hand_prior_3dloss = lhand_prior_3dloss + rhand_prior_3dloss

            else:
                if hand_label == 'right_hand':
                    
                    rhand_prior_3dloss = (rhand_embedding3d.pow(2).sum() * self.hand_prior_weight ** 2) + \
                    torch.abs(hposer3d.decode(rhand_embedding3d, output_type='aa').view(1, -1) - psmplx_rhandGT).sum() * self.data_init_rhand_weight

                    if self.use_hands and self.left_hand_prior is not None:
                        left_hand_prior_loss = torch.sum(
                            self.left_hand_prior(
                                body_model_output.left_hand_pose)) * \
                            self.hand_prior_weight ** 2
                        
                    rhand_prior_smooth = torch.sum(self.robustifier(hposer3d.decode(rhand_embedding3d, output_type='aa').view(1, -1) - psmplx_rhandGT)) * self.data_init_rhand_weight

                    hand_prior_3dloss = rhand_prior_3dloss + left_hand_prior_loss + rhand_prior_smooth

                    #hand_prior_3dloss = rhand_prior_3dloss + left_hand_prior_loss
                
                elif hand_label == "left_hand":

                    lhand_prior_3dloss = (lhand_embedding3d.pow(2).sum() * self.hand_prior_weight ** 2) + \
                torch.abs(hposer3d.decode(lhand_embedding3d, output_type='aa').view(1, -1) - psmplx_lhandGT).sum() * self.data_init_lhand_weight
                    
                    if self.use_hands and self.right_hand_prior is not None:
                        right_hand_prior_loss = torch.sum(
                            self.right_hand_prior(
                                body_model_output.right_hand_pose)) * \
                            self.hand_prior_weight ** 2

                    lhand_prior_smooth = torch.sum(self.robustifier(hposer3d.decode(lhand_embedding3d, output_type='aa').view(1, -1) - psmplx_lhandGT)) * self.data_init_lhand_weight
                    
                    hand_prior_3dloss = lhand_prior_3dloss + right_hand_prior_loss + lhand_prior_smooth
                    #hand_prior_3dloss = lhand_prior_3dloss + right_hand_prior_loss
                    

        else:
            
            if self.use_hands and self.left_hand_prior is not None:
                left_hand_prior_loss = torch.sum(
                    self.left_hand_prior(
                        body_model_output.left_hand_pose)) * \
                    self.hand_prior_weight ** 2

            if self.use_hands and self.right_hand_prior is not None:
                right_hand_prior_loss = torch.sum(
                    self.right_hand_prior(
                        body_model_output.right_hand_pose)) * \
                    self.hand_prior_weight ** 2
            
            hand_prior_3dloss = left_hand_prior_loss + right_hand_prior_loss

        # ---- SOKE VQVAE Hand Prior Loss (NEW, additive) ----
        if use_vqvae_hand and hposer3d_vqvae is not None and not use_hposer3d:
            # Reconstruct through VQVAE and penalize reconstruction error.
            # This regularizes the decoded hand poses to lie on the VQVAE manifold.
            vqvae_weight = kwargs.get('vqvae_recon_loss_weight', 1.0)
            if lhand_embedding3d is not None:
                decoded_lhand = hposer3d_vqvae.decode_aa(lhand_embedding3d)  # (B, 1, 15, 3)
                decoded_lhand_flat = decoded_lhand.view(1, 45)
                # Pad to T_min and run VQVAE encode+decode for reconstruction loss
                lhand_recon, lhand_commit, _ = hposer3d_vqvae(decoded_lhand_flat.unsqueeze(1).expand(1, 8, 45))
                lhand_center = lhand_recon[:, lhand_recon.shape[1]//2, :]
                vqvae_lhand_loss = (decoded_lhand_flat - lhand_center).abs().mean() + 0.02 * lhand_commit
            else:
                vqvae_lhand_loss = 0.0

            if rhand_embedding3d is not None:
                decoded_rhand = hposer3d_vqvae.decode_aa(rhand_embedding3d)
                decoded_rhand_flat = decoded_rhand.view(1, 45)
                rhand_recon, rhand_commit, _ = hposer3d_vqvae(decoded_rhand_flat.unsqueeze(1).expand(1, 8, 45))
                rhand_center = rhand_recon[:, rhand_recon.shape[1]//2, :]
                vqvae_rhand_loss = (decoded_rhand_flat - rhand_center).abs().mean() + 0.02 * rhand_commit
            else:
                vqvae_rhand_loss = 0.0

            hand_prior_3dloss = vqvae_weight * (vqvae_lhand_loss + vqvae_rhand_loss)
            # Also add the standard L2+init regularizer (same as use_hposer3d branch).
            if lhand_embedding3d is not None:
                hand_prior_3dloss += (lhand_embedding3d.pow(2).sum() * self.hand_prior_weight ** 2) + \
                    torch.abs(hposer3d_vqvae.decode_aa(lhand_embedding3d).view(1, -1) - psmplx_lhandGT).sum() * self.data_init_lhand_weight
            if rhand_embedding3d is not None:
                hand_prior_3dloss += (rhand_embedding3d.pow(2).sum() * self.hand_prior_weight ** 2) + \
                    torch.abs(hposer3d_vqvae.decode_aa(rhand_embedding3d).view(1, -1) - psmplx_rhandGT).sum() * self.data_init_rhand_weight

        shape_loss = torch.sum(self.shape_prior(
            body_model_output.betas)) * self.shape_weight ** 2
        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight # angle prior loss



        expression_loss = 0.0
        jaw_prior_loss = 0.0
        if self.use_face:
            expression_loss = torch.sum(self.expr_prior(
                body_model_output.expression)) * \
                self.expr_prior_weight ** 2

            if hasattr(self, 'jaw_prior'):
                jaw_prior_loss = torch.sum(
                    self.jaw_prior(
                        body_model_output.jaw_pose.mul(
                            self.jaw_prior_weight)))

        pen_loss = 0.0
        # Calculate the loss due to interpenetration
        if (self.interpenetration and self.coll_loss_weight.item() > 0):
            batch_size = projected_joints.shape[0]
            triangles = torch.index_select(
                body_model_output.vertices, 1,
                body_model_faces).view(batch_size, -1, 3, 3)

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)

            # Remove unwanted collisions
            if self.tri_filtering_module is not None:
                collision_idxs = self.tri_filtering_module(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(
                    self.coll_loss_weight *
                    self.pen_distance(triangles, collision_idxs))

        # ===== METHOD 5: Sign-Specific Biomechanics =====
        sign_bio_loss = torch.tensor(0.0, device=pred_joints.device)
        use_sign_bio = kwargs.get('use_sign_biomechanics', False)
        if use_sign_bio:
            hand_contact_weight = kwargs.get('hand_contact_weight', 0.0)
            hand_body_contact_weight = kwargs.get('hand_body_contact_weight', 0.0)

            # Get wrist and finger tip positions from SMPL-X output
            # SMPL-X joint indices: 20=L_wrist, 21=R_wrist
            # Hand fingertips in joint space: left_thumb=25, left_index=26, etc.
            all_joints = body_model_output.joints.float()[0]  # (J, 3)

            # Hand-hand contact loss (for two-handed signs)
            if hand_contact_weight > 0 and indp_sign_class != "0":
                # Left wrist (joint 20) vs Right wrist (joint 21)
                l_wrist = all_joints[20]
                r_wrist = all_joints[21]
                wrist_dist = torch.norm(l_wrist - r_wrist)
                # Encourage hands to be close (contact) for two-handed signs
                # Use ReLU to only penalize when distance > threshold
                contact_threshold = 0.3  # meters
                hand_contact_loss = torch.relu(wrist_dist - contact_threshold) ** 2
                sign_bio_loss = sign_bio_loss + hand_contact_weight * hand_contact_loss

            # Hand-body contact loss (for signs that touch the body)
            if hand_body_contact_weight > 0:
                # Body contact zones (approximate with SMPL-X joints):
                # Chest: joints 3 (spine1), 6 (spine2), 9 (spine3)
                # Face: joint 15 (head)
                # Shoulders: joints 16 (L_shoulder), 17 (R_shoulder)
                body_contact_zones = {
                    'chest': [3, 6, 9],
                    'face': [15],
                    'shoulder_l': [16],
                    'shoulder_r': [17],
                }

                # Get hand positions
                l_hand_joints = all_joints[25:46]  # left hand joints
                r_hand_joints = all_joints[46:67]  # right hand joints

                # For each contact zone, check if hands are near
                for zone_name, zone_indices in body_contact_zones.items():
                    zone_center = all_joints[zone_indices].mean(dim=0)

                    # Left hand proximity
                    l_dist = torch.norm(l_hand_joints.mean(dim=0) - zone_center)
                    l_contact = torch.relu(0.15 - l_dist) ** 2  # penalize if too close without reason

                    # Right hand proximity
                    r_dist = torch.norm(r_hand_joints.mean(dim=0) - zone_center)
                    r_contact = torch.relu(0.15 - r_dist) ** 2

                    sign_bio_loss = sign_bio_loss + hand_body_contact_weight * (l_contact + r_contact)

            # Learned finger articulation prior (sign-specific)
            finger_prior_weight = kwargs.get('finger_prior_weight', 0.0)
            if finger_prior_weight > 0:
                # Penalize extreme finger poses (fingers shouldn't bend too much)
                # Left hand pose: joints 25-40 (15 joints × 3 DoF)
                # Right hand pose: joints 40-55 (15 joints × 3 DoF)
                if hasattr(body_model_output, 'left_hand_pose'):
                    lhand_pose = body_model_output.left_hand_pose.float()
                    rhand_pose = body_model_output.right_hand_pose.float()
                    # Penalize large rotations (beyond natural range)
                    finger_loss = (torch.sum(torch.relu(torch.abs(lhand_pose) - 1.5)) +
                                   torch.sum(torch.relu(torch.abs(rhand_pose) - 1.5)))
                    sign_bio_loss = sign_bio_loss + finger_prior_weight * finger_loss

        # ===== APPROACH D: Absolute Depth Loss =====
        absolute_depth_loss = torch.tensor(0.0, device=pred_joints.device)
        if self.absolute_depth_weight.item() > 0 and p3DGT_hand is not None:
            # Get predicted wrist 3D positions from body model
            # Wrist indices in SMPL-X joints: left=20, right=21
            pred_wrist_3d = pred_joints[:, [20, 21], :]  # (1, 2, 3) in intersection space
            # GT wrist 3D from HaMeR (p3DGT_hand first 3 coords of wrist = index 0 per hand)
            if indp_sign_class != "0":
                gt_lwrist_3d = p3DGT_hand[0:1, :]   # left wrist (21,3) -> first row
                gt_rwrist_3d = p3DGT_hand[21:22, :]  # right wrist
                gt_wrists = torch.cat([gt_lwrist_3d, gt_rwrist_3d], dim=0)  # (2, 3)
                abs_err = self.absolute_depth_robustifier(
                    (pred_wrist_3d.squeeze(0) - gt_wrists).norm(dim=-1))
                absolute_depth_loss = abs_err.sum()
            elif hand_label == 'left_hand':
                gt_lwrist_3d = p3DGT_hand[0:1, :]
                pred_lwrist = pred_wrist_3d[:, 0:1, :]  # left only
                abs_err = self.absolute_depth_robustifier(
                    (pred_lwrist.squeeze(0) - gt_lwrist_3d).norm(dim=-1))
                absolute_depth_loss = abs_err.sum()
            elif hand_label == 'right_hand':
                gt_rwrist_3d = p3DGT_hand[0:1, :]
                pred_rwrist = pred_wrist_3d[:, 1:2, :]  # right only
                abs_err = self.absolute_depth_robustifier(
                    (pred_rwrist.squeeze(0) - gt_rwrist_3d).norm(dim=-1))
                absolute_depth_loss = abs_err.sum()

        # ===== APPROACH D: Wrist Alignment Loss =====
        wrist_alignment_loss = torch.tensor(0.0, device=pred_joints.device)
        if self.wrist_alignment_weight.item() > 0 and psmplx_bodyGT is not None:
            # Encourage wrist position consistency: wrist should not drift far from
            # where the kinematic chain (body_pose) would place it
            # Use L1 between current body_pose and SMPLer-X init as proxy
            current_body_pose = body_model_output.body_pose.float()
            init_body_pose = psmplx_bodyGT.float()
            # Focus on arm joints: shoulders(16,17), elbows(18,19), wrists(20,21)
            # In SMPL-X body_pose: joint indices 16-21 correspond to arms
            arm_indices = list(range(16*3, 22*3))  # 48:66
            if current_body_pose.numel() >= 66:
                wrist_err = torch.abs(current_body_pose[0, arm_indices] -
                                      init_body_pose[0, arm_indices])
                wrist_alignment_loss = self.robustifier(wrist_err).sum()

        # ===== APPROACH E: Uncertainty-Aware Hand Weighting =====
        # Modify hand_3d_diff and hand_prior_3dloss by Sapiens wrist confidence
        uncertainty_hand_scale = 1.0
        if kwargs.get('use_uncertainty_hand', False) and joints_conf is not None:
            # joints_conf shape: (1, 133) — indices 9,10 are left/right wrist in Sapiens
            left_wrist_conf = joints_conf[0, 9].item()   # Sapiens left wrist
            right_wrist_conf = joints_conf[0, 10].item()  # Sapiens right wrist
            if hand_label == 'left_hand':
                avg_conf = left_wrist_conf
            elif hand_label == 'right_hand':
                avg_conf = right_wrist_conf
            else:
                avg_conf = (left_wrist_conf + right_wrist_conf) / 2.0
            # Scale: min_weight when conf=0, 1.0 when conf=1
            uncertainty_hand_scale = max(self.uncertainty_min_weight, avg_conf)

        # Scale hand losses by uncertainty
        joint_hand_3d_diff = joint_hand_3d_diff * uncertainty_hand_scale
        hand_prior_3dloss = hand_prior_3dloss * uncertainty_hand_scale

        if use_hposer3d:
            total_loss = (joint_loss + pprior_loss + shape_loss + loss_bio * self.body_biomechanics_loss_weights +
                        angle_prior_loss + pen_loss + temp_loss +
                        jaw_prior_loss + expression_loss + hand_prior_3dloss +
                        hand_2d_loss + sign_bio_loss +
                        self.absolute_depth_weight * absolute_depth_loss +
                        self.wrist_alignment_weight * wrist_alignment_loss)

        else:
            total_loss = (joint_loss + pprior_loss + shape_loss +
                        angle_prior_loss + pen_loss +
                        jaw_prior_loss + expression_loss + hand_prior_3dloss +
                        hand_2d_loss + sign_bio_loss +
                        self.absolute_depth_weight * absolute_depth_loss +
                        self.wrist_alignment_weight * wrist_alignment_loss)
        
        return total_loss



class SMPLifyCameraInitLoss(nn.Module):

    def __init__(self, init_joints_idxs, trans_estimation=None,
                 reduction='sum',
                 data_weight=1.0,
                 depth_loss_weight=1e2, dtype=torch.float32,
                 **kwargs):
        super(SMPLifyCameraInitLoss, self).__init__()
        self.dtype = dtype

        if trans_estimation is not None:
            self.register_buffer(
                'trans_estimation',
                utils.to_tensor(trans_estimation, dtype=dtype))
        else:
            self.trans_estimation = trans_estimation

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer(
            'init_joints_idxs',
            utils.to_tensor(init_joints_idxs, dtype=torch.long))
        self.register_buffer('depth_loss_weight',
                             torch.tensor(depth_loss_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                weight_tensor = torch.tensor(loss_weight_dict[key],
                                             dtype=weight_tensor.dtype,
                                             device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, camera, gt_joints,
                **kwargs):

        projected_joints = camera(body_model_output.joints)

        joint_error = torch.pow(
            torch.index_select(gt_joints, 1, self.init_joints_idxs) -
            torch.index_select(projected_joints, 1, self.init_joints_idxs),
            2)
        joint_loss = torch.sum(joint_error) * self.data_weight ** 2

        depth_loss = 0.0
        if (self.depth_loss_weight.item() > 0 and self.trans_estimation is not
                None):
            depth_loss = self.depth_loss_weight ** 2 * torch.sum((
                camera.translation[:, 2] - self.trans_estimation[:, 2]).pow(2))

        return joint_loss + depth_loss
