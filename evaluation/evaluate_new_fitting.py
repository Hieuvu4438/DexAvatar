from pickletools import string1
from typing import Dict, Tuple, Any, Optional
import time
import os
import os.path as osp
from pathlib import Path
#import scenepic as sp
import json
import pickle
import argparse
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
import re
#import open3d as o3d

from scipy.spatial.transform import Rotation as R

Array = np.ndarray

#from calculate_v2v_error import (
#    compute_similarity_transform,
#)

import matplotlib.pyplot as plt

LEFT_WRIST_INDEX = 20
RIGHT_WRIST_INDEX = 21


class Timer(object):
    def __init__(self, name='', sync=False, verbose=False):
        super(Timer, self).__init__()
        self.elapsed = []
        self.name = name
        self.sync = sync
        self.verbose = verbose

    def __enter__(self):
        if self.sync:
            torch.cuda.synchronize()
        self.start = time.perf_counter()

    def print(self):
        logger.info(f'[{self.name}]: {np.mean(self.elapsed):.3f}')

    def __exit__(self, type, value, traceback):
        if self.sync:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start
        self.elapsed.append(elapsed)
        if self.verbose:
            logger.info(
                f'[{self.name}]: {elapsed:.3f}, {np.mean(self.elapsed):.3f}')


SOMA_PATH = '/home/kaustubh/Desktop/data/smplx_gt'
SOMA_ALT = '/is/cluster/scratch/soma/.tmp/blender_temp/V48_02_ASL_T03/OC_05_G_03_real_000_synt_100/ASL_Unlabeled'
PAOLAS_WORK_FOLDER = '/home/kaustubh/Desktop/PIXIE'
SGNIFY_SV_PATH = osp.join(PAOLAS_WORK_FOLDER, 'SGNify_SV_V1_Vicon')
PIXIE_PATH = osp.join(PAOLAS_WORK_FOLDER, 'output_folder')

METHOD_PATH_DICT = {
    'pixie': PIXIE_PATH,
    'sgnify': SGNIFY_SV_PATH,
    'smplify-sl': SGNIFY_SV_PATH,
}

TO_IGNORE = [
    'SubjectCalibration_1',
]

METHOD_GET_PATH_DICT = {
    'pixie': lambda mesh_name: osp.join(
        PIXIE_PATH, mesh_name),
    'soma': lambda mesh_name: osp.join(
        SOMA_PATH, mesh_name),
    'soma-alt': lambda mesh_name: osp.join(
        SOMA_ALT, 'mesh_files', mesh_name, 'body_mesh'),
    'smplify-sl': lambda mesh_name: osp.join(
        SGNIFY_SV_PATH, 'S1', mesh_name),
    'sgnify': lambda mesh_name: osp.join(
        SGNIFY_SV_PATH, 'S1', mesh_name),
}

####### CHANGE NAME HERE
METHOD_GET_MESH_FNAME = {
    'soma': lambda mesh_name: mesh_name,
    'pixie': lambda mesh_name: osp.join(
        f'low_{mesh_name:03d}', f'low_{mesh_name:03d}.obj'),
    'smplify-sl': lambda mesh_name: osp.join(
        f'low_{mesh_name:03d}', 'output_smplify_SV_star/meshes/000.obj'),
    'sgnify': lambda mesh_name: osp.join(
        f'low_{mesh_name:03d}', 'output_sgnify_nomv/meshes/000.obj'),
}


def load_obj(obj_filename):
    """ Ref: https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
    Load a mesh from a file-like object.
    """
    with open(obj_filename, 'r') as f:
        # lines = [line.strip() for line in f]
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    verts = []
    faces = []
    # startswith expects each line to be a string. If the file is read in as
    # bytes then first decode to strings.
    if lines and isinstance(lines[0], bytes):
        lines = [el.decode("utf-8") for el in lines]

    for line in lines:
        tokens = line.strip().split()
        if line.startswith("v "):  # Line is a vertex.
            vert = [float(x) for x in tokens[1:4]]
            if len(vert) != 3:
                msg = "Vertex %s does not have 3 values. Line: %s"
                raise ValueError(msg % (str(vert), str(line)))
            verts.append(vert)
        elif line.startswith("f "):  # Line is a face.
            # Update face properties info.
            face = tokens[1:]
            face_list = [f.split("/") for f in face]
            faces += (list(map(lambda x: int(x[0]), face_list)))
    return (
        verts,
        faces,
    )


def read_verts_and_faces(
    mesh_path,
    method,
) -> Array:
    vertices, faces = load_obj(mesh_path)
    vertices = np.asarray(vertices)
    faces = np.asarray(faces).reshape(-1, 3) - 1

    if method == 'pixie':
        r = R.from_rotvec(-np.pi * np.array([1, 0, 0])).as_matrix()

        offset = vertices.mean(axis=0, keepdims=True)
        vertices = (vertices - offset) @ r.T + offset
    else:
        pass
    return vertices, faces


def point_error(
    source_points: Array,
    target_points: Array
) -> Array:
    return np.sqrt(np.power(source_points - target_points, 2).sum(axis=-1))


def transl_point_error(
    source_points: Array,
    target_points: Array
) -> Array:
    source_center = np.mean(source_points, axis=0, keepdims=True)
    target_center = np.mean(target_points, axis=0, keepdims=True)

    aligned_source_points = source_points - source_center
    aligned_target_points = target_points - target_center

    return point_error(aligned_source_points, aligned_target_points)


def point_error_common_center(
    source_points: Array,
    target_points: Array,
    center,
):
    centered_source_points = source_points - np.mean(
        source_points, axis=0, keepdims=True) + center
    centered_target_points = target_points - np.mean(
        target_points, axis=0, keepdims=True) + center
    return point_error(centered_source_points, centered_target_points)

def point_error_center(
    source_points: Array,
    target_points: Array,
    source_center,
    target_center,
):
    aligned_source_points = source_points - source_center
    aligned_target_points = target_points - target_center
    return point_error(aligned_source_points, aligned_target_points)

def align_by_pelvis(
    source_joints: Array,
    target_joints: Array,
) -> Array:
    source_pelvis = source_joints[[1, 2]].mean(axis=0, keepdims=True)
    target_pelvis = target_joints[[1, 2]].mean(axis=0, keepdims=True)

    aligned_source_points = source_joints - source_pelvis
    aligned_target_points = target_joints - target_pelvis

    return point_error(aligned_source_points, aligned_target_points)




def get_faces_subset(
    faces,
    vertex_indices,
) -> Array:
    subset_faces = []
    for f in faces:
        mask = np.isin(f, vertex_indices)
        if mask.sum() == 3:
            subset_faces.append(f)
    subset_faces = np.stack(subset_faces)

    vertex_mapping = {}
    for ii, vii in enumerate(vertex_indices):
        vertex_mapping[vii] = ii
    new_faces = subset_faces.copy()
    for fii in range(len(new_faces)):
        new_faces[fii, 0] = vertex_mapping[subset_faces[fii, 0]]
        new_faces[fii, 1] = vertex_mapping[subset_faces[fii, 1]]
        new_faces[fii, 2] = vertex_mapping[subset_faces[fii, 2]]

    return new_faces


def load_gt_obj_paths(folder_list, folder_root, frame_segment):
     
    all_segments = []
    segment_indices = []
    for file_name in folder_list:
                
        segment_range = frame_segment[file_name]
        
        start_idx, end_idx = segment_range[0]*2, segment_range[1]*2
        
        folder = Path(os.path.join(folder_root, file_name))
        
        files = sorted(folder.glob("*.obj"), key=lambda p: int(p.stem))
                
        files_numbers = {int(file.stem): file for file in files}
        
        files_segment = [files_numbers[frame] for frame in range(start_idx, end_idx+1)
            if frame in files_numbers
        ]

        sorted_files = list(files_numbers.values())
        sorted_file_names = [file.name for file in sorted_files]

        if files_segment:
            start_file = files_segment[0].name
            end_file = files_segment[-1].name

            start_index = sorted_file_names.index(start_file)
            end_index = sorted_file_names.index(end_file)

            segment_indices.append((start_index, end_index))

        all_segments.append(files_segment)

    return all_segments, segment_indices


def load_mocap_obj_paths(folder_list, folder_root):
    all_segments = []

    for file_name in folder_list:
        meshes_dir = Path(os.path.join(folder_root, file_name, "smplifyx", "meshes"))
        # Extract the first run of digits from each filename stem
        files = sorted(
            meshes_dir.glob("*.obj"),
            key=lambda p: int(re.search(r"\d+", p.stem).group())
        )
        all_segments.append(files)

    return all_segments


def main(
    soma_to_method: Dict[str, str],
    vertex_indices: Dict[str, Array],
    gt_folder_list,
    mocap_folder_list,
    gt_folder_root,
    mocap_folder_root,
    # left_hand_ids: Array,
    # right_hand_ids: Array,
    # pelvis_segm: Dict[str, Dict[str, Array]],
    J_regressor: Optional[np.ndarray] = None,
    method: str = 'pixie',
    sign_seg_file: Optional[str] = None,
    
) -> None:

    center_on_wrists = J_regressor is not None # If regerssor is present then true

    with open(sign_seg_file, 'r') as file:
            frame_segment = json.load(file)

        
    gt_objs, _ = load_gt_obj_paths(gt_folder_list, gt_folder_root, frame_segment)

    mocap_objs = load_mocap_obj_paths(mocap_folder_list, mocap_folder_root)


    full_errors = {'V2V left wrist': [], 'V2V right wrist': []} # empty dictionary to calculate the wrist errors
    for key in vertex_indices:
        full_errors[f'TR {key}'] = [] # adding an empty list for each key in the vertex_indices

    all_sign_errors = {}


    for idx, (soma_key, method_key) in enumerate(tqdm(soma_to_method.items(), desc='Signs')):
        mesh_ids = []
        mean_values_right = []
        mean_values_left = []
        mean_values_upbody = []

        # print("GT objects loaded: ", gt_objs[idx])

        # print("Mocap objects loaded: ", mocap_objs[idx])
        
        # logger.info(f'{soma_key}, {method_key}')
        if any([ign in method_key for ign in TO_IGNORE]):
            continue
        

        sign_errors = {'V2V left wrist': [], 'V2V right wrist': []}
        for key in vertex_indices:
            sign_errors[f'TR {key}'] = []

        # soma_mesh_fnames = sorted(os.listdir(curr_soma_path))
        # loading the central frames
        #if central:
        

       
        for inter_idx, _ in enumerate(gt_objs[idx]):
            
            """
            method_mesh_path = osp.join(
                curr_method_path, METHOD_GET_MESH_FNAME[method](mesh_id)) # loading the mesh of the sign for the method
            assert method_mesh_path, method_mesh_path

            soma_mesh_name = METHOD_GET_MESH_FNAME['soma'](soma_mesh_id)
            soma_mesh_path = osp.join(curr_soma_path, soma_mesh_name) # loading the mesh of the sign for the soma (GT)
            assert osp.exists(soma_mesh_path), soma_mesh_path
            # continue
            """

            # logger.info(f'{method_mesh_path}, {mesh_name}')
            method_vertices, faces = read_verts_and_faces(
                mocap_objs[idx][inter_idx], method) # method vertices and faces
            faces = np.ascontiguousarray(faces)
            # soma_vertices, faces = read_verts_and_faces(soma_mesh_path, 'soma')
            soma_vertices, soma_faces = read_verts_and_faces(
                gt_objs[idx][inter_idx], 'soma') # soma vertices and faces
            

            if np.isnan(method_vertices).any():
                print("True")
                continue
            

            soma_faces = np.ascontiguousarray(soma_faces)
            np.testing.assert_array_equal(soma_faces, faces)

            if center_on_wrists:
                gt_joints = J_regressor.dot(soma_vertices) #  to get the joints

                soma_left_wrist = gt_joints[LEFT_WRIST_INDEX] # left wrist
                soma_right_wrist = gt_joints[RIGHT_WRIST_INDEX] # right wrist



            for key, vertex_index_set in vertex_indices.items():
                # remove left hand ids in signs belonging to subclass 0 from all evaluations
                if key != 'left hand' and class_sign[soma_key] == '0':
                    vertex_index_set = np.setdiff1d(vertex_index_set, left_hand_ids)

                curr_soma_verts = soma_vertices[vertex_index_set]
                curr_method_verts = method_vertices[vertex_index_set]

                tr_error = transl_point_error(
                    curr_method_verts, curr_soma_verts)
                # logger.info(f'TR: {tr_error.shape}')
                if key == 'left hand' and class_sign[soma_key]== '0':
                    #print("True")
                    continue
                else:
                    sign_errors[f'TR {key}'].append(tr_error)

                if key == 'left hand':
                    left_wrist_error = point_error_common_center(
                        curr_method_verts,
                        curr_soma_verts,
                        soma_left_wrist.reshape(-1, 3),
                    )

                    sign_errors['V2V left wrist'].append(left_wrist_error)
                elif key == 'right hand':
                    right_wrist_error = point_error_common_center(
                        curr_method_verts,
                        curr_soma_verts,
                        soma_right_wrist.reshape(-1, 3),
                    )
                    sign_errors[f'V2V right wrist'].append(right_wrist_error)

            # sys.exit(0)

            #print(str(gt_objs[idx][0]).split('/')[-2])

            # print per frame error
            for key in sign_errors:
                if key == 'TR right hand':
                    mean_value_frame = np.stack(sign_errors[key][-1]).mean() * 1000
                    #print(mean_value_frame)
                    mean_values_right.append(mean_value_frame)
                elif key == 'TR left hand' and class_sign[soma_key] != '0':
                    mean_value_frame = np.stack(sign_errors[key][-1]).mean() * 1000
                    mean_values_left.append(mean_value_frame)
                elif key == 'TR above pelvis upper body':
                    mean_value_frame = np.stack(sign_errors[key][-1]).mean() * 1000
                    mean_values_upbody.append(mean_value_frame)
            #mesh_ids.append(mesh_id)
            plt.clf()
        
        for key in sign_errors:
            assert key in full_errors, key
            full_errors[key] += sign_errors[key]

            if len(sign_errors[key]) > 0:
                sign_errors[key] = np.stack(sign_errors[key])

        # Save all the errors for the current sign
        all_sign_errors[soma_key] = sign_errors

        print(str(gt_objs[idx][0]).split('/')[-2])

        print("Right hand: ", np.array(mean_values_right).mean())
        if np.isnan(np.array(mean_values_left).mean()).any(): 
            print("Left hand: ", 0)
        else:
            print("Left hand: ", np.array(mean_values_left).mean())
        print("TR above pelvis upper body: ", np.array(mean_values_upbody).mean())

        
    


    for key in full_errors:
        # assert key in full_errors, key
        if len(full_errors[key]) > 0:
            full_errors[key] = np.concatenate(full_errors[key], axis=0)
            # logger.info(f'{key}: {full_errors[key].shape}')
            mean_value = full_errors[key].mean() * 1000
            logger.info(f'[{method}]: {key.title()}: {mean_value:.4f} (mm)')

    #per_sign_results_path = osp.join(curr_output_folder, 'per_sign.pkl')
    #logger.info(f'Saving {method} results to: {per_sign_results_path}')


if __name__ == '__main__':
    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''), level='INFO', colorize=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--method',
                        choices=['slrt', 'slrt_prior', 'slrt_3d_prior', 'osx'],
                        default='slrt',
                        type=str,
                        nargs='+',
                        help='The method to evaluate')

    parser.add_argument('--central',
                        required=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='analyze only central frames')
    
    parser.add_argument('--evaluate_folder',
                    required=True,
                    type=str,
                    help='Folder to evaluate')
    
    parser.add_argument('--gt_folder',
                    required=True,
                    type=str,
                    help='GT folder')
    
    parser.add_argument('--sign_file',
                    required=True,
                    type=str,
                    help='sign file to evaluate')
    
    parser.add_argument('--sign_seg',
                    required=True,
                    type=str,
                    help='sign segmentation file to evaluate')
    


    args = parser.parse_args()

    gt_folder_root = args.gt_folder
    gt_folder_list = sorted(os.listdir(gt_folder_root))

    save_vis = False

    mocap_folder_root = args.evaluate_folder

    mocap_folder_list = sorted(os.listdir(mocap_folder_root))

    method = args.method
    central = args.central

    # Loading left and right hand indices
    with open('/home/kaustubh/Desktop/Fitting/experiments/data/MANO_SMPLX_vertex_ids.pkl', 'rb') as f:
        mano_data = pickle.load(f)

    # loading full body smplx model data
    smplx_model_data = np.load('/home/kaustubh/Desktop/Fitting/experiments/data/SMPLX_NEUTRAL.npz', allow_pickle=True)
    # loading J_regressor which is converting vertices to joints
    J_regressor = smplx_model_data['J_regressor']

    # contains the classification of each sign for the method and the gt.

    mapping_path = args.sign_file

    soma_to_method = {}
    class_sign = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            tokens = line.split(' ')
            tokens = list(map(lambda x: x.strip('\n'), tokens))
            soma_to_method[tokens[0]] = args.method
            class_sign[tokens[0]] = tokens[1]

    # loading the mano vertex ids
    class_sign = dict(sorted(class_sign.items()))
    soma_to_method = dict(sorted(soma_to_method.items()))

    left_hand_ids = mano_data['left_hand']
    right_hand_ids = mano_data['right_hand']



    pelvis_segm_paths = {
        #'with pelvis': 'data/sgnify_part_segm',
        'above pelvis': '/home/kaustubh/Desktop/Fitting/experiments/data/sgnify_part_segm_above_pelvis_joint',
        #'no pelvis': 'data/sgnify_part_segm_no_pelvis',
    }

    vertex_indices = {
        'all': np.arange(0, 10475),
        'left hand': left_hand_ids,
        'right hand': right_hand_ids,
    }

    pelvis_segm = {}
    for name, pelvis_segm_path in pelvis_segm_paths.items():
        vertex_indices[f'{name} upper body'] = np.load(osp.join(
            pelvis_segm_path, 'upper_body.npy'))
        vertex_indices[f'{name} minus head'] = np.load(osp.join(
            pelvis_segm_path, 'upper_body_minus_head.npy'))
        vertex_indices[f'{name} minus face'] = np.load(osp.join(
            pelvis_segm_path, 'upper_body_minus_face.npy'))

    # for key, value in vertex_indices.items():
    #     logger.info(f'{key}: {value.shape}')
    # sys.exit(0)

    logger.info(f'Evaluating methods: {method}')

    for method_name in method:
        main(
            soma_to_method,
            vertex_indices,
            gt_folder_list,
            mocap_folder_list,
            gt_folder_root,
            mocap_folder_root,
            # left_hand_ids,
            # right_hand_ids,
            # pelvis_segm,
            J_regressor=J_regressor,
            method=method_name,
            sign_seg_file=args.sign_seg
        )
