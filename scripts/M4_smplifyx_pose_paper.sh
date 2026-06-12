#!/bin/bash
# Mirror of M4_smplifyx_pose_biomech.sh for the paper-formula fitting mode.
# Diff vs. the biomech variant: --config points at fit_smplx_vposer_x_paper.yaml
# (which sets sign_segment: 'segment_paper.json' — the wider paper central window).
#
# This script is launched by runners/run_dexavatar_biomech_paper.sh (PHASE 3).
# It is non-disruptive: nothing in outputs/method_biomech/ is touched; outputs
# go to ${OUTPUT_PATH}/smplifyx/ which lives under outputs/method_biomech_paper/.

cd "${FITTING_EXPERIMENT}"
export PYTHONPATH=$PYTHONPATH:$(pwd)/smplifyx
export PYTHONPATH=$PYTHONPATH:$(pwd)

python script.py --path ${ROOT_PATH} --out_path ${OUTPUT_PATH} --gpu_id 0 --split_num 1 --config cfg_files/fit_smplx_vposer_x_paper.yaml
