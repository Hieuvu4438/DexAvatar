#!/bin/bash
# Stage 4: SMPLify-X pose fitting (NLF-init variant).
# Mirror of scripts/M4_smplifyx_pose.sh with one extra flag
# (--smplx_init_dir nlf/smplx) so the fitter reads NLF body init instead of
# SMPLer-X. The original M4 scripts are NOT modified.

cd "${FITTING_EXPERIMENT}"
export PYTHONPATH=$PYTHONPATH:$(pwd)/smplifyx
export PYTHONPATH=$PYTHONPATH:$(pwd)

python script.py \
    --path ${ROOT_PATH} \
    --out_path ${OUTPUT_PATH} \
    --gpu_id 0 \
    --split_num 1 \
    --smplx_init_dir nlf/smplx
