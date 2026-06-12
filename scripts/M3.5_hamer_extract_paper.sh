#!/bin/bash
# Mirror of M3.5_hamer_extract.sh for the paper-formula fitting mode.
# Differences:
#   - Skips M3_mean_shape_smplerx.py and gender.txt (symlinked from method_biomech)
#   - Writes hamer.pkl to ${OUTPUT_PATH}/hamer/ (a separate output dir)
#   - Consumes the same full input video (data/frames/{sign}/low_*.png) as the
#     original script, so hamer.pkl covers the FULL 30 fps window, not just the
#     sign-active sub-range.
#
# This script is launched by runners/run_dexavatar_biomech_paper.sh
# (PHASE 2). It is non-disruptive: nothing in outputs/method_biomech/ is touched.

set -euo pipefail

cd hamer
CUDA_VISIBLE_DEVICES=0 /home/haipd/miniconda3/envs/hamer/bin/python demo.py \
    --img_folder ${ROOT_PATH} \
    --out_folder ${OUTPUT_PATH}/hamer \
    --batch_size=48 --side_view --save_mesh --full_frame
