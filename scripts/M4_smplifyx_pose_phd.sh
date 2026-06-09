#!/bin/bash
# M4: SMPLify-X Fitting với PHD Diffusion Body Pose Prior
# ========================================================
# Thay thế SignBPoser bằng score-based diffusion prior
#
# Usage:
#   bash scripts/M4_smplifyx_pose_phd.sh
#   ROOT_PATH=data/frames/SIGN_NAME OUTPUT_PATH=outputs/phd/SIGN_NAME bash scripts/M4_smplifyx_pose_phd.sh

set -e

# Default paths (có thể override qua env vars)
ROOT_PATH=${ROOT_PATH:-"data/frames/Tisch"}
OUTPUT_PATH=${OUTPUT_PATH:-"outputs/phd/Tisch"}
FITTING_EXPERIMENT=${FITTING_EXPERIMENT:-"dexavatar_fitting"}

echo "============================================"
echo "M4: SMPLify-X với PHD Diffusion Prior"
echo "============================================"
echo "ROOT_PATH: $ROOT_PATH"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "FITTING_EXPERIMENT: $FITTING_EXPERIMENT"
echo "============================================"

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Run SMPLify-X fitting với PHD config
cd "$FITTING_EXPERIMENT"

python smplifyx/main.py \
    --config cfg_files/fit_smplx_phd.yaml \
    --output_folder "../$OUTPUT_PATH" \
    --data_folder "../$ROOT_PATH" \
    --use_signbposer False \
    --use_phd_prior True \
    --phd_prior_dir './smplifyx/signbposer/snapshots/phd_best.pt' \
    --phd_guidance_scale 1.0 \
    --phd_num_inference_steps 50 \
    --use_direct_optimization True \
    --save_meshes True \
    --visualize False \
    --interactive True \
    2>&1 | tee "../${OUTPUT_PATH}/phd_fitting.log"

echo ""
echo "Done! Results saved to: $OUTPUT_PATH"
