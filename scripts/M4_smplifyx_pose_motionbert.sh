#!/bin/bash
# M4: SMPLify-X Fitting với MotionBERT Body Pose Prior
# =====================================================
# Thay thế SignBPoser bằng pretrained MotionBERT + SMPL head
#
# Usage:
#   bash scripts/M4_smplifyx_pose_motionbert.sh
#   ROOT_PATH=data/frames/SIGN_NAME OUTPUT_PATH=outputs/motionbert/SIGN_NAME bash scripts/M4_smplifyx_pose_motionbert.sh

set -e

# Default paths (có thể override qua env vars)
ROOT_PATH=${ROOT_PATH:-"data/frames/Tisch"}
OUTPUT_PATH=${OUTPUT_PATH:-"outputs/motionbert/Tisch"}
FITTING_EXPERIMENT=${FITTING_EXPERIMENT:-"dexavatar_fitting"}

echo "============================================"
echo "M4: SMPLify-X với MotionBERT Prior"
echo "============================================"
echo "ROOT_PATH: $ROOT_PATH"
echo "OUTPUT_PATH: $OUTPUT_PATH"
echo "FITTING_EXPERIMENT: $FITTING_EXPERIMENT"
echo "============================================"

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Run SMPLify-X fitting với MotionBERT config
cd "$FITTING_EXPERIMENT"

python smplifyx/main.py \
    --config cfg_files/fit_smplx_motionbert.yaml \
    --output_folder "../$OUTPUT_PATH" \
    --data_folder "../$ROOT_PATH" \
    --use_signbposer False \
    --use_motionbert_prior True \
    --motionbert_prior_dir './smplifyx/signbposer/snapshots/motionbert_best.pt' \
    --use_direct_optimization True \
    --save_meshes True \
    --visualize False \
    --interactive True \
    2>&1 | tee "../${OUTPUT_PATH}/motionbert_fitting.log"

echo ""
echo "Done! Results saved to: $OUTPUT_PATH"
