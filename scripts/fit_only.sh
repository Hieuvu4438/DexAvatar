#!/bin/bash
# Fit-only script: Run SMPLifyX fitting on pre-extracted shared data
# Usage: bash scripts/fit_only.sh <method_name> <sign_name> <input_img_dir>
# Example: bash scripts/fit_only.sh hand2d Ablehnen /home/haipd/DexAvatar/data/frames/Ablehnen

set -euo pipefail

METHOD="${1:?Usage: fit_only.sh <method> <sign> <img_dir>}"
SIGN="${2:?}"
IMG_DIR="${3:?}"

PROJECT_DIR="/home/haipd/DexAvatar"
FITTING_EXP="${PROJECT_DIR}/dexavatar_fitting"
METHOD_SIGN="${PROJECT_DIR}/outputs/method_${METHOD}/${SIGN}"

declare -A CONFIGS
CONFIGS[hand2d]="cfg_files/fit_smplx_vposer_x_hand2d.yaml"
CONFIGS[biomech]="cfg_files/fit_smplx_vposer_x_biomech.yaml"
CONFIGS[ensemble]="cfg_files/fit_smplx_vposer_x_ensemble.yaml"

CONFIG="${CONFIGS[$METHOD]}"

# Skip if already fitted
if [ -d "${METHOD_SIGN}/smplifyx" ] && ls "${METHOD_SIGN}/smplifyx/"*.pkl 1>/dev/null 2>&1; then
    echo "[SKIP] ${SIGN} - already fitted"
    exit 0
fi

echo "[FIT] ${SIGN} with ${METHOD} (config: ${CONFIG})"

cd "${FITTING_EXP}"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/smplifyx:$(pwd)

CUDA_VISIBLE_DEVICES=0 python smplifyx/main.py \
    --config "${CONFIG}" \
    --data_folder "${METHOD_SIGN}" \
    --output_folder "${METHOD_SIGN}/smplifyx" \
    --img_folder "${IMG_DIR}" \
    --model_folder ../SMPLer-X/common/utils/human_model_files \
    --part_segm_fn assets/smplx_parts_segm.pkl \
    --visualize False --split_num 1 --cur_num 0

echo "[DONE] ${SIGN} (${METHOD})"
