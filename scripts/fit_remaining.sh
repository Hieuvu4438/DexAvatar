#!/bin/bash
# Fit remaining signs for a given method
# Usage: bash scripts/fit_remaining.sh <method>
# Example: bash scripts/fit_remaining.sh hand2d

set -eo pipefail

# Activate conda environment
set +u
CONDA_PATH=$(conda info --base)
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate dexavatar
set -u

METHOD="${1:?Usage: fit_remaining.sh <method>}"

PROJECT_DIR="/home/haipd/DexAvatar"
INPUT_DIR="${PROJECT_DIR}/data/frames"
FITTING_EXP="${PROJECT_DIR}/dexavatar_fitting"
METHOD_DIR="${PROJECT_DIR}/outputs/method_${METHOD}"

declare -A CONFIGS
CONFIGS[hand2d]="cfg_files/fit_smplx_vposer_x_hand2d.yaml"
CONFIGS[biomech]="cfg_files/fit_smplx_vposer_x_biomech.yaml"
CONFIGS[ensemble]="cfg_files/fit_smplx_vposer_x_ensemble.yaml"

CONFIG="${CONFIGS[$METHOD]}"

echo "=========================================="
echo "Fitting remaining signs for ${METHOD}"
echo "Config: ${CONFIG}"
echo "=========================================="
echo "Start time: $(date)"

FITTED=0
SKIPPED=0

for SIGN_DIR in ${INPUT_DIR}/*/; do
    SIGN_DIR="${SIGN_DIR%/}"  # Remove trailing slash
    SIGN_NAME=$(basename "$SIGN_DIR")
    METHOD_SIGN="${METHOD_DIR}/${SIGN_NAME}"

    # Skip if already fitted
    if [ -d "${METHOD_SIGN}/smplifyx" ] && ls "${METHOD_SIGN}/smplifyx/"*.pkl 1>/dev/null 2>&1; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Skip if no shared data
    if [ ! -d "${METHOD_SIGN}/sapiens_1b" ]; then
        echo "[WARN] No shared data for ${SIGN_NAME}, skipping"
        continue
    fi

    echo "[FIT] ${SIGN_NAME} (${FITTED}/${SKIPPED})"
    FITTED=$((FITTED + 1))

    cd "${FITTING_EXP}"
    export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/smplifyx:$(pwd)

    CUDA_VISIBLE_DEVICES=0 python smplifyx/main.py \
        --config "${CONFIG}" \
        --data_folder "${METHOD_SIGN}" \
        --output_folder "${METHOD_SIGN}/smplifyx" \
        --img_folder "${SIGN_DIR}" \
        --model_folder ../SMPLer-X/common/utils/human_model_files \
        --part_segm_fn assets/smplx_parts_segm.pkl \
        --visualize False --split_num 1 --cur_num 0

    cd "${PROJECT_DIR}"
    echo "[DONE] ${SIGN_NAME}"
done

echo ""
echo "=== ${METHOD} FITTING COMPLETE ==="
echo "Fitted: ${FITTED}, Skipped: ${SKIPPED}"
echo "End time: $(date)"
