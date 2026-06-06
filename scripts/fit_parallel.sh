#!/bin/bash
# Fit multiple signs in parallel for a given method
# Usage: bash scripts/fit_parallel.sh <method> [num_workers]
# Example: bash scripts/fit_parallel.sh hand2d 4

set -eo pipefail

# Activate conda environment
set +u
CONDA_PATH=$(conda info --base)
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate dexavatar
set -u

METHOD="${1:?Usage: fit_parallel.sh <method> [num_workers]}"
NUM_WORKERS="${2:-4}"  # Default 4 parallel workers

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
echo "Parallel Fitting: ${METHOD}"
echo "Config: ${CONFIG}"
echo "Workers: ${NUM_WORKERS}"
echo "=========================================="
echo "Start time: $(date)"

# Get list of signs to fit
SIGNS_TO_FIT=()
for SIGN_DIR in ${INPUT_DIR}/*/; do
    SIGN_DIR="${SIGN_DIR%/}"
    SIGN_NAME=$(basename "$SIGN_DIR")
    METHOD_SIGN="${METHOD_DIR}/${SIGN_NAME}"

    # Skip if already fitted
    if [ -d "${METHOD_SIGN}/smplifyx" ] && ls "${METHOD_SIGN}/smplifyx/"*.pkl 1>/dev/null 2>&1; then
        continue
    fi

    # Skip if no shared data
    if [ ! -d "${METHOD_SIGN}/sapiens_1b" ]; then
        continue
    fi

    SIGNS_TO_FIT+=("${SIGN_NAME}")
done

echo "Signs to fit: ${#SIGNS_TO_FIT[@]}"
echo ""

# Function to fit a single sign
fit_sign() {
    local SIGN_NAME="$1"
    local SIGN_DIR="${INPUT_DIR}/${SIGN_NAME}"
    local METHOD_SIGN="${METHOD_DIR}/${SIGN_NAME}"

    echo "[FIT] ${SIGN_NAME}"

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

    echo "[DONE] ${SIGN_NAME}"
}

# Export function for parallel use
export -f fit_sign
export INPUT_DIR METHOD_DIR FITTING_EXP CONFIG

# Run fitting in parallel
FITTED=0
for SIGN_NAME in "${SIGNS_TO_FIT[@]}"; do
    # Wait if we have too many background processes
    while [ $(jobs -r | wc -l) -ge ${NUM_WORKERS} ]; do
        sleep 1
    done

    fit_sign "${SIGN_NAME}" &
    FITTED=$((FITTED + 1))
done

# Wait for all remaining jobs
wait

echo ""
echo "=== ${METHOD} FITTING COMPLETE ==="
echo "Fitted: ${FITTED} signs"
echo "End time: $(date)"
