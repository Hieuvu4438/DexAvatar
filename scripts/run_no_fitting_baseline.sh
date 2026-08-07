#!/bin/bash
# Direct Forward Pass of Shared Initializations (No Fitting / No VAE) - Parallel
set -euo pipefail

PROJECT_DIR="/home/haipd/DexAvatar"
INPUT_DIR="${PROJECT_DIR}/data/frames"
SHARED_DIR="${PROJECT_DIR}/outputs/shared"
OUTPUT_DIR="${PROJECT_DIR}/outputs/output_baseline_no_fitting"
FITTING_EXP="${PROJECT_DIR}/dexavatar_fitting"
MAX_WORKERS=4

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Running Baseline Initialization Output (No Fitting / No VAE) - Parallel (${MAX_WORKERS} workers)"
echo "=========================================="

process_sign() {
    local SIGN_PATH="$1"
    local SIGN_DIR="${SIGN_PATH%/}"
    local SIGN_NAME=$(basename "${SIGN_DIR}")
    local SIGN_SHARED="${SHARED_DIR}/${SIGN_NAME}"
    local SIGN_OUT="${OUTPUT_DIR}/${SIGN_NAME}"

    if [ ! -d "${SIGN_SHARED}" ]; then
        echo "[SKIP] ${SIGN_NAME} - no shared data found"
        return 0
    fi

    if [ -d "${SIGN_OUT}/smplifyx/images" ] && [ $(ls -1 "${SIGN_OUT}/smplifyx/images"/*.png 2>/dev/null | wc -l) -gt 0 ]; then
        echo "[SKIP] ${SIGN_NAME} - already generated"
        return 0
    fi

    echo "[PROCESSING INIT] ${SIGN_NAME}"
    (
        cd "${FITTING_EXP}"
        export PYTHONPATH=$(pwd)/smplifyx:$(pwd):${PYTHONPATH:-}

        /home/haipd/miniconda3/envs/dexavatar/bin/python smplifyx/main.py \
            --config cfg_files/fit_smplx_vposer_x.yaml \
            --data_folder "${SIGN_SHARED}" \
            --output_folder "${SIGN_OUT}/smplifyx" \
            --img_folder "${SIGN_DIR}" \
            --model_folder ../SMPLer-X/common/utils/human_model_files \
            --visualize False \
            --split_num 1 --cur_num 0 \
            --no_fit True >/dev/null 2>&1
    )
    echo "[DONE] ${SIGN_NAME}"
}

export -f process_sign
export INPUT_DIR SHARED_DIR OUTPUT_DIR FITTING_EXP

ls -d ${INPUT_DIR}/*/ | xargs -n 1 -P ${MAX_WORKERS} bash -c 'process_sign "$0"'

echo "=========================================="
echo "All signs completed!"
echo "=========================================="
