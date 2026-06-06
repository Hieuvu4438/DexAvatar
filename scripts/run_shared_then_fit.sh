#!/bin/bash
# Optimized Pipeline: Shared extraction + Parallel fitting for all methods
# Usage: bash scripts/run_shared_then_fit.sh
set -euo pipefail

PROJECT_DIR="/home/haipd/DexAvatar"
INPUT_DIR="${PROJECT_DIR}/data/frames"
SHARED_DIR="${PROJECT_DIR}/outputs/shared"
FITTING_EXP="${PROJECT_DIR}/dexavatar_fitting"

METHODS=("hand2d" "biomech" "ensemble")
declare -A METHOD_CONFIGS
METHOD_CONFIGS[hand2d]="cfg_files/fit_smplx_vposer_x_hand2d.yaml"
METHOD_CONFIGS[biomech]="cfg_files/fit_smplx_vposer_x_biomech.yaml"
METHOD_CONFIGS[ensemble]="cfg_files/fit_smplx_vposer_x_ensemble.yaml"

echo "=========================================="
echo "Optimized Pipeline: Shared Extract + Fit"
echo "=========================================="
echo "Start time: $(date)"

# ============================================================
# PHASE 1: Shared extraction (Sapiens + SMPLer-X + WiLoR)
# ============================================================
echo ""
echo "=== PHASE 1: Shared Extraction ==="

for SIGN_DIR in ${INPUT_DIR}/*/; do
    SIGN_NAME=$(basename "$SIGN_DIR")
    SIGN_SHARED="${SHARED_DIR}/${SIGN_NAME}"

    # Skip if already extracted
    if [ -f "${SIGN_SHARED}/wilor/wilor.pkl" ] && [ -f "${SIGN_SHARED}/sapiens.pkl" ]; then
        echo "[SKIP] ${SIGN_NAME} - already extracted"
        continue
    fi

    mkdir -p "${SIGN_SHARED}"
    ROOT_PATH="${SIGN_DIR}" OUTPUT_PATH="${SIGN_SHARED}" \
        bash -c 'unset LD_LIBRARY_PATH && bash scripts/shared_extract.sh'
    echo "[DONE] Shared extraction: ${SIGN_NAME}"
done

echo ""
echo "=== PHASE 1 COMPLETE: All shared data ready ==="
echo ""

# ============================================================
# PHASE 2: Copy shared data + Run fitting for each method
# ============================================================
echo "=== PHASE 2: Copy shared data to method dirs ==="

for METHOD in "${METHODS[@]}"; do
    METHOD_DIR="${PROJECT_DIR}/outputs/method_${METHOD}"
    echo "Setting up ${METHOD}..."

    for SIGN_DIR in ${INPUT_DIR}/*/; do
        SIGN_NAME=$(basename "$SIGN_DIR")
        SHARED_SIGN="${SHARED_DIR}/${SIGN_NAME}"
        METHOD_SIGN="${METHOD_DIR}/${SIGN_NAME}"

        # Create symlinks to shared data (saves disk space)
        mkdir -p "${METHOD_SIGN}"

        # Link shared stages (read-only)
        for item in sapiens_1b sapiens.pkl smplerx hamer wilor mean_shape_smplx.npy gender.txt; do
            if [ -e "${SHARED_SIGN}/${item}" ] && [ ! -e "${METHOD_SIGN}/${item}" ]; then
                ln -sf "${SHARED_SIGN}/${item}" "${METHOD_SIGN}/${item}"
            fi
        done
    done
    echo "[DONE] ${METHOD} linked to shared data"
done

echo ""
echo "=== PHASE 2 COMPLETE: All methods linked ==="
echo ""

# ============================================================
# PHASE 3: Run fitting in parallel (tmux sessions)
# ============================================================
echo "=== PHASE 3: Launch parallel fitting ==="

for METHOD in "${METHODS[@]}"; do
    METHOD_DIR="${PROJECT_DIR}/outputs/method_${METHOD}"
    CONFIG="${METHOD_CONFIGS[$METHOD]}"
    LOG="${PROJECT_DIR}/outputs/method_${METHOD}.log"

    echo "Launching fitting for ${METHOD}..."

    # Run fitting only (skip extraction stages)
    tmux new-session -d -s "fit_${METHOD}" \
        "cd ${PROJECT_DIR} && \
         for SIGN_DIR in ${INPUT_DIR}/*/; do \
             SIGN_NAME=\$(basename \"\$SIGN_DIR\"); \
             METHOD_SIGN=\"${METHOD_DIR}/\$SIGN_NAME\"; \
             if [ -f \"\$METHOD_SIGN/smplifyx/results\" ]; then \
                 echo \"[SKIP] \$SIGN_NAME - already fitted\"; \
                 continue; \
             fi; \
             echo \"[FIT] \$SIGN_NAME with ${METHOD}\"; \
             cd ${FITTING_EXP} && \
             export PYTHONPATH=\${PYTHONPATH:-}:\$(pwd)/smplifyx:\$(pwd) && \
             CUDA_VISIBLE_DEVICES=0 python smplifyx/main.py \
                 --config ${CONFIG} \
                 --data_folder \$METHOD_SIGN \
                 --output_folder \$METHOD_SIGN/smplifyx \
                 --img_folder \$SIGN_DIR \
                 --model_folder ../SMPLer-X/common/utils/human_model_files \
                 --part_segm_fn assets/smplx_parts_segm.pkl \
                 --visualize False --split_num 1 --cur_num 0; \
             cd ${PROJECT_DIR}; \
             echo \"[DONE] \$SIGN_NAME\"; \
         done && \
         echo '=== ${METHOD} FITTING COMPLETE ==='" \
        2>&1 | tee "${LOG}"

    echo "[LAUNCHED] fit_${METHOD}"
done

echo ""
echo "=== All fitting sessions launched ==="
echo "Monitor with: tmux list-sessions | grep fit_"
echo "End time: $(date)"
