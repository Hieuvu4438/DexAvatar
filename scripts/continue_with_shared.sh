#!/bin/bash
# Continue pipeline with shared extraction for remaining signs
# Then run fitting-only for each method
set -euo pipefail

PROJECT_DIR="/home/haipd/DexAvatar"
INPUT_DIR="${PROJECT_DIR}/data/frames"
SHARED_DIR="${PROJECT_DIR}/outputs/shared"
FITTING_EXP="${PROJECT_DIR}/dexavatar_fitting"

echo "=========================================="
echo "Continue with Shared Pipeline"
echo "=========================================="
echo "Start time: $(date)"

# ============================================================
# PHASE 1: Shared extraction for remaining signs
# ============================================================
echo ""
echo "=== PHASE 1: Shared Extraction (remaining signs) ==="

REMAINING=0
for SIGN_DIR in ${INPUT_DIR}/*/; do
    SIGN_NAME=$(basename "$SIGN_DIR")
    SIGN_SHARED="${SHARED_DIR}/${SIGN_NAME}"

    # Skip if already extracted
    if [ -f "${SIGN_SHARED}/wilor/wilor.pkl" ] && [ -f "${SIGN_SHARED}/sapiens.pkl" ]; then
        continue
    fi

    REMAINING=$((REMAINING + 1))
    echo "[EXTRACT] ${SIGN_NAME} (${REMAINING})"

    mkdir -p "${SIGN_SHARED}"
    ROOT_PATH="${SIGN_DIR}" OUTPUT_PATH="${SIGN_SHARED}" \
        bash -c 'unset LD_LIBRARY_PATH && bash scripts/shared_extract.sh'
    echo "[DONE] ${SIGN_NAME}"
done

echo ""
echo "=== PHASE 1 COMPLETE: ${REMAINING} signs extracted ==="
echo ""

# ============================================================
# PHASE 2: Link shared data to method directories
# ============================================================
echo "=== PHASE 2: Link shared data ==="

for METHOD in hand2d biomech ensemble; do
    METHOD_DIR="${PROJECT_DIR}/outputs/method_${METHOD}"
    LINKED=0

    for SIGN_DIR in ${INPUT_DIR}/*/; do
        SIGN_NAME=$(basename "$SIGN_DIR")
        SHARED_SIGN="${SHARED_DIR}/${SIGN_NAME}"
        METHOD_SIGN="${METHOD_DIR}/${SIGN_NAME}"

        mkdir -p "${METHOD_SIGN}"

        for item in sapiens_1b sapiens.pkl smplerx hamer wilor mean_shape_smplx.npy gender.txt; do
            if [ -e "${SHARED_SIGN}/${item}" ] && [ ! -e "${METHOD_SIGN}/${item}" ]; then
                ln -sf "${SHARED_SIGN}/${item}" "${METHOD_SIGN}/${item}"
                LINKED=$((LINKED + 1))
            fi
        done
    done

    echo "[DONE] ${METHOD}: ${LINKED} new links"
done

echo ""
echo "=== PHASE 2 COMPLETE ==="
echo ""

# ============================================================
# PHASE 3: Run fitting for each method (parallel tmux)
# ============================================================
echo "=== PHASE 3: Launch fitting ==="

declare -A CONFIGS
CONFIGS[hand2d]="cfg_files/fit_smplx_vposer_x_hand2d.yaml"
CONFIGS[biomech]="cfg_files/fit_smplx_vposer_x_biomech.yaml"
CONFIGS[ensemble]="cfg_files/fit_smplx_vposer_x_ensemble.yaml"

for METHOD in hand2d biomech ensemble; do
    CONFIG="${CONFIGS[$METHOD]}"
    METHOD_DIR="${PROJECT_DIR}/outputs/method_${METHOD}"
    LOG="${PROJECT_DIR}/outputs/method_${METHOD}.log"

    echo "Launching fitting for ${METHOD}..."

    tmux new-session -d -s "fit_${METHOD}" \
        "cd ${PROJECT_DIR} && \
         for SIGN_DIR in ${INPUT_DIR}/*/; do \
             SIGN_NAME=\$(basename \"\$SIGN_DIR\"); \
             METHOD_SIGN=\"${METHOD_DIR}/\$SIGN_NAME\"; \
             if [ -d \"\$METHOD_SIGN/smplifyx/results\" ]; then \
                 continue; \
             fi; \
             echo \"[FIT] \$SIGN_NAME\"; \
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
echo "Monitor: tmux list-sessions | grep fit_"
echo "End time: $(date)"
