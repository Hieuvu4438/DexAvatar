#!/bin/bash
# Phase 3 only: SMPLify-X fitting for all signs with hamer.pkl
# Run in tmux: tmux new -s hamer-fit 'bash run_hamer_fitting.sh'
set -euo pipefail

PROJECT_DIR="/home/haipd/DexAvatar"
INPUT_DIR="${PROJECT_DIR}/data/frames"
OUTPUT_BASE="${PROJECT_DIR}/outputs/method_hamer"
FITTING_EXPERIMENT="${PROJECT_DIR}/dexavatar_fitting"
LOG_DIR="${OUTPUT_BASE}/logs"
mkdir -p "${LOG_DIR}"

SIGNS=()
while IFS= read -r line; do
    sign=$(echo "$line" | awk '{print $1}')
    [ -n "$sign" ] && SIGNS+=("$sign")
done < "${PROJECT_DIR}/data/signs.txt"

echo "=== SMPLify-X Fitting: ${#SIGNS[@]} signs ==="

FAILED_FIT=()
for sign in "${SIGNS[@]}"; do
    ROOT_PATH="${INPUT_DIR}/${sign}"
    OUTPUT_PATH="${OUTPUT_BASE}/${sign}"
    LOG_FILE="${LOG_DIR}/${sign}_fit.log"

    # Skip if meshes already exist
    if [ -d "${OUTPUT_PATH}/smplifyx/meshes" ] && [ "$(ls -1 "${OUTPUT_PATH}/smplifyx/meshes/" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [SKIP] ${sign} — meshes already exist"
        continue
    fi

    # Skip if no hamer.pkl
    if [ ! -f "${OUTPUT_PATH}/hamer/hamer.pkl" ]; then
        echo "  [SKIP] ${sign} — no hamer.pkl"
        FAILED_FIT+=("${sign}")
        continue
    fi

    echo -n "  [RUN]  ${sign} ... "
    cd "${PROJECT_DIR}"
    if ROOT_PATH="${ROOT_PATH}" OUTPUT_PATH="${OUTPUT_PATH}" FITTING_EXPERIMENT="${FITTING_EXPERIMENT}" \
       bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose.sh" > "${LOG_FILE}" 2>&1; then
        MESH_COUNT=$(ls -1 "${OUTPUT_PATH}/smplifyx/meshes/" 2>/dev/null | wc -l)
        echo "OK (${MESH_COUNT} meshes)"
    else
        echo "FAILED (see ${LOG_FILE})"
        FAILED_FIT+=("${sign}")
    fi
done

echo ""
echo "=== Fitting Complete ==="
TOTAL_MESHES=0
SIGNS_DONE=0
for sign in "${SIGNS[@]}"; do
    MESH_DIR="${OUTPUT_BASE}/${sign}/smplifyx/meshes"
    if [ -d "${MESH_DIR}" ]; then
        COUNT=$(ls -1 "${MESH_DIR}" 2>/dev/null | wc -l)
        if [ "$COUNT" -gt 0 ]; then
            SIGNS_DONE=$((SIGNS_DONE + 1))
            TOTAL_MESHES=$((TOTAL_MESHES + COUNT))
        fi
    fi
done
echo "Signs completed: ${SIGNS_DONE}/${#SIGNS[@]}"
echo "Total meshes:    ${TOTAL_MESHES}"
if [ ${#FAILED_FIT[@]} -gt 0 ]; then
    echo "Failed: ${FAILED_FIT[*]}"
fi
echo ""
echo "To evaluate:"
echo "  python eval_mpvpe_regions.py --methods method_hamer --method_names 'DexAvatar-HaMeR' --output_csv outputs/mpvpe_hamer.csv"
