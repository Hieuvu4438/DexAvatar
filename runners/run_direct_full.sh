#!/bin/bash
# Run DexAvatar with Approach A+D+E: Direct optimization + Absolute depth + Uncertainty fusion
# Reuses shared extraction data from method_biomech.
# Only runs SMPLify-X fitting (Phase 4).
set -euo pipefail

PROJECT_DIR="/home/haipd/DexAvatar"
INPUT_DIR="${PROJECT_DIR}/data/frames"
OUTPUT_BASE="${PROJECT_DIR}/outputs/method_direct"
SHARED_SRC="${PROJECT_DIR}/outputs/method_biomech"
FITTING_EXPERIMENT="${PROJECT_DIR}/dexavatar_fitting"
LOG_DIR="${OUTPUT_BASE}/logs"
mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

SIGNS=()
while IFS= read -r line; do
    sign=$(echo "$line" | awk '{print $1}')
    [ -n "$sign" ] && SIGNS+=("$sign")
done < "${PROJECT_DIR}/data/signs.txt"

echo "=== DexAvatar Direct (A+D+E) Fitting: ${#SIGNS[@]} signs ==="

# ── Phase 1: Symlink shared data ──────────────────────────────────────────
echo ""
echo "Phase 1: Linking shared data from ${SHARED_SRC} ..."
for sign in "${SIGNS[@]}"; do
    dest="${OUTPUT_BASE}/${sign}"
    src="${SHARED_SRC}/${sign}"
    mkdir -p "${dest}"

    if [ ! -d "${src}" ]; then
        echo "  [WARN] No shared data for ${sign} in ${SHARED_SRC}, skipping"
        continue
    fi

    for item in sapiens_1b sapiens.pkl smplerx hamer mean_shape_smplx.npy gender.txt; do
        if [ -e "${dest}/${item}" ]; then
            continue
        fi
        if [ -e "${src}/${item}" ]; then
            ln -sf "${src}/${item}" "${dest}/${item}"
        fi
    done
done
echo "Phase 1 done."

# ── Phase 2: SMPLify-X fitting with A+D+E ─────────────────────────────────
echo ""
echo "Phase 2: SMPLify-X fitting with Direct optimization + Absolute depth + Uncertainty fusion ..."
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

    # Check hamer.pkl exists
    if [ ! -f "${OUTPUT_PATH}/hamer/hamer.pkl" ]; then
        echo "  [SKIP] ${sign} — no hamer.pkl"
        FAILED_FIT+=("${sign}")
        continue
    fi

    echo -n "  [RUN]  ${sign} ... "
    cd "${PROJECT_DIR}"
    if ROOT_PATH="${ROOT_PATH}" OUTPUT_PATH="${OUTPUT_PATH}" FITTING_EXPERIMENT="${FITTING_EXPERIMENT}" \
       bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose_direct.sh" > "${LOG_FILE}" 2>&1; then
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
echo "  python evaluation/eval_mpvpe_common_frames.py --methods method_biomech method_hand2d method_hamer output_wilor method_direct --method_names 'Biomech' 'Hand2D' 'HaMeR' 'WiLoR' 'Direct(ADE)' --output_csv outputs/mpvpe_all_methods.csv"
