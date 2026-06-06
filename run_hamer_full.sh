#!/bin/bash
# Run original DexAvatar pipeline with HaMeR (not WiLoR) for all 57 signs.
# Reuses existing Sapiens + SMPLer-X results from method_biomech to save time.
set -euo pipefail

PROJECT_DIR="/home/haipd/DexAvatar"
INPUT_DIR="${PROJECT_DIR}/data/frames"
OUTPUT_BASE="${PROJECT_DIR}/outputs/method_hamer"
SHARED_SRC="${PROJECT_DIR}/outputs/method_biomech"
FITTING_EXPERIMENT="${PROJECT_DIR}/dexavatar_fitting"
LOG_DIR="${OUTPUT_BASE}/logs"

mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

# Get sign list
SIGNS=()
while IFS= read -r line; do
    sign=$(echo "$line" | awk '{print $1}')
    [ -n "$sign" ] && SIGNS+=("$sign")
done < "${PROJECT_DIR}/data/signs.txt"

echo "=== HaMeR Full Pipeline: ${#SIGNS[@]} signs ==="

# ── Phase 1: Symlink shared data ──────────────────────────────────────────
echo ""
echo "Phase 1: Linking shared data (sapiens, smplerx, etc.) from ${SHARED_SRC} ..."
for sign in "${SIGNS[@]}"; do
    dest="${OUTPUT_BASE}/${sign}"
    src="${SHARED_SRC}/${sign}"
    mkdir -p "${dest}"

    if [ ! -d "${src}" ]; then
        echo "  [WARN] No shared data for ${sign} in ${SHARED_SRC}, skipping"
        continue
    fi

    for item in sapiens_1b sapiens.pkl smplerx mean_shape_smplx.npy gender.txt; do
        if [ -e "${dest}/${item}" ]; then
            continue  # already exists
        fi
        if [ -e "${src}/${item}" ]; then
            ln -sf "${src}/${item}" "${dest}/${item}"
        fi
    done
done
echo "Phase 1 done."

# ── Phase 2: HaMeR extraction ─────────────────────────────────────────────
echo ""
echo "Phase 2: HaMeR hand extraction ..."
FAILED_HAMER=()
for sign in "${SIGNS[@]}"; do
    ROOT_PATH="${INPUT_DIR}/${sign}"
    OUTPUT_PATH="${OUTPUT_BASE}/${sign}"
    LOG_FILE="${LOG_DIR}/${sign}_hamer.log"

    if [ -f "${OUTPUT_PATH}/hamer/hamer.pkl" ]; then
        echo "  [SKIP] ${sign} — hamer.pkl already exists"
        continue
    fi

    echo -n "  [RUN]  ${sign} ... "
    cd "${PROJECT_DIR}"
    if ROOT_PATH="${ROOT_PATH}" OUTPUT_PATH="${OUTPUT_PATH}" \
       bash -c "source scripts/config.sh && bash scripts/M3.5_hamer_extract.sh" > "${LOG_FILE}" 2>&1; then
        if [ -f "${OUTPUT_PATH}/hamer/hamer.pkl" ]; then
            echo "OK"
        else
            echo "WARN — no hamer.pkl produced"
            FAILED_HAMER+=("${sign}")
        fi
    else
        echo "FAILED (see ${LOG_FILE})"
        FAILED_HAMER+=("${sign}")
    fi
done
echo "Phase 2 done. Failed: ${#FAILED_HAMER[@]}"
if [ ${#FAILED_HAMER[@]} -gt 0 ]; then
    echo "  Failed signs: ${FAILED_HAMER[*]}"
fi

# ── Phase 3: SMPLify-X fitting ────────────────────────────────────────────
echo ""
echo "Phase 3: SMPLify-X fitting ..."
FAILED_FIT=()
for sign in "${SIGNS[@]}"; do
    ROOT_PATH="${INPUT_DIR}/${sign}"
    OUTPUT_PATH="${OUTPUT_BASE}/${sign}"
    LOG_FILE="${LOG_DIR}/${sign}_fit.log"

    if [ -d "${OUTPUT_PATH}/smplifyx/meshes" ] && [ "$(ls -1 "${OUTPUT_PATH}/smplifyx/meshes/" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [SKIP] ${sign} — smplifyx/meshes already exists"
        continue
    fi

    # Check hamer.pkl exists
    if [ ! -f "${OUTPUT_PATH}/hamer/hamer.pkl" ]; then
        echo "  [SKIP] ${sign} — no hamer.pkl (HaMeR failed?)"
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
echo "Phase 3 done. Failed: ${#FAILED_FIT[@]}"
if [ ${#FAILED_FIT[@]} -gt 0 ]; then
    echo "  Failed signs: ${FAILED_FIT[*]}"
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "=== Pipeline Complete ==="
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
echo ""
echo "To evaluate, run:"
echo "  python eval_mpvpe_regions.py --methods method_hamer --method_names 'DexAvatar-HaMeR' --output_csv outputs/mpvpe_hamer.csv"
