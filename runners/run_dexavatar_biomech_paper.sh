#!/bin/bash
# Re-fit SMPL-X on the SGNify paper's central window (0.5T/8 < t < 7T/8).
#
# This driver builds a parallel fitting mode (outputs/method_biomech_paper/)
# without touching any existing baseline script, config, or output.
#
# Pipeline:
#   PHASE 0  Build segment_paper.json from input video T
#   PHASE 1  Symlink shared assets (Sapiens, SMPLer-X, mean shape, gender)
#            from outputs/method_biomech/{sign}/ to outputs/method_biomech_paper/{sign}/
#   PHASE 2  Re-run HaMeR on the FULL input video (covers buffer frames too)
#   PHASE 3  Re-run SMPLify-X with the paper-window sign segment
#   PHASE 4  Evaluate with --paper_central flag (reuses existing code)
#
# Usage:
#   bash runners/run_dexavatar_biomech_paper.sh
#
# Non-disruptive: nothing in outputs/method_biomech/ is touched.
#
set -euo pipefail

PROJECT_DIR="/home/haipd/DexAvatar"
INPUT_DIR="${PROJECT_DIR}/data/frames"
SHARED_SRC="${PROJECT_DIR}/outputs/method_biomech"
OUTPUT_BASE="${PROJECT_DIR}/outputs/method_biomech_paper"
FITTING_EXPERIMENT="${PROJECT_DIR}/dexavatar_fitting"
LOG_DIR="${OUTPUT_BASE}/logs"

mkdir -p "${OUTPUT_BASE}" "${LOG_DIR}"

echo "=========================================="
echo "Paper-Window SMPL-X Re-fit (non-disruptive)"
echo "=========================================="
echo "Project:  ${PROJECT_DIR}"
echo "Output:   ${OUTPUT_BASE}"
echo "Shared:   ${SHARED_SRC}"
echo "Start:    $(date)"
echo ""

# Get sign list
SIGNS=()
while IFS= read -r line; do
    sign=$(echo "$line" | awk '{print $1}')
    [ -n "$sign" ] && SIGNS+=("$sign")
done < "${PROJECT_DIR}/data/signs.txt"

echo "Signs to process: ${#SIGNS[@]}"
echo ""

# ============================================================
# PHASE 0: Build segment_paper.json
# ============================================================
echo "=== PHASE 0: Build paper-formula segment JSON ==="
python3 "${PROJECT_DIR}/scripts/build_paper_segment.py" \
    --signs_file "${PROJECT_DIR}/data/signs.txt" \
    --frames_root "${INPUT_DIR}" \
    --out_path "${FITTING_EXPERIMENT}/cfg_files/segment_paper.json"
echo ""

# ============================================================
# PHASE 1: Symlink shared assets from method_biomech
# ============================================================
echo "=== PHASE 1: Symlink shared assets (Sapiens, SMPLer-X, mean shape, gender) ==="
FAILED_PHASE1=()
for sign in "${SIGNS[@]}"; do
    dest="${OUTPUT_BASE}/${sign}"
    src="${SHARED_SRC}/${sign}"
    mkdir -p "${dest}"

    if [ ! -d "${src}" ]; then
        echo "  [WARN] No shared source for ${sign} in ${src}; skipping"
        FAILED_PHASE1+=("${sign}")
        continue
    fi

    for item in sapiens_1b sapiens.pkl smplerx mean_shape_smplx.npy gender.txt; do
        # Skip if already a link or real file
        if [ -e "${dest}/${item}" ] || [ -L "${dest}/${item}" ]; then
            continue
        fi
        if [ -e "${src}/${item}" ]; then
            ln -sf "${src}/${item}" "${dest}/${item}"
        fi
    done
done
echo "Phase 1 done. Failed (no shared source): ${#FAILED_PHASE1[@]}"
if [ ${#FAILED_PHASE1[@]} -gt 0 ]; then
    echo "  ${FAILED_PHASE1[*]}"
fi
echo ""

# ============================================================
# PHASE 2: Re-run HaMeR on FULL input window
# ============================================================
echo "=== PHASE 2: HaMeR extraction on full input window ==="
FAILED_HAMER=()
for sign in "${SIGNS[@]}"; do
    ROOT_PATH="${INPUT_DIR}/${sign}"
    OUTPUT_PATH="${OUTPUT_BASE}/${sign}"
    LOG_FILE="${LOG_DIR}/${sign}_hamer.log"

    # Skip if hamer.pkl already exists
    if [ -f "${OUTPUT_PATH}/hamer/hamer.pkl" ]; then
        echo "  [SKIP] ${sign} — hamer.pkl already exists"
        continue
    fi

    # Skip if no symlinked shared data
    if [ ! -e "${OUTPUT_PATH}/sapiens.pkl" ] || [ ! -d "${OUTPUT_PATH}/smplerx" ]; then
        echo "  [SKIP] ${sign} — no shared data (Phase 1 failed?)"
        FAILED_HAMER+=("${sign}")
        continue
    fi

    echo -n "  [RUN]  ${sign} ... "
    cd "${PROJECT_DIR}"
    if ROOT_PATH="${ROOT_PATH}" OUTPUT_PATH="${OUTPUT_PATH}" \
       bash -c "source scripts/config.sh && bash scripts/M3.5_hamer_extract_paper.sh" > "${LOG_FILE}" 2>&1; then
        if [ -f "${OUTPUT_PATH}/hamer/hamer.pkl" ]; then
            COUNT=$(python3 -c "import pickle; print(len(pickle.load(open('${OUTPUT_PATH}/hamer/hamer.pkl','rb'))))")
            echo "OK (${COUNT} hand entries)"
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
    echo "  ${FAILED_HAMER[*]}"
fi
echo ""

# ============================================================
# PHASE 3: Re-run SMPLify-X with paper-window sign segment
# ============================================================
echo "=== PHASE 3: SMPLify-X fitting on paper window ==="
FAILED_FIT=()
for sign in "${SIGNS[@]}"; do
    ROOT_PATH="${INPUT_DIR}/${sign}"
    OUTPUT_PATH="${OUTPUT_BASE}/${sign}"
    LOG_FILE="${LOG_DIR}/${sign}_fit.log"

    # Skip if meshes already exist
    if [ -d "${OUTPUT_PATH}/smplifyx/meshes" ] && \
       [ "$(ls -1 "${OUTPUT_PATH}/smplifyx/meshes/" 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "  [SKIP] ${sign} — smplifyx/meshes already exists"
        continue
    fi

    # Skip if hamer didn't succeed
    if [ ! -f "${OUTPUT_PATH}/hamer/hamer.pkl" ]; then
        echo "  [SKIP] ${sign} — no hamer.pkl (Phase 2 failed?)"
        FAILED_FIT+=("${sign}")
        continue
    fi

    echo -n "  [RUN]  ${sign} ... "
    cd "${PROJECT_DIR}"
    if ROOT_PATH="${ROOT_PATH}" OUTPUT_PATH="${OUTPUT_PATH}" FITTING_EXPERIMENT="${FITTING_EXPERIMENT}" \
       bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose_paper.sh" > "${LOG_FILE}" 2>&1; then
        MESH_COUNT=$(ls -1 "${OUTPUT_PATH}/smplifyx/meshes/" 2>/dev/null | wc -l)
        echo "OK (${MESH_COUNT} meshes)"
    else
        echo "FAILED (see ${LOG_FILE})"
        FAILED_FIT+=("${sign}")
    fi
done
echo "Phase 3 done. Failed: ${#FAILED_FIT[@]}"
if [ ${#FAILED_FIT[@]} -gt 0 ]; then
    echo "  ${FAILED_FIT[*]}"
fi
echo ""

# ============================================================
# Summary
# ============================================================
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
echo "Total meshes:    ${TOTAL_MESHES} (target: ~2,700-2,900 matching paper's 2,872)"
echo ""

if [ ${TOTAL_MESHES} -gt 0 ]; then
    echo "=== PHASE 4: Evaluate with --paper_central ==="
    cd "${PROJECT_DIR}"
    python3 evaluation/eval_mpvpe_regions.py \
        --methods method_biomech_paper \
        --method_names "DexAvatar-Biomech (paper window)" \
        --paper_central \
        --output_csv outputs/method_biomech_paper.eval.csv
fi

echo ""
echo "Done. End: $(date)"
echo ""
echo "To re-evaluate later without re-fitting:"
echo "  python3 evaluation/eval_mpvpe_regions.py \\"
echo "    --methods method_biomech_paper \\"
echo "    --method_names 'DexAvatar-Biomech (paper window)' \\"
echo "    --paper_central \\"
echo "    --output_csv outputs/method_biomech_paper.eval.csv"
