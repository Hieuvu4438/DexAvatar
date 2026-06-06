#!/bin/bash
# Link shared extraction data to method directories
# Usage: bash scripts/link_shared_to_methods.sh [method1 method2 ...]
# Example: bash scripts/link_shared_to_methods.sh hand2d biomech ensemble

set -euo pipefail

PROJECT_DIR="/home/haipd/DexAvatar"
SHARED_DIR="${PROJECT_DIR}/outputs/shared"
INPUT_DIR="${PROJECT_DIR}/data/frames"

# Default: all methods
if [ $# -eq 0 ]; then
    METHODS=("hand2d" "biomech" "ensemble")
else
    METHODS=("$@")
fi

echo "Linking shared data to method directories..."
echo "Shared dir: ${SHARED_DIR}"
echo "Methods: ${METHODS[*]}"

for METHOD in "${METHODS[@]}"; do
    METHOD_DIR="${PROJECT_DIR}/outputs/method_${METHOD}"
    echo ""
    echo "=== Setting up ${METHOD} ==="

    LINKED=0
    SKIPPED=0

    for SIGN_DIR in ${INPUT_DIR}/*/; do
        SIGN_NAME=$(basename "$SIGN_DIR")
        SHARED_SIGN="${SHARED_DIR}/${SIGN_NAME}"
        METHOD_SIGN="${METHOD_DIR}/${SIGN_NAME}"

        # Check if shared data exists
        if [ ! -d "${SHARED_SIGN}" ]; then
            echo "[WARN] No shared data for ${SIGN_NAME}, skipping"
            continue
        fi

        mkdir -p "${METHOD_SIGN}"

        # Link shared stages (sapiens, smplerx, wilor, hamer, metadata)
        for item in sapiens_1b sapiens.pkl smplerx hamer wilor mean_shape_smplx.npy gender.txt; do
            if [ -e "${SHARED_SIGN}/${item}" ] && [ ! -e "${METHOD_SIGN}/${item}" ]; then
                ln -sf "${SHARED_SIGN}/${item}" "${METHOD_SIGN}/${item}"
                LINKED=$((LINKED + 1))
            elif [ -e "${METHOD_SIGN}/${item}" ]; then
                SKIPPED=$((SKIPPED + 1))
            fi
        done
    done

    echo "[DONE] ${METHOD}: ${LINKED} links created, ${SKIPPED} already exist"
done

echo ""
echo "=== All methods linked ==="
