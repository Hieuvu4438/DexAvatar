#!/bin/bash
# Shared Pipeline: Run Sapiens + SMPLer-X + WiLoR/HaMeR ONCE
# Output can be shared across all methods (Hand2D, Biomech, Ensemble, Temporal)
set -euo pipefail

SIGN_NAME=$(basename "${ROOT_PATH}")
echo "=========================================="
echo "Shared Extract: ${SIGN_NAME}"
echo "=========================================="

# Stage 1: Sapiens pose extraction
bash -c "source scripts/config_sapiens.sh && bash scripts/S1_sapiens_extract.sh"
wait
echo "Aggregating Sapiens outputs into sapiens.pkl..."
python scripts/aggregate_sapiens.py --sapiens_dir ${OUTPUT_PATH}/sapiens_1b --output_path ${OUTPUT_PATH} --subfolder ${SIGN_NAME}
wait

# Stage 2: SMPLer-X body estimation
bash -c "source scripts/config_smplerx.sh && bash scripts/S1_smplerx_extract.sh"
wait

# Stage 3: WiLoR/HaMeR hand extraction
bash -c "source scripts/config.sh && bash scripts/M3.5_wilor_extract.sh"
wait

echo "Shared extraction done for ${SIGN_NAME}"
