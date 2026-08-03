#!/usr/bin/env bash
set -euo pipefail

: "${ROOT_PATH:?ROOT_PATH is required}"
: "${OUTPUT_PATH:?OUTPUT_PATH is required}"

SIGN_NAME="$(basename "${ROOT_PATH}")"

bash -c "source scripts/config_sapiens.sh && bash scripts/S1_sapiens_extract.sh"
python scripts/aggregate_sapiens.py \
  --sapiens_dir "${OUTPUT_PATH}/sapiens_1b" \
  --output_path "${OUTPUT_PATH}" \
  --subfolder "${SIGN_NAME}"
bash -c "source scripts/config_smplerx.sh && bash scripts/S1_smplerx_extract.sh"
bash -c "source scripts/config.sh && bash scripts/M3.5_wilor_extract.sh"
bash -c "source scripts/config.sh && bash scripts/M3_ensemble_init.sh"
