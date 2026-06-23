#!/bin/bash
# Stage 1 (NLF body extraction).
# Mirrors scripts/S1_smplerx_extract.sh but invokes the NLF adapter.
# The nlf conda env must be activated by the caller:
#     source scripts/config_nlf.sh && bash scripts/S1_nlf_extract.sh

set -euo pipefail

MODEL_PATH="${NLF_MODEL_PATH:-$DATA_ROOT/models/nlf_l_multi3.torchscript}"
SIGN_NAME=$(basename "${ROOT_PATH}")
SMPLERX_SHARED="${SHARED_SIGN:-/home/haipd/DexAvatar/outputs/shared/${SIGN_NAME}}/smplerx/smplx"
WILOR_PKL="${SHARED_SIGN:-/home/haipd/DexAvatar/outputs/shared/${SIGN_NAME}}/wilor/wilor.pkl"

python scripts/S1_nlf_adapter.py \
    --img_folder "${ROOT_PATH}" \
    --out_folder "${OUTPUT_PATH}" \
    --model_path "${MODEL_PATH}" \
    --batch_size "${NLF_BATCH_SIZE:-1}" \
    --default_focal "${NLF_DEFAULT_FOCAL:-5000.0}" \
    --smplerx_shared_dir "${SMPLERX_SHARED}" \
    --wilor_pkl "${WILOR_PKL}"
