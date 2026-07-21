#!/bin/bash
# Pipeline orchestrator — v2 of the NLF + WiLoR + VQVAE-hand + SIGN-TRAINED
# DPoser-X body prior pipeline.
#
# v2 differences vs Full_running_command_nlf_dposerx_sign.sh (old script untouched):
#   * avoids re-running NLF by COPYING nlf/smplx from the reference method
#     (outputs/method_nlf_dposerx_sign/<SIGN>) when available, so the reference
#     output is never mutated (cp -rL dereferences any symlinks).
#   * runs the existing scripts/smooth_nlf_init.py IN-PLACE on v2's own copy to
#     remove per-frame transl/global_orient outlier jumps (depth/scale jitter).
#   * calls the v2 M4 script (rebalanced config + optim_global_orient).
set -e

SIGN_NAME=$(basename "${ROOT_PATH}")
SHARED_SIGN="/home/haipd/DexAvatar/outputs/shared/${SIGN_NAME}"
REF_NLF="/home/haipd/DexAvatar/outputs/method_nlf_dposerx_sign/${SIGN_NAME}/nlf/smplx"

echo "Processing sign (v2): ${SIGN_NAME}"
echo "=========================================="

mkdir -p "${OUTPUT_PATH}"

echo "Linking shared pre-extracted stages..."
for item in sapiens_1b sapiens.pkl wilor mean_shape_smplx.npy gender.txt hamer; do
    if [ -e "${SHARED_SIGN}/${item}" ] && [ ! -e "${OUTPUT_PATH}/${item}" ]; then
        ln -sf "${SHARED_SIGN}/${item}" "${OUTPUT_PATH}/${item}"
    fi
done

# Populate v2's own NLF init (real copy, never a symlink to shared/reference).
if [ ! -d "${OUTPUT_PATH}/nlf/smplx" ] || [ -z "$(ls -A "${OUTPUT_PATH}/nlf/smplx" 2>/dev/null)" ]; then
    if [ -d "${REF_NLF}" ] && [ -n "$(ls -A "${REF_NLF}" 2>/dev/null)" ]; then
        echo "Copying NLF body extraction from reference method (${REF_NLF})..."
        mkdir -p "${OUTPUT_PATH}/nlf"
        cp -rL "${REF_NLF}" "${OUTPUT_PATH}/nlf/smplx"
    else
        echo "No reference NLF found; running NLF body extraction..."
        bash -c "source scripts/config_nlf.sh && bash scripts/S1_nlf_extract.sh"
    fi
else
    echo "NLF body extraction already present in v2 output, keeping it."
fi

# Smooth per-frame transl/global_orient outlier jumps in v2's own copy (safe,
# idempotent; the reference method is untouched).
echo "Smoothing NLF init (transl / global_orient outliers)..."
bash -c "source scripts/config.sh 2>/dev/null; python /home/haipd/DexAvatar/scripts/smooth_nlf_init.py --pkl_dir '${OUTPUT_PATH}/nlf/smplx' || echo 'smooth_nlf_init.py skipped (non-fatal)'"

echo "Running Stage 4 fitting (v2: rebalanced prior/data + optim_global_orient)..."
bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose_nlf_dposerx_sign_v2.sh"

echo "Sign ${SIGN_NAME} (v2) processing complete."
