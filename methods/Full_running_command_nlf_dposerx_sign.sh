#!/bin/bash
# Pipeline orchestrator for NLF + WiLoR + VQVAE-hand + SIGN-TRAINED DPoser-X body prior.
# (Variant of Full_running_command_nlf_vqvae_dposerx.sh; old script unchanged.)
# Reuses existing pre-extracted Sapiens and WiLoR stages via symbolic links.
set -e

SIGN_NAME=$(basename "${ROOT_PATH}")
SHARED_SIGN="/home/haipd/DexAvatar/outputs/shared/${SIGN_NAME}"

echo "Processing sign: ${SIGN_NAME}"
echo "=========================================="

mkdir -p "${OUTPUT_PATH}"

echo "Linking shared pre-extracted stages..."
for item in sapiens_1b sapiens.pkl wilor mean_shape_smplx.npy gender.txt hamer; do
    if [ -e "${SHARED_SIGN}/${item}" ] && [ ! -e "${OUTPUT_PATH}/${item}" ]; then
        ln -sf "${SHARED_SIGN}/${item}" "${OUTPUT_PATH}/${item}"
    fi
done

if [ ! -d "${OUTPUT_PATH}/nlf/smplx" ] || [ -z "$(ls -A "${OUTPUT_PATH}/nlf/smplx" 2>/dev/null)" ]; then
    echo "Running NLF body extraction..."
    bash -c "source scripts/config_nlf.sh && bash scripts/S1_nlf_extract.sh"
else
    echo "NLF body extraction already exists, skipping."
fi

echo "Running Stage 4 fitting (sign DPoser-X body, active + annealed timestep)..."
bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose_nlf_dposerx_sign.sh"

echo "Sign ${SIGN_NAME} processing complete."
