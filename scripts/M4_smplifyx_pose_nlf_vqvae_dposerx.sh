#!/bin/bash
# Stage 4: SMPLify-X pose fitting with NLF initialization, and VQVAE + DPoser-X priors.
set -e

# Resolve checkpoint paths (allow override via env vars).
VQVAE_CKPT=${VQVAE_CKPT:-/home/haipd/DexAvatar/checkpoints/vqvae_hand/signhposer_vqvae/last.ckpt}
DPOSERX_CKPT=${DPOSERX_CKPT:-/home/haipd/DexAvatar/checkpoints/dposerx_body/body.ckpt}
DPOSERX_CONFIG=${DPOSERX_CONFIG:-/home/haipd/DexAvatar/DPoser-X/configs/body/subvp/timefc.py}
# IMPORTANT: Use the AMASS body normalizer (DPoser-X was trained on AMASS).
DPOSERX_NORMALIZER=${DPOSERX_NORMALIZER:-/home/haipd/DexAvatar/DPoser-X/data/body_data/body_normalizer}

# Check that required checkpoints exist (warn but don't abort on missing).
if [ ! -f "$VQVAE_CKPT" ]; then
    echo "WARNING: VQVAE ckpt not found: $VQVAE_CKPT"
fi
if [ ! -f "$DPOSERX_CKPT" ]; then
    echo "WARNING: DPoser-X ckpt not found: $DPOSERX_CKPT"
fi
if [ ! -d "$DPOSERX_NORMALIZER" ]; then
    echo "WARNING: DPoser-X normalizer dir not found: $DPOSERX_NORMALIZER"
fi

# We must cd to dexavatar_fitting so that relative paths like ../SMPLer-X resolve correctly.
cd /home/haipd/DexAvatar/dexavatar_fitting
export PYTHONPATH=$(pwd)/smplifyx:$(pwd):${PYTHONPATH:-}

python smplifyx/main.py \
    --config cfg_files/fit_smplx_vposer_x_vqvae_dposerx.yaml \
    --data_folder ${OUTPUT_PATH} \
    --output_folder ${OUTPUT_PATH}/smplifyx \
    --img_folder ${ROOT_PATH} \
    --model_folder ${SMPLX_MODEL_DIR:-../SMPLer-X/common/utils/human_model_files} \
    --visualize False \
    --split_num 1 --cur_num 0 \
    --smplx_init_dir nlf/smplx \
    --vqvae_hand_ckpt ${VQVAE_CKPT} \
    --dposerx_ckpt ${DPOSERX_CKPT} \
    --dposerx_config ${DPOSERX_CONFIG} \
    --dposerx_normalizer_dir ${DPOSERX_NORMALIZER} \
    --use_vqvae_hand True \
    --use_dposerx_body True
