#!/bin/bash
# Stage 4: SMPLify-X pose fitting — v2 of the NLF + VQVAE-hand + SIGN-TRAINED
# DPoser-X body pipeline.
#
# v2 differences vs M4_smplifyx_pose_nlf_dposerx_sign.sh (old script untouched):
#   * uses the rebalanced v2 config (prior/data weights + optim_global_orient)
#   * passes --optim_global_orient True (root orientation is optimized)
# (Variant of M4_smplifyx_pose_nlf_dposerx_sign.sh; old script unchanged.)
set -e

VQVAE_CKPT=${VQVAE_CKPT:-/home/haipd/DexAvatar/checkpoints/vqvae_hand/signhposer_vqvae/last.ckpt}
DPOSERX_CKPT=${DPOSERX_CKPT:-/home/haipd/DexAvatar/DPoser-X/checkpoints/dposer/sign/sign_body_ft/last.ckpt}
DPOSERX_CONFIG=${DPOSERX_CONFIG:-/home/haipd/DexAvatar/DPoser-X/configs/body/subvp/timefc.py}
# Sign normalizer (min/max) -- MUST match the sign-trained checkpoint.
DPOSERX_NORMALIZER=${DPOSERX_NORMALIZER:-/home/haipd/DexAvatar/checkpoints/dposerx_body_sign/body_normalizer}

if [ ! -f "$VQVAE_CKPT" ]; then echo "WARNING: VQVAE ckpt not found: $VQVAE_CKPT"; fi
if [ ! -f "$DPOSERX_CKPT" ]; then
    echo "WARNING: sign DPoser-X ckpt not found: $DPOSERX_CKPT"
    echo "         Train it with scripts/train_dposerx_sign_body.sh first."
fi
if [ ! -d "$DPOSERX_NORMALIZER" ]; then echo "WARNING: sign normalizer dir not found: $DPOSERX_NORMALIZER"; fi

cd /home/haipd/DexAvatar/dexavatar_fitting
export PYTHONPATH=$(pwd)/smplifyx:$(pwd):${PYTHONPATH:-}

python smplifyx/main.py \
    --config cfg_files/fit_smplx_vposer_x_dposerx_sign_v2.yaml \
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
