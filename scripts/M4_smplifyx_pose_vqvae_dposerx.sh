#!/bin/bash
# Stage 4: SMPLify-X pose fitting with VQVAE + DPoser-X priors.
# Mirrors scripts/M4_smplifyx_pose.sh but with the new flags set.
set -e

# Resolve checkpoint paths (allow override via env vars).
VQVAE_CKPT=${VQVAE_CKPT:-/home/haipd/DexAvatar/checkpoints/vqvae_hand/signhposer_vqvae/last.ckpt}
DPOSERX_CKPT=${DPOSERX_CKPT:-/home/haipd/DexAvatar/checkpoints/dposerx_body/body.ckpt}
DPOSERX_CONFIG=${DPOSERX_CONFIG:-/home/haipd/DexAvatar/DPoser-X/configs/body/subvp/timefc.py}
DPOSERX_NORMALIZER=${DPOSERX_NORMALIZER:-/home/haipd/DexAvatar/checkpoints/dposerx_body/body_normalizer}

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

# Split settings mirror the baseline (split_num, cur_num).
python dexavatar_fitting/smplifyx/main.py \
    --config dexavatar_fitting/cfg_files/fit_smplx_vposer_x_vqvae_dposerx.yaml \
    --data_folder ${ROOT_PATH} \
    --output_folder ${OUTPUT_PATH}/smplifyx \
    --img_folder ${ROOT_PATH} \
    --model_folder ${SMPLX_MODEL_DIR:-../SMPLer-X/common/utils/human_model_files} \
    --part_segm_fn dexavatar_fitting/assets/smplx_parts_segm.pkl \
    --visualize False \
    --split_num 1 --cur_num 0 \
    --vqvae_hand_ckpt ${VQVAE_CKPT} \
    --dposerx_ckpt ${DPOSERX_CKPT} \
    --dposerx_config ${DPOSERX_CONFIG} \
    --dposerx_normalizer_dir ${DPOSERX_NORMALIZER} \
    --use_vqvae_hand True \
    --use_dposerx_body True
