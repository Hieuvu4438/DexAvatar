#!/bin/bash
# Stage 4: SMPLify-X pose fitting with NLF init, SignHPoser hands, DPoser-X body.
set -e

# Resolve checkpoint paths (allow override via env vars).
DPOSERX_CKPT=${DPOSERX_CKPT:-/home/haipd/DexAvatar/checkpoints/dposerx_body/body.ckpt}
DPOSERX_CONFIG=${DPOSERX_CONFIG:-/home/haipd/DexAvatar/DPoser-X/configs/body/subvp/timefc.py}
# IMPORTANT: Use the AMASS body normalizer (DPoser-X was trained on AMASS).
# The sign-language normalizer was incorrectly computed and maps poses to
# extreme values [-8, 17] instead of [-1, 1], causing NaN gradients.
DPOSERX_NORMALIZER=${DPOSERX_NORMALIZER:-/home/haipd/DexAvatar/DPoser-X/data/body_data/body_normalizer}

# Check that required checkpoints exist.
if [ ! -f "$DPOSERX_CKPT" ]; then
    echo "ERROR: DPoser-X ckpt not found: $DPOSERX_CKPT"
    exit 1
fi
if [ ! -d "$DPOSERX_NORMALIZER" ]; then
    echo "ERROR: DPoser-X normalizer dir not found: $DPOSERX_NORMALIZER"
    exit 1
fi

# cd to dexavatar_fitting so relative paths resolve correctly.
cd /home/haipd/DexAvatar/dexavatar_fitting
export PYTHONPATH=$(pwd)/smplifyx:$(pwd):${PYTHONPATH:-}

python smplifyx/main.py \
    --config cfg_files/fit_smplx_vposer_x_dposerx.yaml \
    --data_folder ${OUTPUT_PATH} \
    --output_folder ${OUTPUT_PATH}/smplifyx \
    --img_folder ${ROOT_PATH} \
    --model_folder ${SMPLX_MODEL_DIR:-../SMPLer-X/common/utils/human_model_files} \
    --visualize False \
    --split_num 1 --cur_num 0 \
    --smplx_init_dir nlf/smplx \
    --use_hposer3d True \
    --use_dposerx_body False \
    --use_dposerx_refine True \
    --use_signbposer False \
    --use_vqvae_hand False \
    --use_motionbert_prior False \
    --use_phd_prior False \
    --dposerx_ckpt ${DPOSERX_CKPT} \
    --dposerx_config ${DPOSERX_CONFIG} \
    --dposerx_normalizer_dir ${DPOSERX_NORMALIZER}
