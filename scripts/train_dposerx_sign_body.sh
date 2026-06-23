#!/bin/bash
# Fine-tune the DPoser-X body prior on merged How2Sign + PHOENIX-2014-T sign data.
#
# Single GPU (RTX 5880 Ada), low CPU (num_workers=2 in the sign config).
# Initializes weights from the released AMASS body checkpoint (via the
# weights-only init ckpt produced by prep_dposerx_finetune_ckpt.py) and trains a
# fresh 30k-step schedule.
#
# Prereqs (run before this):
#   1. python scripts/extract_phoenix_sign.py --frames_per_clip 3 --gpu_id 0
#   2. python scripts/merge_h2s_phoenix_for_dposerx.py --phx_cap_mult 10
#   3. python scripts/convert_sign_to_dposerx_layout.py
#   4. python scripts/fit_sign_normalizer.py
#   5. python scripts/prep_dposerx_finetune_ckpt.py
set -e

REPO=/home/haipd/DexAvatar
CKPT_DIR="${REPO}/DPoser-X/checkpoints/dposer/sign/sign_body_ft"
INIT_CKPT="${CKPT_DIR}/sign_init.ckpt"

if [ ! -f "${INIT_CKPT}" ]; then
    echo "ERROR: init ckpt not found at ${INIT_CKPT}"
    echo "       Run: python scripts/prep_dposerx_finetune_ckpt.py"
    exit 1
fi
if [ ! -f "${REPO}/data/body_data/sign_v1/train/pose_body.pt" ]; then
    echo "ERROR: sign_v1 data not found. Run convert_sign_to_dposerx_layout.py first."
    exit 1
fi

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Use the DPoser-X environment (same one that runs its trainer). Adjust if yours
# differs; the trainer needs torch + pytorch-lightning + ml_collections.
conda activate dexavatar 2>/dev/null || true

cd "${REPO}/DPoser-X"

# Pin to GPU 0; devices=[0] in the sign config will then be the single visible GPU.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python -m run.trainer.body.diffusion \
    -c configs.body.subvp.sign_timefc.get_config \
    --data-root ../data/body_data \
    --version sign_v1 \
    --bodymodel-path /home/haipd/DexAvatar/SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz \
    --resume-ckpt sign_init.ckpt \
    --name sign_body_ft

echo "Training launched. Checkpoints: ${CKPT_DIR}"
echo "TensorBoard:  tensorboard --logdir ${REPO}/DPoser-X/logs/dposer/sign"
