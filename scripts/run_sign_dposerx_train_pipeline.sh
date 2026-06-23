#!/bin/bash
# Master orchestration: produce the sign-trained DPoser-X body prior end-to-end.
# Runs steps 1->6 in order (extract PHOENIX -> merge -> convert -> normalize ->
# prep fine-tune ckpt -> fine-tune). All GPU work is sequential on GPU 0.
#
# Run inside tmux so it survives disconnect:
#   tmux new -s sign_dposerx -d "bash scripts/run_sign_dposerx_train_pipeline.sh"
# Tail the log:
#   tail -f logs/sign_dposerx_pipeline.log
set -eo pipefail

REPO=/home/haipd/DexAvatar
cd "$REPO"
mkdir -p logs
LOG=logs/sign_dposerx_pipeline.log
DONE=logs/sign_dposerx_pipeline.done
FAILED=logs/sign_dposerx_pipeline.failed

# Reset sentinels from any prior run.
rm -f "$DONE" "$FAILED"

# All stdout/stderr -> log file AND console (tmux capture).
exec > >(tee -a "$LOG") 2>&1

# Failure trap: write the fail sentinel so the watcher can notify.
trap 'echo "[$(date)] PIPELINE FAILED at step"; touch "$FAILED"' ERR

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate dexavatar

echo "=========================================================="
echo "[$(date)] Sign-DPoser-X training pipeline START"
echo "GPU:"; nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
echo "=========================================================="

echo "===== STEP 1/6: extract PHOENIX-2014-T body poses (subset, 3 frames/clip, GPU0) ====="
python scripts/extract_phoenix_sign.py --frames_per_clip 3 --gpu_id 0

echo "===== STEP 2/6: merge How2Sign + PHOENIX (cap PHX 10x) ====="
python scripts/merge_h2s_phoenix_for_dposerx.py --phx_cap_mult 10

echo "===== STEP 3/6: convert -> DPoser-X layout (sign_v1) ====="
python scripts/convert_sign_to_dposerx_layout.py

echo "===== STEP 4/6: fit sign normalizer (min/max) -> both consumers ====="
python scripts/fit_sign_normalizer.py

echo "===== STEP 5/6: prep fine-tune init ckpt (reset global_step=0) ====="
python scripts/prep_dposerx_finetune_ckpt.py

echo "===== STEP 6/6: fine-tune DPoser-X body prior on sign data (GPU0) ====="
bash scripts/train_dposerx_sign_body.sh

echo "=========================================================="
echo "[$(date)] Sign-DPoser-X training pipeline DONE."
echo "Checkpoint dir: ${REPO}/DPoser-X/checkpoints/dposer/sign/sign_body_ft"
echo "Next: fit+eval -> python methods/run_dexavatar_nlf_dposerx_sign.py --input_img_folder data/frames"
echo "=========================================================="
touch "$DONE"
