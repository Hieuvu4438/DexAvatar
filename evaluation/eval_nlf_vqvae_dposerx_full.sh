#!/bin/bash
# Evaluation for full 57 signs with NLF + WiLoR + VQVAE + DPoser-X pipeline
cd /home/haipd/DexAvatar

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate dexavatar

echo "=========================================="
echo "Evaluating NLF + WiLoR + VQVAE + DPoser-X"
echo "=========================================="

# 1. Run correct evaluation script with central frames
python evaluation/evaluation_mpvpe_correct.py \
    --pred_root /home/haipd/DexAvatar/outputs/method_nlf_vqvae_dposerx \
    --gt_root /home/haipd/DexAvatar/data/smplx_gt \
    --signs_txt data/signs.txt \
    --segment_json data/segment.json \
    --ubody_indices dexavatar_fitting/assets/smplx_upper_body_minus_face_vidx.npy \
    --lhand_indices dexavatar_fitting/assets/smplx_left_hand_vidx.npy \
    --rhand_indices dexavatar_fitting/assets/smplx_right_hand_vidx.npy \
    --method_name DexAvatar-NLF-WiLoR-VQVAE-DPoserX \
    --central_frames \
    --output_csv /home/haipd/DexAvatar/outputs/method_nlf_vqvae_dposerx/trv2v_frames_central.csv \
    --output_summary /home/haipd/DexAvatar/outputs/method_nlf_vqvae_dposerx/trv2v_summary_central.csv

# 2. Run active-hand evaluation report
python evaluation/evaluate_active_hands.py \
    --csv /home/haipd/DexAvatar/outputs/method_nlf_vqvae_dposerx/trv2v_frames_central.csv \
    --signs_txt data/signs.txt

echo "Evaluation completed!"
