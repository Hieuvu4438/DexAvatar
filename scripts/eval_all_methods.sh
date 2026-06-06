#!/bin/bash
# Evaluate all 3 methods (Hand2D, Biomech, Ensemble)
set -eo pipefail

# Activate conda environment
set +u
CONDA_PATH=$(conda info --base)
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate dexavatar
set -u

PROJECT_DIR="/home/haipd/DexAvatar"
cd "${PROJECT_DIR}"

echo "=========================================="
echo "Evaluating all 3 methods"
echo "=========================================="
echo "Start time: $(date)"

# Method 1: Hand2D
echo ""
echo "=== Evaluating Hand2D ==="
python evaluation/evaluation_mpvpe_correct.py \
    --pred_root /home/haipd/DexAvatar/outputs/method_hand2d \
    --gt_root /home/haipd/DexAvatar/data/smplx_gt \
    --signs_txt data/signs.txt \
    --segment_json data/segment.json \
    --ubody_indices dexavatar_fitting/assets/smplx_upper_body_minus_face_vidx.npy \
    --lhand_indices dexavatar_fitting/assets/smplx_left_hand_vidx.npy \
    --rhand_indices dexavatar_fitting/assets/smplx_right_hand_vidx.npy \
    --method_name DexAvatar-Hand2D \
    --output_csv /home/haipd/DexAvatar/outputs/method_hand2d/hand2d_trv2v_frames.csv \
    --output_summary /home/haipd/DexAvatar/outputs/method_hand2d/hand2d_trv2v_summary.csv

echo ""
echo "Hand2D Summary:"
cat /home/haipd/DexAvatar/outputs/method_hand2d/hand2d_trv2v_summary.csv

# Method 2: Biomech
echo ""
echo "=== Evaluating Biomech ==="
python evaluation/evaluation_mpvpe_correct.py \
    --pred_root /home/haipd/DexAvatar/outputs/method_biomech \
    --gt_root /home/haipd/DexAvatar/data/smplx_gt \
    --signs_txt data/signs.txt \
    --segment_json data/segment.json \
    --ubody_indices dexavatar_fitting/assets/smplx_upper_body_minus_face_vidx.npy \
    --lhand_indices dexavatar_fitting/assets/smplx_left_hand_vidx.npy \
    --rhand_indices dexavatar_fitting/assets/smplx_right_hand_vidx.npy \
    --method_name DexAvatar-Biomech \
    --output_csv /home/haipd/DexAvatar/outputs/method_biomech/biomech_trv2v_frames.csv \
    --output_summary /home/haipd/DexAvatar/outputs/method_biomech/biomech_trv2v_summary.csv

echo ""
echo "Biomech Summary:"
cat /home/haipd/DexAvatar/outputs/method_biomech/biomech_trv2v_summary.csv

# Method 3: Ensemble
echo ""
echo "=== Evaluating Ensemble ==="
python evaluation/evaluation_mpvpe_correct.py \
    --pred_root /home/haipd/DexAvatar/outputs/method_ensemble \
    --gt_root /home/haipd/DexAvatar/data/smplx_gt \
    --signs_txt data/signs.txt \
    --segment_json data/segment.json \
    --ubody_indices dexavatar_fitting/assets/smplx_upper_body_minus_face_vidx.npy \
    --lhand_indices dexavatar_fitting/assets/smplx_left_hand_vidx.npy \
    --rhand_indices dexavatar_fitting/assets/smplx_right_hand_vidx.npy \
    --method_name DexAvatar-Ensemble \
    --output_csv /home/haipd/DexAvatar/outputs/method_ensemble/ensemble_trv2v_frames.csv \
    --output_summary /home/haipd/DexAvatar/outputs/method_ensemble/ensemble_trv2v_summary.csv

echo ""
echo "Ensemble Summary:"
cat /home/haipd/DexAvatar/outputs/method_ensemble/ensemble_trv2v_summary.csv

echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETE"
echo "=========================================="
echo "End time: $(date)"
