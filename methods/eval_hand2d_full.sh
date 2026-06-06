#!/bin/bash
# Evaluation for Method 2: 2D Hand Supervision
cd /home/haipd/DexAvatar

conda activate dexavatar 2>/dev/null || true

python evaluation/evaluation_trv2v_wilor.py \
    --pred_root /home/haipd/DexAvatar/outputs/output_wilor_hand2d \
    --gt_root /home/haipd/DexAvatar/data/smplx_gt \
    --signs_txt data/signs.txt \
    --segment_json data/segment.json \
    --ubody_indices dexavatar_fitting/assets/smplx_upper_body_minus_face_vidx.npy \
    --lhand_indices dexavatar_fitting/assets/smplx_left_hand_vidx.npy \
    --rhand_indices dexavatar_fitting/assets/smplx_right_hand_vidx.npy \
    --method_name DexAvatar-WiLoR-Hand2D \
    --output_csv /home/haipd/DexAvatar/outputs/output_wilor_hand2d/hand2d_trv2v_frames_full.csv \
    --output_summary /home/haipd/DexAvatar/outputs/output_wilor_hand2d/hand2d_trv2v_summary_full.csv

echo "Method 2 Evaluation completed!"
