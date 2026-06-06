#!/bin/bash
# Evaluation for Method 1: Temporal Sliding Window
cd /home/haipd/DexAvatar

conda activate dexavatar 2>/dev/null || true

python evaluation/evaluation_trv2v_wilor.py \
    --pred_root /home/haipd/DexAvatar/outputs/output_wilor_temporal \
    --gt_root /home/haipd/DexAvatar/data/smplx_gt \
    --signs_txt data/signs.txt \
    --segment_json data/segment.json \
    --ubody_indices dexavatar_fitting/assets/smplx_upper_body_minus_face_vidx.npy \
    --lhand_indices dexavatar_fitting/assets/smplx_left_hand_vidx.npy \
    --rhand_indices dexavatar_fitting/assets/smplx_right_hand_vidx.npy \
    --method_name DexAvatar-WiLoR-Temporal \
    --output_csv /home/haipd/DexAvatar/outputs/output_wilor_temporal/temporal_trv2v_frames_full.csv \
    --output_summary /home/haipd/DexAvatar/outputs/output_wilor_temporal/temporal_trv2v_summary_full.csv

echo "Method 1 Evaluation completed!"
