#!/bin/bash
# Evaluation for full 57 signs with WiLoR pipeline
cd /home/haipd/DexAvatar

conda activate dexavatar 2>/dev/null || true

python evaluation/evaluation_mpvpe_correct.py \
    --pred_root /home/haipd/DexAvatar/outputs/output_wilor \
    --gt_root /home/haipd/DexAvatar/data/smplx_gt \
    --signs_txt data/signs.txt \
    --segment_json data/segment.json \
    --ubody_indices dexavatar_fitting/assets/smplx_upper_body_minus_face_vidx.npy \
    --lhand_indices dexavatar_fitting/assets/smplx_left_hand_vidx.npy \
    --rhand_indices dexavatar_fitting/assets/smplx_right_hand_vidx.npy \
    --method_name DexAvatar-WiLoR \
    --output_csv /home/haipd/DexAvatar/outputs/output_wilor/wilor_trv2v_frames_full.csv \
    --output_summary /home/haipd/DexAvatar/outputs/output_wilor/wilor_trv2v_summary_full.csv

echo "Evaluation completed!"
