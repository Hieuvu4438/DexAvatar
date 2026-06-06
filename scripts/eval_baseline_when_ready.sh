#!/bin/bash
# Wait for baseline pipeline to finish, then run evaluation
BASELINE_PID=515339

echo "Waiting for baseline pipeline (PID $BASELINE_PID) to finish..."
while kill -0 $BASELINE_PID 2>/dev/null; do
    sleep 30
done

echo "Baseline pipeline finished. Starting evaluation..."
cd /home/haipd/DexAvatar

conda activate dexavatar 2>/dev/null || true

python evaluation_trv2v_wilor.py \
    --pred_root /home/haipd/DexAvatar/outputs/output_baseline \
    --gt_root /home/haipd/DexAvatar/data/smplx_gt \
    --signs_txt data/signs.txt \
    --segment_json data/segment.json \
    --ubody_indices dexavatar_fitting/assets/smplx_upper_body_minus_face_vidx.npy \
    --lhand_indices dexavatar_fitting/assets/smplx_left_hand_vidx.npy \
    --rhand_indices dexavatar_fitting/assets/smplx_right_hand_vidx.npy \
    --method_name DexAvatar-Baseline-WiLoR \
    --output_csv /home/haipd/DexAvatar/outputs/output_baseline/baseline_trv2v_frames_full.csv \
    --output_summary /home/haipd/DexAvatar/outputs/output_baseline/baseline_trv2v_summary_full.csv

echo ""
echo "=========================================="
echo "Baseline Summary:"
echo "=========================================="
cat /home/haipd/DexAvatar/outputs/output_baseline/baseline_trv2v_summary_full.csv 2>/dev/null

echo ""
echo "=== BASELINE EVAL ALL DONE ==="
echo "End time: $(date)"
