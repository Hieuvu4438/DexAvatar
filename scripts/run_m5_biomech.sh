#!/bin/bash
# M5 Sign-Specific Biomech: Pipeline + Auto Evaluation
cd /home/haipd/DexAvatar

echo "=========================================="
echo "M5: Sign-Specific Biomechanics Pipeline"
echo "=========================================="
echo "Start time: $(date)"

# Run pipeline
python methods/run_dexavatar_wilor_biomech.py \
    --input_img_folder data/frames \
    --output_path outputs/method_biomech \
    --fitting_experiment /home/haipd/DexAvatar/dexavatar_fitting

PIPELINE_EXIT=$?
echo "Pipeline finished at $(date) with exit code $PIPELINE_EXIT"

if [ $PIPELINE_EXIT -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Running TR-V2V Evaluation for M5 Biomech"
    echo "=========================================="

    conda activate dexavatar 2>/dev/null || true

    python evaluation/evaluation_trv2v_wilor.py \
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
    echo "=========================================="
    echo "M5 Biomech Summary:"
    echo "=========================================="
    cat /home/haipd/DexAvatar/outputs/method_biomech/biomech_trv2v_summary.csv 2>/dev/null
else
    echo "Pipeline failed, skipping evaluation"
fi

echo ""
echo "=== M5 Biomech ALL DONE ==="
echo "End time: $(date)"
