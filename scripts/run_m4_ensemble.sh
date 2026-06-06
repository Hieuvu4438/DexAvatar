#!/bin/bash
# M4 Ensemble Init: Pipeline + Auto Evaluation
cd /home/haipd/DexAvatar

echo "=========================================="
echo "M4: Multi-Model Ensemble Init Pipeline"
echo "=========================================="
echo "Start time: $(date)"

# Run pipeline
python methods/run_dexavatar_wilor_ensemble.py \
    --input_img_folder data/frames \
    --output_path outputs/method_ensemble \
    --fitting_experiment /home/haipd/DexAvatar/dexavatar_fitting

PIPELINE_EXIT=$?
echo "Pipeline finished at $(date) with exit code $PIPELINE_EXIT"

if [ $PIPELINE_EXIT -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Running TR-V2V Evaluation for M4 Ensemble"
    echo "=========================================="

    conda activate dexavatar 2>/dev/null || true

    python evaluation_trv2v_wilor.py \
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
    echo "=========================================="
    echo "M4 Ensemble Summary:"
    echo "=========================================="
    cat /home/haipd/DexAvatar/outputs/method_ensemble/ensemble_trv2v_summary.csv 2>/dev/null
else
    echo "Pipeline failed, skipping evaluation"
fi

echo ""
echo "=== M4 Ensemble ALL DONE ==="
echo "End time: $(date)"
