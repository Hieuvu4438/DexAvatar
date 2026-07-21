#!/bin/bash
# Evaluation for the v2 NLF + WiLoR + VQVAE-hand + SIGN-TRAINED DPoser-X pipeline
# (rebalanced prior/data + optim_global_orient). Same metrics as the v1 eval,
# pointed at the v2 prediction folder. Old eval script unchanged.
# Use after methods/run_dexavatar_nlf_dposerx_sign_v2.py has produced predictions.
cd /home/haipd/DexAvatar

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate dexavatar

PRED_ROOT=${PRED_ROOT:-/home/haipd/DexAvatar/outputs/method_nlf_dposerx_sign_v2}

echo "=========================================="
echo "Evaluating NLF + WiLoR + VQVAE + Sign-DPoserX (v2: rebalanced + optim_global_orient)"
echo "pred_root: ${PRED_ROOT}"
echo "=========================================="

python evaluation/evaluation_mpvpe_correct.py \
    --pred_root "${PRED_ROOT}" \
    --gt_root /home/haipd/DexAvatar/data/smplx_gt \
    --signs_txt data/signs.txt \
    --segment_json data/segment.json \
    --ubody_indices dexavatar_fitting/assets/smplx_upper_body_minus_face_vidx.npy \
    --lhand_indices dexavatar_fitting/assets/smplx_left_hand_vidx.npy \
    --rhand_indices dexavatar_fitting/assets/smplx_right_hand_vidx.npy \
    --method_name DexAvatar-NLF-WiLoR-VQVAE-SignDPoserX-v2 \
    --central_frames \
    --output_csv "${PRED_ROOT}/trv2v_frames_central.csv" \
    --output_summary "${PRED_ROOT}/trv2v_summary_central.csv"

python evaluation/evaluate_active_hands.py \
    --csv "${PRED_ROOT}/trv2v_frames_central.csv" \
    --signs_txt data/signs.txt

echo "Evaluation completed!"
