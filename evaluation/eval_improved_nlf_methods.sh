#!/bin/bash
# Run the paper-matched TR-V2V evaluator for the two improved NLF methods.
set -euo pipefail

REPO=/home/haipd/DexAvatar
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate dexavatar

for method in method_nlf_wilor method_nlf_dposerx_hposer; do
    python "${REPO}/evaluation/evaluation_mpvpe_correct.py" \
        --pred_root "${REPO}/outputs/${method}" \
        --gt_root "${REPO}/data/smplx_gt" \
        --signs_txt "${REPO}/data/signs.txt" \
        --segment_json "${REPO}/data/segment.json" \
        --ubody_indices "${REPO}/dexavatar_fitting/assets/smplx_upper_body_minus_face_vidx.npy" \
        --lhand_indices "${REPO}/dexavatar_fitting/assets/smplx_left_hand_vidx.npy" \
        --rhand_indices "${REPO}/dexavatar_fitting/assets/smplx_right_hand_vidx.npy" \
        --method_name "${method}" \
        --central_frames \
        --output_csv "${REPO}/outputs/${method}/trv2v_frames_central.csv" \
        --output_summary "${REPO}/outputs/${method}/trv2v_summary_central.csv"

    python "${REPO}/evaluation/evaluate_active_hands.py" \
        --csv "${REPO}/outputs/${method}/trv2v_frames_central.csv" \
        --signs_txt "${REPO}/data/signs.txt"
done
