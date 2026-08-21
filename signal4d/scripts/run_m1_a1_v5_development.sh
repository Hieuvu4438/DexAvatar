#!/usr/bin/env bash
set -euo pipefail

project="/home/haipd/DexAvatar"
model="${project}/SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz"
cache="signal4d/artifacts/cache/sgnify_smplerx_wilor_a1_leftmirror_v5_all"
calibration="signal4d/artifacts/calibration/sgnify_a1_leftmirror_v5_seed12345/metrics.json"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${project}"
while [[ ! -f "${calibration}" ]]; do
    sleep 5
done

for split in dev test; do
    if [[ "${split}" == "dev" ]]; then
        manifest="signal4d/artifacts/manifests/frozen_seed_20260819_v2/sgnify_development.jsonl"
        gt_cache="signal4d/artifacts/gt_cache/calibration_development"
    else
        manifest="signal4d/artifacts/manifests/frozen_seed_20260819_v2/sgnify_test.jsonl"
        gt_cache="signal4d/artifacts/gt_cache/test_frozen_20260819"
    fi
    output="signal4d/runs/m1_a1_v5_${split}"
    conda run -n signal4d signal4d fit-smplx \
        --config signal4d/configs/method/m1_a1_v5.yaml \
        --manifest "${manifest}" \
        --cache-root "${cache}" \
        --output-root "${output}" \
        --model-path "${model}" \
        --device cuda
    conda run -n signal4d signal4d evaluate-sgnify \
        --manifest "${manifest}" \
        --predictions "${output}/predictions" \
        --gt-root data/smplx_gt \
        --gt-cache-root "${gt_cache}" \
        --model-path "${model}" \
        --upper-indices dexavatar_fitting/assets/smplx_upper_body_vidx.npy \
        --left-indices dexavatar_fitting/assets/smplx_left_hand_vidx.npy \
        --right-indices dexavatar_fitting/assets/smplx_right_hand_vidx.npy \
        --output "${output}/evaluation"
done
