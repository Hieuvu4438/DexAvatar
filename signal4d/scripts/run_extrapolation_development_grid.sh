#!/usr/bin/env bash
set -euo pipefail

alpha="${1:?usage: run_extrapolation_development_grid.sh ALPHA}"
tag="${alpha//./p}"
project="/home/haipd/DexAvatar"
model="${project}/SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz"
upper="${project}/dexavatar_fitting/assets/smplx_upper_body_vidx.npy"
left="${project}/dexavatar_fitting/assets/smplx_left_hand_vidx.npy"
right="${project}/dexavatar_fitting/assets/smplx_right_hand_vidx.npy"
cache="${project}/signal4d/artifacts/cache/sgnify_smplerx_wilor_leftmirror_v3_all"

cd "${project}"
for split in dev test; do
    if [[ "${split}" == "dev" ]]; then
        manifest="signal4d/artifacts/manifests/frozen_seed_20260819_v2/sgnify_development.jsonl"
        candidate="signal4d/runs/dev_v3_m1/predictions"
        baseline="signal4d/runs/dev_v3_legacy_full_fallback/predictions"
        gt_cache="signal4d/artifacts/gt_cache/calibration_development"
    else
        manifest="signal4d/artifacts/manifests/frozen_seed_20260819_v2/sgnify_test.jsonl"
        candidate="signal4d/runs/test_v3_20260819_m1/predictions"
        baseline="signal4d/runs/test_v3_20260819_legacy_full_fallback/predictions"
        gt_cache="signal4d/artifacts/gt_cache/test_frozen_20260819"
    fi
    output="signal4d/runs/extrapolate_alpha_${tag}_${split}"
    conda run -n signal4d signal4d extrapolate \
        --manifest "${manifest}" \
        --candidate-root "${candidate}" \
        --baseline-root "${baseline}" \
        --cache-root "${cache}" \
        --model-path "${model}" \
        --alpha "${alpha}" \
        --output-root "${output}" \
        --device cpu
    conda run -n signal4d signal4d evaluate-sgnify \
        --manifest "${manifest}" \
        --predictions "${output}/predictions" \
        --gt-root data/smplx_gt \
        --gt-cache-root "${gt_cache}" \
        --model-path "${model}" \
        --upper-indices "${upper}" \
        --left-indices "${left}" \
        --right-indices "${right}" \
        --output "${output}/evaluation"
done
