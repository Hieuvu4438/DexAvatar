#!/usr/bin/env bash
set -euo pipefail

split="${1:?usage: run_multigate_development.sh dev|test}"
project="/home/haipd/DexAvatar"
model="${project}/SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz"
upper="${project}/dexavatar_fitting/assets/smplx_upper_body_vidx.npy"
left="${project}/dexavatar_fitting/assets/smplx_left_hand_vidx.npy"
right="${project}/dexavatar_fitting/assets/smplx_right_hand_vidx.npy"
cache="signal4d/artifacts/cache/sgnify_smplerx_wilor_leftmirror_v3_all"
cd "${project}"

if [[ "${split}" == "dev" ]]; then
    manifest="signal4d/artifacts/manifests/frozen_seed_20260819_v2/sgnify_development.jsonl"
    baseline="signal4d/runs/dev_v3_legacy_full_fallback/predictions"
    alpha1="signal4d/runs/dev_v3_m1/predictions"
    gt_cache="signal4d/artifacts/gt_cache/calibration_development"
elif [[ "${split}" == "test" ]]; then
    manifest="signal4d/artifacts/manifests/frozen_seed_20260819_v2/sgnify_test.jsonl"
    baseline="signal4d/runs/test_v3_20260819_legacy_full_fallback/predictions"
    alpha1="signal4d/runs/test_v3_20260819_m1/predictions"
    gt_cache="signal4d/artifacts/gt_cache/test_frozen_20260819"
else
    echo "split must be dev or test" >&2
    exit 2
fi
output="signal4d/runs/multiscale_v4_${split}"
conda run -n signal4d signal4d apply-multigate \
    --manifest "${manifest}" \
    --baseline-root "${baseline}" \
    --cache-root "${cache}" \
    --bundle signal4d/artifacts/gating/multiscale_v4 \
    --hypothesis "m1_alpha_1p0=${alpha1}" \
    --hypothesis "m1_alpha_1p5=signal4d/runs/extrapolate_alpha_1p5_${split}/predictions" \
    --hypothesis "m1_alpha_3p0=signal4d/runs/extrapolate_alpha_3p0_${split}/predictions" \
    --output-root "${output}"
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
