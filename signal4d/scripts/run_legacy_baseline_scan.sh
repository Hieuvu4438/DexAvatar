#!/usr/bin/env bash
set -euo pipefail

method="${1:?usage: run_legacy_baseline_scan.sh METHOD}"
project="/home/haipd/DexAvatar"
manifest="signal4d/artifacts/manifests/sgnify_available_15fps_development.jsonl"
model="${project}/SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz"
fallback="outputs/method_hamer"
case "${method}" in
    hamer) primary="outputs/method_hamer" ;;
    ensemble) primary="outputs/method_ensemble" ;;
    biomech) primary="outputs/method_biomech" ;;
    hand2d) primary="outputs/method_hand2d" ;;
    wilor) primary="outputs/output_wilor" ;;
    direct) primary="outputs/phase2_gates/g1_views/method_direct" ;;
    *) echo "unsupported baseline ${method}" >&2; exit 2 ;;
esac
output="signal4d/runs/baseline_scan_${method}"
cd "${project}"
args=(
    --manifest "${manifest}"
    --primary-root "${primary}"
    --model-path "${model}"
    --output-root "${output}"
    --method-name "legacy_${method}_pinned_decode"
    --device cpu
)
if [[ "${method}" != "hamer" ]]; then
    args+=(--fallback-root "${fallback}")
fi
conda run -n signal4d signal4d materialize-legacy "${args[@]}"
conda run -n signal4d signal4d evaluate-sgnify \
    --manifest "${manifest}" \
    --predictions "${output}/predictions" \
    --gt-root data/smplx_gt \
    --model-path "${model}" \
    --upper-indices dexavatar_fitting/assets/smplx_upper_body_vidx.npy \
    --left-indices dexavatar_fitting/assets/smplx_left_hand_vidx.npy \
    --right-indices dexavatar_fitting/assets/smplx_right_hand_vidx.npy \
    --output "${output}/evaluation"
