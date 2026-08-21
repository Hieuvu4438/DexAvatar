#!/usr/bin/env bash
set -euo pipefail

project="/home/haipd/DexAvatar"
model="${project}/SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz"
upper="${project}/dexavatar_fitting/assets/smplx_upper_body_vidx.npy"
left="${project}/dexavatar_fitting/assets/smplx_left_hand_vidx.npy"
right="${project}/dexavatar_fitting/assets/smplx_right_hand_vidx.npy"
cache="signal4d/artifacts/cache/sgnify_smplerx_wilor_a1_leftmirror_v5_all"
baseline="signal4d/runs/baseline_scan_ensemble/predictions"

cd "${project}"

# The producer writes this file only after both the fit and full evaluator
# complete. Waiting here avoids reading a partially written prediction tree.
while [[ ! -f signal4d/runs/m1_a1_v5_test/evaluation/summary.json ]]; do
    sleep 15
done

for alpha in 1.5 3.0; do
    tag="${alpha//./p}"
    for split in dev test; do
        if [[ "${split}" == "dev" ]]; then
            manifest="signal4d/artifacts/manifests/frozen_seed_20260819_v2/sgnify_development.jsonl"
            gt_cache="signal4d/artifacts/gt_cache/calibration_development"
        else
            manifest="signal4d/artifacts/manifests/frozen_seed_20260819_v2/sgnify_test.jsonl"
            gt_cache="signal4d/artifacts/gt_cache/test_frozen_20260819"
        fi
        output="signal4d/runs/m1_a1_v5_extrap_alpha_${tag}_${split}"
        conda run -n signal4d signal4d extrapolate \
            --manifest "${manifest}" \
            --candidate-root "signal4d/runs/m1_a1_v5_${split}/predictions" \
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
done

for tag in 1p0 1p5 3p0; do
    conda run -n signal4d signal4d train-gate \
        --config "signal4d/configs/gating/a1_m1_alpha_${tag}_v5.yaml" \
        --output "signal4d/artifacts/gating/a1_m1_alpha_${tag}_v5"
done

conda run -n signal4d signal4d build-multigate \
    --config signal4d/configs/gating/a1_multiscale_v5.yaml \
    --output signal4d/artifacts/gating/a1_multiscale_v5

# Eligibility is decided entirely from grouped OOF historical predictions.
# A failed candidate must never advance to the prospective endpoint.
python - <<'PY'
import json
from pathlib import Path

metadata = json.loads(
    Path("signal4d/artifacts/gating/a1_multiscale_v5/metadata.json").read_text()
)
point = float(metadata["oof_clip_macro_delta_mm"])
upper = float(metadata["oof_ci95_clip_bootstrap_mm"][1])
if point > -0.5 or upper >= 0.0:
    raise SystemExit(
        f"historical OOF eligibility failed: delta={point}, CI upper={upper}"
    )
PY

for split in dev test; do
    if [[ "${split}" == "dev" ]]; then
        manifest="signal4d/artifacts/manifests/frozen_seed_20260819_v2/sgnify_development.jsonl"
        gt_cache="signal4d/artifacts/gt_cache/calibration_development"
    else
        manifest="signal4d/artifacts/manifests/frozen_seed_20260819_v2/sgnify_test.jsonl"
        gt_cache="signal4d/artifacts/gt_cache/test_frozen_20260819"
    fi
    output="signal4d/runs/a1_multiscale_v5_${split}"
    conda run -n signal4d signal4d apply-multigate \
        --manifest "${manifest}" \
        --baseline-root "${baseline}" \
        --cache-root "${cache}" \
        --bundle signal4d/artifacts/gating/a1_multiscale_v5 \
        --hypothesis "m1_alpha_1p0=signal4d/runs/m1_a1_v5_${split}/predictions" \
        --hypothesis "m1_alpha_1p5=signal4d/runs/m1_a1_v5_extrap_alpha_1p5_${split}/predictions" \
        --hypothesis "m1_alpha_3p0=signal4d/runs/m1_a1_v5_extrap_alpha_3p0_${split}/predictions" \
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
    conda run -n signal4d signal4d compare \
        --candidate-csv "${output}/evaluation/per_clip.csv" \
        --baseline-csv "signal4d/runs/baseline_scan_ensemble_${split}/evaluation/per_clip.csv" \
        --metric tr_v2v_left_hand_mm \
        --replicates 10000 \
        --output "${output}/comparison_left.json"
done
