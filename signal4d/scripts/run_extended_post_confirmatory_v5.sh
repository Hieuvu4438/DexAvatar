#!/usr/bin/env bash
set -euo pipefail

project="/home/haipd/DexAvatar"
manifest="signal4d/artifacts/manifests/sgnify_extended_post_test_v1.jsonl"
legacy="signal4d/artifacts/legacy_a1_extended_post_v1"
fallback="signal4d/artifacts/legacy_a1_hamer_extended_post_v1"
a0_composed="signal4d/artifacts/legacy_a1_a0_composed_extended_post_v1"
composed="signal4d/artifacts/legacy_a1_composed_extended_post_v1"
fallback_finalize="signal4d/artifacts/legacy_a1_hamer_extended_post_v1/fallback_finalize.json"
cache="signal4d/artifacts/cache/sgnify_extended_post_a1_v5"
baseline="signal4d/runs/extended_post_a1_baseline"
m1="signal4d/runs/extended_post_m1_a1_v5"
alpha15="signal4d/runs/extended_post_m1_a1_v5_extrap_alpha_1p5"
alpha30="signal4d/runs/extended_post_m1_a1_v5_extrap_alpha_3p0"
candidate="signal4d/runs/extended_post_a1_multiscale_v5"
candidate_repro="signal4d/runs/extended_post_a1_multiscale_v5_repro"
gate="signal4d/artifacts/gating/a1_multiscale_v5"
gt_cache="signal4d/artifacts/gt_cache/extended_post_v1_confirmatory"
release="signal4d/artifacts/releases/extended_post_v5_release_freeze.json"
comparison_root="signal4d/runs/extended_post_confirmatory_v5/comparisons"
model="${project}/SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz"
upper="${project}/dexavatar_fitting/assets/smplx_upper_body_vidx.npy"
left="${project}/dexavatar_fitting/assets/smplx_left_hand_vidx.npy"
right="${project}/dexavatar_fitting/assets/smplx_right_hand_vidx.npy"

cd "${project}"

# Wait for the independently running, label-blind primary/fallback finalizer.
# It attempts HaMeR and permits raw SMPLer-X A0 only where both fitted sources
# are unavailable. No GT value is read in this availability decision.
while [[ ! -f "${fallback_finalize}" ]]; do
    sleep 30
done
python - <<'PY'
import json
from pathlib import Path

report = json.loads(
    Path(
        "signal4d/artifacts/legacy_a1_hamer_extended_post_v1/fallback_finalize.json"
    ).read_text()
)
if not report["passed"]:
    raise SystemExit(f"A1 fallback finalization failed: {report['missing']}")
PY

while [[ ! -f "${gate}/metadata.json" ]]; do
    sleep 15
done
python - <<'PY'
import json
from pathlib import Path

metadata = json.loads(
    Path("signal4d/artifacts/gating/a1_multiscale_v5/metadata.json").read_text()
)
if (
    float(metadata["oof_clip_macro_delta_mm"]) > -0.5
    or float(metadata["oof_ci95_clip_bootstrap_mm"][1]) >= 0.0
):
    raise SystemExit("frozen historical OOF gate is not eligible for confirmation")
PY

if [[ -e "${gt_cache}" ]]; then
    echo "Prospective GT cache already exists before release freeze: ${gt_cache}" >&2
    exit 21
fi

conda run -n signal4d signal4d compose-legacy \
    --manifest "${manifest}" \
    --primary-root "${fallback}" \
    --primary-subpath smplifyx/results \
    --fallback-root outputs/output_baseline \
    --fallback-subpath smplerx/smplx \
    --output-root "${a0_composed}" \
    --method-name legacy_a0_hamer_smplerx_terminal

conda run -n signal4d signal4d materialize-legacy \
    --manifest "${manifest}" \
    --primary-root "${legacy}" \
    --primary-subpath smplifyx/results \
    --fallback-root "${a0_composed}" \
    --fallback-subpath smplifyx/results \
    --model-path "${model}" \
    --output-root "${baseline}" \
    --method-name legacy_a1_ensemble_hamer_smplerx_terminal \
    --device cuda

conda run -n signal4d signal4d compose-legacy \
    --manifest "${manifest}" \
    --primary-root "${legacy}" \
    --primary-subpath smplifyx/results \
    --fallback-root "${a0_composed}" \
    --fallback-subpath smplifyx/results \
    --output-root "${composed}" \
    --method-name legacy_a1_ensemble_hamer_smplerx_terminal

conda run -n signal4d signal4d preprocess \
    --manifest "${manifest}" \
    --output-root "${cache}" \
    --body-root outputs/output_baseline \
    --wilor-root outputs/output_baseline \
    --model-path "${model}" \
    --legacy-root "${composed}" \
    --legacy-subpath smplifyx/results \
    --legacy-source-name legacy_a1_balanced \
    --device cuda

conda run -n signal4d signal4d fit-smplx \
    --config signal4d/configs/method/m1_a1_v5.yaml \
    --manifest "${manifest}" \
    --cache-root "${cache}" \
    --output-root "${m1}" \
    --model-path "${model}" \
    --device cuda

for alpha in 1.5 3.0; do
    if [[ "${alpha}" == "1.5" ]]; then output="${alpha15}"; else output="${alpha30}"; fi
    conda run -n signal4d signal4d extrapolate \
        --manifest "${manifest}" \
        --candidate-root "${m1}/predictions" \
        --baseline-root "${baseline}/predictions" \
        --cache-root "${cache}" \
        --model-path "${model}" \
        --alpha "${alpha}" \
        --output-root "${output}" \
        --device cuda
done

apply_gate() {
    local output=$1
    conda run -n signal4d signal4d apply-multigate \
        --manifest "${manifest}" \
        --baseline-root "${baseline}/predictions" \
        --cache-root "${cache}" \
        --bundle "${gate}" \
        --hypothesis "m1_alpha_1p0=${m1}/predictions" \
        --hypothesis "m1_alpha_1p5=${alpha15}/predictions" \
        --hypothesis "m1_alpha_3p0=${alpha30}/predictions" \
        --output-root "${output}"
}
apply_gate "${candidate}"
apply_gate "${candidate_repro}"

conda run -n signal4d signal4d verify-tree \
    --first "${candidate}/predictions" \
    --second "${candidate_repro}/predictions" \
    --output signal4d/reports/reproducibility_extended_post_v5.json

# This is the hard label boundary. Everything that can affect a prediction is
# hashed before cache-gt is allowed to read the prospective OBJ values.
conda run -n signal4d signal4d freeze-release \
    --output "${release}" \
    --config signal4d/configs/method/m1_a1_v5.yaml \
    --config signal4d/configs/gating/a1_m1_alpha_1p0_v5.yaml \
    --config signal4d/configs/gating/a1_m1_alpha_1p5_v5.yaml \
    --config signal4d/configs/gating/a1_m1_alpha_3p0_v5.yaml \
    --config signal4d/configs/gating/a1_multiscale_v5.yaml \
    --config signal4d/reports/preregistration_extended_post_v5.md \
    --manifest "${manifest}" \
    --artifact signal4d/src \
    --artifact signal4d/scripts \
    --artifact signal4d/pyproject.toml \
    --artifact signal4d/environment.json \
    --artifact signal4d/environment.lock.txt \
    --artifact signal4d/artifacts/calibration/sgnify_a1_leftmirror_v5_seed12345 \
    --artifact signal4d/artifacts/gating/a1_m1_alpha_1p0_v5 \
    --artifact signal4d/artifacts/gating/a1_m1_alpha_1p5_v5 \
    --artifact signal4d/artifacts/gating/a1_m1_alpha_3p0_v5 \
    --artifact "${gate}" \
    --artifact "${cache}" \
    --artifact "${legacy}" \
    --artifact "${fallback}" \
    --artifact "${a0_composed}" \
    --artifact "${composed}" \
    --artifact "${baseline}" \
    --artifact "${m1}" \
    --artifact "${alpha15}" \
    --artifact "${alpha30}" \
    --artifact "${candidate}" \
    --artifact "${model}" \
    --artifact "${upper}" \
    --artifact "${left}" \
    --artifact "${right}"

conda run -n signal4d signal4d cache-gt \
    --manifest "${manifest}" \
    --gt-root data/smplx_gt \
    --output-root "${gt_cache}"

for method in baseline candidate; do
    if [[ "${method}" == "baseline" ]]; then root="${baseline}"; else root="${candidate}"; fi
    conda run -n signal4d signal4d evaluate-sgnify \
        --manifest "${manifest}" \
        --predictions "${root}/predictions" \
        --gt-root data/smplx_gt \
        --gt-cache-root "${gt_cache}" \
        --model-path "${model}" \
        --upper-indices "${upper}" \
        --left-indices "${left}" \
        --right-indices "${right}" \
        --output "${root}/evaluation"
done

mkdir -p "${comparison_root}"
for metric in \
    tr_v2v_left_hand_mm tr_v2v_upper_body_mm tr_v2v_right_hand_mm \
    velocity_error acceleration_error jerk_error; do
    conda run -n signal4d signal4d compare \
        --candidate-csv "${candidate}/evaluation/per_clip.csv" \
        --baseline-csv "${baseline}/evaluation/per_clip.csv" \
        --metric "${metric}" \
        --replicates 10000 \
        --output "${comparison_root}/${metric}.json"
done

conda run -n signal4d signal4d assess-confirmatory \
    --candidate-summary "${candidate}/evaluation/summary.json" \
    --baseline-summary "${baseline}/evaluation/summary.json" \
    --comparison-root "${comparison_root}" \
    --reproducibility signal4d/reports/reproducibility_extended_post_v5.json \
    --expected-clips 56 \
    --expected-frames 769 \
    --output signal4d/reports/confirmatory_extended_post_v5.json
