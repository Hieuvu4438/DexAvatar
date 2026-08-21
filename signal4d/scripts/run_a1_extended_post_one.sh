#!/usr/bin/env bash
set -euo pipefail

clip="${1:?usage: run_a1_extended_post_one.sh CLIP}"
project="/home/haipd/DexAvatar"
source_root="${project}/outputs/method_ensemble/${clip}"
destination="${project}/signal4d/artifacts/legacy_a1_extended_post_v1/${clip}"
log="${project}/signal4d/logs/legacy_a1_extended_post_v1/${clip}.log"
parts="${project}/scratch/maps_sign_runtime_code/Ablehnen/dexavatar_fitting/assets/smplx_parts_segm.pkl"

mkdir -p "${destination}" "$(dirname "${log}")"
for item in gender.txt mean_shape_smplx.npy sapiens.pkl hamer smplerx ensemble_smplx; do
    if [[ ! -e "${source_root}/${item}" ]]; then
        echo "Missing frozen A1 input ${source_root}/${item}" >&2
        exit 3
    fi
    if [[ ! -e "${destination}/${item}" ]]; then
        ln -s "${source_root}/${item}" "${destination}/${item}"
    fi
done

cd "${project}/dexavatar_fitting"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/smplifyx:$(pwd)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

python smplifyx/main.py \
    --config cfg_files/fit_smplx_vposer_x_ensemble.yaml \
    --data_folder "${destination}" \
    --output_folder "${destination}/smplifyx" \
    --img_folder "${project}/data/frames/${clip}" \
    --model_folder ../SMPLer-X/common/utils/human_model_files \
    --part_segm_fn "${parts}" \
    --visualize False \
    --split_num 1 \
    --cur_num 0 \
    --sign_segment "${project}/signal4d/configs/data/sgnify_extended_post_segments.json" \
    >"${log}" 2>&1

if [[ -d "${destination}/smplifyx/results" ]]; then
    result_count=$(find "${destination}/smplifyx/results" -type f -name 'low_*.pkl' | wc -l)
else
    result_count=0
fi
if [[ "${result_count}" -eq 0 ]]; then
    echo "No usable ensemble result for ${clip}; A1 requires HaMeR atomic fallback." >>"${log}"
else
    echo "Completed ${clip}: ${result_count} ensemble result files" >>"${log}"
fi
