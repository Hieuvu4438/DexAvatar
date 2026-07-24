#!/usr/bin/env bash
set -euo pipefail

cd /home/haipd/DexAvatar

teacher_session=phase2_how2sign_teacher_v1_20260724
teacher_root=data/phase2_how2sign_teacher_v1
cache_root=cache/phase2/how2sign_t1_v1
config=phase2_refiner/configs/uawsr_t1_how2sign_geometry.yaml
spatial_init=outputs/phase2_training/t1_arctic_geometry_seed42/best.pt
train_output=outputs/phase2_training/t1_how2sign_geometry_seed42

count_npz() {
    local directory=$1
    if [[ -d "${directory}" ]]; then
        find "${directory}" -type f -name '*.npz' | wc -l
    else
        printf '0\n'
    fi
}

while true; do
    train_count=$(count_npz "${teacher_root}/train/clips")
    val_count=$(count_npz "${teacher_root}/val/clips")
    if [[ "${train_count}" -ge 11000 && "${val_count}" -ge 1200 ]]; then
        break
    fi
    printf '[wait] teacher train=%s/11000 val=%s/1200\n' "${train_count}" "${val_count}"
    if ! tmux has-session -t "${teacher_session}" 2>/dev/null; then
        printf '[stalled] teacher session is absent; preserving watcher until a resumable teacher is restarted\n'
    fi
    sleep 60
done

printf '[build] teacher session complete; constructing locked cache\n'
python -u -m phase2_refiner.data.build_how2sign_cache \
    --teacher-root "${teacher_root}" \
    --output "${cache_root}" \
    --minimum-train-clips 10000 \
    --generic-train-manifest cache/phase2/arctic_t1_v1/splits/train.json

python -u -m phase2_refiner.data.audit_training_cache \
    --train-manifest "${cache_root}/splits/train.json" \
    --val-manifest "${cache_root}/splits/val.json" \
    --output "${cache_root}/readiness_report.json" \
    --require-main-go

batch_size=0
for candidate in 64 48 32; do
    preflight_output="outputs/phase2_training/preflight_how2sign_batch${candidate}_20260724"
    printf '[preflight] testing batch=%s output=%s\n' "${candidate}" "${preflight_output}"
    if python -u -m phase2_refiner.train \
        --config "${config}" \
        --spatial-init "${spatial_init}" \
        --output-dir "${preflight_output}" \
        --batch-size "${candidate}" \
        --gradient-accumulation 1 \
        --max-steps 2 \
        --no-validation; then
        batch_size=${candidate}
        break
    fi
done

if [[ "${batch_size}" -eq 0 ]]; then
    printf '[abort] no finite GPU batch preflight passed\n'
    exit 4
fi

printf '[train] selected batch=%s\n' "${batch_size}"
python -u -m phase2_refiner.train \
    --config "${config}" \
    --spatial-init "${spatial_init}" \
    --output-dir "${train_output}" \
    --batch-size "${batch_size}" \
    --gradient-accumulation 1

python -u -m phase2_refiner.evaluate_t1_recovery \
    --config "${config}" \
    --checkpoint "${train_output}/best.pt" \
    --output "${train_output}/t1_recovery_fp32.json" \
    --batch-size "${batch_size}" \
    --eval-precision fp32

python -u -m phase2_refiner.evaluate_t1_vertices \
    --config "${config}" \
    --checkpoint "${train_output}/best.pt" \
    --output "${train_output}/t1_vertex_recovery_fp32.json" \
    --model-folder SMPLer-X/common/utils/human_model_files \
    --vertex-ids SMPLer-X/common/utils/human_model_files/smplx/MANO_SMPLX_vertex_ids.pkl \
    --upper-body-ids data/evaluation_from_author/data/data/sgnify_part_segm_above_pelvis_joint/upper_body_minus_face.npy \
    --batch-size 16 \
    --eval-precision fp32

printf '[complete] How2Sign cache, training, and formal evaluations finished\n'
