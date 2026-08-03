#!/usr/bin/env bash
set -Eeuo pipefail

PHASE2R_REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHASE2R_CACHE="${PHASE2R_REPOSITORY}/cache/phase2r/domain_aligned_v1"
PHASE2R_OUTPUT="${PHASE2R_REPOSITORY}/outputs/phase2r/domain_aligned_v1_seed42"
PHASE2R_LOG_ROOT="${PHASE2R_REPOSITORY}/logs/phase2r/domain_aligned_v1_seed42"
PHASE2R_CONFIG="${PHASE2R_REPOSITORY}/phase2_refiner/configs/phase2r_domain_aligned_v1.yaml"
PHASE2R_MINIMUM_FREE_MIB="${PHASE2R_MINIMUM_FREE_MIB:-40000}"

mkdir -p "${PHASE2R_LOG_ROOT}"
exec > >(tee -a "${PHASE2R_LOG_ROOT}/run.log") 2>&1

run_phase2r() {
  cd "${PHASE2R_REPOSITORY}"
  echo "[phase2r] start $(date --iso-8601=seconds)"
  echo "MATERIALIZING" > "${PHASE2R_LOG_ROOT}/status"
  python -u -m phase2_refiner.data.materialize_phase2r_v4 --resume

  python - <<'PY'
from phase2_refiner.config import load_config, validate_config
from phase2_refiner.data.dataset import SequenceCacheDataset

config = load_config("phase2_refiner/configs/phase2r_domain_aligned_v1.yaml")
validate_config(config, require_data=True, require_validation=True)
SequenceCacheDataset(
    config["data"]["val_glob"],
    max_frames=config["model"]["max_frames"],
    input_dim=config["model"]["input_dim"],
    physical_time_motion=True,
    motion_reference_seconds=config["data"]["motion_reference_seconds"],
    require_phase2r_semantics=True,
)
print("[phase2r] strict cache preflight passed", flush=True)
PY

  echo "WAITING_FOR_GPU" > "${PHASE2R_LOG_ROOT}/status"
  while true; do
    PHASE2R_FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ "${PHASE2R_FREE_MIB}" =~ ^[0-9]+$ ]] && (( PHASE2R_FREE_MIB >= PHASE2R_MINIMUM_FREE_MIB )); then
      break
    fi
    echo "[phase2r] waiting for GPU: free=${PHASE2R_FREE_MIB:-unknown} MiB required=${PHASE2R_MINIMUM_FREE_MIB} MiB"
    sleep 60
  done

  echo "TRAINING" > "${PHASE2R_LOG_ROOT}/status"
  python -u -m phase2_refiner.train \
    --config "${PHASE2R_CONFIG}" \
    --device cuda

  echo "EVALUATING_G4_PROXY" > "${PHASE2R_LOG_ROOT}/status"
  python -u -m phase2_refiner.evaluate_residual_checkpoint \
    --manifest "${PHASE2R_CACHE}/splits/val.json" \
    --config "${PHASE2R_CONFIG}" \
    --checkpoint "${PHASE2R_OUTPUT}/best.pt" \
    --real-residual-audit "${PHASE2R_CACHE}/proxy_residual_audit.json" \
    --output "${PHASE2R_OUTPUT}/g4_proxy.json" \
    --per-clip-output "${PHASE2R_OUTPUT}/g4_proxy_per_clip.json" \
    --batch-size 32 \
    --bootstrap-samples 2000 \
    --device cuda

  echo "EVALUATING_VERTEX_PROXY" > "${PHASE2R_LOG_ROOT}/status"
  python -u -m phase2_refiner.evaluate_t1_vertices \
    --config "${PHASE2R_CONFIG}" \
    --checkpoint "${PHASE2R_OUTPUT}/best.pt" \
    --output "${PHASE2R_OUTPUT}/vertex_proxy_fp32.json" \
    --model-folder "${PHASE2R_REPOSITORY}/SMPLer-X/common/utils/human_model_files" \
    --vertex-ids "${PHASE2R_REPOSITORY}/data/evaluation_from_author/data/data/MANO_SMPLX_vertex_ids.pkl" \
    --upper-body-ids "${PHASE2R_REPOSITORY}/data/evaluation_from_author/data/data/sgnify_part_segm_above_pelvis_joint/upper_body_minus_face.npy" \
    --batch-size 8 \
    --device cuda

  echo "COMPLETE" > "${PHASE2R_LOG_ROOT}/status"
  echo "[phase2r] complete $(date --iso-8601=seconds)"
}

phase2r_failure_trap() {
  PHASE2R_EXIT_CODE="$?"
  if (( PHASE2R_EXIT_CODE != 0 )); then
    echo "FAILED:${PHASE2R_EXIT_CODE}" > "${PHASE2R_LOG_ROOT}/status"
    echo "[phase2r] failed exit=${PHASE2R_EXIT_CODE} $(date --iso-8601=seconds)"
  fi
}

trap phase2r_failure_trap EXIT
run_phase2r
trap - EXIT
