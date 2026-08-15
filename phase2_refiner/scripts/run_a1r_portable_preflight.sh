#!/usr/bin/env bash
set -Eeuo pipefail

A1R_REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
A1R_WORK_ROOT="${A1R_REPOSITORY}/outputs/how2sign_exact_a1_work_v1/batch_000001"
A1R_EXPERT_ROOT="${A1R_WORK_ROOT}/expert_output"
A1R_IMAGE_ROOT="${A1R_WORK_ROOT}/images"
A1R_CONTRACT_ROOT="${A1R_WORK_ROOT}/a1r_portable_contract_v1"
A1R_CACHE="${A1R_REPOSITORY}/cache/phase2r/domain_aligned_v1/clips/train/how2sign_train_0psMj0gsJjs_14-5-rgb_front.npz"
A1R_LOG_ROOT="${A1R_REPOSITORY}/logs/phase2r/a1r_portable_preflight"
# The fitter processes one frame at a time.  Keep >2x observed model headroom
# without imposing the larger mesh-training batch requirement.
A1R_MINIMUM_FREE_MIB="${A1R_MINIMUM_FREE_MIB:-24000}"

mkdir -p "${A1R_LOG_ROOT}"
exec > >(tee -a "${A1R_LOG_ROOT}/run.log") 2>&1

a1r_fail() {
  A1R_EXIT_CODE="$?"
  if ((A1R_EXIT_CODE != 0)); then
    echo "FAILED:${A1R_EXIT_CODE}" > "${A1R_LOG_ROOT}/status"
    echo "[a1r-preflight] failed exit=${A1R_EXIT_CODE} $(date --iso-8601=seconds)"
  fi
}
trap a1r_fail EXIT

cd "${A1R_REPOSITORY}"
python - <<'PY'
from pathlib import Path

from phase2_refiner.data.cache_schema import load_cache_clip
from phase2_refiner.data.materialize_a1r_cache import validate_provider_manifest

repository = Path.cwd()
validate_provider_manifest(
    repository / "phase2_refiner/configs/a1r_portable_ensemble_provider_v1.json"
)
clip = load_cache_clip(
    repository
    / "cache/phase2r/domain_aligned_v1/clips/train/how2sign_train_0psMj0gsJjs_14-5-rgb_front.npz"
)
image_root = repository / "outputs/how2sign_exact_a1_work_v1/batch_000001/images"
expected = {f"{name}.png" for name in clip.frame_names}
actual = {path.name for path in image_root.glob("*.png")}
assert expected == actual, (
    f"preflight image mismatch: missing={sorted(expected - actual)[:3]} "
    f"extra={sorted(actual - expected)[:3]}"
)
assert len(expected) == 32
print("[a1r-preflight] provider manifest and exact 32-frame coverage passed", flush=True)
PY

if [[ -d "${A1R_EXPERT_ROOT}/smplifyx/results" ]] \
  && [[ -n "$(find "${A1R_EXPERT_ROOT}/smplifyx/results" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing non-empty append-only A1R result directory" >&2
  exit 2
fi
if [[ -e "${A1R_CONTRACT_ROOT}" ]]; then
  echo "Refusing existing append-only A1R contract directory: ${A1R_CONTRACT_ROOT}" >&2
  exit 2
fi

echo "WAITING_FOR_GPU" > "${A1R_LOG_ROOT}/status"
while true; do
  A1R_FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ "${A1R_FREE_MIB}" =~ ^[0-9]+$ ]] && ((A1R_FREE_MIB >= A1R_MINIMUM_FREE_MIB)); then
    break
  fi
  echo "[a1r-preflight] waiting for GPU: free=${A1R_FREE_MIB:-unknown} MiB required=${A1R_MINIMUM_FREE_MIB} MiB"
  sleep 60
done

echo "FITTING" > "${A1R_LOG_ROOT}/status"
python -u -m phase2_refiner.data.run_a1r_fitting \
  --cache "${A1R_CACHE}" \
  --image-root "${A1R_IMAGE_ROOT}" \
  --output-root "${A1R_EXPERT_ROOT}" \
  --contract-root "${A1R_CONTRACT_ROOT}" \
  --smplx-init-dir ensemble_smplx \
  --gpu 0

echo "COMPLETE" > "${A1R_LOG_ROOT}/status"
echo "[a1r-preflight] complete $(date --iso-8601=seconds)"
trap - EXIT
