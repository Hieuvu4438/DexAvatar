#!/usr/bin/env bash
set -euo pipefail

project=/home/haipd/DexAvatar
run_root="${project}/outputs/phoenix_soke_full_v1"
h32_root="${project}/data/SignAvatars/datasets/language2motion/annotations/SMPL-X_phoenix"
phoenix_video_root=/home/dongvk/datasets/phoenix14T/videos_phoenix/videos
selection_root="${run_root}/selections"
shard_root="${run_root}/wilor_shards"
wilor_root="${run_root}/wilor_outputs"
cache_root="${project}/cache/signal4d_external"
config="${project}/phase2_refiner/configs/phoenix_soke_full_raw_fusion_v1.yaml"
train_output="${project}/outputs/phase2r/phoenix_soke_full_raw_fusion_v1_seed42"
smoke_output="${project}/outputs/phase2r/phoenix_soke_full_raw_fusion_v1_smoke"
log_root="${run_root}/logs"
wilor_python=/home/haipd/miniconda3/envs/wilor/bin/python
# Two frame-batched workers measured 4.04 aggregate frame/s on a locked
# 64-frame PHOENIX manifest, while retaining ample host/GPU memory for the
# independent SignAvatars queue. Keep the override for other hosts.
wilor_parallel=${WILOR_PARALLEL:-2}

mkdir -p "${log_root}"
cd "${project}"

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

h32_status() {
    python - "${selection_root}" "${h32_root}" "${phoenix_video_root}" "$@" <<'PY'
import json,sys
from pathlib import Path
selection_root=Path(sys.argv[1]); h32_root=Path(sys.argv[2]); video_root=Path(sys.argv[3])
required=[]
for split in sys.argv[4:]:
    payload=json.loads((selection_root/split/'selection.json').read_text())
    required.extend(item['source_clip'] for item in payload['clips'])
missing=[name for name in required if not (h32_root/f'{name}.pkl').is_file()]
print(f"ready={len(required)-len(missing)} required={len(required)} missing={len(missing)}")
if missing:
    print('first_missing=' + ','.join(missing[:5]))
else:
    sys.exit(0)

# The H32 extractor shards the sorted union of train/dev/test videos by
# index modulo N and may further bound a worker partition with
# --assigned-start/--assigned-stop. Require every missing clip to be covered
# by the exact range of at least one live extractor, otherwise a surviving
# sibling slice could hide a dead slice forever.
video_ids=sorted({path.stem for path in video_root.glob('*/*.mp4')})
video_index={name:index for index,name in enumerate(video_ids)}
unknown=sorted({name for name in missing if name not in video_index})
if unknown:
    print('health_error=missing clips absent from extractor video union: ' + ','.join(unknown[:5]))
    sys.exit(4)
live_ranges=[]
for command_path in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        tokens=command_path.read_bytes().split(b'\0')
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    decoded=[token.decode('utf-8', errors='replace') for token in tokens if token]
    if not any(token.endswith('extract_phoenix.py') for token in decoded):
        continue
    try:
        worker=int(decoded[decoded.index('--worker_id') + 1])
        num_workers=int(decoded[decoded.index('--num_workers') + 1])
    except (ValueError, IndexError):
        continue
    try:
        start=int(decoded[decoded.index('--assigned_start') + 1])
    except (ValueError, IndexError):
        start=0
    try:
        stop=int(decoded[decoded.index('--assigned_stop') + 1])
    except (ValueError, IndexError):
        stop=None
    live_ranges.append((worker,num_workers,start,stop))

def covered(name):
    index=video_index[name]
    for worker,num_workers,start,stop in live_ranges:
        if index % num_workers != worker:
            continue
        local=index // num_workers
        if local >= start and (stop is None or local < stop):
            return True
    return False

uncovered=sorted(name for name in missing if not covered(name))
if uncovered:
    print('health_error=missing H32 clips have no live covering range: ' + ','.join(uncovered[:5]))
    sys.exit(4)
print('live_covering_ranges=' + ','.join(
    f'w{worker}/{num_workers}[{start}:{"" if stop is None else stop}]'
    for worker,num_workers,start,stop in sorted(
        set(live_ranges), key=lambda item:(item[0],item[1],item[2],10**18 if item[3] is None else item[3])
    )
))
sys.exit(3)
PY
}

check_h32_health() {
    local splits=("$@")
    if [[ ${#splits[@]} -eq 0 ]]; then
        splits=(train dev)
    fi
    if h32_status "${splits[@]}"; then
        return 0
    else
        local h32_code=$?
    fi
    if [[ ${h32_code} -ne 3 ]]; then
        echo "[$(timestamp)] ERROR: H32 extraction health check failed" >&2
        exit "${h32_code}"
    fi
    return 3
}

audit_wilor_splits() {
    local label=$1
    local output=$2
    shift 2
    local splits=("$@")
    if [[ ! -f "${output}" ]]; then
        sleep 5
        python -m phase2_refiner.data.audit_phoenix_wilor_frontend \
            --run-root "${run_root}" \
            --output "${output}" \
            --splits "${splits[@]}" \
            --minimum-age-seconds 2 \
            --require-all-verified
    fi
    python - "${output}" "${splits[@]}" <<'PY'
import json,sys
from pathlib import Path
payload=json.loads(Path(sys.argv[1]).read_text())
expected=set(sys.argv[2:])
actual=set(payload.get('splits',{}))
if payload.get('schema') != 'signal4d-phoenix-wilor-frontend-audit-v1':
    raise SystemExit('invalid WiLoR audit schema')
if actual != expected:
    raise SystemExit(f'WiLoR audit splits {sorted(actual)} != {sorted(expected)}')
if not payload.get('all_verified'):
    raise SystemExit('WiLoR audit is not all_verified')
for split,item in payload['splits'].items():
    if not item.get('all_verified') or item.get('verified_shards') != item.get('declared_shards'):
        raise SystemExit(f'incomplete WiLoR audit split: {split}')
print(f'WiLoR audit gate passed: {sorted(expected)}')
PY
    echo "[$(timestamp)] WiLoR audit passed label=${label} output=${output}"
}

echo "[$(timestamp)] PHOENIX/SOKE full Transformer pipeline started"
echo "[$(timestamp)] Starting target-independent H32/WiLoR frontend overlap"
check_h32_health || true

run_wilor_split() {
    local split_name=$1
    local monitor_h32=${2:-false}
    local split_shards="${shard_root}/${split_name}"
    local split_output="${wilor_root}/${split_name}"
    mkdir -p "${split_output}"
    echo "[$(timestamp)] WiLoR split=${split_name} parallel=${wilor_parallel}"
    for manifest in "${split_shards}"/shard_[0-9][0-9][0-9][0-9].json; do
        local stem
        stem=$(basename "${manifest}" .json)
        local destination="${split_output}/${stem}"
        local logfile="${log_root}/wilor_${split_name}_${stem}.log"
        if [[ -f "${destination}/hamer/hamer.pkl" && -f "${destination}/wilor/wilor.pkl" ]]; then
            echo "[$(timestamp)] WiLoR already complete ${split_name}/${stem}"
            continue
        fi
        if [[ -e "${destination}" ]]; then
            echo "[$(timestamp)] ERROR: incomplete WiLoR artifact requires inspection: ${destination}" >&2
            exit 4
        fi
        while (( $(jobs -rp | wc -l) >= wilor_parallel )); do
            wait -n
            if [[ "${monitor_h32}" == true ]]; then
                check_h32_health || true
            fi
        done
        (
            cd "${project}/WiLoR"
            CUDA_VISIBLE_DEVICES=0 "${wilor_python}" export_hamer_pkl.py \
                --frame_manifest "${manifest}" \
                --out_folder "${destination}" \
                --fast \
                --frame_batch_size 16 \
                --hand_batch_size 16
        ) >"${logfile}" 2>&1 &
        echo "[$(timestamp)] launched ${split_name}/${stem} log=${logfile}"
    done
    wait
    echo "[$(timestamp)] WiLoR complete split=${split_name}"
}

# Only gradient and checkpoint-selection inputs are processed before freezing.
run_wilor_split train true
run_wilor_split dev true
audit_wilor_splits train_dev "${run_root}/wilor_train_dev_frontend_audit.json" train dev

echo "[$(timestamp)] Waiting for complete target-independent H32 train/dev coverage"
while ! check_h32_health; do
    sleep 60
done

h32_audit="${run_root}/h32_train_dev_frontend_audit.json"
if [[ ! -f "${h32_audit}" ]]; then
    # H32 workers predating the atomic writer may expose the final filename
    # during pickle.dump.  A short settle plus full payload audit prevents the
    # cache stage from racing that last write.
    sleep 5
    python -m phase2_refiner.data.audit_phoenix_h32_frontend \
        --selection-root "${selection_root}" \
        --h32-root "${h32_root}" \
        --splits train dev \
        --minimum-age-seconds 2 \
        --output "${h32_audit}"
fi

build_cache_split() {
    local official_split=$1
    local phase2_split=$2
    local output=$3
    if [[ -f "${output}/splits/${phase2_split}.json" ]]; then
        echo "[$(timestamp)] cache already complete ${official_split}: ${output}"
        return
    fi
    if [[ -e "${output}" ]]; then
        echo "[$(timestamp)] ERROR: incomplete cache requires inspection: ${output}" >&2
        exit 5
    fi
    python -m phase2_refiner.data.build_phoenix_soke_full_cache \
        --selection "${selection_root}/${official_split}/selection.json" \
        --smplerx-root "${h32_root}" \
        --wilor-shard-manifest-root "${shard_root}/${official_split}" \
        --wilor-root "${wilor_root}/${official_split}" \
        --output "${output}"
}

train_cache="${cache_root}/phoenix_soke_full_train_v1"
dev_cache="${cache_root}/phoenix_soke_full_dev_v1"
test_cache="${cache_root}/phoenix_soke_full_test_v1"
build_cache_split train train "${train_cache}"
build_cache_split dev val "${dev_cache}"

if [[ ! -f "${smoke_output}/last.pt" ]]; then
    if [[ -e "${smoke_output}" ]]; then
        echo "[$(timestamp)] ERROR: incomplete smoke output requires inspection: ${smoke_output}" >&2
        exit 6
    fi
    echo "[$(timestamp)] Transformer smoke test"
    CUDA_VISIBLE_DEVICES=0 python -m phase2_refiner.train \
        --config "${config}" \
        --device cuda \
        --no-validation \
        --max-steps 2 \
        --batch-size 8 \
        --output-dir "${smoke_output}"
fi

if [[ ! -f "${train_output}/last.pt" ]]; then
    resume_args=()
    if [[ -e "${train_output}" ]]; then
        if resume_checkpoint=$(python - "${train_output}" <<'PY'
import sys
from pathlib import Path
import torch

root=Path(sys.argv[1])
candidates=[]
for path in [root/'best.pt', *sorted(root.glob('step_*.pt'))]:
    if not path.is_file():
        continue
    try:
        state=torch.load(path, map_location='cpu', weights_only=False)
        candidates.append((int(state['step']), path))
    except Exception as error:
        print(f'Ignoring unreadable checkpoint {path}: {error}', file=sys.stderr)
if not candidates:
    raise SystemExit(1)
print(max(candidates, key=lambda item:item[0])[1])
PY
        ); then
            resume_args=(--resume "${resume_checkpoint}")
            echo "[$(timestamp)] Resuming PHOENIX Transformer from ${resume_checkpoint}"
        else
            echo "[$(timestamp)] ERROR: incomplete training has no readable resume checkpoint: ${train_output}" >&2
            exit 7
        fi
    fi
    echo "[$(timestamp)] Full PHOENIX Transformer training"
    CUDA_VISIBLE_DEVICES=0 python -m phase2_refiner.train \
        --config "${config}" \
        --device cuda \
        --output-dir "${train_output}" \
        "${resume_args[@]}"
fi

if [[ ! -f "${train_output}/best.pt" ]]; then
    echo "[$(timestamp)] ERROR: completed training lacks best.pt: ${train_output}" >&2
    exit 8
fi

checkpoint="${train_output}/best.pt"
calibration="${train_output}/phoenix_dev_benefit_calibration.json"
if [[ ! -f "${calibration}" ]]; then
    echo "[$(timestamp)] Freeze region thresholds on official dev"
    CUDA_VISIBLE_DEVICES=0 python -m phase2_refiner.evaluate_phoenix_soke_pampjpe \
        --mode calibrate \
        --config "${config}" \
        --checkpoint "${checkpoint}" \
        --manifest "${dev_cache}/splits/val.json" \
        --output "${calibration}" \
        --device cuda
fi

# Test RGB is target-free, but its WiLoR/cache materialization is deliberately
# deferred until both checkpoint and thresholds above are immutable.
run_wilor_split test
audit_wilor_splits test "${run_root}/wilor_test_frontend_audit.json" test

echo "[$(timestamp)] Waiting for complete target-independent H32 test coverage"
while ! check_h32_health test; do
    sleep 60
done

h32_test_audit="${run_root}/h32_test_frontend_audit.json"
if [[ ! -f "${h32_test_audit}" ]]; then
    sleep 5
    python -m phase2_refiner.data.audit_phoenix_h32_frontend \
        --selection-root "${selection_root}" \
        --h32-root "${h32_root}" \
        --splits test \
        --minimum-age-seconds 2 \
        --output "${h32_test_audit}"
fi
build_cache_split test test "${test_cache}"

evaluation="${train_output}/phoenix_test_soke_pampjpe.json"
if [[ ! -f "${evaluation}" ]]; then
    echo "[$(timestamp)] Final official-test PA-MPJPE evaluation"
    CUDA_VISIBLE_DEVICES=0 python -m phase2_refiner.evaluate_phoenix_soke_pampjpe \
        --mode evaluate \
        --config "${config}" \
        --checkpoint "${checkpoint}" \
        --manifest "${test_cache}/splits/test.json" \
        --calibration "${calibration}" \
        --output "${evaluation}" \
        --device cuda
fi

evaluation_report="${train_output}/phoenix_test_soke_pampjpe.md"
if [[ ! -f "${evaluation_report}" ]]; then
    python scripts/render_phoenix_soke_evaluation.py \
        --input "${evaluation}" \
        --output "${evaluation_report}"
fi

echo "[$(timestamp)] COMPLETE evaluation=${evaluation} report=${evaluation_report}"
