#!/usr/bin/env bash
set -euo pipefail

minimum_free_mib=12000
maximum_wait_seconds=3600
poll_seconds=30
start_seconds=${SECONDS}

while true; do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
    if [[ ${free_mib} =~ ^[0-9]+$ ]] && (( free_mib >= minimum_free_mib )); then
        printf '[resource] ready free_mib=%s threshold_mib=%s\n' "${free_mib}" "${minimum_free_mib}"
        break
    fi
    elapsed=$((SECONDS - start_seconds))
    if (( elapsed >= maximum_wait_seconds )); then
        printf '[resource] wait_timeout elapsed_seconds=%s free_mib=%s\n' "${elapsed}" "${free_mib}"
        exit 75
    fi
    printf '[resource] waiting elapsed_seconds=%s free_mib=%s\n' "${elapsed}" "${free_mib}"
    sleep "${poll_seconds}"
done

exec timeout --signal=TERM --kill-after=30s 90m \
    python -m phase2_refiner.train \
    --config phase2_refiner/configs/sign_domain_raw_fusion_v1.yaml \
    --device cuda \
    --resume outputs/phase2r/sign_domain_raw_fusion_v1_seed42/best.pt
