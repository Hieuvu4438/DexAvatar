#!/bin/bash
# Exp2: ISOLATE INIT. Baseline config (fit_smplx_vposer_x.yaml) + nlf/smplx init.
# Same config as output_baseline; ONLY the init differs (NLF+WiLoR vs SMPLer-X).
# If hands recover -> the direct.yaml flags (frozen global_orient/transl + direct refine)
# were the culprit. If hands stay bad -> the NLF init is the culprit.
SIGNS="${*:-Glas Tisch Ablehnen}"
cd /home/haipd/DexAvatar
for s in $SIGNS; do
    echo "===== exp2(base+nlf) $s ====="
    bash scripts/exp2_base_nlf_fit.sh "$s" 2>&1 | tail -3
done
echo "ALL EXP2 DONE: $SIGNS"
