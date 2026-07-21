#!/bin/bash
# Batch runner for remaining Exp1 signs (sequential, single GPU).
SIGNS="${*:-Muell Tisch Dort Ablehnen}"
cd /home/haipd/DexAvatar
for s in $SIGNS; do
    echo "===== exp1 $s ====="
    bash scripts/exp1_paper_nlf_fit.sh "$s" 2>&1 | tail -3
done
echo "ALL EXP1 DONE: $SIGNS"
