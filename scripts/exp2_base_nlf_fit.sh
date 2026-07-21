#!/bin/bash
# Experiment 1: ISOLATE CONFIG.
# Uses nlf_wilor's DATA (NLF body + WiLoR hand init, HaMeR hand kpts, Sapiens body kpts)
# but fits with paper.yaml (the baseline config: sign_biomechanics ON, global_orient/transl optimized,
# NO direct_optimization). Goal: see if switching config alone recovers the hand error.
#
# If hands recover -> direct.yaml config was the culprit (NLF+WiLoR experts are fine).
# If hands stay bad -> the NLF init itself is the culprit.
set -e
SIGN="${1:?usage: exp2_base_nlf_fit.sh <SIGN>}"
DEX=/home/haipd/DexAvatar
DATA="$DEX/outputs/method_nlf_wilor/${SIGN}"
OUT="$DEX/outputs/exp2_base_nlf/${SIGN}/smplifyx"
FRAMES="$DEX/data/frames/${SIGN}"

source /home/haipd/miniconda3/etc/profile.d/conda.sh
conda activate dexavatar
cd "$DEX/dexavatar_fitting"
export PYTHONPATH="$(pwd)/smplifyx:$(pwd)"
mkdir -p "$OUT"

echo "[exp1] SIGN=$SIGN config=base(baseline) init=nlf/smplx"
python smplifyx/main.py \
    --config cfg_files/fit_smplx_vposer_x.yaml \
    --data_folder "$DATA" \
    --output_folder "$OUT" \
    --img_folder "$FRAMES" \
    --model_folder "$DEX/SMPLer-X/common/utils/human_model_files" \
    --visualize False --split_num 1 --cur_num 0 \
    --smplx_init_dir nlf/smplx \
    --sign_segment ../data/segment.json
echo "[exp1] $SIGN done -> meshes: $(ls "$OUT/meshes/" 2>/dev/null | wc -l)"
