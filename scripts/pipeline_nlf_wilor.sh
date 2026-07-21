#!/bin/bash
# Pipeline: NLF body + WiLoR hands + Original DexAvatar fitting (no VQVAE/DPoserX).
# Reuses pre-extracted Sapiens/WiLoR from shared directory.
# Usage: ROOT_PATH=<img_dir> OUTPUT_PATH=<out_dir> bash scripts/pipeline_nlf_wilor.sh
set -e

SIGN_NAME=$(basename "${ROOT_PATH}")
SHARED_SIGN="/home/haipd/DexAvatar/outputs/shared/${SIGN_NAME}"
FITTING_EXPERIMENT="${FITTING_EXPERIMENT:-/home/haipd/DexAvatar/dexavatar_fitting}"

echo "=== Processing ${SIGN_NAME} ==="

# 1. Link shared pre-extracted stages
mkdir -p "${OUTPUT_PATH}"
echo "  Linking shared stages..."
for item in sapiens_1b sapiens.pkl wilor mean_shape_smplx.npy gender.txt hamer; do
    if [ -e "${SHARED_SIGN}/${item}" ] && [ ! -e "${OUTPUT_PATH}/${item}" ]; then
        ln -sf "${SHARED_SIGN}/${item}" "${OUTPUT_PATH}/${item}"
    fi
done

# 2. NLF body extraction (skip only if complete)
NLF_DIR="${OUTPUT_PATH}/nlf/smplx"
NLF_COMPLETE=false
if [ -d "${NLF_DIR}" ] && [ -n "$(ls -A "${NLF_DIR}" 2>/dev/null)" ]; then
    # Count images in the source folder and compare with pkl count
    N_IMGS=$(ls "${ROOT_PATH}"/*.png "${ROOT_PATH}"/*.jpg 2>/dev/null | wc -l)
    N_PKLS=$(ls "${NLF_DIR}"/*.pkl 2>/dev/null | wc -l)
    if [ "${N_PKLS}" -ge "${N_IMGS}" ]; then
        NLF_COMPLETE=true
    else
        echo "  NLF extraction incomplete (${N_PKLS}/${N_IMGS} pkls), re-running..."
        rm -rf "${NLF_DIR}"
    fi
fi

if [ "${NLF_COMPLETE}" = true ]; then
    echo "  NLF extraction already done (${N_PKLS} pkls), skipping."
else
    echo "  Running NLF body extraction..."
    bash -c "source scripts/config_nlf.sh && bash scripts/S1_nlf_extract.sh"
fi

# 2.5 Temporal smoothing of NLF init (detect and fix transl/orient outliers)
echo "  Running NLF init smoothing..."
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate dexavatar
python scripts/smooth_nlf_init.py --pkl_dir "${OUTPUT_PATH}/nlf/smplx" || echo "  Smoothing skipped (non-critical)"

# 3. Original DexAvatar M4 fitting with NLF body init
echo "  Running Stage 4 fitting (original DexAvatar + NLF init)..."

# Activate dexavatar conda env
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate dexavatar

cd "${FITTING_EXPERIMENT}"
export PYTHONPATH=$(pwd)/smplifyx:$(pwd):${PYTHONPATH:-}

# Config: use the baseline config (fit_smplx_vposer_x.yaml), NOT fit_smplx_vposer_x_direct.yaml.
# direct.yaml freezes global_orient/transl and adds a direct-refinement stage that L2-locks
# hands to the init, which DEGRADES hands despite the NLF+WiLoR init being good. Measured on
# Glas (TR-V2V active-hand, central): direct.yaml LHand/RHand = 15.55/15.31, baseline config
# (this) LHand/RHand = 11.62/9.53, beating the original SMPLer-X+HaMeR baseline (12.26/12.02).
python smplifyx/main.py \
    --config cfg_files/fit_smplx_vposer_x.yaml \
    --data_folder ${OUTPUT_PATH} \
    --output_folder ${OUTPUT_PATH}/smplifyx \
    --img_folder ${ROOT_PATH} \
    --model_folder ../SMPLer-X/common/utils/human_model_files \
    --visualize False \
    --split_num 1 --cur_num 0 \
    --smplx_init_dir nlf/smplx

echo "=== ${SIGN_NAME} complete ==="
