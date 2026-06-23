#!/bin/bash
# Pipeline orchestrator: NLF + WiLoR + SignHPoser + DPoser-X.
# Reuses existing pre-extracted Sapiens, WiLoR, and NLF stages via symbolic links.
# Usage: ROOT_PATH=<img_dir> OUTPUT_PATH=<out_dir> bash methods/Full_running_command_nlf_dposerx.sh
set -e

SIGN_NAME=$(basename "${ROOT_PATH}")
SHARED_SIGN="/home/haipd/DexAvatar/outputs/shared/${SIGN_NAME}"
# Reuse NLF from an existing run if available (avoid re-extraction).
# Use NLF init from method_nlf_wilor — it has correct camera translation
# (Z ≈ 17.8m, matching SMPLer-X).  method_nlf_vqvae_dposerx has broken
# camera init (Z ≈ 0.085m) which puts the mesh 8.5cm from the camera.
NLF_SOURCE="/home/haipd/DexAvatar/outputs/method_nlf_wilor/${SIGN_NAME}"

echo "=========================================="
echo "Processing sign: ${SIGN_NAME}"
echo "=========================================="

# Ensure output directory exists
mkdir -p "${OUTPUT_PATH}"

# 1. Link shared pre-extracted stages (SAPIENS, WiLoR, HaMeR, mean_shape)
echo "[1/4] Linking shared pre-extracted stages..."
for item in sapiens_1b sapiens.pkl wilor mean_shape_smplx.npy gender.txt hamer; do
    if [ -e "${SHARED_SIGN}/${item}" ] && [ ! -e "${OUTPUT_PATH}/${item}" ]; then
        ln -sf "${SHARED_SIGN}/${item}" "${OUTPUT_PATH}/${item}"
        echo "  linked ${item}"
    fi
done

# 2. NLF body extraction (reuse from existing run if available)
echo "[2/4] Setting up NLF body init..."
if [ -d "${NLF_SOURCE}/nlf/smplx" ] && [ -n "$(ls -A "${NLF_SOURCE}/nlf/smplx" 2>/dev/null)" ]; then
    # Reuse NLF from existing run (always update symlink).
    mkdir -p "${OUTPUT_PATH}/nlf"
    rm -f "${OUTPUT_PATH}/nlf/smplx"
    ln -sf "${NLF_SOURCE}/nlf/smplx" "${OUTPUT_PATH}/nlf/smplx"
    echo "  linked NLF from ${NLF_SOURCE}/nlf/smplx"
else
    # Run NLF extraction fresh
    echo "  Running NLF body extraction..."
    bash -c "source scripts/config_nlf.sh && bash scripts/S1_nlf_extract.sh"
fi

# 2.5 Temporal smoothing of NLF init
echo "[2.5/4] Smoothing NLF init..."
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate dexavatar 2>/dev/null || true
python scripts/smooth_nlf_init.py --pkl_dir "${OUTPUT_PATH}/nlf/smplx" || echo "  Smoothing skipped (non-critical)"

# 3. Stage 4 fitting: SignHPoser + DPoser-X
echo "[3/4] Running Stage 4 fitting (SignHPoser + DPoser-X)..."
bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose_nlf_dposerx.sh"

echo "[4/4] Sign ${SIGN_NAME} complete."
