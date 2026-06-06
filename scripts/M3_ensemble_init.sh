#!/bin/bash
# Method 4: Multi-Model Ensemble Initialization
# Run after all estimators have produced their outputs.
# This selects the best per-frame init from multiple estimators.

set -euo pipefail

# Default estimator dirs (SMPLer-X is always included)
ESTIMATOR_DIRS="${OUTPUT_PATH}/smplerx/smplx"

# Add more estimators if available
if [ -d "${OUTPUT_PATH}/pixie/smplx" ]; then
    ESTIMATOR_DIRS="$ESTIMATOR_DIRS ${OUTPUT_PATH}/pixie/smplx"
fi
if [ -d "${OUTPUT_PATH}/pymaf/smplx" ]; then
    ESTIMATOR_DIRS="$ESTIMATOR_DIRS ${OUTPUT_PATH}/pymaf/smplx"
fi
if [ -d "${OUTPUT_PATH}/osx/smplx" ]; then
    ESTIMATOR_DIRS="$ESTIMATOR_DIRS ${OUTPUT_PATH}/osx/smplx"
fi

echo "Ensemble init from: $ESTIMATOR_DIRS"

cd "${FITTING_EXPERIMENT}"
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/smplifyx
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)

python smplifyx/ensemble_init.py \
    --estimator_dirs $ESTIMATOR_DIRS \
    --sapiens_pkl ${OUTPUT_PATH}/sapiens.pkl \
    --output_dir ${OUTPUT_PATH}/ensemble_smplx \
    --img_folder ${ROOT_PATH}

echo "Ensemble init saved to ${OUTPUT_PATH}/ensemble_smplx"
