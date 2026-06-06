#!/bin/bash
# Extract sign name from ROOT_PATH
SIGN_NAME=$(basename "${ROOT_PATH}")

bash -c "source scripts/config_sapiens.sh && bash scripts/S1_sapiens_extract.sh"
wait
echo "Aggregating Sapiens outputs into sapiens.pkl..."
python scripts/aggregate_sapiens.py --sapiens_dir ${OUTPUT_PATH}/sapiens_1b --output_path ${OUTPUT_PATH} --subfolder ${SIGN_NAME}
wait
bash -c "source scripts/config_smplerx.sh && bash scripts/S1_smplerx_extract.sh"
wait
bash -c "source scripts/config.sh && bash scripts/M3.5_wilor_extract.sh"
wait
bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose_temporal.sh"
wait
echo "All finished (Method 1: Temporal Sliding Window)"
