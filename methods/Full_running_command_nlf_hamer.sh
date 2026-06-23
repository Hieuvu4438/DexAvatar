#!/bin/bash
# Full running command for the NLF + HaMeR variant.
# Pipeline: Sapiens -> NLF body -> HaMeR hands -> default M4 fitting
# Mirror of methods/Full_running_command.sh with these swaps:
#   - SMPLer-X extract -> NLF extract
#   - default M4       -> M4 (NLF-init)

# Extract sign name from ROOT_PATH
SIGN_NAME=$(basename "${ROOT_PATH}")

bash -c "source scripts/config_sapiens.sh && bash scripts/S1_sapiens_extract.sh"
wait
echo "Aggregating Sapiens outputs into sapiens.pkl..."
python scripts/aggregate_sapiens.py --sapiens_dir ${OUTPUT_PATH}/sapiens_1b --output_path ${OUTPUT_PATH} --subfolder ${SIGN_NAME}
wait
bash -c "source scripts/config_nlf.sh && bash scripts/S1_nlf_extract.sh"
wait
bash -c "source scripts/config.sh && bash scripts/M3.5_hamer_extract.sh"
wait
bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose_nlf.sh"
wait
echo "All finished (NLF body + HaMeR hands, default M4 fitting)"
