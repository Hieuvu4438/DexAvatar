#!/bin/bash
# Full running command for the NLF + WiLoR variant.
# Pipeline: Sapiens -> NLF body -> WiLoR hands -> default M4 fitting
# Mirror of methods/Full_running_command_wilor_biomech.sh with these swaps:
#   - SMPLer-X extract -> NLF extract
#   - M4 biomech       -> M4 (NLF-init)

# Extract sign name from ROOT_PATH
SIGN_NAME=$(basename "${ROOT_PATH}")

bash -c "source scripts/config_sapiens.sh && bash scripts/S1_sapiens_extract.sh"
wait
echo "Aggregating Sapiens outputs into sapiens.pkl..."
python scripts/aggregate_sapiens.py --sapiens_dir ${OUTPUT_PATH}/sapiens_1b --output_path ${OUTPUT_PATH} --subfolder ${SIGN_NAME}
wait
bash -c "source scripts/config_nlf.sh && bash scripts/S1_nlf_extract.sh"
wait
bash -c "source scripts/config.sh && bash scripts/M3.5_wilor_extract.sh"
wait
bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose_nlf.sh"
wait
echo "All finished (NLF body + WiLoR hands, default M4 fitting)"
