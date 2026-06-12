#!/bin/bash
# Pipeline orchestrator for the VQVAE-hand + DPoser-X-body prior combination.
# Mirrors methods/Full_running_command_wilor.sh but uses the new
# `vqvae_dposerx` config that sets use_hposer3d=False, use_signbposer=False,
# use_vqvae_hand=True, use_dposerx_body=True.

set -e

SIGN_NAME=$(basename "${ROOT_PATH}")

# Stage 1A: Sapiens extraction (body keypoints, hand keypoints)
bash -c "source scripts/config_sapiens.sh && bash scripts/S1_sapiens_extract.sh"
wait
echo "Aggregating Sapiens outputs into sapiens.pkl..."
python scripts/aggregate_sapiens.py --sapiens_dir ${OUTPUT_PATH}/sapiens_1b --output_path ${OUTPUT_PATH} --subfolder ${SIGN_NAME}
wait

# Stage 1B: SMPLer-X extraction (SMPL-X init)
bash -c "source scripts/config_smplerx.sh && bash scripts/S1_smplerx_extract.sh"
wait

# Stage 3.5: WiLoR extraction (3D hand init)
bash -c "source scripts/config.sh && bash scripts/M3.5_wilor_extract.sh"
wait

# Stage 4: SMPLify-X pose fitting with VQVAE + DPoser-X priors
bash -c "source scripts/config.sh && bash scripts/M4_smplifyx_pose_vqvae_dposerx.sh"
wait

echo "All finished"
