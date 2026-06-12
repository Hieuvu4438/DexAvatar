"""
Runner for the VQVAE-hand + DPoser-X-body prior combination.

This is a NEW method (additive). It does not modify any existing runner
(so the SignHPoser + SignBPoser baseline and the other variants
(temporal, hand2d, ensemble, biomech) continue to work unchanged).

Usage:
    python methods/run_dexavatar_wilor_vqvae_dposerx.py \\
        --input_img_folder data/frames \\
        --output_path outputs/method_vqvae_dposerx \\
        --fitting_experiment vqvae_dposerx
"""
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--input_img_folder', type=str, default='')
parser.add_argument('--output_path', type=str, default='')
parser.add_argument('--fitting_experiment', type=str, default='vqvae_dposerx')
args = parser.parse_args()

inp_img_folder = args.input_img_folder
base_output_dir = args.output_path

sub_folder_list = os.listdir(inp_img_folder)
sub_folder_list.sort()

for sub_folder in sub_folder_list:
    input_folder = os.path.abspath(os.path.join(inp_img_folder, sub_folder))
    out_folder = os.path.abspath(os.path.join(base_output_dir, sub_folder))

    os.makedirs(out_folder, exist_ok=True)

    cmd = (
        f"cd {PROJECT_DIR} && "
        f"ROOT_PATH={input_folder} "
        f"OUTPUT_PATH={out_folder} "
        f"FITTING_EXPERIMENT={args.fitting_experiment} "
        f"bash -c 'unset LD_LIBRARY_PATH && bash {SCRIPT_DIR}/Full_running_command_wilor_vqvae_dposerx.sh'"
    )
    print("Running:", cmd)
    os.system(cmd)
