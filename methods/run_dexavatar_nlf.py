#!/usr/bin/env python3
"""NLF body + WiLoR hands entry point.

Mirror of methods/run_dexavatar_wilor_biomech.py with the bash invocation
pointing at Full_running_command_nlf.sh.

Usage:
    python methods/run_dexavatar_nlf.py \
        --input_img_folder data/frames \
        --output_path outputs/nlf_wilor \
        --fitting_experiment /home/haipd/DexAvatar/dexavatar_fitting
"""

import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--input_img_folder', type=str, default='')
parser.add_argument('--output_path',    type=str, default='')
parser.add_argument('--fitting_experiment', type=str, default='')
args = parser.parse_args()

inp_img_folder   = args.input_img_folder
base_output_dir  = args.output_path

sub_folder_list = os.listdir(inp_img_folder)
sub_folder_list.sort()

for sub_folder in sub_folder_list:
    input_folder = os.path.abspath(os.path.join(inp_img_folder, sub_folder))
    out_folder   = os.path.abspath(os.path.join(base_output_dir, sub_folder))
    os.makedirs(out_folder, exist_ok=True)

    cmd = (
        f"cd {PROJECT_DIR} && "
        f"ROOT_PATH={input_folder} "
        f"OUTPUT_PATH={out_folder} "
        f"FITTING_EXPERIMENT={args.fitting_experiment} "
        f"bash -c 'unset LD_LIBRARY_PATH && bash {SCRIPT_DIR}/Full_running_command_nlf.sh'"
    )
    print("Running:", cmd)
    os.system(cmd)
