"""
Runner for the NLF + WiLoR + VQVAE + DPoser-X fitting pipeline.
Reuses existing pre-extracted Sapiens/WiLoR stages.

Usage:
    python methods/run_dexavatar_nlf_vqvae_dposerx.py \
        --input_img_folder data/frames \
        --output_path outputs/method_nlf_vqvae_dposerx \
        --fitting_experiment /home/haipd/DexAvatar/dexavatar_fitting
"""
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--input_img_folder', type=str, default='data/frames')
parser.add_argument('--output_path', type=str, default='outputs/method_nlf_vqvae_dposerx')
parser.add_argument('--fitting_experiment', type=str, default='/home/haipd/DexAvatar/dexavatar_fitting')
args = parser.parse_args()

inp_img_folder = args.input_img_folder
base_output_dir = args.output_path

sub_folder_list = os.listdir(inp_img_folder)
sub_folder_list.sort()

# Sequences with >1 result pkl are already complete; skip them.
# Sequences with 0 results or only the broken first-frame placeholder
# (created when absolute_depth_loss caused NaN) are reprocessed.
import glob
for sub_folder in sub_folder_list:
    input_folder = os.path.abspath(os.path.join(inp_img_folder, sub_folder))
    if not os.path.isdir(input_folder):
        continue
    out_folder = os.path.abspath(os.path.join(base_output_dir, sub_folder))

    os.makedirs(out_folder, exist_ok=True)

    results_dir = os.path.join(out_folder, 'smplifyx', 'results')
    existing = glob.glob(os.path.join(results_dir, '*.pkl')) if os.path.isdir(results_dir) else []
    if len(existing) > 1:
        print(f"Skipping {sub_folder}: already has {len(existing)} result pkls.")
        continue

    cmd = (
        f"cd {PROJECT_DIR} && "
        f"ROOT_PATH={input_folder} "
        f"OUTPUT_PATH={out_folder} "
        f"FITTING_EXPERIMENT={args.fitting_experiment} "
        f"bash -c 'unset LD_LIBRARY_PATH && bash {SCRIPT_DIR}/Full_running_command_nlf_vqvae_dposerx.sh'"
    )
    print("Running:", cmd)
    os.system(cmd)
