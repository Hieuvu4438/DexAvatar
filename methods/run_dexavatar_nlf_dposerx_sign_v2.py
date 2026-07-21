"""
Runner for the v2 NLF + WiLoR + VQVAE-hand + SIGN-TRAINED DPoser-X body pipeline.

v2 = rebalanced prior/data weights + NLF-init smoothing + root-orientation
optimization, to fix the poor body fits of method_nlf_dposerx_sign. Reuses the
pre-extracted Sapiens/WiLoR/NLF stages; writes to a SEPARATE output folder so the
old method is never affected.

Usage:
    python methods/run_dexavatar_nlf_dposerx_sign_v2.py \\
        --input_img_folder data/frames \\
        --output_path outputs/method_nlf_dposerx_sign_v2 \\
        --fitting_experiment /home/haipd/DexAvatar/dexavatar_fitting
"""
import os
import argparse
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--input_img_folder', type=str, default='data/frames')
parser.add_argument('--output_path', type=str,
                    default='outputs/method_nlf_dposerx_sign_v2')
parser.add_argument('--fitting_experiment', type=str,
                    default='/home/haipd/DexAvatar/dexavatar_fitting')
args = parser.parse_args()

inp_img_folder = args.input_img_folder
base_output_dir = args.output_path

sub_folder_list = os.listdir(inp_img_folder)
sub_folder_list.sort()

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
        f"bash -c 'unset LD_LIBRARY_PATH && bash {SCRIPT_DIR}/Full_running_command_nlf_dposerx_sign_v2.sh'"
    )
    print("Running:", cmd)
    os.system(cmd)
