"""
Runner for the NLF + WiLoR + SignHPoser + DPoser-X fitting pipeline.
Reuses existing pre-extracted Sapiens/WiLoR/NLF stages.

Usage:
    python methods/run_dexavatar_nlf_dposerx.py \
        --input_img_folder data/frames \
        --output_path outputs/method_nlf_dposerx_hposer
"""
import os
import argparse
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def already_complete(out_folder, min_results=2):
    """Check if the sequence already has enough result pkls."""
    results_dir = os.path.join(out_folder, 'smplifyx', 'results')
    if not os.path.isdir(results_dir):
        return False
    existing = glob.glob(os.path.join(results_dir, '*.pkl'))
    n_img = count_images(out_folder)
    # Consider complete if we have results for at least half the frames
    return len(existing) >= max(min_results, n_img // 2)


def count_images(out_folder):
    """Count images by checking the shared directory structure."""
    shared_name = os.path.basename(out_folder)
    shared_dir = os.path.join(PROJECT_DIR, 'outputs', 'shared', shared_name)
    # Check sapiens_1b for frame count
    sapiens_dir = os.path.join(out_folder, 'sapiens_1b')
    if os.path.isdir(sapiens_dir):
        jsons = glob.glob(os.path.join(sapiens_dir, '*.json'))
        return len(jsons)
    return 0


parser = argparse.ArgumentParser()
parser.add_argument('--input_img_folder', type=str, default='data/frames')
parser.add_argument('--output_path', type=str, default='outputs/method_nlf_dposerx_hposer')
parser.add_argument('--skip_complete', action='store_true', default=True,
                    help='Skip sequences that already have results')
parser.add_argument('--dry_run', action='store_true', default=False,
                    help='Print commands without running')
parser.add_argument('--force', action='store_true', default=False,
                    help='Delete existing SMPLify-X outputs and refit every sequence')
args = parser.parse_args()

inp_img_folder = args.input_img_folder
base_output_dir = args.output_path

sub_folder_list = os.listdir(inp_img_folder)
sub_folder_list.sort()

print(f"Input: {inp_img_folder} ({len(sub_folder_list)} sequences)")
print(f"Output: {base_output_dir}")
print(f"Skip complete: {args.skip_complete}")
print(f"Dry run: {args.dry_run}")
print("=" * 60)

for sub_folder in sub_folder_list:
    input_folder = os.path.abspath(os.path.join(inp_img_folder, sub_folder))
    if not os.path.isdir(input_folder):
        continue
    out_folder = os.path.abspath(os.path.join(base_output_dir, sub_folder))

    if args.force:
        smplifyx_dir = os.path.join(out_folder, 'smplifyx')
        if os.path.isdir(smplifyx_dir):
            import shutil
            shutil.rmtree(smplifyx_dir)
    elif args.skip_complete and already_complete(out_folder):
        results_dir = os.path.join(out_folder, 'smplifyx', 'results')
        n = len(glob.glob(os.path.join(results_dir, '*.pkl')))
        print(f"Skipping {sub_folder}: already has {n} result pkls.")
        continue

    os.makedirs(out_folder, exist_ok=True)

    cmd = (
        f"cd {PROJECT_DIR} && "
        f"ROOT_PATH={input_folder} "
        f"OUTPUT_PATH={out_folder} "
        f"bash methods/Full_running_command_nlf_dposerx.sh"
    )

    if args.dry_run:
        print(f"[DRY RUN] {sub_folder}: {cmd[:120]}...")
    else:
        print(f"Running: {sub_folder}")
        ret = os.system(f"bash -c 'unset LD_LIBRARY_PATH && {cmd}'")
        if ret != 0:
            print(f"  WARNING: {sub_folder} exited with code {ret}")
