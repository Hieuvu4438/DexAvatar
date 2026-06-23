#!/usr/bin/env python3
"""Runner: Parallel NLF + WiLoR + Original DexAvatar (no VQVAE/DPoserX) for all signs.

Reuses shared pre-extracted Sapiens/WiLoR stages.
Launches multiple signs in parallel using ThreadPoolExecutor.
Each sign's output is logged to its own pipeline.log to avoid interleaved console prints.

Usage:
    python scripts/run_nlf_wilor_parallel.py \
        --input_img_folder data/frames \
        --output_path outputs/method_nlf_wilor \
        --fitting_experiment /home/haipd/DexAvatar/dexavatar_fitting \
        --num_workers 6
"""
import os
import sys
import json
import glob
import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

def run_sign(sub_folder, input_folder, out_folder, expected_frames, fitting_experiment, idx, total):
    start_time = time.time()
    log_path = os.path.join(out_folder, "pipeline.log")
    
    # Prepend OMP_NUM_THREADS/MKL_NUM_THREADS to limit CPU thread thrashing per process
    cmd = (
        f"cd {PROJECT_DIR} && "
        f"export OMP_NUM_THREADS=2 && "
        f"export MKL_NUM_THREADS=2 && "
        f"ROOT_PATH={input_folder} "
        f"OUTPUT_PATH={out_folder} "
        f"FITTING_EXPERIMENT={fitting_experiment} "
        f"bash -c 'unset LD_LIBRARY_PATH && bash {SCRIPT_DIR}/pipeline_nlf_wilor.sh'"
    )
    
    print(f"[{idx}/{total}] START: {sub_folder} (expected: {expected_frames} meshes) -> Log: outputs/method_nlf_wilor/{sub_folder}/pipeline.log")
    
    try:
        with open(log_path, "w") as log_file:
            res = subprocess.run(cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        
        duration = time.time() - start_time
        duration_str = time.strftime('%M:%S', time.gmtime(duration))
        
        if res.returncode == 0:
            print(f"[{idx}/{total}] SUCCESS: {sub_folder} (took {duration_str})")
            return (sub_folder, True, duration_str)
        else:
            print(f"[{idx}/{total}] ERROR: {sub_folder} failed with exit code {res.returncode} (took {duration_str})")
            return (sub_folder, False, f"Exit code {res.returncode}")
    except Exception as e:
        duration = time.time() - start_time
        duration_str = time.strftime('%M:%S', time.gmtime(duration))
        print(f"[{idx}/{total}] EXCEPTION: {sub_folder} failed with error: {str(e)} (took {duration_str})")
        return (sub_folder, False, str(e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img_folder', type=str, default='data/frames')
    parser.add_argument('--output_path', type=str, default='outputs/method_nlf_wilor')
    parser.add_argument('--fitting_experiment', type=str, default='/home/haipd/DexAvatar/dexavatar_fitting')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of parallel signs to process')
    args = parser.parse_args()

    inp_img_folder = os.path.join(PROJECT_DIR, args.input_img_folder)
    base_output_dir = os.path.join(PROJECT_DIR, args.output_path)

    # Load GT segment to compute expected frame count per sign
    with open(os.path.join(PROJECT_DIR, 'data', 'segment.json')) as f:
        frame_seg = json.load(f)

    sub_folder_list = sorted(os.listdir(inp_img_folder))
    
    # First filter out valid directories and count total
    valid_folders = []
    for sub_folder in sub_folder_list:
        input_folder = os.path.abspath(os.path.join(inp_img_folder, sub_folder))
        if os.path.isdir(input_folder):
            valid_folders.append((sub_folder, input_folder))
            
    total = len(valid_folders)
    print(f"Found {total} sign directories in {inp_img_folder}")
    print(f"Using {args.num_workers} parallel workers...")

    # Determine which ones need running
    to_run = []
    skipped_count = 0
    for idx, (sub_folder, input_folder) in enumerate(valid_folders, 1):
        out_folder = os.path.abspath(os.path.join(base_output_dir, sub_folder))
        os.makedirs(out_folder, exist_ok=True)

        seg = frame_seg.get(sub_folder, [0, 0])
        expected_frames = (seg[1] - seg[0]) // 2 + 1

        meshes_dir = os.path.join(out_folder, 'smplifyx', 'meshes')
        if os.path.isdir(meshes_dir):
            existing_meshes = glob.glob(os.path.join(meshes_dir, '*.obj'))
            mesh_count = len(existing_meshes)
        else:
            mesh_count = 0

        if mesh_count == expected_frames:
            print(f"[{idx}/{total}] SKIP: {sub_folder} (already complete with {mesh_count} meshes)")
            skipped_count += 1
        else:
            to_run.append({
                'sub_folder': sub_folder,
                'input_folder': input_folder,
                'out_folder': out_folder,
                'expected_frames': expected_frames,
                'idx': idx
            })

    print(f"\nSkipped: {skipped_count}/{total} already complete. Need to run: {len(to_run)}/{total} signs.\n")

    if not to_run:
        print("All signs are already processed!")
        return

    # Run in parallel
    results = []
    start_time_all = time.time()
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                run_sign,
                job['sub_folder'],
                job['input_folder'],
                job['out_folder'],
                job['expected_frames'],
                args.fitting_experiment,
                job['idx'],
                total
            ): job['sub_folder'] for job in to_run
        }
        
        for future in as_completed(futures):
            sub_folder = futures[future]
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Future for {sub_folder} raised an exception: {e}")
                results.append((sub_folder, False, str(e)))

    # Print summary
    success_list = [r[0] for r in results if r[1]]
    failed_list = [r for r in results if not r[1]]
    
    elapsed_all = time.time() - start_time_all
    elapsed_all_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_all))
    
    print("\n=================== Execution Summary ===================")
    print(f"Total time elapsed: {elapsed_all_str}")
    print(f"Successfully processed: {len(success_list)} signs")
    print(f"Failed: {len(failed_list)} signs")
    if failed_list:
        print("Failures:")
        for name, _, err in failed_list:
            print(f"  - {name}: {err}")
    print("=========================================================")

if __name__ == "__main__":
    main()
