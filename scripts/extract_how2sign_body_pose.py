#!/usr/bin/env python3
"""
Extract frames từ How2Sign videos và chạy SMPLer-X để lấy body_pose.
Tối ưu disk: chỉ lưu body_pose pkl, xóa frames sau khi xử lý.

Usage:
    python scripts/extract_how2sign_body_pose.py \
        --video_dir /home/shared_data/sign_language/How2Sign/train/subset_2000/raw_videos \
        --output_dir data/signbposer_data/raw/how2sign \
        --max_videos 10  # test với 10 videos trước
"""

import os
import sys
import argparse
import subprocess
import pickle
import shutil
import glob
import numpy as np
from pathlib import Path

# SMPLer-X extraction script path
SMPLERX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'SMPLer-X', 'main')


def extract_frames_from_video(video_path, output_dir, fps=24, max_frames=0):
    """
    Extract frames từ video sử dụng ffmpeg.
    Args:
        max_frames: nếu > 0, chỉ lấy tối đa N frames (uniformly sampled)
    Returns: list of frame paths
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get video duration
    probe_cmd = [
        'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
        '-of', 'csv=p=0', video_path
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    try:
        duration = float(probe_result.stdout.strip())
    except:
        duration = 5.0

    if max_frames > 0 and max_frames < duration * fps:
        # Sample uniformly
        interval = duration / max_frames
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'fps=1/{interval:.4f}',
            '-q:v', '2',
            os.path.join(output_dir, 'frame_%06d.png'),
            '-y', '-loglevel', 'error'
        ]
    else:
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'fps={fps}',
            '-q:v', '2',
            os.path.join(output_dir, 'frame_%06d.png'),
            '-y', '-loglevel', 'error'
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Warning: ffmpeg failed for {video_path}: {result.stderr[:200]}")
        return []

    frames = sorted(glob.glob(os.path.join(output_dir, 'frame_*.png')))
    return frames


def run_smplerx_on_frames(frames_dir, output_dir, gpu_id=0):
    """
    Chạy SMPLer-X trên frames directory.
    Returns: path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Run SMPLer-X inference (must use conda env and correct working directory)
    smplerx_main = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'SMPLer-X', 'main')
    inference_script = os.path.join(smplerx_main, 'inference.py')

    cmd = [
        'conda', 'run', '-n', 'smpler_x', 'python3', inference_script,
        '--num_gpus', '1',
        '--exp_name', 'output',
        '--pretrained_model', 'smpler_x_h32',
        '--agora_benchmark', 'agora_model',
        '--img_path', os.path.abspath(frames_dir),
        '--output_folder', os.path.abspath(output_dir),
        '--show_verts',
        '--show_bbox',
        '--save_mesh',
        '--split_num', '1',
        '--cur_num', '0',
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    result = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=smplerx_main, env=env)

    if result.returncode != 0:
        err_msg = result.stderr[:500] if result.stderr else "unknown error"
        print(f"  Warning: SMPLer-X failed: {err_msg}")
        return None

    return output_dir


def collect_body_poses(smplx_dir):
    """
    Thu thập body_pose từ SMPLer-X output pkl files.
    Returns: list of body_pose arrays (63-dim each)
    """
    body_poses = []

    pkl_files = sorted(glob.glob(os.path.join(smplx_dir, '*.pkl')))

    for pkl_path in pkl_files:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict) and 'body_pose' in data:
                bp = np.array(data['body_pose']).flatten()
                if len(bp) == 63:
                    # Filter outlier
                    if np.linalg.norm(bp) < 10.0:
                        body_poses.append(bp)
        except Exception as e:
            continue

    return body_poses


def process_single_video(video_path, temp_dir, output_dir, gpu_id=0, max_frames=0):
    """
    Xử lý 1 video: extract frames → SMPLer-X → collect body_pose → cleanup
    """
    video_name = Path(video_path).stem
    frames_dir = os.path.join(temp_dir, video_name)
    smplx_dir = os.path.join(output_dir, video_name, 'smplx')

    # Step 1: Extract frames
    frames = extract_frames_from_video(video_path, frames_dir, max_frames=max_frames)
    if not frames:
        return []

    # Step 2: Run SMPLer-X
    run_smplerx_on_frames(frames_dir, os.path.join(output_dir, video_name), gpu_id)

    # Step 3: Collect body poses
    body_poses = collect_body_poses(smplx_dir)

    # Step 4: Cleanup frames (save disk space)
    shutil.rmtree(frames_dir, ignore_errors=True)

    return body_poses


def main():
    parser = argparse.ArgumentParser(description='Extract body_pose from How2Sign videos')
    parser.add_argument('--video_dir', type=str,
                        default='/home/shared_data/sign_language/How2Sign/train/subset_2000/raw_videos')
    parser.add_argument('--output_dir', type=str,
                        default='data/signbposer_data/raw/how2sign')
    parser.add_argument('--temp_dir', type=str,
                        default='/tmp/how2sign_frames')
    parser.add_argument('--max_videos', type=int, default=10,
                        help='Max videos to process (0=all)')
    parser.add_argument('--max_frames', type=int, default=10,
                        help='Max frames per video (0=all, 10=sample uniformly)')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--fps', type=int, default=24,
                        help='Frame extraction FPS')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    # Get video list
    videos = sorted(glob.glob(os.path.join(args.video_dir, '*.mp4')))
    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    print(f"Found {len(videos)} videos to process")
    print(f"Output: {args.output_dir}")
    print(f"Temp: {args.temp_dir}")
    print()

    all_body_poses = []
    failed_videos = []

    for i, video_path in enumerate(videos):
        video_name = Path(video_path).stem
        print(f"[{i+1}/{len(videos)}] Processing: {video_name}")

        try:
            body_poses = process_single_video(
                video_path, args.temp_dir, args.output_dir, args.gpu_id,
                max_frames=args.max_frames)

            if body_poses:
                all_body_poses.extend(body_poses)
                print(f"  → {len(body_poses)} body poses extracted")
            else:
                print(f"  → No body poses extracted")
                failed_videos.append(video_name)
        except Exception as e:
            print(f"  → Error: {e}")
            failed_videos.append(video_name)

    # Save all body poses
    if all_body_poses:
        output_path = os.path.join(args.output_dir, 'body_poses.npy')
        np.save(output_path, np.array(all_body_poses, dtype=np.float32))
        print(f"\nSaved {len(all_body_poses)} body poses to {output_path}")
        print(f"Shape: {np.array(all_body_poses).shape}")

    if failed_videos:
        print(f"\nFailed videos: {len(failed_videos)}")

    # Cleanup temp
    shutil.rmtree(args.temp_dir, ignore_errors=True)

    print("Done!")


if __name__ == '__main__':
    main()
