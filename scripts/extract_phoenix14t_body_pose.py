#!/usr/bin/env python3
"""
Extract body_pose từ PHOENIX14T dataset.
Chạy SMPLer-X trên PHOENIX14T video frames.

Usage:
    python scripts/extract_phoenix14t_body_pose.py \
        --phoenix_dir data/signbposer_data/raw/phoenix/phoenix-2014-T \
        --output_dir data/signbposer_data/raw/phoenix \
        --max_frames_per_video 10
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

SMPLERX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'SMPLer-X', 'main')


def find_phoenix_videos(phoenix_dir):
    """
    Tìm video files trong PHOENIX14T dataset.
    PHOENIX14T có thể ở dạng video (.mpg) hoặc frames (.png).
    """
    videos = []

    # Tìm video files
    for ext in ['*.mpg', '*.mp4', '*.avi']:
        videos.extend(glob.glob(os.path.join(phoenix_dir, '**', ext), recursive=True))

    # Nếu không có video, tìm frame directories
    if not videos:
        # PHOENIX14T có thể có cấu trúc: features/fullFrame-210x260px/train/*.png
        frame_dirs = []
        for split in ['train', 'dev', 'test']:
            split_dir = os.path.join(phoenix_dir, 'features', 'fullFrame-210x260px', split)
            if os.path.exists(split_dir):
                frame_dirs.append(split_dir)
        return frame_dirs, 'frames'

    return videos, 'videos'


def extract_frames_from_video(video_path, output_dir, max_frames=0):
    """Extract frames từ video."""
    os.makedirs(output_dir, exist_ok=True)

    # Get video duration
    probe_cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                 '-of', 'csv=p=0', video_path]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    try:
        duration = float(probe_result.stdout.strip())
    except:
        duration = 5.0

    if max_frames > 0 and max_frames < duration * 25:
        interval = duration / max_frames
        cmd = ['ffmpeg', '-i', video_path, '-vf', f'fps=1/{interval:.4f}',
               '-q:v', '2', os.path.join(output_dir, 'frame_%06d.png'),
               '-y', '-loglevel', 'error']
    else:
        cmd = ['ffmpeg', '-i', video_path, '-vf', 'fps=25',
               '-q:v', '2', os.path.join(output_dir, 'frame_%06d.png'),
               '-y', '-loglevel', 'error']

    subprocess.run(cmd, capture_output=True, text=True)
    return sorted(glob.glob(os.path.join(output_dir, 'frame_*.png')))


def run_smplerx(frames_dir, output_dir, gpu_id=0):
    """Chạy SMPLer-X trên frames."""
    os.makedirs(output_dir, exist_ok=True)

    smplerx_main = SMPLERX_DIR
    inference_script = os.path.join(smplerx_main, 'inference.py')

    cmd = [
        'conda', 'run', '-n', 'smpler_x', 'python3', inference_script,
        '--num_gpus', '1',
        '--exp_name', 'output',
        '--pretrained_model', 'smpler_x_h32',
        '--agora_benchmark', 'agora_model',
        '--img_path', os.path.abspath(frames_dir),
        '--output_folder', os.path.abspath(output_dir),
        '--show_verts', '--show_bbox', '--save_mesh',
        '--split_num', '1', '--cur_num', '0',
    ]

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    result = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=smplerx_main, env=env)

    if result.returncode != 0:
        return None
    return output_dir


def collect_body_poses(smplx_dir):
    """Thu thập body_pose từ pkl files."""
    body_poses = []
    for pkl_path in sorted(glob.glob(os.path.join(smplx_dir, '*.pkl'))):
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'body_pose' in data:
                bp = np.array(data['body_pose']).flatten()
                if len(bp) == 63 and np.linalg.norm(bp) < 10.0:
                    body_poses.append(bp)
        except:
            continue
    return body_poses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phoenix_dir', type=str,
                        default='data/signbposer_data/raw/phoenix/phoenix-2014-T')
    parser.add_argument('--output_dir', type=str,
                        default='data/signbposer_data/raw/phoenix')
    parser.add_argument('--temp_dir', type=str, default='/tmp/phoenix_frames')
    parser.add_argument('--max_frames_per_video', type=int, default=10)
    parser.add_argument('--max_videos', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)

    # Find PHOENIX data
    items, data_type = find_phoenix_videos(args.phoenix_dir)
    print(f"Found {len(items)} {data_type}")
    print(f"Output: {args.output_dir}")
    print()

    all_poses = []

    if data_type == 'videos':
        if args.max_videos > 0:
            items = items[:args.max_videos]

        for i, video_path in enumerate(items):
            video_name = Path(video_path).stem
            print(f"[{i+1}/{len(items)}] Processing: {video_name}")

            frames_dir = os.path.join(args.temp_dir, video_name)
            smplx_dir = os.path.join(args.output_dir, video_name, 'smplx')

            frames = extract_frames_from_video(video_path, frames_dir,
                                              args.max_frames_per_video)
            if not frames:
                print(f"  → No frames extracted")
                continue

            run_smplerx(frames_dir, os.path.join(args.output_dir, video_name), args.gpu_id)
            poses = collect_body_poses(smplx_dir)
            all_poses.extend(poses)
            print(f"  → {len(poses)} body poses")

            shutil.rmtree(frames_dir, ignore_errors=True)

    elif data_type == 'frames':
        # PHOENIX14T đã có sẵn frames
        for frame_dir in items:
            split_name = Path(frame_dir).name
            print(f"Processing split: {split_name}")

            # Lấy list of video directories (mỗi directory = 1 video)
            video_dirs = sorted([d for d in os.listdir(frame_dir)
                                if os.path.isdir(os.path.join(frame_dir, d))])

            if args.max_videos > 0:
                video_dirs = video_dirs[:args.max_videos]

            for j, video_name in enumerate(video_dirs):
                video_frames_dir = os.path.join(frame_dir, video_name)
                smplx_dir = os.path.join(args.output_dir, split_name, video_name, 'smplx')

                # Sample frames nếu cần
                all_frames = sorted(glob.glob(os.path.join(video_frames_dir, '*.png')))
                if args.max_frames_per_video > 0 and len(all_frames) > args.max_frames_per_video:
                    indices = np.linspace(0, len(all_frames)-1, args.max_frames_per_video, dtype=int)
                    selected_frames = [all_frames[k] for k in indices]

                    # Copy selected frames to temp
                    temp_frames_dir = os.path.join(args.temp_dir, video_name)
                    os.makedirs(temp_frames_dir, exist_ok=True)
                    for frame in selected_frames:
                        shutil.copy2(frame, temp_frames_dir)

                    run_smplerx(temp_frames_dir,
                               os.path.join(args.output_dir, split_name, video_name),
                               args.gpu_id)
                    shutil.rmtree(temp_frames_dir, ignore_errors=True)
                else:
                    run_smplerx(video_frames_dir,
                               os.path.join(args.output_dir, split_name, video_name),
                               args.gpu_id)

                poses = collect_body_poses(smplx_dir)
                all_poses.extend(poses)

                if (j + 1) % 10 == 0:
                    print(f"  [{j+1}/{len(video_dirs)}] {len(poses)} poses (total: {len(all_poses)})")

            print(f"  Split {split_name} done. Total poses: {len(all_poses)}")

    # Save
    if all_poses:
        output_path = os.path.join(args.output_dir, 'body_poses.npy')
        np.save(output_path, np.array(all_poses, dtype=np.float32))
        print(f"\nSaved {len(all_poses)} body poses to {output_path}")

    shutil.rmtree(args.temp_dir, ignore_errors=True)
    print("Done!")


if __name__ == '__main__':
    main()
