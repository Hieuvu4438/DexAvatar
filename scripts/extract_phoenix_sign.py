#!/usr/bin/env python3
"""
Extract SMPL body_pose (21x3 axis-angle = 63-dim) from PHOENIX-2014-T for the
sign-language DPoser-X body prior.

READ-ONLY on the source dataset:
    /home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T
We only glob + read PNG frames there. We never write to /home/dongvk.

WRITE only under /home/haipd/DexAvatar/data:
    data/signbposer_data/raw/phoenix_sign/<split>/{smplx/*.pkl, body_poses.npy}
    data/_tmp/phoenix_<split>/... (temp frames, cleaned up)

Efficiency vs. the existing per-clip extractor:
    The legacy extract_phoenix14t_body_pose.py spawns one SMPLer-X process per
    clip (~8257 clips -> days of model re-init). Here we batch ALL sampled frames
    of a split into a SINGLE SMPLer-X call, so the whole job is 3 calls (one per
    split). Frame files are renamed <clip>__<frame>.png so identities stay unique.

Sampling: <= N frames per clip, uniformly (default 3 -> "subset").

Usage:
    python scripts/extract_phoenix_sign.py \\
        --frames_per_clip 3 --gpu_id 0
"""
import os
import sys
import argparse
import subprocess
import pickle
import shutil
import glob
from pathlib import Path

import numpy as np

# --- paths (defaults honor the read-only / write-only contract) ---
PHOENIX_SRC_DEFAULT = (
    "/home/dongvk/datasets/phoenix14T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T"
)
OUT_DEFAULT = "/home/haipd/DexAvatar/data/signbposer_data/raw/phoenix_sign"
TMP_DEFAULT = "/home/haipd/DexAvatar/data/_tmp/phoenix_sign"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMPLERX_MAIN = os.path.join(REPO_ROOT, "SMPLer-X", "main")

SPLITS = ["train", "dev", "test"]  # PHOENIX split names (dev == validation)


def sample_clip_frames(clip_dir, frames_per_clip):
    """Return <= frames_per_clip uniformly-sampled PNG paths from a clip dir (READ-only)."""
    frames = sorted(glob.glob(os.path.join(clip_dir, "*.png")))
    if not frames:
        return []
    if 0 < frames_per_clip < len(frames):
        idx = np.linspace(0, len(frames) - 1, frames_per_clip, dtype=int)
        frames = [frames[k] for k in idx]
    return frames


def stage_split_frames(src_split_dir, tmp_dir, frames_per_clip, max_clips=0):
    """Copy <= frames_per_clip frames per clip into tmp_dir as <clip>_<idx>.png.

    READ-only on src; WRITE only to tmp_dir (under DexAvatar/data).
    Filenames are <clip>_<NNNNNN>.png so SMPLer-X's image-sort lambda
    (int(fn.split('_')[-1])) works -- PHOENIX native names like 'images0072'
    have no trailing int and crash it. The tmp dir is wiped first so stale
    files from a prior failed run don't leak in.
    Returns the number of staged frames.
    """
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)
    clips = sorted(d for d in os.listdir(src_split_dir)
                   if os.path.isdir(os.path.join(src_split_dir, d)))
    if max_clips > 0:
        clips = clips[:max_clips]
    counter = 0
    for clip in clips:
        clip_dir = os.path.join(src_split_dir, clip)
        for fr in sample_clip_frames(clip_dir, frames_per_clip):
            dst = os.path.join(tmp_dir, f"{clip}_{counter:06d}.png")
            shutil.copy2(fr, dst)  # src -> tmp (never the reverse)
            counter += 1
    return counter


def run_smplerx(img_path, output_dir, gpu_id=0):
    """Run SMPLer-X inference once over a flat folder of images (one model init)."""
    os.makedirs(output_dir, exist_ok=True)
    inference_script = os.path.join(SMPLERX_MAIN, "inference.py")
    cmd = [
        # --no-capture-output: stream SMPLer-X output instead of conda buffering it.
        "conda", "run", "--no-capture-output", "-n", "smpler_x",
        "python3", inference_script,
        "--num_gpus", "1",
        "--exp_name", "output",
        "--pretrained_model", "smpler_x_h32",
        "--agora_benchmark", "agora_model",
        "--img_path", os.path.abspath(img_path),
        "--output_folder", os.path.abspath(output_dir),
        # NOTE: --show_verts/--show_bbox/--save_mesh dropped -- we only need the
        # smplx/*.pkl (body_pose). Mesh saving dominated runtime (~70 frames/min).
        "--split_num", "1", "--cur_num", "0",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # SMPLer-X inference.py does `from utils.inference_utils import ...`; put main/ on
    # the path explicitly so the import resolves regardless of how `conda run` sets cwd.
    env["PYTHONPATH"] = SMPLERX_MAIN + os.pathsep + env.get("PYTHONPATH", "")
    # If the parent shell has a conda env active (e.g. dexavatar) that ships a compiler
    # package, `conda run -n smpler_x` re-activation can abort with
    # "This cross-compiler package contains no program ...-g++". Clear conda-active and
    # compiler vars so smpler_x activates cleanly from any parent.
    for k in list(env):
        if (k in {"CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL",
                  "CONDA_PROMPT_MODIFIER", "CC", "CXX", "CFLAGS", "CXXFLAGS",
                  "LDFLAGS", "CONDA_BUILD_SYSROOT", "CONDA_BUILD_CROSS_COMPILATION",
                  "HOST", "BUILD"} or k.startswith("CONDA_BACKUP")):
            del env[k]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=SMPLERX_MAIN, env=env)
    if result.returncode != 0:
        print("  SMPLer-X STDOUT (tail):")
        print("\n".join((result.stdout or "").splitlines()[-25:]))
        print("  SMPLer-X STDERR (tail):")
        print("\n".join((result.stderr or "").splitlines()[-25:]))
        return False
    return True


def collect_body_poses(smplx_dir):
    """Collect 63-dim body_pose vectors from SMPLer-X pkls (mirrors legacy collector)."""
    out = []
    for pkl_path in sorted(glob.glob(os.path.join(smplx_dir, "*.pkl"))):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "body_pose" in data:
                bp = np.array(data["body_pose"]).flatten()
                if len(bp) == 63 and np.linalg.norm(bp) < 10.0:
                    out.append(bp.astype(np.float32))
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phoenix_dir", default=PHOENIX_SRC_DEFAULT,
                    help="READ-only PHOENIX-2014-T root (contains features/fullFrame-210x260px).")
    ap.add_argument("--output_dir", default=OUT_DEFAULT,
                    help="WRITE destination under DexAvatar/data.")
    ap.add_argument("--temp_dir", default=TMP_DEFAULT,
                    help="WRITE temp dir for staged frames (cleaned up).")
    ap.add_argument("--frames_per_clip", type=int, default=3)
    ap.add_argument("--max_clips", type=int, default=0, help="0 = all clips per split.")
    ap.add_argument("--splits", nargs="*", default=SPLITS, help="Subset of splits to run.")
    ap.add_argument("--gpu_id", type=int, default=0)
    args = ap.parse_args()

    frames_root = os.path.join(args.phoenix_dir, "features", "fullFrame-210x260px")
    if not os.path.isdir(frames_root):
        sys.exit(f"READ source not found (expected features/fullFrame-210x260px): "
                 f"{frames_root}")

    print(f"READ (source, never written): {args.phoenix_dir}")
    print(f"WRITE (outputs):              {args.output_dir}")
    print(f"WRITE (temp, cleaned):         {args.temp_dir}")
    print(f"frames_per_clip={args.frames_per_clip}  gpu_id={args.gpu_id}")

    for split in args.splits:
        src_split_dir = os.path.join(frames_root, split)
        if not os.path.isdir(src_split_dir):
            print(f"\n[SKIP] split '{split}' not found at {src_split_dir}")
            continue

        out_split = os.path.join(args.output_dir, split)
        tmp_split = os.path.join(args.temp_dir, split)

        print(f"\n=== split {split} === staging frames (READ {src_split_dir})")
        n_staged = stage_split_frames(src_split_dir, tmp_split,
                                      args.frames_per_clip, args.max_clips)
        print(f"  staged {n_staged} frames -> {tmp_split}")
        if n_staged == 0:
            shutil.rmtree(tmp_split, ignore_errors=True)
            continue

        print(f"  running SMPLer-X (single batched call, GPU {args.gpu_id})...")
        ok = run_smplerx(tmp_split, out_split, args.gpu_id)
        if not ok:
            print(f"  [WARN] SMPLer-X failed for split {split}; keeping temp for debug.")
            continue

        poses = collect_body_poses(os.path.join(out_split, "smplx"))
        if poses:
            arr = np.stack(poses, axis=0)
            np.save(os.path.join(out_split, "body_poses.npy"), arr)
            print(f"  saved {arr.shape} -> {out_split}/body_poses.npy")
        else:
            print(f"  [WARN] no valid body poses for split {split}")

        shutil.rmtree(tmp_split, ignore_errors=True)

    print("\nDone. Outputs under:", args.output_dir)


if __name__ == "__main__":
    main()
