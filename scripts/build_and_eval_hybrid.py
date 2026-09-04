#!/usr/bin/env python3
"""
Thực nghiệm Phương án 1 (Hybrid Initializer) cho Stage 1 DexAvatar.

Quy trình:
1. Giữ nguyên 100% code nguồn của DexAvatar và SignEFT-X.
2. Ghép Body & Global Orientation từ SMPLer-X + Hands từ SMPLest-X.
3. Chạy Stage 1 fitting với initializer hybrid.
4. Đánh giá kết quả bằng script chính thức của tác giả:
   data/evaluation_from_author/evaluate_new_fitting.py
"""

from pathlib import Path
import pickle
import subprocess
import json
import numpy as np

WORKSPACE = Path("/home/haipd/DexAvatar")
OUTPUT_EXP = WORKSPACE / "outputs/experiments/hybrid_stage1"
WIRLOR_ROOT = WORKSPACE / "outputs/output_wilor"
SMPLEST_ROOT = WORKSPACE / "SignEFT-X/outputs/smplest_x_stage1_allframes"

FROZEN_INPUTS = (
    "sapiens.pkl",
    "sapiens_1b",
    "hamer",
    "wilor",
    "gender.txt",
    "mean_shape_smplx.npy",
)


def prepare_hybrid_sign(sign: str) -> Path:
    """Tạo cấu trúc thư mục độc lập và khởi tạo hybrid .pkl cho 1 sign."""
    dest_sign = OUTPUT_EXP / sign
    dest_sign.mkdir(parents=True, exist_ok=True)
    src_sign = WIRLOR_ROOT / sign
    smplest_sign = SMPLEST_ROOT / sign

    # 1. Symlink các observations gốc (WiLoR, HaMeR, Sapiens)
    for item in FROZEN_INPUTS:
        src = (src_sign / item).resolve()
        dst = dest_sign / item
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)

    # 2. Tạo thư mục hybrid_init/smplx
    init_dir = dest_sign / "hybrid_init" / "smplx"
    init_dir.mkdir(parents=True, exist_ok=True)

    # 3. Ghép Body từ SMPLer-X và Hands từ SMPLest-X
    smpler_pkls = sorted((src_sign / "smplerx" / "smplx").glob("*.pkl"))
    for spkl in smpler_pkls:
        stem = spkl.stem
        with open(spkl, "rb") as f:
            d_smpler = pickle.load(f)

        smplest_path = smplest_sign / "smplest_x" / "smplx" / f"{stem}.pkl"
        d_hyb = dict(d_smpler)
        if smplest_path.is_file():
            with open(smplest_path, "rb") as f:
                d_smplest = pickle.load(f)
            d_hyb["left_hand_pose"] = d_smplest["left_hand_pose"]
            d_hyb["right_hand_pose"] = d_smplest["right_hand_pose"]

        with open(init_dir / f"{stem}.pkl", "wb") as f:
            pickle.dump(d_hyb, f)

    return dest_sign


def run_fitting(sign: str, gpu: int = 0) -> None:
    """Chạy stage 1 fitting của DexAvatar với hybrid initializer."""
    cmd = [
        "/home/haipd/miniconda3/envs/dexavatar/bin/python",
        "smplifyx/main.py",
        "--config", "cfg_files/fit_smplx_vposer_x.yaml",
        "--data_folder", str(OUTPUT_EXP / sign),
        "--output_folder", str(OUTPUT_EXP / sign / "smplifyx"),
        "--img_folder", str(WORKSPACE / "data/frames" / sign),
        "--model_folder", str(WORKSPACE / "SMPLer-X/common/utils/human_model_files"),
        "--part_segm_fn", "assets/smplx_parts_segm.pkl",
        "--visualize", "False",
        "--split_num", "1",
        "--cur_num", "0",
        "--smplx_init_dir", "hybrid_init/smplx",
    ]
    env = dict(
        PYTHONPATH=f"{WORKSPACE / 'dexavatar_fitting/smplifyx'}:{WORKSPACE / 'dexavatar_fitting'}",
        CUDA_VISIBLE_DEVICES=str(gpu),
    )
    subprocess.run(cmd, cwd=WORKSPACE / "dexavatar_fitting", env=env, check=True)


def evaluate_author(sign: str, method_name: str, folder: Path) -> dict:
    """Chạy script evaluate_new_fitting.py chính thức của tác giả."""
    sign_file = OUTPUT_EXP / f"{sign}_single.txt"
    # Lấy class của sign từ signs.txt gốc
    with open(WORKSPACE / "data/signs.txt") as f:
        cls_map = dict(line.strip().split() for line in f if line.strip())
    sign_file.write_text(f"{sign} {cls_map.get(sign, '0')}\n")

    cmd = [
        "python3",
        str(WORKSPACE / "data/evaluation_from_author/evaluate_new_fitting.py"),
        "--central", "True",
        "--evaluate_folder", str(folder),
        "--gt_folder", str(WORKSPACE / "data/smplx_gt"),
        "--sign_file", str(sign_file),
        "--sign_seg", str(WORKSPACE / "data/evaluation_from_author/data/data/segment.json"),
        "--method", method_name,
    ]
    res = subprocess.run(cmd, cwd=WORKSPACE, capture_output=True, text=True)
    return res.stdout + res.stderr
