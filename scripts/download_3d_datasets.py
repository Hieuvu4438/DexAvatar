#!/usr/bin/env python3
"""
Script to download 3D Sign Language datasets from Google Drive.
Supports downloading individual files and folders with retry and resume.
"""

import os
import sys
import time
import argparse
import datetime
import gdown

DATASETS = [
    {
        "name": "HamNoSys subset with default shape",
        "filename": "hamnosys_pkls_default_shape.zip",
        "type": "file",
        "id": "1_R3KwFyb_RaGmZdPssBJLBIjlhXHaQUQ",
        "url": "https://drive.google.com/file/d/1_R3KwFyb_RaGmZdPssBJLBIjlhXHaQUQ/view",
        "size_gb": 1.12,
    },
    {
        "name": "HamNoSys subset with optimized shape",
        "filename": "hamnosys_pkls_cropFalse_shapeTrue.zip",
        "type": "file",
        "id": "1mFEdYsQaKCZoQbGrQ5xavDWfBsx3CdDq",
        "url": "https://drive.google.com/file/d/1mFEdYsQaKCZoQbGrQ5xavDWfBsx3CdDq/view",
        "size_gb": 1.44,
    },
    {
        "name": "Language-level subset (ASL) with default shape",
        "filename": "how2sign_pkls_default_shape.zip",
        "type": "file",
        "id": "1lwjEx6FuF04ena9JeskvvWdK9x-BVYqu",
        "url": "https://drive.google.com/file/d/1lwjEx6FuF04ena9JeskvvWdK9x-BVYqu/view",
        "size_gb": 13.60,
    },
    {
        "name": "Language-level subset (ASL) with optimized shape",
        "filename": "how2sign_pkls_cropTrue_shapeTrue.zip",
        "type": "file",
        "id": "19Vf5TK2r2w796gMqy_Xk-KqNhpzaR8zY",
        "url": "https://drive.google.com/file/d/19Vf5TK2r2w796gMqy_Xk-KqNhpzaR8zY/view",
        "size_gb": 11.80,
    },
    {
        "name": "Language-level subset (GSL) with default shape",
        "filename": "phonex_pkls_cropFalse_shapeFalse.zip",
        "type": "file",
        "id": "1MCQLQPAP0yI-oi8eQJ3NawgI88zWfasr",
        "url": "https://drive.google.com/file/d/1MCQLQPAP0yI-oi8eQJ3NawgI88zWfasr/view",
        "size_gb": 0.91,
    },
    {
        "name": "Word-level ASL subset",
        "filename": "wlasl_pkls_cropFalse_defult_shape.zip",
        "type": "folder",
        "id": "1JN9l9s5cOg3VE_KL_WY3NETjLTemiyKC",
        "url": "https://drive.google.com/drive/folders/1JN9l9s5cOg3VE_KL_WY3NETjLTemiyKC",
        "size_gb": 0.17,
    },
    {
        "name": "How2sign-synth3D: Greenscreen SMPL-H",
        "filename": "how2sign-synth3D-greenscreen-smplh.zip",
        "type": "file",
        "id": "1zS_X4FbiBnX5Fe9503_Hz3b08zDlrA0l",
        "url": "https://drive.google.com/file/d/1zS_X4FbiBnX5Fe9503_Hz3b08zDlrA0l/view",
        "size_gb": 3.68,
    },
    {
        "name": "How2sign-synth3D: Panoptic SMPL-H",
        "filename": "how2sign-synth3D-panoptic-smplh.zip",
        "type": "file",
        "id": "1VzAdN56Atb5dx3J63TsetuqtYetOJDR1",
        "url": "https://drive.google.com/file/d/1VzAdN56Atb5dx3J63TsetuqtYetOJDR1/view",
        "size_gb": 0.22,
    },
]


def log(msg):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def print_summary():
    print("=" * 85)
    print(f"{'#':<3} | {'Dataset / Subset':<45} | {'File/Folder Name':<35} | {'Size':<10}")
    print("-" * 85)
    total_gb = 0.0
    for idx, item in enumerate(DATASETS, start=1):
        print(f"{idx:<3} | {item['name']:<45} | {item['filename']:<35} | {item['size_gb']:>6.2f} GB")
        total_gb += item['size_gb']
    print("=" * 85)
    print(f"Tổng dung lượng cần tải: {total_gb:.2f} GB (~{round(total_gb)} GB)")
    print("=" * 85)


def download_datasets(output_dir: str, skip_existing: bool = True, max_retries: int = 3):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        log(f"Đã tạo thư mục đích: {output_dir}")

    log(f"Bắt đầu tải các dataset vào: {output_dir}")
    total_items = len(DATASETS)

    for idx, item in enumerate(DATASETS, start=1):
        name = item["name"]
        dest_path = os.path.join(output_dir, item["filename"])
        size_gb = item["size_gb"]

        log(f"[{idx}/{total_items}] Đang xử lý: {name} ({size_gb:.2f} GB)")

        if skip_existing and os.path.exists(dest_path):
            curr_size_gb = os.path.getsize(dest_path) / (1024 ** 3)
            # Check if file has non-trivial size
            if curr_size_gb > 0.01:
                log(f"  -> File đã tồn tại: {dest_path} ({curr_size_gb:.2f} GB). Bỏ qua.")
                continue

        success = False
        for attempt in range(1, max_retries + 1):
            try:
                log(f"  -> Lần thử {attempt}/{max_retries}...")
                if item["type"] == "folder":
                    gdown.download_folder(
                        id=item["id"],
                        output=output_dir,
                        quiet=False,
                        use_cookies=False,
                        resume=True,
                    )
                else:
                    gdown.download(
                        id=item["id"],
                        output=dest_path,
                        quiet=False,
                        fuzzy=True,
                        resume=True,
                    )
                success = True
                log(f"  -> Tải thành công: {item['filename']}")
                break
            except Exception as e:
                log(f"  -> [CẢNH BÁO] Lỗi khi tải: {e}")
                time.sleep(5)

        if not success:
            log(f"  -> [LỖI] Không thể tải {name} sau {max_retries} lần thử.")

    log("=" * 85)
    log("Quá trình tải hoàn tất! Danh sách file trong thư mục:")
    for f in sorted(os.listdir(output_dir)):
        p = os.path.join(output_dir, f)
        if os.path.isfile(p):
            sz = os.path.getsize(p) / (1024 ** 3)
            log(f" - {f}: {sz:.2f} GB")
    log("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tải các dataset 3D Sign Language từ Google Drive")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/haipd/3d_sign_language",
        help="Đường dẫn thư mục lưu dataset",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Chỉ in bảng thống kê dung lượng, không tải",
    )
    args = parser.parse_args()

    print_summary()

    if not args.check_only:
        download_datasets(args.output_dir)
