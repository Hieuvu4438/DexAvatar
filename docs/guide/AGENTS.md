# Agent Context: DexAvatar Project

This document provides essential context for AI agents to understand the DexAvatar framework, environment management, and execution workflow.

## 1. Project Overview
**DexAvatar** is a framework for 3D Sign Language Reconstruction from monocular videos, focusing on bio-mechanically accurate hand and body poses using 3D priors (SignBPoser and SignHPoser). 
- **Paper:** DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors [WACV 2026]
- **Core Technology:** SMPL-X model, Sapiens-lite, SMPLer-X, and custom motion priors.

## 2. Environment Matrix
The project operates across **three separate Conda environments**. AI should verify the active environment before suggesting commands:

| Environment Name | Python | Key Purpose |
| :--- | :--- | :--- |
| `dexavatar` | 3.10 | Main execution and fitting logic. |
| `sapiens_lite` | (per repo) | Person detection and whole-body pose estimation. |
| `smpler_x` | 3.8 | 3D human shape and pose estimation (SMPLer-X). |

> **Note:** Always run `bash scripts/bug_fix_dexavatar.sh` after setting up `dexavatar` and `smpler_x` environments.

## 3. Critical Directory Structure
AI agents must respect the following layout for data and checkpoints:

```text
DexAvatar/
├── data/
│   └── images_sgnify/          # Sign language video frames
│       └── [sign_id]/images/   # Input .png files
├── checkpoints/
│   ├── smpler_x_h32.pth.tar    # SMPLer-X weight
│   └── mmdet/                  # Detector weights
├── SMPLer-X/
│   └── common/utils/human_model_files/ # SMPL-X model files
├── dexavatar_fitting/
│   └── smplifyx/
│       ├── signbposer/         # Body pose priors
│       └── signhposer/         # Hand pose priors
└── sapiens/
    └── lite/torchscript/       # Sapiens models (RTMPose & 1B)

4. Execution Workflow
To help the user run the project, follow these steps in order:

Preprocessing (Detection/Pose): Requires sapiens_lite and smpler_x.

Fitting (DexAvatar Core): - Switch to dexavatar environment.

Command template:

Bash
python run_dexavatar.py \
  --input_img_folder ./data/images_sgnify/[SIGN_NAME]/images \
  --output_path ./output/[SIGN_NAME] \
  --fitting_experiment ./dexavatar_fitting
5. Key Dependencies & Quirks
Sapiens-lite: Uses torchscript models for high-performance inference.

SMPLer-X: Requires specific mmcv-full==1.7.1 and torch==1.12.0+cu116.

Segmentation: Pre-defined segmentations for SGNify are in ./data. Custom data requires external generation (refer to project README).

6. Citation Reference
Đoạn mã
@article{kundu2025dexavatar,
  title={DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors},
  author={Kundu, Kaustubh et al.},
  journal={arXiv preprint arXiv:2512.21054},
  year={2025}
}

---

### Cách sử dụng file này:
1. Bạn tạo một file mới tên là `AGENTS.md` ngay tại thư mục gốc của project.
2. Dán nội dung trên vào.
3. Nếu bạn dùng **Cursor**, bạn có thể gõ `@AGENTS.md` trong khung chat để AI tập trung vào các hướng dẫn này.
4. Nếu bạn dùng **GitHub Copilot**, nó sẽ tự động quét file này nếu bạn mở nó trong editor.

### Một vài lưu ý nhỏ cho bạn khi chạy:
* **Vấn đề Môi trường:** Vì project này dùng tới 3 môi trường khác nhau, khi bạn nhờ AI viết script chạy tự động, hãy nhắc nó sử dụng lệnh `conda run -n [tên_môi_trường] python ...` để tránh việc phải deactivate/activate thủ công.
* **Đường dẫn:** Hãy đảm bảo bạn đã chạy các lệnh `bash scripts/...` để fix các lỗi import (thường là do đường dẫn tuyệt đối/tương đối trong các thư viện con).
