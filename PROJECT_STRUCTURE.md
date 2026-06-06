# Project Structure

> **Quy tắc:** Mọi file mới PHẢI được đặt đúng thư mục theo quy định dưới đây. Không được tạo file `.py` hoặc `.sh` ở thư mục gốc.

---

## Cấu trúc thư mục

```
DexAvatar/
├── PROJECT_STRUCTURE.md           # ← File này
├── README.md                      # Hướng dẫn cài đặt & sử dụng
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
├── .gitignore
│
├── runners/                       # ★ Entry points (chạy pipeline)
├── pipelines/                     # ★ Pipeline orchestrators
├── evaluation/                    # ★ Evaluation scripts & metrics
├── scripts/                       # Pipeline stages & utilities
├── methods/                       # Method variants (Temporal, Hand2D, etc.)
│
├── dexavatar_fitting/             # ★ CORE: Fitting engine
├── SMPLer-X/                      # 3D body estimation
├── SGNify/                        # Sign language processing
├── docs/                          # Documentation
├── assets/                        # Static assets (images, etc.)
│
├── data/                          # Input data (gitignored)
├── outputs/                       # Runtime outputs (gitignored)
├── checkpoints/                   # Model weights (gitignored)
└── _DATA/                         # Shared checkpoints (gitignored)
```

---

## Chi tiết từng thư mục

### `runners/` — Entry Points

**Quy tắc:** Chỉ chứa script **chạy pipeline** từ đầu đến cuối. Mỗi file = 1 cách chạy.

```
runners/
├── run_dexavatar_wilor.py         # Entry point chính: WiLoR pipeline
├── run_direct_full.sh             # Direct optimization (A+D+E)
├── run_hamer_fitting.sh           # Chỉ SMPLify-X fitting (có hamer.pkl)
└── run_hamer_full.sh              # Full HaMeR pipeline (extraction + fitting)
```

**Cách dùng:**
```bash
python runners/run_dexavatar_wilor.py --input_img_folder data/frames --output_path outputs/output_wilor
bash runners/run_direct_full.sh
```

---

### `pipelines/` — Pipeline Orchestrators

**Quy tắc:** Chứa shell script orchestrate **nhiều stage** (Sapiens → SMPLer-X → WiLoR → SMPLify-X).

```
pipelines/
└── Full_running_command_wilor.sh  # Pipeline đầy đủ cho 1 ký hiệu
```

**Lưu ý:** Các variant (biomech, ensemble, hand2d, temporal) nằm trong `methods/`.

---

### `evaluation/` — Evaluation Scripts

**Quy tắc:** Chứa **tất cả** script đánh giá metrics (TR-V2V, MPVPE).

```
evaluation/
├── evaluation_trv2v_wilor.py      # ★ TR-V2V metric (chính)
├── evaluation_mpvpe_correct.py    # MPVPE metric (corrected version)
├── eval_mpvpe_common_frames.py    # MPVPE trên common frames (fair comparison)
├── eval_mpvpe_regions.py          # MPVPE theo vùng body
└── eval_wilor_full.sh             # Shell chạy eval 57 signs
```

**Cách dùng:**
```bash
python evaluation/evaluation_trv2v_wilor.py \
    --pred_root outputs/output_wilor \
    --gt_root data/smplx_gt \
    --signs_txt data/signs.txt \
    --method_name DexAvatar-WiLoR
```

---

### `scripts/` — Pipeline Stages & Utilities

**Quy tắc:** Chứa script chạy **một stage** cụ thể hoặc utility functions.

```
scripts/
├── config.sh                      # Activate conda env dexavatar
├── config_sapiens.sh              # Activate conda env sapiens_fix
├── config_smplerx.sh              # Activate conda env smpler_x
│
├── S1_sapiens_extract.sh          # Stage 1A: Sapiens extraction
├── S1_smplerx_extract.sh          # Stage 1B: SMPLer-X extraction
├── aggregate_sapiens.py           # Stage 1B: Gộp Sapiens outputs
│
├── M3_mean_shape_smplerx.py       # Stage 3: Tính mean shape
├── M3.5_hamer_extract.sh          # Stage 3.5: HaMeR extraction
├── M3.5_wilor_extract.sh          # Stage 3.5: WiLoR extraction
├── M3_ensemble_init.sh            # Stage 3: Ensemble initialization
│
├── M4_smplifyx_pose.sh            # Stage 4: SMPLify-X fitting (baseline)
├── M4_smplifyx_pose_direct.sh     # Stage 4: Direct optimization
├── M4_smplifyx_pose_biomech.sh    # Stage 4: Biomechanics constraints
├── M4_smplifyx_pose_ensemble.sh   # Stage 4: Ensemble init
├── M4_smplifyx_pose_hand2d.sh     # Stage 4: 2D hand supervision
├── M4_smplifyx_pose_temporal.sh   # Stage 4: Temporal window
│
├── eval_all_methods.sh            # Utility: Eval tất cả methods
├── eval_baseline_when_ready.sh    # Utility: Eval baseline khi sẵn sàng
├── fit_parallel.sh                # Utility: Chạy fitting song song
├── fit_remaining.sh               # Utility: Chạy fitting cho signs còn lại
└── ...
```

---

### `methods/` — Method Variants

**Quy tắc:** Chứa **pair** (Python runner + Shell orchestrator) cho mỗi method variant.

```
methods/
├── run_dexavatar.py                        # Baseline (HaMeR)
├── Full_running_command.sh                 # Pipeline orchestrator (baseline)
│
├── run_dexavatar_wilor_biomech.py          # WiLoR + Biomechanics
├── Full_running_command_wilor_biomech.sh   # Pipeline orchestrator
│
├── run_dexavatar_wilor_ensemble.py         # WiLoR + Ensemble Init
├── Full_running_command_wilor_ensemble.sh  # Pipeline orchestrator
│
├── run_dexavatar_wilor_hand2d.py           # WiLoR + 2D Hand Supervision
├── Full_running_command_wilor_hand2d.sh    # Pipeline orchestrator
│
├── run_dexavatar_wilor_temporal.py         # WiLoR + Temporal Window
├── Full_running_command_wilor_temporal.sh  # Pipeline orchestrator
│
├── eval_hand2d_full.sh                     # Eval Hand2D variant
└── eval_temporal_full.sh                   # Eval Temporal variant
```

---

### `dexavatar_fitting/` — Core Fitting Engine

**Quy tắc:** Không sửa đổi trừ khi hiểu rõ impact.

```
dexavatar_fitting/
├── script.py                       # Main fitting script
├── script_direct.py                # Direct optimization variant
├── script_hand2d.py                # 2D hand supervision variant
├── script_temporal.py              # Temporal window variant
├── rewrite_body_model.py           # SMPL-X body model modifications
│
├── cfg_files/                      # Configuration files
│   ├── fit_smplx_vposer_x.yaml    # Baseline config
│   ├── fit_smplx_vposer_x_biomech.yaml
│   ├── fit_smplx_vposer_x_direct.yaml
│   ├── fit_smplx_vposer_x_ensemble.yaml
│   ├── fit_smplx_vposer_x_hand2d.yaml
│   └── fit_smplx_vposer_x_temporal.yaml
│
├── smplifyx/                       # Fitting engine core
│   ├── main.py                     # Main loop: load → iterate → fit
│   ├── fit_single_frame.py         # Single-frame optimization
│   ├── fitting.py                  # ★ Loss functions (QUAN TRỌNG)
│   ├── data_parser.py              # Data loading
│   ├── cmd_parser.py               # CLI argument parsing
│   ├── prior.py                    # Pose priors (L2, angle, GMM)
│   ├── camera.py                   # Camera model
│   ├── utils.py                    # Utilities
│   ├── ensemble_init.py            # Ensemble initialization
│   ├── fit_temporal_window.py      # Temporal sliding window
│   ├── signbposer/                 # Body pose prior (VAE)
│   └── signhposer/                 # Hand pose prior (VAE)
│
└── assets/                         # Region indices
    ├── smplx_upper_body_minus_face_vidx.npy
    ├── smplx_left_hand_vidx.npy
    └── smplx_right_hand_vidx.npy
```

---

### `docs/` — Documentation

```
docs/
├── CODEBASE_GUIDE.md               # ★ Hướng dẫn đọc codebase
├── PROJECT_STRUCTURE.md            # ← File này (cấu trúc project)
├── eval_setup.md                   # Hướng dẫn evaluation
├── RESEARCH_REPORT.md              # Báo cáo nghiên cứu
├── research_analysis.md            # Phân tích nghiên cứu
├── research_trv2v_surpass_sota.md  # Hướng dẫn surpass SOTA
├── signbposer_replacement_strategy.md
├── vae_replacement_methods.md
└── dexavatar3d_slp_paper_research_proposal.md
```

---

## Quy tắc đặt tên

### Python files
- `run_*.py` → Entry point scripts (chỉ trong `runners/` hoặc `methods/`)
- `eval_*.py` hoặc `evaluation_*.py` → Evaluation scripts (chỉ trong `evaluation/`)
- `*_parser.py` → Data/config parsing
- `*_utils.py` → Utility functions

### Shell scripts
- `run_*.sh` → Entry point runners (chỉ trong `runners/`)
- `Full_running_command*.sh` → Pipeline orchestrators (trong `pipelines/` hoặc `methods/`)
- `M*.sh` → Pipeline stage scripts (trong `scripts/`)
- `config*.sh` → Environment configuration (trong `scripts/`)
- `eval_*.sh` → Evaluation runners (trong `evaluation/`)

### Config files
- `fit_smplx_*.yaml` → Fitting configurations (trong `dexavatar_fitting/cfg_files/`)

---

## Quy tắc khi tạo file mới

1. **Entry point mới** → Đặt trong `runners/`
2. **Pipeline orchestrator mới** → Đặt trong `pipelines/` hoặc `methods/`
3. **Evaluation script mới** → Đặt trong `evaluation/`
4. **Pipeline stage mới** → Đặt trong `scripts/`
5. **Config mới** → Đặt trong `dexavatar_fitting/cfg_files/`
6. **Documentation mới** → Đặt trong `docs/`

**KHÔNG được phép:**
- Tạo file `.py` hoặc `.sh` ở thư mục gốc
- Tạo thư mục mới ở thư mục gốc mà không có lý do chính đáng
- Di chuyển file mà không cập nhật tất cả references

---

## Dependency Graph

```
runners/run_dexavatar_wilor.py
  └── pipelines/Full_running_command_wilor.sh
        ├── scripts/config_sapiens.sh
        ├── scripts/S1_sapiens_extract.sh
        ├── scripts/aggregate_sapiens.py
        ├── scripts/config_smplerx.sh
        ├── scripts/S1_smplerx_extract.sh
        ├── scripts/config.sh
        ├── scripts/M3.5_wilor_extract.sh
        │     └── scripts/M3_mean_shape_smplerx.py
        └── scripts/M4_smplifyx_pose.sh
              └── dexavatar_fitting/script.py

methods/run_dexavatar_wilor_*.py
  └── methods/Full_running_command_wilor_*.sh
        └── (same as above)

evaluation/evaluation_trv2v_wilor.py  (standalone)
evaluation/evaluation_mpvpe_correct.py (standalone)
evaluation/eval_mpvpe_common_frames.py (standalone, self-resolving paths)
evaluation/eval_mpvpe_regions.py       (standalone, self-resolving paths)
```

---

## Conda Environments

| Env name | Python | Dùng cho |
|----------|--------|----------|
| `dexavatar` | 3.10 | Fitting engine, evaluation |
| `sapiens_fix` | 3.9 | Sapiens pose estimation |
| `smpler_x` | 3.8 | SMPLer-X body estimation |
| `hamer` | 3.9 | HaMeR hand extraction |
| `wilor` | 3.9 | WiLoR hand extraction |

**Lưu ý:** Mỗi pipeline stage cần activate đúng env. Các script `config*.sh` xử lý việc này.
