# Hướng dẫn Chi tiết Từ Đầu đến Cuối: Chạy Suy Luận (Inference) DexAvatar

Tài liệu này cung cấp hướng dẫn từng bước (step-by-step), chi tiết và đầy đủ nhất để bất kỳ ai cũng có thể làm theo và chạy thành công pipeline tái tạo chuyển động cơ thể & bàn tay 3D (**DexAvatar**) từ đầu vào là **Video** hoặc **Thư mục ảnh (Frames)**.

---

## 📌 Bảng tra cứu Môi trường Conda (Environment Matrix)

Do DexAvatar tích hợp nhiều mô hình SOTA với các phiên bản PyTorch / CUDA khác nhau, hệ thống sử dụng các môi trường Conda chuyên biệt:

| Tên Môi Trường | Phiên bản Python | Mục đích sử dụng | Khi nào cần kích hoạt? |
| :--- | :---: | :--- | :--- |
| **`dexavatar`** | Python 3.10 | Môi trường chính: Điều phối runner, Hand Extraction (WiLoR/HaMeR), SMPLify-X Fitting, Đánh giá TR-V2V | **Mặc định khi bắt đầu chạy runner** |
| **`sapiens_fix`** *(hoặc `sapiens_lite`)* | Python 3.10 | Trích xuất 133 điểm khớp 2D toàn thân (Whole-body pose) qua Sapiens-1B | Tự động gọi (hoặc thủ công ở Stage 1A) |
| **`smpler_x`** | Python 3.8 | Trích xuất hình học và tư thế 3D thô của thân người qua SMPLer-X H32 | Tự động gọi (hoặc thủ công ở Stage 1B) |

---

## 🛠️ BƯỚC 0: Kiểm tra Checkpoints Tiền đề (Chỉ làm 1 lần)

Trước khi chạy, hãy đảm bảo các file trọng số (weights) đã nằm đúng vị trí trong thư mục dự án `/home/haipd/DexAvatar`:

```text
DexAvatar/
├── checkpoints/
│   ├── smpler_x_h32.pth.tar                 # Trọng số SMPLer-X
│   └── mmdet/
│       ├── faster_rcnn_r50_fpn_1x_coco_...pth
│       └── mmdet_faster_rcnn_r50_fpn_coco.py
├── SMPLer-X/
│   └── common/utils/human_model_files/      # Chứa models smplx, neutral, female, male
├── dexavatar_fitting/
│   └── smplifyx/
│       ├── signbposer/                      # Body pose prior (SignBPoser)
│       └── signhposer/                      # Hand pose prior (SignHPoser)
└── sapiens/
    └── lite/torchscript/
        ├── detector/checkpoints/rtmpose/rtmdet_m_...pth
        └── pose/checkpoints/sapiens_1b/sapiens_1b_coco_wholebody_...pt2
```

---

## 📁 BƯỚC 1: Chuẩn bị Dữ liệu Đầu vào (Input Data)

DexAvatar nhận đầu vào là các khung hình ảnh `.png` hoặc `.jpg`.

### 1.1. Trường hợp có sẵn Video (`.mp4`, `.avi`, `.mov`)
Chuyển đổi video thành chuỗi ảnh bằng công cụ `ffmpeg`:

```bash
# 1. Tạo thư mục chứa frame của video
mkdir -p /home/haipd/DexAvatar/data/my_inputs/video_01

# 2. Tách video thành các file ảnh đánh số thứ tự 5 chữ số: 00001.png, 00002.png,...
ffmpeg -i /duong/dan/toi/video.mp4 -qscale:v 2 /home/haipd/DexAvatar/data/my_inputs/video_01/%05d.png
```

### 1.2. Cấu trúc thư mục đầu vào chuẩn
Tổ chức thư mục đầu vào theo phân cấp: 1 thư mục gốc chứa các thư mục con của từng video:

```text
/home/haipd/DexAvatar/data/my_inputs/
├── video_01/
│   ├── 00001.png
│   ├── 00002.png
│   └── ...
└── video_02/
    ├── 00001.png
    ├── 00002.png
    └── ...
```

---

## 🚀 BƯỚC 2: Hướng dẫn Chạy Lệnh Tự Động (Khuyên Dùng)

*Hệ thống runner tự động chuyển đổi qua lại giữa các môi trường Conda (`sapiens_fix` $\rightarrow$ `smpler_x` $\rightarrow$ `dexavatar`), bạn chỉ cần kích hoạt môi trường `dexavatar` ở đầu lệnh.*

---

### Cách 2.1: Chạy Hàng Loạt (Batch Mode) cho cả thư mục chứa nhiều video

Mở Terminal và thực thi chuỗi lệnh sau:

```bash
# 1. Di chuyển vào thư mục gốc của dự án
cd /home/haipd/DexAvatar

# 2. Kích hoạt môi trường conda chính
source ~/miniconda3/etc/profile.d/conda.sh   # hoặc source ~/.bashrc
conda activate dexavatar

# 3. Chạy Pipeline tối ưu với WiLoR Hand Expert (Khuyên dùng - Độ chính xác tay cao nhất)
python runners/run_dexavatar_wilor.py \
    --input_img_folder /home/haipd/DexAvatar/data/my_inputs \
    --output_path /home/haipd/DexAvatar/outputs/my_outputs_wilor \
    --fitting_experiment ./dexavatar_fitting
```

> *(Tùy chọn) Nếu muốn chạy bản DexAvatar gốc theo bài báo (dùng HaMeR thay vì WiLoR):*
> ```bash
> python methods/run_dexavatar.py \
>     --input_img_folder /home/haipd/DexAvatar/data/my_inputs \
>     --output_path /home/haipd/DexAvatar/outputs/my_outputs_hamer \
>     --fitting_experiment ./dexavatar_fitting
> ```

---

### Cách 2.2: Chạy Đơn Lẻ cho 1 Video / 1 Chuỗi Frame duy nhất (Single Sequence)

Nếu bạn chỉ muốn chạy thử nghiệm nhanh trên đúng 1 thư mục video:

```bash
# 1. Di chuyển vào thư mục gốc dự án
cd /home/haipd/DexAvatar

# 2. Kích hoạt môi trường dexavatar
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dexavatar

# 3. Khai báo biến đường dẫn
export ROOT_PATH="/home/haipd/DexAvatar/data/my_inputs/video_01"
export OUTPUT_PATH="/home/haipd/DexAvatar/outputs/my_outputs_wilor/video_01"
export FITTING_EXPERIMENT="/home/haipd/DexAvatar/dexavatar_fitting"

# 4. Tạo thư mục output
mkdir -p "${OUTPUT_PATH}"

# 5. Chạy toàn bộ pipeline tự động
bash pipelines/Full_running_command_wilor.sh
```

---

## 🔍 BƯỚC 3: Hướng dẫn Chạy Thủ Công Từng Bước (Manual Step-by-Step)

*Chỉ sử dụng phần này khi bạn muốn debug, can thiệp vào từng bước hoặc kiểm tra trung gian.*

Giả sử ta đang chạy cho video tại: `ROOT_PATH="/home/haipd/DexAvatar/data/my_inputs/video_01"` và `OUTPUT_PATH="/home/haipd/DexAvatar/outputs/debug/video_01"`.

```bash
cd /home/haipd/DexAvatar
export ROOT_PATH="/home/haipd/DexAvatar/data/my_inputs/video_01"
export OUTPUT_PATH="/home/haipd/DexAvatar/outputs/debug/video_01"
export FITTING_EXPERIMENT="/home/haipd/DexAvatar/dexavatar_fitting"
SIGN_NAME=$(basename "${ROOT_PATH}")
mkdir -p "${OUTPUT_PATH}"
```

### 🔹 Giai đoạn 1: Trích xuất 2D 133 Keypoints toàn thân (Sapiens 1B)
```bash
# 1. Kích hoạt môi trường Sapiens
conda activate sapiens_fix

# 2. Chạy trích xuất Sapiens Pose
bash scripts/S1_sapiens_extract.sh

# 3. Kích hoạt môi trường dexavatar để gộp kết quả keypoint thành file sapiens.pkl
conda activate dexavatar
python scripts/aggregate_sapiens.py \
    --sapiens_dir "${OUTPUT_PATH}/sapiens_1b" \
    --output_path "${OUTPUT_PATH}" \
    --subfolder "${SIGN_NAME}"
```

### 🔹 Giai đoạn 2: Trích xuất 3D Body thô (SMPLer-X H32)
```bash
# 1. Kích hoạt môi trường SMPLer-X
conda activate smpler_x

# 2. Chạy SMPLer-X extraction
bash scripts/S1_smplerx_extract.sh
```

### 🔹 Giai đoạn 3: Tính Mean Shape và Trích xuất 3D Bàn tay (WiLoR)
```bash
# 1. Kích hoạt môi trường dexavatar
conda activate dexavatar

# 2. Chạy trích xuất hình học tay WiLoR
bash scripts/M3.5_wilor_extract.sh
```

### 🔹 Giai đoạn 4: Tối ưu hóa Khớp với Sign Priors (SMPLify-X Fitting)
```bash
# 1. Giữ môi trường dexavatar
conda activate dexavatar

# 2. Chạy tối ưu hóa fitting tạo 3D Mesh
bash scripts/M4_smplifyx_pose.sh
```

---

## 📊 BƯỚC 4: Kiểm tra và Xem Kết quả Đầu ra (Output Verification)

Sau khi pipeline chạy xong, truy cập vào thư mục kết quả:

```text
/home/haipd/DexAvatar/outputs/my_outputs_wilor/video_01/
├── sapiens.pkl                  # Toàn bộ 133 điểm khớp 2D
├── smplerx/                     # Kết quả ước lượng 3D ban đầu từ SMPLer-X
├── mean_shape_smplx.npy         # Hình dáng 3D (beta) trung bình của nhân vật
├── gender.txt                   # Nhãn giới tính (neutral)
├── wilor/                       # Khớp bàn tay 3D cục bộ
└── smplifyx/                    # KẾT QUẢ CUỐI CÙNG (FINAL OUTPUT)
    ├── meshes/                  # ★ Chứa file 3D Mesh (.obj) cho từng frame
    │   ├── 00001.obj
    │   ├── 00002.obj
    │   └── ...
    └── results/                 # ★ Chứa file tham số SMPL-X (.pkl) đã tối ưu
        ├── 00001.pkl
        └── ...
```

> **Cách hiển thị xem 3D Mesh:**
> * Bạn có thể mở các file `.obj` trong thư mục `smplifyx/meshes/` bằng các phần mềm: **Blender**, **MeshLab**, hoặc **3D Viewer** trên máy tính.

---

## 📈 BƯỚC 5: Đánh giá Sai số Benchmark (TR-V2V Metric)

Nếu dữ liệu của bạn có Ground Truth 3D SMPL-X (ví dụ tập SGNify), chạy lệnh đánh giá sau:

```bash
cd /home/haipd/DexAvatar
conda activate dexavatar

python evaluation/evaluation_trv2v_wilor.py \
    --pred_root /home/haipd/DexAvatar/outputs/my_outputs_wilor \
    --gt_root /home/haipd/DexAvatar/data/smplx_gt \
    --signs_txt /home/haipd/DexAvatar/data/signs.txt \
    --method_name DexAvatar-WiLoR
```

---

## ⚠️ BƯỚC 6: Xử lý Các Lỗi Thường Gặp (Troubleshooting)

| Hiện tượng lỗi | Nguyên nhân | Cách khắc phục |
| :--- | :--- | :--- |
| `ImportError` hoặc lỗi CUDA liên quan đến `libstdc++` / `libcudart` | Xung đột thư viện `LD_LIBRARY_PATH` giữa các môi trường conda. | Luôn chạy `unset LD_LIBRARY_PATH` trước khi chuyển môi trường Conda. |
| `No images found. Check your input directory` | Thư mục đầu vào không chứa ảnh đuôi `.png` hoặc `.jpg`. | Kiểm tra lại lệnh `ffmpeg` ở Bước 1, đảm bảo ảnh được lưu dạng `00001.png`. |
| `FileNotFoundError: smpler_x_h32.pth.tar` | Chưa tải checkpoint của SMPLer-X. | Đặt file trọng số vào đúng thư mục `checkpoints/smpler_x_h32.pth.tar`. |
| `CUDA out of memory` | Video có độ phân giải quá cao hoặc batch size lớn. | Giảm batch size trong các file script hoặc giảm độ phân giải ảnh đầu vào. |
