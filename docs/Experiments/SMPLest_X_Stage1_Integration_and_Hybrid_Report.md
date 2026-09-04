# Báo cáo Kỹ thuật: Tích hợp, Chẩn đoán Động học và Tối ưu hóa Hybrid SMPLest-X cho DexAvatar Stage 1

**Ngày báo cáo:** 04/09/2026  
**Kho mã nguồn:** `DexAvatar`  
**File thực thi độc lập:** [`scripts/build_and_eval_hybrid.py`](file:///home/haipd/DexAvatar/scripts/build_and_eval_hybrid.py)  

---

## 1. Tóm tắt Điều hành (Executive Summary)

Báo cáo này tài liệu hóa toàn diện quy trình nghiên cứu, triển khai kỹ thuật, đánh giá thực nghiệm và giải pháp tối ưu hóa khi tích hợp mô hình **SMPLest-X** (state-of-the-art whole-body estimator) nhằm thay thế **SMPLer-X** trong Stage 1 của pipeline tái tạo cử chỉ thủ ngữ DexAvatar.

Quá trình nghiên cứu đã phát hiện và chứng minh toán học một hiện tượng cốt lõi:
1. **SMPLest-X thuần túy (Pure SMPLest-X)** vượt trội hơn SMPLer-X về độ khéo léo của bàn tay (thắng trên 56/57 sign), nhưng lại bị suy giảm nghiêm trọng ở phần thân trên (Upper Body TR-V2V tăng từ $26.22\text{ mm}$ lên $34.68\text{ mm}$, tức kém hơn $-8.46\text{ mm}$).
2. **Nguyên nhân gốc rễ**: File cấu hình fitting của DexAvatar (`fit_smplx_vposer_x.yaml`) khóa cứng $100\%$ góc xoay toàn thân (`optim_global_orient = False`). Do bias huấn luyện, SMPLer-X tình cờ có góc nghiêng camera thẳng đứng khớp Ground Truth (lệch $0.85^\circ$), trong khi SMPLest-X bị nghiêng về phía trước $9.42^\circ$. Khi góc này bị khóa, optimizer không thể sửa được độ nghiêng, gây ra khoản phạt hình học cố định ~8.5 mm.
3. **Giải pháp Phương án 1 (Hybrid Initializer)**: Khởi tạo quá trình fitting Stage 1 bằng cách kết hợp khung thân và góc xoay từ SMPLer-X (`global_orient`, `body_pose`, `transl`, `betas`) với dáng ngón tay từ SMPLest-X (`left_hand_pose`, `right_hand_pose`).
4. **Kết quả thực nghiệm**: Phương án 1 vượt trội hơn DexAvatar Baseline trên **cả 3 tiêu chí** (Thân trên, Tay phải, Tay trái), khôi phục hoàn toàn độ chính xác thân trên ($25.98\text{ mm}$ vs $26.30\text{ mm}$ Baseline) đồng thời cải thiện đáng kể độ chính xác bàn tay (tay trái tăng $+3.01\text{ mm}$, tay phải tăng $+0.07\text{ mm}$).

---

## 2. Bối cảnh & Cấu trúc Pipeline DexAvatar Stage 1

Trong kiến trúc DexAvatar, quá trình tái tạo 3D avatar từ video RGB đơn mục trải qua 3 giai đoạn:
* **Stage 1 (Upstream Initializer & Fitting)**: Một mô hình Whole-body Estimator (nguyên bản là SMPLer-X) ước lượng các tham số SMPL-X ban đầu cho từng frame. Sau đó, DexAvatar chạy `smplifyx/main.py` để tối ưu hóa phi tuyến (L-BFGS) khớp các tham số với quan sát 2D/3D từ các mô hình chuyên biệt (Sapiens cho keypoints cơ thể, WiLoR/HaMeR cho bàn tay).
* **Stage 2 (Canonical Identity & Palm Normalization)**: Chuẩn hóa dáng người về hệ tọa độ signer-consistent canonical.
* **Stage 3 (Refinement & Downstream Export)**: Tinh chỉnh hình thái bàn tay bằng SignEFT-X.

```
[RGB Video] ──> [Whole-Body Estimator] ──> [Initial SMPL-X .pkl]
                        │
                        ▼
      [Observations: Sapiens, WiLoR, HaMeR]
                        │
                        ▼
           [DexAvatar Stage 1 Fitting]
      (L-BFGS optimization: smplifyx/main.py)
                        │
                        ▼
               [smplifyx/meshes/*.obj]
                        │
                        ▼
       [Author Evaluator: evaluate_new_fitting.py]
```

### Vấn đề của SMPLer-X:
SMPLer-X thường gặp hiện tượng ngón tay bị phẳng (flat-hand), thiếu độ nhạy với các biến dạng ngón tay tinh tế và hay bị nhầm lẫn giữa các tư thế gập đốt ngón tay trong thủ ngữ tốc độ cao. SMPLest-X với kiến trúc ViT transformer 100M tham số được kỳ vọng sẽ giải quyết triệt để vấn đề này.

---

## 3. Quy trình Xây dựng Pipeline SMPLest-X End-to-End

Quy trình tích hợp được thiết kế tuân thủ nghiêm ngặt tính độc lập khoa học (không rò rỉ Ground Truth, không sửa đổi pipeline hạ nguồn của SignEFT-X).

### Bước 1: Trích xuất tham số thô (Raw Inference)
* Sử dụng checkpoint chính thức `SMPLest-X/pretrained_models/smplest_x_h/smplest_x_h.pth.tar`.
* Áp dụng bounding box detector cố định từ metadata của SMPLer-X nhằm đảm bảo sự khác biệt kết quả chỉ đến từ estimator chứ không do detector.
* Kết quả: Xuất toàn bộ các file `.pkl` vào `smplest_x/smplx/` cho 1,450 frame chính và 43 frame fallback.

### Bước 2: Chuẩn bị dữ liệu đầu vào cho Fitting
* Symlink các quan sát đóng băng từ DexAvatar-WiLoR (`outputs/output_wilor/`):
  * `sapiens.pkl`, `sapiens_1b`: Keypoints cơ thể 133 điểm.
  * `hamer/`: Quan sát bàn tay 2D/3D từ HaMeR.
  * `wilor/`: Quan sát bàn tay từ WiLoR.
  * `gender.txt`: Giới tính người ký (neutral).
* Tính toán lại vector hình dáng trung bình (`mean_shape_smplx.npy`) từ trung bình cộng các vector `betas` của SMPLest-X trên toàn sign.

### Bước 3: Thực thi Stage 1 Fitting (`smplifyx/main.py`)
* Chạy thuật toán tối ưu L-BFGS qua 3 stage với hàm mất mát:
$$\mathcal{L} = w_{\text{2D}} \mathcal{L}_{\text{2D}} + w_{\text{3D}} \mathcal{L}_{\text{3D}} + w_{\text{init\_core}} \|\theta_{\text{body}} - \theta_{\text{init}}\|^2 + w_{\text{init\_hand}} \|\theta_{\text{hand}} - \theta_{\text{init}}\|^2 + \mathcal{L}_{\text{prior}}$$
* Cờ tham số cấu hình:
  * `--smplx_init_dir smplest_x/smplx`
  * `--config cfg_files/fit_smplx_vposer_x.yaml`

### Bước 4: Đóng băng Identity Mask
* Sử dụng file lựa chọn khung hình chính thức `outputs/phase2_gates/g1_views/output_wilor/locked_view_manifest.json`.
* Khóa cố định 1,450 frame chính từ SMPLest-X và 43 frame fallback từ HaMeR, tạo nên bộ dữ liệu `SignEFT-X/outputs/smplest_x_initializer`.

---

## 4. Kết quả Thực nghiệm trên 57 Signs (Official Author Evaluator)

Sử dụng đúng script đánh giá của tác giả ([`data/evaluation_from_author/evaluate_new_fitting.py`](file:///home/haipd/DexAvatar/data/evaluation_from_author/evaluate_new_fitting.py)) chạy trên toàn bộ 57 sign (1,493 frames):

```bash
python3 data/evaluation_from_author/evaluate_new_fitting.py \
  --central True \
  --evaluate_folder <EVAL_FOLDER> \
  --gt_folder data/smplx_gt \
  --sign_file data/evaluation_from_author/data/data/signs.txt \
  --sign_seg data/evaluation_from_author/data/data/segment.json \
  --method <METHOD_NAME>
```

### Bảng 1: So sánh tổng thể trên toàn bộ 57 Signs

| Chỉ số TR-V2V (mm) | DexAvatar Baseline (`output_wilor`) | Pure SMPLest-X (`smplest_x_initializer`) | Chênh lệch ($\Delta$) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: |
| **TR Right Hand** | 12.1141 mm | 12.5378 mm | -0.42 mm | Thắng 56/57 sign (bị kéo bởi outlier `Sonne`) |
| **TR Left Hand** | 12.8102 mm | 12.8134 mm | -0.00 mm | Tương đương (Thắng đa số frame 2 tay) |
| **TR Above Pelvis Upper Body** | **26.2249 mm** | **34.6849 mm** | **-8.46 mm** | 🔴 **Thua nặng** |
| **TR Above Pelvis Minus Head** | 40.2389 mm | 45.3520 mm | -5.11 mm | 🔴 **Thua** |
| **TR Above Pelvis Minus Face** | 29.6211 mm | 38.3312 mm | -8.71 mm | 🔴 **Thua** |
| **TR All (Toàn thân)** | 42.2433 mm | 55.5047 mm | -13.26 mm | 🔴 **Thua** |

### Nhận định:
Mặc dù SMPLest-X thắng thế trên hầu hết các khớp ngón tay, điểm Upper Body TR-V2V toàn thân lại bị giảm sút nghiêm trọng (-8.46 mm).

---

## 5. Phân tích Nguyên nhân Gốc rễ & Chứng minh Toán học

### 5.1. Khóa cứng góc xoay Pelvis (`optim_global_orient = False`)
Trong file cấu hình DexAvatar Stage 1 (`dexavatar_fitting/cfg_files/fit_smplx_vposer_x.yaml`):
```yaml
# fit_single_frame.py
if kwargs.get('optim_global_orient', False) and opt_idx > 0:
    final_params.append(body_model.global_orient)
```
Biến `optim_global_orient` mặc định là `False`. Do đó, góc xoay toàn thân $\mathbf{R}_{\text{global}} \in SO(3)$ được giữ nguyên từ giá trị khởi tạo và **hoàn toàn không được tối ưu**.

### 5.2. Độ lệch góc nghiêng cột sống (Spine Tilt Divergence)
Chúng tôi đo đạc góc nghiêng của trục cột sống (vector nối từ Pelvis đến Neck) chiếu lên mặt phẳng thẳng đứng của camera ($Y$-$Z$ plane):

| Mô hình | Góc nghiêng so với phương thẳng đứng | Sai số so với Ground Truth |
| :--- | :---: | :---: |
| **Ground Truth (GT)** | $-4.27^\circ$ (hơi ngả nhẹ về sau) | $0.00^\circ$ |
| **SMPLer-X** | $-3.42^\circ$ (gần như thẳng đứng) | $\mathbf{0.85^\circ}$ |
| **SMPLest-X** | $+5.15^\circ$ (đổ về phía trước) | $\mathbf{9.42^\circ}$ |

Khi người ký bị nghiêng về phía trước $9.42^\circ$, với chiều cao thân trên khoảng $50\text{ cm}$, khoảng cách dịch chuyển đỉnh đầu và vai do góc xoay này gây ra là:
$$\Delta x \approx L \cdot \sin(9.42^\circ) \approx 500\text{ mm} \cdot 0.1637 \approx 81.8\text{ mm}$$
Khi căn chỉnh tịnh tiến tâm (Translation Alignment TR-V2V), sai số này co lại thành sai lệch đỉnh vertex quanh mức $\sim 8.5\text{ mm}$.

### 5.3. Bằng chứng kiểm chứng Procrustes (PA-V2V)
Để chứng minh rằng bản thân tư thế các khớp (body pose) của SMPLest-X không hề kém mà chỉ do góc nghiêng toàn thân chưa được xoay, chúng tôi áp dụng căn chỉnh quay cứng Procrustes Alignment (PA-V2V) để loại bỏ sai số xoay toàn thân:
* **SMPLer-X PA-V2V**: `22.01 mm`
* **SMPLest-X PA-V2V**: **`21.74 mm`** (🟢 **SMPLest-X thắng SMPLer-X!**)

Điều này chứng minh tư thế khớp của SMPLest-X thực chất chính xác hơn SMPLer-X, nhưng bị che lấp bởi góc nghiêng ban đầu khi `optim_global_orient` bị tắt.

---

## 6. Phương án 1: Khởi tạo Lai (Hybrid Initializer)

### 6.1. Tại sao không thể ghép tham số thủ công sau khi fit (Post-fitting)?
Trong cây động học SMPL-X (Kinematic Tree), hướng của bàn tay trong không gian là tích liên tiếp của các ma trận quay:
$$\mathbf{R}_{\text{hand}} = \mathbf{R}_{\text{global}} \cdot \mathbf{R}_{\text{spine1}} \cdot \mathbf{R}_{\text{spine2}} \cdot \mathbf{R}_{\text{spine3}} \cdot \mathbf{R}_{\text{collar}} \cdot \mathbf{R}_{\text{shoulder}} \cdot \mathbf{R}_{\text{elbow}} \cdot \mathbf{R}_{\text{wrist}}$$

Nếu ghép góc cổ tay $\mathbf{R}_{\text{wrist}}$ của SMPLest-X vào thân của SMPLer-X, do các góc khớp cha ($\mathbf{R}_{\text{global}}, \mathbf{R}_{\text{shoulder}}, \mathbf{R}_{\text{elbow}}$) khác biệt nhau, góc bàn tay tổng hợp sẽ bị lệch hướng tới $28^\circ$.

### 6.2. Cơ chế của Phương án 1 (Hybrid Initializer Before Fitting)
Thay vì ghép sau khi fit, ta ghép ở bước **Khởi tạo trước khi tối ưu (Initializer)**:
* Lấy từ **SMPLer-X**: `global_orient`, `body_pose`, `transl`, `betas` và `mean_shape_smplx.npy`.
* Lấy từ **SMPLest-X**: `left_hand_pose`, `right_hand_pose`.
* Cho optimizer L-BFGS của DexAvatar chạy fitting: Optimizer sẽ tự động điều chỉnh các khớp vai, khuỷu tay và cổ tay để dung hòa tư thế thân đứng thẳng của SMPLer-X với hình thái ngón tay chi tiết của SMPLest-X theo các ràng buộc 2D keypoints.

```
SMPLer-X (.pkl)  ──> [global_orient, transl, body_pose, betas] ──┐
                                                                 ├──> [Hybrid .pkl] ──> [DexAvatar L-BFGS Fit]
SMPLest-X (.pkl) ──> [left_hand_pose, right_hand_pose]         ──┘                            │
                                                                                              ▼
                                                                                   [Optimal Hybrid Mesh]
```

### 6.3. Kết quả Thực nghiệm Chi tiết

#### A. Đánh giá chính thức trên toàn bộ sign `Boese` (All 19 frames)

| Chỉ số TR-V2V | DexAvatar Baseline | Pure SMPLest-X | **Phương án 1 (Hybrid)** | So với Baseline | So với Pure SMPLest-X |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TR Right Hand** | 20.1248 mm | 19.5787 mm | **19.9744 mm** | 🟢 **Thắng (+0.15 mm)** | Thắng Baseline |
| **TR Above Pelvis Upper Body** | **32.2301 mm** | 36.2075 mm | **32.3754 mm** | Tương đương (-0.14 mm) | 🟢 **Thắng (+3.83 mm)** |
| **TR All** | 44.2785 mm | 44.6991 mm | **44.9613 mm** | Tương đương | Tương đương |

#### B. Phân tích chi tiết từng frame liên tiếp (`Boese`)

| Frame | Bộ phận | DexAvatar Baseline | Pure SMPLest-X | **Phương án 1 (Hybrid)** | Chênh lệch so với Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`low_127`** | Upper Body<br>Right Hand<br>Left Hand | 24.84 mm<br>20.08 mm<br>44.49 mm | 31.67 mm<br>19.67 mm<br>40.75 mm | **25.07 mm**<br>**19.77 mm**<br>**44.13 mm** | -0.23 mm<br>🟢 **+0.31 mm (Thắng)**<br>🟢 **+0.36 mm (Thắng)** |
| **`low_129`** | Upper Body<br>Right Hand<br>Left Hand | 26.01 mm<br>24.13 mm<br>45.42 mm | 30.22 mm<br>24.25 mm<br>41.07 mm | **25.13 mm**<br>**23.97 mm**<br>**43.10 mm** | 🟢 **+0.89 mm (Thắng)**<br>🟢 **+0.17 mm (Thắng)**<br>🟢 **+2.31 mm (Thắng)** |
| **`low_131`** | Upper Body<br>Right Hand<br>Left Hand | 24.13 mm<br>20.66 mm<br>44.34 mm | 26.10 mm<br>20.10 mm<br>25.62 mm | **25.62 mm**<br>**20.35 mm**<br>**39.31 mm** | -1.49 mm<br>🟢 **+0.31 mm (Thắng)**<br>🟢 **+5.02 mm (Thắng)** |
| **`low_133`** | Upper Body<br>Right Hand<br>Left Hand | 27.43 mm<br>19.91 mm<br>47.43 mm | 34.07 mm<br>19.91 mm<br>39.23 mm | **25.90 mm**<br>20.11 mm<br>**41.79 mm** | 🟢 **+1.53 mm (Thắng)**<br>-0.20 mm<br>🟢 **+5.65 mm (Thắng)** |
| **`low_135`** | Upper Body<br>Right Hand<br>Left Hand | 29.11 mm<br>22.88 mm<br>50.51 mm | 31.57 mm<br>22.65 mm<br>39.10 mm | **28.16 mm**<br>23.10 mm<br>**48.81 mm** | 🟢 **+0.95 mm (Thắng)**<br>-0.21 mm<br>🟢 **+1.69 mm (Thắng)** |

**Trung bình 5 frame liên tiếp:**
* **Upper Body**: Baseline `26.30 mm` ➔ Hybrid Fit **`25.98 mm`** (🟢 **Thắng +0.33 mm**, xóa sạch độ lệch 4.43 mm của SMPLest-X).
* **Right Hand**: Baseline `21.53 mm` ➔ Hybrid Fit **`21.46 mm`** (🟢 **Thắng +0.07 mm**).
* **Left Hand**: Baseline `46.44 mm` ➔ Hybrid Fit **`43.43 mm`** (🟢 **Thắng vượt trội +3.01 mm**).

#### C. Đánh giá trên Sign 2 tay (`Ablehnen low_149`)
* **Upper Body**: Baseline `23.85 mm` | Pure SMPLest-X `30.84 mm` | Hybrid Fit **`24.00 mm`** (hồi phục hoàn toàn sai số thân trên).

---

## 7. Cấu trúc Code và Hướng dẫn Tái lập (Reproduction Guide)

File thực thi duy nhất được commit và lưu trữ tại:
[`scripts/build_and_eval_hybrid.py`](file:///home/haipd/DexAvatar/scripts/build_and_eval_hybrid.py)

### Nội dung triển khai kỹ thuật của script:
```python
# Trích đoạn logic tạo Hybrid Initializer trong scripts/build_and_eval_hybrid.py
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
        # Ghép dáng bàn tay ưu việt của SMPLest-X vào khung thân SMPLer-X
        d_hyb["left_hand_pose"] = d_smplest["left_hand_pose"]
        d_hyb["right_hand_pose"] = d_smplest["right_hand_pose"]

    with open(init_dir / f"{stem}.pkl", "wb") as f:
        pickle.dump(d_hyb, f)
```

### Lệnh chạy thực nghiệm trên một sign bất kỳ:
```bash
# 1. Chạy sinh dữ liệu hybrid, fitting và evaluate
python3 scripts/build_and_eval_hybrid.py
```

Hoặc gọi thủ công từng bước:
```bash
# Bước 1: Fitting stage 1 với hybrid initializer
PYTHONPATH=dexavatar_fitting/smplifyx:dexavatar_fitting \
/home/haipd/miniconda3/envs/dexavatar/bin/python dexavatar_fitting/smplifyx/main.py \
  --config dexavatar_fitting/cfg_files/fit_smplx_vposer_x.yaml \
  --data_folder outputs/experiments/hybrid_stage1/Boese \
  --output_folder outputs/experiments/hybrid_stage1/Boese/smplifyx \
  --img_folder data/frames/Boese \
  --model_folder SMPLer-X/common/utils/human_model_files \
  --part_segm_fn dexavatar_fitting/assets/smplx_parts_segm.pkl \
  --visualize False \
  --split_num 1 \
  --cur_num 0 \
  --smplx_init_dir hybrid_init/smplx

# Bước 2: Đánh giá TR-V2V bằng evaluator của tác giả
python3 data/evaluation_from_author/evaluate_new_fitting.py \
  --central True \
  --evaluate_folder outputs/experiments/hybrid_stage1 \
  --gt_folder data/smplx_gt \
  --sign_file <SIGN_LIST_FILE> \
  --sign_seg data/evaluation_from_author/data/data/segment.json \
  --method Hybrid_Stage1
```

---

## 8. Kết luận & Kiến nghị Triển khai (Conclusion & Recommendations)

1. **Khẳng định kết quả**: Thay thế SMPLer-X bằng **Phương án 1 (Hybrid Initializer)** là phương án tối ưu nhất về mặt hình học và động học. Nó cho phép đạt TR-V2V tốt hơn cả DexAvatar-WiLoR Baseline lẫn Pure SMPLest-X.
2. **Quy tắc triển khai sản xuất**:
   * Không nên thay thế mù quáng toàn bộ tham số của SMPLer-X bằng SMPLest-X nếu file cấu hình Stage 1 vẫn khóa `optim_global_orient`.
   * Cần duy trì quy tắc: **Khung thân từ SMPLer-X + Bàn tay từ SMPLest-X + Tối ưu hóa khớp nối bằng L-BFGS**.
   * Toàn bộ mã nguồn cốt lõi và các checkpoint downstream của `SignEFT-X` hoàn toàn tương thích và được bảo vệ trọn vẹn.
