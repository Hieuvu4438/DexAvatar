# SIGNAL-4D V5: phương pháp tốt nhất hiện tại, đặc tả end-to-end và biên giới tuyên bố

Ngày khóa báo cáo: **2026-08-21**
Trạng thái: **current best frozen method trong codebase này**
Tên thực nghiệm chính xác: `SIGNAL4D M1-v5 + GT-free multiscale gate`

## 1. Tóm tắt điều hành

SIGNAL-4D V5 không thay thế DexAvatar bằng một mạng end-to-end mới. Nó giữ nguyên các chuyên gia khởi tạo và fitting đã có, biểu diễn tất cả giả thuyết trong cùng không gian SMPL-X, ước lượng độ bất định theo nguồn/vùng, chỉ tối ưu một tập con khớp có rủi ro cao trên đa tạp SO(3), rồi dùng một gate đa tỉ lệ có tính nhất quán thời gian để chọn giữa kết quả V5 và baseline A1.

Trong so sánh đầy đủ 57 sign/1.493 frame theo code evaluator SGNify của tác giả, V5 tốt hơn output DexAvatar `method_hamer` trên mọi vùng được báo cáo. Với metric mục tiêu chính thức `UBody(-F)` (`tr_upper_body_minus_face_mm`), kết quả vertex-micro là **29,9074 mm → 29,5932 mm**, giảm **0,3142 mm**. Hai tay giảm lần lượt **1,9084 mm** ở tay trái và **1,0942 mm** ở tay phải trên metric TR-V2V riêng từng tay.

Ranh giới quan trọng:

- Đây là **kết quả tốt nhất nội bộ trên protocol và dữ liệu đã nêu**, không tự động đồng nghĩa với SOTA công bố trên một leaderboard bên ngoài.
- Kết quả 1.493 frame là diagnostic đầy đủ và đã chịu ảnh hưởng của chu trình phát triển. Nó không phải một test set còn bí mật.
- Bộ 56 sign/769 frame được freeze trước khi tạo GT cache cho endpoint đó, chứng minh gate không đọc GT tại inference; tuy nhiên các clip/sign identity vẫn chồng với dữ liệu lịch sử và không chứng minh tổng quát hóa unseen-signer.
- “V5” chỉ là số phiên bản kỹ thuật/thực nghiệm. Contribution khoa học nằm ở uncertainty-calibrated selective SO(3) refinement và temporal multiscale gating, không nằm ở tên phiên bản.

## 2. Bài toán, input và output

### 2.1 Bài toán

Cho chuỗi ảnh RGB của một người thực hiện ngôn ngữ ký hiệu, cần tái dựng chuỗi mesh SMPL-X sao cho:

1. thân trên, cánh tay, cổ tay và bàn tay đúng hình học tương đối;
2. chuyển động không rung và không bị làm mượt quá mức ở các pha chuyển động nhanh;
3. output giữ đúng topology SMPL-X neutral của evaluator;
4. không sử dụng GT tại inference;
5. luôn có đường fallback byte-exact về baseline mạnh đã tồn tại.

### 2.2 Input thực tế của V5

V5 dùng ba nguồn quan sát đã được canonicalize:

| Source ID | Nguồn | Vai trò |
|---:|---|---|
| 0 | SMPLer-X | body/camera/shape hypothesis và khung SMPL-X ban đầu |
| 1 | WiLoR | keypoint 3D, keypoint 2D và rotation MANO cho tay |
| 2 | fitted legacy A1 | giả thuyết pose đã coherent sau fitting DexAvatar |

Mỗi nguồn được đưa về:

- hệ tọa độ camera OpenCV `x-right, y-down, z-forward`;
- đơn vị mét trong optimization;
- rotation matrix local parent-to-child;
- canonical 55 khớp phục vụ fitting;
- cùng frame manifest, tuyệt đối không tự bỏ frame lỗi.

Với tay trái WiLoR, điểm và rotation MANO được mirror theo phép biến đổi đúng handedness trước khi đặt vào canonical skeleton. Đây là một sửa lỗi kỹ thuật quan trọng; không được coi tay trái là bản sao chỉ đổi dấu tùy ý.

### 2.3 Output

Output chính là tham số SMPL-X và mesh được forward trực tiếp từ SMPL-X neutral:

- 10.475 vertex;
- 20.908 face;
- không nội suy OBJ;
- một prediction cho mọi frame trong manifest;
- có provenance, hash, source selection và diagnostics theo frame/clip.

## 3. A1 baseline chính xác là gì?

Có hai ngữ cảnh tên A1 cần phân biệt.

### 3.1 Baseline trong so sánh đầy đủ 1.493 frame

Baseline DexAvatar chính thức trong báo cáo đầy đủ là:

```text
outputs/method_hamer
  = HaMeR initialization
  + SignBPoser fitting
  + SignHPoser fitting
  = native DexAvatar fitted SMPL-X output
```

Nó có đúng 57 sign và 1.493 frame key giống SIGNAL-4D. Vì vậy so sánh full-1.493 không lấy một baseline rút gọn hay một model body-only khác.

### 3.2 A1 hierarchy trong endpoint prospective V5

Để không dùng evaluator score khi tạo baseline khả dụng trên toàn endpoint 56 sign/769 frame, A1 được resolve bằng thứ tự cố định:

1. `Ensemble` nếu có: 607 frame;
2. HaMeR A0 fallback: 145 frame;
3. raw SMPLer-X terminal fallback: 17 frame.

Thứ tự này được freeze trước khi mở GT endpoint, cho coverage 100%. Nó không có nghĩa mọi frame của A1 prospective đều là cùng một folder `method_hamer`. Khi viết paper phải nêu rõ endpoint nào dùng baseline nào.

## 4. Pipeline end-to-end

```mermaid
flowchart LR
    I[RGB frames + frozen manifest] --> E0[SMPLer-X]
    I --> E1[WiLoR]
    A1[DexAvatar fitted A1, read-only] --> C
    E0 --> C[Canonical SMPL-X observation cache]
    E1 --> C
    C --> U[Calibrated uncertainty by source and region]
    C --> P[Rule-based change probability]
    U --> S[Selective SO(3) M1 solver]
    P --> S
    S --> H1[M1 alpha 1.0]
    S --> H2[M1 alpha 1.5]
    S --> H3[M1 alpha 3.0]
    A1 --> G[GT-free-at-inference multiscale temporal gate]
    H1 --> G
    H2 --> G
    H3 --> G
    G --> X[Direct SMPL-X params and 10,475-vertex meshes]
    X --> V[Strict author-protocol evaluation and visualization]
```

Toàn pipeline fail-closed: thiếu frame, sai topology, sai hash model, không có rotation hypothesis hợp lệ hoặc sinh loss không hữu hạn đều gây lỗi thay vì âm thầm loại mẫu.

## 5. Cache quan sát và feature độ bất định

Tensor quan sát có dạng:

```text
joints_3d:    [T, S, J, 3]
rotations:    [T, S, J, 3, 3]
valid_3d:     [T, S, J]
valid_rot:    [T, S, J]
features:     [T, S, J, 8]
keypoints_2d: [T, S, J, 2]
```

Vector feature 8 chiều hiện tại gồm:

1. prior risk cố định theo source;
2. nghịch đảo kích thước hand crop cho WiLoR;
3. độ mơ hồ của score handedness WiLoR;
4. độ dài chuỗi frame thiếu liên tiếp;
5. độ lệch 3D của source so với SMPLer-X;
6. cờ invalid observation;
7. vị trí thời gian đã chuẩn hóa trong clip;
8. source ID.

Cache ghi hash của mọi file input, hash SMPL-X model, quy ước camera, đơn vị, shape trung bình và frame IDs. Các output legacy chỉ được đọc; không file nguồn/baseline nào bị sửa.

## 6. Calibrated uncertainty

### 6.1 Calibrator

V5 dùng MLP nhỏ:

```text
8 -> Linear(32) -> SiLU -> Linear(7)
```

Bảy output gồm `sigma_xyz[3]`, `sigma_rot[3]` và `risk_logit`. Spatial sigma bị chặn trong khoảng cấu hình `[0,002; 0,5]` mét. Calibrator được fit bằng Student-t negative log-likelihood để giảm ảnh hưởng outlier:

\[
\mathcal L_{t}(r,\sigma)
= \log \sigma
+ \frac{\nu+1}{2}\log\left(1+\frac{(r/\sigma)^2}{\nu}\right),
\qquad \nu=3.
\]

Artifact V5:

```text
signal4d/artifacts/calibration/sgnify_a1_leftmirror_v5_seed12345
```

Calibrator dùng 12 clip/260 frame: 8 clip để fit và 4 clip cho conformal calibration.

### 6.2 Grouped split-conformal scaling

Residual được nhóm theo tích Descartes:

```text
source_{0,1,2} x {body, left_hand, right_hand}
```

Với mỗi nhóm \(g\), scale \(q_g\) là quantile hữu hạn-mẫu của nonconformity score \(|r|/\sigma\). Sigma cuối:

\[
\hat\sigma_{t,s,j}=\operatorname{clip}
\left(q_{g(s,j)}\sigma_{t,s,j},\sigma_{min},\sigma_{max}\right).
\]

Nominal coverage là 90%; coverage thực tế từng nhóm nằm khoảng 90,0–91,1% trong artifact calibration.

### 6.3 Giới hạn calibration cần nêu thẳng

MLP có sinh `sigma_rot`, nhưng solver V5 hiện tái sử dụng relative weight từ spatial sigma cho rotation factor. Vì vậy chỉ có thể claim **spatial conformal calibration**. Không được claim interval góc SO(3) đã được calibrate độc lập. Đây là một mục tiêu sửa cho V6.

## 7. Phát hiện change-point và trọng số thời gian

Từ source body chính, V5 tính sáu thống kê theo frame:

- mean/max joint speed;
- mean/max joint acceleration magnitude;
- mean/max hand speed.

Mỗi chuỗi được robust-normalize bằng median/MAD rồi ánh xạ thành xác suất change \(p_t\). Trọng số temporal cho khớp \(j\):

\[
w_{t,j}=\operatorname{clip}\left[
w_0(1-p_t)^{\gamma}
\left(1+\alpha\frac{u_{t,j}}{\operatorname{median}(u)}\right),
0,05,5,0\right],
\]

với `gamma=2`. Ý nghĩa:

- source càng bất định thì cần regularization thời gian mạnh hơn;
- gần change-point thì hạ regularization để không xóa chuyển động ký hiệu nhanh;
- lower/upper clamp tránh mất hoàn toàn temporal prior hoặc oversmooth cực đoan.

## 8. Selective uncertainty-aware SO(3) refinement

### 8.1 Phần nào thực sự được tối ưu ở V5?

Cấu hình frozen `m1_a1_v5.yaml`:

| Nhóm tham số | Trạng thái |
|---|---|
| global orientation | freeze |
| translation | freeze |
| lower body và phần lớn body | freeze |
| body joint indices 17, 19 | optimize |
| left-hand pose | optimize |
| right-hand pose | freeze |
| face/jaw/eyes/expression | freeze |
| shape | lấy từ hypothesis, không mở tự do trong M1 |

Hai body index hiện hành tương ứng với left elbow/left wrist theo convention mà code đang dùng. Mọi bản mở rộng phải resolve joint bằng tên SMPL-X, không copy raw index giữa repository.

### 8.2 Khởi tạo

Mặc dù code hỗ trợ lựa chọn source theo uncertainty, V5 frozen dùng:

```yaml
initializer_mode: legacy_full
```

Nghĩa là pose khởi tạo coherent lấy từ fitted legacy A1 khi có. Đây là quyết định bảo thủ giúp không phá nghiệm tốt. Không được mô tả V5 là “uncertainty argmin initialization”.

### 8.3 Biểu diễn rotation

Biến tối ưu dùng rotation-6D và được chiếu thành ma trận quay hợp lệ. Residual giữa hai rotation là geodesic log-map:

\[
r_R(R,\bar R)=\left\|\log(\bar R^T R)\right\|_2.
\]

Temporal rotation dùng angular velocity và angular acceleration trên SO(3):

\[
\omega_t=\log(R_{t-1}^{T}R_t)f,
\qquad
a^R_t=\omega_t-\omega_{t-1}.
\]

Cách này tránh lấy hiệu trực tiếp axis-angle hoặc Euler, vốn không tôn trọng hình học của rotation.

### 8.4 Objective

Với tập tham số mở \(\Theta_{open}\), solver tối thiểu hóa:

\[
\begin{aligned}
\mathcal L ={}&
\lambda_{3D}\mathcal L_{obs3D}
+0,1\mathcal L_{rot}
+\lambda_{2D}\mathcal L_{obs2D}\\
&+\lambda_T\mathcal L_{acc-pos}
+\lambda_R\mathcal L_{acc-SO(3)}
+\lambda_P\mathcal L_{prior}.
\end{aligned}
\]

Trọng số frozen:

```yaml
observation: 1.0
observation_2d: 1000.0
prior: 0.01
temporal: 0.001
temporal_rotation: 0.001
contact: 0.0
collision: 0.0
```

Observation 3D được chuẩn hóa bằng calibrated sigma. Position/rotation temporal dùng pseudo-Huber để không cho vài outlier chi phối toàn clip.

Optimizer: Adam, learning rate `5e-4`, tối đa 60 step, patience 12, gradient clip 10. V5 không chạy contact/collision; mọi claim contact-aware là không hợp lệ.

## 9. Windowing và consensus

Thiết kế tổng quát dùng window length 64, stride 32, context 8. Nếu một frame xuất hiện ở nhiều window:

- translation/shape được average có trọng số;
- rotation được ghép bằng weighted Karcher mean trên SO(3);
- uncertainty và diagnostics được tổng hợp có trọng số.

Tuy nhiên các clip SGNify hiện tại dài không quá 48 frame, ngắn hơn window 64. Vì vậy mỗi clip hiện chỉ tạo một optimization window. Không được dùng kết quả này để claim lợi ích thực nghiệm từ overlap consensus; đó mới chỉ là năng lực implementation đã tồn tại.

## 10. Multiscale hypotheses và temporal gate

### 10.1 Candidate set

Gate V5 chọn giữa bốn state:

| State | Ý nghĩa |
|---|---|
| A1 | baseline/fallback |
| M1×1,0 | selective refinement ở scale chuẩn |
| M1×1,5 | refinement mạnh hơn |
| M1×3,0 | refinement mạnh nhất trong candidate set |

Các scale thay đổi mức correction của cùng M1, không phải bốn model backbone độc lập.

### 10.2 Gate học gì?

Gate dùng ExtraTrees regressors. Target lịch sử của V5 là delta theo frame:

\[
y_{t,h}=TRV2V^{left}_{t,h}-TRV2V^{left}_{t,A1}.
\]

Feature gồm correction so với A1, disagreement giữa nguồn, uncertainty/risk/abstention, motion/change, ngữ cảnh lân cận ±1/±2 frame và thống kê clip. Dữ liệu gate gồm 45 clip/1.233 frame, cross-validation được group theo clip.

Sự thật cần trình bày chính xác:

- gate **có dùng GT lịch sử để học regressor**;
- gate **không dùng GT của frame/clip đang suy luận**;
- do đó tên đúng là **GT-free at inference**, không phải “self-supervised gate” hay “never trained with GT”.

### 10.3 Giải mã nhất quán thời gian

Sau khi dự đoán cost cho từng candidate, Viterbi tìm đường state có tổng cost thấp nhất:

\[
\min_{z_{1:T}}
\sum_t \hat c_{t,z_t}
+\lambda_{sw}\sum_{t=2}^{T}\mathbf 1[z_t\ne z_{t-1}],
\qquad \lambda_{sw}=8\text{ mm}.
\]

Switch penalty được freeze từ endpoint lịch sử trước prospective GT. Nó ngăn việc nhảy hypothesis từng frame chỉ để đạt lợi thế position nhỏ nhưng làm hỏng velocity/acceleration.

Trên endpoint prospective, gate chọn A1/M1×1,0/M1×1,5/M1×3,0 lần lượt 127/442/7/193 frame, với 0 switch bên trong clip. Output lặp lại byte-identical trên 112 file.

## 11. Official SGNify evaluation protocol

Nguồn chuẩn trong repository:

```text
signal4d/evaluate_author_protocol.py
data/evaluation_from_author/evaluate_new_fitting.py
signal4d/src/signal4d/evaluation/author_sgnify.py
```

### 11.1 `UBody(-F)` chính xác

Field chính thức:

```text
tr_upper_body_minus_face_mm
```

Vùng được tạo bằng vertex ở phía trên pelvis rồi loại face. Mỗi frame/vùng được căn chỉnh **translation-only bằng centroid của chính vùng đó**:

\[
TRV2V(P,G;V)=
\frac{1}{|V|}\sum_{i\in V}
\left\|
(P_i-\bar P_V)-(G_i-\bar G_V)
\right\|_2.
\]

Không dùng rotation alignment, scale alignment hay Procrustes. Vì vậy đổi camera/global translation không thể trực tiếp làm giảm `UBody(-F)`; phải sửa hình học/pose tương đối của torso–shoulder–arm–wrist–hand.

### 11.2 Quy tắc bắt buộc

- Prediction frame `k` ghép với GT frame `2*k` theo convention tác giả.
- Topology phải đúng SMPL-X neutral 10.475 vertex và đúng face array.
- Thiếu/thừa frame đều fail.
- Với class `0` one-handed sign, bỏ metric tay trái và loại vertex tay trái khỏi các region metric còn lại.
- Author primary aggregation là vertex-micro; clip-macro và paired bootstrap được báo thêm để đánh giá độ ổn định qua sign.

## 12. Kết quả đã đạt

### 12.1 Full 57 sign/1.493 frame — author vertex-micro

| Metric (mm, thấp hơn tốt hơn) | DexAvatar HaMeR + SignBPoser/SignHPoser | SIGNAL-4D V5 | Delta V5 − baseline | Relative reduction |
|---|---:|---:|---:|---:|
| TR all | 42,5867 | **42,1434** | **−0,4434** | 1,04% |
| TR upper body | 26,4560 | **26,1935** | **−0,2625** | 0,99% |
| TR left hand | 13,5735 | **11,6651** | **−1,9084** | 14,06% |
| TR right hand | 12,9271 | **11,8329** | **−1,0942** | 8,46% |
| **Official UBody(-F)** | 29,9074 | **29,5932** | **−0,3142** | 1,05% |
| TR upper body minus head | 40,7960 | **40,2643** | **−0,5317** | 1,30% |

Coverage: 57/57 sign, 1.493/1.493 frame, 100% cho cả hai method.

Clip-macro `UBody(-F)` cũng giảm từ **30,4200** xuống **30,0835 mm**. Full set được dùng trong development/audit, nên bảng này là bằng chứng diagnostic mạnh về coverage và độ tương thích protocol, không phải một confirmatory test còn độc lập.

### 12.2 Prospective extended-post 56 sign/769 frame — clip-macro

| Metric | A1 | SIGNAL-4D V5 | Delta | Paired 95% CI |
|---|---:|---:|---:|---:|
| Left hand | 24,9751 | **22,8340** | **−2,1411** | [−2,9547; −1,4191] |
| Upper body endpoint của report | 39,8024 | **38,7279** | **−1,0745** | [−1,5975; −0,6002] |
| Right hand | 13,1358 | 13,1378 | +0,0020 | [−0,0002; +0,0042] |
| Velocity error | 6,5377 | 6,5234 | −0,0143 | [−0,0265; +0,0013] |
| Acceleration error | 137,7112 | **136,8182** | **−0,8929** | [−1,2831; −0,4353] |
| Jerk error | 3.678,6309 | **3.657,0457** | **−21,5851** | [−34,8538; −7,1724] |

“Upper body endpoint” trong bảng prospective trên là endpoint lịch sử của report, **không được đồng nhất** với official `tr_upper_body_minus_face_mm`. Khi chạy structured author evaluator trên cùng prospective set, official vertex-micro `UBody(-F)` là **33,2424 → 33,2035 mm**, delta **−0,0390 mm**; clip-macro là **33,5162 → 33,4099 mm**.

### 12.3 Tại sao tay phải cải thiện dù M1 freeze tay phải?

M1 selective solver không optimize right-hand pose. Gain tay phải ở full diagnostic đến từ việc gate chọn giữa baseline và các hypothesis toàn-SMPL-X đã được canonicalize/ghép theo clip, cùng ảnh hưởng gián tiếp của body/wrist hypothesis. Không được mô tả đây là “explicit right-hand refinement”. V6 phải có ablation riêng nếu mở tay phải.

## 13. Artifact và đường dẫn xem kết quả

| Nội dung | Đường dẫn |
|---|---|
| Full prediction release | `signal4d/runs/signal4d_v5_full1493_20260820` |
| Strict SIGNAL-4D OBJ | `signal4d/outputs/strict_dexavatar_obj_20260820/full_1493/SIGNAL4D_v5` |
| Strict DexAvatar baseline OBJ registry | `signal4d/outputs/strict_dexavatar_obj_20260820/full_1493/DexAvatar_HaMeR` |
| 1.493 fitting overlays + mesh links | `signal4d/outputs/reconstruction_signal4d_v5_full1493_20260820` |
| Structured author evaluation | `signal4d/reports/author_evaluator_strict_obj_20260820/full_1493` |
| Full release report | `signal4d/reports/full1493_signal4d_v5_20260820/REPORT.md` |
| Prospective report | `signal4d/reports/confirmatory_extended_post_v5.md` |
| Material passport | `signal4d/reports/material_passport.md` |
| Frozen method config | `signal4d/configs/method/m1_a1_v5.yaml` |
| Frozen gate config | `signal4d/configs/gating/a1_multiscale_v5.yaml` |

## 14. Reproducibility và isolation

- Toàn implementation mới nằm dưới `signal4d/`; legacy code/output chỉ là read-only input.
- Môi trường frozen: Python 3.10.19, PyTorch 2.1.1+cu121, CUDA 12.1.
- Hardware đã dùng: NVIDIA RTX 5880 Ada, 49.140 MiB VRAM.
- Seed: 12345.
- SMPL-X neutral model SHA-256: `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`.
- Full manifest SHA-256: `02e06c946f9400d8eb2b238c0297b07e188912121748db68ee1d66d12ea7c362`.
- Release prospective freeze hash: `0c5808308b6de3f965fa50f0b05bad21cfe04e9e9161e211f2b03740f825975d`.
- Calibration, cache, gate, predictions, OBJ và evaluator output đều có hash/provenance.

## 15. Contribution có thể viết trong paper

### 15.1 Calibrated uncertainty-aware selective SO(3) refinement

Một optimizer SMPL-X theo chuỗi sử dụng uncertainty đã được spatial conformal-calibrate theo nguồn/vùng để cân observation, đồng thời chỉ mở các khớp có mục tiêu rõ ràng. Rotation được tối ưu và regularize trên SO(3); change probability làm giảm temporal regularization ở chuyển động ký hiệu nhanh. Contribution không phải “dùng optimizer”, mà là tổ hợp có kiểm soát giữa calibration, selective parameter opening, geometry đúng của rotation và temporal adaptation.

### 15.2 GT-free-at-inference temporally coherent multiscale gating

Một gate học từ historical paired GT để ước lượng lợi ích của nhiều mức refinement, nhưng suy luận chỉ từ observation/uncertainty/motion features. Dynamic programming với switch cost chọn đường hypothesis coherent theo thời gian và có state fallback A1. Contribution phải gọi đúng là **GT-free at inference**, không phải unsupervised.

### 15.3 Protocol-safe expert composition

Các estimator khác topology/handedness/camera được đưa vào contract SMPL-X duy nhất, fail-closed và xuất direct mesh đúng topology để đánh giá byte/coverage-comparable với DexAvatar. Đây là contribution hệ thống/reproducibility, hỗ trợ hai contribution thuật toán ở trên.

## 16. Điều V5 chưa làm và không được claim

1. Không có contact-aware improvement thật: contact/collision weight bằng 0; M2/G4/G5 đã bị loại vì thiếu label và không có incremental value.
2. Không có semantic sign evaluator đáng tin cậy; chưa chứng minh semantic preservation.
3. Không có external dataset hoặc unseen-signer claim.
4. Không optimize explicit right hand trong M1.
5. Không calibrate rotation uncertainty độc lập.
6. Không dùng Sapiens adapter trong frozen V5 dù adapter đã tồn tại.
7. Không có optical flow, M3 hay explicit body–hand seam module.
8. Không có empirical overlap-window gain vì clip ngắn hơn window.
9. Gate target V5 thiên về tay trái, chưa trực tiếp tối ưu official `UBody(-F)`.
10. Full-1.493 result không được gọi là confirmatory SOTA do test exposure.

## 17. Kết luận giữ V5

V5 nên được giữ nguyên như một immutable control và fallback state vì:

- đã vượt native DexAvatar `method_hamer` trên toàn bộ 1.493 frame theo cùng evaluator;
- output đúng SMPL-X topology, coverage đủ và có visualization;
- gate inference không cần GT và có byte-exact reproducibility;
- mọi module mới có thể được thêm dưới dạng candidate rồi quay về V5 khi bất định.

Kế hoạch nghiên cứu tiếp theo không sửa trực tiếp V5. Tất cả module mới phải chứng minh cải thiện official `UBody(-F)` trên split mới, đồng thời không làm xấu hai tay, trước khi được phép đổi tên thành một release tốt hơn.
