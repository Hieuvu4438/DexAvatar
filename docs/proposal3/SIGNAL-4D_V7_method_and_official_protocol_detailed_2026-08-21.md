# SIGNAL-4D V7: phương pháp và protocol đánh giá chính thức

## Material Passport

| Trường | Giá trị |
|---|---|
| Method | `SIGNAL4D_V7_EXT_How2SignResidualTransport` |
| Tên ngắn dùng trong bảng | `V7_EXT` |
| Ngày khóa | 2026-08-21, Asia/Ho_Chi_Minh |
| Algorithm-lock commit | `5f9f7a35ce2d2c5a4d4478f552ef4a7205bc3006` |
| Training supervision | How2Sign, không dùng SGNify target |
| Official evaluation | SGNify author code, 57 signs, 1.493 frames |
| Trạng thái kiểm chứng | `VERIFIED`, deterministic rerun |
| Phạm vi claim | cải thiện UBody và UBody-F so với frozen V6 |
| Mức tin cậy | `CAUTION`: kết quả đúng protocol nhưng gain nhỏ và All bị giảm |

Tài liệu này mô tả đúng implementation đã tạo ra kết quả chính thức. Trong
paper, tên **V7** phải chỉ phiên bản `V7_EXT` ở tài liệu này. Không được dùng
tên V7 chung cho router NLF exploratory đã học từ nhãn SGNify.

## 1. Bài toán

Đầu vào là chuỗi ảnh RGB đơn camera của một sign. Hệ thống cần khôi phục chuỗi
SMPL-X có cùng frame ID với manifest đánh giá. Với frame (t), output hình học
được biểu diễn bởi

\[
\mathcal{M}_t =
\operatorname{SMPLX}(\boldsymbol{\theta}_t,
\boldsymbol{\beta},\boldsymbol{\psi}_t,\mathbf{c}_t),
\]

trong đó (\boldsymbol{\theta}_t\) là các rotation cục bộ, (\boldsymbol{\beta})
là body shape, (\boldsymbol{\psi}_t) là expression và
(\mathbf{c}_t) là translation. Mesh cuối có 10.475 vertices.

Mục tiêu riêng của V7 là giảm sai số upper body của frozen SIGNAL-4D V6 mà
không thay model expert, không học trên SGNify, không thay pose ngón tay và
không ghép vertices từ nhiều mesh. V7 thực hiện mục tiêu này bằng cách học một
residual expert trên How2Sign, sau đó vận chuyển residual đó lên pose V6 trong
không gian (SO(3)).

## 2. Tổng quan end-to-end

```mermaid
flowchart LR
    H[How2Sign RGB + 133-point 2D tracks] --> P[2D-guided temporal pseudo-target]
    P --> T[Factorized spatio-temporal SO(3) refiner]
    T --> C[Frozen external checkpoint]
    S[SGNify RGB, không GT] --> O[51-joint observation cache]
    O --> E[External inference: R_init và R_ext]
    C --> E
    V[Frozen SIGNAL-4D V6] --> X[SO(3) residual transport]
    E --> X
    X --> W[Global-wrist compensation]
    W --> M[One coherent SMPL-X forward]
    M --> Y[57 signs / 1.493-frame safetensors]
    Y --> A[Released author TR-V2V evaluator]
```

Training và official evaluation là hai pha tách biệt. How2Sign tạo toàn bộ
tham số học và quyết định checkpoint. SGNify chỉ cung cấp image observations
khi inference; SGNify ground-truth mesh chỉ được mở trong evaluator sau khi
output V7 đã bị khóa.

## 3. Frozen V6 reference

V7 không chạy lại hoặc sửa code V5/V6. Reference pose cho frame (t) là

\[
\mathbf{R}^{V6}_t =
\{R^{V6}_{t,j}\}_{j=0}^{54}, \qquad R^{V6}_{t,j}\in SO(3).
\]

Artifact được dùng trực tiếp:

`signal4d/runs/signal4d_v6_final_full1493_20260821/predictions`

V6 cung cấp 55 local rotations, translation, frame IDs và các trường
diagnostic. Shape và expression để decode được đọc từ frozen DexAvatar
parameter source `outputs/method_hamer`, là nguồn hình thể đã dùng trong lane
V6. V7 không tối ưu lại shape, expression, root translation hoặc camera.

## 4. External How2Sign supervision

### 4.1 Tách dữ liệu

Training data được lấy từ How2Sign và chia theo source group, trong đó source
group là 11 ký tự video ID. Train và validation không giao nhau theo source
group.

| Split | Accepted clips | Frames | Vai trò |
|---|---:|---:|---|
| Train | 10.822 | 346.304 | cập nhật model |
| Validation | 498 | 15.936 | chọn checkpoint |
| Calibration | 497 | 15.904 | tồn tại trong corpus nhưng không dùng để chọn V7_EXT |

Các manifest đều khai báo `dataset=How2Sign`, `sgnify_excluded=true` và
`target_provider=How2Sign 2D-track temporal bundle adjustment v1`. Số frame
SGNify tham gia training hoặc calibration bằng **0**.

### 4.2 Khởi tạo và pseudo-target

Mỗi clip How2Sign bắt đầu từ SMPLer-X H32 frozen per-frame. Ordered 133-point
2D tracks được ánh xạ về 51 joints gồm 21 body joints, 15 left-hand joints và
15 right-hand joints. Pseudo-target không phải SGNify 3D ground truth; nó là
kết quả bundle adjustment được dẫn hướng bởi 2D tracks.

Với pose initializer (\theta^0), target builder tối ưu correction
(\delta\theta) trong 30 Adam iterations, learning rate 0,03. Correction được
giới hạn 12 độ cho body và 18 độ cho hands. Objective target builder là

\[
\mathcal{L}_{BA} =
\mathcal{L}_{2D}
+ 0.02\mathcal{L}_{anchor}
+ 0.10\mathcal{L}_{velocity}
+ 0.05\mathcal{L}_{acceleration}.
\]

(\mathcal{L}_{2D}) là confidence-weighted Smooth-L1 giữa projected SMPL-X
joints và How2Sign tracks. Anchor phạt correction quá lớn. Hai temporal terms
phạt first- và second-order differences của correction theo chuỗi.

Một clip chỉ được giữ nếu:

- tổng reprojection error giảm ít nhất 0,5%;
- mỗi vùng body, left hand và right hand không giảm chất lượng quá 2%;
- correction không phải identity.

Pseudo-target này là external 2D-guided supervision. Nó không được mô tả như
motion-capture ground truth hoặc exact SMPL-X annotation.

Target builder dùng bounded additive correction trong local axis-angle để tạo
pseudo-label. Đây chỉ là bước chuẩn bị supervision. Learned refiner và V7
integration ở các phần sau dùng valid (SO(3)) composition; paper không được
đồng nhất hai phép cập nhật này.

## 5. Observation representation

Mỗi frame-joint được biểu diễn bởi token 45 chiều:

| Thành phần | Số chiều | Nội dung |
|---|---:|---|
| Initial rotation | 6 | continuous 6D rotation |
| Rotation velocity | 6 | first difference của rotation-6D |
| Rotation acceleration | 6 | second difference của rotation-6D |
| Observation state | 8 | confidence, presence, missing, crop scale, truncation, motion innovation, duplicate/disagreement và auxiliary flag |
| 2D keypoint | 2 | tọa độ ảnh chuẩn hóa |
| 2D-valid flag | 1 | keypoint có hợp lệ hay không |
| Fixed reliability (u_0) | 1 | độ tin cậy observation |
| Torso-frame position | 3 | joint position trong torso coordinate |
| Torso-valid flag | 1 | validity của torso position |
| Wrist-local position | 3 | hand position trong wrist coordinate |
| Wrist-valid flag | 1 | validity của wrist-local position |
| Palm normal | 3 | hướng palm cho các hand tokens |
| Palm-valid flag | 1 | validity của palm normal |
| Time delta | 1 | normalized inter-frame interval |
| 2D reprojection residual | 2 | projected initializer trừ observed keypoint |
| **Tổng** | **45** | |

How2Sign keypoints từ ([0,1]) được đổi thành model coordinates ([-1,1]).
Reprojection residual được nhân scale 10 trước khi đi vào network. Missing
coordinates được đặt bằng zero và luôn đi kèm validity, nên zero không bị hiểu
nhầm thành observation thật.

How2Sign cache cung cấp trực tiếp fixed reliability
(u_0=confidence\times valid). Với các cache không cung cấp trường này,
schema có fallback kết hợp confidence, presence, missing, truncation,
duplicate flag và motion innovation. V7_EXT không dự đoán uncertainty:
`predict_uncertainty=false`. Vì vậy paper phải gọi thành phần này là
**reliability-conditioned attention**, không gọi là learned uncertainty head.

Cụ thể, khi cache không cung cấp sẵn (u_0), implementation dùng

\[
u_0=c\,p\,(1-m)(1-t)(1-0.5d)\exp(-2\nu),
\]

với (c,p,m,t,d,\nu) lần lượt là confidence, presence, missing, truncation,
duplicate và normalized motion innovation.

## 6. External spatio-temporal residual expert

### 6.1 Kiến trúc

Network nhận tensor ([B,T,51,45]), với (T\leq64). Cấu hình frozen gồm:

| Tham số | Giá trị |
|---|---:|
| Hidden size | 256 |
| Factorized blocks | 6 |
| Attention heads | 8 |
| MLP ratio | 4 |
| Dropout | 0,1 |
| Maximum frames | 64 |
| Temporal mode | bidirectional, `causal=false` |
| Body residual bound | 25 degrees |
| Hand residual bound | 35 degrees |
| Learned uncertainty | disabled |
| Reprojection skip | enabled, zero-initialized |

Mỗi factorized block lần lượt áp dụng:

1. spatial self-attention giữa 51 joints trong cùng frame;
2. temporal self-attention của cùng joint qua các frame, với learned relative
   temporal bias;
3. group attention giữa torso, left arm, right arm, left hand và right hand;
4. feed-forward network với GELU.

Reliability (u_0\in[0,1]) điều chỉnh key/value bằng
(0.1+0.9u_0). Observation kém tin cậy vẫn giữ một residual weight 0,1 để
network không mất hoàn toàn context. Reprojection skip là linear map từ toàn bộ
51 x 2 residuals tới 51 x 3 raw rotation corrections.

### 6.2 Bounded residual trên SO(3)

Network sinh raw tangent vector (\mathbf{d}_{t,j}\in\mathbb{R}^3) và gate
(g_{t,j}\in[0,1]). Norm được chặn mềm:

\[
\bar{\mathbf d}_{t,j} =
\frac{\theta^{max}_j\tanh(\lVert\mathbf d_{t,j}\rVert/
\theta^{max}_j)}{\lVert\mathbf d_{t,j}\rVert}\mathbf d_{t,j}.
\]

Refined external rotation là left composition:

\[
R^{ext}_{t,j} =
\operatorname{Exp}(g_{t,j}\bar{\mathbf d}_{t,j})R^{init}_{t,j}.
\]

Cách biểu diễn này luôn tạo rotation hợp lệ. Axis-angle vectors không được
trung bình hoặc cộng trực tiếp trong V7 integration.

### 6.3 Training objective

Checkpoint dùng các loss có weight khác zero sau:

\[
\begin{aligned}
\mathcal L_{train}={}&
1.0\mathcal L_{rot}
+2.0\mathcal L_{vertex}
+0.25\mathcal L_{vel}
+0.10\mathcal L_{acc}\\
&+0.05\mathcal L_{anchor}
+0.01\mathcal L_{bio}.
\end{aligned}
\]

- (\mathcal L_{rot}): shortest geodesic distance giữa predicted và target
  rotations.
- (\mathcal L_{vertex}): balanced SMPL-X vertex loss, chuẩn hóa độc lập cho
  upper body, left hand và right hand trước khi tổng hợp.
- (\mathcal L_{vel}), (\mathcal L_{acc}): khớp motion first/second order
  với pseudo-target thay vì chỉ ép chuyển động phẳng.
- (\mathcal L_{anchor}): geodesic distance tới reliable initializer, có
  observation-confidence weighting.
- (\mathcal L_{bio}): soft penalty khi raw correction norm vượt 0,5 radian.

Joint, fingertip, palm, direct observation và uncertainty losses có weight 0
trong checkpoint này. Chúng tồn tại trong framework nhưng không phải thành phần
thực nghiệm của V7_EXT và không được liệt kê như contribution đạt kết quả.

### 6.4 Corruption mixture và optimization

Training batch sử dụng 50% real residual, 25% synthetic corruption và 25%
clean identity cases. Synthetic bursts dài 2--16 frames gồm upper-body,
single/both-hand, finger-chain, wrist-attachment, keypoint dropout, hand swap
và crop truncation.

Optimizer là AdamW với learning rate (2\times10^{-4}), weight decay 0,05,
gradient clipping 1,0, BF16, batch size 48, EMA decay 0,999 và seed 42. Schedule
gồm 5% linear warm-up rồi cosine decay. Run đặt giới hạn 5.000 steps; checkpoint
tốt nhất thực tế ở step 1.500 và dùng EMA weights.

Model được khởi tạo không gian từ T1 How2Sign geometry checkpoint
`c86a95a7e900dda02a8f8ebc1bbe0ef36c656e4186ec4ef24507da65286b1b9e`.
166 tensors tương thích được nạp. Khi input tăng từ 43 lên 45 chiều, learned
prefix được giữ và hai feature reprojection mới được zero-initialize;
reprojection skip cũng bắt đầu từ zero. Vì vậy model khởi đầu đúng hành vi T1
thay vì thêm một correction ngẫu nhiên.

Checkpoint selection chỉ dùng How2Sign validation. Với region
(r\in\{UBody,LHand,RHand\}), đặt
(q_r=E^{pred}_r/E^{init}_r). Score là

\[
S=\frac{1}{3}\sum_r q_r
+0.5\sum_r\max(0,q_r-1.01).
\]

Score nhỏ hơn tốt hơn; checkpoint frozen đạt 0,873438. Công thức phạt một vùng
nếu regression vượt 1%, ngăn việc đổi gain của một vùng lấy degradation lớn ở
vùng khác. Không metric SGNify nào tham gia checkpoint selection.

## 7. V7 cross-initializer residual transport

External inference trên SGNify tạo hai rotations cho mỗi joint: initializer
(R^{init}) và external refined result (R^{ext}). V7 không thay toàn bộ V6
bằng (R^{ext}). Nó trích residual tương đối từ external expert:

\[
\Delta R_{t,j}=R^{ext}_{t,j}(R^{init}_{t,j})^\top.
\]

Sau đó residual được ánh xạ qua tangent space và vận chuyển lên V6:

\[
R^{V7}_{t,j}=
\operatorname{Exp}\!\left(\alpha\operatorname{Log}(\Delta R_{t,j})\right)
R^{V6}_{t,j}.
\]

(\alpha=1.0) được khóa trước evaluation. Không có alpha search trên SGNify.
Residual transport tách “correction learned by external expert” khỏi absolute
pose của initializer, nhờ đó giữ V6 làm reference system thay vì thay model
expert.

### 7.1 Joint set

Chỉ 10 upper-body joints được transport:

| Cache index | SMPL-X index | Joint |
|---:|---:|---|
| 2 | 3 | spine1 |
| 5 | 6 | spine2 |
| 8 | 9 | spine3 |
| 11 | 12 | neck |
| 12 | 13 | left collar |
| 13 | 14 | right collar |
| 15 | 16 | left shoulder |
| 16 | 17 | right shoulder |
| 17 | 18 | left elbow |
| 18 | 19 | right elbow |

Pelvis/root, hips, legs, feet, head, jaw, eyes và toàn bộ local finger
rotations giữ nguyên V6. Wrist indices 20 và 21 không nhận external residual;
chúng được giải bằng kinematic compensation ở bước tiếp theo.

## 8. Global-wrist preservation

Nếu thay shoulder/elbow nhưng giữ nguyên local wrist rotation, global wrist
orientation vẫn thay đổi do ancestor chain đã đổi. Điều này có thể phá hand
orientation dù local finger pose giống V6. V7 áp đặt invariant rõ ràng.

Với global rotation

\[
G_{t,j}=G_{t,p(j)}R_{t,j},
\]

local wrist rotation mới được giải bởi

\[
R^{V7}_{t,w}=
(G^{V7}_{t,p(w)})^\top G^{V6}_{t,w},
\qquad w\in\{20,21\}.
\]

Suy ra (G^{V7}_{t,w}=G^{V6}_{t,w}). Hai global wrists vì thế giữ nguyên
đến numerical tolerance, trong khi shoulder và elbow có thể cải thiện. Local
finger rotations 25:54 giữ nguyên V6. Đây là kinematic compensation, không
phải vertex replacement.

Observed maximum global-wrist error của full run là
(9.5367\times10^{-7}), nhỏ hơn threshold (10^{-5}). Maximum rotation error
trên tập joint bị khóa bằng 0.

## 9. Coherent SMPL-X decoding

Sau residual transport và wrist compensation, toàn bộ 55 rotations được đổi
từ matrix về rotation vector và đưa qua **một** SMPL-X neutral forward pass:

- `use_pca=false`;
- `flat_hand_mean=true`;
- 10 betas và 10 expression coefficients;
- topology 10.475 vertices;
- unit của output là meter;
- coordinate convention là OpenCV: x-right, y-down, z-forward.

Internal SMPL-X decode dùng phép quay 180 độ quanh camera x-axis qua vector
`[1,-1,-1]`, sau đó vertices và joints được đổi lại về convention output.
Không region nào lấy vertices trực tiếp từ external mesh. Vì vậy surface cuối
liên tục và tuân theo một body shape cùng một kinematic tree.

## 10. Safety và leakage contract

Implementation dừng với lỗi nếu bất kỳ điều kiện nào sau bị vi phạm:

- checkpoint hash khác checkpoint frozen;
- checkpoint train/val path không chứa How2Sign hoặc chứa SGNify/SMPL-X GT;
- manifest không khai báo `How2Sign` và `sgnify_excluded=true`;
- inference cache chứa `target_axis_angle` hoặc `target_joint_positions`;
- Lane inference không được tạo bởi đúng checkpoint hash;
- V6 frame IDs khác cache frame IDs;
- output directory đã tồn tại;
- locked-joint error lớn hơn (10^{-6});
- global-wrist error lớn hơn (10^{-5});
- prediction có non-finite values hoặc sai topology trong evaluator.

Inference cache chính thức có `has_target=false`; calibration path trong Lane
run là `null`; số `fallback_group_frames` là 0. SGNify GT không được load trong
method run (`sgnify_gt_loaded=false`).

Training config trỏ `geometry.assets_root` tới thư mục author assets chỉ để
đọc neutral SMPL-X topology và immutable region masks cho differentiable
vertex loss. Không SGNify OBJ, per-frame target hay author metric nào được đọc
trong training. Việc sử dụng model topology/vertex indices không phải benchmark
supervision.

## 11. Output contract

Mỗi sign có cấu trúc:

```text
predictions/<SIGN>/
├── prediction.safetensors
└── metadata.json
```

Các tensor chính gồm:

| Tensor | Shape | Ý nghĩa |
|---|---|---|
| `frame_ids` | `[T]` | exact manifest frame IDs |
| `rotations` | `[T,55,3,3]` | local SMPL-X rotations |
| `translation` | `[T,3]` | frozen V6 translation |
| `vertices` | `[T,10475,3]` | coherent SMPL-X surface |
| `joints_3d` | `[T,55,3]` | decoded joints |
| `uncertainty` | `[T,55]` | trường kế thừa từ V6, không phải V7 learned uncertainty |
| `risk_score` | `[T,3]` | trường kế thừa từ V6 |
| `abstain` | `[T,3]` | trường kế thừa từ V6 |

Metadata ghi frame IDs, model/config/checkpoint hashes, coordinate convention,
unit, training corpus và artifact hash. Official full output nằm tại:

`signal4d_v7_nlf_fusion/runs/v7_ext_how2sign_residual_transport_v1_full1493_20260821/predictions`

## 12. Official SGNify evaluation protocol

### 12.1 Code và population

Evaluator adapter import trực tiếp ba hàm của code tác giả:
`read_verts_and_faces`, `transl_point_error` và
`point_error_common_center` từ:

`data/evaluation_from_author/evaluate_new_fitting.py`

Population được khóa bằng
`signal4d/artifacts/manifests/sgnify_available_15fps_development.jsonl`, gồm
57 signs và 1.493 frames. `frame-policy=manifest` yêu cầu mỗi method có đúng
frame IDs theo manifest. Với prediction frame ID (f), GT được đọc tại
`data/smplx_gt/<SIGN>/<2f:05d>.obj`, đúng quan hệ 15-fps prediction với author
GT indexing.

V6 và V7 phải cùng:

- 57/57 signs;
- 1.493/1.493 frames;
- SMPL-X neutral model hash;
- coordinate convention;
- topology và vertex ordering;
- official region index files.

### 12.2 TR-V2V

Cho region vertex set (\mathcal I_r), prediction vertices (P_i) và GT
vertices (Q_i), evaluator căn chỉnh **translation riêng cho từng region**:

\[
\tilde P_i=P_i-\frac{1}{|\mathcal I_r|}\sum_{k\in\mathcal I_r}P_k,
\quad
\tilde Q_i=Q_i-\frac{1}{|\mathcal I_r|}\sum_{k\in\mathcal I_r}Q_k.
\]

Sai số vertex là

\[
e_i=\lVert\tilde P_i-\tilde Q_i\rVert_2.
\]

Official summary dùng author vertex-micro aggregation:

\[
TR_r = 1000\times
\frac{\sum_{t}\sum_{i\in\mathcal I_r}e_{t,i}}
{\sum_t|\mathcal I_r|}\;\text{mm}.
\]

Đây không phải Procrustes alignment: evaluator chỉ loại translation, không
xoay và không scale prediction.

### 12.3 Official regions và class-0 rule

Regions gồm All, left hand, right hand, above-pelvis upper body,
upper-body-minus-head và upper-body-minus-face. Hand indices lấy từ
`MANO_SMPLX_vertex_ids.pkl`; body regions lấy đúng các `.npy` đi kèm code tác
giả.

Với sign class `0`, tác giả loại left-hand vertices khỏi mọi non-left-hand
region và không ghi LHand metric cho sign đó. Vì vậy LHand bootstrap chỉ có 42
eligible signs; các metric khác có 57 signs. Adapter giữ nguyên rule này.

### 12.4 Statistical analysis

Official vertex-micro score là primary number. Để ước lượng uncertainty theo
đơn vị độc lập hơn, secondary analysis resample paired sign clips 100.000 lần,
seed 12345. Estimand secondary là equal-weight sign-macro delta
(V7-V6). Negative delta nghĩa là V7 tốt hơn.

UBody-F là preregistered primary endpoint. All, UBody, LHand và RHand là
supporting/safety endpoints. Không claim family-wise multiple-testing
correction.

## 13. Official results

TR-V2V, mm; thấp hơn tốt hơn:

| Region | V6 | V7_EXT | V7 - V6 | Relative change |
|---|---:|---:|---:|---:|
| All | 42,111111 | 42,376460 | +0,265349 | +0,630% |
| UBody | 26,139380 | **25,983900** | **-0,155480** | **-0,595%** |
| UBody-H | 40,127560 | **39,917173** | **-0,210387** | **-0,524%** |
| UBody-F | 29,519389 | **29,368032** | **-0,151357** | **-0,513%** |
| LHand | 11,633903 | 11,634084 | +0,000181 | +0,002% |
| RHand | 11,805594 | 11,808403 | +0,002809 | +0,024% |

Paired sign bootstrap:

| Region | Mean sign-macro delta | 95% percentile CI | Signs improved / worse |
|---|---:|---:|---:|
| All | +0,289281 | [+0,224117, +0,353340] | 5 / 52 |
| UBody | -0,131790 | [-0,228553, -0,033603] | 36 / 21 |
| UBody-F | -0,126628 | [-0,221707, -0,030069] | 34 / 23 |
| LHand | +0,000024 | [-0,000534, +0,000582] | 21 / 21, 42 eligible |
| RHand | +0,002853 | [+0,002179, +0,003540] | 9 / 48 |

UBody và UBody-F cải thiện dưới cả vertex-micro lẫn sign-macro aggregation.
Gain tuyệt đối nhỏ. All regression cho thấy thay đổi upper-body kinematics lan
ra surface vertices ngoài official upper-body mask. RHand regression có hướng
nhất quán nhưng độ lớn thực tế chỉ khoảng 0,003 mm.

Temporal safety so với V6:

| Metric | V6 | V7_EXT | Relative change |
|---|---:|---:|---:|
| Velocity error | 6,542203 | 6,542363 | +0,0024% |
| Acceleration error | 142,894706 | 142,905806 | +0,0078% |
| Jerk error | 3811,576867 | 3811,870788 | +0,0077% |

Các thay đổi đều thấp hơn ngưỡng safety 2%.

## 14. Reproducibility commands

### 14.1 Materialize V7

```bash
PYTHONPATH=.:phase2_refiner/src python \
  signal4d_v7_nlf_fusion/external_how2sign_residual_transport.py \
  --config signal4d_v7_nlf_fusion/configs/v7_external_how2sign_residual_transport_v1.json \
  --cache-root cache/phase2/lane_l_a1_ensemble_reprojection_v2 \
  --external-prediction-root outputs/phase2_lane_l_reprojection_v6_seed42 \
  --external-checkpoint outputs/phase2_training/t2_how2sign_2d_temporal_reprojection_v6_seed42/best.pt \
  --external-train-manifest cache/phase2/t2_how2sign_2d_temporal_reprojection_v2/splits/train.json \
  --external-val-manifest cache/phase2/t2_how2sign_2d_temporal_reprojection_v2/splits/val.json \
  --v6-root signal4d/runs/signal4d_v6_final_full1493_20260821/predictions \
  --baseline-parameter-root outputs/method_hamer \
  --model-path data/evaluation_from_author/data/data/SMPLX_NEUTRAL.npz \
  --output-root <NEW_APPEND_ONLY_OUTPUT>
```

`<NEW_APPEND_ONLY_OUTPUT>` bắt buộc là path mới vì implementation từ chối ghi
vào directory đã tồn tại.

### 14.2 Run official author evaluation

```bash
PYTHONPATH=signal4d/src python -m signal4d.cli.main evaluate-author-sgnify \
  --manifest signal4d/artifacts/manifests/sgnify_available_15fps_development.jsonl \
  --method V6=signal4d/runs/signal4d_v6_final_full1493_20260821/predictions \
  --method V7_EXT=signal4d_v7_nlf_fusion/runs/v7_ext_how2sign_residual_transport_v1_full1493_20260821/predictions \
  --baseline V6 \
  --gt-root data/smplx_gt \
  --author-source data/evaluation_from_author/evaluate_new_fitting.py \
  --author-asset-root data/evaluation_from_author/data/data \
  --author-sign-file data/evaluation_from_author/data/data/signs.txt \
  --author-segment-file data/evaluation_from_author/data/data/segment.json \
  --frame-policy manifest \
  --prediction-format safetensors \
  --output <NEW_EVALUATION_OUTPUT>
```

### 14.3 Test

```bash
PYTHONPATH=.:phase2_refiner/src pytest -q signal4d_v7_nlf_fusion/tests
```

Kết quả hiện tại: 12 tests passed. Một independent full rerun tạo 57
prediction files và 456 tensors giống chính xác tensor-by-tensor với primary
run.

## 15. Artifact provenance

| Artifact | Path / SHA-256 |
|---|---|
| V7 implementation | `signal4d_v7_nlf_fusion/external_how2sign_residual_transport.py` |
| Implementation hash | `e0af1b87b0086deaf39a1f4c3fde42866f7515aa1dd67f60721bd4b06c3073b2` |
| V7 config | `signal4d_v7_nlf_fusion/configs/v7_external_how2sign_residual_transport_v1.json` |
| Config hash | `2fbb01d4d88f99e7f33047860630394c380c4882d7751a592a2c8f5f67ca3eba` |
| External checkpoint hash | `8c4e8c011fd69e51b6bc492012f1eb1667384cb095b2996a14935b0a26d8a482` |
| Train-manifest hash | `e597083a95d46b1c29bc99e3777c48bdb1a49d8f8cabff5f7e88fa5c6eb08758` |
| Validation-manifest hash | `7d2cfdc28146ce0d978b3f5a85e2cca44eeaacbe1827d4c00a879d822641667f` |
| SMPL-X model hash | `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992` |
| Official author-source hash | `e43e12a6659f0604752f0adb8b3c06cfb6ff8d910ed29137036351ee8fc44513` |
| Frozen frame-manifest hash | `02e06c946f9400d8eb2b238c0297b07e188912121748db68ee1d66d12ea7c362` |
| Official comparison hash | `6c7c001430315a66c993c8f789ed8b74192b18debc9716b871a6d3e87dcedb1c` |
| Bootstrap result hash | `b3d8e95991acd4ecc4dafc8b794177cc8f458c391a3b4bc13639ed453b8bcf4c` |

Official result files:

- `signal4d_v7_nlf_fusion/reports/author_v7_ext_how2sign_residual_transport_v1_full1493/comparison.json`
- `signal4d_v7_nlf_fusion/reports/author_v7_ext_how2sign_residual_transport_v1_full1493/paired_sign_bootstrap.json`
- `signal4d_v7_nlf_fusion/runs/v7_ext_how2sign_residual_transport_v1_full1493_20260821/run.json`
- `signal4d_v7_nlf_fusion/runs/v7_ext_how2sign_residual_transport_v1_full1493_20260821/clip_audit.json`

## 16. Contributions có thể viết trong paper

Ba đóng góp kỹ thuật đúng với artifact hiện tại là:

1. **External sign-domain spatio-temporal residual learning.** Một refiner
   factorized học bounded local SO(3) corrections từ How2Sign 2D tracks,
   reliability và temporal evidence mà không cần SGNify supervision.
2. **Cross-initializer Lie-group residual transport.** Correction học quanh
   frozen external initializer được tách thành relative (SO(3)) residual và
   vận chuyển lên frozen SIGNAL-4D V6, thay vì thay model expert hoặc averaging
   axis-angle/vertices.
3. **Kinematic-invariant specialist integration.** Upper-body ancestors được
   sửa trong khi hai global wrist orientations và local finger articulation
   được bảo toàn, sau đó toàn bộ body được decode bằng một coherent SMPL-X
   forward pass.

Đây là contribution description, chưa phải bằng chứng novelty tuyệt đối so với
toàn bộ literature. Novelty claim cuối cùng cần citation matrix và comparison
với các phương pháp residual fusion/whole-body hand integration gần nhất.

## 17. Paper-safe claim và giới hạn

Claim có thể dùng:

> Trained and selected exclusively on external How2Sign data, the proposed
> SO(3) residual-transport module reduces SIGNAL-4D V6 TR-V2V from 29.519 to
> 29.368 mm on UBody-F and from 26.139 to 25.984 mm on UBody under the released
> SGNify author protocol, with complete 1,493-frame coverage.

Không được claim từ experiment này:

- V7 cải thiện full-body, vì All tăng 0,265 mm;
- V7 cải thiện hand reconstruction, vì LHand trung tính và RHand giảm nhẹ;
- learned uncertainty-aware V7, vì checkpoint này tắt uncertainty head;
- global SOTA, nếu chưa có like-for-like literature table cùng protocol;
- pristine held-out confirmation, vì SGNify đã được xem trong các vòng nghiên
  cứu V5--V7 trước khi integration này được khóa.

Để nâng kết quả thành strong main-paper claim, cần freeze nguyên checkpoint,
config, joint set và alpha hiện tại, sau đó chạy đúng một lần trên một dataset
hoặc signer split sealed bổ sung. V6 và V7 phải được đánh giá cùng coverage và
paired sign-level confidence intervals.
