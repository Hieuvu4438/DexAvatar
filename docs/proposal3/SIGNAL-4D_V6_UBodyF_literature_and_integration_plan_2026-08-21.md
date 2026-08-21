# SIGNAL-4D V6 UBody(-F): literature review và kế hoạch ghép diffusion pose prior vào V5

Ngày nghiên cứu: **2026-08-21**
Trạng thái: **research and implementation plan — chưa phải kết quả thực nghiệm**
Ưu tiên metric: **official `tr_upper_body_minus_face_mm` / UBody(-F)**
Ràng buộc thiết kế: **không thay expert hiện tại; không sửa release V5; mọi module mới là candidate có fallback về V5**

## 1. Quyết định nghiên cứu sau literature review

Hướng chính được chọn không phải thay SMPLer-X bằng SMPLest-X, AiOS hay SAM 3D Body. Việc đổi backbone có thể tăng accuracy nhưng novelty thấp và làm khó tách contribution của SIGNAL-4D.

Hướng chính là ghép phương pháp của [DPoser-X, ICCV 2025 Oral](https://arxiv.org/abs/2508.00599) vào factor graph V5 dưới dạng **diffusion whole-body pose-prior factor**, sau đó tạo ba cải tiến riêng cho miền sign language:

1. **uncertainty-adaptive diffusion prior:** calibrated uncertainty của SIGNAL-4D quyết định vùng/frame nào cần prior mạnh;
2. **change-aware prior suppression:** giảm prior ở pha chuyển động ký hiệu nhanh để không ép sign pose về chuyển động “generic human”;
3. **geodesic body–wrist–MCP coupling:** dùng ngữ cảnh torso/arm và palm/MCP của WiLoR để sửa wrist/upper limb trên SO(3), nhưng không copy wrist rotation ngây thơ từ hand expert.

Tên làm việc của method mới:

```text
SIGNAL-4D V6 UQ-DiffPrior
= frozen V5 experts
+ region-calibrated DPoser-X regularization
+ change-aware SO(3) upper-body solver
+ body–wrist–MCP seam factor
+ UBody-first safe temporal gate
```

PAD-Hand là module giai đoạn sau cho LHand/RHand. Nó không phải đường chính để cải thiện UBody(-F). SMPLest-X và các expert mới chỉ được giữ trong danh sách external control/upper-bound, không nằm trong contribution hoặc default candidate.

## 2. Câu hỏi nghiên cứu

### 2.1 Câu hỏi chính

Giữ nguyên ba expert và output V5 hiện tại, một diffusion whole-body pose prior được điều tiết theo calibrated uncertainty và change-point, kết hợp với body–wrist–MCP coupling trên SO(3), có giảm official SGNify `UBody(-F)` mà không làm xấu LHand/RHand hay temporal dynamics không?

### 2.2 Population, intervention, comparator, outcome

| Thành phần | Định nghĩa |
|---|---|
| Population | monocular sign-language clips có prediction/GT theo SGNify author protocol |
| Intervention | V5 + UQ/change-aware DPoser-X factor + wrist/MCP seam factor |
| Comparator | immutable SIGNAL-4D V5; bảng phụ so với native DexAvatar `method_hamer` |
| Primary outcome | vertex-micro `tr_upper_body_minus_face_mm` |
| Secondary outcomes | clip-macro UBody(-F), TR all, LHand, RHand, velocity, acceleration, jerk |
| Safety outcomes | 100% frame/topology coverage, no-NaN, no hand regression, byte-exact fallback |

### 2.3 Giả thuyết cơ chế

V5 hiện chỉ mở left elbow/left wrist và tay trái; prior pose chỉ giữ nghiệm gần initialization. Vì `UBody(-F)` bỏ translation nhưng giữ sai lệch pose tương đối, dư địa chính nằm ở spine–clavicle–shoulder–elbow–wrist của cả hai bên. DPoser-X cung cấp gradient về pose manifold hợp lý ở các vùng observation bất định. Change-aware weighting ngăn prior generic làm mất các articulation nhanh, còn MCP seam đưa evidence hand-centric trở lại wrist và cánh tay theo đúng kinematic chain.

## 3. Điều official UBody(-F) buộc thiết kế phải làm

Code chuẩn:

```text
signal4d/evaluate_author_protocol.py
data/evaluation_from_author/evaluate_new_fitting.py
signal4d/src/signal4d/evaluation/author_sgnify.py
```

`UBody(-F)` là field:

```text
tr_upper_body_minus_face_mm
```

Mỗi frame và vùng được centroid-align translation-only:

\[
E_{UB-F}=\frac{1}{|V_{UB-F}|}\sum_{i\in V_{UB-F}}
\|(P_i-\bar P)-(G_i-\bar G)\|_2.
\]

Hệ quả:

- sửa camera translation hoặc global translation không trực tiếp hạ metric;
- world trajectory của WHAM/GVHMR không phải ưu tiên;
- cần sửa relative rotations/geometry của torso, vai, cánh tay, cổ tay và bàn tay;
- face bị loại, nên thêm face estimator không phục vụ objective chính;
- class-0 one-handed sign loại left-hand vertices khỏi các region còn lại, nên training/evaluation phải áp đúng mask theo class;
- output phải là direct SMPL-X 10.475 vertex, cùng face topology, cùng frame key và quy tắc GT frame `2*k`.

Đây là lý do DPoser-X factor và wrist–MCP coupling phù hợp hơn một module camera/global-motion.

## 4. Phương pháp literature review

### 4.1 Phạm vi tìm kiếm

Các domain được khảo sát:

- 3D sign-language reconstruction và holistic sign motion;
- expressive human pose and shape/whole-body mesh recovery;
- optimization-based HMR và learned pose prior;
- temporal HMR/motion refinement;
- monocular hand reconstruction, hand motion refinement và two-hand interaction;
- body–hand integration/wrist alignment;
- calibrated uncertainty, abstention và safe candidate selection.

Nguồn ưu tiên: CVPR/ICCV/ECCV/WACV/TPAMI, arXiv chính thức của paper, project page và repository chính chủ. Ngày kiểm tra cuối: 2026-08-21.

### 4.2 Query families

```text
3D sign language SMPL-X reconstruction temporal fitting
expressive whole-body mesh recovery upper body wrist hand public code checkpoint
diffusion whole-body pose prior test-time optimization SMPL-X
temporal human mesh refinement velocity acceleration SMPL-X
body hand wrist MCP integration whole-body mesh
physics-aware hand motion diffusion uncertainty checkpoint
SO(3) human motion prior partial pose fitting
```

### 4.3 Tiêu chí chọn

Một method được ưu tiên khi thỏa phần lớn các điều kiện:

1. tác động trực tiếp vào relative upper-body pose thay vì global translation;
2. ghép được sau expert hiện tại hoặc thành factor trong optimizer;
3. không yêu cầu thay backbone SIGNAL-4D;
4. biểu diễn tương thích SMPL/SMPL-X/MANO;
5. có code và checkpoint chính thức;
6. có license rõ;
7. có bằng chứng trên mesh recovery, pose completion hoặc motion denoising;
8. có fallback an toàn và có thể ablate độc lập.

### 4.4 Tiêu chí loại khỏi đường chính

- chỉ là model expert mới;
- lợi ích chủ yếu ở camera/world trajectory bị TR alignment loại bỏ;
- không có code/checkpoint dù implementation lớn;
- phụ thuộc interaction/object annotation không tồn tại trong SGNify;
- license chưa rõ hoặc xung đột với phân phối dự kiến;
- không thể xuất exact SMPL-X topology;
- có nguy cơ oversmooth sign articulation nhưng không có cơ chế uncertainty/change control.

## 5. Literature map

### 5.1 3D sign-language reconstruction

**SGNify.** [Project chính thức](https://sgnify.is.tue.mpg.de/) đặt bài toán fitting sign language với body/hand priors. Đây là nguồn protocol/GT lineage nhưng evaluator đính kèm trong repository mới là executable specification phải tuân thủ.

**DexAvatar.** [WACV 2026 paper](https://openaccess.thecvf.com/content/WACV2026/papers/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.pdf) và [official repository](https://github.com/kaustesseract/DexAvatar) kết hợp expert initialization với SignBPoser/SignHPoser fitting. `outputs/method_hamer` là comparator thực tế của chúng ta.

**SignAvatars.** [ECCV 2024 project](https://signavatars.github.io/) và [official repository](https://github.com/ZhengdiYu/SignAvatars) cung cấp 8,34 triệu SMPL-X annotation, 70.000 sequence, 153 signer và MANO version. Dữ liệu cần request cho nghiên cứu phi thương mại; RGB không được tái phân phối. Đây là nguồn tốt nhất để học một adapter sign-specific cho pose prior mà không cần GT SGNify mới.

**SignAvatar.** [Paper](https://arxiv.org/abs/2405.07974) và [repository](https://github.com/dongludeeplearning/SignAvatar) tập trung generation/reconstruction bằng latent motion model. Có liên quan tới distribution sign motion nhưng không phải drop-in test-time pose prior; checkpoint/data còn cần quy trình xin riêng.

Kết luận domain: dữ liệu sign có articulation khác generic AMASS/AGORA. Bất kỳ learned generic prior nào cũng phải có change-aware suppression và ablation sign-specific.

### 5.2 Whole-body mesh recovery

**SMPLer-X.** [NeurIPS 2023 paper](https://arxiv.org/abs/2309.17448) và [code](https://github.com/MotrixLab/SMPLer-X) là body expert hiện tại.

**SMPLest-X.** [TPAMI paper](https://arxiv.org/abs/2501.09782) và [code/checkpoint](https://github.com/MotrixLab/SMPLest-X) cho kết quả UBody tốt hơn SMPLer-X trong paper và public ViT-H checkpoint. Tuy nhiên nó chỉ tạo một expert initialization mạnh hơn. **Không chọn làm contribution hay default source V6.** Chỉ dùng làm external upper-bound nếu sau này cần biết giới hạn chất lượng observation.

**AiOS.** [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_AiOS_All-in-One-Stage_Expressive_Human_Pose_and_Shape_Estimation_CVPR_2024_paper.html) và [code](https://github.com/MotrixLab/AiOS) dùng progressive body/whole-body refinement và xuất SMPL-X. Đây vẫn là expert replacement, nên không chọn.

**OSX/UBody.** [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_One-Stage_3D_Whole-Body_Mesh_Recovery_With_Component_Aware_Transformer_CVPR_2023_paper.pdf) và [code](https://github.com/IDEA-Research/OSX) giới thiệu component-aware transformer và UBody dataset. Nó cung cấp bằng chứng rằng component-level reasoning quan trọng, nhưng model đã cũ hơn và là backbone thay thế.

**PyMAF-X.** [TPAMI paper](https://arxiv.org/abs/2207.06400) và [code](https://github.com/HongwenZhang/PyMAF-X) cho mesh-alignment feedback và adaptive wrist integration. Ý tưởng feedback hữu ích, nhưng dùng toàn model sẽ lại là expert replacement.

**SAM 3D Body.** [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_SAM_3D_Body_Robust_Full-Body_Human_Mesh_Recovery_CVPR_2026_paper.html), [official code](https://github.com/facebookresearch/sam-3d-body) và [MHR→SMPL-X converter](https://github.com/facebookresearch/MHR/tree/main/tools/mhr_smpl_conversion) có thể tạo expert đa dạng. Tuy nhiên output native là MHR, conversion thêm fitting error, license riêng và novelty chính vẫn là đổi model. Không chọn.

Kết luận domain: estimator scaling là control hữu ích nhưng không phải research contribution mong muốn.

### 5.3 Learned pose prior và inverse problems

**DPoser-X.** [ICCV 2025 Oral paper](https://arxiv.org/html/2508.00599v1), [official code](https://github.com/moonbow721/DPoser-X) và pretrained body/hand/whole-body checkpoints được phát hành dưới MIT license. Method xem pose task như inverse problem và đưa one-step diffusion denoiser vào objective test-time. Paper demo pose completion, inverse kinematics, motion denoising, body/hand/whole-body mesh recovery; repository hỗ trợ SMPL-X/MANO/FLAME. Đây là lựa chọn chính vì nó là **regularizer ghép vào solver**, không phải một image expert thay thế.

DPoser-X original dùng:

\[
x_t=\alpha_t x_0+\sigma_t\epsilon,
\qquad
\hat x_0(t)=\frac{x_t-\sigma_t\epsilon_\phi(x_t;t)}{\alpha_t},
\]

\[
L_{DPoser}=w_t\|x_0-\operatorname{sg}[\hat x_0(t)]\|_2^2.
\]

Paper còn chỉ ra pose refinement chủ yếu hữu ích ở timestep nhỏ và đề xuất schedule thường khoảng `[0,15; 0,05]`. Whole-body prior dùng frozen part models cùng fused module và mixed whole-body/part-only training.

**Neural Riemannian Motion Fields (NRMF).** [CVPR 2026 paper/project](https://circle-group.github.io/research/NRMF/) rất phù hợp về mặt hình học: distance fields cho pose, angular velocity và acceleration trên rotation manifold. Tuy nhiên [repository chính thức](https://github.com/ZhengdiYu/NRMF) tại thời điểm audit chưa có implementation/checkpoint/license sử dụng được. Chỉ giữ làm future comparison; không dựng lại từ paper trong V6 đầu tiên.

Kết luận domain: DPoser-X là module duy nhất vừa có cơ chế inverse-problem đúng với SIGNAL-4D, vừa có code/checkpoint/license đủ để tích hợp ngay.

### 5.4 Temporal HMR và motion refinement

**HTD-Refine.** [Official CVPR 2026 repository](https://github.com/ant-research/HTD-Refine) nhận initial SMPL/SMPL-X và dùng learned 2D keypoints, 3D velocity, acceleration cùng full-sequence optimization. Nó chứng minh high-order temporal dynamics có thể refine HMR. Tuy nhiên pipeline nhắm natural global motion, yêu cầu camera intrinsics/extrinsics, demo 30 FPS và code AGPL-3.0. SGNify là 15 FPS, clip ngắn và metric bỏ translation; vì vậy chỉ mượn cách ablate velocity/acceleration, không ghép nguyên pipeline.

**WHAM.** [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/html/Shin_WHAM_Reconstructing_World-grounded_Humans_with_Accurate_3D_Motion_CVPR_2024_paper.html) và [code](https://github.com/yohanshin/WHAM) tối ưu world-grounded motion/contact. Lợi ích global trajectory phần lớn không đi vào UBody(-F), và locomotion prior có thể làm mượt quá mức sign arms.

**GVHMR.** [Official repository](https://github.com/zju3dv/GVHMR) cũng tập trung world-grounded motion. Giữ làm related work, không tích hợp.

Kết luận domain: V5 đã có position/rotation acceleration factor. Cần learned pose plausibility theo vùng hơn là thay bằng global-motion pipeline.

### 5.5 Hand reconstruction và body–hand integration

**WiLoR.** [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.pdf) là hand expert hiện tại và được giữ nguyên.

**HaMeR.** [CVPR 2024 project/code](https://github.com/geopavlakos/hamer) là nguồn khởi tạo/fallback trong DexAvatar/A1. Nó không được thay trong kế hoạch chính.

**Hand4Whole.** [CVPRW 2022 paper](https://arxiv.org/abs/2011.11534) và [official code/checkpoint](https://github.com/mks0601/Hand4Whole_RELEASE) cho insight quan trọng: wrist rotation cần body context và các MCP roots, trong khi finger rotation không nên dùng coarse body feature.

**Hand4Whole++.** [CVPR 2026 paper](https://arxiv.org/html/2603.14726v1) và [MIT code/checkpoint](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE) giữ frozen whole-body/hand estimators, chỉ học Conditional Hands Modulator và dùng differentiable rigid alignment. Ablation của paper cho thấy copy wrist trực tiếp từ hand model làm full-body/hand error tăng mạnh; coupling có body context mới tốt. Đây là bằng chứng trực tiếp cho việc không “copy-paste” WiLoR wrist vào SMPL-X.

**PAD-Hand.** [CVPR 2026 Highlight paper](https://arxiv.org/html/2603.26068v1) và [official demo/checkpoint](https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026) dùng physics-aware conditional diffusion, Euler–Lagrange residual như virtual observations và last-layer Laplace variance theo joint/time. Repo chạy trực tiếp từ WiLoR nên phù hợp với source hiện tại. Hạn chế: repo còn nhỏ, cần Python 3.7/`torch-scatter`, demo mặc định chỉ xuất video và chưa thấy license file rõ tại thời điểm audit. Vì vậy nó là phase phụ, không là dependency của UBody path.

**Dyn-HaMR.** [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.pdf) và [code](https://github.com/ZhengdiYu/Dyn-HaMR) mạnh về tracking, handedness và dynamic-camera interacting hands. Nó nặng về camera/SLAM và MANO world motion, nên chỉ mượn kiểm tra handedness/track continuity.

**InterWild.** [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/html/Moon_Bringing_Inputs_to_Shared_Domains_for_3D_Interacting_Hands_Recovery_CVPR_2023_paper.html) và [code](https://github.com/facebookresearch/InterWild) hữu ích với two-hand interaction, nhưng không trực tiếp sửa torso/arms.

Kết luận domain: body–wrist–MCP coupling phải là factor rõ ràng trong solver; tay chi tiết vẫn do WiLoR/PAD-Hand xử lý riêng.

## 6. Source verification matrix

| Paper/module | Peer-reviewed | Code | Checkpoint | License/status | Quyết định |
|---|---|---|---|---|---|
| DPoser-X | ICCV 2025 Oral | official | body/hand/whole-body | MIT | **chọn làm core prior** |
| Hand4Whole++ | CVPR 2026 | official | có | MIT | dùng nguyên tắc coupling/alignment; không thay expert |
| Hand4Whole | CVPRW 2022 Oral | official | có | MIT | bằng chứng wrist–MCP |
| SignAvatars | ECCV 2024 | official | annotation theo request | non-commercial research conditions | dùng cho sign adapter nếu được cấp |
| PAD-Hand | CVPR 2026 Highlight | official demo | có | chưa thấy license rõ | optional, chờ license/parameter export audit |
| NRMF | CVPR 2026 | repo placeholder | chưa có | chưa rõ | defer |
| HTD-Refine | CVPR 2026 | official | pipeline có | AGPL-3.0, 30 FPS demo | không dùng core |
| SMPLest-X | TPAMI 2025/2026 issue | official | có | S-Lab non-commercial | external control, **không thay expert** |
| SAM 3D Body | CVPR 2026 | official | có điều kiện | custom SAM license | loại khỏi core |

Không có claim nào trong tài liệu này dựa trên một repository không kiểm chứng. “Checkpoint có” chỉ nghĩa nguồn chính thức cung cấp download, không bảo đảm asset license cho phép tái phân phối.

## 7. Novelty audit: tại sao không chọn expert replacement?

| Hướng | Khả năng tăng metric | Novelty với SIGNAL-4D | Tách contribution | Quyết định |
|---|---:|---:|---:|---|
| đổi SMPLer-X → SMPLest-X | có | rất thấp | khó, gain thuộc backbone | loại khỏi core |
| thêm AiOS/SAM3DB candidate | có thể | thấp | khó, thêm expert | loại khỏi core |
| thêm Sapiens/RTMW 2D evidence | có thể | thấp–trung bình | dễ | optional ablation |
| cắm DPoser-X nguyên bản | có cơ sở | trung bình | dễ | reproduction baseline D1 |
| **UQ/change-aware geodesic DPoser factor** | **có cơ sở trực tiếp** | **cao hơn** | **rất rõ qua ablation** | **core D2–D4** |
| **body–wrist–MCP SO(3) coupling** | **tác động UBody và tay** | **cao khi ghép với V5** | **rõ** | **core D5** |
| PAD-Hand parameter refinement | mạnh cho dynamics tay | trung bình | rõ | optional D7 |

Phần có novelty không phải checkpoint DPoser-X tự thân. Novelty dự kiến nằm ở cách biến diffusion prior thành một **selective, calibrated, change-aware, region-wise geodesic factor** và ghép nó với safe temporal gating cho sign reconstruction. Trước khi viết “first”, vẫn phải chạy một novelty search cập nhật ở thời điểm nộp paper.

## 8. Kiến trúc V6 được chọn

```mermaid
flowchart LR
    RGB[RGB + frozen manifest] --> V5E[V5 experts unchanged: SMPLer-X, WiLoR, DexAvatar A1]
    V5E --> C[V5 canonical cache + calibrated uncertainty + change probability]
    C --> H0[H0 immutable V5]
    C --> O[Open named upper-body joints on both sides]
    DP[DPoser-X frozen checkpoint] --> UQ[UQ/change-aware geodesic diffusion factor]
    C --> UQ
    W[WiLoR wrist + MCP observations] --> S[SO(3) wrist–palm seam factor]
    O --> F[Selective factor-graph optimization]
    UQ --> F
    S --> F
    F --> H1[H1..Hk V6 hypotheses]
    H0 --> G[UBody-first constrained temporal gate]
    H1 --> G
    G --> X[Exact SMPL-X output]
    X --> E[Author UBody(-F), L/R hand, dynamics]
```

Ba expert của V5 là frozen inputs. Không fine-tune, không đổi checkpoint và không sửa output cũ.

## 9. Module A — named-joint upper-body selective solver

V5 hiện mở raw indices 17/19 và left hand. V6 phải resolve theo tên từ canonical SMPL-X joint map:

```text
spine1, spine2, spine3, neck,
left_collar, right_collar,
left_shoulder, right_shoulder,
left_elbow, right_elbow,
left_wrist, right_wrist
```

Candidate nhỏ hơn được ablate trước:

- `arms_only`: shoulders, elbows, wrists;
- `arms_clavicle`: collars + shoulders + elbows + wrists;
- `full_ubody_chain`: spine1/2/3 + neck + collars + shoulders + elbows + wrists.

Luôn freeze:

```text
global orientation, translation, pelvis/lower body,
betas, face/jaw/eyes/expression
```

Finger pose vẫn giữ từ V5 trong core UBody stage. Chỉ phase hand mới mở finger rotations.

## 10. Module B — DPoser-X factor ghép vào SIGNAL-4D

### 10.1 Reproduction factor D1

Trước tiên tái tạo đúng DPoser-X regularizer trên cùng pose vector và frozen checkpoint:

\[
L_{DP}^{Euc}=w_{\tau}\|x_0-\operatorname{sg}(\hat x_0(\tau))\|_2^2.
\]

Chạy unit test trên example của official repo và một synthetic SMPL-X pose. Nếu không tái tạo được chiều gradient/denoising behavior, dừng integration.

### 10.2 Geodesic factor D2

DPoser-X xuất pose vector; SIGNAL-4D optimizer giữ rotation hợp lệ. Chuyển current và denoised pose về rotation matrices rồi dùng:

\[
L_{DP}^{SO(3)}=
\sum_{t,r}\sum_{j\in\mathcal J_r}
\lambda_{t,r,j}
\rho\left(
\left\|\log\left(\hat R_{t,j}^{T}R_{t,j}\right)\right\|_2
\right).
\]

`stop_gradient` được giữ ở \(\hat R\), đúng tinh thần one-step diffusion regularization. Pseudo-Huber \(\rho\) giảm ảnh hưởng denoised outlier. Module phải có test quanh góc \(\pi\), left/right reflection và batch/frame ordering.

### 10.3 Uncertainty-adaptive weight D3

Với spatial uncertainty đã conformal-calibrate \(u_{t,r}\), disagreement \(d_{t,r}\) và change probability \(p_t\):

\[
\tilde u_{t,r}=\operatorname{clip}
\left(\frac{u_{t,r}}{\operatorname{median}_{t}(u_{t,r})},0,3\right),
\]

\[
\lambda_{t,r}=lambda_0m_r
\left(1+\alpha_u\tilde u_{t,r}+\alpha_d\tilde d_{t,r}\right)
(1-p_t)^{\gamma_d}.
\]

Ý nghĩa:

- observation chắc chắn: prior yếu, giữ nghiệm expert;
- source bất đồng/bất định: prior mạnh hơn;
- pha change cao: giảm prior dù uncertainty lớn, tránh xóa articulation của sign;
- region mask chỉ tác động khớp upper-body đang mở.

Mọi hệ số được giới hạn `[lambda_min, lambda_max]`. Không học weight từ test GT.

### 10.4 Timestep schedule D4

Baseline dùng truncated schedule từ DPoser-X:

\[
\tau_i=\tau_{max}-
\frac{\tau_{max}-\tau_{min}}{N-1}i,
\quad [\tau_{max},\tau_{min}]=[0,15;0,05].
\]

Sau khi D1–D3 pass mới ablate adaptive interval:

- low uncertainty: `[0,10; 0,03]`;
- high uncertainty/disagreement: `[0,20; 0,05]`;
- high change: cap `tau_max <= 0,10`.

Adaptive timestep là hypothesis, không mặc định. Nếu nó không hơn weight-only D3 trên development CI, loại để giữ method đơn giản.

### 10.5 Tại sao module này có khả năng cải thiện thực tế?

1. DPoser-X đã chứng minh prior có thể ghép vào HMR/IK/pose completion bằng test-time optimization.
2. Metric mục tiêu nhạy với relative pose, đúng biến mà diffusion prior regularize.
3. V5 cung cấp initialization mạnh nên DPoser chỉ cần sửa cục bộ, không generate pose từ noise.
4. Uncertainty chọn đúng vùng cần sửa; change-point giảm rủi ro oversmooth.
5. V5 vẫn là candidate fallback, nên prior generic không bắt buộc được chọn.

Đây là cơ sở hợp lý, không phải bảo đảm kết quả. Chỉ evaluator mới quyết định.

## 11. Module C — body–wrist–MCP seam factor

### 11.1 Lý do

Hand4Whole/Hand4Whole++ chỉ ra wrist phải dùng cả body context và MCP cues; copy wrist từ hand-only model có thể làm full-body error tăng. SIGNAL-4D hiện có WiLoR MCP observations nhưng chưa có explicit seam factor.

### 11.2 Palm frame từ MCP

Với wrist \(W\), index MCP \(M_i\), middle MCP \(M_m\), ring MCP \(M_r\), pinky MCP \(M_p\):

\[
e_1=\operatorname{normalize}(M_m-W),
\]

\[
e_2'=\operatorname{normalize}(M_i-M_p),\quad
e_3=\operatorname{normalize}(e_1\times e_2'),\quad
e_2=e_3\times e_1.
\]

Palm frame \(P=[e_1,e_2,e_3]\) được tính cho prediction và WiLoR observation sau khi áp đúng left-hand mirror/camera convention.

### 11.3 Loss

\[
L_{palm}=
\sum_{t,h\in\{L,R\}}
w^{hand}_{t,h}
\rho\left(\|\log((P^{obs}_{t,h})^TP^{pred}_{t,h})\|_2\right),
\]

\[
L_{mcp-pos}=\sum_{t,h,k}
w_{t,h,k}
\rho\left(
\left\|
\frac{M^{pred}_k-W^{pred}}{s^{pred}}
-
\frac{M^{obs}_k-W^{obs}}{s^{obs}}
\right\|_2
\right).
\]

Scale normalization làm factor tập trung orientation/cấu trúc palm, không kéo global translation. Weight lấy từ calibrated WiLoR uncertainty, handedness ambiguity, crop size và validity.

### 11.4 Gradient routing

Core UBody experiment:

- gradient từ `L_palm/L_mcp-pos` chỉ đi vào collar/shoulder/elbow/wrist;
- finger rotation bị detach/freeze;
- không copy WiLoR global wrist rotation;
- không thay hand mesh bằng MANO mesh trong output official;
- output cuối vẫn forward từ một SMPL-X parameter vector duy nhất.

Phase hand sau đó mới cho gradient vào finger pose với weight nhỏ và non-inferiority gate.

## 12. Module D — sign-specific diffusion adapter

Generic DPoser-X có thể xem articulation sign nhanh là OOD. Sau khi frozen-prior D1–D5 có tín hiệu tích cực, mới huấn luyện một adapter nhỏ thay vì train DPoser-X từ đầu.

### 12.1 Dữ liệu

Xin quyền dùng SignAvatars annotations. Dùng:

- smoothed SMPL-X sequences cho motion prior;
- unsmoothed annotations cho pose/frame robustness ablation;
- split theo signer, không split ngẫu nhiên theo frame;
- không dùng SGNify final confirmatory GT để train.

### 12.2 Cấu trúc adapter

Hai lựa chọn, theo thứ tự ưu tiên:

1. LoRA/low-rank residual trên fused whole-body blocks của DPoser-X;
2. residual MLP nhỏ chỉ trên upper-body score, cộng vào frozen score.

Frozen DPoser-X là anchor:

\[
\epsilon_{sign}(x_t,t)=
\epsilon_{DP}(x_t,t)+g_{ubody}\Delta_{\psi}(x_t,t).
\]

Chỉ train \(\psi\); body/hand/face part networks frozen. Dùng masked training để adapter chịu được observation thiếu, theo tinh thần mixed training của DPoser-X.

### 12.3 Mục tiêu

- noise-prediction loss trên upper-body rotations;
- tangent-space angular velocity/acceleration consistency;
- tăng sampling của high hand-speed/change segments;
- left/right symmetry augmentation nhưng không đảo nhãn handedness sai;
- không dùng sign label/semantic label nếu không cần.

Adapter chỉ được giữ nếu hơn frozen DPoser-X trên signer-disjoint validation và official SGNify development folds.

## 13. Module E — optional PAD-Hand branch

Chỉ bắt đầu sau khi UBody core ổn định.

Pipeline:

```text
existing WiLoR MANO sequence
  -> PAD-Hand official checkpoint
  -> extract refined MANO rotations and per-joint/time physics variance
  -> left/right handedness validation
  -> SMPL-X hand rotation mapping
  -> seam factor with frozen upper-body result
  -> candidate, never unconditional overwrite
```

Các blocker phải giải quyết trước:

1. xác nhận license bằng văn bản/file LICENSE;
2. sửa demo để xuất parameters, không reverse-engineer từ rendered video;
3. xác minh MANO order, units, handedness và frame rate;
4. tách Python 3.7 environment khỏi `signal4d`;
5. calibration physics variance trên historical split;
6. kiểm tra PAD-Hand không làm official UBody(-F) xấu vì hand vertices vẫn thuộc vùng upper-body-minus-face.

Nếu bất kỳ blocker nào không pass, giữ V5/WiLoR và bỏ PAD-Hand khỏi paper.

## 14. UBody-first constrained temporal gate

Gate V5 lịch sử học target left-hand. V6 phải đổi primary target thành:

\[
\Delta^{UB-F}_{c}=E^{UB-F}_{c}-E^{UB-F}_{V5}.
\]

Secondary targets:

```text
delta_left_hand, delta_right_hand,
delta_all, delta_velocity, delta_acceleration, delta_jerk
```

### 14.1 Safe selection rule

Với candidate \(c\), dự đoán mean và uncertainty/quantile của delta. Candidate chỉ hợp lệ khi:

\[
UCB_{95}(\Delta^{UB-F}_c)<0,
\]

\[
UCB_{95}(\Delta^{LH}_c)\le \delta_H,
\qquad
UCB_{95}(\Delta^{RH}_c)\le \delta_H.
\]

Nếu không candidate nào pass, chọn immutable V5. `delta_H` đề xuất preregister là `+0,25 mm`; phải cố định trước confirmatory reveal.

### 14.2 Temporal decoding

Viterbi/semi-Markov objective:

\[
\min_{z_{1:T}}
\sum_t C_{t,z_t}
+\lambda_{sw}\sum_t\mathbf1[z_t\ne z_{t-1}]
+\lambda_{short}\sum_s\mathbf1[\operatorname{len}(s)<L_{min}].
\]

State V5 luôn tồn tại. Gate được học từ historical GT nhưng không được đọc GT lúc inference; wording vẫn là **GT-free at inference**.

## 15. End-to-end work packages

### WP0 — freeze và isolation

- Hash toàn V5 source/config/artifacts/outputs.
- Cấm write vào `signal4d/runs/signal4d_v5_full1493_20260820` và legacy outputs.
- Tạo code mới trong `signal4d/extensions/v6_uqdiff/`.
- Tạo environment riêng `signal4d-dposer` cho DPoser-X.
- Vendor/clone third-party ở `signal4d/third_party/DPoser-X` hoặc đường external read-only, ghi commit hash/license.
- Mọi config V6 có `parent_release_hash` trỏ V5 freeze.

Pass: rerun V5 hash và byte-exact output không đổi.

### WP1 — DPoser-X reproduction

- Download checkpoint bằng official downloader.
- Lưu URL, SHA-256, license và exact commit.
- Chạy official generation/completion/whole-body demo.
- Viết adapter `pose_vector <-> canonical named SMPL-X rotations`.
- Unit test round-trip axis-angle/matrix/6D.

Pass: official example chạy, round-trip geodesic error dưới tolerance, deterministic fixed-seed denoiser.

### WP2 — original DPoser factor D1

- Implement one-step denoiser với stop-gradient.
- Reproduce direction gradient theo equation paper.
- Thêm config `dposer.enabled`, checkpoint hash, region mask, schedule, weight.
- Chạy synthetic corrupted-pose recovery.

Pass: corrupted joint error giảm mà valid rotation/topology không hỏng.

### WP3 — geodesic factor D2

- Chuyển denoised pose về SO(3).
- Implement log-map pseudo-Huber loss.
- Test near-identity, near-π, left/right batches và gradient finite.
- So sánh raw Euclidean vs geodesic trên grouped development folds.

Pass: D2 không kém D1 trên primary delta và giảm rotation artifacts.

### WP4 — uncertainty/change adaptation D3–D4

- Calibrate spatial uncertainty như V5, đồng thời thêm angular residual calibration riêng.
- Build region aggregates cho spine, left/right arm, left/right wrist/hand.
- Implement bounded lambda schedule.
- Ablate uncertainty-only, change-only, both, adaptive timestep.

Pass: `both` hơn original DPoser factor và không làm dynamics vượt safety threshold.

### WP5 — body–wrist–MCP seam D5

- Implement palm-frame construction với degeneracy checks.
- Dùng four-MCP/wrist valid mask và WiLoR calibrated weight.
- Route gradient chỉ vào named upper-body joints.
- Test mirror-equivariance cho tay trái.
- Visualize wrist axes/palm frames cùng overlay.

Pass: UBody(-F) hoặc hand metrics cải thiện; không có wrist flip/outlier clip.

### WP6 — candidate generation và gate

- Materialize D1–D5 hypotheses song song, không overwrite.
- Train grouped out-of-fold multi-output gate trên historical folds.
- Fit prediction intervals/conformal residual cho delta.
- Decode temporal path với V5 fallback.
- Kiểm tra inference graph không import GT/evaluator package.

Pass: label-blind replay, 100% coverage, byte-exact repeated gate.

### WP7 — sign adapter D6

- Chỉ chạy nếu D5 đã có tín hiệu UBody(-F).
- Xin/kiểm tra SignAvatars license và split metadata.
- Train adapter trong tmux với checkpoint/log/config/hash.
- Validate signer-disjoint.
- Không fine-tune base experts hay full DPoser-X.

Pass: D6 hơn frozen D5 trên signer-disjoint validation và SGNify development OOF.

### WP8 — PAD-Hand D7

- Chỉ chạy sau license audit.
- Export MANO params/variance.
- Map sang SMPL-X và calibrate variance.
- Chạy left/right/seam ablations.

Pass: ít nhất một hand cải thiện có CI và hand còn lại non-inferior; UBody(-F) không regress.

### WP9 — strict export/evaluation/render

- Export direct SMPL-X params/OBJ.
- Validate 10.475 vertices/20.908 faces và exact author faces.
- Same manifest/same GT mapping cho V5 và candidate.
- Chạy structured author evaluator và original author `main()` replay.
- Render toàn frame theo format fitting reconstruction hiện có.

Pass: 100% frame/sign coverage, structured/original numbers khớp precision in ra.

## 16. Experiment matrix bắt buộc

| ID | Experts | Open joints | DPoser | UQ | Change | MCP seam | Sign adapter | Mục đích |
|---|---|---|---|---|---|---|---|---|
| H0 | frozen V5 | V5 | no | V5 | V5 | no | no | immutable comparator |
| D0 | same | bilateral upper body | no | no | V5 | no | no | effect của mở khớp |
| D1 | same | bilateral upper body | original Euclidean | no | no | no | no | reproduce paper graft |
| D2 | same | bilateral upper body | geodesic | no | no | no | no | SO(3) contribution |
| D3 | same | bilateral upper body | geodesic | yes | no | no | no | uncertainty contribution |
| D4 | same | bilateral upper body | geodesic | yes | yes | no | no | change-aware contribution |
| D5 | same | bilateral upper body | geodesic | yes | yes | yes | no | core V6 |
| D6 | same | bilateral upper body | geodesic | yes | yes | yes | yes | sign-domain extension |
| D7-L/R | same | + selected fingers | D6/core | yes | yes | yes | optional | PAD-Hand effect |

Mỗi row phải báo:

- official vertex-micro UBody(-F);
- clip-macro UBody(-F);
- LHand/RHand/all;
- per-clip paired delta distribution;
- velocity/acceleration/jerk;
- coverage, runtime, peak VRAM;
- failure/outlier count;
- candidate/state selection histogram.

Không được chỉ báo row tốt nhất.

## 17. Data split và chống leakage

Toàn bộ 1.493 frame hiện đã được xem qua khi phát triển/audit V5. Vì vậy:

- có thể dùng grouped/nested leave-sign-out cross-validation để chọn module và hyperparameter;
- không thể gọi một subset được tách lại từ 1.493 frame là “unseen confirmatory” theo nghĩa nghiêm ngặt;
- kết quả full-1.493 V6 phải ghi là development/diagnostic nếu không có dữ liệu mới;
- để claim improvement xác nhận, cần freeze code/config/gate rồi mới thu/nhận GT của sign/clip mới hoặc một evaluation server kín;
- external SignAvatars signer-disjoint validation chứng minh prior không collapse nhưng không thay thế official SGNify confirmatory protocol.

Split development đề xuất:

1. outer fold group theo sign/clip;
2. inner fold chọn lambda/timestep/joint set;
3. gate OOF prediction chỉ từ model không train trên clip đó;
4. bootstrap unit là clip/sign, không phải frame;
5. nếu có signer ID mới, group theo signer ở cấp cao nhất.

## 18. Statistics và acceptance gates

Primary estimator vẫn là author vertex-micro. Để tránh một sign dài chi phối kết luận, báo thêm clip-macro và paired clip bootstrap 10.000 lần.

Trước khi mở confirmatory GT, preregister:

1. coverage 100%, no topology mismatch, no NaN;
2. primary mean delta UBody(-F) `<= -0,15 mm` và upper bound paired 95% CI `< 0`;
3. LHand và RHand non-inferiority: upper 95% CI `<= +0,25 mm` mỗi tay;
4. TR all không regress quá `+0,10 mm`;
5. velocity/acceleration/jerk không xấu quá 2% tương đối;
6. không clip nào có catastrophic UBody(-F) regression trên ngưỡng được freeze;
7. gate replay byte-exact và `gt_used_for_selection=false`;
8. original author `main()` và structured evaluator khớp số in ra.

Các ngưỡng `0,15/0,25/0,10` là đề xuất để preregister, không phải kết quả đã đạt. Nếu thay phải thay trước confirmatory freeze và ghi lý do.

## 19. Hyperparameter search có kiểm soát

Không grid-search hàng trăm cấu hình trên cùng GT. Dùng coarse-to-fine với tối đa:

```text
joint set: {arms_only, arms_clavicle, full_ubody_chain}
lambda_0: {0.001, 0.003, 0.01}
alpha_u: {0, 1, 2}
gamma_d: {1, 2}
timestep: {original, adaptive}
seam weight: {0, 0.1, 0.3, 1.0}
```

Successive halving theo inner folds; primary ranking UBody(-F), constraints hai tay/dynamics. Không dùng full-1.493 aggregate để chọn rồi lại báo cùng aggregate như confirmatory.

## 20. Folder và artifact contract

Đề xuất layout mới, không đụng code cũ:

```text
signal4d/extensions/v6_uqdiff/
  adapters/dposer_x.py
  factors/diffusion_prior_so3.py
  factors/wrist_mcp_seam.py
  models/region_risk.py
  optimization/ubody_solver.py
  gating/ubody_safe_gate.py
  evaluation/experiment_registry.py
  tests/

signal4d/configs/v6_uqdiff/
  d1_original.yaml
  d2_geodesic.yaml
  d3_uncertainty.yaml
  d4_change.yaml
  d5_core.yaml
  d6_sign_adapter.yaml

signal4d/artifacts/v6_uqdiff/
  third_party_registry.json
  calibration/
  gates/
  releases/

signal4d/runs/v6_uqdiff_<timestamp>/
signal4d/reports/v6_uqdiff_<timestamp>/
signal4d/outputs/reconstruction_v6_uqdiff_<timestamp>/
```

Mỗi run lưu:

- git commit/dirty status;
- config resolved;
- seed và environment lock;
- GPU/driver/runtime/peak memory;
- input/model/checkpoint hashes;
- frame manifest hash;
- per-iteration factor logs;
- per-frame uncertainty/change/lambda/timestep;
- selected hypothesis;
- exact output hashes;
- evaluator command/log/raw JSON.

Training adapter nếu cần chạy trong tmux và lưu stdout/stderr, TensorBoard, last/best checkpoint, scheduler state và resume command.

## 21. Rủi ro, mitigation và abort criteria

### R1 — generic prior làm mất sign articulation

Mitigation: change-aware suppression, region mask, low timestep, V5 fallback, SignAvatars adapter.

Abort: D1/D2 làm hand speed/acceleration xấu hoặc UBody(-F) regress trên đa số outer folds.

### R2 — DPoser representation không khớp SMPL-X convention

Mitigation: named-joint map, rotation round-trip tests, exact left/right convention, official examples.

Abort: không đạt rotation round-trip hoặc denoised gradient không ổn định quanh π.

### R3 — prior kéo pose khỏi image evidence

Mitigation: uncertainty-bounded lambda, strong observation factors, stop-gradient denoiser, line search/early stopping.

Abort: 2D/3D observation residual tăng quá threshold hoặc xuất hiện catastrophic clip.

### R4 — seam factor gây wrist flip

Mitigation: palm-frame degeneracy mask, temporal sign consistency của axes, geodesic robust loss, handedness tests.

Abort: wrist angular jump/flip count cao hơn V5 hoặc tay metric regress.

### R5 — gate overfit historical GT

Mitigation: grouped OOF, prediction interval, V5 abstention, prospective freeze.

Abort: OOF gain không lặp qua outer folds hoặc gate chọn candidate với calibration coverage thấp.

### R6 — SignAvatars domain/license

Mitigation: dùng annotations theo research agreement, signer-disjoint, không tái phân phối data.

Abort: không được quyền hoặc không có signer split đáng tin; giữ frozen DPoser core.

### R7 — PAD-Hand code/license chưa trưởng thành

Mitigation: isolated env, parameter-export audit, hỏi tác giả/license.

Abort: không có license rõ hoặc không xuất được MANO parameters; không đưa vào release/paper claim.

## 22. Compute plan

- Không train SMPLest-X/DPoser-X từ đầu.
- DPoser-X frozen inference/optimization chạy trong environment riêng; GPU 49 GB hiện có đủ dư địa hợp lý nhưng phải đo peak VRAM thực tế.
- Bắt đầu 3–5 clip representative: slow sign, fast transition, occluded left/right hand, two-hand interaction.
- Sau unit/smoke pass, chạy inner folds; chỉ sau đó full diagnostic.
- Sign adapter là training duy nhất dự kiến; dùng mixed precision nếu stable và tmux checkpoint resume.
- PAD-Hand dùng Python 3.7 env riêng, không hạ dependency của `signal4d`.

Không đưa ETA/GPU-hour giả trước khi benchmark throughput thực tế.

## 23. Paper contribution dự kiến nếu D5/D6 pass

Wording an toàn:

1. **Uncertainty- and change-aware diffusion pose-prior refinement.** Một diffusion whole-body prior được ghép vào selective SMPL-X optimizer, với calibrated region risk điều tiết độ mạnh và sign change-point bảo vệ chuyển động nhanh.
2. **Geodesic cross-part body–wrist–MCP coupling.** Một factor SO(3) kết nối torso/arm context với palm/MCP evidence mà không copy hand-expert wrist rotation hoặc làm nhiễu finger articulation.
3. **UBody-first risk-controlled temporal composition.** Một gate GT-free at inference chọn giữa immutable V5 và diffusion hypotheses theo official UBody(-F), đồng thời áp ràng buộc non-inferiority cho hai tay và dynamics.
4. **Strict protocol-compatible sign mesh reconstruction.** Direct SMPL-X output, full coverage và author-evaluator replay có provenance.

Không được viết:

- “thay bằng expert mạnh hơn” như contribution;
- “SOTA” nếu chưa có benchmark/test độc lập;
- “unsupervised gate”;
- “physics-aware” nếu PAD-Hand không được tích hợp/evaluate;
- “rotation uncertainty calibrated” trước khi angular conformal calibration pass;
- “semantic preservation” khi chưa có semantic evaluator.

## 24. Thứ tự triển khai khuyến nghị

```text
P0  Freeze V5 and reproduce DPoser-X
P1  D1 original DPoser regularizer
P2  D2 geodesic SO(3) factor
P3  D3 calibrated uncertainty weighting
P4  D4 change-aware weighting
P5  D5 wrist–MCP seam + UBody-first safe gate
P6  strict OOF evaluation and visualization
P7  only if D5 works: SignAvatars adapter D6
P8  only if UBody is safe: PAD-Hand D7
P9  prospective freeze and final author-protocol evaluation
```

Go/no-go sau mỗi phase. Không nhảy thẳng sang D6/PAD-Hand vì sẽ không biết gain đến từ đâu.

## 25. Kết luận

Kế hoạch có novelty hợp lý nhất là giữ nguyên toàn bộ expert của V5 và thay đổi cơ chế refinement: đưa DPoser-X vào như một learned diffusion factor, nhưng biến nó thành region-calibrated, change-aware và geodesic cho sign-language upper body. Body–wrist–MCP seam giải quyết đúng điểm nối giữa body expert và hand expert mà literature cho thấy không thể xử lý bằng copy pose ngây thơ.

SMPLest-X không còn là đường chính. Nó chỉ là control ngoài nếu cần. Release mới chỉ được công nhận khi D5/D6 vượt immutable V5 trên official UBody(-F) với paired CI, không làm xấu hai tay/dynamics và giữ đủ 100% frame/topology.

## 26. Tài liệu tham khảo chính

1. [DPoser-X: Diffusion Model as Robust 3D Whole-body Human Pose Prior, ICCV 2025](https://arxiv.org/abs/2508.00599); [official code](https://github.com/moonbow721/DPoser-X).
2. [Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator, CVPR 2026](https://arxiv.org/abs/2603.14726); [official code](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE).
3. [Hand4Whole, CVPRW 2022](https://arxiv.org/abs/2011.11534); [official code](https://github.com/mks0601/Hand4Whole_RELEASE).
4. [PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery, CVPR 2026](https://arxiv.org/abs/2603.26068); [official code](https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026).
5. [SignAvatars, ECCV 2024](https://signavatars.github.io/); [official code](https://github.com/ZhengdiYu/SignAvatars).
6. [DexAvatar, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.pdf); [official code](https://github.com/kaustesseract/DexAvatar).
7. [SGNify project](https://sgnify.is.tue.mpg.de/).
8. [SMPLer-X, NeurIPS 2023](https://arxiv.org/abs/2309.17448); [official code](https://github.com/MotrixLab/SMPLer-X).
9. [SMPLest-X, TPAMI](https://arxiv.org/abs/2501.09782); [official code](https://github.com/MotrixLab/SMPLest-X).
10. [AiOS, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Sun_AiOS_All-in-One-Stage_Expressive_Human_Pose_and_Shape_Estimation_CVPR_2024_paper.html); [official code](https://github.com/MotrixLab/AiOS).
11. [OSX/UBody, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_One-Stage_3D_Whole-Body_Mesh_Recovery_With_Component_Aware_Transformer_CVPR_2023_paper.pdf); [official code](https://github.com/IDEA-Research/OSX).
12. [PyMAF-X, TPAMI 2023](https://arxiv.org/abs/2207.06400); [official code](https://github.com/HongwenZhang/PyMAF-X).
13. [WiLoR, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.pdf); [official code](https://github.com/rolpotamias/WiLoR-mini).
14. [HaMeR, CVPR 2024](https://geopavlakos.github.io/hamer/); [official code](https://github.com/geopavlakos/hamer).
15. [Dyn-HaMR, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.pdf); [official code](https://github.com/ZhengdiYu/Dyn-HaMR).
16. [InterWild, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Moon_Bringing_Inputs_to_Shared_Domains_for_3D_Interacting_Hands_Recovery_CVPR_2023_paper.html); [official code](https://github.com/facebookresearch/InterWild).
17. [HTD-Refine, CVPR 2026 official code/project](https://github.com/ant-research/HTD-Refine).
18. [WHAM, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Shin_WHAM_Reconstructing_World-grounded_Humans_with_Accurate_3D_Motion_CVPR_2024_paper.html); [official code](https://github.com/yohanshin/WHAM).
19. [Neural Riemannian Motion Fields, CVPR 2026](https://circle-group.github.io/research/NRMF/); [repository status](https://github.com/ZhengdiYu/NRMF).
20. [SAM 3D Body, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_SAM_3D_Body_Robust_Full-Body_Human_Mesh_Recovery_CVPR_2026_paper.html); [official code](https://github.com/facebookresearch/sam-3d-body).

## 27. Research integrity và AI disclosure

Tài liệu được AI hỗ trợ tìm kiếm, sàng lọc và tổng hợp vào ngày 2026-08-21. Các claim về paper/code/checkpoint/license đều được đối chiếu với paper, project page hoặc repository chính chủ được liên kết ở trên. Các kết quả V5 được truy vết từ artifact/evaluator cục bộ; tài liệu này **không tuyên bố bất kỳ kết quả empirical V6 nào**. Mọi novelty/SOTA claim tương lai phải qua literature search cập nhật, ablation đầy đủ và confirmatory evaluation độc lập.
