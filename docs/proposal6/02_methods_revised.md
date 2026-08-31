# Methods viết lại — Dynamic-contact-guided monocular 4D sign reconstruction

> **Working name:** CoSign4D. `[AUTHOR DECISION REQUIRED: cân nhắc đổi tên vì CoSIGN đã tồn tại trong diffusion inverse problems.]`  
> **Trạng thái:** bản Methods có cấu trúc toán học nhất quán để triển khai và viết paper; mọi giá trị chưa có bằng chứng được giữ dưới dạng `[AUTHOR INPUT REQUIRED]` thay vì được suy đoán.

## 1. Problem formulation và ranh giới claim

Cho một clip RGB đơn nhãn quan

\[
Y_{1:T}=\{Y_t\}_{t=1}^{T},
\]

mục tiêu là phục hồi một trajectory SMPL-X có tính toàn cục

\[
X_{1:T}=\{X_t\}_{t=1}^{T},
\]

đồng thời suy luận một chuỗi contact graph động

\[
C_{1:T}=\{C_t\}_{t=1}^{T}.
\]

Mỗi trạng thái hình học được biểu diễn bởi

\[
X_t=(R_t^{root},\,p_t^{root},\,\theta_t^{body},\,
\theta_t^{lh},\,\theta_t^{rh},\,[\theta_t^{face}]),
\]

với body shape `beta` dùng chung cho toàn clip và camera `Pi_t`. Thành phần face đặt trong ngoặc vì chỉ được tối ưu/đánh giá nếu dữ liệu cung cấp supervision tương ứng. Network dùng rotation representation liên tục—ví dụ 6D—và chuyển về rotation matrix trước forward kinematics.

**Đầu ra chính** là `K` trajectory hypotheses cùng contact-event sequences và một ranking score không dùng ground truth. Phương pháp không được gọi là “calibrated posterior” cho tới khi vượt qua protocol calibration tại §10.6.

`[AUTHOR INPUT REQUIRED]` Chốt loại camera, cách xử lý camera motion, frame rate chuẩn hóa, window length, overlap và những SMPL-X degrees of freedom thực sự được tối ưu.

## 2. Observation set và reliability/visibility

Từ video, hệ thống trích tập quan sát

\[
O=\{O^{kp},O^{mask},O^{track},O^{depth},[O^{flow}]\},
\]

gồm body/hand keypoints, silhouettes hoặc part masks, temporal tracks, relative depth ordering và optical-flow cues nếu có. Mỗi cue đi kèm reliability

\[
M=\{M^{(k)}_{t,j}\in[0,1]\},
\]

được định nghĩa ở cấp frame–joint/part thay vì một binary mask toàn ảnh.

### 2.1 Reliability calibration

Detector confidence không được mặc định là visibility probability. Trên validation subset có visibility/occlusion labels, fit một calibration map

\[
g_k:s^{raw}_{t,j}\mapsto M^{(k)}_{t,j}
\]

bằng temperature scaling, isotonic regression hoặc một phương pháp được khóa trước. Báo expected calibration error và reliability diagram trên validation set; test set chỉ được dùng một lần cho kết quả cuối.

Đối với keypoint cue, một lựa chọn có thể audit là robust likelihood với variance phụ thuộc reliability:

\[
-\log p_{kp}(O^{kp}\mid X,M)=
\sum_{t,j}\rho_{Huber}\!\left(
\frac{\|\Pi_t(J_j(X_t))-u_{t,j}\|_2}
{\sigma_{min}+(1-M^{kp}_{t,j})\sigma_{occ}}
\right).
\]

Khi cue hoàn toàn thiếu, hạng tương ứng bị mask; không gán tọa độ giả. Silhouette và depth-order cues cũng phải dùng cùng nguyên tắc: reliability làm tăng uncertainty hoặc giảm influence, không thay đổi target geometry.

`[AUTHOR INPUT REQUIRED]` Nguồn từng cue, calibration subset, reliability target, loss cụ thể và threshold dùng để định nghĩa các visibility bins.

## 3. Dynamic contact graph

### 3.1 Node và edge ontology

Đặt `V` là tập surface patches cố định trên SMPL-X, tối thiểu bao gồm:

- left/right fingertip groups và palm;
- head/face regions có ý nghĩa đối với sign;
- upper-torso và arm patches;
- các patch bổ sung chỉ khi có đủ dữ liệu để gán nhãn ổn định.

Tập cạnh hợp lệ `E_adm` loại bỏ các cặp kề nhau về giải phẫu hoặc các cặp không thể phân biệt tin cậy ở độ phân giải dữ liệu. Tại frame `t`, mỗi cạnh `e=(a,b)` có trạng thái sự kiện

\[
z_{e,t}\in\{\texttt{off},\texttt{onset},\texttt{hold},\texttt{release}\}.
\]

Contact graph là `C_t=(V,{z_e,t}_{e in E_adm})`. Mô hình không dùng một bit “contact/no-contact” duy nhất vì bit đó không biểu diễn duration và dễ flicker quanh threshold.

### 3.2 Geometric observables

Với mỗi cạnh, tính từ mesh:

- khoảng cách patch `d_e(X_t)`;
- normal compatibility `n_a^T n_b`, với hai mặt tiếp xúc kỳ vọng có normal đối nhau;
- relative patch velocity `v_e(X_t)`;
- signed penetration depth `p_e(X_t)`.

Một contact-compatibility factor tường minh là

\[
\psi_{geo}(X,C)=\exp[-E_{geo}(X,C)],
\]

trong đó

\[
E_{geo}=E_{positive}+E_{negative}
+\lambda_{pen}E_{penetration}(X),
\]

\[
E_{positive}=\sum_{t,e}w^+_{e,t}\left[
\rho(d_e/\sigma_d)
+\lambda_n\rho((n_a^Tn_b+1)/\sigma_n)
+I_{hold}(z_{e,t})\lambda_v\rho(\|v_e\|/\sigma_v)
\right],
\]

\[
E_{negative}=\sum_{t,e}w^-_{e,t}
\rho([\delta_{sep}-d_e]_+/\sigma_d).
\]

Hàm chỉ báo `I_hold(z)` bằng 1 khi `z=hold` và bằng 0 ở các trạng thái khác.

`w+` và `w-` bằng 0 đối với nhãn uncertain. Negative separation chỉ áp dụng cho annotated/hard-negative edges; không ép mọi cặp `off` phải cách xa nhau. Khoảng cách, normal và velocity phải được tính bằng cùng mesh scale và frame rate.

### 3.3 Semi-Markov event prior

Mỗi edge dùng một duration-aware transition model

\[
p_\rho(C)=\prod_e p_\rho(z_{e,1})
\prod_{t=2}^{T}p_\rho(z_{e,t}\mid z_{e,t-1},d_{e,t-1}),
\]

với transition hợp lệ như `off→onset→hold→release→off` và duration `d`. Có thể hiện thực bằng semi-Markov CRF hoặc explicit-duration HMM. Hysteresis chỉ dùng để tạo pseudo-label ban đầu; nó không thay thế event model trong claim cuối.

`[AUTHOR INPUT REQUIRED]` Mesh patch map, admissible-edge list, contact/near-contact thresholds, duration rule và tolerance theo frame.

## 4. Target distribution và vai trò của inference network

Để tránh double-counting bằng chứng ảnh, tách **target distribution** khỏi **amortized proposal**.

### 4.1 Unnormalized target

Mô hình tối ưu/sampling từ

\[
\boxed{
\pi(X,C\mid O,M)\propto
p_\phi(X\mid C,M)\;p_\rho(C)\;\psi_{geo}(X,C)
\prod_k p_k(O^{(k)}\mid X,M^{(k)})
}
\]

trong đó:

- `p_phi(X|C,M)` là holistic trajectory diffusion prior, condition trên contact-event sequence và missingness/reliability pattern, không trực tiếp nhân lại raw detector predictions;
- `p_rho(C)` là temporal event prior;
- `psi_geo(X,C)` kiểm tra graph có được hiện thực hóa về hình học hay không;
- `p_k` là các observation likelihood đã định nghĩa và hiệu chỉnh.

Đây là một target chưa chuẩn hóa; paper không báo exact NLL nếu không có estimator hợp lệ cho normalizing constant.

### 4.2 Contact proposal

Mạng

\[
q_\eta(C\mid O,X,M)
\]

được dùng để khởi tạo và cập nhật graph trong approximate inference. Nó **không** được nhân thêm vào target như một generative factor. Việc phân vai này giữ cho ảnh không bị tính hai lần và giải thích vì sao contact update phụ thuộc cả evidence lẫn geometry hiện tại.

## 5. Graph-conditioned holistic trajectory diffusion

### 5.1 Trajectory representation

Mỗi window ghép root-relative translation/velocity, body rotations, left/right hand rotations và các state được chọn thành tensor `x_0`. Tất cả channel được normalize bằng training statistics; shape và camera có nhánh riêng hoặc được tối ưu ngoài diffusion state.

Graph event tokens được mã hóa theo edge identity, event state, duration và time. Part-aware cross-attention cho phép hand, body và face patches trao đổi thông tin, nhưng architecture phải được matched về capacity trong ablation no-graph/static-graph.

### 5.2 Denoising objective

Với diffusion time `tau`,

\[
x_\tau=\sqrt{\bar\alpha_\tau}x_0+
\sqrt{1-\bar\alpha_\tau}\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

Mạng dự đoán noise hoặc velocity theo một lựa chọn cố định. Với noise prediction:

\[
\mathcal L_{diff}=
\mathbb E_{x_0,C,M,\tau,\epsilon}
\left[w(\tau)\|\epsilon-epsilon_\phi(x_\tau,\tau,C,M)\|_2^2\right].
\]

Mask dropout được dùng trong training để mô phỏng missing cues, nhưng robustness và calibration vẫn phải đánh giá riêng. Không gọi `L_diff` là một energy tại test time trừ khi paper định nghĩa rõ surrogate và gradient của nó.

### 5.3 Contact proposal loss

Trên samples có graph labels,

\[
\mathcal L_{graph}=
\mathcal L_{class-balanced-event}
+\lambda_{dur}\mathcal L_{duration}
+\lambda_{trans}\mathcal L_{invalid-transition}.
\]

Class-balanced focal loss hoặc logit adjustment có thể dùng do contact rất thưa. Kết quả phải so với weighted cross-entropy trên validation set; không chọn loss chỉ vì phổ biến.

### 5.4 Total training objective

\[
\mathcal L=
\mathcal L_{diff}
+\lambda_g\mathcal L_{graph}
+\lambda_{geom}\mathcal L_{geometry-consistency}
+[\lambda_{obs}\mathcal L_{observation}],
\]

với mask theo dataset để loss chỉ được áp dụng khi supervision tồn tại. Mọi `lambda` được chọn trên validation set và công bố đầy đủ; không tune trên test.

## 6. Data và contact-label protocol

### 6.1 Data roles

Ba loại dữ liệu có vai trò khác nhau:

1. **Sign reconstruction data:** học sign-specific temporal dynamics và đánh giá task chính.
2. **Self-contact data:** ví dụ Goliath-SC/TUCH-style assets nếu license cho phép; học geometry/contact plausibility.
3. **Interaction data:** bimanual và hand–face datasets để tăng coverage của edge types; chỉ dùng khi topology và coordinate mapping được kiểm chứng.

Không trộn dataset bằng cách giả định tất cả có cùng label. Mỗi sample có supervision mask cho pose, trajectory, graph, visibility và semantics.

### 6.2 Gold contact subset

Một subset của sign data phải được double-annotated theo protocol:

- annotator xem synchronized RGB và, nếu có, multi-view/3D reference;
- gán patch pair, onset, hold interval, release và uncertain flag;
- hai annotator làm độc lập; disagreement được adjudicate;
- báo agreement theo edge class và transition time;
- threshold hình học chỉ hỗ trợ annotation, không tự động thay thế human decision.

Pseudo-labels từ fitted SMPL-X chỉ được dùng sau khi ước lượng precision/recall trên gold subset. Samples gần threshold được gắn uncertain, không ép thành hard labels.

### 6.3 Split và leakage control

Primary split phải độc lập theo signer; nếu mục tiêu còn bao gồm unseen signs thì thêm split độc lập theo gloss/sign identity. Pretraining data và evaluation sequence được kiểm tra near-duplicate ở cấp clip/identity. Công bố manifest và hash.

`[AUTHOR INPUT REQUIRED]` Dataset names/versions, licenses, số signer/sign/clip, split manifest, annotation sample size và label agreement target.

## 7. Approximate alternating inference

### 7.1 Initialization

1. Trích `O,M` từ clip.
2. Khởi tạo `X^(0)` bằng một reconstruction system cố định, ví dụ pipeline DexAvatar hoặc estimator tương đương.
3. Tính `q_eta(C|O,X^(0),M)` rồi semi-Markov decode để có `C^(0)`.

Baseline và proposed method phải dùng cùng initialization để tránh attribution sai.

### 7.2 Posterior-score guidance cho trajectory

Tại diffusion step `tau`, score có guidance là

\[
\hat s(x_\tau,\tau)=
s_\phi(x_\tau,\tau\mid C,M)
+s_{obs}(x_\tau,\tau)+s_{contact}(x_\tau,\tau),
\]

với observation guidance

\[
s_{obs}(x_\tau,\tau)=\sum_k\lambda_k(\tau)
\nabla_{x_\tau}\log p_k
(O^{(k)}\mid D_\tau(x_\tau),M^{(k)}),
\]

và contact guidance

\[
s_{contact}(x_\tau,\tau)=\lambda_c(\tau)
\nabla_{x_\tau}\log\psi_{geo}(D_\tau(x_\tau),C).
\]

trong đó `D_tau` là clean-state estimate của sampler. Guidance schedules được khóa trên validation set. Công thức này dùng diffusion đúng vai trò score prior; không cần một `E_diff` giả định.

`[AUTHOR INPUT REQUIRED]` Sampler, number of reverse steps, guidance schedule, clean-state parameterization và gradient stabilization.

### 7.3 Graph update

Sau mỗi trajectory round, cập nhật riêng cho từng hypothesis:

\[
C^{(r+1,k)}=
\operatorname{SemiMarkovDecode}
\left(q_\eta(C\mid O,X^{(r,k)},M),p_\rho(C)\right).
\]

Không collapse các hypothesis về một graph trung bình nếu việc đó phá đa mode. Lặp `R` rounds cố định hoặc dừng theo criterion được công bố trước. Gọi quy trình này là **approximate alternating inference**, không tuyên bố exact posterior sampling.

### 7.4 Hypothesis ranking

Top-1 được chọn bằng score không nhìn ground truth:

\[
S(X,C)=
\sum_k \omega_k\log p_k(O^{(k)}\mid X,M^{(k)})
+\omega_c\log\psi_{geo}(X,C)
+\omega_t\log p_\rho(C)
+[S_{rank}(X,C,O)].
\]

Weights/ranker được fit trên validation set. Semantic score không được dùng làm default ranker cho tới khi chứng minh correlation với human sign judgments trên standardized renders.

## 8. Training curriculum và implementation controls

Một curriculum có thể kiểm thử:

1. **Stage A — geometry/contact pretraining:** học patch encoder và contact proposal trên self-contact/interaction data.
2. **Stage B — sign trajectory prior:** học holistic diffusion trên sign trajectories với graph/mask dropout.
3. **Stage C — sign contact adaptation:** fine-tune graph proposal trên gold + filtered pseudo-labels.
4. **Stage D — optional joint fine-tuning:** chỉ thực hiện nếu các stage độc lập đã qua validation gates.

Mỗi stage lưu config, data manifest, seed, checkpoint-selection rule và compute. Mixed batches phải công bố sampling ratio; batch size không được coi là dataset weighting ngầm.

`[AUTHOR INPUT REQUIRED]` Architecture, parameter count, optimizer, learning rate, epochs/steps, batch composition, augmentation, seeds, hardware, training time và inference time.

## 9. Hypotheses và ablation design

### 9.1 Preregistered hypotheses

- **H1 — Dynamic contact:** dynamic event graph giảm root-aligned hand placement error trên contact+occluded subset so với static-contact và no-contact models có cùng capacity/compute.
- **H2 — Visibility weighting:** calibrated cue reliability cải thiện high-occlusion performance mà không gây practical regression trên low-occlusion/non-contact frames.
- **H3 — Multi-hypothesis utility:** tăng `K` cải thiện risk–coverage hoặc selection quality; best-of-K chỉ là oracle upper bound.

### 9.2 Minimum matched baselines

| ID | Trajectory prior | Contact | Visibility | Mục đích |
|---|---|---|---|---|
| B0 | DexAvatar/current optimization | penetration only | detector confidence hiện tại | Anchor reproduction |
| B1 | Holistic pose/trajectory prior | none | fixed | Tách gain của holistic prior |
| B2 | PAPoseDiff/DPoser-X-style prior | static geometric loss | fixed | Closest static-contact baseline |
| B3 | Matched trajectory diffusion | none | calibrated | No-graph control |
| B4 | Matched trajectory diffusion | static graph | calibrated | Test dynamics |
| B5 | Matched trajectory diffusion | dynamic graph | constant cue weights | Test visibility mechanism |
| B6 | Full | dynamic graph | calibrated | Proposed model |
| B7 | Full | oracle graph | calibrated | Mechanism upper bound; gold subset only |

Parameter count, training data, initialization, sampler steps và optimization budget phải được match hoặc báo riêng.

## 10. Evaluation protocol

### 10.1 Primary endpoint

Đề xuất primary endpoint là **clip-macro root-aligned hand PVE** trên test set, với alignment tại global/root body frame chứ không tại từng wrist. Điều này đo placement của bàn tay tương đối với cơ thể. Chốt một endpoint duy nhất trước khi chạy test.

`[AUTHOR INPUT REQUIRED]` Root definition, vertex set, unit, alignment transform, aggregation rule và smallest practically important effect.

### 10.2 Articulation và body geometry

- **Wrist-aligned hand PVE:** local finger articulation; báo riêng trái/phải và dominant/non-dominant nếu protocol có nhãn.
- **Body MPJPE/PVE:** theo alignment đã công bố, không trộn PA và non-PA trong cùng claim.
- **Penetration:** maximum/mean penetration depth và affected surface area.

Evaluator phải có unit tests với known rigid transforms. File evaluator và commit hash là một phần của artifact.

### 10.3 Contact-event metrics

Chỉ trên gold subset:

- macro precision/recall/F1 theo admissible edge;
- onset và release timing error với tolerance cố định;
- event interval IoU hoặc segmental F1;
- graph edit distance nếu ontology đủ ổn định;
- kết quả theo edge group: hand–hand, hand–face, hand–torso.

Không tính patch-ID accuracy khi patch ground truth không tồn tại. Báo support của từng class và cả micro/macro để lộ class imbalance.

### 10.4 Temporal metrics

Raw acceleration/jerk thấp không mặc nhiên tốt. Báo:

- acceleration/jerk **error so với ground truth**;
- velocity/acceleration spectral distance;
- contact-transition timing error;
- oversmoothing diagnostic: motion amplitude và high-frequency energy ratio.

### 10.5 Visibility-stratified analysis

Khóa visibility bins từ validation protocol rồi báo primary/secondary metrics theo:

- low, medium, high occlusion;
- contact vs non-contact;
- hand–hand vs hand–face/torso;
- seen vs unseen signer/sign nếu thiết kế split cho phép.

Interaction effect giữa contact và occlusion quan trọng hơn một average gain duy nhất.

### 10.6 Uncertainty và multi-hypothesis evaluation

Tách ba chế độ:

- **top-1:** hypothesis được ranking mà không dùng GT;
- **mean/expected:** nếu có weighted samples hợp lệ;
- **oracle best-of-K:** chỉ là upper bound.

Báo risk–coverage curve, area under risk–coverage, empirical coverage–interval width và error–uncertainty rank correlation. AUSE có thể bổ sung nhưng không thay calibration. Không gọi unnormalized energy là NLL.

### 10.7 Semantic evaluation

SiLVERScore hoặc sign-recognizer score chỉ là secondary/exploratory cho tới khi:

1. renderer, camera và background được chuẩn hóa;
2. metric được validation trên language/domain của benchmark;
3. correlation với human expert judgment được báo.

Headline claim vẫn dựa trên geometry/contact metrics trực tiếp.

### 10.8 Statistics

- Aggregate theo clip trước, không coi frame là mẫu độc lập.
- Dùng signer/sign cluster bootstrap; với nhiều signer, dùng hierarchical bootstrap.
- Báo effect size, 95% confidence interval và full per-sequence distribution.
- Chọn một primary endpoint; secondary tests dùng Holm/FDR nếu diễn giải inferentially.
- Hyperparameter và stopping rule khóa trước test evaluation.

## 11. Falsification criteria

Claim trung tâm bị bác bỏ hoặc phải thu hẹp nếu xảy ra một trong các trường hợp:

1. dynamic graph không hơn static contact baseline khi capacity/data/compute được match;
2. gain chỉ xuất hiện ở oracle best-of-K nhưng top-1 không cải thiện;
3. placement tốt hơn nhưng articulation, non-contact frames hoặc temporal fidelity xấu đi vượt practical margin;
4. contact labels có agreement thấp hoặc pseudo-label error quá lớn;
5. PAPoseDiff-style prior + temporal smoothing đạt kết quả tương đương;
6. gain biến mất khi bootstrap ở cấp signer/sign thay vì frame;
7. semantic score tăng nhưng direct geometry/contact metrics không tăng.

## 12. Claim được phép sau từng mức bằng chứng

| Bằng chứng đạt được | Claim tối đa nên dùng |
|---|---|
| Chỉ có metric repair | “We identify and correct a hand-placement evaluation failure.” |
| Static contact prior thắng | “Contact-aware refinement improves sign hand placement.” |
| Dynamic graph thắng matched static baseline | “Temporal contact events improve reconstruction under contact/occlusion.” |
| Joint alternating inference thắng no-alternation | “Joint geometry–contact updates provide additional gains.” |
| Top-1 + calibration protocol thắng | “Multiple hypotheses improve risk-aware selection.” |
| Chưa có normalized likelihood/calibration | Không dùng “calibrated posterior” hoặc “probabilistically calibrated reconstruction.” |

## 13. Reproducibility checklist

- [ ] Evaluator source, unit tests và commit hash.
- [ ] Dataset manifests, licenses và duplicate/leakage check.
- [ ] SMPL-X version, patch map và admissible-edge definition.
- [ ] Gold annotation guide, agreement và adjudication log.
- [ ] Pseudo-label thresholds và error analysis.
- [ ] Model architecture, parameter count và full configs.
- [ ] Training ratios, seeds, compute và checkpoint rule.
- [ ] Inference sampler, steps, guidance schedules, `R` và `K`.
- [ ] Primary endpoint và statistical analysis plan.
- [ ] Per-sequence outputs và failure cases.

## 14. Phát biểu contribution đã hiệu chỉnh

Nếu các falsification tests được vượt qua, contribution có thể viết theo ba ý:

1. **Một contact-event representation sign-specific** mô tả identity, duration và onset–hold–release thay vì chỉ chống penetration.
2. **Một approximate alternating inference procedure** kết hợp graph-conditioned holistic trajectory diffusion với calibrated visibility-weighted observation guidance, đồng thời tách rõ target factors và amortized contact proposal.
3. **Một evaluation protocol audit được** tách hand placement khỏi articulation, đánh giá contact events và kiểm tra multi-hypothesis utility ở cấp clip/signer.

Phát biểu này cố ý không claim rằng self-contact, diffusion prior, contact graph hay visibility awareness tự thân là mới.

## 15. Closest-prior positioning bắt buộc

Related Work và experiment phải đối chiếu trực tiếp ít nhất với:

- [DexAvatar](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html): anchor sign reconstruction;
- [PAPoseDiff / Goliath-SC](https://arxiv.org/html/2509.23393v1): shape-dependent self-contact diffusion và single-view refinement;
- [TUCH](https://arxiv.org/abs/2104.03176): self-contact-aware pose/fitting;
- [DPoser-X](https://arxiv.org/html/2508.00599v2): whole-body diffusion pose prior;
- [HandX](https://arxiv.org/html/2603.28766v1): bimanual temporal/contact-event modeling;
- [Visibility-Aware HOI Tracking](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_Visibility_Aware_Human-Object_Interaction_Tracking_From_Single_RGB_Camera_CVPR_2023_paper.pdf): visibility, temporal contact và monocular tracking;
- [GraphiContact](https://arxiv.org/html/2603.20310v1): graph-based contact prediction + reconstruction;
- [HACO](https://arxiv.org/html/2505.11152v2): dense hand-contact estimation và imbalance;
- [Decaf](https://dl.acm.org/doi/10.1145/3618329) / [DICE](https://arxiv.org/abs/2406.17988): monocular hand–face interaction reconstruction.

Khoảng trống cần chứng minh bằng thực nghiệm là giao của **sign-specific trajectory**, **dynamic self-contact events**, **visibility-conditioned evidence** và **joint inverse inference**—không phải từng thành phần riêng lẻ.
