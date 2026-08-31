# Review kỹ thuật và Methods proposal cuối cho 3D Sign Language Reconstruction

**Ngày rà soát:** 27-08-2026  
**Code được kiểm tra:** Hand4Whole++ `f81d35d`, DexAvatar `a0dfd42`, SMPLest-X `fdebd88`, SGNify `bae2a71`.

## 1. Kết luận điều hành

Không nên triển khai nguyên trạng proposal **SMPLest-X + WiLoR + CHAM + SAR + SRG** đã dán. Ba điều cần sửa là:

1. **CHAM công khai được xây trên SMPLer-X-L32, không phải SMPLest-X.** Port checkpoint CHAM sang SMPLest-X không phải thao tác cắm-và-chạy: số block, luồng token và cách wrist rotation đi vào mesh đều khác.
2. **WiLoR hand replacement và global palm/wrist orientation không còn là novelty đủ mạnh.** SOKE/Signs as Tokens và Tamaththul3D đã dùng các ý tưởng gần như vậy cho sign reconstruction; H4W++ cũng cho thấy hard wrist copy có thể làm kết quả xấu đi đáng kể.
3. **Không thể bảo đảm vô điều kiện một phương pháp monocular sẽ luôn giảm TR-V2V.** Có các cấu hình 3D khác nhau nhưng ảnh chiếu 2D giống nhau. Điều có thể chứng minh là objective mới bám đúng phần sai số mà TR-V2V đo; việc thắng metric vẫn phải được chứng minh bằng thí nghiệm khóa protocol.

Phương án cuối được khuyến nghị là:

> **SignAlign-TR: Uncertainty-Calibrated Translation-Residual Articulator Fitting**

- Front-end mặc định: **Hand4Whole++ chính thức = SMPLer-X-L32 + WiLoR + CHAM**.
- SMPLest-X: giữ làm **hypothesis/ablation thứ hai**, chỉ thay front-end mặc định nếu nó thắng trên development set cùng một evaluator.
- Contribution chính: một **Translation-Residual Articulator Graph (TRAG)** dự đoán các quan hệ 3D bất biến tịnh tiến và độ bất định từ ảnh; sau đó fitting trực tiếp SMPL-X bằng hand-local geometry, palm orientation có gating, torso-relative location và ordinal/relative depth.
- SignBPoser/SignHPoser: không làm contribution chính, nhưng **không bỏ cứng**; dùng làm fallback khi visual evidence yếu nếu checkpoint hợp lệ có sẵn.
- Semantic contact và phonological labels: để ở ablation/extension, không đặt vào đường găng của bản đầu tiên.

Đây là formulation có cơ hội tốt nhất để giảm đồng thời **UBody(-F), LHand và RHand** mà vẫn có một research story mới, đúng metric và triển khai được.

---

## 2. Khóa đúng bài toán và evaluator trước khi thiết kế method

SGNify đánh giá trên 57 sign DGS, tổng 2.872 frame RGB. Các method tạo mesh cùng topology SMPL-X. **TR-V2V** căn chỉnh bằng **một phép tịnh tiến cho toàn mesh ở mỗi frame**, rồi tính khoảng cách vertex-to-vertex trung bình; không được xoay hay rescale riêng. Ba vùng báo cáo là Upper Body không gồm face, Left Hand và Right Hand. Baseline được công bố là SGNify `55.63 / 19.22 / 17.50` và DexAvatar `30.13 / 13.53 / 13.08`. Xem [SGNify paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.pdf), [SGNify code](https://github.com/MPForte/SGNify) và [DexAvatar paper](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html).

Điều này có ba hệ quả thiết kế:

- Root/camera translation tuyệt đối phần lớn bị triệt tiêu; **quan hệ tương đối giữa torso–arm–hand và giữa hai tay** là thứ quyết định metric.
- Hand-only wrist-relative improvement không tự động kéo UBody(-F) xuống. Nếu wrist/forearm/elbow sai, hand mesh dù đẹp vẫn nằm sai vị trí trong global body.
- Một mesh hybrid thay vertex MANO vào SMPL-X có thể đạt metric tốt nhưng không nhất thiết còn được sinh chính xác bởi bộ SMPL-X parameters báo cáo. Cần công bố riêng mesh hybrid và pure parametric mesh.

### 2.1 Liên hệ toán học giữa TR-V2V và relation graph

Gọi sai số vertex là (e_i=v_i^{pred}-v_i^{gt}), và 

\[
\bar e=\frac{1}{N}\sum_i e_i.
\]

Sau khi center bằng một translation, squared residual có đẳng thức:

\[
\frac{1}{N}\sum_i\lVert e_i-\bar e\rVert_2^2
=\frac{1}{2N^2}\sum_{i,j}\lVert e_i-e_j\rVert_2^2.
\]

TR-V2V thực tế lấy mean Euclidean distance, không phải mean squared distance, nhưng theo Jensen/Cauchy:

\[
\frac{1}{N}\sum_i\lVert e_i-\bar e\rVert_2
\leq
\sqrt{\frac{1}{N}\sum_i\lVert e_i-\bar e\rVert_2^2}.
\]

Do đó tối ưu các **relative offsets** (e_i-e_j) làm giảm một upper bound tự nhiên của TR-V2V. Với một graph anchor liên thông, spectral gap của graph Laplacian còn cho một Poincaré bound giữa tổng sai số centered và tổng edge residual. Đây là lý do có cơ sở toán học để dùng torso↔wrist, wrist↔palm, palm↔finger và L-hand↔R-hand edges thay vì chỉ thêm một latent pose prior.

Đẳng thức trên **chứng minh objective alignment**, không chứng minh model image-to-3D sẽ dự đoán đúng target.

### 2.2 Vì sao không thể hứa “chắc chắn thắng metric” trước thí nghiệm

Dưới weak-perspective/orthographic camera, lấy hai pose hợp lệ có cùng (x,y) nhưng khác depth (z) của bàn tay. Hai pose tạo cùng keypoints/ảnh chiếu 2D nhưng có TR-V2V khác nhau. Một thuật toán deterministic nhận cùng ảnh buộc phải trả cùng output, nên không thể đúng cho cả hai ground truths.

Vì vậy, lời khẳng định khoa học đúng là:

> Method mới tối ưu một surrogate bám trực tiếp vào translation-aligned vertex error và dùng thêm learned/depth evidence để giảm ambiguity. Khả năng thắng DexAvatar có bằng chứng hỗ trợ, nhưng chỉ paired evaluation mới xác lập kết quả.

---

## 3. Audit codebase

### 3.1 Hand4Whole++

Hand4Whole++ dùng SMPLer-X-L32 làm whole-body estimator, WiLoR làm hand specialist, và CHAM để đưa hand features ngược vào body encoder. Các base networks được freeze; CHAM dùng cross-attention trái↔phải, zero-initialized adapters và chèn feature vào từng ViT block. Xem [module CHAM ở commit đã audit](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE/blob/f81d35ddd2b74206c40142243eb62b6d64ce0d65/common/nets/module.py) và [forward/model integration](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE/blob/f81d35ddd2b74206c40142243eb62b6d64ce0d65/main/model.py).

Các điểm quan trọng:

- WiLoR feature dimension và adapter được viết cho kiến trúc SMPLer-X-L32 24 block. Không có checkpoint CHAM cho SMPLest-X 32 block.
- H4W++ không chỉ fusion feature: code còn rigid-align MANO từ WiLoR theo wrist/MCP, rồi chép 778 hand vertices vào các chỉ số hand của SMPL-X và smooth boundary.
- Vì MANO hand shape độc lập không nằm hoàn toàn trong shared SMPL-X β, mesh sau replacement không được bảo đảm tái sinh đúng từ bộ SMPL-X params xuất ra.
- Benchmark H4W++ dùng pelvis/root-aligned full-body MPVPE, wrist-relative hand MPVPE và hand-to-hand MRRPE; đó không phải evaluator SGNify TR-V2V.
- Ablation của paper cho thấy naive wrist copy có thể làm AGORA tệ mạnh, trong khi CHAM giúp phục hồi. Vì vậy palm/global wrist orientation phải được **confidence-gated**, không hard replace. Xem [Hand4Whole++ paper](https://arxiv.org/html/2603.14726v1).

**Kết luận:** đây là front-end công khai mạnh và dễ tái lập nhất, nhưng không được viện dẫn kết quả H4W++ như bằng chứng trực tiếp rằng TR-V2V sẽ giảm.

### 3.2 DexAvatar

DexAvatar là optimization pipeline dùng SMPLer-X, HaMeR, Sapiens, SignBPoser/SignHPoser, temporal/biomechanics/interpenetration. Paper đã có signer-space motivation, one-vs-two-hand decision và contact-aware fitting; do đó “sign-space”, generic contact, temporal hoặc pose prior riêng lẻ không còn đủ mới. Xem [DexAvatar paper](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html) và [repository](https://github.com/kaustesseract/DexAvatar).

Audit implementation phát hiện các điểm cần sửa trước khi tái sử dụng:

| Điểm trong code | Hệ quả | Sửa bắt buộc |
|---|---|---|
| `data_3d_weights: [0,0,0]` trong [config fitting](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/cfg_files/fit_smplx_vposer_x.yaml) | Nhánh HaMeR relative-depth chính thức đang tắt | Không được coi depth branch là baseline đã hoạt động; validate rồi mới bật |
| Hand joint slices trong [fitting.py](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fitting.py) trộn các khoảng left/right và bỏ wrist trong một số mode | Nếu bật 3D loss, supervision có thể gắn sai joint | Dùng explicit named mapping và unit test từng index |
| Chuẩn hóa z theo per-axis standard deviation không có ε | Pose gần degenerate có thể tạo NaN/gradient bùng | `std.clamp_min(eps)` và mask low-spread samples |
| Fitting chủ yếu tối ưu latent pose embedding; camera, shape và nhiều global variables bị giữ từ init trong [fit_single_frame.py](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fit_single_frame.py) | Image evidence khó sửa lỗi root/chain/camera đã có | Tối ưu trực tiếp Lie increments của upper chain, camera và clip-shared shape |
| Data terms tether mạnh về SMPLer-X init ở mọi stage | Specialist evidence dễ bị prior đè | Weight theo uncertainty và anneal theo stage |
| Temporal là first-order pose penalty mạnh | Có thể oversmooth chuyển động ký hiệu nhanh | Dùng centered-anchor acceleration, motion-adaptive gating |
| SignB/H checkpoints không nằm trong git | Reproduction phụ thuộc asset ngoài repo | Hash/version checkpoint; có fallback không cần private data |

Paper ablation cũng không ủng hộ việc bỏ cứng SignBPoser/SignHPoser: filtered pose priors cải thiện đáng kể so với unfiltered, còn biomechanics lúc train chỉ giúp ít hoặc đôi khi giảm nhẹ. Cách hợp lý hơn là đặt prior weight tăng khi image-conditioned uncertainty tăng.

### 3.3 SMPLest-X

SMPLest-X dùng ViT-H, 32 encoder blocks và task-token decoder. Trong đường tạo mesh hiện tại, predicted left/right hand-root tokens chủ yếu phục vụ consistency supervision; wrist orientation thực tế đi qua body kinematic chain. Vì vậy chỉ chèn WiLoR vào hand-root token không sửa được global hand placement. Xem [SMPLest_X.py](https://github.com/MotrixLab/SMPLest-X/blob/fdebd887a317f9004b435c57812d1a8936295360/models/SMPLest_X.py) và [token/output modules](https://github.com/MotrixLab/SMPLest-X/blob/fdebd887a317f9004b435c57812d1a8936295360/models/module.py).

Hệ quả:

- Port CHAM cần adapter cho 32 blocks hoặc decoder-side gated cross-attention mới; phải retrain.
- WiLoR orientation phải tác động vào shoulder–elbow–wrist chain, không chỉ finger pose.
- Checkpoint lớn và training guide nhiều GPU làm ablation port tốn kém hơn dùng H4W++ chính thức.
- SMPLest-X vẫn đáng giữ vì output pure SMPL-X và backbone mạnh, nhưng chưa có bằng chứng tin cậy rằng nó thắng SMPLer-X trên SGNify. Xem [SMPLest-X paper](https://arxiv.org/html/2501.09782v1) và [repository](https://github.com/MotrixLab/SMPLest-X).

### 3.4 SGNify baseline đã có gì

SGNify code/paper đã có robust 2D reprojection, 2D bone orientation, body/hand priors, previous-frame temporal terms, representative hand-pose symmetry/invariance, standing prior và generic self-contact/interpenetration. Vì vậy các contribution sau bị trùng hoặc quá incremental nếu đứng một mình:

- symmetry/invariance;
- generic self-contact;
- simple temporal smoothness;
- hand pose distribution prior;
- một 2D keypoint fitting term khác.

---

## 4. Kiểm tra novelty với literature gần nhất

| Work | Thành phần liên quan | Ý nghĩa đối với proposal |
|---|---|---|
| [SOKE / Signs as Tokens, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zuo_Signs_as_Tokens_A_Retrieval-Enhanced_Multilingual_Sign_Language_Generator_ICCV_2025_paper.html) | Curate SMPL-X bằng OSX + WiLoR hand pose/global orientation + upper-body refinement | Palm/global wrist transfer và WiLoR fusion không còn mới; báo cáo `46.73 / 10.55 / 8.94` trên SGNify |
| [Tamaththul3D, 2026 preprint](https://arxiv.org/html/2605.05367v2) | SMPLer-X/SMPLest-X + WiLoR, forearm alignment, swing–twist, shoulder 2D optimization | Rất gần phần palm/arm refinement; báo cáo `29.28 / 10.65 / 8.90`, nhưng chưa thấy official code và cách gọi metric trong paper cần được xác minh độc lập |
| [A2P, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Han_From_2D_Alignment_to_3D_Plausibility_Unifying_Heterogeneous_2D_Priors_CVPR_2026_paper.pdf) | Fuse 2D keypoint/segmentation/depth features cho two-hand reconstruction | Hỗ trợ giả thuyết rằng depth giảm z-error/MRRPE; cũng có nghĩa “dùng depth” chung chung không đủ novelty |
| [ProsePose, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Subramanian_Pose_Priors_from_Language_Models_CVPR_2025_paper.pdf) | Biến semantic contact descriptions thành optimization losses | Semantic contact graph khả thi, nhưng không hoàn toàn mới ở cấp kỹ thuật |
| [Phonology-guided generation, CVPRW 2026](https://arxiv.org/pdf/2603.17388) | Handshape/location/orientation conditioning cho generation | Cho thấy các attributes có ý nghĩa, nhưng không tự động biến một geometric loss không có labels thành “phonology-grounded” |
| [WiLoR](https://arxiv.org/html/2409.12259v2) | In-the-wild MANO hand reconstruction | Là hand expert tốt nhưng có scale, flip và global orientation uncertainty cần calibrate |

Các số cross-paper ở bảng trên chỉ là **evidence envelope**, không phải leaderboard hợp lệ nếu evaluator, crop hoặc mesh variant khác nhau. Mọi claim cuối phải chạy lại chung một script.

---

## 5. Review từng thành phần proposal ban đầu

| Thành phần | Phán quyết | Lý do và cách giữ lại |
|---|---|---|
| SMPLest-X làm backbone mặc định | **Chưa chấp nhận** | Chưa có CHAM checkpoint tương thích và chưa chứng minh tốt hơn trên SGNify; giữ làm secondary hypothesis |
| H4W++ front-end chính thức | **Chấp nhận** | Reproducible, có hand↔body feature fusion; phải đánh giá pure-SMPL-X và hybrid mesh riêng |
| Handshape geometric descriptor | **Giữ** | Tốt hơn L2 trên axis-angle; dùng wrist-local joints/bones/fingertips và robust uncertainty, nhưng không claim là novelty duy nhất |
| Palm orientation | **Giữ có điều kiện** | Có ích cho wrist/forearm nhưng đã có prior work; dùng SO(3) geodesic + confidence gate, không hard copy |
| Body-relative signing-space location | **Giữ và đổi formulation** | Đúng TR-V2V, nhưng target 3D không thể suy ra chỉ từ x,y; cho relation head dự đoán mean + variance trong torso frame |
| Relative depth map loss | **Giữ nhưng không dùng metric depth trực tiếp** | Monocular depth thường chỉ đúng order/affine; dùng ordinal pairs + uncertainty, học magnitude từ 3D relation data |
| Sign Relational Graph | **Nâng thành contribution chính** | Đổi thành graph dự đoán relative 3D observations và covariance, gắn thẳng với translation-residual metric |
| Semantic contact graph | **Optional ablation** | Chỉ có tác dụng ở frame có contact; detector/labels là bottleneck, generic contact đã có |
| Dominant/non-dominant roles | **Optional** | Cần classifier/label đáng tin; hard one-hand decision có thể tắt nhầm articulator |
| Bỏ SignHPoser/SignBPoser | **Bác bỏ** | Ablation DexAvatar cho thấy prior hữu ích; dùng confidence-adaptive fallback thay vì contribution |
| Tên “Phonology-grounded” | **Chưa dùng** | Không có explicit DGS phonological labels/classifier được validate; nên gọi “articulator-aware” và language-agnostic |

---

## 6. Methods proposal cuối: SignAlign-TR

### 6.1 Mục tiêu thiết kế

Method phải đồng thời giải bốn loại sai số:

1. **Local hand articulation:** ngón tay, thumb opposition, fingertip arrangement.
2. **Global hand orientation:** palm/wrist orientation trong camera và torso frame.
3. **Upper-chain placement:** shoulder–elbow–wrist–palm, quyết định UBody(-F).
4. **Cross-articulator depth/relations:** L↔R hand và hand↔torso, là phần 2D reprojection không quan sát đủ.

### 6.2 Front-end và output contract

Cho clip (I_{1:T}):

- Chạy H4W++ chính thức để lấy SMPL-X initialization, body features và WiLoR hand features.
- Chạy body keypoint/segmentation detector độc lập và một frozen monocular depth feature extractor.
- Nếu ngân sách cho phép, chạy SMPLest-X thành hypothesis thứ hai. Chọn hypothesis theo một score chỉ dùng image evidence đã calibrate; không dùng SGNify test GT.
- Output chính là **pure SMPL-X vertices từ parameters**. H4W++ MANO-insert hybrid mesh chỉ báo cáo như variant phụ.

### 6.3 Translation-Residual Articulator Graph (TRAG)

Định nghĩa anchor set:

\[
\mathcal A=\{neck, shoulders, elbows, wrists, palms, MCPs, fingertips\}.
\]

Graph gồm:

- kinematic edges: shoulder→elbow→wrist→palm→finger;
- long-range edges: torso→palm;
- cross-hand edges: L-palm↔R-palm và các fingertip pairs quan trọng.

Một lightweight relation head nhận body feature, WiLoR L/R features, keypoint heatmaps và depth features. Với mỗi edge (e=(i,j)), head dự đoán:

\[
(\mu_e,\log\sigma_e^2,p_e),
\]

trong đó (mu_e) là 3D relative offset trong torso-normalized coordinate, (sigma_e) là aleatoric uncertainty và (p_e) là visibility/reliability. Không dự đoán full SMPL-X pose lần nữa.

Từ current mesh (M(q)), relative edge là

\[
\Delta a_e(q)=R_T(q)^\top\frac{a_j(q)-a_i(q)}{s_T(q)}.
\]

Loss:

\[
\mathcal L_{TRAG}
=\sum_e p_e\left[
\rho\!\left(\frac{\Delta a_e(q)-\mu_e}{\sigma_e}\right)
+\log \sigma_e^2
\right].
\]

Việc học covariance ngăn một cue sai như flipped palm hoặc noisy depth áp đảo optimizer. TTA flip/scale/crop variance có thể được cộng vào epistemic uncertainty lúc test.

**Training:** dùng ground-truth/pseudo-ground-truth 3D trên AGORA, ARCTIC, InterHand2.6M/ReInterHand và sign-domain data được tách khỏi SGNify test. Huấn luyện thêm centered-vertex surrogate trên data có SMPL-X:

\[
\mathcal L_{CTR}=\frac{1}{N}\sum_i
\left\|(v_i^{pred}-\bar v^{pred})-(v_i^{gt}-\bar v^{gt})\right\|_1.
\]

Đây là phần khiến contribution bám evaluator hơn một semantic graph chỉ có contact distance.

### 6.4 Hand-local geometric observation

Rigid/similarity-align WiLoR MANO vào SMPL-X wrist/MCP frame, sau đó so các quantities có ý nghĩa hình học:

\[
\mathcal L_{hand-local}=
\sum_{h\in\{L,R\}}c_h
\left(
\sum_k \rho(\hat b_{hk}-b_{hk}(q))
+\sum_m\rho(\hat d_{hm}-d_{hm}(q))
\right),
\]

với (b) là normalized bone vectors và (d) là selected fingertip/thumb distances. Không dùng L2 trực tiếp giữa MANO axis-angle và SMPL-X hand parameters.

### 6.5 Palm orientation với SO(3) và gating

Tạo palm frame từ wrist và MCPs. Dùng geodesic residual:

\[
\mathcal L_{ori}
=\sum_h c_h^{ori}
\left\|\log\left(\hat R_h^\top R_h(q)\right)\right\|_2.
\]

(c_h^{ori}) giảm khi WiLoR flip-TTA không nhất quán, hand crop nhỏ, arm direction mâu thuẫn hoặc palm basis gần suy biến. Term này tác động vào Lie increments của wrist và upper chain; không chép cứng global wrist rotation.

### 6.6 Torso-relative location và depth

Signing-space coordinate vẫn hữu ích, nhưng target đến từ TRAG chứ không ghép 2D với một depth scalar chưa calibrate:

\[
\ell_h(q)=R_T(q)^\top\frac{C_h(q)-O_T(q)}{s_T(q)}.
\]

TRAG supervise (ell_h) theo mean/variance. Generic monocular depth chỉ tạo ordinal constraints:

\[
\mathcal L_{ord-z}
=\sum_{(i,j)} c_{ij}
\operatorname{softplus}\!\left(-y_{ij}\frac{z_i(q)-z_j(q)}{\tau}\right),
\]

với pairs palm↔chest và L-palm↔R-palm. (y_{ij}\in\{-1,+1\}) là front/behind order. Cách này không giả vờ monocular depth có scale metric tuyệt đối.

### 6.7 Adaptive prior, temporal và physical constraints

- Shape β được share theo clip/signer và estimate từ các frame torso rõ; sau đó freeze hoặc regularize mạnh.
- SignB/H hoặc generic pose prior có weight (lambda_{prior}(u)) tăng theo predicted visual uncertainty (u). Khi cue rõ, prior không kéo pose về mean sign manifold.
- Temporal term dùng acceleration của **centered anchors**, không phạt mọi chuyển động first-order:

\[
\mathcal L_{acc}=\sum_t w_t
\left\|\tilde a_{t+1}-2\tilde a_t+\tilde a_{t-1}\right\|_1,
\]

trong đó (w_t) giảm ở frame có motion evidence lớn.
- Giữ joint limits, penetration và differentiable biomechanics. Semantic contact chỉ bật khi contact probability vượt threshold và depth/mask cùng đồng thuận.

### 6.8 Objective đầy đủ

\[
\begin{aligned}
E(q)=&\;\lambda_{2D}\mathcal L_{2D}
+\lambda_G\mathcal L_{TRAG}
+\lambda_H\mathcal L_{hand-local}
+\lambda_R\mathcal L_{ori}\\
&+\lambda_Z\mathcal L_{ord-z}
+\lambda_S\mathcal L_{sil}
+\lambda_T\mathcal L_{acc}
+\lambda_P(u)\mathcal L_{prior}\\
&+\lambda_B\mathcal L_{biomech}
+\lambda_I\mathcal L_{penetration}
+\lambda_C\mathcal L_{contact}^{optional}.
\end{aligned}
\]

Mọi residual image-conditioned dùng robust penalty và uncertainty; không dùng một bộ fixed weights cực lớn cho mọi frame.

### 6.9 Lịch tối ưu

1. **Calibration stage:** estimate clip-shared β, camera scale/focal hypothesis và torso orientation từ frame tin cậy; không tối ưu finger.
2. **Upper-chain stage:** tối ưu spine, shoulders, elbows, wrists bằng 2D + TRAG + orientation + ordinal depth; hand articulation giữ từ WiLoR/H4W++.
3. **Hand stage:** tối ưu finger pose bằng hand-local geometry; vẫn cho wrist/forearm điều chỉnh nhỏ trong trust region.
4. **Joint stage:** tối ưu toàn upper body + hands với centered temporal/physical terms; anneal initialization tether.
5. **Safety selection:** chỉ nhận refined result nếu calibrated image-evidence score tốt hơn initialization và không tăng penetration/joint-limit violations; nếu không, fallback theo vùng hoặc toàn frame.

Pose update nên dùng SO(3) exponential-map increments. Camera, pose, shape và units phải có gradient/finite-difference tests riêng.

---

## 7. Evaluation đủ mạnh để “chứng minh” claim

### 7.1 Khóa protocol

- Reproduce đúng ba con số published của ít nhất SGNify và DexAvatar trước khi dùng evaluator.
- Xác nhận vertex subsets, face exclusion, đơn vị mm và translation centering bằng unit tests.
- Không tune trên 2.872 SGNify test frames có ground truth. Tách external/dev data hoặc một development split được khai báo trước.
- Report pure SMPL-X parametric mesh và optional hybrid mesh ở hai dòng khác nhau.

### 7.2 Ablation tối thiểu

| ID | Method |
|---|---|
| A0 | SMPLer-X official |
| A1 | SMPLest-X official |
| A2 | Hand4Whole++ official |
| A3 | A2 + clip-shared shape/camera calibration |
| A4 | A3 + hand-local geometry |
| A5 | A4 + gated SO(3) palm orientation |
| A6 | A5 + TRAG (x,y) only |
| A7 | A6 + learned/ordinal (z) |
| A8 | A7 + uncertainty calibration |
| A9 | A8 + adaptive pose prior |
| A10 | A9 + motion-adaptive temporal |
| A11 | A10 + optional semantic contact |
| Full | Best preregistered configuration |

Thêm ablation `hard wrist copy` để chứng minh gating thật sự cần thiết, và `SMPLest Token-CHAM` chỉ khi đã retrain đúng kiến trúc.

### 7.3 Metrics

Primary:

- SGNify TR-V2V UBody(-F), LHand, RHand.

Diagnostic:

- palm orientation error theo degree;
- torso-relative palm location/depth error;
- hand-to-hand MRRPE;
- local fingertip/bone error;
- penetration/contact precision trên subset có annotation;
- acceleration/jitter và error theo motion-speed bins.

### 7.4 Statistical decision rule

Frame trong cùng sign tương quan, nên không bootstrap frame độc lập. Dùng paired bootstrap theo **sign clip** hoặc paired permutation/Wilcoxon ở mức sign, báo 95% CI.

Go/no-go nên được preregister:

- **Gate 1:** Full method phải tốt hơn reproduced DexAvatar ở cả ba vùng, với 95% paired CI của chênh lệch nằm dưới 0 hoặc ít nhất không suy giảm vùng nào và có gain chính được định trước.
- **Gate 2:** để claim state of the art, chạy cùng evaluator và nhắm envelope có margin: UBody(-F) `< 28.5`, LHand `< 10.5`, RHand `< 8.9` mm. Đây là ngưỡng mục tiêu, không phải dự đoán được bảo đảm.
- **Gate 3:** đa số sign clips phải cải thiện; không chấp nhận mean gain do vài clip outlier.
- **Gate 4:** pure SMPL-X result phải được báo; hybrid-only gain không đủ cho claim parametric reconstruction.

---

## 8. Feasibility và rủi ro

| Hạng mục | Khả thi | Rủi ro | Quyết định |
|---|---:|---:|---|
| Dùng H4W++ official làm init | Cao | Thấp–TB | Làm đầu tiên |
| Clip-shared β/camera calibration | Cao | Thấp | Làm đầu tiên |
| Hand-local WiLoR geometry | Cao | Thấp | Làm đầu tiên |
| Gated SO(3) palm orientation | Cao | TB | Làm sau calibration |
| TRAG (x,y) relations | Cao | TB | Core contribution |
| Learned/ordinal depth (z) | TB | Cao | Core research bet, cần uncertainty |
| Direct SMPLest-X CHAM port | TB | Cao/đắt | Chỉ ablation sau baseline |
| Adaptive SignB/H prior | TB | Asset-dependent | Fallback, không core |
| Semantic contact | Thấp–TB | Cao | Optional |
| Explicit phonology classifier/loss | Thấp khi thiếu DGS labels | Cao | Không đưa vào v1 |

### Kế hoạch triển khai giảm rủi ro

1. Reproduce official evaluator và các baselines.
2. Sửa DexAvatar mappings/numerics; chạy `data_3d_weights=0` và corrected non-zero branch như một ablation riêng.
3. Đưa H4W++ pure mesh vào evaluator, chưa refinement.
4. Thêm clip calibration, local hand, gated orientation; dừng nếu upper-body regression.
5. Huấn luyện TRAG (x,y), sau đó mới thêm depth (z).
6. Chỉ port CHAM sang SMPLest-X nếu A1 thật sự có lợi hoặc compute budget cho phép.
7. Chỉ thêm contact/phonology sau khi core model vượt DexAvatar có ý nghĩa thống kê.

---

## 9. Claim khoa học nên dùng

Không nên claim:

> “Chúng tôi bảo đảm giảm TR-V2V” hoặc “lần đầu dùng palm orientation/phonology trong sign reconstruction.”

Claim defensible hơn:

> “We formulate monocular sign reconstruction as uncertainty-calibrated fitting of translation-invariant articulator relations. The objective is aligned with translation-only vertex evaluation, while learned relative-depth observations address ambiguities left unresolved by 2D reprojection and local hand reconstruction.”

Evidence hiện có ủng hộ hướng này theo ba mảnh độc lập:

- H4W++ cho thấy hand-to-body feature feedback tốt hơn naive wrist replacement trên whole-body benchmarks.
- SOKE/Tamaththul3D cho thấy WiLoR có thể đưa hand error SGNify xuống khoảng 9–11 mm, nhưng body placement vẫn là bottleneck.
- A2P cho thấy depth feature giảm z-error và hand-to-hand relative error trong two-hand reconstruction.

Ba mảnh này làm hypothesis **hợp lý và có khả năng thắng DexAvatar**, nhưng paired experiments mới là bằng chứng kết quả.

## 10. Chốt cuối

Nếu mục tiêu là paper vừa có novelty vừa có xác suất metric tốt, lựa chọn nên là:

> **H4W++ initialization + pure-SMPL-X SignAlign-TR refinement**, với TRAG, uncertainty-gated palm/hand observations, torso-relative learned relations và ordinal depth.

Không chọn SMPLest-X + CHAM làm đường mặc định trước khi retrain adapter và có evidence trên development set. Không bỏ cứng SignH/B priors; không để semantic contact hay phonology labels trở thành dependency bắt buộc. Phần “chứng minh” hợp lệ gồm: (i) đẳng thức translation-residual ở mục 2.1, (ii) correctness/unit tests cho coordinate và evaluator, và (iii) paired clip-level evaluation theo protocol ở mục 7. Không có chứng minh lý thuyết nào thay thế được phần (iii) đối với một bài toán monocular bất định.
