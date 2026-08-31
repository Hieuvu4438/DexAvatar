# Change log, evidence ledger và source map

**Áp dụng cho:** `Bao_cao_nghien_cuu_CoSign4D_DexAvatar(1).md/.docx`  
**Ngày cutoff của lượt tìm kiếm:** 21-08-2026  
**Nguyên tắc:** nguồn sơ cấp trước; tách accepted/peer-reviewed khỏi preprint; không xác nhận code khi file không được cung cấp.

## 1. Đối chiếu hai file đầu vào

- Nội dung text của `.md` và `.docx` trùng nhau sau khi trích xuất.
- Khác biệt thực tế chỉ nằm ở cách đóng gói hình: `.md` nhúng Figure 1 dạng data URI; `.docx` lưu hình trong media package.
- Bản Word có 19 trang, dùng hệ màu navy–orange, cover, header/footer và bảng định dạng sẵn. Không phát hiện một phiên bản Methods khác ẩn trong Word.
- Hai file gốc được giữ nguyên.

## 2. Mapping phần gốc → thay đổi trong bản mới

| Phần bản gốc | Vấn đề | Thay đổi đã thực hiện | Tệp đích |
|---|---|---|---|
| `Quyết định nghiên cứu` | Khóa direction và venue story quá sớm | Đổi verdict thành conditional; thêm bốn evidence gates trước full model | Review §1, §7 |
| `DexAvatar: paper đang làm gì` | Lập luận đúng nhưng dùng “contact” hơi nhị phân | Giữ gap positive contact; phân biệt rõ interpenetration avoidance với identity/event modeling | Review §3 |
| `Evaluator hiện chưa đo đúng` | Tuyên bố đã audit exact code/line nhưng evaluator không được đính kèm | Hạ status thành conditional/unverified; yêu cầu source + unit tests | Review CR-01; Methods §10.2 |
| `Literature review đa miền` | Rộng nhưng bỏ sót prior gần nhất về self-contact diffusion và dynamic hand contact | Thêm PAPoseDiff/Goliath-SC, TUCH, HandX, GraphiContact, HACO, Decaf, DICE, ProsePose, Visibility-Aware HOI | Review CR-03; Methods §15 |
| `Bảy hướng ứng viên` | Fusion rule còn mang tính danh sách module | Chuyển thành một hypothesis có thể bác bỏ: dynamic events giúp placement dưới occlusion | Review §9; Methods §9 |
| `7.2 State representation` | `C_t` chưa có ontology/state operational | Định nghĩa surface patches, admissible edges và `off/onset/hold/release` | Methods §3 |
| `7.3 Probabilistic factorization` | Trộn `p(C|Y)` discriminative với generative target; double-count `Y` | Tách target `pi(X,C|O,M)` khỏi proposal `q_eta(C|O,X,M)` | Methods §4 |
| `7.4 Module design` | Chưa định nghĩa visibility, geometry compatibility, event duration | Thêm calibrated reliability, contact observables, semi-Markov prior và losses | Methods §2–§5 |
| `7.5 Inference algorithm` | Dùng `E_diff` chưa định nghĩa; graph update không khớp factorization | Thay bằng posterior-score guidance + graph decode; gọi đúng là approximate alternating inference | Methods §7 |
| `7.6 Borrowed vs new` | Claim novelty rộng | Thu hẹp novelty vào giao của sign trajectory + dynamic event graph + visibility-conditioned evidence + joint inverse inference | Review §5; Methods §14–§15 |
| `Binary novelty matrix` | DPoser-X gắn temporal `Partial`; thiếu closest priors | Sửa DPoser-X thành non-temporal; tạo novelty map cập nhật | Review §5 |
| `Data curriculum` | Không có nhãn graph audit được; pseudo-label risk chưa xử lý | Thêm gold subset, double annotation, uncertain labels, leakage control và supervision masks | Methods §6 |
| `Implementation roadmap` | Có schedule nhưng thiếu config quyết định khả năng tái lập | Thêm architecture/compute/sampler/data-manifest placeholders và checklist | Methods §8, §13 |
| `Metric suite` | Raw jerk, NLL/energy proxy, AUSE và contact metrics chưa đủ điều kiện | Dùng jerk error/spectral metrics; bỏ pseudo-NLL; thêm risk–coverage/coverage–width; contact chỉ trên gold subset | Review M-03; Methods §10 |
| `Factorial ablation` | Chưa có closest-prior matched baselines | Thêm PAPoseDiff/DPoser-X static-contact, no-graph/static-graph và oracle-contact controls | Methods §9.2 |
| `Go/no-go thresholds` | `5%/1%` mang tính quản trị nhưng trình bày như ngưỡng khoa học | Yêu cầu pilot variance, practical effect và confidence interval | Review §7; Methods §10.8 |
| `Conference-ready story` | Dùng “posterior” và “calibrated” trước bằng chứng | Hạ claim; cung cấp claim ladder theo evidence | Methods §12, §14 |
| `Curated bibliography` | Status/coverage chưa đầy đủ | Bổ sung source ledger, publication status và adversarial prior search | Tệp này §4–§5 |

## 3. Claim disposition

### 3.1 Giữ nguyên về bản chất

- Monocular sign reconstruction chịu failure mode mạnh ở hands, contact và occlusion.
- Cần tách hand placement khỏi local articulation trong evaluator.
- Holistic body–hand modeling hợp lý hơn việc ghép prior độc lập nếu interaction là contribution trung tâm.
- Dynamic contact có khả năng cung cấp bằng chứng bổ sung khi 2D cues yếu.
- Falsification tests và matched ablations phải quyết định có tiếp tục hướng hay không.

### 3.2 Giữ nhưng thu hẹp

| Claim cũ | Claim sau review |
|---|---|
| “DexAvatar không contact-aware” | DexAvatar có terminology/contact-aware penalties nhưng objective công bố chưa mô hình positive contact identity/event tường minh. |
| “CoSign4D là graph-conditioned whole-body diffusion mới” | Novelty có thể nằm ở temporal sign contact-event process được suy luận đồng thời với trajectory; các thành phần self-contact/diffusion/graph riêng lẻ đã có prior. |
| “Visibility-aware posterior” | Visibility-weighted conditional inference với reliability được hiệu chỉnh. |
| “Calibrated multiple hypotheses” | Multiple hypotheses; chỉ claim calibration sau coverage/risk protocol. |
| “Semantic preservation qua SiLVERScore” | Secondary exploratory metric sau domain/human validation. |
| “Metric bug đã verified” | Algebraically valid conditional diagnosis; source code verification còn thiếu. |

### 3.3 Bỏ hoặc tạm dừng

- Bỏ exact code-line citation cho evaluator cho tới khi có file/commit.
- Bỏ DPoser-X temporal capability trong novelty matrix.
- Tạm dừng claim exact NLL hoặc calibrated posterior.
- Tạm dừng patch-ID/contact-F1 trên toàn benchmark nếu không có ground truth.
- Không dùng raw jerk giảm như bằng chứng độc lập cho temporal quality.
- Không dùng best-of-K làm headline deployed performance.
- Không công bố “Best Paper Award Finalist” nếu tác giả chưa cung cấp nguồn chính thức.

## 4. Evidence ledger — nguồn neo và prior gần nhất

| Nguồn | Năm/status | Nội dung đã kiểm tra | Tác động lên review |
|---|---|---|---|
| [DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html) | WACV 2026, proceedings | Pipeline, priors, dataset size, metrics/results, optimization framing | Xác nhận anchor; gap phải được mô tả là positive/dynamic contact, không phải mọi dạng contact. |
| [DexAvatar—arXiv full text](https://arxiv.org/html/2512.21054v1) | arXiv v1 | Objective/loss details, ablations và benchmark protocol | Đối chiếu formal objective với wording “contact-aware”. |
| [Generative Modeling of Shape-Dependent Self-Contact Human Poses](https://arxiv.org/html/2509.23393v1) | arXiv 2025 | Goliath-SC, shape-conditioned PAPoseDiff, SMPL-X self-contact, image refinement | Closest omitted prior; làm yếu claim “whole-body contact diffusion/refinement”. |
| [On Self-Contact and Human Pose](https://arxiv.org/abs/2104.03176) | CVPR 2021 / arXiv | Self-contact data/constraints và contact-aware fitting | Chứng minh positive self-contact constraints đã có trong human pose estimation. |
| [DPoser-X](https://arxiv.org/html/2508.00599v2) | ICCV 2025 / arXiv | Unconditional whole-body pose diffusion prior | Sửa novelty matrix: pose prior, không phải temporal trajectory model. |
| [HandX: Scaling Bimanual Motion and Interaction Generation](https://arxiv.org/html/2603.28766v1) | arXiv 2026 | Bimanual motion, contact events và contact precision/recall/F1 | Prior trực tiếp cho dynamic hand contact representation/evaluation, dù là generation. |
| [Visibility Aware Human-Object Interaction Tracking From Single RGB Camera](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_Visibility_Aware_Human-Object_Interaction_Tracking_From_Single_RGB_Camera_CVPR_2023_paper.pdf) | CVPR 2023 | Temporal human/object/contact tracking dưới visibility constraints | “Visibility-aware contact tracking” không thể là novelty rộng. |
| [GraphiContact](https://arxiv.org/html/2603.20310v1) | arXiv 2026 | Graph-based contact perception cùng 3D mesh reconstruction | Adjacent closest prior cho graph/contact/reconstruction dưới nhiễu và occlusion. |
| [Learning Dense Hand Contact Estimation from Imbalanced Data](https://arxiv.org/html/2505.11152v2) | arXiv 2025 | Dense hand contact và class/spatial imbalance | Cơ sở cho `q_eta`, label taxonomy và imbalance protocol. |
| [Decaf](https://dl.acm.org/doi/10.1145/3618329) | ACM TOG 2023 | Monocular hand–face interaction/deformation capture | Làm yếu claim rộng về hand–face contact novelty. |
| [DICE](https://arxiv.org/abs/2406.17988) | arXiv 2024 | End-to-end hand–face interaction reconstruction từ một ảnh | Closest adjacent prior cho hand–face branch. |
| [Pose Priors from Language Models / ProsePose](https://arxiv.org/html/2405.03689v2) | arXiv 2024 | Image-derived body-region contact pairs đưa vào pose optimization | Contact-pair constraint từ ảnh không còn mới tự thân. |
| [Score-Guided Diffusion for 3D Human Recovery / ScoreHMR](https://openaccess.thecvf.com/content/CVPR2024/papers/Stathopoulos_Score-Guided_Diffusion_for_3D_Human_Recovery_CVPR_2024_paper.pdf) | CVPR 2024 | Score-guided inverse human recovery | Bắt buộc mô tả score guidance đúng; không thay score bằng energy tùy ý. |
| [DPMesh](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_DPMesh_Exploiting_Diffusion_Prior_for_Occluded_Human_Mesh_Recovery_CVPR_2024_paper.pdf) | CVPR 2024 | Diffusion prior cho occluded human mesh recovery | Occlusion + diffusion reconstruction đã có prior. |
| [Neural Sign Actors](https://openaccess.thecvf.com/content/CVPR2024/papers/Baltatzis_Neural_Sign_Actors_A_Diffusion_Model_for_3D_Sign_Language_CVPR_2024_paper.pdf) | CVPR 2024 | Diffusion cho 3D sign language generation | Sign-specific diffusion đã có ở generation; inverse reconstruction vẫn khác. |
| [SiLVERScore](https://aclanthology.org/2025.ranlp-1.54/) | RANLP 2025 | Semantic metric cho sign generation trên PHOENIX-14T/CSL-Daily | Không tự động chuyển sang German isolated-sign reconstruction/rendering. |
| [From 2D Alignment to 3D Plausibility](https://openaccess.thecvf.com/content/CVPR2026/html/Han_From_2D_Alignment_to_3D_Plausibility_Unifying_Heterogeneous_2D_Priors_CVPR_2026_paper.html) | CVPR 2026 | Heterogeneous 2D priors cho 3D plausibility | Adjacent prior cho cue fusion/likelihood design. |
| [CoSIGN: Few-Step Guidance of ConSIstency Model to Solve General INverse Problems](https://arxiv.org/abs/2407.12676) | arXiv 2024 | Tên CoSIGN trong diffusion inverse problems | Naming/discoverability conflict với CoSign4D. |

## 5. Evidence ledger — nguồn bổ trợ có trong bản gốc

Các nguồn sau vẫn hữu ích nhưng không thay closest-prior comparison ở §4:

| Nguồn | Vai trò hợp lý sau review | Giới hạn |
|---|---|---|
| [DanceHMR](https://arxiv.org/abs/2605.18102) | Video/dance whole-body reconstruction và temporal cues | Không phải sign/contact method. |
| [OmniHands](https://dl.acm.org/doi/10.1145/3807943) | Hand modeling/occlusion prior | Không chứng minh dynamic sign contact graph. |
| [CoToGrasp](https://arxiv.org/abs/2608.19776) | Contact-aware bimanual interaction transfer | Hand–object/grasp setting; publication status phải ghi chính xác theo thời điểm. |
| [Tamaththul3D](https://arxiv.org/abs/2605.05367) | Sign-specific 3D direction/representation | Không thay direct reconstruction baseline. |

## 6. Search protocol đã dùng

### 6.1 Query clusters

1. **Anchor verification:** DexAvatar WACV 2026, dataset size, objective, hand metrics, ablation, repository/status.
2. **Whole-body/contact diffusion:** SMPL-X self-contact diffusion, contact-aware pose prior, whole-body diffusion refinement.
3. **Dynamic interaction:** bimanual contact events, hand–hand motion generation, temporal contact graph.
4. **Visibility/occlusion:** visibility-aware contact tracking, occluded human mesh recovery, uncertainty-aware reconstruction.
5. **Contact estimation:** dense hand contact, hand–face capture, body-region contact pairs, contact graph reconstruction.
6. **Sign semantics:** 3D sign generation, sign semantic metrics, domain transfer limitations.
7. **Naming:** CoSign/CoSIGN inverse problem, method-name collision.

### 6.2 Inclusion

- Nguồn chính chủ hoặc proceedings/arXiv/DOI.
- Direct task prior, mechanism prior hoặc evaluation prior có ảnh hưởng thực sự tới novelty/method design.
- Ưu tiên công trình 2021–2026; giữ paper cũ hơn nếu là nguồn gốc trực tiếp của contact constraint.

### 6.3 Exclusion

- Blog, aggregator hoặc secondary summary khi full paper/project chính chủ có sẵn.
- Paper chỉ dùng từ khóa “contact/graph/visibility” nhưng không thay đổi novelty hoặc baseline.
- Claim status/award không truy được nguồn chính thức.

### 6.4 Caveat

Đây là **targeted adversarial literature review**, không phải systematic review theo PRISMA. Nó được thiết kế để tìm prior gần nhất và phá thử novelty claim. Literature sau ngày cutoff, supplementary/code chưa công khai, hoặc dataset license không truy cập được có thể làm thay đổi kết luận.

## 7. Các thay đổi có ảnh hưởng lớn nhất tới hướng nghiên cứu

1. **Novelty confidence giảm từ Medium–High xuống Medium/contingent** vì PAPoseDiff đã phủ self-contact SMPL-X diffusion + single-view refinement.
2. **Factorization được thay hoàn toàn:** `q_eta(C|O,X,M)` là inference proposal, không phải generative factor; target có event prior và geometry compatibility riêng.
3. **Diffusion inference được viết lại bằng score guidance**, loại bỏ `E_diff` không định nghĩa.
4. **Contact supervision trở thành gate bắt buộc:** cần gold subset và agreement trước khi claim graph metrics.
5. **Evaluation chuyển sang primary endpoint + cluster bootstrap**, tách placement/articulation, top-1/oracle và ranking/calibration.
6. **Tên CoSign4D được gắn author decision** vì có CoSIGN gần về không gian kỹ thuật.

## 8. Unresolved questions cho vòng review kế tiếp

1. File evaluator thực tế có đúng phép biến đổi được mô tả không, và metric repair thay đổi ranking baseline thế nào?
2. SGNify/DexAvatar-related assets cho phép annotation hoặc phát hành contact labels ở mức nào?
3. Có multi-view/3D reference nào đủ để tạo gold contact subset, hay chỉ có monocular RGB?
4. Goliath-SC/PAPoseDiff assets và license có thể dùng cho pretraining/baseline không?
5. `C_t` cần patch resolution nào để vừa đủ ngữ nghĩa sign mà vẫn annotatable?
6. Có bao nhiêu signer/sign sequence; signer-independent statistical power có đủ không?
7. Camera và root alignment hiện tại có làm placement error bị che khuất ở mức body–camera không?
8. Semantic evaluation nhắm German isolated signs hay một benchmark khác?
9. Tác giả muốn đóng góp chính là metric audit, contact representation hay full generative inference?
10. Tên phương pháp cuối cùng là gì?

## 9. Recommended evidence package cho second pass

- evaluator + unit tests + commit hash;
- dataset/split manifest;
- 20–50 clip pilot với double-annotated contact events;
- B0/B2/B3/B4 pilot results, per-sequence outputs và compute-matched configs;
- render protocol cho semantic evaluation;
- đầy đủ hyperparameter/sampler table;
- một trang claim–evidence matrix đã được tác giả xác nhận.

Khi các artifact này có mặt, vòng review thứ hai có thể chuyển các marker từ `UNVERIFIED` sang `VERIFIED`, chốt primary endpoint và quyết định liệu full CoSign4D story có đủ bằng chứng để nộp hay nên thu hẹp thành metric/contact-mechanism paper.
