# CUSP-SL: Calibrated Uncertainty-Triggered Selective Pose Reconstruction for Sign Language

**Research topic:** monocular 3D sign-language reconstruction
**Target benchmark:** SGNify, using the DexAvatar author-supplied TR-V2V evaluator and central-sign metadata
**Document status:** design specification and pre-implementation protocol; **not** an empirical claim
**Audit date:** 19 August 2026
**Primary source snapshot:** DexAvatar repository commit `a0dfd427f60f5811aadb35c8657b3856d47f56b5`

> This document deliberately separates (i) facts verified in the attached paper, repository, and evaluator; (ii) adaptations of prior work; and (iii) proposed CUSP-SL components. No statement in this document establishes state of the art or superiority over DexAvatar. Every unresolved implementation or protocol choice is explicitly marked.

## Executive methodological verdict

The proposed study is a **quantitative, experimental, comparative machine-learning study** with a developmental phase (model construction and validation) and a locked confirmatory benchmark phase (SGNify test evaluation). Its current scientific objective is not yet an achieved result: it is to test whether selective, uncertainty-triggered temporal residual reconstruction and a sign-form-sensitive candidate selector improve reconstruction relative to matched deterministic controls under the same official evaluation protocol.

The revised pipeline is technically coherent at the level of interfaces and objectives, but it is **not yet ready for full implementation or a conference claim**. Four blockers remain: (1) the attached evaluator's frame-count semantics must be resolved with the authors; (2) paired training videos and SMPL-X annotations must be obtained and versioned; (3) the MANO-to-SMPL-X coordinate adapter must be implemented and round-trip tested; and (4) the counterfactual scorer requires linguistically validated perturbations and an available or reproducible sign-video encoder. The appropriate final status is therefore **Cần chỉnh sửa lớn trước khi triển khai**.

---

# A. Kết quả xử lý các nhận xét

## A.1. Xác định thiết kế nghiên cứu sau hiệu chỉnh

| Hạng mục | Xác định có căn cứ | Nội dung chưa được phép suy đoán |
|---|---|---|
| Loại hình | Định lượng (quantitative). | Không có thành phần định tính trong nghiên cứu lõi. Nếu bổ sung đánh giá bởi người dùng/người ký hiệu ngôn ngữ, phải thiết kế thành nghiên cứu riêng. |
| Thiết kế | Nghiên cứu phát triển phương pháp và thực nghiệm đối chứng (method-development and controlled comparative benchmark study), có ablation và kiểm định trên holdout cố định. | Chưa có preregistration, primary endpoint cuối cùng, hay ngưỡng thành công được khóa. |
| Mục tiêu | Xây dựng và kiểm tra một phương pháp tái dựng 3D upper-body và hands từ monocular sign video, so sánh công bằng với DexAvatar và các control mạnh trên SGNify. | Không được viết “outperforms” hoặc “SOTA” trước khi có kết quả theo protocol đã khóa. |
| Câu hỏi chính | Với cùng dữ liệu đầu vào, backbone, frame manifest và evaluator, CUSP-SL có giảm TR-V2V so với DexAvatar reproduction và `SMPLer-X + WiLoR` hay không? | `[CẦN TÁC GIẢ XÁC NHẬN: metric/vùng nào là primary endpoint và tiêu chí thắng đa vùng]`. |
| Giả thuyết kỹ thuật | H1: lỗi của strong frontend tập trung ở các joint/window không đáng tin cậy; H2: residual distribution theo thời gian bao phủ nghiệm đúng tốt hơn point estimate; H3: scorer nhạy với sign form chọn candidate tốt hơn geometry-only selector. | Các giả thuyết này chưa được xác nhận; phải có kill criteria và ablation. |
| Đối tượng nghiên cứu | Video đơn mắt chứa sign language; đơn vị huấn luyện là clip/window; đơn vị đánh giá chính là sign sequence và frame/vertex theo evaluator. | Chưa xác định đầy đủ ngôn ngữ, signer, license và phiên bản của tập huấn luyện. |
| Input | Chuỗi RGB có timestamp/frame ID; 2D evidence và initial SMPL-X/MANO estimates từ frozen frontends. | Resolution, FPS, crop policy, color preprocessing và detector checkpoints. |
| Output | Chuỗi SMPL-X mesh/pose cho upper body và hai tay, một selected candidate, candidate disagreement, coverage và failure flags. | Không có output face được tinh chỉnh trong v1; face/expression chỉ truyền qua từ frontend. |
| Biến độc lập | Method variant, gate, candidate count, selector type, corruption/occlusion stratum. | Mức/cấu hình chính xác phải khóa trên validation. |
| Biến phụ thuộc | Official TR-V2V cho UBody(-F), LHand, RHand; secondary per-sign effects, coverage, runtime, temporal and calibration diagnostics. | Composite endpoint hoặc non-inferiority margin chưa được xác định. |
| Reporting standard | Checklist tái lập của conference/ML venue, model/data documentation và complete experiment manifest; CONSORT/STROBE/PRISMA/COREQ/TRIPOD không phù hợp với benchmark Computer Vision thuần túy này. Nếu thêm human perceptual study, phải dùng protocol và reporting checklist riêng phù hợp. | `[CẦN TÁC GIẢ XÁC NHẬN: conference cụ thể để áp đúng checklist và policy về external data]`. |

## A.2. Bảng quyết định đối với từng nhận xét review

| STT | Nhận xét review | Quyết định | Lý do | Cách xử lý trong bản mới |
|---:|---|---|---|---|
| 1 | Objective mới chỉ nói “vượt DexAvatar”, chưa có research question và endpoint khóa trước. | Chấp nhận | Một mục tiêu xếp hạng không đủ định nghĩa thiết kế hay falsification. | Viết lại thành ba giả thuyết cơ chế; để primary endpoint và success rule ở placeholder bắt buộc. |
| 2 | Pipeline chủ yếu ghép module có sẵn, novelty chưa tách khỏi engineering integration. | Chấp nhận | SMPLer-X + WiLoR và geometric fusion đã được nghiên cứu trực tiếp trong Tamaththul3D; backbone swap không còn là đóng góp mới. | Gắn provenance cho từng module. Chỉ coi interface `Q + gated residual G + frozen counterfactual S` là proposed research mechanism, có điều kiện qua ablation. |
| 3 | Stronger hand frontend có thể giải thích toàn bộ gain. | Chấp nhận | Đây là confounder trung tâm và có control rẻ. | Bắt buộc baseline `SMPLer-X + WiLoR direct retargeting`, cùng cached outputs cho mọi ablation. |
| 4 | Chưa định nghĩa dataset huấn luyện, split, data lineage và chống leakage. | Chấp nhận | Không thể đánh giá validity hoặc tái lập nếu thiếu data manifest. | Dùng SignAvatars làm nguồn paired video/pseudo-SMPL-X có điều kiện tiếp cận; split theo signer và source video trước khi cắt window; SGNify chỉ dùng test. |
| 5 | “Cỡ mẫu” chưa được tính. | Chấp nhận một phần | Benchmark SGNify là tập cố định; power calculation kiểu clinical trial không tạo thêm sign. Tuy nhiên precision và tính ổn định vẫn phải báo. | Báo toàn bộ 57 sign, cluster bootstrap theo sign, paired effects và external validation; không giả tạo một phép tính cỡ mẫu không phù hợp. |
| 6 | Định nghĩa reliability (q) và chiều của gate không nhất quán. | Chấp nhận | Nếu (q) lúc là uncertainty, lúc là probability of correctness thì gate đảo nghĩa. | Định nghĩa duy nhất (q=P(e\le e_0\mid f^q)); (q) cao nghĩa là đáng tin cậy; gate correction giảm đơn điệu theo (q). |
| 7 | Residual target đã nhân gate rồi candidate assembly lại nhân gate (“double gating”). | Chấp nhận | Làm co residual hai lần và làm sai target distribution. | Training dùng corruption mask (m); flow học residual không gate. Inference chỉ áp (g) đúng một lần trong (R^0\operatorname{Exp}(g\delta)). |
| 8 | Learned relation/contact head thiếu labels và calibration plan. | Chấp nhận | Không có dataset sign-specific contact/depth đã xác nhận; một head như vậy chưa thể tái lập. | Loại khỏi core v1. Chỉ giữ deterministic ROM/penetration validity energy; learned contact là extension riêng sau khi có annotation. |
| 9 | “Posterior” từ softmax của handcrafted energy chưa được calibrate. | Chấp nhận | Energy weights không tự tạo xác suất hậu nghiệm. | Đổi tên thành **energy-normalized candidate weights** π; uncertainty là candidate disagreement, không tuyên bố Bayesian posterior. |
| 10 | Semantic scorer có thể học gloss/signer/background shortcut và không nhạy với 3D articulation. | Chấp nhận | Đây là nguy cơ validity nghiêm trọng. | Đổi thành **form-consistency scorer**; train bằng same-video, one-factor, physically valid counterfactuals; signer/source-disjoint validation và per-axis probes; freeze trước khi train/đánh giá generator. |
| 11 | Dùng “SignDINO-like” là không tái lập và không xác định checkpoint. | Chấp nhận | Tên gần giống paper không định nghĩa model hay weights. | Chỉ cho hai nhánh: checkpoint chính thức có checksum/license, hoặc reproduction theo paper với config công khai. Nếu không có, bỏ S khỏi core experiment. |
| 12 | Architecture size, learning rate, thresholds và search grid trong draft là số đề xuất không có căn cứ. | Chấp nhận | Các số này chưa được benchmark và có nguy cơ trở thành fabricated implementation details. | Thay toàn bộ bằng placeholders; yêu cầu khóa bằng validation protocol và lưu search budget. |
| 13 | Chưa tương thích rõ giữa MANO hand output và SMPL-X body/camera. | Chấp nhận | Hai model có local frames, handedness, shape và camera conventions khác nhau. | Thêm explicit retargeting contract, SE(3) wrist alignment, rest-pose/chirality/round-trip tests; không chuyển MANO shape vào SMPL-X nếu chưa có calibrated map. |
| 14 | Face được nhắc như output nhưng không có modeling/loss/evaluation. | Chấp nhận | Tạo overclaim “whole-body/holistic” trong khi benchmark chính loại face. | V1 chỉ tinh chỉnh selected upper-body and hand rotations; copy expression/jaw/face từ SMPLer-X và nêu limitation. |
| 15 | Temporal alignment và missing frontend output chưa có contract. | Chấp nhận | Index-based pairing có thể dịch frame; residual model không tồn tại nếu thiếu base pose. | Mọi record dùng immutable `(video_id, frame_id, timestamp)`; no silent drop. Complete base failure dẫn tới abstention trong v1, không hallucinate output. |
| 16 | Tổng loss chỉ liệt kê tên, chưa có công thức và domain áp dụng. | Chấp nhận | Không biết loss tác động module nào, dữ liệu nào hoặc giả định nào. | Viết công thức riêng cho `Q`, `G`, `S`, endpoint geometry và candidate energy; staged training, không giả định end-to-end. |
| 17 | Factorial ablation trước đây không đầy đủ và không tách coverage khỏi selection. | Chấp nhận | Không thể quy gain cho sampling hay semantic selection. | Thiết kế control dùng cùng candidate set: top-1, oracle, random, geometry-only, form-only, combined; thêm matched-compute deterministic restarts. |
| 18 | Có nguy cơ tune trên SGNify test; DexAvatar supplement còn mô tả chọn hyperparameter trên DEV và TEST. | Chấp nhận | Test-informed model selection làm invalid confirmatory comparison. | Cấm dùng SGNify GT, signs/classes hoặc test scores để train/tune. Chạy test sau khi config/hash khóa; mọi exploratory test exposure phải khai báo. |
| 19 | Statistical plan thiếu effect size, CI, seed policy và unit of resampling. | Chấp nhận | Một mean duy nhất trên benchmark nhỏ không cho biết uncertainty hay sign-level stability. | Official global mean vẫn báo để so sánh; thêm paired per-sign deltas, cluster bootstrap CI, seed distribution và sensitivity to missing frames. |
| 20 | Ethical/data-governance reporting còn thiếu. | Chấp nhận một phần | Core study dùng secondary data, nhưng sign-language data chứa identity, culture và licensing constraints. | Ghi license/consent provenance, access restrictions, signer-disjoint splits, non-release of identifiable frames, language scope và không suy rộng ASL/DGS tùy tiện. |
| 21 | Reproducibility package chưa đủ. | Chấp nhận | Repo DexAvatar không chứa evaluator/prior-training source đầy đủ; draft mới phải nâng chuẩn. | Bắt buộc environment lock, model/data hashes, manifest, seeds, exact evaluator track, audited track, unit tests và failure ledger. |
| 22 | Có thể gọi evaluator đính kèm là “official” và dùng ngay không cần audit. | Không chấp nhận | Nguồn từ authors làm nó phù hợp cho comparability, nhưng code vẫn có hành vi cần xác nhận; provenance không thay thế validation. | Giữ immutable author-supplied track, đồng thời có audited track; không sửa rồi gọi kết quả là cùng protocol. |
| 23 | Nên giữ learned contact factor vì sign language phụ thuộc contact. | Chưa đủ thông tin để quyết định | Tầm quan trọng ngôn ngữ không tự cung cấp label hoặc detector đáng tin cậy. | Chỉ tái đưa vào khi có annotation protocol, inter-rater agreement, held-out calibration và ablation contact subset. |
| 24 | Có thể tuyên bố novelty vì chưa thấy exact combination. | Không chấp nhận | Literature search không chứng minh tuyệt đối tính mới; novelty còn phụ thuộc empirical interaction. | Gọi là “proposed integration mechanism”; claim novelty chỉ sau prior-art update và ablation cho thấy không phải additive engineering. |

## A.3. Những ràng buộc rút ra trực tiếp từ DexAvatar

| Evidence đã kiểm tra | Diễn giải phương pháp đúng | Hệ quả đối với CUSP-SL |
|---|---|---|
| DexAvatar khởi tạo body/camera/shape bằng SMPLer-X, hands bằng HaMeR và body 2D keypoints bằng Sapiens, sau đó tối ưu SMPL-X với sign-specific priors. | DexAvatar là optimization-based refinement quanh strong initial estimates, không phải một video-to-mesh network huấn luyện end-to-end. | Reproduction phải giữ đúng initializer/config; CUSP phải tách gain của frontend khỏi learned refinement. |
| Released YAML dùng LBFGS, ba stages, `data_3d_weights = 0`, và các initializer-target weights bằng 1,200 ở cả ba stages. | Trong released configuration, initial body/hand estimates là tether rất mạnh; 3D supervision term được tắt. | Không được mặc định rằng reported gain chỉ do SignBPoser/SignHPoser; cần matched-initializer controls. |
| Paper mô tả SignHPoser được huấn luyện từ mocap riêng (8 signers, 93 fingerspelling words) và SignBPoser từ filtered SignAvatars/How2Sign-derived data. | Priors có sign-specific data, nhưng training code/data split/checkpoint provenance không đầy đủ trong repo đã audit. | Reproduction uncertainty phải được báo; CUSP không thể coi prior-training pipeline này là fully reproducible. |
| Paper's Supplement S4 states that best hyperparameters are selected on “DEV and TEST”. | Nếu TEST ở đây là benchmark test used for final claims, đây là test-informed model selection; nếu là internal prior-test split, wording vẫn cần phân biệt rõ. | CUSP cấm SGNify test-informed tuning và phải định danh từng split theo dataset/source. |
| DexAvatar reports 57 SGNify signs and 2,872 central frames; attached metadata satisfy \(\sum(end-start)=2872\), while supplied code doubles endpoints and iterates inclusively. | Paper text, metadata arithmetic and evaluator implementation do not yet define a unique frame set. | Direct score comparison is blocked until expected frame IDs are confirmed and DexAvatar is reproduced. |
| DexAvatar Supplement S6 notes implausible hand configurations in some SGNify ground-truth meshes. | TR-V2V measures agreement with that reference, not anatomical or linguistic correctness in isolation. | Physical/form metrics are secondary diagnostics; they cannot replace official TR-V2V post hoc. |

---

# B. Pipeline sau chỉnh sửa

## B.1. Luồng tổng quát

`Monocular RGB video + immutable frame IDs → deterministic preprocessing → frozen SMPLer-X/WiLoR reconstruction → tested MANO–SMPL-X retargeting → calibrated reliability gate → selective temporal residual candidates → geometric/physical checks + frozen sign-form scoring → selected SMPL-X sequence + disagreement/failure flags → exact-author and audited SGNify evaluation`

## B.2. Contract input–operation–output

| Bước | Input | Operation | Output | Trạng thái bằng chứng |
|---:|---|---|---|---|
| 1. Ingestion | RGB video `I[1:T]`, original timestamps/frame IDs | Decode without silent dropping; write immutable manifest | Ordered frames and metadata | Proposed reproducibility requirement |
| 2. Preprocessing | Frames and metadata | Body/hand detection, resize/crop, 2D keypoints and confidence; record missingness | Image tensors, crops, `k[t,j]`, `c[t,j]`, detector status | Exact tools/checkpoints `[CẦN TÁC GIẢ XÁC NHẬN]` |
| 3. Body frontend | Full frame/crop | Frozen SMPLer-X | SMPL-X body rotations, global orientation, shape, expression and camera state | Inherited, not novel |
| 4. Hand frontend | Hand crops | Frozen WiLoR | Per-hand MANO pose/orientation/shape and camera/translation metadata | Inherited, not novel |
| 5. Retargeting | SMPL-X body + MANO hands | Convert units, handedness and joint frames; align wrist/forearm; retain SMPL-X shape | Common SMPL-X base Θ⁰ and validity flags | Adapted engineering; mandatory tests |
| 6. Reliability | Base output, detector scores, reprojection and consistency features | Calibrated model `Q` estimates probability that each base joint is within a predeclared tolerance | `q[t,j]`, correction gate `g[t,j]`, ambiguous windows | Proposed component |
| 7. Residual generation | Active windows, base rotations, video/2D context, `q,g` | Conditional rectified flow samples ungated tangent-space residuals; gate applied once during assembly | `K` valid SMPL-X pose candidates | Adapted generative mechanism + proposed selective interface |
| 8. Candidate checks | Candidate meshes and observations | 2D reprojection, visible-motion consistency, ROM and penetration checks | Standardized energy terms and validity mask | Adapted deterministic evaluation |
| 9. Form scoring | Same input video + pose candidates | Frozen video–pose scorer trained with validated one-factor counterfactuals | Candidate form-consistency scores | Proposed; conditional on data/checkpoint feasibility |
| 10. Selection | Valid candidates and validation-fitted energies | Select minimum-energy candidate; compute energy-normalized weights and disagreement | Final pose/mesh, uncertainty diagnostic, failure flag | Proposed structured inference |
| 11. Evaluation | Per-frame predicted meshes + locked manifests | Exact-author evaluator for comparability; audited evaluator for robustness | Three official TR-V2V values plus coverage and paired diagnostics | Two tracks must remain separate |

## B.3. Tensor and representation contract

Let batch size be \(B\), clip length \(T\), image height/width be \(H,W\), and the set of refined joints be \(\mathcal J_R\) with cardinality \(J_R\). Dimensions not fixed by a source are not guessed.

| Quantity | Required structure | Coordinate/frame | Known / unresolved |
|---|---|---|---|
| RGB input `I` | `B × T × 3 × H × W` after preprocessing | Pixel/color convention of selected checkpoint | `H,W`, normalization and FPS `[CẦN TÁC GIẢ XÁC NHẬN]` |
| Base rotations `R0` | `B × T × J_R × 3 × 3` | SMPL-X local parent frames; global root separately | Joint list and rotation convention must be serialized |
| Base shape β | `B × d_beta`, constant per clip/signer where justified | SMPL-X shape space | `d_beta` follows model asset; estimation policy must be fixed |
| Expression ψ | `B × T × d_psi` | SMPL-X/FLAME | Pass-through only; not refined or claimed |
| Camera γ | Per-frame structure required by projection Π | Explicit weak-/full-perspective convention | `[CẦN TÁC GIẢ XÁC NHẬN: model and units]` |
| 2D joints `k` | `B × T × J_2D × 2` plus confidence `c` | Original image pixels after inverse crop transform | Mapping to SMPL-X joints required |
| Reliability `q,g` | `B × T × J_R` | Scalar probability and scalar edit gate | \(q,g\in[0,1]\) |
| Residual δ | `B × T × J_R × 3` | Tangent space of each \(R^0_{t,j}\in SO(3)\) | Exp/Log convention must pass unit tests |
| Mesh `V` | `B × T × 10475 × 3` for the referenced neutral SMPL-X topology | Explicit model/camera or pelvis frame | Exact asset/checksum and units required |
| Candidate set | `B × K × T × J_R × 3` residuals or rotations | Same as base | `K` selected on validation and reported |

## B.4. Module provenance and integration risk

| Module | Paper/source | Chức năng gốc | Cách dùng trong nghiên cứu | Thay đổi cần thiết | Rủi ro tích hợp |
|---|---|---|---|---|---|
| SMPL-X | Pavlakos et al.; official model | Unified parametric body, hands and face | Common output mesh and kinematic graph | Fix topology, asset version, units and joint list | License/assets; vertex-map mismatch |
| SMPLer-X | Moon et al., NeurIPS 2023 | Expressive whole-body pose/shape recovery | Frozen body/global/camera/shape initializer | Export complete metadata and original-frame projections | Domain shift; hand quality; camera convention |
| WiLoR | Potamias et al., CVPR 2025 | In-the-wild MANO hand detection and reconstruction | Frozen detailed hand initializer | Handedness, crop-to-image, MANO-to-SMPL-X retargeting | Mirror errors, wrist-frame mismatch, inconsistent scale |
| Direct hand–body fusion | Tamaththul3D, 2026 preprint | Fuse SMPLer-X and WiLoR with geometric alignment | Mandatory strong baseline and starting point | Independently reproduce exact transformation or implement a tested alternative | Cannot be claimed as novelty; source code/metric equivalence uncertain |
| Sign-aware static priors | DexAvatar, WACV 2026 | VAE body/hand priors inside optimization | Baseline to reproduce; not inserted automatically into CUSP | Same official config/checkpoints; report unavailable training code | Fixed prior may conflict with stronger frontend; slow fitting |
| Conditional rectified flow | HandFlow and rectified-flow literature | Generate temporally coherent MANO trajectories | Adapted to joint upper-body/bimanual **residuals around a frozen SMPL-X base** | New tokenization, residual target, identity gate, common body-hand graph | Recent work; code/weights availability; mode collapse; compute |
| Confidence/masked generation | HandFlow; MaskHand/MMHMR | Handle uncertain or masked hand evidence | Mechanistic inspiration for training corruptions and conditions | Calibrated error probability and separate train/inference masks | Detector confidence may be miscalibrated |
| Sign-video encoder | SignDINO, CVPR 2026 | Self-supervised sign representation for gloss-free translation | Conditional initialization of frozen video tower | Official checkpoint+license or exact reproduction | May be unavailable; may encode signer/background rather than 3D form |
| Paired 3D sign data | SignAvatars, ECCV 2024 | Pseudo-SMPL-X/MANO annotations for large sign corpora | Candidate training source for `Q,G,S` | Re-associate licensed source videos, QC, source/signer split | Pseudo-label confirmation bias; heterogeneous languages |
| Phonetic/form metadata | ASLLVD; 3D-LEX | Lexical/form attributes and 3D pose resources | Validate counterfactual axes and form probes, if licensed and mappable | Retarget/ontology map and Deaf/sign-language expert review | Cross-language non-equivalence; representation mismatch |
| Learned contact head | No confirmed source/data for this exact study | Predict contact/depth relation | **Removed from core v1** | Require labels, annotation protocol and calibration before reintroduction | Sticky false contact and linguistic distortion |

## B.5. Compatibility verdict by interface

| Interface | Verdict | Required adapter/test |
|---|---|---|
| WiLoR MANO → SMPL-X hands | Compatible only conditionally | Fixed rest-pose basis conversion; chirality tests; wrist global transform; joint-order map; synthetic pose round trip; overlay audit. |
| Per-frame frontends → temporal flow | Compatible | Timestamp-keyed cache, missingness tokens, clip sampling without frame shifts, window boundary tests. |
| Flow residual → SMPL-X layer | Compatible | Stable \(SO(3)\) Exp/Log, geodesic tests near π, one-time gating, differentiable SMPL-X forward. |
| Video encoder → pose generator | Technically compatible but not required | Projection layer dimensions `[CẦN TÁC GIẢ XÁC NHẬN]`; keep frozen; ablate to test contribution. |
| Video–pose scorer → candidate selection | Compatible only after validity tests | Counterfactual sensitivity, shortcut probes, validation-only normalization, same candidate sets across selectors. |
| Learned contact → candidate selection | Not currently justified | Remove from core; deterministic penetration/ROM only. |
| Output meshes → attached evaluator | Not yet proven | Exact topology/face order, units in metres, name/frame manifest, complete output counts and NaN tests. |

## B.6. Audit of the attached SGNify evaluator

Reference artifacts:

- `evaluate_new_fitting.py`: SHA-256 `400bfbd736fc59fcc1867af7650188b61772136982f64b623df31494e6116877`.
- `segment.json`: SHA-256 `e5d9bd504ef8158695a09d2fa279ba33bccda78d43a9575435ee838223d1aac2`.
- `signs.txt`: SHA-256 `bc5b0da75a3af8f6ecf6914bd83d5ce55faba698ab08217fbd0e3979a58d596e`.
- The attached segment/class files are byte-identical to the files in the audited DexAvatar repository snapshot.
- The supplied evaluator itself was not found in that repository snapshot; it is therefore treated as a separately author-supplied artifact whose provenance and expected outputs still require confirmation.

| Finding in supplied code | Consequence | Required treatment |
|---|---|---|
| `segment.json` has 57 signs and \(\sum(end-start)=2872\), matching the paper's stated frame total only under an end-exclusive interpretation. | Metadata alone does not specify whether endpoints are inclusive or whether indices refer to 25/50 fps. | Obtain author confirmation and an expected frame manifest. |
| Lines 239–248 multiply both endpoints by two and iterate inclusively. | Depending on which GT filenames exist, the code can select 2,929 even-index meshes or up to 5,801 consecutive meshes, not unambiguously 2,872. | `[CẦN TÁC GIẢ XÁC NHẬN: exact GT filenames/FPS and expected evaluated count per sign]`. |
| Missing GT frame numbers are silently omitted; prediction meshes are indexed by the resulting list position (lines 247–249, 342–361). | A missing GT mesh can shift all later prediction–GT pairs; names/timestamps are not matched. | Exact-author track preserves behavior but records manifest; audited track matches explicit IDs and asserts equality. |
| Prediction length/name equality is not asserted. | Extra predictions are ignored; too few predictions may fail late; wrong ordering may yield a valid-looking number. | Add preflight checks in audited evaluator. |
| `transl_point_error` centers each vertex subset independently (lines 159–169, 380–395). | “TR” removes a separate centroid for each reported region, including each hand; it is not one global/pelvis translation. | Describe this exact alignment and do not equate it with PA-MPVPE or global-TR metrics. |
| NaN prediction meshes are skipped (lines 364–366). | Failures improve denominator-conditioned mean and coverage is hidden. | Preserve for comparability only; always report NaN/missing coverage; audited track marks failure explicitly. |
| For class `0`, left hand is excluded and left-hand vertices are also removed from all other regions (lines 380–395). | Region membership and denominators depend on supplied sign class. There are 15 class-`0` and 42 class-`~0` signs. | Freeze class file/hash; report behavior; do not infer class from test videos. |
| Final score concatenates every per-frame per-vertex error then takes one mean (lines 455–461). | It is vertex- and frame-weighted, not an equal-weight mean across 57 signs. | Report official aggregate plus secondary per-sign paired distribution/CI. |
| `--central` is parsed but not used to choose a branch; segment loading always occurs. | Command-line label does not change behavior. | Record this in exact track; test explicit audited behavior. |
| Paths to SMPL-X assets and region arrays are hard coded (lines 521–571). | Script is not portable or hash-stable after ordinary path edits. | Archive immutable source; make a path-only wrapper/patch with diff and numerical equivalence tests. |
| The script logs means but does not save a result manifest. | Results cannot be fully audited after the run. | Capture stdout, environment, input hashes, frame-level errors and coverage in a separate wrapper. |

**Protocol decision.** The paper must report two clearly labeled result tracks:

1. **Author-comparability track:** functionally preserve the supplied computations and sign metadata, with only documented path externalization. This is the only track used for direct comparison to the DexAvatar table, subject to reproducing the reference score and resolving the frame manifest.
2. **Audited track:** explicit `(sign, frame_id)` matching, strict topology/unit assertions, no silent omission, complete failure accounting, saved per-frame errors, and per-sign summaries. These numbers must never be presented as if they were generated by the unchanged author protocol.

---

# C. Các quyết định kỹ thuật quan trọng

1. **Narrow the modeled output.** CUSP-SL v1 refines upper-body and two-hand rotations only. Shape, camera, global state and face/expression are inherited from the deterministic frontend unless a separately specified adapter changes the wrist/forearm chain. This matches the actual benchmark target and prevents an unsupported holistic-face claim.

2. **Make the strongest simple integration a baseline.** SMPLer-X + WiLoR retargeting is required before any learned CUSP component. It is engineering integration and prior art, not novelty.

3. **Use a residual, not full-pose generation.** Generating (SO(3)) residuals around a strong base lowers the learning burden and makes exact identity possible on reliable joints. This is an adaptation of generative hand-recovery ideas, not a claim that rectified flow itself is new.

4. **Separate training corruption from inference reliability.** A binary/continuous training mask (m) determines supervised residual locations; calibrated gate (g) controls deployed editing. This removes the prior draft's double-gating error.

5. **Treat complete frontend failure honestly.** Without (R^0), a residual model has no reference. V1 abstains and reports a failure. Full detector-dropout completion would require a separate absolute-pose branch and is outside the current implementable claim.

6. **Remove the learned contact head from the core.** ROM and penetration checks remain deterministic candidate energies. Positive contact prediction is deferred until labels and calibration exist.

7. **Rename “semantic posterior.”** The scorer is a frozen video–pose **form-consistency** model, and the softmax outputs are energy-normalized candidate weights, not calibrated posteriors. Cross-language meaning is not assumed.

8. **Stage training and freeze interfaces.** Frontends are frozen; \(S\) is trained then frozen; \(Q\) is trained/calibrated; \(G\) is trained afterward; energy weights are fitted only on validation. This limits gradient gaming and makes module attribution testable.

9. **Use two evaluator tracks.** Exact-author comparability and audited validity answer different questions. The manuscript will not mix them.

10. **Make novelty falsifiable.** The research contribution survives only if the learned residual candidates have an oracle advantage over deterministic controls and the form scorer selects them better than geometry-only/random selection using identical candidates and matched compute.

---

# D. Revised Methods

> The following section is written in academic English for direct adaptation into a conference manuscript. Bracketed placeholders are intentional and must remain until verified.

## 3. Method

### 3.1. Problem Formulation and Scope

Given a monocular RGB sign-language video \(I_{1:T}=\{I_t\}_{t=1}^{T}\), with immutable frame identifiers and timestamps, our goal is to recover a temporally coherent sequence of SMPL-X meshes for the signer's upper body and hands. Each frame is represented by

\[
\Theta_t=\left(\mathbf R^{B}_t,\mathbf R^{L}_t,\mathbf R^{R}_t,
\boldsymbol\beta,\boldsymbol\psi_t,\boldsymbol\gamma_t\right),
\]

where \(\mathbf R^{B}_t\) denotes the selected upper-body joint rotations, \(\mathbf R^{L}_t\) and \(\mathbf R^{R}_t\) denote the left- and right-hand joint rotations, \(\boldsymbol\beta\) is the SMPL-X shape vector, \(\boldsymbol\psi_t\) contains facial-expression parameters, and \(\boldsymbol\gamma_t\) contains the global and camera variables required by the projection model. All rotations are represented internally as matrices in \(SO(3)\); alternative parameterizations are used only as network inputs and are converted back to rotation matrices before kinematic evaluation.

The SMPL-X forward model is denoted by

\[
\mathbf V_t,\mathbf J_t
=\mathcal M_{\mathrm X}(\Theta_t),
\]

where \(\mathbf V_t\in\mathbb R^{N_V\times3}\) is the posed mesh and \(\mathbf J_t\in\mathbb R^{N_J\times3}\) is the corresponding joint set. For the model topology used by the supplied evaluator, \(N_V=10{,}475\). `[CẦN TÁC GIẢ XÁC NHẬN: exact SMPL-X asset, gender setting, model version, joint regressor, and checksums]`.

The first version of our method refines only the rotation subset

\[
\mathcal J_R=\mathcal J_{\mathrm{upper}}\cup
\mathcal J_{\mathrm{left\ hand}}\cup
\mathcal J_{\mathrm{right\ hand}}.
\]

Shape, facial expression, jaw/eye pose, global state, and camera parameters are copied from a deterministic frontend, except for any explicitly documented wrist/forearm coordinate alignment. We therefore do not claim that the proposed component reconstructs non-manual facial signals. This scope is aligned with the primary SGNify regions, namely upper body excluding the face, left hand, and right hand.

The method returns (i) a selected SMPL-X sequence \(\widehat\Theta_{1:T}\), (ii) the generated candidate sequences for edited windows, (iii) energy-normalized candidate weights, (iv) joint/window disagreement scores, and (v) coverage and failure flags. These weights are diagnostic quantities and are not described as calibrated Bayesian posterior probabilities.

### 3.2. Method Overview and Scientific Provenance

Our pipeline, termed **CUSP-SL** (**C**alibrated **U**ncertainty-triggered **S**elective **P**ose reconstruction for **S**ign **L**anguage), starts from a strong deterministic body–hand reconstruction and edits it only where the initial estimate is predicted to be unreliable. The deterministic frontend combines frozen SMPLer-X body estimates with frozen WiLoR hand estimates after an explicitly tested MANO-to-SMPL-X retargeting step. This frontend and its geometric fusion are inherited engineering components, not a scientific contribution of CUSP-SL \cite{cai2023smplerx,potamias2025wilor,alghamdi2026tamaththul3d}.

For each frame and refinable joint, a calibrated reliability model estimates the probability that the base rotation lies within a prespecified error tolerance. Contiguous low-reliability tokens activate a temporal conditional-flow model that samples rotation residuals around the base reconstruction. Candidate sequences are checked against the available image evidence and deterministic physical constraints. A separately trained and frozen video–pose form-consistency model may then distinguish candidates that are geometrically similar but differ in signing-relevant handshape, orientation, location, movement, or bimanual relation. The selected residual is composed with the base rotation exactly once. High-reliability tokens follow an architectural identity path and remain unchanged.

The provenance of the method is as follows:

- **Used without conceptual modification:** SMPL-X as the output representation and MANO as the source hand representation; frozen SMPLer-X and WiLoR as initial estimators, subject to using their official implementations and checkpoints \cite{pavlakos2019smplx,romero2017mano,cai2023smplerx,potamias2025wilor}.
- **Adapted from prior work:** body–hand coordinate fusion and temporal conditional rectified flow with confidence-conditioned evidence \cite{alghamdi2026tamaththul3d,xu2026handflow}. The exact ROM and penetration formulations remain `[CITATION NEEDED]` until the implementation is selected.
- **Proposed in this study:** the calibrated identity-preserving interface that triggers a joint upper-body/bimanual residual distribution only on unreliable tokens, together with a frozen counterfactual form-consistency selector evaluated on identical candidate sets. This is a proposed research contribution, not an established novelty claim until the literature audit and ablations are complete.

### 3.3. Data Sources, Partitions, and Data Governance

#### 3.3.1. Development data

The conditional residual model and form-consistency scorer require paired RGB video and time-aligned 3D pose. The primary candidate source is SignAvatars, which reports SMPL-X and MANO annotations for multiple sign-language video corpora \cite{yu2024signavatars}. Its annotations are automatically reconstructed and are therefore treated as **pseudo-ground truth**, not motion-capture truth. Use of this source is conditional on obtaining the annotations under their license and separately obtaining each source video from its authorized distributor.

`[CẦN TÁC GIẢ XÁC NHẬN: SignAvatars access approval, annotation release/version, source subsets, source-video availability, license compatibility, and exact usable clip/frame counts]`.

Before extracting temporal windows, we will construct an immutable source-level manifest containing at least `dataset`, `source_video_id`, `signer_id` where available, `language`, `clip_id`, `frame_id`, `timestamp`, annotation provenance, and checksum. Splits will be made at the signer level when signer identity is available and at the original source-video level in all cases. No frames or overlapping windows from the same source video may occur in more than one split. If a source lacks a reliable signer identifier, its clips will not be used to support signer-independent claims.

Clean 3D supervision is desirable for calibrating the reliability target and for quantifying pseudo-label bias. Potential sources include 3D-LEX or an independently licensed motion-capture set, but these data may not share the SMPL-X parameterization \cite{ranum2024threeDLEX}. Such data will be used only after a documented retargeting procedure and a held-out retargeting-error analysis.

`[CẦN TÁC GIẢ XÁC NHẬN: clean 3D/mocap dataset, language, participant split, skeleton/mesh representation, calibration accuracy, and permission for the proposed use]`.

ASLLVD or comparable lexicons may provide articulatory labels for validating counterfactual factors [CITATION NEEDED: ASLLVD]. These labels are not assumed to transfer semantically across ASL, DGS, or other sign languages. Cross-language data may teach geometric form factors, but language-specific lexical meaning is outside the claimed scope unless independently evaluated.

#### 3.3.2. SGNify benchmark isolation

SGNify is reserved for final evaluation and is not used to train, calibrate, select thresholds, choose energy weights, construct counterfactuals, or select checkpoints \cite{forte2023sgnify}. The supplied metadata contain 57 German signs. The DexAvatar paper states that evaluation uses 2,872 central frames \cite{kundu2026dexavatar}; however, the attached evaluator doubles both segment endpoints and iterates inclusively. The exact expected frame manifest must therefore be confirmed before any benchmark result is considered valid.

`[CẦN TÁC GIẢ XÁC NHẬN: whether segment endpoints are inclusive/exclusive, whether indices refer to 25-fps video or 50-fps meshes, the exact GT filenames selected for every sign, and the expected evaluated count]`.

The `signs.txt` labels are evaluation metadata only. They will not be inferred from or exposed to the reconstruction model. Any exploratory use of SGNify images, meshes, labels, or scores during development will be logged and will invalidate a strictly confirmatory test claim unless a new untouched benchmark is used.

#### 3.3.3. Quality control and exclusions

All inclusion and exclusion decisions will be rule based and applied before model comparison. A frame may be excluded from training for an unreadable image, corrupted annotation, incompatible topology, or failed timestamp association. It may not be excluded merely because the proposed model or a baseline has a high error. Every exclusion will be recorded with a reason code. Evaluation frames are never silently removed; missing or invalid predictions generate explicit failure records.

Pseudo-3D labels will receive provenance and quality flags based on reprojection consistency, mesh validity, temporal continuity, and, where available, agreement between independent estimators or views. These flags may weight training losses but will not turn estimator consensus into motion-capture truth. `[CẦN TÁC GIẢ XÁC NHẬN: exact QC thresholds and whether a human audit is feasible]`.

#### 3.3.4. Privacy, ethics, and licensing

This study uses secondary video data containing identifiable signers. The authors will document the consent/licensing basis for every source, restrict raw-video redistribution accordingly, and release only derived artifacts permitted by the relevant licenses. Dataset access, deletion obligations, and any institutional ethics determination will be reported. Performance will not be generalized to all sign languages or Deaf communities from a benchmark containing 57 DGS signs. `[CẦN TÁC GIẢ XÁC NHẬN: institutional ethics review or exemption and data-management plan]`.

#### 3.3.5. Prespecified role of each candidate dataset

| Dataset/resource | Permitted role in the planned study | Prohibited or unsupported use | Entry condition |
|---|---|---|---|
| SignAvatars + authorized source videos | Primary paired pseudo-SMPL-X/MANO data for (G); error labels for (Q) after QC; paired clips for (S). | May not be described as mocap truth; source duplicates may not cross splits. | Annotation access, source-video license, timestamp association, signer/source split and QC. |
| 3D-LEX | Optional clean/form-aware 3D supervision, retargeting validation or form probes. | May not directly supervise SMPL-X residuals if skeleton/mesh correspondence is unverified. | Asset access, representation map, retargeting error and language-aware scope. |
| ASLLVD | Optional articulatory/form labels and counterfactual validation. | Not a direct SMPL-X flow-training set without synchronized compatible 3D. | License, label ontology, exact video–pose alignment and expert review. |
| How2Sign | Authorized RGB source corresponding to a SignAvatars subset, if used. | The same source clip may not enter twice via “How2Sign” and “SignAvatars”; pseudo labels cannot be treated as independent estimators. | Deduplication by source ID/hash and compatible license. |
| DexYCB, HOT3D, ARCTIC or generic hand data | Optional hand-only pretraining/control, clearly labeled external data. | Cannot establish sign-language form preservation or body–hand compatibility by itself. | A separate external-data ablation and MANO/SMPL-X interface validation. |
| DexAvatar private mocap/prior data | None unless the authors grant documented access. | No assumed reproduction or hidden use. | Explicit access, split, license and provenance. |
| SGNify/DexAvatar benchmark assets | Locked final evaluation only. | No training, calibration, threshold/weight selection, counterfactual mining or checkpoint selection. | Confirmed frame manifest and protocol hashes. |

### 3.4. Input Processing and Deterministic Reconstruction

#### 3.4.1. Immutable temporal indexing

Video decoding preserves the original frame identifier, presentation timestamp, nominal frame rate, and decode status. Every downstream record is keyed by `(source_video_id, frame_id, timestamp)`. Crops and resized tensors retain the invertible affine transform back to original-image coordinates. Predictions from different frontends are joined by these keys, not by list position. If model-specific temporal sampling is required, interpolation or resampling rules are defined once and tested on synthetic timestamps. `[CẦN TÁC GIẢ XÁC NHẬN: decoder, variable-frame-rate handling, target FPS, and resampling rule]`.

#### 3.4.2. Frozen body and hand frontends

Let

\[
\Theta^{B,0}_{1:T}=F_B(I_{1:T})
\]

denote the SMPLer-X output, including SMPL-X body rotations, global orientation, shape, expression, and camera variables. Let

\[
\mathcal H^{h,0}_{1:T}=F_H^h(I_{1:T}),\qquad h\in\{L,R\},
\]

denote WiLoR's MANO output for each detected hand, including local hand rotations, a global wrist transform, hand shape, and detector/crop metadata. Both frontends are frozen. `[CẦN TÁC GIẢ XÁC NHẬN: repository commits, checkpoints, licenses, input resolutions, crop expansion, left/right assignment, and whether temporal test-time augmentation is disabled]`.

The base model is

\[
\Theta^0_{1:T}
=\mathcal A\!\left(\Theta^{B,0}_{1:T},
\mathcal H^{L,0}_{1:T},\mathcal H^{R,0}_{1:T}\right),
\]

where \(\mathcal A\) is a deterministic retargeting and fusion operator. This direct-fusion model is evaluated as a standalone baseline before any CUSP-SL component is trained.

#### 3.4.3. MANO-to-SMPL-X retargeting

For hand \(h\), let \(C_j^h\in SO(3)\) map the MANO local basis of joint \(j\) to the corresponding SMPL-X local basis. A local rotation may then be expressed in the target basis as

\[
R_{t,j}^{X,h}=C_j^h R_{t,j}^{M,h}(C_j^h)^{\top},
\]

provided that the two assets use the assumed rest-pose and parent-frame conventions. This equation specifies the required basis change but does not establish the values of \(C_j^h\). Those matrices must be derived from the exact licensed model assets and serialized with the experiment.

The global hand-to-body relationship is represented by a per-frame rigid transform \(A_t^h\in SE(3)\) estimated from the wrist and anatomically corresponding palm/forearm frames. `[CẦN TÁC GIẢ XÁC NHẬN: whether to reproduce Tamaththul3D swing–twist alignment, use a calibrated rigid alignment, or optimize a documented shoulder/forearm objective]`. The transformation may update the wrist/forearm chain but must not silently alter the root camera, body shape, or non-refined joints.

WiLoR MANO shape coefficients are not copied into SMPL-X. The SMPL-X body shape \(\boldsymbol\beta\) is retained unless a separately calibrated hand-shape map is developed and evaluated. This avoids combining incompatible shape spaces by assumption.

The adapter is accepted only after the following tests pass: identity/rest-pose reconstruction, known single-axis rotations, left–right chirality, joint-order consistency, forward-kinematic wrist continuity, MANO-to-SMPL-X-to-MANO round-trip error, unit consistency, original-image overlay, and no topology change. `[CẦN BỔ SUNG THỰC NGHIỆM: adapter test tolerances and results]`.

#### 3.4.4. Base failure contract

A residual model requires a valid base rotation. If \(R^0_{t,j}\) is valid but its image evidence is weak, CUSP-SL may refine it. If a complete base pose is unavailable or numerically invalid for a required span, v1 emits an abstention/failure record rather than synthesizing an absolute pose without a defined reference. Handling complete detector dropout would require an additional absolute-pose completion branch and is not part of the present method.

### 3.5. Calibrated Reliability and the Identity Gate

The reliability model \(Q_\omega\) predicts whether a base joint rotation is sufficiently accurate for direct use. For joint \(j\) at frame \(t\), we define a target error

\[
e_{t,j}=d_{SO(3)}(R^0_{t,j},R^*_{t,j}),
\qquad
d_{SO(3)}(R_1,R_2)
=\left\|\operatorname{Log}(R_1^{\top}R_2)\right\|_2,
\]

where \(R^*_{t,j}\) is compatible clean or quality-controlled pseudo-3D supervision and \(\operatorname{Log}:SO(3)\rightarrow\mathbb R^3\) is the rotation logarithm. If only 3D joint supervision is available, a separately declared positional error target is used; the two target definitions are not mixed within one calibration model.

The binary acceptability label is

\[
y_{t,j}=\mathbb 1[e_{t,j}\le e_{0,j}],
\]

where \(e_{0,j}\) is a joint-group-specific tolerance fixed using development data. `[CẦN TÁC GIẢ XÁC NHẬN: error definition and tolerances before training Q]`.

The input feature vector \(f^q_{t,j}\) contains only quantities available at inference, such as detector confidence, 2D keypoint confidence, reprojection residual, crop truncation, frontend disagreement or equivariance consistency, temporal discontinuity, and explicit missingness indicators. The exact feature list, window radius, and architecture are `[CẦN TÁC GIẢ XÁC NHẬN]`; ground-truth errors are never input features.

The model produces

\[
q_{t,j}=Q_\omega(f^q_{t-w:t+w,j})
\approx P(e_{t,j}\le e_{0,j}\mid f^q),
\]

and is trained on the training partition using weighted binary cross-entropy,

\[
\mathcal L_Q
=-\frac{1}{|\Omega_Q|}
\sum_{(t,j)\in\Omega_Q}
a_{y_{t,j}}
\left[y_{t,j}\log q_{t,j}
+(1-y_{t,j})\log(1-q_{t,j})\right],
\]

where \(\Omega_Q\) is the set of supervised tokens and \(a_0,a_1\) are class weights computed from the training partition only. The raw scores are calibrated on a held-out validation partition using `[CẦN TÁC GIẢ XÁC NHẬN: temperature scaling, isotonic regression, or another prespecified method]`. Calibration is assessed with reliability diagrams, Brier score, expected calibration error with fixed bins, and selective risk/coverage curves.

The edit gate is a monotone function of calibrated reliability:

\[
g_{t,j}=
\operatorname{clip}\!\left(
\frac{\tau_{\mathrm{hi}}-q_{t,j}}
{\tau_{\mathrm{hi}}-\tau_{\mathrm{lo}}},0,1
\right),
\qquad \tau_{\mathrm{lo}}<\tau_{\mathrm{hi}}.
\]

Thus, a highly reliable base estimate has \(g_{t,j}=0\), an unreliable estimate approaches \(g_{t,j}=1\), and intermediate estimates are softly edited. The thresholds and any temporal dilation/grouping rule are selected on validation data and then frozen. Reliability, correction magnitude, and missingness are kept as distinct variables.

### 3.6. Selective Temporal Residual Flow

#### 3.6.1. Rotation-residual target

For a supervised token, the target correction from the base rotation \(R^0_{t,j}\) to the reference rotation \(R^*_{t,j}\) is represented in the tangent space around the base:

\[
\delta^*_{t,j}
=\operatorname{Log}\!\left((R^0_{t,j})^{\top}R^*_{t,j}\right)
\in\mathbb R^3.
\]

This convention implies \(R^*_{t,j}=R^0_{t,j}\operatorname{Exp}(\delta^*_{t,j})\) when the principal logarithm is valid. Rotations close to the \(\pi\) singularity are handled by `[CẦN TÁC GIẢ XÁC NHẬN: numerical convention and exclusion/continuity policy]` and covered by unit tests.

Training uses a corruption/supervision mask \(m_{t,j}\in[0,1]\), derived from natural frontend errors, quality flags, or synthetic evidence corruption. Importantly, \(m\) is **not** the deployed reliability gate \(g\). The residual target is not multiplied by \(g\) during flow construction.

#### 3.6.2. Conditional rectified-flow objective

Let \(\epsilon\sim\mathcal N(0,I)\) have the same shape as the residual sequence and let \(s\sim\mathcal U(0,1)\) be the flow time. We construct the linear probability path

\[
x_s=(1-s)\epsilon+s\delta^*,
\]

whose target velocity is \(\delta^*-\epsilon\). A conditional velocity network \(v_\phi\) receives \(x_s\), \(s\), and conditioning tensor \(C\), and is trained with

\[
\mathcal L_{\mathrm{FM}}
=\mathbb E_{s,\epsilon}
\left[
\frac{
\sum_{t,j} \bar m_{t,j}
\left\|v_\phi(x_s,s,C)_{t,j}
-(\delta^*_{t,j}-\epsilon_{t,j})\right\|_2^2
}{
\sum_{t,j}\bar m_{t,j}+\varepsilon
}
\right],
\]

where \(\bar m_{t,j}\) is the product of the training mask and the 3D-label quality weight, \(\varepsilon>0\) prevents division by zero, and all weights are derived without using SGNify test labels.

The conditioning tensor contains only quantities available at deployment, except for training-only corruption indicators:

\[
C=\operatorname{Proj}\!left[
\operatorname{RotRep}(R^0),
q,
m_{\mathrm{obs}},
f^{2D},
f^{\mathrm{video}},
f^{\mathrm{kin}},
e_{\mathrm{token}}
\right].
\]

Here, \(\operatorname{RotRep}\) is a continuous network representation converted from \(R^0\); \(m_{\mathrm{obs}}\) encodes observed/missing evidence; \(f^{2D}\) contains confidence-aware 2D evidence in the original image frame; \(f^{\mathrm{video}}\) is frozen visual context; \(f^{\mathrm{kin}}\) contains torso-normalized joint and relative-hand geometry; and \(e_{\mathrm{token}}\) identifies joint, side, and body/hand type. `[CẦN TÁC GIẢ XÁC NHẬN: exact feature dimensions, normalization, video backbone, and whether all listed features survive ablation]`.

The network must allow temporal communication and communication along the upper-body/hand kinematic graph. A factorized temporal–kinematic Transformer is an implementable candidate, but the number of layers, hidden dimension, attention heads, positional encoding, and parameter budget are not fixed by the present evidence and remain `[CẦN TÁC GIẢ XÁC NHẬN]`. These values must be selected within a predeclared validation budget and matched against a deterministic residual network of comparable capacity.

#### 3.6.3. Endpoint supervision

During training, integrating the predicted velocity field from an initial noise sample yields an endpoint residual \(\widehat\delta\). Let

\[
\widehat R_{t,j}
=R^0_{t,j}\operatorname{Exp}(m_{t,j}\widehat\delta_{t,j})
\]

denote the training reconstruction. Endpoint losses are evaluated only when the corresponding reference is available and compatible. The rotation loss is

\[
\mathcal L_{\mathrm{rot}}
=\frac{
\sum_{t,j} \bar m_{t,j}
d_{SO(3)}^2(\widehat R_{t,j},R^*_{t,j})
}{\sum_{t,j}\bar m_{t,j}+\varepsilon}.
\]

Let \(\widehat\Theta_t\) be the parameter set obtained by inserting \(\widehat R_{t,j}\) into the otherwise unchanged base parameters, and let \(\widehat{\mathbf J}_t,\widehat{\mathbf V}_t=\mathcal M_X(\widehat\Theta_t)\). For available 3D labels, we use a torso-relative joint/vertex term

\[
\mathcal L_{\mathrm{3D}}
=\frac{1}{|\Omega_J|}\sum_{(t,j)\in\Omega_J}
w^{J}_{t,j}
\left\|
(\widehat J_{t,j}-\widehat J_{t,r})
-(J^*_{t,j}-J^*_{t,r})
\right\|_1
+\lambda_V
\frac{1}{|\Omega_V|}\sum_{(t,n)\in\Omega_V}
w^{V}_{t,n}
\left\|
(\widehat V_{t,n}-\widehat J_{t,r})
-(V^*_{t,n}-J^*_{t,r})
\right\|_1,
\]

where \(r\) is the declared torso/root joint, \(\Omega_J\) and \(\Omega_V\) are supervised joint and vertex sets, and \(w^J,w^V\) are quality weights. The vertex term is used only when the reference has the identical topology and correspondence. The value of \(\lambda_V\) is `[CẦN TÁC GIẢ XÁC NHẬN]`.

For observed 2D keypoints \(k_{t,j}\) with confidence \(c_{t,j}\), the reprojection loss is

\[
\mathcal L_{\mathrm{2D}}
=\frac{
\sum_{(t,j)\in\Omega_{2D}}
c_{t,j}\,
\rho\!\left(
\left\|\Pi_{\gamma_t}(\widehat J_{t,j})-k_{t,j}\right\|_2
\right)
}{
\sum_{(t,j)\in\Omega_{2D}}c_{t,j}+\varepsilon
},
\]

where \(\Pi_{\gamma_t}\) is the explicitly declared camera projection and \(\rho\) is `[CẦN TÁC GIẢ XÁC NHẬN: robust penalty and scale]`. Coordinates are transformed back to the original image before this loss is evaluated.

To preserve rapid but coherent signing motion, temporal supervision compares predicted dynamics with reference dynamics rather than penalizing motion magnitude directly. For torso-relative 3D joints \(X_{t,j}=J_{t,j}-J_{t,r}\),

\[
\mathcal L_{\mathrm{dyn}}
=\frac{1}{|\Omega_{\Delta}|}
\sum_{(t,j)\in\Omega_{\Delta}}
w^{\Delta}_{t,j}\,
\rho\!\left(\left\|\Delta\widehat X_{t,j}-\Delta X^*_{t,j}\right\|_2\right)
+\lambda_{\Delta^2}
\frac{1}{|\Omega_{\Delta^2}|}
\sum_{(t,j)\in\Omega_{\Delta^2}}
w^{\Delta^2}_{t,j}\,
\rho\!\left(\left\|\Delta^2\widehat X_{t,j}-\Delta^2X^*_{t,j}\right\|_2\right),
\]

where \(\Delta\) and \(\Delta^2\) are first- and second-order finite differences computed using the true timestamps. This term is disabled where reference timing or pose quality is insufficient.

The generator objective is

\[
\mathcal L_G
=\lambda_{\mathrm{FM}}\mathcal L_{\mathrm{FM}}
+\lambda_{\mathrm{rot}}\mathcal L_{\mathrm{rot}}
+\lambda_{\mathrm{3D}}\mathcal L_{\mathrm{3D}}
+\lambda_{\mathrm{2D}}\mathcal L_{\mathrm{2D}}
+\lambda_{\mathrm{dyn}}\mathcal L_{\mathrm{dyn}}.
\]

All \(\lambda\) values, the use of differentiable endpoint integration, and any schedule that activates auxiliary losses are `[CẦN TÁC GIẢ XÁC NHẬN]` and must be selected on the development validation set. If differentiable integration is not implemented, only \(\mathcal L_{\mathrm{FM}}\) is a justified mandatory objective; endpoint terms may not be claimed as used.

#### 3.6.4. Inference and one-time gating

At inference, the ODE

\[
\frac{dx_s}{ds}=v_\phi(x_s,s,C),\qquad x_0=\epsilon^{(k)},
\]

is integrated from \(s=0\) to \(1\) for each candidate \(k\in\{1,\ldots,K\}\), producing residual \(\delta^{(k)}\). The solver, number of function evaluations, tolerances, candidate count, and random-seed policy are `[CẦN TÁC GIẢ XÁC NHẬN]`.

Each candidate rotation is assembled as

\[
R^{(k)}_{t,j}
=R^0_{t,j}\operatorname{Exp}\!\left(g_{t,j}\delta^{(k)}_{t,j}\right).
\]

This is the only point at which the inference gate multiplies the residual. Consequently, \(g_{t,j}=0\) gives \(R^{(k)}_{t,j}=R^0_{t,j}\) exactly, up to numerical precision. Contiguous active tokens are grouped into temporal windows with validation-fixed context and overlap. Overlapping residuals are combined in the same tangent convention using a deterministic taper before the exponential map; `[CẦN TÁC GIẢ XÁC NHẬN: window length, overlap, taper, and boundary policy]`.

### 3.7. Counterfactual Video–Pose Form Consistency

#### 3.7.1. Scope and encoders

Monocular evidence can support multiple 3D candidates with similar reprojection error. We therefore test a frozen form-consistency score that is explicitly required to distinguish signing-relevant articulatory changes. This component does not infer a language-independent meaning and is not called a semantic oracle.

Let \(E_v\) encode an RGB clip and \(E_p\) encode a synchronized pose sequence:

\[
z^v_i=E_v(I^i_{1:T}),\qquad
z^p_i=E_p(\Theta^i_{1:T}),\qquad
s(I^i,\Theta^j)
=\frac{(z^v_i)^{\top}z^p_j}
{\|z^v_i\|_2\|z^p_j\|_2}.
\]

The video tower may be initialized from an official SignDINO checkpoint only if the checkpoint, preprocessing, license, and checksum are available \cite{gan2026signdino}. Otherwise, the authors must reproduce the cited architecture/training procedure or omit this initialization. “SignDINO-like” is not a reproducible specification.

The pose tower consumes body/hand rotations, torso-normalized wrist trajectories, palm normals, relative hand locations, and visibility masks. Its architecture and representation dimensions are `[CẦN TÁC GIẢ XÁC NHẬN]`. Face features are excluded from the v1 scorer unless a valid face branch and evaluation are added.

#### 3.7.2. Paired contrastive objective

For a minibatch of \(B_S\) paired video–pose clips, the video-to-pose loss is

\[
\mathcal L_{v\rightarrow p}
=-\frac{1}{B_S}\sum_{i=1}^{B_S}
\log
\frac{\exp(s(I^i,\Theta^i)/\tau_S)}
{\sum_{j=1}^{B_S}\exp(s(I^i,\Theta^j)/\tau_S)},
\]

and \(\mathcal L_{p\rightarrow v}\) is defined symmetrically. The paired objective is

\[
\mathcal L_{\mathrm{NCE}}
=\tfrac12\left(\mathcal L_{v\rightarrow p}
+\mathcal L_{p\rightarrow v}\right),
\]

where \(\tau_S>0\) is a learned or validation-fixed temperature `[CẦN TÁC GIẢ XÁC NHẬN]`.

#### 3.7.3. One-factor counterfactuals

For a paired pose \(\Theta_i^+\), a counterfactual operator \(T_a\) changes one articulatory axis \(a\) while preserving the remaining measured factors as closely as possible:

\[
\Theta^-_{i,a}=T_a(\Theta_i^+).
\]

Candidate axes are handshape, palm orientation, hand location relative to the torso, local movement trajectory, and bimanual relation/contact. An axis is included only when its operator has (i) an explicit kinematic definition, (ii) ROM and non-penetration checks, (iii) a matched low-level-motion control, and (iv) validation by `[CẦN TÁC GIẢ XÁC NHẬN: qualified sign-language experts and annotation protocol]`. In particular, learned positive-contact perturbations are not part of the core method without labels.

For validated counterfactuals, we use a margin loss

\[
\mathcal L_{\mathrm{cf}}
=\frac{1}{|\Omega_{\mathrm{cf}}|}
\sum_{(i,a)\in\Omega_{\mathrm{cf}}}
\max\left(
0,
m_a-s(I^i,\Theta_i^+)+s(I^i,\Theta^-_{i,a})
\right),
\]

where \(m_a>0\) is a validation-fixed margin and \(\Omega_{\mathrm{cf}}\) contains only counterfactuals that pass the declared validity tests. The scorer objective is

\[
\mathcal L_S
=\mathcal L_{\mathrm{NCE}}
+\lambda_{\mathrm{cf}}\mathcal L_{\mathrm{cf}}.
\]

The scorer is accepted for candidate selection only if it passes source/signer-disjoint retrieval, same-video one-factor discrimination for each claimed axis, background/crop invariance probes, and tests on real reconstruction candidates. It is then frozen. It is not optimized jointly with \(G\) in the main method, preventing the generator from learning to exploit scorer-specific artifacts.

### 3.8. Candidate Validity, Selection, and Disagreement

For candidate \(k\), observation consistency is measured in original-image coordinates:

\[
E_{\mathrm{obs}}^{(k)}
=\frac{
\sum_{(t,j)\in\Omega_{2D}}
c_{t,j}\rho\!\left(
\|\Pi_{\gamma_t}(J^{(k)}_{t,j})-k_{t,j}\|_2
\right)
}{\sum_{(t,j)\in\Omega_{2D}}c_{t,j}+\varepsilon}.
\]

Visible motion consistency is

\[
E_{\mathrm{mot}}^{(k)}
=\frac{
\sum_{(t,j)\in\Omega_{\Delta 2D}}
\bar c_{t,j}\rho\!\left(
\|\Delta\Pi_{\gamma_t}(J^{(k)}_{t,j})-\Delta k_{t,j}\|_2
\right)
}{\sum_{(t,j)\in\Omega_{\Delta 2D}}\bar c_{t,j}+\varepsilon}
+\eta_{\mathrm{acc}}
\frac{1}{|\Omega_{\mathrm{acc}}|}
\sum_{(t,j)\in\Omega_{\mathrm{acc}}}
\rho\!\left(\|\Delta^2X^{(k)}_{t,j}\|_2\right),
\]

where \(\bar c_{t,j}\) combines adjacent-frame confidence and the acceleration coefficient \(\eta_{\mathrm{acc}}\) is validation selected. The first term preserves observed rapid movement; the second is only a weak plausibility penalty and must be ablated to detect oversmoothing.

Physical validity is represented by

\[
E_{\mathrm{phys}}^{(k)}
=E_{\mathrm{ROM}}^{(k)}
+\lambda_{\mathrm{pen}}E_{\mathrm{pen}}^{(k)},
\]

where \(E_{\mathrm{ROM}}\) penalizes rotation components outside a documented anatomical convention and \(E_{\mathrm{pen}}\) penalizes mesh interpenetration [CITATION NEEDED: exact adopted formulations]. The exact joint limits, Euler/basis conversion, collision implementation, and thresholds are `[CẦN TÁC GIẢ XÁC NHẬN]`. These terms do not reward positive contact and are not interpreted as linguistic contact models.

Let \(s^{(k)}=s(I,\Theta^{(k)})\) be the frozen form score. Each scalar term is robustly standardized using median and median absolute deviation computed on the development validation partition:

\[
\widetilde E
=\frac{E-\operatorname{median}_{\mathrm{val}}(E)}
{\operatorname{MAD}_{\mathrm{val}}(E)+\varepsilon}.
\]

The candidate energy is

\[
E_k
=w_o\widetilde E_{\mathrm{obs}}^{(k)}
+w_m\widetilde E_{\mathrm{mot}}^{(k)}
+w_p\widetilde E_{\mathrm{phys}}^{(k)}
-w_s\widetilde s^{(k)}.
\]

Weights \(w_o,w_m,w_p,w_s\ge0\) are fitted within a prespecified validation budget and frozen before SGNify evaluation. The candidate with minimum valid energy is selected:

\[
\widehat k=\arg\min_{k\in\mathcal K_{\mathrm{valid}}}E_k.
\]

If \(\mathcal K_{\mathrm{valid}}\) is empty, the method emits a failure. The base candidate is included in \(\mathcal K\) so that the selector can decline a learned edit.

For diagnostics, we define energy-normalized weights

\[
\pi_k
=\frac{\exp(-E_k/T_E)}
{\sum_{\ell\in\mathcal K_{\mathrm{valid}}}\exp(-E_\ell/T_E)},
\]

where \(T_E\) is fixed on validation. These weights do not constitute a calibrated posterior. Candidate disagreement at token \((t,j)\) is

\[
u_{t,j}
=\sum_{k\in\mathcal K_{\mathrm{valid}}}
\pi_k\,
d_{SO(3)}^2(R^{(k)}_{t,j},R^{(\widehat k)}_{t,j}),
\]

and the window-level score is the declared aggregate of \(u_{t,j}\) over active tokens. `[CẦN TÁC GIẢ XÁC NHẬN: aggregate and warning threshold]`. We report disagreement, energy margin, and empirical error separately; we do not label disagreement as calibrated uncertainty until a held-out calibration analysis supports that claim.

### 3.9. Training Procedure

Training is staged to preserve causal attribution and prevent leakage.

1. **Manifest and split construction.** Resolve licenses and create source-/signer-disjoint train, validation, and internal test manifests before window extraction. Deduplicate source videos and near-duplicate frames across datasets. Freeze the SGNify test manifest separately.
2. **Frontend and adapter validation.** Run the frozen frontends, test \(\mathcal A\), and cache base parameters, 2D observations, transforms, detector metadata, missingness, and features using immutable keys. All primary ablations consume the same cache.
3. **Form-scorer training.** Train \(E_v,E_p\) using \(\mathcal L_S\), validate per-axis sensitivity and shortcuts, select one checkpoint, and freeze it. If feasibility/validity criteria fail, set \(w_s=0\) and report a geometry-only method.
4. **Reliability training and calibration.** Construct error labels from clean or quality-controlled references, train \(Q_\omega\) with \(\mathcal L_Q\), calibrate on held-out signers/sources, and freeze \(Q\), tolerances, and gate thresholds.
5. **Residual-flow training.** Build natural-error and synthetic-corruption windows, train \(G_\phi\) with \(\mathcal L_G\), and monitor mode coverage, rotation validity, temporal fidelity, and edits on high-reliability controls. Counterfactual scorer gradients do not enter \(G\).
6. **Candidate-energy fitting.** Generate validation candidates with frozen \(F,Q,G,S\); compute normalization statistics and select energy weights using only validation data. Lock the complete configuration and hashes.
7. **Final evaluation.** Run the locked model on SGNify once per predeclared seed/sampling policy, capture all predictions including failures, and evaluate through the two protocol tracks.

The implementation details required for reproducibility are not yet available and must be completed in the final manuscript:

| Item | Required report |
|---|---|
| Optimizers and schedulers | `[CẦN TÁC GIẢ XÁC NHẬN: optimizer, learning rate, weight decay, schedule, warm-up]` for `Q,G,S` separately |
| Training budget | `[CẦN TÁC GIẢ XÁC NHẬN: epochs/steps, batch/window sampling, early stopping, checkpoint rule]` |
| Architecture | `[CẦN TÁC GIẢ XÁC NHẬN: layer counts, widths, heads, tokenization, parameter counts]` |
| Windows | `[CẦN TÁC GIẢ XÁC NHẬN: FPS, length in frames/seconds, overlap, boundary padding]` |
| Flow sampling | `[CẦN TÁC GIẢ XÁC NHẬN: solver, steps/tolerance, K, seed handling]` |
| Losses | `[CẦN TÁC GIẢ XÁC NHẬN: all λ values, robust losses/scales, label-quality weighting]` |
| Gate | `[CẦN TÁC GIẢ XÁC NHẬN: e0, tau_lo, tau_hi, dilation/grouping, calibration method]` |
| Hardware/software | `[CẦN TÁC GIẢ XÁC NHẬN: GPU/CPU, CUDA, framework and dependency versions, precision]` |
| Randomness | `[CẦN TÁC GIẢ XÁC NHẬN: training seeds, deterministic settings, primary inference seed/policy]` |
| Search budget | `[CẦN TÁC GIẢ XÁC NHẬN: search space, number of trials, selection metric, compute]` |

### 3.10. Inference Procedure

For each test video, inference proceeds as follows:

1. Decode all frames and construct the immutable temporal manifest.
2. Run frozen SMPLer-X and WiLoR and apply the tested retargeting operator \(\mathcal A\).
3. If the base contract fails for a required span, emit an abstention and continue logging; do not remove the frame.
4. Compute calibrated reliability \(q\), edit gate \(g\), and active temporal windows.
5. If no token is active, return the base sequence exactly.
6. For each active window, generate \(K\) residual candidates, apply \(g\) once, and retain the base as an additional no-edit candidate.
7. Reject only candidates that violate predeclared numerical/topological validity rules; calculate observation, motion, physical, and optional frozen form scores for all remaining candidates.
8. Select \(\widehat k\), merge overlapping tangent residuals using the frozen boundary rule, and restore the exact base rotation wherever \(g=0\).
9. Decode SMPL-X meshes and save pose, mesh, candidate energies, disagreement, runtime, memory, coverage, and failure reason for every expected frame.

Training-only masks, reference labels, sign-class labels, and SGNify ground truth are unavailable during inference. An online/causal version is outside the present claim unless it is implemented and evaluated separately.

### 3.11. Evaluation Protocol and Statistical Analysis

#### 3.11.1. Baselines and controlled ablations

The following systems are required, subject to code/checkpoint availability:

- published DexAvatar output or a reproduction using its official repository/configuration;
- frozen SMPLer-X alone;
- frozen SMPLer-X + WiLoR with direct tested retargeting;
- the strongest deterministic temporal residual model with a matched parameter/compute budget;
- CUSP variants using identical cached frontend outputs and identical candidate sets.

The minimum ablation sequence is:

| ID | Variant | Question isolated |
|---|---|---|
| A0 | SMPLer-X | Original whole-body frontend |
| A1 | SMPLer-X + WiLoR | Gain from stronger hand estimator/fusion |
| A2 | A1 + simple temporal filter | Gain from inexpensive smoothing |
| A3 | A1 + deterministic residual model | Gain from learned temporal correction without sampling |
| A4 | A1 + `Q + G`, `K=1` | Gate/residual effect without multi-candidate selection |
| A5 | A1 + `Q + G`, `K>1`, random select | Candidate distribution without informed selection |
| A6 | Same candidates, oracle select | Whether the correct/better mode is covered; diagnostic only |
| A7 | Same candidates, geometry-only select | Contribution of observation/motion/physical energy |
| A8 | Same candidates, form-only select | Whether `S` has independent selection value |
| A9 | Same candidates, combined select | Proposed full selector |
| A10 | A9 with `Q` disabled/always-on | Whether selective identity adds value and saves compute |
| A11 | Matched-compute deterministic restarts | Whether gains are only due to additional samples/compute |

Oracle selection is never reported as deployable performance. If A6 does not improve meaningfully over A4, the multi-candidate hypothesis is rejected. If A9 does not outperform A7 on held-out real candidates, the form scorer is removed from the main method and the novelty claim is revised.

`[CẦN TÁC GIẢ XÁC NHẬN: quantitative kill thresholds and smallest effect of interest before running the benchmark]`.

#### 3.11.2. Author-comparability evaluation

Direct comparison with the DexAvatar table uses the author-supplied evaluator identified by SHA-256 above, its exact `segment.json` and `signs.txt`, the same SMPL-X topology/region arrays, and predictions in metres. The script independently translation-centers each reported vertex subset and multiplies the final mean distance by 1,000 to report millimetres. Primary comparison regions are UBody(-F), LHand, and RHand. For class-`0` signs, the evaluator omits the left hand and removes left-hand vertices from other regions.

Because the script's frame selection is currently ambiguous relative to the stated 2,872 frames, no official comparison will be reported until the expected per-sign manifest and a DexAvatar reference run are reproduced. Only documented path externalization is permitted in this track; the immutable original, patch diff, environment, input hashes, stdout, and selected GT/prediction filenames will be archived.

#### 3.11.3. Audited evaluation

The audited evaluator uses explicit `(sign, frame_id)` joins, asserts equal mesh topology and vertex order, verifies units, requires one status record per expected frame, saves frame-level errors, and never shifts pairing after a missing file. NaN or missing predictions are failures and are reported in coverage; they are not silently conditioned out. The audited report includes the same region-wise translation alignment for diagnostic comparability, plus `[CẦN TÁC GIẢ XÁC NHẬN: any additional global/pelvis-aligned metric]`. Values from this track are labeled “audited” and are not substituted into the author-protocol comparison table.

#### 3.11.4. Outcomes and uncertainty

The official evaluator's global vertex-frame mean is reported exactly because it is the direct comparison quantity. In addition, for each sign and region we compute a sign-level mean using the same frame/vertex error definition, and report paired method differences. Confidence intervals are obtained by resampling the 57 signs as clusters with `[CẦN TÁC GIẢ XÁC NHẬN: number and type of bootstrap replicates]`. This secondary analysis prevents long signs from being the sole determinant of inferential uncertainty.

For trained components, results are reported across `[CẦN TÁC GIẢ XÁC NHẬN: number of independent training seeds]`; candidate-sampling randomness is separated from training-seed variation. We report absolute and relative paired effects with confidence intervals, not only p-values. If hypothesis tests are used for three co-primary regions, the multiplicity strategy is `[CẦN TÁC GIẢ XÁC NHẬN]`. The primary endpoint and rule for claiming improvement across UBody(-F), LHand, and RHand must be fixed before test access.

Because SGNify is a fixed benchmark rather than a newly sampled participant study, no conventional recruitment sample-size calculation is applicable. The limited 57-sign support constrains external validity; confidence intervals and an independent dataset are therefore required for broad claims.

Secondary diagnostics include frame coverage, NaN/failure rate, runtime and peak memory on matched hardware, geodesic error where compatible 3D parameters exist, velocity/acceleration fidelity, gate calibration and risk–coverage, edit magnitude on reliable joints, candidate oracle gap, selector top-1 regret, ROM violation, and penetration. Any high-occlusion subset must be defined from annotations independent of model errors and before comparing methods.

### 3.12. Reproducibility and Reporting

The release will include, where licenses permit: source commit IDs; environment lockfiles; model checkpoint hashes; exact preprocessing; SMPL-X/MANO asset identifiers; joint/vertex maps; coordinate-adapter parameters; immutable train/validation/test manifests; data provenance; all random seeds; complete configuration files; training and inference logs; evaluator original and patch; unit tests; per-frame prediction status; and scripts that regenerate every table. Restricted datasets and model assets will be referenced by acquisition instructions rather than redistributed.

The study will follow the target venue's reproducibility checklist and disclose external training data, compute, model-selection budget, failed runs, and test-set exposure. `[CẦN TÁC GIẢ XÁC NHẬN: target conference and its current checklist]`.

### 3.13. Methodological Limitations

CUSP-SL v1 does not reconstruct missing base poses from scratch, does not refine facial non-manual markers, and does not contain a learned positive-contact model. Its 3D supervision may inherit bias from pseudo-SMPL-X annotations. A form scorer trained on heterogeneous sign languages may capture geometric regularities but cannot be assumed to preserve lexical meaning in DGS. The SGNify benchmark is small and its supplied ground truth/evaluator contain unresolved protocol and plausibility issues. Accordingly, any positive result is limited to the declared protocol, data, and regions and requires independent validation before a general sign-language reconstruction claim.

---

# E. Change Log

| Vị trí | Nội dung cũ | Nội dung mới | Lý do sửa | Nguồn/căn cứ |
|---|---|---|---|---|
| Title/claim | “Counterfactual Uncertainty-gated Semantic Posterior” và định vị như ứng viên SOTA. | “Calibrated Uncertainty-triggered Selective Pose”; design specification, không có superiority/posterior claim. | Tên mới khớp đúng gate đã calibration và output không phải posterior. | Mathematical/statistical audit và yêu cầu của tác giả. |
| Problem formulation | “Whole-body pose” dù chỉ xử lý selected upper body/hands. | Output scope là upper-body + hands; face/camera/shape pass-through. | Khớp biến thực sự được can thiệp và metric. | DexAvatar paper/evaluator regions. |
| Frontend | `SMPLer-X + WiLoR` nằm trong method contribution. | Thành strong baseline/inherited engineering. | Prior art đã có direct fusion. | Tamaththul3D; SMPLer-X; WiLoR. |
| Data | Nêu chung paired videos/pseudo-3D. | SignAvatars có điều kiện access; source/signer split; clean-3D option; SGNify test-only. | Chống leakage, xác định lineage và feasibility. | SignAvatars repository/paper; SGNify protocol. |
| Temporal indexing | Dùng thứ tự frame ngầm định. | Join bằng `(video_id, frame_id, timestamp)`; no silent drop. | Ngăn lệch alignment. | Audit evaluator lines 231–361. |
| Hand/body fusion | “Coordinate conversion + swing–twist” như chi tiết đã quyết định. | Abstract tested adapter \(\mathcal A\); exact algorithm là placeholder; no MANO-shape copy. | Không bịa transform hoặc giả định shape-space compatibility. | MANO/SMPL-X interface; Tamaththul3D prior art. |
| Reliability | `q` vừa là confidence vừa là uncertainty. | \(q=P(e\le e_0\mid f)\) duy nhất; gate giảm theo \(q\). | Sửa lỗi logic. | Calibration/selective prediction principles. |
| Mask/gate | Target residual nhân (g), output lại nhân (g). | Training mask (m) tách khỏi inference gate (g); output chỉ gate một lần. | Loại double gating và residual shrinkage. | Mathematical audit. |
| Flow output | Full hand/body trajectory không có identity path rõ. | Tangent residual quanh base, exact identity khi (g=0). | Giảm burden và bảo toàn evidence chắc chắn. | Adaptation of HandFlow/rectified flow. |
| Generator architecture | Cố định 8 blocks/512/8 heads và nhiều hyperparameter chưa thử. | Mọi architecture/hyperparameter không có evidence được đánh dấu xác nhận. | Tránh fabricated details và overfitting design. | Reproducibility standard. |
| Loss | Chỉ liệt kê tên. | Công thức cho FM, geodesic, 3D, 2D và temporal dynamics; chỉ dùng khi label/solver phù hợp. | Cho phép triển khai và audit gradient. | Rotation geometry and differentiable SMPL-X. |
| Contact | Learned relation/contact head là core. | Loại khỏi v1; deterministic ROM/penetration only. | Thiếu labels/calibration; collision không phải positive contact. | Dataset audit; DexAvatar ground-truth limitation. |
| Semantic module | “SignDINO-like semantic scorer.” | Exact-checkpoint/reproduction branch; form-consistency scorer; per-axis counterfactual validation. | Tái lập và giảm shortcut/cross-language overclaim. | SignDINO paper; sign linguistics constraints. |
| Scorer–generator training | Có khả năng end-to-end. | Train scorer trước, freeze; không backprop score vào generator ở main method. | Tránh generator gaming và hỗ trợ causal attribution. | Experimental design logic. |
| Candidate probability | Softmax energy gọi là posterior. | Energy-normalized candidate weights \(\pi\); no probability calibration claim. | Softmax không tạo Bayesian posterior. | Statistical calibration logic. |
| Candidate selection | Một full energy gồm observation/contact/semantic. | Same candidate set; observation, visible motion, physical validity, optional form score; base candidate included. | Tách contribution và cho phép decline edit. | Controlled ablation design. |
| Missing base | Ngầm dùng mask token để recover mọi dropout. | Complete invalid base → abstention in v1. | Residual không có reference; tránh che giấu missing functionality. | Interface validity. |
| Evaluation | “Official 2,872 frames” được mặc định. | Nêu mismatch 2,872 vs doubled inclusive code; author-comparability + audited tracks. | Protocol chưa xác định duy nhất. | Attached evaluator and metadata hashes. |
| TR alignment | Mô tả chung translation alignment. | Nêu rõ center riêng từng vertex region. | Quyết định giá trị metric và khả năng so sánh. | `transl_point_error`, lines 159–169. |
| Failure denominator | Không có policy. | Exact track ghi behavior; audited track báo missing/NaN failures và coverage. | Tránh conditioning-on-success. | Evaluator lines 364–366. |
| Aggregation | Một mean duy nhất. | Official vertex-frame mean + secondary paired per-sign effects/cluster CI. | Thể hiện uncertainty và sign-level stability. | Evaluator lines 455–461. |
| Ablation | G/S/R factorial chưa hoàn chỉnh. | A0–A11, identical candidates, oracle/random/geometry/form/combined, matched compute. | Tách coverage, selection, backbone và compute. | Causal experiment design. |
| Test usage | Có nguy cơ chọn config từ test. | SGNify test isolated; lock config/hash trước final evaluation. | Ngăn leakage. | DexAvatar supplement explicitly mentions DEV and TEST selection; standard practice. |
| Reproducibility | Danh sách chung. | Hashes, manifests, patch diff, unit tests, status per expected frame, source provenance. | Đủ để audit và tái lập. | Repo/evaluator audit. |

---

# F. Nội dung chưa thể hoàn thiện

## F.1. Thông tin tác giả cần xác nhận

1. Conference đích và deadline/policy về external data, anonymized repository và reproducibility checklist.
2. Primary endpoint: UBody(-F), LHand, RHand là co-primary hay có một endpoint chính; tiêu chí “better than DexAvatar” khi ba vùng không cùng cải thiện.
3. SGNify frame semantics: FPS, endpoint convention, GT filenames và expected frame count per sign.
4. Quyền truy cập và phiên bản cụ thể của SignAvatars annotations, source RGB videos và mọi mocap/3D-LEX/ASLLVD resource.
5. Exact frozen frontends, commits/checkpoints, input resolutions and licenses.
6. Camera model, units, global coordinate convention and target SMPL-X assets.
7. MANO-to-SMPL-X adapter: reproduce Tamaththul3D or implement a new calibrated transformation; wrist/forearm/shoulder policy.
8. Which sign-form axes have expert-validated counterfactual operators and which languages they cover.
9. Architecture/training/search budgets for \(Q,G,S\), including every hyperparameter currently marked as a placeholder.
10. Independent training seeds, candidate-sampling policy, effect-size threshold, bootstrap and multiplicity plan.

## F.2. Chi tiết triển khai còn thiếu

- Environment lock and hardware specification.
- Exact frame decoder, FPS conversion, crop affine conventions and 2D detector mapping.
- Full SMPL-X/MANO joint-order and basis maps.
- Rest-pose, chirality, round-trip and overlay test thresholds.
- Reliability feature schema, label definition and calibration algorithm.
- Flow architecture, ODE solver, sampling steps, window length and overlap merge.
- Pose/video encoder details and checkpoint provenance.
- Counterfactual construction code, validity checks and annotation form.
- All loss weights and robust-penalty scales.
- Candidate-energy fitting algorithm and search budget.
- Failure/abstention serialization and audited-evaluator penalty/reporting policy.
- Exact baseline reproduction commands and acceptable tolerance to DexAvatar's reported score.

## F.3. Citation cần bổ sung hoặc xác minh trong manuscript

The English Methods uses verified citation keys where a primary bibliographic source was identified and retains `[CITATION NEEDED]` only where the exact adopted formulation/resource is not yet selected. At minimum, the final `.bib` must verify and cite:

- SMPL-X and MANO model papers/assets;
- SMPLer-X and WiLoR official papers/checkpoints;
- DexAvatar and SGNify;
- Tamaththul3D for prior SMPLer-X/WiLoR fusion;
- HandFlow and the foundational rectified-flow reference actually used by the implementation;
- MaskHand/MMHMR only if its mechanism is discussed;
- SignDINO only if its exact model or training procedure is used;
- SignAvatars, ASLLVD and 3D-LEX only if those resources are actually accessed;
- the exact ROM, penetration/collision and robust-loss formulations implemented.

## F.4. Thực nghiệm bắt buộc trước khi viết claim

| Priority | Experiment | Decision enabled |
|---:|---|---|
| P0 | Build exact SGNify frame manifest and obtain author confirmation | Whether “official protocol” is defined and reproducible |
| P0 | Run toy evaluator tests for missing GT, missing prediction, NaN, ordering, units and per-region alignment | Whether metrics/denominators behave as documented |
| P0 | Reproduce DexAvatar with author artifacts | Whether direct numerical comparison is valid |
| P0 | Implement and test SMPLer-X + WiLoR adapter | Whether the base pipeline is technically valid |
| P0 | Run strong direct-fusion and simple temporal controls | Whether any learned method is necessary |
| P1 | Measure base error by reliability/occlusion bins on development 3D data | Whether a selective gate is learnable/useful |
| P1 | Deterministic residual vs (K>1) oracle curve with matched compute | Whether a multi-modal generator has useful coverage |
| P1 | Real-candidate form-scorer test with per-axis counterfactuals | Whether `S` detects 3D sign form rather than shortcuts |
| P1 | Identity test for (g=0) and calibration/risk–coverage | Whether selective editing preserves strong estimates |
| P1 | Full A0–A11 ablation on internal test data | Whether the proposed interaction is more than module addition |
| P2 | Locked SGNify author-comparability and audited runs | Final benchmark evidence |
| P2 | Independent dataset or language validation | External-validity boundary |
| P2 | Runtime/memory and matched-compute Pareto analysis | Whether gains justify sampling cost |

## F.5. Rủi ro phương pháp chưa giải quyết

| Risk | Severity | Observable failure signature | Mitigation / stop rule |
|---|---|---|---|
| Author evaluator frame ambiguity | Nghiêm trọng | Evaluated counts differ from expected; reproduction score unstable | Do not report official comparison until manifest is confirmed. |
| Strong frontend leaves no oracle room | Nghiêm trọng | Best-of-​K oracle ≈ deterministic top-1 | Stop multi-candidate work; retain simpler baseline. |
| Pseudo-label confirmation bias | Nghiêm trọng | Improvement on pseudo labels but not mocap/real candidates | Add clean calibration data; report source-stratified effects. |
| Form scorer shortcut | Nghiêm trọng | High retrieval but fails same-video orientation/location changes | Remove scorer claim/component. |
| Cross-language mismatch | Nghiêm trọng | ASL-trained score changes DGS candidates inconsistently | Restrict claim to geometric form; add DGS validation or omit S. |
| Adapter chirality/frame error | Nghiêm trọng | Mirrored fingers, wrist discontinuity, good hand-local but bad body-relative pose | Block all downstream training until adapter tests pass. |
| Oversmoothing fast articulation | Trung bình–nghiêm trọng | Lower acceleration but attenuated motion and worse hand TR-V2V | Compare motion amplitude; reduce/remove acceleration penalty. |
| Candidate weights misread as probability | Trung bình | Poor error calibration despite sharp π | Report only disagreement/margin; no posterior claim. |
| Selective model misses complete dropout | Trung bình | No valid base for long span | Abstain and report coverage; separate future absolute-pose branch. |
| Face/non-manual omission | Trung bình | Geometric hand success but incomplete linguistic intelligibility | Limit scope; add face branch only with data and evaluation. |
| Compute confounding | Trung bình | Gain disappears under matched wall time/restarts | Report Pareto curve; prefer simpler model if dominated. |
| Tiny fixed benchmark | Trung bình | Large per-sign CI, unstable ranking | Paired sign bootstrap and independent validation. |

## F.6. Mười câu hỏi quan trọng nhất nhóm nghiên cứu phải trả lời

1. **Danh sách chính xác các GT mesh/frame mà authors mong evaluator chấm cho 57 signs là gì, và tổng phải là 2,872, 2,929 hay một con số khác?**
2. **Primary endpoint và success rule được khóa trước test là gì khi UBody(-F), LHand và RHand có thể cho kết luận khác nhau?**
3. **Nhóm có thực sự truy cập được paired RGB–SMPL-X training data nào, với license, language, signer ID và source-level split nào?**
4. **MANO-to-SMPL-X transformation chính xác là gì và bằng chứng nào cho thấy chirality, wrist orientation, scale và joint order đều đúng?**
5. **Sau strong baseline `SMPLer-X + WiLoR + simple temporal filter`, còn bao nhiêu oracle headroom trên development mocap/pseudo-3D data?**
6. **Reliability target \(e_{t,j}\), tolerance \(e_{0,j}\), features và calibration method nào cho phép dự đoán lỗi mà không dùng thông tin test?**
7. **Counterfactual nào được sign-language experts xác nhận là chỉ thay một axis, và scorer có thắng geometry-only trên real candidates chứ không chỉ synthetic negatives hay không?**
8. **SignDINO checkpoint chính thức có sẵn, được phép dùng và tái lập preprocessing hay không; nếu không, video tower cụ thể nào sẽ được train?**
9. **Budget architecture/hyperparameter/compute được giới hạn thế nào để so sánh công bằng, và các random seed nào được khóa trước SGNify?**
10. **Nếu CUSP không thắng matched deterministic control hoặc form scorer không thêm giá trị, nhóm sẽ pivot sang strong baseline/reproducibility paper hay tiếp tục đổi endpoint?**

---

# G. Đánh giá cuối cùng

## G.1. Self-review sau chỉnh sửa

| Checklist | Đánh giá hiện tại | Căn cứ / blocker |
|---|---|---|
| Alignment với research objective | Đạt nhưng cần khóa endpoint | Pipeline trực tiếp xử lý upper-body/hand ambiguity; success rule còn thiếu. |
| Scientific validity | Chưa đạt hoàn toàn | Pseudo-label bias, scorer validity và fixed-benchmark inference chưa được kiểm chứng. |
| Technical feasibility | Có điều kiện | Mọi module có implementation path; adapter/data/checkpoint còn là blocker. |
| Module compatibility | Có điều kiện | Contracts đã rõ; MANO→SMPL-X và camera/frame transforms chưa test. |
| Mathematical correctness | Đạt ở mức specification | Gate direction và double-gating đã sửa; numerical (SO(3)) details vẫn cần unit tests. |
| Novelty clarity | Đạt ở mức hypothesis | Engineering vs proposed mechanism đã tách; novelty empirical/literature vẫn chưa xác nhận. |
| Reproducibility | Chưa đạt | Nhiều placeholder về data, checkpoints, architecture và training. |
| Citation integrity | Đạt có điều kiện | Không bịa citation; tất cả chỗ chưa chốt dùng `[CITATION NEEDED]`. |
| Terminology consistency | Đạt | “Form consistency”, “energy-normalized weights”, “upper body/hands” thay các overclaim. |
| Official SGNify compliance | Chưa đạt | Frame-selection semantics và DexAvatar reproduction chưa được xác nhận. |
| Conference readiness | Chưa đạt | Chưa có implementation, experiment, statistics hoặc final citations. |

## G.2. Kết luận

**Cần chỉnh sửa lớn trước khi triển khai.**

Đây không có nghĩa pipeline khái niệm là vô hiệu. Bản sửa đã tạo một phương pháp có interface, loss, training/inference split và falsification logic nhất quán. Tuy nhiên, ba yếu tố quyết định tính hợp lệ—official frame manifest, paired training data và tested coordinate adapter—vẫn chưa tồn tại dưới dạng artifact đã xác nhận. Thành phần form-consistency trung tâm của novelty cũng chưa có checkpoint/annotation validation. Vì vậy, bắt đầu full-scale training lúc này sẽ có nguy cơ tiêu tốn compute cho một pipeline không thể so sánh hoặc không đủ novelty.

## G.3. Hành động tiếp theo theo thứ tự ưu tiên

1. Gửi authors của DexAvatar/SGNify câu hỏi frame-manifest; khóa evaluator original, asset hashes và expected per-sign counts.
2. Reproduce DexAvatar score và viết unit-tested audited evaluator trước khi thay model.
3. Xác nhận data access/license; tạo signer/source-level manifest và kiểm tra duplicate/leakage.
4. Implement/test MANO–SMPL-X adapter; chạy `SMPLer-X + WiLoR` và simple temporal controls.
5. Trên development data, đo reliability curve và best-of-​K oracle headroom. Dừng G nếu không có headroom.
6. Chỉ khi bước 5 đạt, train/calibrate \(Q\) và một residual \(K=1\) control trước flow đa candidate.
7. Xây dựng bộ counterfactual nhỏ có expert validation; dừng \(S\) nếu không thắng geometry-only trên real candidates.
8. Khóa architecture, hyperparameters, seeds, primary endpoint, statistical plan và compute budget.
9. Chạy ablation A0–A11 trên internal test; chỉ sau đó mới mở SGNify test.
10. Viết claim theo kết quả thực tế: full method nếu interaction pass; geometry-only/pivot nếu \(S\) fail; reproducibility/strong-baseline result nếu learned components bị simple controls thống trị.

---

# Source and Evidence Map

The links below are primary papers, official project pages, or official repositories used to verify provenance. They are not a substitute for the final BibTeX audit.

1. [DexAvatar paper](https://arxiv.org/abs/2512.21054) and [official repository](https://github.com/kaustesseract/DexAvatar).
2. [SGNify project](https://sgnify.is.tue.mpg.de/) and [official repository](https://github.com/MPForte/SGNify).
3. [SMPL-X official project](https://smpl-x.is.tue.mpg.de/) and [MANO official project](https://mano.is.tue.mpg.de/).
4. [SMPLer-X official repository](https://github.com/caizhongang/SMPLer-X) and [project page](https://caizhongang.com/projects/SMPLer-X/).
5. [WiLoR project and paper](https://rolpotamias.github.io/WiLoR/).
6. [Tamaththul3D paper](https://arxiv.org/abs/2605.05367).
7. [HandFlow paper](https://arxiv.org/abs/2607.11221).
8. [MaskHand/MMHMR paper](https://arxiv.org/abs/2412.13393).
9. [Two-hand penetration-free diffusion paper](https://arxiv.org/abs/2503.17788).
10. [SignDINO CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Gan_Learning_Effective_Sign_Features_without_Text_for_Gloss-free_Sign_Language_CVPR_2026_paper.html).
11. [SignAvatars official repository](https://github.com/ZhengdiYu/SignAvatars) and [ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00653.pdf).
12. [ASLLVD official page](https://www.bu.edu/asllrp/av/dai-asllvd.html).
13. [3D-LEX paper](https://arxiv.org/abs/2409.01901).

## Verified citation keys used in the Revised Methods

```bibtex
@inproceedings{pavlakos2019smplx,
  title     = {Expressive Body Capture: 3D Hands, Face, and Body From a Single Image},
  author    = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {10975--10985},
  year      = {2019}
}

@article{romero2017mano,
  title   = {Embodied Hands: Modeling and Capturing Hands and Bodies Together},
  author  = {Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
  journal = {ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia)},
  volume  = {36},
  number  = {6},
  pages   = {245:1--245:17},
  year    = {2017}
}

@inproceedings{cai2023smplerx,
  title     = {{SMPLer-X}: Scaling Up Expressive Human Pose and Shape Estimation},
  author    = {Cai, Zhongang and Yin, Wanqi and Zeng, Ailing and Wei, Chen and Sun, Qingping and Wang, Yanjun and Pang, Hui En and Mei, Haiyi and Zhang, Mingyuan and Zhang, Lei and Loy, Chen Change and Yang, Lei and Liu, Ziwei},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {36},
  year      = {2023}
}

@inproceedings{potamias2025wilor,
  title     = {{WiLoR}: End-to-end 3D Hand Localization and Reconstruction in-the-wild},
  author    = {Potamias, Rolandos Alexandros and Zhang, Jinglei and Deng, Jiankang and Zafeiriou, Stefanos},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {12242--12254},
  year      = {2025}
}

@inproceedings{kundu2026dexavatar,
  title     = {{DexAvatar}: 3D Sign Language Reconstruction with Hand and Body Pose Priors},
  author    = {Kundu, Kaustubh and Barua, Hrishav Bakul and Robertson-Bell, Lucy and Cai, Zhixi and Stefanov, Kalin},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages     = {5842--5852},
  year      = {2026}
}

@inproceedings{forte2023sgnify,
  title     = {Reconstructing Signing Avatars From Video Using Linguistic Priors},
  author    = {Forte, Maria-Paola and Kulits, Peter and Huang, Chun-Hao and Choutas, Vasileios and Tzionas, Dimitrios and Kuchenbecker, Katherine J. and Black, Michael J.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2023}
}

@article{alghamdi2026tamaththul3d,
  title   = {{Tamaththul3D}: High-Fidelity 3D Saudi Sign Language Avatars from Monocular Video},
  author  = {Alghamdi, Eyad and Altuuaim, Sattam and Ghulam, Obay and Qutah, Abdulrahman and Basoodan, Yousef},
  journal = {arXiv preprint arXiv:2605.05367},
  year    = {2026}
}

@inproceedings{yu2024signavatars,
  title     = {{SignAvatars}: A Large-scale 3D Sign Language Holistic Motion Dataset and Benchmark},
  author    = {Yu, Zhengdi and Huang, Shaoli and Cheng, Yongkang and Birdal, Tolga},
  booktitle = {European Conference on Computer Vision},
  year      = {2024}
}

@article{xu2026handflow,
  title   = {{HandFlow}: Fully Generative 4D Hand Recovery with Flow Matching},
  author  = {Xu, Mingxi and Duan, Bowen and Gu, Yi and Shen, Zhengyang and Xu, Renjing and Yue, Yutao},
  journal = {arXiv preprint arXiv:2607.11221},
  year    = {2026}
}

@inproceedings{gan2026signdino,
  title     = {Learning Effective Sign Features without Text for Gloss-free Sign Language Translation},
  author    = {Gan, Shiwei and Liu, Xiao and Yin, Yafeng and Liu, Nan and Liu, Kuizhuang and Tuerdaken, Desibieer and Jiang, Zhiwei and Xie, Lei and Lu, Sanglu and Wen, Hongkai},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}

@inproceedings{ranum2024threeDLEX,
  title     = {3D-LEX v1.0 -- 3D Lexicons for American Sign Language and Sign Language of the Netherlands},
  author    = {Ranum, Oline and Otterspeer, Gomèr and Andersen, Jari I. and Belleman, Robert G. and Roelofsen, Floris},
  booktitle = {Proceedings of the LREC-COLING 2024 11th Workshop on the Representation and Processing of Sign Languages: Evaluation of Sign Language Resources},
  pages     = {290--301},
  year      = {2024}
}
```
