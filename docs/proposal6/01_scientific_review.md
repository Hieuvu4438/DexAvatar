# Phản biện khoa học độc lập: DexAvatar → CoSign4D

**Tài liệu được phản biện:** `Bao_cao_nghien_cuu_CoSign4D_DexAvatar(1).md/.docx`  
**Ngày rà soát nguồn:** 21-08-2026  
**Phạm vi:** tính đúng của lập luận, novelty, thiết kế phương pháp, dữ liệu–giám sát, giao thức đánh giá, khả năng tái lập và mức sẵn sàng để viết paper.

> Kết luận ngắn: tài liệu là một **research blueprint mạnh**, nhưng chưa phải một Methods section sẵn sàng nộp. Khuyến nghị **MAJOR REVISION / METHODOLOGICAL REDESIGN REQUIRED** trước khi đóng băng claim, mô hình và kế hoạch thực nghiệm.

## 1. Executive verdict

| Trục đánh giá | Kết luận | Lý do chính |
|---|---|---|
| Giá trị của research question | Mạnh | Occlusion, hand placement và self-contact là nút thắt thực của monocular sign reconstruction. |
| Đọc DexAvatar | Phần lớn đúng | DexAvatar dùng prior tay/thân tách rời và penalty chống xuyên; chưa mô hình hóa positive contact identity/event một cách tường minh. |
| Novelty hiện tại | **Trung bình, còn điều kiện** | Một số closest prior rất sát bị thiếu, đặc biệt PAPoseDiff/Goliath-SC, TUCH, HandX, Visibility-Aware HOI và các công trình hand–face contact. |
| Tính nhất quán toán học | **Cần tái thiết kế** | Factorization hiện tại trộn generative factors với amortized inference, có nguy cơ double-counting quan sát và dùng diffusion score như một scalar energy chưa được định nghĩa. |
| Dữ liệu và supervision | Chưa đủ | Chưa có protocol tạo/kiểm định dynamic contact-graph labels cho dữ liệu sign. |
| Giao thức đánh giá | Có nền tốt nhưng chưa operational | Nhiều metric đúng hướng, nhưng contact GT, primary endpoint, calibration, bootstrap unit và multiplicity chưa được chốt. |
| Khả năng tái lập | Chưa đạt | Thiếu evaluator source, hyperparameter, graph ontology, annotation rule, split manifest và compute budget. |
| Khuyến nghị | **Không khóa paper claim lúc này** | Chỉ tiến lên full model sau bốn gate: sửa evaluator, chứng minh feasibility của contact labels, thắng closest-prior baseline, rồi mới mở rộng joint model. |

## 2. Những điểm nên giữ

1. **Problem selection tốt.** Tài liệu không chỉ đề xuất “thêm một prior”, mà xác định đúng hai failure mode liên kết với nhau: visibility thấp và quan hệ contact động.
2. **Phân biệt placement với articulation là đúng và quan trọng.** Báo cáo root-aligned hand error cùng wrist-aligned articulation error sẽ ngăn metric che khuất sai lệch vị trí toàn cục.
3. **Tư duy falsification tốt.** Các câu hỏi “contact có thực sự giúp vùng occluded không?”, “graph động có hơn graph tĩnh không?” và “multi-hypothesis có tạo đa dạng hữu ích không?” phù hợp với reviewer conference.
4. **Có risk register và go/no-go mindset.** Đây là ưu điểm hiếm của một bản đề xuất sớm; chỉ cần chuyển các ngưỡng quản trị thành tiêu chí dựa trên pilot variance và practical effect size.
5. **Story sign-specific có tiềm năng.** Tính mới thuyết phục nhất không nằm ở từng module riêng lẻ, mà ở việc suy luận đồng thời trajectory SMPL-X và contact-event sequence dưới bằng chứng visibility-aware trong miền sign language.

## 3. Audit các claim quan trọng

| Claim trong bản gốc | Kết quả kiểm tra | Hành động đề xuất |
|---|---|---|
| DexAvatar dùng SMPLer-X/HaMeR initialization, Sapiens/HaMeR 2D cues, prior thân và tay tách rời | **Đã xác nhận** trong paper chính thức | Giữ, nhưng trích paper chính thức thay vì chỉ repo/summary. |
| DexAvatar có 57 German signs, 2.872 central frames; TR-V2V 30,13 / 13,53 / 13,08 mm | **Đã xác nhận** | Giữ, nêu rõ benchmark và alignment protocol. |
| DexAvatar “contact-aware” nhưng objective chủ yếu là interpenetration avoidance, chưa có positive contact identity/event | **Được hỗ trợ** bởi objective công bố | Viết chính xác: collision avoidance không đồng nghĩa positive contact modeling. Tránh nói paper “không có contact” theo nghĩa tuyệt đối. |
| Evaluator triệt tiêu common wrist center và lặp lại centered articulation error | **Đúng về đại số nếu code đúng như mô tả**, nhưng file evaluator không được cung cấp | Hạ từ `[V]` thành `[UNVERIFIED—SOURCE FILE REQUIRED]`; không nêu số dòng cho tới khi kiểm tra file thực. |
| DexAvatar là “Best Paper Award Finalist” | **Chưa xác nhận độc lập** từ nguồn hội nghị đã kiểm tra | Gắn `[AUTHOR VERIFY]` hoặc bỏ khỏi paper claim. |
| DPoser-X có temporal modeling “Partial” | **Không chính xác** | DPoser-X là whole-body **pose** prior; không chứng minh trajectory/temporal prior. Đổi thành `No / not modeled`. |
| CoSign4D là tên riêng an toàn | **Có rủi ro nhầm lẫn** | Đã tồn tại CoSIGN cho diffusion inverse problems. Cần quyết định đổi tên hoặc disambiguate trước khi public release. |
| Model tạo “calibrated posterior” | **Chưa được hỗ trợ** | Chỉ dùng “multiple conditional hypotheses” cho tới khi có normalized likelihood hoặc protocol calibration đúng nghĩa. |
| SiLVERScore đo semantic fidelity của reconstruction | **Chưa được xác nhận ngoài miền** | Chỉ dùng secondary/exploratory trên standardized renders và phải có validation với human/sign-language judgment. |

Nguồn neo cho phần audit: [DexAvatar—WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html), [DexAvatar—arXiv full text](https://arxiv.org/html/2512.21054v1), [DPoser-X](https://arxiv.org/html/2508.00599v2), [SiLVERScore](https://aclanthology.org/2025.ranlp-1.54/), [CoSIGN](https://arxiv.org/abs/2407.12676).

## 4. Các vấn đề bắt buộc phải sửa

### CR-01 — Evidence chain của evaluator chưa hợp lệ

Bản gốc ghi đã audit `evaluate_new_fitting(3).py`, bao gồm cả số dòng, nhưng tệp này không nằm trong bộ đầu vào hiện có. Lập luận đại số là hợp lý: nếu source và target đều được center độc lập tại wrist rồi cộng cùng một center, common center sẽ triệt tiêu trong hiệu, khiến metric không còn đo placement. Tuy nhiên, chưa thể xác nhận:

- code thực sự dùng đúng phép biến đổi được mô tả;
- broadcasting, joint indices và unit conversion có đúng không;
- các nhánh one-hand/two-hand có dùng cùng protocol không;
- kết quả đã báo trước đây bị ảnh hưởng ở mức nào.

**Yêu cầu tối thiểu:** cung cấp evaluator, tạo unit test với ba perturbation độc lập—global wrist translation, local finger articulation và rigid hand rotation—rồi chứng minh metric mới phản ứng đúng với từng perturbation.

### CR-02 — Factorization và inference hiện không mô tả cùng một mô hình

Bản gốc viết:

\[
p(X,C\mid Y)\propto p_\phi(X\mid C,M_{vis})\,p_\eta(C\mid Y)\prod_k\psi_k(Y_k\mid X,M_{vis}).
\]

Nhưng thuật toán sau đó lại cập nhật contact graph từ cả ảnh lẫn geometry mẫu. Ba vấn đề phát sinh:

1. `p_eta(C|Y)` là một posterior/discriminative predictor, không phải prior của generative model. Nếu dùng nó như factor rồi lại dùng các likelihood chứa cùng `Y`, bằng chứng ảnh có thể bị tính hai lần.
2. Nếu `C` được cập nhật từ `X`, conditional đúng của inference network phải có dạng `q_eta(C|Y,X)`; hoặc target distribution phải có compatibility factor `psi_C(X,C)` rõ ràng.
3. Diffusion model cung cấp score `∇_X log p(X)` hoặc denoising prediction, không mặc nhiên cung cấp một scalar `E_diff(X)`. Vì vậy hạng `lambda_score E_diff` chưa có nghĩa toán học nếu không định nghĩa score-distillation/denoising regularizer cụ thể.

**Sửa đề xuất:** tách rõ target distribution và proposal network:

\[
\pi(X,C\mid Y,M)\propto
p_\phi(X\mid C,M)\;p_\rho(C)\;\psi_{geo}(X,C)
\prod_k p_k(Y^{(k)}\mid X,M^{(k)}),
\]

trong đó `p_rho(C)` là semi-Markov event prior; `q_eta(C|Y,X)` chỉ là amortized proposal/initializer. Inference dùng posterior score guidance thay vì gọi score là một energy chưa chuẩn hóa. Bản Methods viết lại đi theo formulation này.

### CR-03 — Novelty review bỏ sót closest prior trực tiếp

Thiếu quan trọng nhất là [Generative Modeling of Shape-Dependent Self-Contact Human Poses / PAPoseDiff](https://arxiv.org/html/2509.23393v1): công trình này xây Goliath-SC với khoảng 383 nghìn SMPL-X self-contact poses, dùng shape-conditioned latent diffusion và còn trình bày single-view refinement với diffusion prior. Vì vậy các claim rộng như “whole-body contact-aware diffusion prior” hoặc “diffusion for self-contact refinement” đã không còn mới.

Các prior khác cần được đưa vào related work và baseline design:

- [On Self-Contact and Human Pose / TUCH](https://arxiv.org/abs/2104.03176): self-contact constraints và contact-aware SMPL-X fitting.
- [HandX](https://arxiv.org/html/2603.28766v1): bimanual motion generation với contact events và contact metrics.
- [Visibility Aware Human-Object Interaction Tracking](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_Visibility_Aware_Human-Object_Interaction_Tracking_From_Single_RGB_Camera_CVPR_2023_paper.pdf): tracking người–vật–contact dưới visibility/occlusion.
- [GraphiContact](https://arxiv.org/html/2603.20310v1): contact prediction đồng thời với 3D mesh reconstruction trong điều kiện ảnh nhiễu/occlusion.
- [Learning Dense Hand Contact Estimation from Imbalanced Data / HACO](https://arxiv.org/html/2505.11152v2): dense hand contact và xử lý class/spatial imbalance.
- [Decaf](https://dl.acm.org/doi/10.1145/3618329) và [DICE](https://arxiv.org/abs/2406.17988): hand–face interaction/contact reconstruction.
- [Pose Priors from Language Models / ProsePose](https://arxiv.org/html/2405.03689v2): image-derived body-region contact pairs được chuyển thành optimization loss.

**Novelty còn lại có thể bảo vệ:** một **latent temporal contact-event process dành cho sign**, được suy luận luân phiên với holistic SMPL-X trajectory và visibility-weighted observations. Tính mới không nên đặt ở “contact”, “graph”, “whole body”, “diffusion” hay “visibility” riêng lẻ.

### CR-04 — Chưa có contact supervision có thể audit

Tài liệu chưa chỉ ra SGNify hoặc benchmark sign hiện có cung cấp dynamic contact graph ground truth. Tự sinh nhãn từ pseudo-SMPL-X có nguy cơ đóng băng chính lỗi hình học mà phương pháp muốn sửa. Methods phải mô tả:

- ontology của node/edge và danh sách edge hợp lệ;
- contact threshold theo khoảng cách, normal consistency và thời lượng tối thiểu;
- cách phân biệt touch với near-contact và penetration;
- onset/hold/release annotation;
- double annotation, adjudication và inter-annotator agreement;
- hard negatives, class imbalance và uncertain labels;
- split theo signer/sign để tránh leakage.

Nếu không thể tạo một gold subset đủ tin cậy, claim về contact graph chỉ nên là latent mechanism, không được báo `contact F1` hoặc `patch-ID accuracy` như metric chính.

## 5. Novelty map sau khi cập nhật

| Phương pháp | Sign-specific | Temporal | Positive contact | Dynamic event graph | Diffusion prior | Monocular inverse reconstruction |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| DexAvatar | ✓ | △ smoothing | ✗ | ✗ | ✗ | ✓ |
| TUCH / SMPLify-XMC | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| PAPoseDiff / Goliath-SC | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ refinement |
| DPoser-X | ✗ | ✗ | △ plausibility | ✗ | ✓ | ✓ downstream prior |
| HandX | ✗ | ✓ | ✓ | ✓ hand events | ✓ | ✗ generation |
| Visibility-Aware HOI | ✗ | ✓ | ✓ person–object | △ tracks | ✗ | ✓ |
| GraphiContact | ✗ | ✗ | ✓ person–scene | graph encoding | ✗ | ✓ |
| Decaf / DICE | ✗ | △ / image | ✓ hand–face | ✗ | ✗ | ✓ |
| **Đề xuất đã thu hẹp** | **✓** | **✓** | **✓** | **✓ onset–hold–release** | **✓** | **✓** |

`△` nghĩa là chỉ bao phủ một phần hoặc không phải contribution được chứng minh trực tiếp. Bảng này là bản đồ chức năng, không thay thế ablation hoặc legal/priority novelty analysis.

## 6. Các major issue về thiết kế thực nghiệm

### M-01 — Visibility chưa được định nghĩa và hiệu chỉnh

`M_vis` hiện là ký hiệu giàu ý nghĩa nhưng thiếu phép đo. Cần xác định nó đến từ segmentation visibility, detector confidence, depth ordering hay learned reliability. Confidence của detector không đồng nghĩa xác suất visibility đã calibrated. Nên:

1. tạo occlusion/visibility labels trên validation subset;
2. fit hoặc kiểm tra calibration của reliability predictor;
3. dùng reliability để temper từng observation likelihood;
4. báo performance theo visibility bins đã khóa trước.

Classifier-free dropout có thể cải thiện robustness, nhưng không tự tạo calibrated uncertainty.

### M-02 — Contact dynamics chưa phải một temporal model hoàn chỉnh

Hysteresis theo threshold là heuristic hữu ích cho pseudo-labeling, nhưng không đủ để claim latent event model. Nên dùng semi-Markov/CRF state sequence với ít nhất `no-contact`, `onset`, `hold`, `release`, kèm minimum duration hoặc transition constraints. Điều này cho phép định nghĩa transition-timing error và tránh graph flicker.

### M-03 — Metric temporal và uncertainty có thể dẫn đến kết luận sai

- Raw jerk thấp có thể chỉ phản ánh oversmoothing; ưu tiên **jerk error so với GT**, spectral distance hoặc transition-timing error.
- “NLL/energy proxy” không phải NLL nếu model không có normalized density.
- AUSE đo chất lượng ranking uncertainty nhiều hơn calibration. Cần thêm empirical coverage–width, risk–coverage và calibration curve.
- Với `K` hypotheses, phải tách **top-1**, **mean**, và **oracle/best-of-K**; oracle chỉ là upper bound, không phải deployed performance.
- Patch-ID accuracy và contact F1 chỉ hợp lệ trên subset có ground truth và định nghĩa tolerance cố định.

### M-04 — Thống kê chưa xử lý correlation và multiplicity

Frame không phải đơn vị bootstrap độc lập. Nên bootstrap ở cấp signer/sign sequence; nếu có nhiều signer, dùng hierarchical cluster bootstrap. Chọn trước một primary endpoint; các metric còn lại là secondary/exploratory. Nếu diễn giải nhiều p-value, dùng Holm hoặc FDR. Báo effect size và confidence interval, không chỉ tỷ lệ phần trăm cải thiện.

### M-05 — Baseline chưa đủ để chứng minh mechanism

Ít nhất phải có matched-capacity comparisons:

1. PAPoseDiff-style self-contact prior + temporal smoothing;
2. DPoser-X/holistic prior + static contact loss;
3. trajectory diffusion không graph;
4. graph-conditioned model nhưng graph tĩnh;
5. dynamic graph nhưng bỏ visibility weighting;
6. full model;
7. oracle-contact upper bound trên gold subset.

Nếu full model chỉ thắng nhờ nhiều parameter, nhiều data hoặc nhiều optimization steps, claim mechanism không đứng vững.

### M-06 — Tên và ngôn ngữ claim đang vượt quá bằng chứng

“Posterior” và “calibrated multiple hypotheses” là claim toán học/đánh giá mạnh. Trước khi có normalized model và calibration protocol, dùng:

> *a visibility-aware structured conditional model for dynamic-contact-guided sign reconstruction*

Tên `CoSign4D` cũng nên được xét lại vì [CoSIGN](https://arxiv.org/abs/2407.12676) đã dùng trong diffusion inverse problems—một vùng kỹ thuật đủ gần để gây nhầm lẫn khi tìm kiếm và review.

## 7. Lộ trình go/no-go được khuyến nghị

| Gate | Thí nghiệm tối thiểu | Điều kiện để đi tiếp |
|---|---|---|
| G0 — Metric validity | Unit test evaluator trên translation/articulation/rotation; rerun baseline | Metric phản ứng đúng và kết quả cũ được định lượng lại. |
| G1 — Label feasibility | Gold contact subset nhỏ nhưng double-annotated | Agreement và error analysis đủ để định nghĩa contact GT có ý nghĩa. |
| G2 — Closest-prior test | PAPoseDiff/DPoser-X + static/dynamic contact baselines trên cùng data/compute | Dynamic event representation tạo gain ở contact+occluded subset, không chỉ ở oracle metric. |
| G3 — Minimal mechanism | No-graph, static-graph, dynamic-graph, visibility ablations | Gain lặp lại qua cluster bootstrap và không đi kèm regression thực tiễn ở non-contact frames. |
| G4 — Full posterior sampling | K hypotheses + preregistered ranking | Top-1 cải thiện; best-of-K chỉ được báo như upper bound; uncertainty có coverage/risk calibration. |

Các ngưỡng `≥5%` hoặc `≤1%` trong bản gốc nên được xem là management thresholds tạm thời. Sau pilot, thay chúng bằng smallest practically important effect và confidence interval phù hợp với variance thực tế.

## 8. Những thông tin tác giả phải cung cấp

- `[AUTHOR INPUT REQUIRED]` File evaluator thực tế và commit/hash đã dùng để sinh bảng kết quả.
- `[AUTHOR INPUT REQUIRED]` Dataset manifest, license, signer/sign splits và cách xử lý frame trùng.
- `[AUTHOR INPUT REQUIRED]` Contact patch ontology, threshold/tolerance và annotation protocol.
- `[AUTHOR INPUT REQUIRED]` Window length, overlap, state parameterization, architecture, noise schedule và số hypotheses.
- `[AUTHOR INPUT REQUIRED]` Training curriculum, sampling ratio, loss weights và compute budget.
- `[AUTHOR INPUT REQUIRED]` Primary endpoint, smallest effect đáng quan tâm và statistical analysis plan.
- `[AUTHOR DECISION REQUIRED]` Tên phương pháp cuối cùng; giữ `CoSign4D` hay đổi để tránh nhầm với CoSIGN.
- `[AUTHOR VERIFY]` Award/finalist status của DexAvatar nếu muốn nhắc trong paper.

## 9. Khuyến nghị cuối

Không nên bỏ hướng nghiên cứu. Gap “dynamic sign contact under occlusion” vẫn có giá trị, nhưng paper chỉ thuyết phục nếu chuyển từ một tổ hợp module hấp dẫn sang một hypothesis có thể bác bỏ:

> **Khi quan sát bàn tay bị che khuất, một contact-event sequence có cấu trúc và được suy luận đồng thời với trajectory sẽ giảm lỗi hand placement mà không làm xấu articulation hoặc non-contact motion, so với holistic diffusion prior và static contact constraints có cùng capacity/compute.**

Đây nên là claim trung tâm. Self-contact diffusion, graph encoding, visibility weighting và multi-hypothesis sampling là phương tiện kiểm tra claim đó, không phải bốn novelty claim độc lập.
