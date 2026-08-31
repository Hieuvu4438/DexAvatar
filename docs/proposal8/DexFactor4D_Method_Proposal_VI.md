# DexFactor-4D: method proposal có kiểm chứng cho DexAvatar

**Bản nghiên cứu kỹ thuật và preregistration, cập nhật đến 25-08-2026**  
**Đối tượng:** DexAvatar / SGNify / TR-V2V / tái dựng 3D sign language đơn camera  
**Trạng thái bằng chứng:** đã đọc paper DexAvatar kèm supplement, paper và supplement SGNify, audit tĩnh code công khai, đối chiếu paper–code, và đọc paper/code của các phương pháp gần nhất. Chưa chạy benchmark SGNify vì dữ liệu, checkpoint hạn chế quyền truy cập và evaluator chính thức không có trong repo công khai.

---

## 0. Kết luận điều hành

### Method được chốt

Tôi đề xuất **DexFactor-4D**, một bộ tối ưu hóa chuỗi SMPL-X theo factor graph, không huấn luyện lại DexAvatar và không tạo một prior “tưởng tượng”. Method dùng:

1. các prior SignBPoser/SignHPoser sẵn có của DexAvatar;
2. các quan sát frozen từ SMPLer-X, Sapiens, HaMeR và WiLoR;
3. ghép vết tay trái/phải xuyên thời gian thay cho quyết định cứng từng frame;
4. quan sát bàn tay 3D đầy đủ theo hệ xương, thay cho loss chỉ dùng trục `z`;
5. silhouette và thứ tự độ sâu tương đối từ Sapiens khi hai tay che nhau;
6. regularization vận tốc/gia tốc trên SO(3), có chuẩn hóa theo khoảng thời gian thực và độ tin cậy;
7. biomechanical barrier, collision và contact-persistence chỉ kích hoạt khi có bằng chứng;
8. PAD-Hand công khai làm **proposal temporal có switch**, chỉ ở cửa sổ độ tin cậy thấp, không được quyền ghi đè quan sát tốt.

Đây là lựa chọn có xác suất thành công hợp lý nhất vì nó trực tiếp sửa các failure mode có bằng chứng trong code DexAvatar, đồng thời chỉ mượn những thành phần đã được các công trình công bố chứng minh trên bài toán gần. Nó không cần gắn nhãn SGNify, không fine-tune trên test, và mọi thành phần mới đều có ablation độc lập.

### Điều tôi không thể trung thực bảo đảm trước thí nghiệm

Không có quy trình đọc paper nào có thể **bảo đảm** điểm TR-V2V tốt hơn trước khi chạy đúng evaluator trên đúng 57 sign. Tôi có thể bảo đảm ba điều kiểm chứng được hơn:

- proposal không phụ thuộc một module hay dataset không tồn tại;
- mỗi thay đổi đều truy về một failure mode trong code hoặc một kết quả thực nghiệm đã công bố;
- protocol có tiêu chí bác bỏ trước khi xem test, nên kết quả xấu sẽ không bị che hoặc “tune cho đẹp”.

Vì ground truth SGNify đôi lúc có bàn tay phi giải phẫu, một method hợp lý hơn về sinh cơ học thậm chí có thể bị TR-V2V phạt. Vì vậy, mục tiêu chính vẫn là TR-V2V; plausibility và temporal quality chỉ là metric phụ, không được dùng để thay thế một kết quả TR-V2V không tốt.

---

## 1. Quy trình nghiên cứu và phạm vi bằng chứng

### 1.1 Câu hỏi nghiên cứu

> Với cùng đầu vào monocular RGB và cùng topology SMPL-X, thay đổi nhỏ nhất nhưng có cơ sở nào có thể giảm TR-V2V ở upper body, left hand và right hand so với **code DexAvatar đã sửa đúng contract**, mà không sử dụng SGNify test để train/tune?

Các câu hỏi phụ:

- DexAvatar paper thực sự tối ưu biến nào, dùng prior nào, và đánh giá ra sao?
- repo công khai có thực hiện đúng mô tả paper và đúng benchmark contract không?
- failure mode nào có thể sửa bằng quan sát, temporal model, interaction prior hoặc uncertainty gating đã được chứng minh ở công trình khác?
- dataset nào có thể dùng để train/tune mà không gây test leakage?
- metric/protocol nào đủ chặt để phân biệt cải thiện thật với bug fix, dropped-frame bias hoặc hyperparameter selection trên test?

### 1.2 Quy trình đã thực hiện

Quy trình theo chuỗi **research architecture → bibliography → source verification → code inspection → evidence synthesis → devil's-advocate review → editor decision**:

1. khóa paper, commit và benchmark contract trước khi đề xuất method;
2. ưu tiên nguồn gốc: paper chính thức, supplement, project/repo chính thức; không dùng blog làm bằng chứng kỹ thuật;
3. trích riêng các claim về data, training, loss, metric và code availability;
4. audit luồng thực thi thật của repo, không chỉ README;
5. lập evidence matrix: kết quả nào chuyển được, kết quả nào không;
6. dựng proposal tối thiểu;
7. phản biện theo bốn hướng: leakage, domain shift, benchmark mismatch, alternative explanation;
8. bỏ các nhánh không vượt qua phản biện;
9. preregister ablation và tiêu chí thành công trước khi mở SGNify test.

### 1.3 Corpus chính và mức tin cậy

| Nhóm | Nguồn đã đọc | Paper | Code | Vai trò trong quyết định |
|---|---|---:|---:|---|
| Target | [DexAvatar, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html), [repo](https://github.com/kaustesseract/DexAvatar) | Có, kèm supplement | Có | Target, baseline, prior, audit |
| Benchmark | [SGNify, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.pdf), [supplement](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Forte_Reconstructing_Signing_Avatars_CVPR_2023_supplemental.pdf), [repo](https://github.com/MPForte/SGNify) | Có | Có, nhưng không có evaluator/mask/manifest hoàn chỉnh | Định nghĩa TR-V2V và dữ liệu |
| Sign data | [SignAvatars, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00653.pdf), [repo](https://github.com/ZhengdiYu/SignAvatars); [How2Sign](https://how2sign.github.io/) | Có | Annotation/visualization một phần | Nguồn prior body hiện hữu, dev phụ |
| Hand image | [HaMeR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html), [repo](https://github.com/geopavlakos/hamer); [WiLoR, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.html), [repo](https://github.com/rolpotamias/WiLoR) | Có | Có | Ensemble observation |
| Hai tay | [ACR, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_ACR_Attention_Collaboration-Based_Regressor_for_Arbitrary_Two-Hand_Reconstruction_CVPR_2023_paper.pdf); [4DHands](https://arxiv.org/html/2405.20330v1) | Có | ACR có; 4DHands không thấy release chính thức | Bằng chứng về coupling hai tay |
| Chuỗi tay | [Dyn-HaMR, CVPR 2025 Highlight](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html), [repo](https://github.com/ZhengdiYu/Dyn-HaMR); [PAD-Hand, CVPR 2026 Highlight](https://openaccess.thecvf.com/content/CVPR2026/html/Ismayilzada_PAD-Hand_Physics-Aware_Diffusion_for_Hand_Motion_Recovery_CVPR_2026_paper.html), [repo](https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026) | Có | Có, với khác biệt paper–demo | Temporal/physics proposal |
| Biomechanics | [KNOWN-Hand, ECCV 2024](https://arxiv.org/html/2407.12307v1), [repo](https://github.com/zhangy76/KNOWN-Hand); [DIP-Hand, ICCV 2025](https://arxiv.org/html/2508.01835v1), [repo](https://github.com/zhangy76/DIP-Hand) | Có | Có nhưng không hoàn chỉnh như một drop-in module | Công thức/gating, không copy mù |
| Multi-cue | [A2P / From 2D Alignment to 3D Plausibility, CVPR 2026](https://arxiv.org/html/2503.17788), [project](https://gaogehan.github.io/A2P/); [Sapiens, ECCV 2024](https://arxiv.org/html/2408.12569v1), [repo](https://github.com/facebookresearch/sapiens) | Có | Sapiens có; A2P chưa thấy code chính thức | Segmentation/depth evidence |
| Whole-body temporal | [Motions as Queries, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Motions_as_Queries_One-Stage_Multi-Person_Holistic_Human_Motion_Capture_CVPR_2025_paper.pdf); [Neural Sign Actors](https://arxiv.org/abs/2312.02702) | Có | Không có fitter đầy đủ | Đối chiếu, không là dependency |

**Giới hạn corpus:** đây là systematic deep review có mục tiêu, không phải systematic review toàn bộ 3D human reconstruction. Tiêu chí chọn là liên quan trực tiếp tới sign/hand/SMPL-X/temporal và có chi tiết đủ để kiểm tra chuyển giao.

---

## 2. Benchmark contract: SGNify và TR-V2V

### 2.1 Dataset SGNify dùng để đánh giá

Theo paper/supplement SGNify:

- 57 isolated signs tiếng Đức (DGS), một signer bản ngữ thuận tay phải;
- Vicon 120 fps, camera RGB frontal 4112×3008 ở 60 fps;
- tổng 16.608 frame mocap và 8.304 frame RGB;
- input benchmark được downsample thành 514×300 ở 30 fps và crop phía trên pelvis;
- chỉ đoạn biểu đạt trung tâm được đánh dấu thủ công và chấm: tổng 2.872 RGB frame;
- ground truth là SMPL-X cùng topology với prediction.

Do chỉ có một signer, một ngôn ngữ và 57 isolated signs, kết quả này không chứng minh generalization sang signer/ngôn ngữ/continuous signing khác.

### 2.2 Định nghĩa chính xác của TR-V2V

Với prediction vertices \(V_t\), ground truth \(V_t^*\), subset vertex \(S_r\) của region \(r\), benchmark thực hiện **translation-only alignment cho từng frame** rồi lấy mean per-vertex distance:

\[
\mathrm{TRV2V}_r =
\frac{1}{\sum_t |S_r|}
\sum_t \sum_{i\in S_r}
\left\| (V_{t,i}-c_t) - (V^*_{t,i}-c_t^*) \right\|_2 .
\]

Không Procrustes rotation, không scale alignment. Ba region báo cáo:

- `UBody (-F)`: vertices phía trên pelvis, gồm head nhưng loại face;
- `LHand`;
- `RHand`.

Paper SGNify nói face bị loại vì hệ mocap chỉ dùng 27 marker mặt. Exact mask được minh họa trong supplement, nhưng evaluator, danh sách vertex và manifest 2.872 frame không nằm trong repo công khai đã kiểm tra. Vì vậy:

> Chỉ được gọi là **official TR-V2V** khi dùng đúng manifest và mask do tác giả cung cấp. Nếu phải dựng mask tương đương, báo cáo phải ghi “internal translation-aligned V2V”, không được nhập nhằng với bảng paper.

### 2.3 Bảng baseline được công bố

| Method | UBody (-F) ↓ | LHand ↓ | RHand ↓ |
|---|---:|---:|---:|
| FrankMoCap | 78.07 | 20.47 | 19.62 |
| PIXIE | 60.11 | 25.02 | 22.42 |
| PyMAF-X | 68.61 | 21.46 | 19.19 |
| SMPLify-SL | 56.07 | 22.23 | 18.83 |
| SGNify | 55.63 | 19.22 | 17.50 |
| OSX | 47.32 | 18.34 | 18.12 |
| Neural Sign Actors | 46.42 | 16.17 | 15.23 |
| EVA* | 40.38 | 13.73 | 13.68 |
| **DexAvatar** | **30.13** | **13.53** | **13.08** |

Theo phép tính từ bảng, DexAvatar giảm so với EVA* khoảng 25,38% / 1,46% / 4,39% ở ba region. Khoảng cách ở hai tay, đặc biệt left hand, nhỏ; chỉ một thay đổi routing hoặc frame inclusion cũng có thể ảnh hưởng kết luận. Vì vậy reproducibility contract quan trọng hơn một cải thiện vài phần mười mm.

---

## 3. DexAvatar: paper thực sự làm gì

### 3.1 Pipeline

Pipeline paper:

1. **SMPLer-X** ước lượng SMPL-X/camera/body ban đầu;
2. **Sapiens** cung cấp 2D body keypoints;
3. **HaMeR** cung cấp 2D/3D hand keypoints và MANO pose;
4. classifier của **SGNify** quyết định one-hand/two-hand;
5. **SignBPoser** và **SignHPoser** regularize body và hai tay trong không gian latent;
6. fitting thêm reprojection, interpenetration, temporal và biomechanical terms.

### 3.2 Hai prior

**SignBPoser** là VAE cho body pose. Dữ liệu là pseudo-3D SMPL-X từ SignAvatars/How2Sign, được lọc bằng ràng buộc sinh cơ học và signer-space. **SignHPoser** là VAE cho hand pose, dùng mocap riêng:

- 8 người tham gia: 6 người thành thạo Auslan và 2 người thông thạo ASL;
- 93 từ fingerspelling;
- Vicon 9 camera và Manus gloves;
- retarget sang SMPL-X.

Cả hai dùng ba linear layer, width 512, Adam \(10^{-3}\). Body latent 33; hand latent 23 trong cấu hình được chọn. Đây là prior domain-specific có thật; không có lý do đủ mạnh để bỏ chúng và train một latent prior mới bằng pseudo-label tương tự.

### 3.3 Ablation quan trọng

Body prior trong pipeline:

| Variant | FBody | UBody | UBody(-H) | UBody(-F) |
|---|---:|---:|---:|---:|
| BPu | 43.18 | 29.95 | 44.72 | 34.06 |
| BPf | 42.32 | 26.78 | 41.35 | 30.28 |
| BPf+bio | 42.38 | 26.93 | 41.88 | 30.44 |

Hand prior trong pipeline:

| Variant | UBody(-F) | LHand | RHand |
|---|---:|---:|---:|
| HPu | 31.34 | 14.19 | 13.92 |
| HPf | 30.17 | 13.55 | **13.06** |
| HPf+bio | **30.13** | **13.53** | 13.08 |

Kết luận đúng từ bảng không phải “biomechanics luôn tốt”. Filtering data giúp rõ; thêm biomechanical loss có hiệu ứng rất nhỏ, và right hand xấu đi 0,02 mm. Vì vậy DexFactor-4D dùng biomechanics như barrier có gate, không như prior nặng trên mọi frame.

### 3.4 Hai vấn đề báo cáo cần được coi là rủi ro

1. Supplement viết rõ chọn best hyperparameter trên cả `Evaluation (DEV) and TEST data`. Nếu hiểu theo nghĩa đen, đây là test-set selection. Protocol mới phải chọn trên dev, khóa toàn bộ cấu hình, rồi chỉ chạy SGNify test một lần.
2. Main paper nói HPf+bio làm right hand xấu 0,2% so với HPf; supplement S5 lại nói VPoser HPf+bio cải thiện right hand 1,7%. Hai kết quả có thể thuộc hai prior khác nhau, nhưng phần diễn giải dễ gây lẫn. Bản tái lập phải gắn mỗi claim với đúng decoder/checkpoint/table.

---

## 4. Audit code DexAvatar tại commit công khai

Audit dùng commit `a0dfd427f60f5811aadb35c8657b3856d47f56b5` của repo DexAvatar. Đây là **static audit** theo execution path `run_dexavatar.py → Full_running_command.sh → dexavatar_fitting`; chưa chạy numerical reproduction vì thiếu assets/checkpoints/data benchmark.

### 4.1 Findings có thể thay đổi kết quả

| Mức | Finding | Bằng chứng code | Tác động |
|---|---|---|---|
| Critical | Main config đặt `data_3d_weights = [0,0,0]` | [`fit_smplx_vposer_x.yaml#L54-L57`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/cfg_files/fit_smplx_vposer_x.yaml#L54-L57) | Nhánh hand 3D không đóng góp trong cấu hình chính, dù paper mô tả 3D hand term. |
| Critical | Không tìm thấy implementation hand-biomechanics trong fitting path | [repo tại commit audit](https://github.com/kaustesseract/DexAvatar/tree/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting) | Paper mô tả body và hand biomechanical regularizer, code chỉ có body term. |
| Critical | Khi one-hand/right có nhiều detection, code tìm `idx_r` nhưng đọc pose và 3D ở index cố định `1` | [`data_parser.py#L484-L518`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/data_parser.py#L484-L518) | Detector order không được bảo đảm; có thể gắn nhầm tay. |
| Critical | Index hand “3D” one-hand không tương ứng một bàn tay hoàn chỉnh | [`fitting.py#L466-L496`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fitting.py#L466-L496) | `53:63` chủ yếu là fingertip của cả hai tay; `12:42` trộn internal joints của cả hai tay theo mapping common-joints. |
| High | “3D” loss chỉ lấy `[..., 2:3]`, tức depth z | [`fitting.py#L468-L496`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fitting.py#L468-L496) | Bỏ toàn bộ hình học x/y và direction 3D; yếu khi self-occlusion. |
| High | Frame thiếu HaMeR hoặc SMPLer-X bị loại im lặng | [`data_parser.py#L180-L199`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/data_parser.py#L180-L199) | Có thể đổi denominator và temporal gap; nếu evaluator chỉ đọc output tồn tại sẽ gây selection bias. |
| High | Numeric sort bị ghi đè bằng lexicographic `sorted(temp)` | [`data_parser.py#L140-L145`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/data_parser.py#L140-L145), [`#L199`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/data_parser.py#L199) | Sai thứ tự nếu filename không zero-pad, làm temporal prior vô nghĩa. |
| High | Resume bằng output image bỏ qua frame nhưng không restore `joints_temp` | [`main.py#L301-L330`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/main.py#L301-L330) | Kết quả resume khác clean run. |
| High | File `.pkl` lưu decoded body pose nhưng không thay hand pose bằng decoded SignHPoser; OBJ lại dùng decoded hand | [`fit_single_frame.py#L593-L609`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fit_single_frame.py#L593-L609), [`#L627-L640`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fit_single_frame.py#L627-L640) | Metric từ PKL và metric từ mesh OBJ có thể đánh giá hai pose khác nhau. |
| Medium | Two-hand branch lặp `range(2)` nhưng chỉ kiểm tra có ít nhất một detection | [`data_parser.py#L397-L421`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/data_parser.py#L397-L421) | Có thể crash khi classifier báo two-hand nhưng detector chỉ thấy một tay. |
| Medium | Fallback frame đầu dùng `prev_* = None` | [`data_parser.py#L528-L542`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/data_parser.py#L528-L542) | Có thể crash hoặc tạo dữ liệu rỗng nếu detection đầu tiên sai phía. |
| Medium | `normalize_points_torch` không có epsilon | [`fitting.py#L457-L474`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fitting.py#L457-L474) | Zero variance có thể sinh NaN. |
| Medium | Temporal loss là chênh axis-angle frame trước, hệ số hard-coded 2000 | [`fitting.py#L499`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fitting.py#L499) | Không đúng geodesic SO(3), không có \(\Delta t\), acceleration hay hand temporal. |
| Medium | Init anchors là 1200 ở cả ba stage | [`fit_smplx_vposer_x.yaml#L58-L73`](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/cfg_files/fit_smplx_vposer_x.yaml#L58-L73) | Latent khó rời initial estimate kể cả khi quan sát khác cho tín hiệu tốt. |

### 4.2 Những biến nào thực sự được tối ưu

Trong `fit_single_frame.py`, optimizer nhận `pose_embedding` và một/hoặc hai `hand_embedding3d`; camera, translation, global orientation, shape, expression và direct SMPL-X pose không được thêm vào `final_params` ([code lines 476–503](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fit_single_frame.py#L476-L503)). Nói cách khác, public path là **latent-only refinement quanh initialization**, không phải full bundle adjustment như một số mô tả tổng quát có thể khiến người đọc hiểu.

Điều này giải thích vì sao proposal phải mở lại root/translation/body/hand state ở cấp sequence, nhưng vẫn giữ intrinsics và shape ổn định để tránh bài toán quá tự do.

### 4.3 Contract-fixed baseline bắt buộc

Trước khi thử method mới, cần tạo baseline `DexAvatar-CF` chỉ gồm:

- dùng numeric frame order duy nhất;
- giữ đủ manifest frame; missing observation được mask, không drop;
- deterministic left/right routing dựa trên handedness, không index cố định;
- đúng mapping wrist + 20 joint cho mỗi hand;
- thêm epsilon normalization;
- PKL/OBJ cùng decoded pose;
- resume khôi phục state hoặc chạy clean-only;
- log đúng count 2.872 frame và hash manifest/masks;
- không thay loss/weight/prior ngoài những gì cần để code đúng nghĩa đã mô tả.

Mọi điểm tăng từ `DexAvatar → DexAvatar-CF` phải báo là **correctness/reproducibility gain**, không phải đóng góp DexFactor-4D.

---

## 5. Datasets: dùng gì, không dùng gì

| Dataset | Quy mô/phạm vi liên quan | Nhãn | Vai trò hợp lệ | Không được làm |
|---|---|---|---|---|
| SGNify mocap | 57 DGS signs, 1 signer; 2.872 frame chấm | SMPL-X mocap | **Test duy nhất** theo official manifest | Train, tune threshold/weight, chọn ablation |
| SignAvatar / SignAvatars | 70K clips, 8,34M frames, 117 giờ, 153 signer; WLASL, PJM, DGS, GRSL, LSF, How2Sign, Phoenix | SMPL-X pseudo-GT từ multi-estimator + fitting | Dev phụ cho stability, missing/blur/occlusion stress; prior SignBPoser hiện hữu | Xem pseudo-GT như mocap hoặc claim metric tuyệt đối |
| How2Sign | continuous ASL multi-view corpus | video/linguistic, reconstruction-derived khi qua SignAvatars | Clip sign tự nhiên cho sanity check | Lẫn signer/video giữa train/dev nếu train module mới |
| SignHPoser private mocap | 8 signer, 93 fingerspelling words | Vicon + Manus, retarget SMPL-X | Dùng checkpoint prior sẵn có | Claim công khai tái train được khi raw data chưa release |
| InterHand2.6M | single/interacting hands, multi-view | MANO/3D hand | Dev geometry/routing; là training data của nhiều frozen model | Gọi là sign-language data |
| Re:InterHand | temporal interacting hand sequences | 3D hand | Dev temporal/interaction | Dùng để chứng minh sign generalization |
| DexYCB, HO3D | hand-object video | MANO/3D | Dữ liệu training đã công bố của PAD-Hand/WiLoR; dev physics/temporal | Fine-tune theo SGNify behavior |
| ARCTIC | hand-object interaction sequences | MANO/contact | Nguồn HMP/Dyn-HaMR; tham khảo contact/temporal | Áp contact attraction vào mọi sign |

### Quyết định training

**Bản DexFactor-4D chốt không train neural network mới.** Tất cả backbone/prior/diffusion dùng checkpoint frozen. Chỉ chọn hyperparameter của factor graph trên:

1. `InterHand2.6M validation` và `Re:InterHand validation` cho mapping, handedness, temporal và interaction;
2. một split signer-disjoint cố định của SignAvatars/How2Sign cho stability dưới sign motion, blur và occlusion;
3. synthetic corruption được tạo trước: drop detection, swap handedness, blur, occlusion, variable frame gap.

Không dùng SGNify frame, sign name hay GT để chọn weight. Nếu không thể tạo signer-disjoint vì metadata thiếu, chỉ dùng các clip đó làm qualitative smoke test, không dùng để tune.

### Tuyên bố nguồn học của từng component frozen

DexFactor-4D không cập nhật trọng số neural, nhưng provenance của pretrained model vẫn phải được ghi để thấy domain shift:

| Component | Trạng thái trong proposal | Dữ liệu học được paper/repo công bố | Ghi chú áp dụng |
|---|---|---|---|
| SignBPoser | frozen, latent body 33 | SignAvatars/How2Sign pseudo-3D body sau filtering | prior sign-specific hiện hữu |
| SignHPoser | frozen, latent hand 23 | mocap riêng 8 signer, 93 fingerspelling words, Vicon + Manus | không thể tái train đầy đủ nếu raw set chưa công khai |
| SMPLer-X | official frozen checkpoint | giữ nguyên checkpoint/recipe của public DexAvatar; không đổi corpus | chỉ initialization/camera/body cue |
| Sapiens | official frozen task checkpoints | encoder pretrain Humans-300M; task heads pose/segmentation/depth theo paper Sapiens | chạy inference, không distill/train FAE như A2P |
| HaMeR | official frozen checkpoint | khoảng 2,7M examples từ FreiHAND, HO3D, MTC, RHD, InterHand2.6M, H2O3D, DEX-YCB, COCO WholeBody, Halpe, MPII NZSL | observation, không phải temporal prior |
| WiLoR | official frozen checkpoint | detector WHIM hơn 2M auto-annotated images; pose khoảng 4,2M images từ 14 datasets, gồm các hand datasets trên, Re:InterHand và BEDLAM hands | observation thứ hai + disagreement signal |
| PAD-Hand | official frozen dataset-specific checkpoint | official train splits DexYCB hoặc HO3D với WiLoR estimates; sequence length 16 | chọn checkpoint theo dev protocol, không fine-tune SGNify |

Nếu license/checkpoint của một component không cho phép sử dụng, ablation tương ứng phải bị loại trước test; không thay bằng model khác sau khi đã nhìn kết quả SGNify.

---

## 6. Evidence synthesis: điều gì thực sự chuyển được

### 6.1 HaMeR và WiLoR

HaMeR là ViT-H hồi quy MANO, train từ khoảng 2,7 triệu example nhiều dataset; mạnh ở single-frame nhưng không temporal. WiLoR thêm detector và mesh-aligned refinement, train pose trên khoảng 4,2 triệu image từ 14 dataset. Paper WiLoR báo trên FreiHAND PA-MPJPE/PA-MPVPE 5,5/5,1 mm so với HaMeR 6,0/5,7; trên dynamic HO3D, WiLoR có MPFVE 4,43, MPFJE 0,762 và jitter 5,92, tốt hơn HaMeR 10,60/1,768/20,43 trong protocol paper.

**Chuyển được:** hai estimator có error mode khác nhau, nên disagreement là reliability signal có thật; WiLoR là observation bổ sung hợp lý.  
**Không chuyển được:** WiLoR vẫn bottom-up, xử lý hai tay độc lập và paper nêu giới hạn ở extreme pose/small crowded hands. Không thay HaMeR bằng WiLoR một cách mù.

### 6.2 ACR và 4DHands

ACR dùng attention collaboration/cross-hand prior cho arbitrary two-hand reconstruction; trên protocol InterHand2.6M của paper, MPJPE 8,09 mm overall, 9,08 interacting, 6,85 single. Chính paper nêu không có collision nên vẫn interpenetration. 4DHands cho thấy temporal context và đặc biệt cross-hand relative modeling làm giảm lỗi; trên InterHand 30 fps paper báo MPJPE 7,37 và acceleration error 2,81.

**Chuyển được:** left/right association và relative-hand factor là cấu trúc cần thiết.  
**Không chuyển được:** không có code 4DHands chính thức tìm thấy, nên không biến nó thành dependency; không copy con số sang SGNify.

### 6.3 Dyn-HaMR

Dyn-HaMR dùng tracking/infill, sequence/global optimization, geodesic smoothness, motion/interaction priors, biomechanics và penetration. Paper báo trên InterHand2.6M HaMeR 9,84/10,13/5,13 (MPJPE/MPVPE/AccErr) và Dyn-HaMR 7,94/8,15/2,76; trên H2O, MPJPE/AccErr 32,9/9,21 xuống 22,5/4,2.

Audit repo hiện tại cho thấy một số default không bật toàn bộ thành phần paper (`run_prior=False`, HMP/biomechanics/penetration tùy config hoặc weight 0). Vì vậy chỉ mượn cấu trúc geodesic + sequence + interaction; không tuyên bố “cắm Dyn-HaMR là xong”.

### 6.4 A2P và Sapiens multi-cue

A2P dùng Sapiens keypoint, segmentation và depth, cộng diffusion/collision. Ablation paper trên InterHand cho baseline MPJPE/MPVPE 7,77/7,93; +keypoint 6,48/6,72; +segmentation 6,19/6,34; +depth 5,74/5,98; +diffusion 5,36/5,58. Sapiens chính thức có checkpoint cho pose, body-part segmentation và monocular human depth.

**Chuyển được:** silhouette và depth-order là cue bổ sung khi keypoint bị occlusion.  
**Không chuyển được:** depth Sapiens không phải metric depth tuyệt đối đáng tin cho bàn tay nhỏ; chỉ dùng thứ tự trước/sau hoặc residual đã chuẩn hóa. A2P chưa có code chính thức được xác minh, nên implement factor đơn giản, không tuyên bố tái lập A2P.

### 6.5 KNOWN-Hand, DIP-Hand, PAD-Hand

KNOWN-Hand cho bằng chứng rằng biomechanical/functional constraints kết hợp heteroscedastic uncertainty có ích trong weak supervision. Code công khai là tham khảo công thức, chưa đủ sạch để copy trực tiếp như package. DIP-Hand dùng diffusion conditional cho chuỗi 16 frame, nhưng demo là single-hand và phụ thuộc assets ngoài.

PAD-Hand (CVPR 2026 Highlight) dùng physics-aware conditional diffusion, Euler–Lagrange virtual observations và uncertainty. Paper báo với WiLoR initialization:

| Dataset | WiLoR PA-MPJPE / MPJPE / ACCEL | PAD-Hand |
|---|---:|---:|
| DexYCB | 4.88 / 12.75 / 6.70 | 4.63 / 10.56 / 3.34 |
| HO3D | 7.50 / — / 4.98 | 7.43 / — / 2.71 |
| TACO unseen | 8.37 / 25.13 / 5.47 | 8.02 / 24.38 / 1.87 |

Nhưng audit release cho thấy demo:

- chọn một tay (ưu tiên right nếu có), không xử lý bimanual joint inference;
- dùng window 16 stride 16, không overlap;
- nếu thiếu một frame thì bỏ cả window và fallback;
- không expose đầy đủ LLLA uncertainty/physics path như paper.

**Quyết định:** chỉ dùng output pretrained như **soft temporal proposal**, chạy riêng từng hand sau canonicalization, window 16 overlap 8, có missing mask. Weight do reliability/switch variable của DexFactor-4D quyết định; không dùng uncertainty mà release không cung cấp.

### 6.6 Tại sao không train một sign diffusion mới

Một sign-specific temporal diffusion nghe hấp dẫn nhưng không qua vòng phản biện:

- SignAvatars là pseudo-GT được tạo bởi các estimator/fitting; train lại dễ học chính bias cần sửa;
- SignHPoser mocap riêng không công khai đủ để mở rộng/retrain;
- SGNify quá nhỏ và là test;
- continuous sign data và fingerspelling khác phân phối isolated DGS;
- chi phí ablation lớn, khó quy attribution.

Do đó proposal giữ learned priors frozen và đặt đổi mới ở inference-time factor graph có thể kiểm tra từng residual.

---

## 7. DexFactor-4D: method end-to-end

### 7.1 Input và output

**Input:** chuỗi RGB monocular \(I_{1:T}\), frame timestamps, camera intrinsics khởi tạo từ SMPLer-X.  
**Output:** một SMPL-X mesh cho **mọi frame trong manifest**, gồm \(\beta\), global orientation, translation, body pose, left/right hand pose; PKL và OBJ cùng tham số decoded.

Không cần text, gloss, HamNoSys hay class one/two-hand GT.

### 7.2 Frozen perception front-end

Chạy một lần và cache:

- SMPLer-X: camera, root, translation, shape, body/hand initialization;
- Sapiens pose: 2D whole-body/hand keypoints và heatmap confidence;
- Sapiens segmentation: human/arm/hand-related silhouette;
- Sapiens depth: depth-order cue trên vùng human;
- HaMeR: hand boxes, handedness, 2D/3D keypoints, MANO pose;
- WiLoR: hand boxes, handedness, 2D/3D/MANO estimate;
- PAD-Hand: proposal temporal riêng cho mỗi track hand, chỉ nếu checkpoint chạy được.

Tất cả checkpoint phải ghi hash/version. Không fine-tune.

### 7.3 Deterministic temporal hand association

Tạo tối đa hai track semantic \(h\in\{L,R\}\). Mỗi detection \(d\) ở frame \(t\) có cost:

\[
C_{t,d,h} =
\lambda_{lr} C_{handedness}
+ \lambda_{box}(1-\mathrm{IoU}(b_{t,d},\hat b_{t,h}))
+ \lambda_{w}\frac{\|w_{t,d}-\hat w_{t,h}\|_2}{\sqrt{A_{body}}}
+ \lambda_{pose}D_{2D}(d,\hat d_{t,h}).
\]

Giải assignment toàn clip bằng Viterbi/min-cost flow với trạng thái `observed`, `missing`, `ambiguous`. Không dùng vị trí detection index làm identity. `missing` giữ frame trong chuỗi và chỉ tắt observation factor tương ứng.

Thay classifier one/two-hand cứng bằng active probability \(a_t^h\in[0,1]\), suy ra từ detection confidence, motion và track persistence. Tay không hoạt động vẫn có state, nhưng observation/temporal/contact weight được giảm; không ép pose về zero.

### 7.4 State tối ưu

Cho clip/window:

\[
\mathcal X = \left\{
\beta, K,
R_t^g,\tau_t,
z_t^B\in\mathbb R^{33},
z_t^L,z_t^R\in\mathbb R^{23}
\right\}_{t=1}^{T}.
\]

- \(z^B\) được SignBPoser decode thành body axis-angle;
- \(z^L,z^R\) được SignHPoser decode thành hand axis-angle;
- \(\beta\) khởi tạo bằng robust Huber/coordinate median của SMPLer-X qua các frame tin cậy, rồi giữ chung cho clip;
- \(K\) giữ cố định trong official profile;
- \(R_t^g,\tau_t\) được phép tối ưu nhẹ với anchor. TR-V2V bỏ translation khi chấm, nhưng translation vẫn cần cho reprojection và depth ordering.

### 7.5 Reliability thay cho “uncertainty” bịa đặt

Không giả định HaMeR/WiLoR xuất calibrated per-joint variance. Với mỗi tay/frame, định nghĩa reliability quan sát được:

\[
q_t^h = m_{coord}\,\operatorname{clip}_{[0,1]}\!\left[
(c_{det}+\epsilon)^{\alpha_c}
(\mathrm{IoU}_{sil}+\epsilon)^{\alpha_s}
\exp\!\left(-\alpha_d d_{HW}/\sigma_d-\alpha_v e_{track}/\sigma_v\right)
\right],
\]

Ở đây các dòng liền nhau trong ngoặc vuông được **nhân** với nhau; \(\alpha_c+\alpha_s+\alpha_d+\alpha_v=1\), các \(\alpha\ge0\), và \(m_{coord}\in\{0,1\}\) là cờ unit-test coordinate/reprojection. Đây là geometric reliability: một cue yếu làm giảm score thay vì được một cue lớn cộng bù.

trong đó:

- \(c_{det}\): confidence công khai của detector/heatmap;
- \(d_{HW}\): median normalized 2D disagreement HaMeR–WiLoR sau mapping;
- \(e_{track}\): innovation so với prediction của track;
- \(\mathrm{IoU}_{sil}\): consistency giữa projected/rendered hand region và segmentation.

Hàm gộp và \(\sigma\) được chọn trên dev, sau đó khóa. Báo cáo gọi đây là **reliability score**, không gọi calibrated uncertainty.

### 7.6 Observation factors

#### (a) Body/hand 2D reprojection

\[
L_{2D} = \sum_{t,j} m_{t,j}\,w_{t,j}\,
\rho\!\left(\pi_K(J_j(\mathcal X_t))-u_{t,j}\right),
\]

với Geman–McClure hoặc Huber \(\rho\), confidence \(w\), missing mask \(m\). Body ưu tiên Sapiens; hand hợp nhất HaMeR/WiLoR bằng reliability, không average vô điều kiện.

#### (b) Full-3D normalized hand geometry

Vì các estimator có coordinate/scale khác, không so absolute translation. Với wrist \(w\), scale robust \(s\) là median chiều dài metacarpal, chuẩn hóa:

\[
\bar J_{t,k}^h = \frac{J_{t,k}^h-J_{t,w}^h}{s_t^h+\epsilon}.
\]

Trên 20 edge MANO/SMPL-X, dùng cả joint geometry và unit bone directions:

\[
L_{3D}^{hand} = \sum_{t,h} a_t^h q_t^h
\left[
\sum_k \rho(\bar J_{t,k}^h-\bar Y_{t,k}^h)
+ \eta\sum_{(i,j)\in E}\rho(\hat b_{ij}(J)-\hat b_{ij}(Y))
\right].
\]

Đây là thay thế trực tiếp cho nhánh z-only hiện tại. Mọi transform MANO↔camera↔SMPL-X phải qua unit test: projected 3D detector joints phải trùng 2D detector trong tolerance đã đăng ký; nếu fail thì tắt factor 3D cho observation đó.

#### (c) Silhouette factor

Render differentiable silhouette \(S(\mathcal X_t)\), so với mask Sapiens trong ROI hand/forearm:

\[
L_{sil}=1-\mathrm{IoU}_{soft}(S,M)+\mathrm{BCE}(S,M).
\]

Chỉ tính trên pixel được segmentation confidence cao; không để torso mask kéo sai hand mesh.

#### (d) Relative depth-order factor

Ở pixel/vertex pairs nơi hai tay hoặc hand–torso overlap, Sapiens depth chỉ cho dấu/order:

\[
L_{ord} = \sum_{(p,q)\in\Omega_t}
\operatorname{softplus}\left[-s_{pq}\big(z_q(\mathcal X_t)-z_p(\mathcal X_t)\big)/\tau_z\right],
\]

trong đó \(s_{pq}\in\{-1,+1\}\) là thứ tự front/back từ depth map. Không dùng absolute depth và không scale depth Sapiens thành millimet.

### 7.7 Prior và anchors

\[
L_{prior} = \sum_t
\lambda_B\|z_t^B\|_2^2
+\lambda_H\left(\|z_t^L\|_2^2+\|z_t^R\|_2^2\right)
+L_{shape}.
\]

Giữ decoded-pose anchor tới initialization ở stage đầu, nhưng **anneal** qua stage thay vì 1200 ở cả ba stage. Ví dụ chỉ quy định tỷ lệ `1.0 → 0.25 → 0.05`; absolute scale được xác định sau robust residual normalization trên dev, không bịa một weight tối ưu từ test.

### 7.8 Temporal factors đúng SO(3)

Với rotation matrix của joint \(j\), vận tốc geodesic:

\[
\omega_{t,j} = \frac{\log(R_{t-1,j}^{\top}R_{t,j})^\vee}{\Delta t_t},
\]

và acceleration:

\[
\alpha_{t,j} = \frac{\omega_{t,j}-\omega_{t-1,j}}
{(\Delta t_t+\Delta t_{t-1})/2}.
\]

\[
L_{temp}=\sum_{t,j}
\gamma_{t,j}^{v}\rho(\omega_{t,j})
+\gamma_{t,j}^{a}\rho(\alpha_{t,j}).
\]

Gate được định nghĩa là \(\gamma_{t,j}=\gamma_{min}+(\gamma_{max}-\gamma_{min})(1-q_{t,j})\): tăng khi observation reliability thấp và giảm khi keypoint/3D/silhouette đồng thuận. Như vậy fast signing có bằng chứng không bị oversmooth; gap do missing frame được xử lý bằng \(\Delta t\) thật. \(\gamma_{min},\gamma_{max}\) được khóa từ dev.

### 7.9 Biomechanics, collision và contact

#### Biomechanical barrier

Áp dụng joint-angle limits/functional couplings được định nghĩa công khai trong Dyn-HaMR/KNOWN-Hand và kiểm tra lại trên SMPL-X convention:

\[
L_{bmc}=\sum_{t,h,j}
(1-q_t^h)\left[
\mathrm{ReLU}(\theta_{t,j}-\theta^{max}_j)^2
+\mathrm{ReLU}(\theta^{min}_j-\theta_{t,j})^2
\right] + L_{coupling}.
\]

Term bằng zero khi pose nằm trong feasible interval; không kéo pose hợp lệ về một mean pose.

#### Collision

Dùng signed-distance/interpenetration loss cho hand–hand và hand–body. Weight theo collision severity nhưng capped để metric không hy sinh alignment chỉ vì GT có interpenetration.

#### Contact persistence có bằng chứng

Tạo contact candidate khi đồng thời thỏa:

- khoảng cách projected fingertip/palm nhỏ;
- silhouette/depth cho biết overlap hoặc cùng surface neighborhood;
- tồn tại ít nhất \(k\) frame liên tiếp.

Khi candidate tồn tại, penalize thay đổi distance/relative transform, **không** attraction tuyệt đối nếu chưa có contact. Điều này tránh kéo hai tay chạm nhau ở sign không contact.

### 7.10 PAD-Hand proposal factor có switch

Chạy PAD-Hand riêng cho left/right track sau mirror-canonicalization, sequence 16, overlap 8. Với decoded proposal \(P_{t}^{h}\):

\[
L_{PAD}=\sum_{t,h} a_t^h(1-q_t^h)s_t^h\,
\rho\big(D_{SO(3)}(R_t^h,P_t^h)\big)
+\lambda_s\Phi(s_t^h),
\]

trong đó \(s_t^h\in[0,1]\) là switchable-constraint variable. Nếu PAD proposal bất đồng mạnh với keypoint/silhouette/full-3D residual, optimization đẩy \(s\) về 0. Vì release PAD-Hand không expose uncertainty paper, DexFactor-4D không sử dụng một variance giả.

### 7.11 Objective tổng

Sau khi chuẩn hóa mỗi residual theo robust scale trên dev:

\[
\begin{aligned}
L(\mathcal X)=
&L_{2D}^{body}+L_{2D}^{hand}
+L_{3D}^{hand}+L_{sil}+L_{ord}\\
&+L_{prior}+L_{init}+L_{root}
+L_{temp}\\
&+L_{bmc}+L_{coll}+L_{contact}+L_{PAD}.
\end{aligned}
\]

Không có một bộ weight “đúng” được suy ra từ literature vì scale/protocol khác nhau. Proposal chỉ cho phép một grid nhỏ đã định trước trên dev; chọn bằng objective tổng hợp gồm 2D holdout, 3D dev, acceleration và failure rate; khóa trước test.

### 7.12 Lịch tối ưu

Chuỗi dài được chia window 64 frame, overlap 16; overlap dùng consensus và warm-start. Nếu GPU không đủ, profile 32/8 phải được đăng ký trước và không trộn trong cùng benchmark.

1. **Stage 0 — contract/data:** validate manifest, coordinate transforms, L/R track; không optimize.
2. **Stage 1 — root/body:** optimize \(R^g,\tau,z^B\); hand latent cố định; strong init anchor.
3. **Stage 2 — observations:** mở \(z^L,z^R\); bật 2D, full-3D, silhouette, order; giảm anchor.
4. **Stage 3 — sequence/interaction:** bật SO(3) velocity/acceleration, collision, gated BMC/contact/PAD; anchor thấp.
5. **Stage 4 — overlap consensus:** tối ưu lại overlap state và kiểm tra tất cả frame có output.

Dùng LBFGS line-search như baseline cho khả năng so sánh; nếu memory không phù hợp, Adam warm-start + LBFGS polish chỉ được đổi sau dev và phải dùng cho mọi test clip.

### 7.13 Pseudocode

```text
inputs: frames I[1:T], timestamps, frozen checkpoints, fixed config

obs = run_smplerx_sapiens_hamer_wilor(I)
tracks = global_LR_assignment(obs, allow_missing=True)
pad_proposals = run_PAD_per_track(I, tracks, window=16, overlap=8)

assert manifest_count_and_order()
assert coordinate_reprojection_unit_tests(obs)

X = initialize_SMPLX_latents_and_root(obs, robust_clip_shape=True)
for window in sliding_windows(T, size=64, overlap=16):
    optimize root/body with body observations + priors
    optimize body/hands with 2D + full-3D + silhouette + depth-order
    optimize sequence with SO3 temporal + switched PAD/BMC/collision/contact
    write overlap state to consensus buffer

X = overlap_consensus_polish(X)
export identical decoded SMPL-X parameters to PKL and OBJ for every frame
evaluate only with frozen official manifest/masks
```

---

## 8. Tích hợp vào codebase DexAvatar

Không nên vá trực tiếp mọi thứ vào `fitting.py`. Tách module để test độc lập:

| Module đề xuất | Trách nhiệm | Thay/đọc file hiện tại |
|---|---|---|
| `benchmark_contract.py` | load/hash manifest, masks, frame coverage, exact TR alignment | mới; chặn dropped frame trước evaluator |
| `hand_tracks.py` | L/R global association, missing/ambiguous state | thay branch index-based trong `data_parser.py` |
| `joint_mapping.py` | named SMPL-X↔MANO/Sapiens mapping, unit tests | thay magic indices trong `fitting.py` |
| `observations.py` | cache HaMeR/WiLoR/Sapiens/PAD với version/hash | đọc front-end output |
| `factors/hand3d.py` | full xyz normalized geometry/bone direction | thay z-only branch |
| `factors/silhouette.py` | differentiable mask loss | mới |
| `factors/depth_order.py` | overlap-relative order only | mới |
| `factors/temporal_so3.py` | \(\Delta t\)-aware velocity/acceleration | thay axis-angle previous-frame loss |
| `factors/interaction.py` | collision, gated BMC/contact | hoàn thiện paper-code gap |
| `sequence_fitter.py` | window stages, overlap consensus, logging | thay per-frame latent optimizer path |
| `export.py` | một decoded state cho PKL/OBJ | sửa mismatch export |

### Unit/integration tests bắt buộc

- detection order permutation không đổi L/R output;
- filename `1,2,10` giữ numeric order;
- missing frame không đổi manifest count;
- one-hand mapping chỉ chạm đúng 21 joint của hand đó;
- mirror-left → canonical → inverse trả lại pose trong tolerance;
- 3D observation project về đúng 2D sau coordinate transform;
- zero-variance observation không NaN;
- resumed run bằng clean run trong tolerance;
- PKL render và OBJ vertices trùng nhau;
- loss term có gradient tới đúng state và bằng zero khi mask off;
- evaluator từ chối mask/manifest hash lạ.

---

## 9. Evaluation protocol đã đăng ký trước

### 9.1 Primary endpoint

Ba co-primary metrics trên đúng 57 sign/2.872 frame:

- mean TR-V2V UBody(-F);
- mean TR-V2V LHand;
- mean TR-V2V RHand.

Translation-only per frame, official exact masks. Không rotation/scale alignment.

### 9.2 Secondary endpoints

- median, p90/p95 TR-V2V và per-sign distribution;
- missing-output/failure rate;
- acceleration và jerk của joint/vertex;
- interpenetration volume/count;
- joint-limit violation rate;
- contact precision/persistence trên subset được annotation thủ công **sau khi** primary run bị khóa;
- runtime, peak VRAM, số optimizer failure/NaN.

Secondary metrics không cứu một primary result xấu.

### 9.3 Baseline và ablation ladder

| ID | Cấu hình | Câu hỏi |
|---|---|---|
| A | DexAvatar paper numbers | Chỉ là reference, không paired run |
| B | Public DexAvatar commit | Repo hiện tại làm được gì? |
| C | `DexAvatar-CF` | Bao nhiêu gain là correctness fix? |
| D | C + WiLoR observation/track | Ensemble/routing có ích? |
| E | D + full-3D hand | z-only là bottleneck? |
| F | E + silhouette/depth order | Multi-cue giúp overlap/blur? |
| G | F + SO(3) sequence | Temporal factor giúp mà không oversmooth? |
| H | G + reliability gating | Có tránh negative transfer không? |
| I | H + gated BMC/collision/contact | Plausibility có hạ primary metric? |
| J | I + switched PAD proposal | Temporal diffusion proposal thêm gain hay gây domain shift? |

Full method là J. Nếu J fail tiêu chí non-inferiority so với I, paper phải báo J fail và không được đổi tên I thành “full” sau khi xem test; một version kế tiếp có thể preregister I làm final.

### 9.4 Thống kê

- paired difference theo frame, nhưng bootstrap cluster theo **57 sign** để giữ tương quan thời gian;
- 10.000 bootstrap resamples, fixed seed công bố;
- báo 95% percentile hoặc BCa CI, cùng per-sign scatter;
- không chỉ báo relative percent;
- báo cả micro-average theo frame và macro-average theo sign;
- cấu hình thắng khi cả ba point estimate không xấu hơn C, và CI 95% của composite normalized paired delta nằm dưới 0;
- guardrail: upper CI của từng region không vượt `+0,5 mm`; threshold này phải khóa trước test;
- nếu làm significance test nhiều region/ablation, dùng Holm correction.

Composite chỉ dùng cho quyết định preregistered:

\[
\Delta_{comp} = \frac{1}{3}\sum_r
\frac{\mathrm{TRV2V}_{r}^{method}-\mathrm{TRV2V}_{r}^{CF}}
{\mathrm{TRV2V}_{r}^{CF}}.
\]

Không dùng composite để che một bàn tay suy giảm mạnh.

### 9.5 Freeze discipline

Trước khi giải mã SGNify GT:

- commit code/config và container digest;
- hash checkpoint, official manifest và region masks;
- khóa preprocessing, crop, frame rate, hand mapping, window/overlap;
- khóa grid và tiêu chí chọn;
- tạo report dev tự động;
- đánh dấu test run duy nhất. Nếu phải sửa bug sau test, version bump và báo toàn bộ history.

---

## 10. Vòng phản biện và thay đổi proposal

### Review 1 — “Method quá rộng, attribution không thể làm”

**Phản biện:** ý tưởng đầu gồm train sign temporal prior mới, camera SLAM, language prior và multi-cue. Dữ liệu/attribution không đủ.  
**Sửa:** bỏ new training, bỏ SLAM và text/linguistic inference. Giữ factor graph quan sát được và prior sẵn có.

### Review 2 — “Cải thiện có thể chỉ là bug fix”

**Phản biện:** mapping/routing/frame/export findings có thể tự hạ metric. Nếu so full method với public code thì claim algorithm không hợp lệ.  
**Sửa:** thêm `DexAvatar-CF` làm baseline bắt buộc và báo B→C riêng.

### Review 3 — “WiLoR/PAD không phải sign model; có thể phá extreme handshape”

**Phản biện:** các model này train chủ yếu hand/hand-object; sign handshape và fast motion khác phân phối.  
**Sửa:** không thay HaMeR; dùng ensemble disagreement, reliability gate và switchable PAD factor. High-confidence sign evidence thắng learned proposal.

### Review 4 — “Biomechanics tốt về hình nhưng có thể xấu TR-V2V”

**Phản biện:** DexAvatar ablation và supplement đã cho thấy loss sinh cơ học có gain rất nhỏ hoặc làm xấu right hand; SGNify GT đôi khi phi giải phẫu.  
**Sửa:** dùng barrier chỉ ngoài feasible set, weight theo low confidence; collision capped; BMC/PAD/contact là ablation riêng và phải qua non-inferiority.

### Review 5 — “Sapiens depth không phải metric hand depth”

**Phản biện:** monocular depth scale/offset không đáng tin, đặc biệt crop bàn tay nhỏ.  
**Sửa:** chỉ dùng pairwise front/back order tại overlap và tắt khi depth confidence/consistency thấp.

### Review 6 — “Temporal smoothing có thể xóa phonologically important fast transitions”

**Phản biện:** acceleration nhỏ không đồng nghĩa sign đúng.  
**Sửa:** SO(3) robust factor theo \(\Delta t\), giảm weight khi multi-cue đồng thuận với chuyển động nhanh; không dùng low-pass postprocess.

### Editor verdict

**Accept for implementation / reject any superiority claim before experiment.** Proposal hiện đủ chặt để code và falsify. Lý do chấp nhận: dependency có thật, train/apply rõ, test leakage được chặn, alternative explanations được tách bằng ablation. Lý do chưa thể claim SOTA: chưa có official evaluator/masks/manifest trong repo công khai, chưa chạy checkpoint/data end-to-end, và target test chỉ một signer.

---

## 11. Dự báo định tính và tiêu chí dừng

### Thành phần có xác suất đóng góp cao nhất

1. contract fixes cho frame/mapping/export;
2. global L/R association + no-drop observation masks;
3. full xyz hand geometry và WiLoR–HaMeR disagreement gating;
4. SO(3) temporal factor cho tay;
5. silhouette/depth-order ở overlap.

### Thành phần rủi ro nhất

1. PAD-Hand proposal do domain shift và release single-hand;
2. BMC/contact do GT limitation;
3. mở quá nhiều root/camera degrees of freedom.

### Kill criteria trên dev

- module làm tăng failure rate hoặc NaN;
- permutation/mirror/coordinate tests fail;
- degradation >0,5 mm trên bất kỳ region dev chính;
- acceleration tốt hơn nhưng positional error xấu hơn guardrail;
- PAD/BMC residual thường bị switch về 0 ở >80% low-confidence window: module không tạo thông tin hữu ích;
- runtime/VRAM vượt budget đã đăng ký mà gain dưới practical threshold.

---

## 12. Checklist triển khai tái lập

- [ ] Có quyền dùng SMPL-X/MANO và checkpoint theo license.
- [ ] Nhận official SGNify manifest, exact region masks và evaluator.
- [ ] Lưu hash DexAvatar commit, configs và checkpoints.
- [ ] Tạo `DexAvatar-CF`; test B→C trước method.
- [ ] Unit-test named joint mapping và coordinate projection.
- [ ] Cache frozen observations, không rerun khác version giữa ablations.
- [ ] Xây hand tracks toàn clip với missing state.
- [ ] Implement full-3D factor trước; kiểm chứng gradient bằng synthetic case.
- [ ] Implement sequence SO(3) và timestamps.
- [ ] Thêm Sapiens silhouette/depth-order sau khi core pass.
- [ ] Thêm gated BMC/contact, rồi switched PAD cuối cùng.
- [ ] Freeze dev config, commit/container digest.
- [ ] Chạy official test đúng một lần, kiểm count 2.872.
- [ ] Báo mọi failure, per-sign CI và full ablation.

---

## 13. Những claim được phép và không được phép

### Được phép sau khi implement nhưng trước test

- “DexFactor-4D là một proposal inference-time, training-free ở cấp hệ thống.”
- “Method sửa hand routing/3D/temporal failure modes đã xác định trong public DexAvatar code.”
- “Method dùng frozen components đã được công bố và switch/gate để hạn chế negative transfer.”

### Chỉ được phép nếu official test xác nhận

- “tốt hơn DexAvatar trên TR-V2V”;
- “SOTA trên SGNify”;
- “PAD/BMC làm tăng accuracy sign reconstruction”;
- bất kỳ con số mm hoặc phần trăm gain mới nào.

### Không được phép từ benchmark hiện tại

- “generalize mọi sign language/signer”;
- “understands sign semantics”;
- “physically correct ground truth”;
- “real-time” nếu chưa đo;
- “uncertainty calibrated” nếu chỉ dùng disagreement reliability.

---

## 14. Tài liệu tham khảo chính

1. Kundu et al., [DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors](https://arxiv.org/html/2512.21054v1), WACV 2026; [code](https://github.com/kaustesseract/DexAvatar).
2. Forte et al., [Reconstructing Signing Avatars From Video Using Linguistic Priors (SGNify)](https://openaccess.thecvf.com/content/CVPR2023/papers/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.pdf), CVPR 2023; [supplement](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Forte_Reconstructing_Signing_Avatars_CVPR_2023_supplemental.pdf); [code](https://github.com/MPForte/SGNify).
3. Yu et al., [SignAvatars: A Large-scale 3D Sign Language Holistic Motion Dataset and Benchmark](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00653.pdf), ECCV 2024; [code/data tools](https://github.com/ZhengdiYu/SignAvatars).
4. Pavlakos et al., [Reconstructing Hands in 3D with Transformers (HaMeR)](https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html), CVPR 2024; [code](https://github.com/geopavlakos/hamer).
5. Potamias et al., [WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild](https://openaccess.thecvf.com/content/CVPR2025/html/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.html), CVPR 2025; [code](https://github.com/rolpotamias/WiLoR).
6. Yu et al., [ACR: Attention Collaboration-based Regressor for Arbitrary Two-Hand Reconstruction](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_ACR_Attention_Collaboration-Based_Regressor_for_Arbitrary_Two-Hand_Reconstruction_CVPR_2023_paper.pdf), CVPR 2023; [code](https://github.com/ZhengdiYu/Arbitrary-Hands-3D-Reconstruction).
7. Lin et al., [4DHands: Reconstructing Interactive Hands in 4D](https://arxiv.org/html/2405.20330v1), 2024; [project](https://4dhands.github.io/).
8. Yu et al., [Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html), CVPR 2025 Highlight; [code](https://github.com/ZhengdiYu/Dyn-HaMR).
9. Zhang et al., [KNOWN-Hand](https://arxiv.org/html/2407.12307v1), ECCV 2024; [code](https://github.com/zhangy76/KNOWN-Hand).
10. Zhang et al., [DIP-Hand](https://arxiv.org/html/2508.01835v1), ICCV 2025; [code](https://github.com/zhangy76/DIP-Hand).
11. Ismayilzada et al., [PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery](https://arxiv.org/html/2603.26068v1), CVPR 2026 Highlight; [code](https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026).
12. Han et al., [From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction (A2P)](https://arxiv.org/html/2503.17788), CVPR 2026; [project](https://gaogehan.github.io/A2P/).
13. Khirodkar et al., [Sapiens: Foundation for Human Vision Models](https://arxiv.org/html/2408.12569v1), ECCV 2024; [code](https://github.com/facebookresearch/sapiens).
14. Liu et al., [Motions as Queries](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Motions_as_Queries_One-Stage_Multi-Person_Holistic_Human_Motion_Capture_CVPR_2025_paper.pdf), CVPR 2025.
15. Baltatzis et al., [Neural Sign Actors](https://arxiv.org/abs/2312.02702), 2023/2024; [project](https://baltatzisv.github.io/neural-sign-actors/).
16. Duarte et al., [How2Sign](https://how2sign.github.io/), CVPR 2021.

---

## 15. Reproducibility và disclosure

- PDF DexAvatar đính kèm đã được structural preflight và đối chiếu phần main/supplement.
- Các PDF SGNify, WiLoR và các paper liên quan được đọc từ nguồn chính thức; các con số trong tài liệu này là số paper, không phải kết quả tôi chạy lại.
- Code audit là static inspection ở các commit công khai; syntax check các file trọng yếu pass, nhưng không phải functional reproduction.
- Không có dữ liệu/checkpoint restricted nào được suy đoán hoặc tạo giả.
- Nội dung được tổng hợp với hỗ trợ AI; các quyết định experiment và claim cuối cùng cần tác giả dự án/human researcher kiểm tra, chạy và ký xác nhận.

**Bottom line:** DexFactor-4D là proposal có thể triển khai và bác bỏ được; thành phần cốt lõi đáng đặt cược là correctness-first sequence fitting với named hand mapping, no-drop tracking, full-3D geometry và reliability-gated SO(3) temporal optimization. Bất kỳ claim “tốt hơn DexAvatar” nào chỉ hợp lệ sau official preregistered run.
