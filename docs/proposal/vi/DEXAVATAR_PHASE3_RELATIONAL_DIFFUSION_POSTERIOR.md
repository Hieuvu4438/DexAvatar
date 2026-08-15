# Kế hoạch Xây dựng Phase 3: Quan hệ Khuếch tán Hậu phương (Relational Diffusion Posterior)

- **Dự án:** Chương trình giảm thiểu rủi ro DexAvatar / SignPosterior4D
- **Giai đoạn (Phase):** Chỉ Phase 3
- **Tên phương pháp:** `RDP` (Relational Diffusion Posterior - Phân phối Hậu phương Khuếch tán Quan hệ)
- **Ngày đề xuất:** 3 tháng 8 năm 2026
- **Mục tiêu chính:** suy luận nhiều giả thuyết SMPL-X toàn chuỗi được phối hợp từ các quan sát cố định không chắc chắn, mô hình hóa rõ ràng mối quan hệ giữa tay–tay và tay–thân thể, đồng thời chọn giả thuyết bằng cách sử dụng bằng chứng video thay vì ground truth của benchmark.
- **Hợp đồng đánh giá chính:** tập dữ liệu quần thể SGNify 57 ký hiệu / 1.493 khung hình do tác giả phát hành được chọn bởi chủ sở hữu dự án
- **Nguồn thiết kế:** [đề xuất SignPosterior4D tập trung vào phương pháp](DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md) và [kế hoạch và báo cáo thực thi UAWSR Phase 2](DEXAVATAR_PHASE2_UNCERTAINTY_AWARE_WHOLE_SEQUENCE_REFINER.md)

---

## 1. Quyết định điều hành

Tiến hành Phase 3 dưới dạng một **nhánh nghiên cứu mới, biệt lập**, đồng thời đóng băng Phase 2 để có thể tái sử dụng sau này.

Phase 2 đã tạo ra cơ sở hạ tầng hữu ích và các kết quả phục hồi tổng hợp được kiểm soát mạnh mẽ, nhưng nó không vượt qua hợp đồng chuyển tiếp chính thức đầy đủ:

| Cổng Phase 2 | Quyết định hiện tại | Hệ quả đối với Phase 3 |
|---|:---:|---|
| G0 bộ đánh giá/độ bao phủ | GO | tái sử dụng hợp đồng đánh giá bất biến của tác giả |
| G1 bộ khởi tạo mạnh nhất | GO | sử dụng tập hợp/fallback A1 đã khóa làm bộ khởi tạo benchmark cuối cùng |
| G2 độ sẵn sàng của dữ liệu | GO do chủ sở hữu ủy quyền | chỉ tái sử dụng cache với nhãn mục tiêu/nguồn gốc trung thực |
| G3 phục hồi tổng hợp | GO | tái sử dụng các kiểm thử hình học, SO(3), biến dạng (corruption) và đỉnh được giải mã |
| G4 giá trị thặng dư thực tế | GO ủy nhiệm / NO-GO chính thức | không coi T2 là khung xương hậu phương (posterior backbone) đã được xác thực |
| G5 độ không chắc chắn | GO ủy nhiệm / NO-GO chính thức | không bật U1 trong cấu hình Phase 3 chính |
| G6 chuyển giao đã khóa | NO-GO | không khởi tạo mô hình chính từ kết quả T2/T5 Lane-L thất bại |
| G7 phạm vi đánh giá của tác giả | GO | 1.493 khung hình là quần thể chuẩn chính thức |

Sự chuyển hướng này được cố ý thực hiện khác với việc giả định rằng Phase 2 đã thành công. Tuyến Phase 3 chính phải có khả năng huấn luyện và chạy **mà không cần** checkpoint chất lượng của Phase 2:

```text
các quan sát cố định + độ tin cậy U0 cố định
  -> chuỗi thân thể/cổ tay/hai tay chuẩn (canonical)
  -> điểm số không gian toàn thân DPoser-X cố định
  -> các bộ thích ứng điểm số thời gian-quan hệ có thể huấn luyện
  -> đồ thị quan hệ/tiếp xúc xác suất
  -> phân phối khuếch tán có điều kiện bị che (masked conditional diffusion posterior)
  -> K ứng viên chuỗi hoàn chỉnh
  -> bộ chọn chỉ dựa trên bằng chứng (evidence-only selector)
  -> tinh chỉnh quan sát an toàn tùy chọn
  -> các PKL chuẩn + lưới (meshes) + chẩn đoán
```

Phase 2 vẫn có thể tái sử dụng theo ba cách:

1. tái sử dụng ngay lập tức cấu trúc cache, các tiện ích xoay/hệ tọa độ, bộ giải mã SMPL-X, bộ đánh giá nghiêm ngặt, bộ dựng hình (renderer), tiện ích nguồn gốc và kiểm thử biến dạng;
2. coi checkpoint ARCTIC T1 tổng quát là một **thử nghiệm phân tách (ablation) khởi tạo tùy chọn**, vì nó đã vượt qua G3 tổng quát nhưng chưa qua G6 miền ký hiệu; và
3. thêm một tuyến khởi tạo `Phase2-GO -> Phase3` sau này nếu việc huấn luyện Phase 2 exact-A1 cuối cùng vượt qua G4–G6.

Câu hỏi khoa học trung tâm cho giai đoạn này là:

> Phân phối hậu phương toàn chuỗi đa thức, quan hệ có thể phục hồi chuyển động tay mơ hồ và tương tác mà cả bộ khởi tạo cố định lẫn bộ hiệu chỉnh thời gian xác định đều không thể phục hồi hay không?

Phase 3 chỉ thành công nếu sự khuếch tán và lý luận quan hệ cải thiện độ chính xác không gian trên các tập con khó được khai báo trước. Sự đa dạng của ứng viên, độ mượt mà thị giác hoặc độ giật (jerk) thấp hơn không phải là lý do chấp nhận GO.

---

## 2. Phạm vi chính xác của Phase 3

### 2.1 Trong phạm vi

- phân phối chuỗi thống nhất trên thân trên, cổ tay và cả hai tay;
- khuếch tán có điều kiện bị che trên các ký hiệu hoàn chỉnh hoặc cửa sổ 64 khung hình;
- đồ thị quan hệ tay–tay và tay–thân thể gọn nhẹ;
- tiếp xúc xác suất, thứ tự chiều sâu và sự duy trì tiếp xúc;
- huấn luyện score đầy đủ quan sát, che một phần và che theo bùng nổ (burst-masked);
- khởi tạo hoặc chưng cất (distillation) prior không gian toàn thân DPoser-X cố định;
- mặt nạ giám sát đặc thù theo nguồn cho các tập dữ liệu toàn thân và chỉ có tay;
- suy luận `K = 1` kiểu xác định và `K = 4` đa giả thuyết;
- xếp hạng mẫu chỉ dựa trên bằng chứng với dự đoán quan sát được giữ lại (held-out);
- tinh chỉnh khả vi ngắn tùy chọn sau khi chọn ứng viên;
- độ bao phủ chính xác, fallback an toàn, nguồn gốc và đánh giá với 3 seed; và
- mã và artifact mang tính bổ sung (additive) không làm thay đổi các phương pháp trước đó.

### 2.2 Hoãn lại một cách rõ ràng sang Phase 4

- SHuBERT hoặc một bộ mã hóa ngữ nghĩa video ngôn ngữ ký hiệu khác;
- điều kiện hóa HamNoSys;
- các token ngữ nghĩa hình dạng tay/hướng/vị trí/chuyển động;
- các lớp duy trì/chuyển tiếp/lặp lại/pha tiếp xúc được học;
- tính nhất quán chu trình ngữ nghĩa–động học;
- tính ưu thế hoặc đối xứng xác suất dưới dạng biến ngôn ngữ học; và
- bất kỳ tuyên bố nào cho rằng mô hình đã được điều kiện hóa theo âm vận học (phonology).

Các mối quan hệ trong Phase 3 mang tính **hình học và dựa trên tương tác**, không phải âm vận học. Ranh giới này rất quan trọng cho thang phân tách bắt buộc trong đề xuất chính:

```text
mô hình xác định nhận biết độ không chắc chắn A4
  -> đồ thị quan hệ/tiếp xúc A5
  -> khuếch tán bị che A6, K=1
  -> lựa chọn bằng chứng K-giả thuyết A7
  -> âm vận học và pha A8/A9 ở Phase 4
```

### 2.3 Các mục tiêu ngoài phạm vi khác

- thay đổi bộ đánh giá của tác giả hoặc quần thể khung hình;
- huấn luyện trên `data/smplx_gt` hoặc `data/evaluation_from_author`;
- tinh chỉnh (tuning) trên 57 ký hiệu SGNify;
- fine-tuning WiLoR, SMPLer-X, Sapiens, hoặc chuyên gia quan sát khác;
- trình bày DPoser-X, FUSION, hoặc một tập hợp chuyên gia như là điểm mới của bài báo;
- lựa chọn best-of-`K` bằng cách sử dụng GT;
- ép buộc tiếp xúc bất cứ khi nào hai bề mặt gần nhau;
- khuếch tán camera, dịch chuyển, hình dạng thân thể, hoặc tham số khuôn mặt trong mô hình đầu tiên; và
- ghi đè các thư mục đầu ra của Phase 1, Phase 2, DexAvatar, hoặc legacy.

---

## 3. Thực tế repo và dữ liệu rằng buộc kế hoạch

### 3.1 Các tài sản đã sẵn sàng

Các phép đo sau đến từ repo hiện tại và các kiểm toán Phase 2 đã có.

| Tài sản | Trạng thái địa phương | Vai trò Phase 3 | Giới hạn |
|---|---:|---|---|
| Dữ liệu thô How2Sign | khoảng 86 GB | quan sát RGB/2D miền ký hiệu và thích ứng không trùng lặp nguồn | không có GT SMPL-X mét sạch đồng nhất |
| `cache/phase2/how2sign_t1_v1` | 11.000 đoạn video huấn luyện / 352.000 khung hình; 1.200 đoạn kiểm định / 38.400 khung hình | thử nghiệm chuyển động ký hiệu và che mặt nạ | giáo viên giả SMPL-X H32, không phải A1 chính xác |
| Cache chiếu lại How2Sign | 10.822 đoạn huấn luyện / 498 kiểm định / 497 hiệu chỉnh | thử nghiệm điều kiện hóa quan sát | miền thặng dư khác biệt mạnh mẽ với Lane A1 |
| Dữ liệu địa phương ARCTIC | khoảng 20 GB; 301 chuỗi / 218.273 khung hình thô | tiền huấn luyện SMPL-X hoàn chỉnh và hình học quan hệ/tiếp xúc | tiếp xúc tay–vật thể chứ không phải tay–ký hiệu |
| `cache/phase2/arctic_t1_v1` | 2.351 đoạn huấn luyện / 146.781 khung hình; 511 đoạn kiểm định / 31.822 khung hình | khuếch tán bị che toàn thân sạch và kiểm định thời gian | miền tổng quát |
| Chú thích địa phương InterHand2.6M | 1.148 đoạn huấn luyện / 16.096 khung hình; 47 đoạn kiểm định / 2.736 khung hình | tiền huấn luyện khớp nối trái/phải và quan hệ tay–tay | chú thích tay một phần, tập con địa phương bị giới hạn |
| Bản phát hành địa phương PHOENIX-2014T | 4.121 khung hình PNG đã trích xuất có sẵn địa phương cộng với tài sản đánh giá | thử nghiệm nhất quán 2D/định tính đa ngôn ngữ tùy chọn | không hoàn chỉnh cho huấn luyện quy mô lớn và không có mục tiêu 3D mét tương thích |
| Checkpoint ký hiệu địa phương DPoser-X | `checkpoints/dposer/sign/sign_body_ft/last.ckpt` | thử nghiệm phân tách bộ thích ứng chỉ dành cho thân tùy chọn | chỉ có thân; không phải mô hình toàn thân thân-cộng-tay |
| Checkpoint Phase 2 ARCTIC T1 | G3 tổng quát GO | thử nghiệm khởi tạo ấm xác định tùy chọn | không phải checkpoint miền ký hiệu hoặc đạt Lane-G6 |
| Checkpoint Phase 2 U1 v7 | hiệu chỉnh số liệu miền H32 mạnh mẽ | chỉ thử nghiệm chẩn đoán | G5 chính thức NO-GO và sai miền khởi tạo cuối cùng |

### 3.2 Các tài sản chưa sẵn sàng và do đó bị cấm dưới dạng phụ thuộc ẩn

| Tài sản | Phát hiện địa phương hiện tại | Hành động yêu cầu |
|---|---|---|
| Trọng số toàn thân DPoser-X công khai | `DPoser-X/pretrained_models/` không chứa trọng số sử dụng được | tải xuống và băm (hash) `wholebody/mixed/last.ckpt` cộng với các mô hình phụ thân/tay yêu cầu |
| SignAvatars | không có sẵn địa phương | thu thập theo điều khoản nghiên cứu của nó và xây dựng cache phân biệt người ký/nguồn |
| Motion-X | tệp nén 498 MB địa phương thất bại trong việc xác thực thư mục trung tâm ZIP | tải lại và xác minh, hoặc bỏ qua vì DPoser-X đã cung cấp tiền huấn luyện tổng quát |
| WHIM | kho lưu trữ huấn luyện nhiều phần địa phương không hoàn chỉnh | hoàn thành và xác minh chỉ khi thử nghiệm phân tách dữ liệu tay yêu cầu |
| DexYCB | thư mục địa phương thực sự trống | thu thập chỉ khi yêu cầu; đây không phải điểm chặn Phase 3 |
| FUSION | không ghi nhận phụ thuộc được phê duyệt hoặc quyền phân phối lại | giữ tùy chọn phía sau rào cản giấy phép/trọng số phái sinh bằng văn bản |
| Đầu ra exact-A1 How2Sign | ngăn xếp bị đóng băng thất bại ở hợp đồng phân đoạn/ký hiệu tiếng Đức hard-code | không bắt buộc bởi tuyến Phase 3 độc lập; không bao giờ giả mạo nhãn ký hiệu để bỏ qua nó |

Bản phát hành chính thức của [DPoser-X](https://dposer.github.io/) mô tả một prior khuếch tán toàn thân bị che và xuất bản mô hình `wholebody/mixed/last.ckpt` thông qua [kho lưu trữ trọng số chính thức](https://huggingface.co/Moon-bow/DPoser-X). Repo hiện tại chưa chứa tệp đó. Do đó, việc tải hoàn tất và độ tương thích checkpoint là các cổng chính thức, không phải giả định.

### 3.3 Diễn giải tập dữ liệu

- [SignAvatars](https://signavatars.github.io/) cung cấp tập dữ liệu SMPL-X đặc thù ký hiệu lớn nhất liên quan trực tiếp và là nguồn thích ứng cuối cùng được ưu tiên. Các chú thích SMPL-X của nó là chú thích tái dựng/giả, vì vậy chúng yêu cầu lọc chất lượng và không được gọi là mocap GT.
- [How2Sign](https://how2sign.github.io/index.html) cung cấp hơn 80 giờ ASL liên tục và các phân chia huấn luyện/kiểm định/kiểm thử chính thức. Trong dự án này, các chuỗi H32 SMPL-X của nó có ích làm mục tiêu giả chuyển động ký hiệu và các quan sát RGB/2D của nó có ích cho việc điều kiện hóa bằng chứng.
- [ARCTIC](https://arctic.is.tue.mpg.de/) cung cấp hình học SMPL-X/MANO đồng bộ và tiếp xúc tay–vật thể động. Nó có giá trị cho việc học hình học thời gian chính xác, nhưng các tiếp xúc vật thể của nó không thể được dán nhãn lại thành tiếp xúc tay–mặt hoặc tay–ngực ký hiệu.
- [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) cung cấp các chú thích tay đơn/tương tác chính thức. Nó có giá trị cho định danh tay và hình học tay–tay, nhưng không giám sát cử chỉ toàn thân trên.
- [Motion-X](https://motion-x-dataset.github.io/) là một nguồn toàn thân tổng quát lớn tùy chọn sau khi lưu trữ địa phương được sửa chữa và giấy phép được ghi nhận. Dữ liệu tổng quát phải được giảm trọng số trong quá trình thích ứng ký hiệu.
- Dữ liệu đánh giá SGNify giữ nguyên chế độ chỉ-đánh-giá dưới mọi cấu hình.

---

## 4. Các giả thuyết khoa học

### H3.1 Điều kiện hóa quan hệ

Hình học tương đối rõ ràng giữa cổ tay, lòng bàn tay, đầu ngón tay, khuôn mặt, ngực, vai và cánh tay trên sẽ cải thiện các khung hình bị che/tương tác vì các mối quan hệ này vẫn cung cấp thông tin khi một quan sát tay cục bộ bị thất bại.

### H3.2 Khuếch tán toàn chuỗi bị che

Một phân phối chuỗi có điều kiện có thể đại diện cho nhiều sự hoàn thiện hợp lý cho sự mất mát tay dài, sự mơ hồ về chiều sâu, lật lòng bàn tay và tay bắt chéo, nơi một mô hình thặng dư xác định có xu hướng trung bình hóa hoặc sao chép giáo viên.

### H3.3 Các giả thuyết được lựa chọn theo bằng chứng

Khi lựa chọn ứng viên sử dụng các quan sát được giữ lại từ quá trình tạo ứng viên, các giả thuyết `K = 4` được chọn sẽ vượt trội hơn phân phối hậu phương một đường duy nhất và thu hẹp một phần đáng kể khoảng cách ứng viên oracle mà không cần dùng GT.

### H3.4 Phase 3 độc lập

Một điểm số không gian được huấn luyện trước cộng với các bộ thích ứng thời gian/quan hệ mới có thể cung cấp giá trị ngay cả khi checkpoint xác định Phase 2 không thể chuyển giao. Nếu giả thuyết này thất bại, độ phức tạp của posterior không được biện minh và dự án nên quay lại hình học mục tiêu/quan sát thay vì tiến sang Phase 4.

---

## 5. Phân phối hậu phương hình thức và trạng thái

Gọi các quan sát cố định là `O`, độ tin cậy cố định là `U`, và các mối quan hệ hình học dự đoán là `R`. Phase 3 mô hình hóa:

$$
p_{\theta}(X_{1:T}\mid O_{1:T}, U_{1:T}, R_{1:T}).
$$

Không có biến âm vận học nào xuất hiện trong Phase 3.

Đối với mỗi khung hình:

$$
X_t = [
R_t^{body,21},
R_t^{left,15},
R_t^{right,15}
],
$$

với 51 phép xoay khớp cục bộ. Mô hình nhận tất cả 51 khớp làm ngữ cảnh, nhưng tập hợp có thể thay đổi mặc định là tập hợp an toàn về danh tính được sử dụng bởi Phase 2:

- spine1, spine2, spine3;
- cổ và cả hai xương đòn;
- cả hai vai, khuỷu tay và cổ tay; và
- tất cả 15 khớp của mỗi tay.

Các phép xoay thân dưới, đầu, hàm, mắt, biểu cảm, định hướng toàn cục, dịch chuyển, camera và hình dạng chia sẻ được sao chép từ bộ khởi tạo trong cấu hình chính.

### 5.1 Biểu diễn khuếch tán

Sử dụng hệ tọa độ xoay 6D liên tục đã chuẩn hóa cho SDE tiến:

$$
z_0 = \operatorname{rot6d}(X_{1:T}),
$$

sau đó chiếu từng phép xoay sạch được dự đoán trở lại `SO(3)` bằng Gram–Schmidt trước khi tính toán tổn thất hình học hoặc giải mã SMPL-X. Trục-góc (Axis-angle) chỉ được sử dụng cho tính tương thích cache và xuất dữ liệu.

Đây là một biểu diễn khuếch tán Euclidean với sự chiếu đa tạp rõ ràng. Nó phải vượt qua kiểm thử vòng lặp (round-trip) và độ dốc (gradient) trước khi huấn luyện. Các giá trị trục-góc thô không bao giờ được trung bình hóa giữa các ứng viên hoặc cửa sổ.

### 5.2 Các khung tọa độ chuẩn

- các khớp thân thể và quan hệ xuyên bộ phận: khung tọa độ trung tâm thân;
- khớp nối tay: khung tọa độ cục bộ cổ tay;
- định hướng lòng bàn tay: cả cục bộ cổ tay và tương đối với thân;
- khoảng cách tay–thân thể: khung tọa độ thân sử dụng các điểm neo SMPL-X giải mã;
- bằng chứng 2D: khung tọa độ camera với ma trận nội suy đã được cache;
- hình dạng: một vectơ `betas` trung vị mạnh mẽ cho mỗi chuỗi; và
- thời gian chuỗi: giây cộng với chỉ số khung hình chuẩn hóa, không chỉ dùng chỉ số khung hình.

Các phép biến đổi tọa độ Phase 2 và kiểm thử vòng lặp được tái sử dụng dưới dạng chỉ đọc. Cache Phase 3 chỉ có thể tham chiếu một đoạn Phase 2 nếu phiên bản schema và băm biến đổi của nó được chấp nhận.

---

## 6. Hợp đồng quan sát và độ tin cậy

### 6.1 Tuyến quan sát chính

Tuyến có thể tái lập chính đóng băng ngăn xếp quan sát A1 đã được chọn cho đánh giá cuối cùng và tiêu thụ:

- các phép xoay SMPL-X và khớp được giải mã của bộ khởi tạo;
- quan sát tay tương thích WiLoR/HaMeR khi có sẵn;
- các khớp toàn thân 2D Sapiens và độ tin cậy ban đầu;
- ma trận nội suy camera;
- hình học 3D cục bộ thân/cổ tay;
- tâm/pháp tuyến lòng bàn tay, đầu ngón tay, tâm MCP và đính kèm cổ tay;
- các đặc trưng bị thiếu, cắt xén, đột biến thời gian (temporal innovation) và bất đồng chuyên gia; và
- mặt nạ nguồn/nguồn gốc.

### 6.2 Quyết định độ tin cậy

Sử dụng độ tin cậy `U0` xác định trong thử nghiệm Phase 3 chính.

Lý do:

1. nó sẵn có cho tất cả các cache;
2. nó không kế thừa sự không phù hợp chính thức giữa H32 và A1;
3. nó cô lập đóng góp của Phase 3 khỏi một mô hình độ không chắc chắn chưa được chấp nhận; và
4. nó cung cấp một phương án dự phòng ổn định khi vắng mặt quan sát.

U1 v7 chỉ có thể xuất hiện dưới dạng thử nghiệm phân tách miền H32. Nó chỉ có thể trở thành một phần của cấu hình chính sau này nếu được hiệu chỉnh lại trên một cache biệt lập nguồn có nguồn gốc khởi tạo khớp với miền quan sát Phase 3 và vượt qua tất cả các tiêu chuẩn Phase 2 G5 chính thức.

### 6.3 Phân chia điều kiện/bằng chứng

Việc lựa chọn bằng chứng không được chấm điểm cho một ứng viên bằng chính các quan sát đã ép buộc tạo ra ứng viên đó.

Đối với mọi chuỗi huấn luyện và kiểm định:

1. phân tầng các quan sát 2D/3D có độ tin cậy cao theo vùng và thời gian;
2. giữ lại 20% làm token bằng chứng, đảm bảo mọi vùng đều giữ lại quan sát điều kiện hóa;
3. tạo các ứng viên từ 80% còn lại;
4. xếp hạng ứng viên theo dự đoán của chúng đối với bằng chứng được giữ lại; và
5. tinh chỉnh ứng viên được chọn tùy chọn với tất cả quan sát.

Khi đánh giá, sử dụng bốn nếp gấp (fold) bằng chứng xác định và trung bình hóa điểm số ứng viên của chúng. Seed phân chia được dẫn xuất từ băm của đoạn video, không chọn theo từng kết quả.

---

## 7. Đồ thị quan hệ tay–thân thể

### 7.1 Các nút (Nodes)

Sử dụng một đồ thị cố định, nhỏ thay vì toàn bộ 10.475 đỉnh:

- cổ tay trái/phải;
- tâm và pháp tuyến lòng bàn tay trái/phải;
- 10 điểm neo MCP;
- 10 đầu ngón tay;
- đầu, cằm, ngực trên, xương ức và xương chậu/gốc;
- vai và cánh tay trên trái/phải; và
- điểm neo bề mặt SMPL-X gần nhất tùy chọn cho tiếp xúc dự đoán.

### 7.2 Các cạnh ứng viên (Edges)

- cổ tay-đến-cổ tay;
- lòng bàn tay-đến-lòng bàn tay;
- đầu ngón tay-đến-đầu ngón tay đối diện;
- đầu ngón tay-đến-lòng bàn tay đối diện;
- mỗi lòng bàn tay/đầu ngón tay đến điểm neo mặt/cằm/ngực/vai/cánh tay trên;
- cổ tay-đến-khuỷu tay của chính nó và lòng bàn tay-đến-cẳng tay của chính nó; và
- cạnh tự thân theo thời gian cho sự duy trì tiếp xúc.

Mỗi token cạnh chứa:

- vectơ 3D tương đối và khoảng cách;
- vận tốc và gia tốc tương đối;
- định hướng lòng bàn tay tương đối;
- thứ tự chiều sâu trước/sau có dấu;
- độ chồng phủ 2D và khả năng nhìn thấy;
- độ tin cậy của cả hai điểm đầu cuối;
- tính hợp lệ của cạnh; và
- xác suất tiếp xúc của khung hình trước đó.

### 7.3 Bộ mã hóa quan hệ

Cấu hình bắt đầu:

| Mục | Giá trị |
|---|---:|
| kích thước ẩn nút | 256 |
| kích thước ẩn cạnh | 128 |
| số lớp đồ thị | 4 |
| số đầu attention | 8 |
| dropout | 0.1 |
| ngữ cảnh cạnh thời gian | 5 khung hình căn giữa khung hình hiện tại |
| đầu ra | token quan hệ, logit tiếp xúc, logit thứ tự chiều sâu, logit duy trì |

Đồ thị phát ra các token quan hệ cấp khung hình và xác suất theo từng cạnh. Những giá trị này điều kiện hóa attention giữa các bộ phận trong bộ giải nhiễu khuếch tán.

### 7.4 Nhãn tiếp xúc và hành vi an toàn dự phòng

Tiếp xúc giả được dẫn xuất từ hình học chỉ dương tính khi tất cả điều kiện sẵn có đồng thuận:

- khoảng cách bề mặt được giải mã dưới 12 mm cho khởi phát tiếp xúc;
- tiếp xúc vẫn duy trì cho đến khi khoảng cách vượt quá 20 mm;
- tốc độ tiếp tuyến tương đối dưới 0,15 m/s đối với tiếp xúc duy trì;
- cả hai điểm đầu cuối đều hợp lệ; và
- thâm nhập nằm dưới mức dung sai an toàn cho phép.

Đây là các ngưỡng bắt đầu. Chúng chỉ có thể thay đổi bằng cách sử dụng tập quan hệ bên ngoài được xác minh thủ công, không bao giờ dùng Lane-L.

Tiếp xúc luôn mang tính xác suất. Năng lượng tiếp xúc được đưa về 0 khi:

- xác suất dưới 0,6;
- độ không chắc chắn điểm đầu cuối nằm ngoài phạm vi huấn luyện;
- cạnh ứng viên không hợp lệ hoặc nằm ngoài khung hình;
- entropy vị trí tiếp xúc vượt quá ngưỡng kiểm định; hoặc
- việc ép buộc tiếp xúc làm tăng lỗi quan sát đáng tin cậy vượt quá mức dung sai đóng băng.

Tiếp xúc vật thể ARCTIC có thể tiền huấn luyện các khái niệm tổng quát về tiếp cận, duy trì và trượt, nhưng nó không thể dán nhãn trực tiếp cho các cạnh tay–thân thể ký hiệu. InterHand cung cấp hình học tay–tay. Tiếp xúc tay–thân thể đặc thù ký hiệu được dẫn xuất từ SMPL-X SignAvatars/How2Sign đã lọc và được kiểm tra trên một tập con được xác minh thủ công.

---

## 8. Kiến trúc khuếch tán quan hệ

### 8.1 Khung xương (Backbone)

Sử dụng một mạng điểm số không-thời gian phân tách trên các token khớp `B x T x 51`:

1. điểm số không gian toàn thân DPoser-X theo từng khung hình cố định;
2. phép chiếu có thể huấn luyện từ biểu diễn DPoser-X sang trạng thái Phase 3 51-khớp;
3. attention nội bộ bộ phận cho thân/cánh tay, tay trái và tay phải;
4. attention xuyên bộ phận thông qua các token cổ tay và quan hệ;
5. attention thời gian hai chiều cho mỗi khớp và cạnh quan hệ; và
6. attention chéo tới các token quan sát, tính hợp lệ, U0 và thời gian khuếch tán.

Điểm số dự đoán là phần thặng dư xung quanh prior không gian cố định:

$$
s_{RDP}(z_t,t,c) =
s_{DPoserX}(z_t,t)
+
\Delta s_{temporal-rel}(z_t,t,c).
$$

Phép chiếu điểm số thặng dư được khởi tạo bằng 0. Trước khi huấn luyện, RDP phải tái tạo điểm số DPoser-X cố định với độ chính xác số khi nhánh thặng dư bị tắt.

### 8.2 Kích thước mạng bắt đầu

| Thành phần | Giá trị bắt đầu |
|---|---:|
| cửa sổ tối đa | 64 khung hình |
| token khớp | 51 |
| chiều rộng mô hình | 384 |
| các khối luân phiên | 8 |
| đầu attention | 8 |
| tỷ lệ MLP | 4 |
| dropout | 0.1 |
| độ lệch thời gian tương đối | cắt ở 64 khung hình |
| token nhóm | thân, cánh tay trái, cánh tay phải, tay trái, tay phải, quan hệ |
| mục tiêu tham số huấn luyện được | khoảng 45–70 triệu, không tính DPoser-X cố định |
| checkpointing kích hoạt | bật cho các khối thời gian và xuyên bộ phận |

Attention phân tách là bắt buộc. Full attention trên tất cả `64 x 51` token chỉ là một thử nghiệm phân tách nếu phân tích bộ nhớ và thời gian chạy biện minh cho nó.

### 8.3 Quá trình khuếch tán

Khớp với công thức sub-VP liên tục của DPoser-X công khai trong tuyến chính:

| Mục | Giá trị |
|---|---:|
| SDE | sub-VP liên tục |
| `beta_min` | 0.1 |
| `beta_max` | 20.0 |
| tỷ lệ định danh (nominal scales) | 1.000 |
| phạm vi thời gian huấn luyện | `[1e-3, 1.0]` |
| mục tiêu | khớp điểm số khử nhiễu (denoising score matching), cân bằng vùng |
| trọng số khả năng (likelihood weighting) | tắt ban đầu, khớp DPoser-X |
| tỷ lệ điểm số (score scaling) | theo độ lệch chuẩn lề |
| EMA | 0.9999 |

Giữ cho SDE và sự chuẩn hóa tương thích cho phép điểm số DPoser-X cố định được sử dụng trực tiếp. Một tuyến cosine/DDPM/EDM khác phải là một thử nghiệm phân tách riêng biệt và không thể tái sử dụng tuyên bố prior được huấn luyện trước mà không qua chuyển đổi đã được xác thực.

### 8.4 Điều kiện hóa bị che

Đối với mỗi mẫu, độc lập chọn một mặt nạ hợp lệ cho giám sát:

| Mặt nạ/Biến dạng | Xác suất bắt đầu |
|---|---:|
| quan sát đầy đủ / khuếch tán thông thường | 20% |
| bùng nổ tay trái (left-hand burst) | 12% |
| bùng nổ tay phải (right-hand burst) | 12% |
| bùng nổ cả hai tay (both-hand burst) | 10% |
| một chuỗi ngón tay | 10% |
| đính kèm cổ tay/cẳng tay | 10% |
| bùng nổ thân trên/cánh tay | 8% |
| tráo đổi tay hoặc mơ hồ đối xứng | 6% |
| mơ hồ lòng bàn tay/chiều sâu | 6% |
| bỏ rơi 2D/cắt xén | 6% |

Độ dài bùng nổ được lấy mẫu từ 4, 8 và 16 khung hình với xác suất bằng nhau trong quá trình huấn luyện phục hồi chính thức. Các bùng nổ bổ sung 2–12 khung hình có thể được dùng cho tăng cường dữ liệu thông thường.

Bỏ rơi điều kiện classifier-free bắt đầu ở mức 10%. Hướng dẫn (guidance) bắt đầu ở mức `1.0` và có thể so sánh với `1.2` và `1.5` trên kiểm định bên ngoài. Hướng dẫn cao hơn bị cấm nếu nó làm giảm sự đa dạng của ứng viên hoặc làm tồi tệ hơn bất kỳ vùng sạch nào.

---

## 9. Chiến lược mô hình huấn luyện trước

### 9.1 Chính: checkpoint trộn toàn thân DPoser-X công khai

Tải xuống và ghi ghim:

```text
DPoser-X/pretrained_models/body/BaseMLP/last.ckpt
DPoser-X/pretrained_models/hand/BaseMLP/last.ckpt
DPoser-X/pretrained_models/wholebody/mixed/last.ckpt
```

Mô hình khuôn mặt không cần thiết cho Phase 3 vì trạng thái khuôn mặt bị đóng băng. Nếu bộ tải toàn thân chính thức yêu cầu nhánh khuôn mặt của nó, hãy tải nó ở trạng thái đóng băng nhưng loại trừ khuôn mặt khỏi trạng thái và tổn thất của Phase 3.

Nạp checkpoint phải ghi lại:

- URL nguồn và ngày lấy;
- SHA-256 của từng tệp;
- commit mã nguồn;
- giấy phép mã nguồn và giấy phép trọng số mô hình riêng biệt;
- thống kê chuẩn hóa;
- biểu diễn xoay dự kiến và thứ tự khớp; và
- độ bao phủ khóa tensor chính xác trong quá trình tải.

### 9.2 Cổng tương thích

Tái sử dụng điểm số không gian trực tiếp chỉ được chấp nhận nếu:

- 100% các tensor prior thân và tay yêu cầu tải được mà không cần ép hình dạng (shape coercion);
- thứ tự khớp DPoser-X được ánh xạ rõ ràng sang thứ tự 51 khớp Phase 3;
- lỗi vòng lặp chuẩn hóa dưới `1e-6` trong tọa độ chuẩn hóa;
- bộ thích ứng tái tạo điểm số theo khung hình chính thức trong vòng lỗi tuyệt đối tối đa `1e-5` trên 100 tư thế cố định; và
- các kiểm thử kiểm tra hoàn thành/tạo mới vẫn duy trì giá trị hữu hạn.

Nếu chuyển giao trực tiếp thất bại, **không** tải một phần các tên khớp ngẫu nhiên. Sử dụng DPoser-X như một giáo viên đóng băng:

1. lấy mẫu các tư thế sạch bị biến dạng;
2. thu được tư thế/điểm số đã khử nhiễu từ DPoser-X;
3. chưng cất nó vào nhánh không gian Phase 3; và
4. so sánh chưng cất với khởi tạo không gian ngẫu nhiên.

### 9.3 Checkpoint ký hiệu chỉ dành cho thân địa phương

`DPoser-X/checkpoints/dposer/sign/sign_body_ft/last.ckpt` chỉ có thể khởi tạo đường dẫn 21 khớp thân. Nó không bao giờ được khởi tạo hoặc mô tả là prior của tay. Bao gồm ba hàng được kiểm soát:

- prior thân DPoser-X công khai;
- prior thân công khai cộng với bộ thích ứng thân-ký hiệu địa phương; và
- bộ thích ứng thân-ký hiệu địa phương không có chuyển giao tay.

Chỉ giữ lại bộ thích ứng thân-ký hiệu nếu nó cải thiện kiểm định ký hiệu biệt lập nguồn mà không làm suy giảm cả hai tay.

### 9.4 Khởi tạo Phase 2 tùy chọn

Checkpoint ARCTIC T1 có thể khởi tạo các nhúng thời gian hoặc các phép chiếu hình học được chọn chỉ sau khi có báo cáo tương thích ở cấp độ tensor. Nó vẫn là một thử nghiệm phân tách vì kết quả được chấp nhận của nó là G3 tổng hợp tổng quát, không phải G6 miền ký hiệu.

Không có kết quả Lane T2/T5/U1 nào được sử dụng trong tuyến Phase 3 chính.

### 9.5 Tuyến FUSION tùy chọn

[FUSION](https://arxiv.org/abs/2601.03959) có liên quan về mặt kỹ thuật như một prior khuếch tán thời gian thân–tay. Nó không phải là một phần của cấu hình tái lập chính trừ khi dự án ghi nhận phép ủy quyền bằng văn bản để:

- chỉnh sửa/fine-tune mã và checkpoint;
- phân phối lại trọng số phái sinh hoặc các bộ thích ứng; và
- xuất bản kết quả nghiên cứu dự kiến.

Nếu không có phép ủy quyền này, FUSION chỉ có thể là một so sánh đóng băng nếu các điều khoản của nó cho phép. Bản phát hành không được phụ thuộc vào sự phê duyệt giấy phép muộn.

### 9.6 Các mô hình cố ý không được sử dụng trong Phase 3

- SHuBERT: dành riêng cho âm vận học/pha ở Phase 4;
- một khung xương RGB có thể huấn luyện mới: không cần thiết cho kiểm thử posterior-quan hệ đầu tiên;
- fine-tuned WiLoR/SMPLer-X/Sapiens: các chuyên gia quan sát giữ nguyên trạng thái đóng băng; và
- U1 v7: chỉ thử nghiệm chẩn đoán miền H32 cho đến khi có hiệu chỉnh chính thức theo miền chính xác.

---

## 10. Chiến lược tập dữ liệu

### 10.1 Ba tầng giám sát

| Tầng | Mục đích | Nguồn chính | Giám sát |
|---|---|---|---|
| A: hình học sạch/tổng quát | học điểm số toàn thân/tay hợp lệ và hình học thời gian | Các miền DPoser-X, ARCTIC, Motion-X đã sửa | Tham số SMPL-X/MANO, khớp, đỉnh, mặt nạ hợp lệ |
| B: quan hệ/tiếp xúc | học định danh tay, hình học tương đối, khởi phát/duy trì tiếp xúc | InterHand2.6M, ARCTIC, tập con ký hiệu đã xác minh | Các cạnh, khoảng cách, thứ tự, tiếp xúc, trượt |
| C: thích ứng ký hiệu | học phân phối chuyển động ký hiệu và sự thất bại quan sát | SignAvatars, How2Sign, PHOENIX không nhãn | Mục tiêu giả SMPL-X đã lọc, bằng chứng RGB/2D, sự thiếu hụt |

### 10.2 Phối hợp dữ liệu cấp bài báo

Thu thập SignAvatars trước mô hình cuối cùng. Quy mô và sự đa dạng miền ký hiệu của nó khiến nó trở thành nguồn thích ứng Phase 3 chính. Bộ lấy mẫu chuỗi cuối cùng nên sử dụng các tỷ lệ cấp đoạn video này:

| Nguồn | Tỷ lệ trong quá trình thích ứng ký hiệu | Mục đích |
|---|---:|---|
| SignAvatars | 50% | phân phối toàn thân đặc thù ký hiệu chính |
| How2Sign | 25% | ASL liên tục, thất bại quan sát thực tế |
| ARCTIC | 15% | giữ lại hình học toàn thân/hai tay chính xác |
| InterHand2.6M | 10% | giữ lại khớp nối và định danh tay tương tác |

Motion-X có thể thay thế tối đa một nửa phần của ARCTIC trong quá trình tiền huấn luyện tổng quát. Chuyển động tổng quát không được vượt quá 30% trong quá trình thích ứng ký hiệu cuối cùng.

### 10.3 Phối hợp thử nghiệm sẵn sàng ngay

Trước khi SignAvatars có sẵn, một bản thử nghiệm kỹ thuật/khoa học có thể sử dụng:

| Nguồn | Tỷ lệ thử nghiệm |
|---|---:|
| Chuỗi giả H32 How2Sign | 45% |
| ARCTIC | 30% |
| InterHand2.6M | 15% |
| Các đoạn quan sát tự giám sát PHOENIX | 0–10% tùy chọn, lấy từ phần của How2Sign chỉ sau khi kiểm toán chuỗi/độ bao phủ |

Nếu không có PHOENIX, sử dụng `55% How2Sign / 30% ARCTIC / 15% InterHand`. Bản thử nghiệm này có thể vượt qua các cổng triển khai, quan hệ, che mặt nạ và lấy mẫu. Nó không thể thiết lập tuyên bố prior ký hiệu cấp bài báo cuối cùng vì How2Sign H32 là một mục tiêu giả giáo viên đơn lẻ và PHOENIX không có mục tiêu 3D tương thích.

### 10.4 Mặt nạ tổn thất đặc thù theo nguồn

- ARCTIC: đầy đủ 51 phép xoay khớp và hình học SMPL-X; quan hệ/tiếp xúc chỉ ở nơi nhãn tương ứng hợp lệ.
- InterHand: khớp nối tay trái/phải, các khớp tay và quan hệ tay–tay; tổn thất thân thể bằng 0.
- SignAvatars: mục tiêu đầy đủ ở nơi mặt nạ chất lượng vượt qua; trọng số độ tin cậy hình học giả và loại trừ các khung hình thảm họa.
- How2Sign: chuyển động ký hiệu và tính nhất quán quan sát; trọng số hình học mục tiêu giả bắt đầu ở mức 0,25 tương đối với dữ liệu sạch.
- PHOENIX: chỉ nhất quán quan sát 2D/thời gian; không tạo tổn thất 3D bịa đặt.
- Motion-X: hình học đầy đủ sau khi kiểm toán kho lưu trữ, giấy phép, tọa độ và chất lượng vượt qua.

Các nhãn bị thiếu phải được đại diện bằng mặt nạ. Chúng không bao giờ được thay thế bằng một tư thế bằng 0.

### 10.5 Chính sách phân chia và rò rỉ dữ liệu

Sử dụng nhóm nguồn, video, người ký và chuỗi — không dùng cửa sổ — làm đơn vị phân chia.

- giữ nguyên các phân chia kiểm thử tập dữ liệu chính thức;
- sử dụng phân chia biệt lập người ký ở bất kỳ đâu có ID người ký;
- đảm bảo các cửa sổ liền kề từ một video nguồn nằm trong cùng một phần phân chia;
- khử trùng lặp các khung hình RGB bằng băm cảm nhận và SHA-256 chính xác;
- khử trùng lặp các chuỗi SMPL-X bằng băm tư thế chuẩn hóa và siêu dữ liệu nguồn;
- từ chối bất kỳ đường dẫn hoặc băm nào dưới `data/smplx_gt` hoặc `data/evaluation_from_author`;
- giữ tất cả 57 ký hiệu SGNify nằm ngoài huấn luyện/kiểm định/hiệu chỉnh; và
- xuất bản bản kê khai (manifests) phân chia và các băm.

### 10.6 Lọc chất lượng mục tiêu

Đối với các nguồn SMPL-X giả, tính toán điểm chất lượng chuỗi từ:

- lỗi chiếu lại 2D;
- tính nhất quán định danh trái/phải;
- độ ổn định hình dạng;
- tính hợp lệ của xoay và lưới;
- tính nhất quán độ dài xương tay;
- bước nhảy pháp tuyến lòng bàn tay;
- đính kèm cổ tay/cẳng tay;
- điểm ngoại lệ gia tốc/độ giật (jerk);
- thâm nhập lẫn nhau; và
- sự đồng thuận với chuyên gia đóng băng thứ hai khi có sẵn.

Sử dụng ba dải:

| Dải | Quy tắc | Sử dụng |
|---|---|---|
| Q0 sạch | không có cờ thảm họa; chất lượng 60% top đầu | tất cả tổn thất hình học và khuếch tán |
| Q1 dùng được | không có cờ thảm họa; 30% giữa | tổn thất hình học theo trọng số độ tin cậy và nhất quán quan sát |
| Q2 yếu | 10% dưới cùng hoặc bất đồng nghiêm trọng | chỉ làm đầu vào bị biến dạng/bằng chứng không nhãn; không bao giờ là mục tiêu sạch |

Đánh giá thủ công ít nhất 300 đoạn ký hiệu biệt lập nguồn, lấy mẫu vượt trội các đoạn hai tay/tiếp xúc và chuyển động cao. Ít nhất 10% phải được đánh giá kép. Lỗi mục tiêu thảm họa phải dưới 10%, và các nhãn tiếp xúc phải báo cáo sự đồng thuận giữa các người đánh giá trước khi chạy chính thức.

---

## 11. Giáo trình biến dạng và khôi phục

Làm biến dạng quan sát, không làm biến dạng mục tiêu sạch.

### 11.1 Biến dạng tổng hợp

- mất hoàn toàn tay trong 4/8/16 khung hình;
- mất một chuỗi ngón tay trong 2–8 khung hình;
- lỗi cổ tay 10–45 độ;
- lật định hướng lòng bàn tay độc lập;
- lỗi gãy cổ tay kết hợp thân/tay;
- tráo đổi định danh trái/phải;
- tráo đổi chiều sâu tay tương tác;
- nhiễu 2D tỷ lệ với độ phân giải cắt xén;
- cắt xén hình ở mức 10%, 25% và 40%;
- mờ chuyển động và giảm độ phân giải khi có RGB;
- bất đồng chuyên gia thân/tay được lấy mẫu từ thặng dư kiểm định bên ngoài; và
- khoảng trống khung hình và tốc độ khung hình biến đổi.

### 11.2 Thặng dư thực tế

Chỉ sử dụng cặp thặng dư chuyên gia đóng băng thực tế khi mục tiêu độc lập với chuyên gia đầu vào. Cặp định danh H32-đến-H32 không phải là giám sát hiệu chỉnh thực tế.

Các ví dụ hợp lệ bao gồm:

- ARCTIC/InterHand GT so với đầu ra chuyên gia đóng băng;
- mục tiêu đồng thuận chất lượng cao SignAvatars so với các quan sát đóng băng;
- mục tiêu tinh chỉnh đa góc nhìn hoặc ngoại tuyến mạnh hơn so với đầu ra chuyên gia đơn góc nhìn; và
- các chuỗi được chấp nhận thủ công với bằng chứng hình học độc lập.

Cache How2Sign H32 có thể dạy phân phối ký hiệu, nhưng kết quả dịch chuyển miền Phase 2 của nó cấm việc dán nhãn lại nó thành giám sát thặng dư exact-A1.

### 11.3 Điều kiện hóa sạch

Ít nhất 20% các batch giữ tất cả các quan sát không bị che. Điều này dạy sự tập trung posterior xung quanh một quan sát chính xác và là bắt buộc cho sự an toàn của tập sạch.

---

## 12. Các hàm tổn thất (Losses)

Mục tiêu bắt đầu là:

$$
\begin{aligned}
\mathcal{L} ={}&
\lambda_s L_{score}
+ \lambda_R L_{rot}
+ \lambda_V L_{region-vertex}
+ \lambda_J L_{joint}
+ \lambda_F L_{fingertip}
+ \lambda_P L_{palm} \\
&+ \lambda_{rel} L_{relation}
+ \lambda_C L_{contact}
+ \lambda_{slip} L_{slip}
+ \lambda_O L_{observation}
+ \lambda_M L_{motion}
+ \lambda_A L_{anchor}
+ \lambda_D L_{DPoser-distill}.
\end{aligned}
$$

### 12.1 Trọng số bắt đầu

| Tổn thất | Trọng số |
|---|---:|
| khớp điểm số (score matching) | 1.0 |
| xoay trắc địa (geodesic rotation) | 0.5 |
| đỉnh vùng bằng nhau (equal-region vertex) | 1.0 |
| vị trí khớp | 0.5 |
| đầu ngón tay | 2.0 |
| pháp tuyến/định hướng lòng bàn tay | 0.5 |
| hình học tương đối | 0.5 |
| phân loại/khoảng cách tiếp xúc | 0.25 |
| trượt/duy trì tiếp xúc | 0.10 |
| khả năng quan sát theo trọng số U0 | 0.50 |
| vận tốc/gia tốc mục tiêu | 0.25 |
| neo quan sát đáng tin cậy | 0.10 |
| chưng cất DPoser-X | 0.25, chỉ khi không thể tái sử dụng điểm số trực tiếp |

Đây là các giá trị bắt đầu, không phải các hằng số đã được tinh chỉnh cho benchmark. Thay đổi từng họ tổn thất tại một thời điểm bằng cách sử dụng kiểm định bên ngoài và ghi lại từng thử nghiệm.

### 12.2 Quy tắc tổn thất

- tổn thất điểm số được chuẩn hóa riêng cho thân trên, tay trái và tay phải;
- tổn thất hình học sử dụng ước lượng `x0` được chiếu của mô hình và mặt nạ hợp lệ nguồn;
- tổn thất hình học giải mã chạy ở FP32 ngay cả khi mạng sử dụng BF16;
- tổn thất đầu ngón tay và lòng bàn tay không bị pha giãng bởi số lượng đỉnh thân trên;
- tổn thất chuyển động khớp với vận tốc/gia tốc mục tiêu thay vì giảm thiểu chúng về 0;
- các quan sát đáng tin cậy nhận được một mỏ neo định danh mạnh hơn;
- tổn thất tiếp xúc sử dụng focal BCE cho các cạnh thưa cộng với một mục tiêu khoảng cách mềm;
- sự thâm nhập và sinh cơ học vẫn là các điều khoản an toàn, không phải mục tiêu chính; và
- hình dạng, dịch chuyển, camera và mặt không nhận cập nhật huấn luyện nào trong cấu hình đầu tiên.

Sử dụng trọng số hình học phụ trợ cắt gọt SNR với `gamma = 5`. Không áp dụng độ dốc đỉnh giải mã lớn ở các bước thời gian nhiễu cao nhất.

---

## 13. Chương trình huấn luyện theo thứ tự

Mỗi giai đoạn phụ thuộc vào giai đoạn trước. Dừng lại khi cổng của nó thất bại.

### Giai đoạn R0: hợp đồng, dữ liệu và rò rỉ

Xây dựng `cache/phase3/v1` dưới dạng các bản kê khai chỉ-thêm (append-only) và sidecar quan hệ tham chiếu các đoạn Phase 2 bất biến ở nơi có thể.

Đầu ra yêu cầu:

- bảng tập dữ liệu/giấy phép;
- các bản kê khai biệt lập nguồn/người ký/video;
- quét nguồn bị cấm;
- báo cáo tọa độ/thứ tự khớp;
- các băm checkpoint DPoser-X;
- báo cáo chất lượng/tiếp xúc thủ công 300 đoạn; và
- các băm dựng lại cache xác định.

**GO:** tất cả các phần phân chia đều biệt lập; không rò rỉ SGNify/đánh giá; tất cả giấy phép dữ liệu và checkpoint yêu cầu được ghi lại; thất bại mục tiêu thảm họa dưới 10%.

**NO-GO:** dừng trước khi huấn luyện. Không bù đắp cho dữ liệu bị thiếu bằng các mục tiêu benchmark.

### Giai đoạn R1: Nạp prior không gian DPoser-X

1. tải xuống các trọng số thân, tay và trộn toàn thân chính thức;
2. tái tạo một trường hợp kiểm tra hoàn thiện chính thức;
3. triển khai bộ thích ứng chuẩn hóa và ánh xạ 51 khớp rõ ràng;
4. xác minh tính tương đương điểm số đóng băng; và
5. đánh giá benchmark tái sử dụng trực tiếp so với chưng cất giáo viên trên kiểm định ARCTIC.

**GO:** tiêu chí tương thích ở Phần 9.2 vượt qua, và khởi tạo DPoser-X cải thiện kiểm định ARCTIC bị che ít nhất 5% so với khởi tạo ngẫu nhiên sau cùng ngân sách 10.000 bước.

**NO-GO:** sử dụng chưng cất giáo viên. Nếu cả chuyển giao trực tiếp lẫn chưng cất đều không giúp ích, hãy huấn luyện nhánh không gian một cách độc lập và loại bỏ tuyên bố prior được huấn luyện trước.

### Giai đoạn R2: Tiền huấn luyện đồ thị quan hệ/tiếp xúc

Huấn luyện trước tiên trên InterHand và ARCTIC, sau đó thích ứng trên tập con ký hiệu đã xác minh.

**GO:** trên kiểm định biệt lập nguồn:

- MAE khoảng cách tay–tay cải thiện ít nhất 10% so với MLP chỉ hình học;
- độ chính xác thứ tự chiều sâu đạt ít nhất 80% trên các nhãn không mơ hồ;
- F1 tiếp xúc đạt ít nhất 0,65 tổng thể và ít nhất 0,60 trên các cạnh tay–thân thể ký hiệu;
- trượt tiếp xúc giảm ít nhất 15% so với không có tổn thất duy trì; và
- thêm đồ thị vào tái tạo đóng băng không làm tồi tệ hơn bất kỳ vùng nào quá 1%.

**NO-GO:** giữ lại các token hình học tương đối nhưng tắt năng lượng tiếp xúc. Không ép buộc các nhãn tiếp xúc chất lượng thấp.

### Giai đoạn R3: Khuếch tán không gian bị che

Huấn luyện trên các khung hình riêng lẻ và các đoạn 8 khung hình ngắn từ các nguồn Tầng A/B sạch. Không có bộ thích ứng thời gian ngoài ngữ cảnh ngắn, không có bộ chọn đa giả thuyết.

**GO:** phục hồi đỉnh giải mã vượt quá 30% cho mọi vùng sẵn có dưới mặt nạ tay/ngón tay/cổ tay cố định, với sự suy giảm tập sạch dưới 1% mỗi vùng.

**NO-GO:** sửa lỗi chuẩn hóa trạng thái, ánh xạ điểm số, mặt nạ và tổn thất hình học. Không thêm mô hình hóa thời gian dài.

### Giai đoạn R4: Khuếch tán thời gian-quan hệ tổng quát

Huấn luyện các cửa sổ 64 khung hình trên ARCTIC cộng với Motion-X đã sửa nếu có sẵn. Khởi tạo từ R3 và thêm các khối thời gian/quan hệ hai chiều.

**GO:** trên kiểm định ARCTIC nguyên bản:

- phục hồi vùng 4/8/16 khung hình đạt ít nhất 35% trong mọi vùng sẵn có;
- điều kiện hóa quan hệ cải thiện tập con tương tác được định nghĩa trước ít nhất 5%;
- sự suy giảm vùng sạch dưới 1%;
- lỗi quỹ đạo pháp tuyến lòng bàn tay và đầu ngón tay được cải thiện; và
- các ứng viên được tạo ra là hữu hạn và không bị sụp đổ (non-collapsed).

**NO-GO:** giảm độ phức tạp chuỗi hoặc liên kết quan hệ. Sự khuếch tán chỉ làm giảm độ giật sẽ bị từ chối.

### Giai đoạn R5: Thích ứng miền ký hiệu

Fine-tune trên hỗn hợp cấp bài báo. Đóng băng các prior bộ phận DPoser-X trong 20.000 bước đầu tiên, sau đó tùy chọn mở đóng băng chỉ mô hình toàn thân hợp nhất ở mức `2e-5` nếu kiểm định bên ngoài cải thiện.

Giữ lại ít nhất 25% các batch tay/toàn thân tổng quát sạch để ngăn ngừa sự trôi dạt (drift).

**GO:** trên kiểm định ký hiệu biệt lập nguồn và người ký:

- lỗi vùng bằng nhau cải thiện ít nhất 3% so với bộ khởi tạo quan sát đóng băng;
- không có vùng nào tồi tệ hơn quá 1%;
- tập con bị che/tương tác cải thiện ít nhất 8%;
- các đoạn điều kiện hóa tiếp xúc cải thiện mà không suy giảm không-tiếp-xúc quá 1%; và
- lỗi chuyển tiếp/vận tốc cao không bị suy giảm.

**NO-GO:** giữ R4 làm kết quả nghiên cứu và báo cáo thích ứng mục tiêu giả ký hiệu là tiêu cực. Không tinh chỉnh trên Lane-L.

### Giai đoạn R6: Điều kiện hóa thặng dư tái tạo

Phối hợp 40% cặp thặng dư độc lập thực tế, 30% bùng nổ tổng hợp, 20% quan sát đầy đủ sạch và 10% nhất quán quan sát không nhãn. Huấn luyện chung attention chéo quan sát và điểm số thặng dư.

**GO:** kiểm định bên ngoài đạt các ngưỡng R5 và cải thiện tập khó thêm 2% tương đối so với R5.

**NO-GO:** quay lại R5. Một prior tổng quát không thể sử dụng quan sát không phải là một posterior tái tạo thành công.

### Giai đoạn R7: Bộ chọn bằng chứng K-giả thuyết

Tạo `K = 4` ứng viên từ posterior R6 đóng băng và huấn luyện một bộ chọn riêng biệt. Việc tạo ứng viên bị đóng băng trong khi huấn luyện bộ chọn.

Đầu vào bộ chọn:

- khả năng quan sát 2D/3D được giữ lại;
- lỗi chiếu lại theo trọng số U0;
- điểm số/năng lượng DPoser-X và RDP;
- tính nhất quán quan hệ/tiếp xúc;
- tính nhất quán quỹ đạo đầu ngón tay/lòng bàn tay;
- tính hợp lệ chuyển động và sinh cơ học; và
- sự đa dạng của ứng viên tương đối so với bộ khởi tạo và các ứng viên khác.

Sử dụng tổn thất xếp hạng softmax danh sách (listwise) so với lỗi mục tiêu vùng bằng nhau trên dữ liệu huấn luyện/kiểm định. GT chỉ là nhãn huấn luyện; nó không bao giờ là một đặc trưng suy luận.

**GO:** `K = 4` được chọn:

- cải thiện tập khó ít nhất 2% vượt qua `K = 1`, hoặc thu hẹp ít nhất 25% khoảng cách oracle-`K=4`;
- không làm suy giảm bất kỳ vùng sạch nào quá 0,5%;
- đánh bại ứng viên ngẫu nhiên và lựa chọn năng lượng prior tối thiểu;
- sử dụng độ bao phủ hoàn toàn giống hệt nhau; và
- giữ lại khoảng cách oracle khác 0, chứng minh rằng các ứng viên đa dạng chứ không phải bản sao.

**NO-GO:** phát hành `K = 1`. Không báo cáo lựa chọn oracle như một kết quả phương pháp.

### Giai đoạn R8: Đóng băng và đánh giá cuối cùng

Đóng băng tất cả kiến trúc, hỗn hợp dữ liệu, ngưỡng, bước lấy mẫu, trọng số bộ chọn và quy tắc an toàn trước khi mở các dự đoán Phase 3 Lane-L.

Huấn luyện ba seed cố định: `42`, `123` và `456`. Benchmark cuối cùng sử dụng bộ khởi tạo A1 đã khóa và bản kê khai tác giả 1.493 khung hình bất biến.

Nếu seed 42 vi phạm dứt quát kích thước hiệu ứng hoặc ngưỡng an toàn, dừng các seed khác và đưa ra NO-GO. Nếu không, hoàn thành cả ba seed.

---

## 14. Cấu hình tối ưu hóa ban đầu

### 14.1 Mạng chính

| Mục | Giá trị bắt đầu |
|---|---:|
| optimizer | AdamW |
| tốc độ học module mới | `2e-4` |
| LR module hợp nhất DPoser nếu mở đóng băng | `2e-5` |
| tốc độ học bộ chọn | `1e-4` |
| weight decay | `0.05`, không tính norm/bias/embedding |
| warm-up | 5% tuyến tính |
| lịch trình (schedule) | cosine về 10% LR ban đầu |
| độ chính xác | mạng BF16; phép xoay/SMPL-X/tổng hợp tổn thất ở FP32 |
| physical batch | 4–8 cửa sổ 64 khung hình, được chọn bởi kiểm tra trước bộ nhớ |
| tích lũy độ dốc | đủ cho effective batch 32 |
| xén độ dốc (gradient clipping) | chuẩn toàn cục `1.0` |
| dropout | `0.1` |
| EMA | `0.9999` |
| công nhân/luồng CPU | tối đa 4 |
| kiểm định xác định | bật với mặt nạ/nhiễu/seed ứng viên cố định |
| seed thử nghiệm được chấp nhận | 42, 123, 456 |

### 14.2 Ngân sách huấn luyện

| Giai đoạn | Cập nhật tối đa | Khoảng thời gian kiểm định | Dừng sớm |
|---|---:|---:|---:|
| Bộ thích ứng/chưng cất R1 | 10.000 cho mỗi so sánh | 1.000 | 5 lần kiểm định |
| Đồ thị quan hệ R2 | 50.000 | 2.000 | 8 lần kiểm định |
| Khuếch tán không gian bị che R3 | 75.000 | 2.500 | 8 lần kiểm định |
| Tiền huấn luyện thời gian-quan hệ R4 | 150.000 | 5.000 | 8 lần kiểm định |
| Thích ứng ký hiệu R5 | 100.000 | 2.500 | 10 lần kiểm định |
| Posterior quan sát R6 | 50.000 | 2.500 | 8 lần kiểm định |
| Bộ chọn R7 | 20.000 | 1.000 | 8 lần kiểm định |

Đây là các mức trần. Phân tích 1.000 bước trước mỗi lần chạy dài và ghi lại thông lượng, VRAM đã cấp/dự trữ, kích thước checkpoint, thời gian kiểm định và thời gian thực tế dự kiến. Không dự trữ bộ nhớ GPU chỉ để đạt mục tiêu bộ nhớ.

### 14.3 Lựa chọn checkpoint

Sử dụng điểm số kiểm định bên ngoài vùng bằng nhau đã khai báo trước:

$$
S = \frac{1}{3}\sum_{r\in\{U,L,R\}}
\frac{E_r^{model}}{E_r^{baseline}}
+ 0.5\sum_r\max(0, E_r^{model}/E_r^{baseline}-1.01)
+ 0.25(1-G_{hard}),
$$

trong đó `G_hard` là mức tăng tương đối của tập khó bị cắt gọt. Càng thấp càng tốt. F1 tiếp xúc là một cổng, không phải một trọng số ẩn có thể đánh đổi hình học.

Lưu `last`, `best`, EMA và các checkpoint phục hồi định kỳ cùng với optimizer, scheduler, scaler, RNG, băm bản kê khai, cấu hình đã giải quyết, git SHA và băm checkpoint đã tiền huấn luyện.

---

## 15. Suy luận và lựa chọn hậu phương

### 15.1 RDP-Fast

- `K = 1`;
- dòng xác suất xác định (deterministic probability-flow) hoặc đường dẫn nhiễu cố định tương đương;
- 20 bước giải ban đầu;
- đầu ra trực tiếp, với tinh chỉnh an toàn 5 bước tùy chọn chỉ khi được chấp nhận bên ngoài; và
- hoàn thành ký hiệu trong một lần duyệt khi `T <= 64`.

### 15.2 RDP-Best

- `K = 4` ứng viên posterior độc lập;
- 30 bước giải ban đầu;
- bốn nếp gấp bằng chứng được giữ lại;
- bộ chọn dựa trên bằng chứng;
- tinh chỉnh quan sát 10 bước tùy chọn sau khi chọn; và
- dự phòng bộ khởi tạo theo nhóm sau khi kiểm toán an toàn cuối cùng.

`K = 2` và `K = 8`, 10/20/30/50 bước giải, và hướng dẫn 1.0/1.2/1.5 là các thử nghiệm phân tách kiểm định bên ngoài được khai báo trước. Thiết lập cuối cùng được đóng băng trước Lane-L.

### 15.3 Các chuỗi liên tục dài

Đối với `T > 64`:

- sử dụng cửa sổ 64 khung hình với bước nhảy 32;
- chia sẻ cùng hình dạng đoạn video và chuẩn hóa quan sát;
- tái sử dụng nhiễu cho các khung hình chồng phủ của một ứng viên;
- hòa trộn các phép xoay bằng căn chỉnh bán cầu quaternion/trung bình trắc địa;
- hòa trộn độ tin cậy ứng viên bằng trọng số Hann; và
- xác nhận độ bao phủ khung hình chính xác sau khi hợp nhất.

Các ký hiệu benchmark tác giả biệt lập được xử lý dưới dạng các đoạn hoàn chỉnh.

### 15.4 Ứng viên cơ sở (Baseline)

Luôn bao gồm bộ khởi tạo đóng băng làm ứng viên 0. Điều này tự nó không đảm bảo an toàn; nó cung cấp cho bộ chọn và fail-safe một sự thay thế chưa đổi hợp lệ.

Lựa chọn ứng viên theo mặc định là toàn chuỗi. Thay thế theo nhóm chỉ được phép khi nếp gấp cổ tay/cẳng tay và đồ thị quan hệ vẫn hợp lệ.

---

## 16. Tinh chỉnh cuối cùng tùy chọn

Đầu ra posterior trực tiếp là kết quả Phase 3 chính. Một sự tinh chỉnh ngắn chỉ được chấp nhận sau một thử nghiệm phân tách bên ngoài đóng băng.

Tối ưu hóa tối đa 10 bước Adam trên các độ lệch tư thế cục bộ bị chặn bằng cách sử dụng:

- các quan sát 2D/3D đầy đủ và giữ lại theo trọng số U0;
- tính nhất quán điểm số posterior;
- đính kèm cổ tay/cẳng tay;
- các điều khoản tiếp xúc/quan hệ mềm;
- prior chuyển động mục tiêu từ chuỗi được chọn; và
- an toàn sinh cơ học/không thâm nhập.

Hình dạng, dịch chuyển, định hướng toàn cục, camera, mặt và thân dưới giữ nguyên đóng băng.

**GO:** ít nhất một vùng cải thiện 0,2 mm bên ngoài; không có vùng nào tồi tệ hơn 0,1 mm; dự phòng vẫn dưới 1%; thời gian chạy có thể chấp nhận được.

**NO-GO:** tắt tinh chỉnh và giữ đầu ra RDP trực tiếp. Kết quả T5 Phase 2 thất bại không được tái sử dụng.

---

## 17. An toàn và cơ chế dự phòng (Fallback)

Quay lại bộ khởi tạo A1 cho một nhóm/đoạn khi:

- bất kỳ tham số, điểm số, độ không chắc chắn/độ tin cậy, khớp hoặc đỉnh nào không hữu hạn;
- hiệu chỉnh xoay vượt quá giới hạn thân/tay bị đóng băng bên ngoài;
- chiếu lại được giữ lại đáng tin cậy tồi tệ hơn quá mức dung sai;
- lòng bàn tay, đầu ngón tay, xương hoặc đính kèm cổ tay trở nên không hợp lệ;
- tiếp xúc dự đoán tạo ra sự thâm nhập hoặc trượt cao;
- ứng viên nằm ngoài phạm vi chuẩn hóa huấn luyện;
- hình học, định danh khung hình hoặc băm nguồn bị khác biệt; hoặc
- độ tin cậy của bộ chọn nằm dưới ngưỡng hiệu chỉnh của nó.

Mọi lần dự phòng đều ghi lại vùng, phạm vi khung hình, ứng viên và nguyên nhân. Dự phòng hơn 1% khung hình-nhóm trên kiểm định bên ngoài sạch hoặc Lane-L cuối cùng là NO-GO.

---

## 18. Gói đánh giá

### 18.1 Các phương pháp cơ sở (Baselines)

| ID | Cấu hình | Câu hỏi |
|---|---|---|
| A0 | `method_hamer` gốc | tham chiếu lịch sử |
| A1 | tập hợp đóng băng + HaMeR fallback | bộ khởi tạo mạnh nhất được chấp nhận |
| P2 | đầu ra trực tiếp Phase 2 tốt nhất giữ lại | tinh chỉnh xác định có giúp ích không? |
| R0 | chỉ prior DPoser-X theo khung hình | bao nhiêu phần đến từ prior không gian công khai? |
| R1 | khuếch tán bị che không có khối thời gian/quan hệ | sự hoàn thiện không gian sinh ra có giúp ích không? |
| R2 | khuếch tán thời gian, không có quan hệ/tiếp xúc | posterior toàn chuỗi có giúp ích không? |
| R3 | R2 + đồ thị tương đối, tắt tiếp xúc | các đặc trưng tương đối có giúp ích không? |
| R4 | R3 + tiếp xúc xác suất | tiếp xúc có thêm giá trị một cách an toàn không? |
| R5 | R4, `K = 1` | posterior một đường được chấp nhận |
| R6 | R4, `K = 4` + bộ chọn | suy luận đa giả thuyết có thêm giá trị không? |
| R7 | R6 + tinh chỉnh cuối cùng tùy chọn | tối ưu hóa còn hữu ích không? |

### 18.2 Các chỉ số (Metrics)

Chính:

- TR-V2V kiểu tác giả cho thân trên không tính mặt, tay trái và tay phải;
- độ bao phủ khung hình/vùng chính xác;
- khác biệt theo cặp từng ký hiệu;
- khoảng tin cậy 95% bootstrap theo cụm ký hiệu;
- lỗi ký hiệu trung bình, trung vị và decile tồi nhất; và
- mức tăng tương đối vùng bằng nhau.

Chẩn đoán posterior/quan hệ:

- phục hồi vùng 4/8/16 khung hình bị che;
- lỗi quỹ đạo đầu ngón tay và lỗi trắc địa pháp tuyến lòng bàn tay;
- lỗi đính kèm cổ tay/cẳng tay;
- lỗi khoảng cách tương đối tay–tay và tay–thân thể;
- độ chính xác thứ tự chiều sâu;
- độ chính xác (precision), độ gợi nhớ (recall), F1 và trượt của tiếp xúc;
- sự đa dạng theo cặp ứng viên và khoảng cách oracle;
- độ chính xác top-1 của bộ chọn và tỷ lệ khoảng cách oracle được thu hẹp;
- NLL quan sát được giữ lại / độ bao phủ rủi ro;
- MPJVE, lỗi gia tốc và độ giật (jerk);
- các tập con sạch, mờ, bị che, tương tác và vận tốc cao;
- tỷ lệ dự phòng và nguyên nhân; và
- thời gian chạy, VRAM đỉnh, kích thước checkpoint, số bước và `K`.

### 18.3 Đơn vị thống kê

Bootstrap theo ký hiệu/đoạn. Không bao giờ bootstrap các đỉnh hoặc khung hình riêng lẻ như các quan sát độc lập.

### 18.4 Các tuyến đánh giá (Lanes)

- **Phát triển bên ngoài:** Kiểm định ARCTIC, InterHand, SignAvatars/How2Sign. Tất cả tinh chỉnh diễn ra ở đây.
- **Đa ngôn ngữ/Tổng quát hóa:** Tính nhất quán 2D/định tính PHOENIX và một tập con ngôn ngữ ký hiệu chưa thấy khi có sẵn.
- **Lane-L đã khóa:** 57 ký hiệu tác giả / 1.493 khung hình, chỉ mở sau khi đóng băng.

---

## 19. Các cổng GO/NO-GO tổng thể của Phase 3

| Cổng | Yêu cầu GO | Nếu NO-GO |
|---|---|---|
| P3-G0 hợp đồng/dữ liệu | không rò rỉ; phân chia biệt lập; băm/giấy phép; thất bại thủ công <10% | dừng trước khi huấn luyện |
| P3-G1 prior tiền huấn luyện | bộ thích ứng điểm số đạt tính tương đương; tiền huấn luyện đánh bại ngẫu nhiên ≥5% cùng ngân sách | chưng cất hoặc huấn luyện prior độc lập |
| P3-G2 quan hệ/tiếp xúc | mức tăng MAE quan hệ ≥10%; F1 tiếp xúc ≥0,65 tổng thể/≥0,60 ký hiệu; không có vùng nào suy giảm >1% | tắt tiếp xúc, giữ lại đặc trưng quan hệ an toàn |
| P3-G3 phục hồi không gian bị che | phục hồi ≥30% mọi vùng; suy giảm tập sạch <1% | sửa biểu diễn/dữ liệu; không dùng mô hình thời gian |
| P3-G4 posterior thời gian | phục hồi 4/8/16 khung hình ≥35%; tăng tương tác ≥5%; suy giảm tập sạch <1% | đơn giản hóa mô hình thời gian/quan hệ |
| P3-G5 thích ứng ký hiệu | mức tăng vùng bằng nhau ≥3%; không vùng nào tồi hơn >1%; tập khó ≥8% | giữ kết quả tổng quát; từ chối thích ứng mục tiêu giả |
| P3-G6 posterior quan sát | các ngưỡng R5 cộng với ≥2% mức tăng bổ sung cho tập khó | quay lại prior ký hiệu mà không tinh chỉnh quan sát |
| P3-G7 lựa chọn K | tăng tập khó ≥2% vượt K1 hoặc thu hẹp ≥25% khoảng cách oracle; suy giảm sạch <0,5% | phát hành K1 |
| P3-G8 benchmark tác giả đã khóa | tất cả tiêu chuẩn bên dưới đều vượt qua | Phase 3 NO-GO; không bắt đầu Phase 4 như một sự tiến triển được tuyên bố |

### Yêu cầu khóa chính xác P3-G8

Tương đối với bộ khởi tạo A1 đóng băng:

- chính xác 1.493 khung hình tác giả và quần thể tay trái dự kiến;
- không thiếu hoặc dư khung hình dự đoán nào;
- không có vùng nào suy giảm quá 0,20 mm trong bất kỳ seed nào được chấp nhận;
- ít nhất hai vùng cải thiện với 95% CI bootstrap cụm ký hiệu loại trừ 0 trong mọi seed;
- mức tăng tương đối vùng bằng nhau đạt ít nhất 3% trong mọi seed;
- mức tăng tập con bị che/tương tác/bất đồng được định nghĩa trước đạt ít nhất 8%;
- sự suy giảm độ không chắc chắn thấp tập sạch nằm dưới 1% trong mọi vùng;
- dự phòng nằm dưới 1% khung hình-nhóm;
- ba seed hoàn thành; và
- độ lệch chuẩn vùng xuyên seed dưới 0,20 mm.

Vượt qua P3-G8 cho phép tiến hành công việc âm vận học/pha ở Phase 4. Thất bại nó sẽ giữ lại các artifact A1/RDP tốt nhất làm kết quả tiêu cực và chuyển hướng nỗ lực sang căn chỉnh dữ liệu/quan sát.

---

## 20. Bảng chuyển hướng theo thất bại

| Thất bại | Nguyên nhân gốc có khả năng | Phản ứng yêu cầu |
|---|---|---|
| Bất đồng điểm số DPoser | chuẩn hóa/thứ tự khớp/checkpoint không tương thích | fail closed; sử dụng ánh xạ rõ ràng hoặc chưng cất |
| Khuếch tán cải thiện tính hợp lý nhưng không giảm lỗi | điều kiện hóa quan sát yếu hoặc thiên vị giáo viên | tăng cường bằng chứng độc lập; kiểm tra chất lượng mục tiêu |
| Các ứng viên gần như giống hệt nhau | sụp đổ posterior/hướng dẫn quá mạnh | giảm hướng dẫn, xác minh nhiễu độc lập và mặt nạ |
| Các ứng viên đa dạng nhưng đều xấu | bất đồng prior/miền | cải thiện thích ứng ký hiệu; không mở rộng `K` |
| Oracle K giúp ích nhưng bộ chọn thì không | bằng chứng giữ lại/bộ xếp hạng yếu | cải thiện đặc trưng bộ chọn hoặc phát hành K1 |
| Đồ thị quan hệ giúp tiếp xúc nhưng làm hỏng tay | tiếp xúc giả hoặc liên kết quá mức | tắt năng lượng tiếp xúc; giữ lại các token tương đối |
| Thân cải thiện, tay tồi đi | mất cân bằng tổn thất vùng/dữ liệu tổng quát chiếm ưu thế | cân bằng lại các vùng và lấy mẫu ký hiệu/tay |
| Tay cải thiện, thân tồi đi | nếp gấp cổ tay/cẳng tay hoặc attention xuyên bộ phận | đóng băng thân, giảm cập nhật xuyên bộ phận |
| Quan sát sạch bị trôi dạt | không đủ batch quan sát đầy đủ/mỏ neo | tăng tỷ lệ tập sạch và mỏ neo đáng tin cậy |
| How2Sign cải thiện, ARCTIC/InterHand suy giảm | học quá mức giáo viên giả | giảm trọng số mục tiêu H32; giữ dữ liệu sạch |
| Kiểm định bên ngoài GO, Lane thất bại | bất đồng chuyển giao bộ khởi tạo/miền | báo cáo NO-GO; không bao giờ tinh chỉnh trên Lane |
| Dự phòng cao | bất đồng bộ chọn hoặc an toàn | hiệu chỉnh lại bên ngoài; ưu tiên ứng viên 0/K1 |
| Lấy mẫu chậm không tăng K | quá nhiều bước/giả thuyết | giảm bước giải hoặc phát hành RDP-Fast |

---

## 21. Các thử nghiệm phân tách (Ablations) bắt buộc

### 21.1 Tiền huấn luyện

- nhánh không gian ngẫu nhiên so với điểm số DPoser-X trực tiếp so với chưng cất DPoser-X;
- thân DPoser-X công khai so với bộ thích ứng thân-ký hiệu địa phương;
- có/không có khởi tạo ARCTIC T1 Phase 2 tùy chọn;
- tuyến DPoser-X chính so với FUSION chỉ khi giấy phép cho phép; và
- module hợp nhất DPoser đóng băng so với thích ứng LR thấp ở giai đoạn cuối.

### 21.2 Kiến trúc

- theo khung hình so với thời gian;
- ngữ cảnh 16/32/64 khung hình;
- nhân quả (causal) so với hai chiều (bidirectional);
- không quan hệ so với đặc trưng quan hệ so với quan hệ + tiếp xúc;
- không có token cổ tay/lòng bàn tay;
- từng tay một so với mô hình hóa chung hai tay;
- điểm số tuyệt đối so với điểm số thặng dư xung quanh DPoser-X; và
- 6 so với 8 khối, chỉ thay đổi nếu mô hình bắt đầu bị thiếu khớp (underfit) bên ngoài.

### 21.3 Dữ liệu và biến dạng

- chỉ ARCTIC/InterHand so với thêm How2Sign so với thêm SignAvatars;
- quy mô SignAvatars 10/25/50/100%;
- không có mặt nạ bùng nổ so với mặt nạ 4/8/16;
- không tráo đổi tay/mơ hồ chiều sâu;
- trọng số mục tiêu giả 0.1/0.25/0.5;
- tỷ lệ dữ liệu tổng quát 15/25/40%; và
- U0 so với chẩn đoán U1 miền H32.

### 21.4 Lấy mẫu và lựa chọn

- `K = 1, 2, 4, 8`;
- 10/20/30/50 bước giải;
- hướng dẫn 1.0/1.2/1.5;
- ngẫu nhiên so với năng lượng prior so với bộ chọn bằng chứng so với oracle;
- không giữ lại bằng chứng so với 20% bằng chứng giữ lại;
- ứng viên 0 bị loại trừ/bao gồm; và
- tinh chỉnh cuối cùng tắt/bật.

Mọi thử nghiệm phân tách đều báo cáo cả ba vùng và tập khó. Không chỉ báo cáo F1 tiếp xúc hoặc chỉ oracle-`K`.

---

## 22. Bố cục triển khai bổ sung (Additive)

Tạo một gói mới. Không thêm các nhánh Phase 3 khắp Phase 2 hoặc mã fitting DexAvatar.

```text
phase3_posterior/
  README.md
  config.py
  provenance.py
  configs/
    rdp_r2_relation.yaml
    rdp_r3_spatial_diffusion.yaml
    rdp_r4_temporal_generic.yaml
    rdp_r5_sign_adaptation.yaml
    rdp_r6_observation_posterior.yaml
    rdp_r7_selector.yaml
  data/
    cache_schema.py
    build_phase3_index.py
    build_relation_targets.py
    quality_filter.py
    evidence_split.py
    dataset.py
    corruptions.py
  geometry/
    relation_anchors.py
    contact.py
    state_adapter.py
  models/
    dposer_adapter.py
    relation_graph.py
    contact_head.py
    temporal_score.py
    relational_diffusion.py
    evidence_selector.py
  losses/
    diffusion.py
    geometry.py
    relation.py
    selector.py
  train_relation.py
  train_diffusion.py
  train_selector.py
  sample.py
  infer.py
  render.py
  evaluate.py
  gates.py
  tests/
```

Các tiện ích Phase 2 ổn định có thể được nhập ở dạng chỉ đọc. Nếu một tiện ích chung cần thay đổi hành vi, hãy sao chép hoặc tổng quát hóa nó chỉ sau khi các kiểm thử hồi quy chứng minh tất cả hành vi Phase 2 vẫn giữ nguyên.

### 22.1 Bố cục Artifact

```text
cache/phase3/v1/
  manifest.json
  sources/*.json
  splits/{train,val,calibration,test}.json
  relations/<source>/<clip>.npz
  quality/<source>/<clip>.json

outputs/phase3_training/<experiment>/
  resolved_config.json
  provenance.json
  best.pt
  last.pt
  checkpoints/
  validation.jsonl

outputs/phase3_gates/<gate>/<experiment>/
  decision.json
  summary.json
  per_clip.jsonl
  per_frame.csv
  hashes.sha256

outputs/phase3_rdp_<mode>/
  <sign>/smplifyx/results/*.pkl
  <sign>/smplifyx/meshes/*.obj
  <sign>/phase3_diagnostics/*.json
```

Các thư mục output/cache hiện tại luôn là chỉ đọc.

---

## 23. Hợp đồng lệnh và thực thi

Các công việc dài chạy trong các phiên tmux có tên với nhật ký chỉ-thêm và tối đa bốn luồng CPU.

Các dạng lệnh ví dụ:

```bash
python -m phase3_posterior.data.build_phase3_index \
  --sources phase3_posterior/configs/data_sources_v1.yaml \
  --output cache/phase3/v1

python -m phase3_posterior.train_relation \
  --config phase3_posterior/configs/rdp_r2_relation.yaml

python -m phase3_posterior.train_diffusion \
  --config phase3_posterior/configs/rdp_r5_sign_adaptation.yaml \
  --init outputs/phase3_training/rdp_r4_temporal_generic/best.pt

python -m phase3_posterior.train_selector \
  --config phase3_posterior/configs/rdp_r7_selector.yaml \
  --posterior outputs/phase3_training/rdp_r6_observation_posterior/best.pt

python -m phase3_posterior.infer \
  --config phase3_posterior/configs/rdp_r7_selector.yaml \
  --cache cache/phase2/lane_l_a1_ensemble_v1 \
  --checkpoint outputs/phase3_training/rdp_r6_observation_posterior/best.pt \
  --selector outputs/phase3_training/rdp_r7_selector/best.pt \
  --output outputs/phase3_rdp_best_seed42
```

Mẫu khởi chạy Tmux:

```bash
tmux new-session -d -s phase3_r5_seed42 \
  "cd /home/haipd/DexAvatar && set -o pipefail && \
   OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
   PYTHONUNBUFFERED=1 python -u -m phase3_posterior.train_diffusion \
   --config phase3_posterior/configs/rdp_r5_sign_adaptation.yaml \
   2>&1 | tee -a logs/phase3/rdp_r5_sign_seed42.txt"
```

Mọi lệnh đều ghi lại git SHA, trạng thái worktree dơ (dirty), môi trường, phiên bản CUDA/PyTorch, cấu hình đã giải quyết, bản kê khai nguồn, băm tiền huấn luyện, seed và băm đầu ra.

---

## 24. Các kiểm thử bắt buộc trước khi huấn luyện đầy đủ

### 24.1 Kiểm thử đơn vị (Unit tests)

- vòng lặp trục-góc/ma trận/6D Phase 2 giữ nguyên;
- nhiễu tiến 6D và chiếu `SO(3)` là hữu hạn;
- thứ tự khớp DPoser-X và vòng lặp chuẩn hóa;
- tính tương đương của bộ thích ứng điểm số đóng băng;
- mặt nạ giám sát đặc thù theo nguồn;
- thứ tự nút/cạnh quan hệ và tính thuận tay (handedness);
- hướng pháp tuyến lòng bàn tay dưới phép đối xứng qua gương;
- trễ tiếp xúc (contact hysteresis) và sự trượt;
- tính biệt lập của phân chia điều kiện/bằng chứng;
- khuếch tán bị che không bao giờ coi nhãn thiếu là 0;
- sự độc lập nhiễu ứng viên và phát lại xác định;
- bộ chọn không thể truy cập các trường GT khi suy luận;
- hợp nhất chồng phủ quaternion/trắc địa gần `pi`;
- tính đồng nhất ứng viên 0; và
- hợp đồng PKL/hình học/độ bao phủ của kết quả.

### 24.2 Kiểm thử tích hợp (Integration tests)

- một đoạn sạch dựng lại qua cache -> score -> sample -> PKL -> mesh;
- tắt điểm số thặng dư tái tạo đúng điểm số DPoser đóng băng;
- mặt nạ tay 8 khung hình cố định được phục hồi trên ARCTIC;
- điều kiện hóa quan hệ chỉ thay đổi các cạnh hợp lệ;
- tiếp xúc giả không thể ép buộc tư thế khi xác suất tiếp xúc thấp;
- `K = 4` phát ra bốn ứng viên tái lập khác nhau;
- lựa chọn bằng chứng giữ lại không bao giờ đọc GT giữ lại;
- `K = 1` và ứng viên 0 là các phương án dự phòng hợp lệ;
- huấn luyện BF16 64 khung hình có độ dốc hình học FP32 hữu hạn;
- suy luận lặp lại với một seed/cấu hình có các băm giống hệt nhau; và
- đánh giá nghiêm ngặt từ chối các khung hình thiếu/cũ/trùng lặp.

### 24.3 Kiểm thử Red-team

- không có tay trong toàn bộ đoạn video;
- cả hai tay chồng phủ hoàn toàn;
- tráo đổi định danh trái/phải;
- định向 lòng bàn tay bị lật 180 độ;
- sai thông số nội suy camera;
- băm checkpoint DPoser không hợp lệ;
- thống kê chuẩn hóa bị hỏng;
- khoảng trống khung hình và FPS biến đổi;
- đầu vào chuyên gia bị NaN;
- tất cả logit tiếp xúc đều cao;
- tất cả ứng viên sụp đổ về một tư thế;
- độ tin cậy bộ chọn nằm ngoài phạm vi hiệu chỉnh;
- lưới cũ hoặc tệp kết quả dư thừa; và
- cố gắng dùng đường dẫn SGNify trong bản kê khai huấn luyện/kiểm định.

Trước và sau mỗi thay đổi của Phase 3:

```bash
ruff check phase2_refiner phase3_posterior
pytest -q phase2_refiner/tests phase3_posterior/tests
python -m compileall -q phase2_refiner phase3_posterior
git diff --check
```

Không có triển khai Phase 3 nào được chấp nhận nếu nó làm hỏng kiểm thử Phase 2.

---

## 25. Chiến lược tính toán và lưu trữ

VGA RTX 5880 Ada hiện có là đủ cho mô hình phân tách, nhưng việc sử dụng bộ nhớ phải được đo lường thay vì tối đa hóa.

- bắt đầu với batch 4, 64 khung hình, BF16, activation checkpointing;
- chỉ tăng batch vật lý trong khi giữ lại ít nhất 15% khoảng trống bộ nhớ headroom;
- nhắm tới effective batch 32 thông qua tích lũy;
- giữ luồng CPU bằng hoặc dưới 4;
- chỉ giải mã các mẫu hình học được yêu cầu bởi tổn thất hiện tại, không giải mã mọi trạng thái khuếch tán;
- cache các neo quan hệ tĩnh và chuẩn hóa DPoser;
- gom batch `K` ứng viên theo chiều batch khi bộ nhớ cho phép;
- chỉ giữ lại các checkpoint `best`, `last` và phục hồi đã khai báo trước sau khi chạy được chấp nhận; và
- chỉ lưu các lưới ứng viên lớn cho các tập con cổng và đánh giá cuối cùng.

Chạy thử 1.000 bước thông lượng/bộ nhớ trước mỗi giai đoạn. Một cuộc chạy thử thành công không phải là kết quả độ chính xác.

---

## 26. Giấy phép và cổng khả năng tái lập

Ghi lại riêng giấy phép mã nguồn và trọng số cho DPoser-X. Ghi lại các điều khoản tập dữ liệu cho SignAvatars, How2Sign, ARCTIC, InterHand2.6M, Motion-X, và bất kỳ nguồn tay tùy chọn nào.

WiLoR vẫn là một chuyên gia quan sát đóng băng và phải tuân theo các hạn chế được phát hành của nó. Không sửa đổi hoặc phân phối lại các trọng số WiLoR phái sinh.

FUSION là tùy chọn cho đến khi phép ủy quyền sửa đổi/phân phối lại bằng văn bản được lưu trữ. Checkpoint RDP chính phải duy trì khả năng tái lập mà không cần FUSION.

Các tệp mô hình SMPL-X và MANO không được phân phối lại trừ khi giấy phép của chúng cho phép rõ ràng; hướng dẫn thiết lập nên yêu cầu người dùng lấy chúng từ các nguồn chính thức.

---

## 27. Định nghĩa hoàn thành (Definition of Done)

Phase 3 chỉ hoàn thành khi:

1. trạng thái NO-GO Phase 2 vẫn được tài liệu hóa và không có checkpoint thất bại nào được trình bày như một nền tảng đã xác thực;
2. prior toàn thân DPoser-X công khai được ghi ghim băm và được tích hợp trực tiếp hoặc chưng cất trung thực;
3. mọi nguồn dữ liệu đều có báo cáo giấy phép, phân chia, nguồn gốc và chất lượng;
4. các kiểm tra rò rỉ SGNify/đánh giá vượt qua;
5. đồ thị quan hệ vượt qua các cổng hình học/tiếp xúc hoặc tiếp xúc được tắt một cách rõ ràng;
6. khuếch tán bị che vượt qua các cổng phục hồi 4/8/16 khung hình giải mã và an toàn tập sạch;
7. điều kiện hóa posterior miền ký hiệu cải thiện hình học bên ngoài, không chỉ mượt mà;
8. `K > 1` chỉ được sử dụng nếu bộ chọn bằng chứng vượt qua cổng của nó;
9. mỗi khung hình đầu ra có PKL chuẩn, lưới 10.475 đỉnh và hồ sơ chẩn đoán;
10. thiết lập cuối cùng được chấp nhận được đóng băng trước Lane-L;
11. cả ba seed cuối cùng vượt qua P3-G8, hoặc một NO-GO Phase 3 rõ ràng được đưa ra;
12. tất cả kiểm thử Phase 2 và Phase 3 đều vượt qua; và
13. các cấu hình, checkpoint, bản kê khai, nhật ký, chỉ số và băm được tài liệu hóa.

---

## 28. Thứ tự thực thi ngay lập tức được đề xuất

1. đóng băng và gắn thẻ (tag) nhánh/artifact Phase 2 hiện tại cho khả năng tái sử dụng sau này;
2. tạo khung gói `phase3_posterior/` biệt lập và lệnh kiểm thử hồi quy Phase 2;
3. tải xuống và băm các trọng số thân/tay/toàn thân trộn chính thức của DPoser-X;
4. triển khai và vượt qua cổng tương thích chuẩn hóa/51-khớp DPoser;
5. cụ thể hóa các sidecar quan hệ Phase 3 cho ARCTIC và InterHand;
6. xây dựng tập kiểm toán tiếp xúc/quan hệ bên ngoài 300 đoạn, không tính SGNify;
7. huấn luyện và kiểm duyệt cổng R2 quan hệ/tiếp xúc;
8. huấn luyện R3 khuếch tán không gian bị che và dừng trừ khi phục hồi giải mã vượt qua;
9. huấn luyện R4 khuếch tán thời gian-quan hệ trên các chuỗi tổng quát sạch;
10. thu thập và kiểm toán SignAvatars song song với thử nghiệm, hoặc chỉ định một nguồn SMPL-X ký hiệu tương đương;
11. huấn luyện R5 thích ứng ký hiệu và R6 posterior quan sát;
12. huấn luyện R7 bộ chọn chỉ sau khi posterior tạo ra một khoảng cách oracle hữu ích khác 0;
13. đóng băng tất cả thiết lập và chạy một kiểm tra đã khóa cho seed 42;
14. nếu seed 42 vẫn khả thi, chạy các seed 123 và 456; và
15. bổ sung báo cáo số liệu Phase 3 hoàn chỉnh vào tài liệu này trước khi quyết định Phase 4 có được ủy quyền hay không.

---

## 29. Đề xuất cuối cùng

Cấu hình Phase 3 chính được đề xuất là:

> **Quan sát A1 đóng băng khi đánh giá + độ tin cậy U0 cố định + điểm số không gian toàn thân DPoser-X công khai + bộ thích ứng thời gian-quan hệ hai chiều được huấn luyện độc lập + tiếp xúc xác suất + khuếch tán sub-VP bị che + `K = 4` chọn dựa trên bằng chứng, với ứng viên 0 và dự phòng an toàn theo nhóm.**

Dữ liệu địa phương sẵn sàng hiện tại là đủ để triển khai và kiểm thử lát cắt dọc hoàn chỉnh và chạy một thử nghiệm nghiêm túc. Mô hình cấp bài báo cuối cùng nên chờ SignAvatars đã được kiểm toán hoặc một tập hợp SMPL-X ký hiệu tương đương, vì chỉ với How2Sign H32 sẽ lặp lại điểm yếu giáo viên/miền đã chặn Phase 2.

Nếu quan hệ giúp ích nhưng `K > 1` thì không, hãy phát hành posterior quan hệ `K = 1`. Nếu khuếch tán không đánh bại các phương pháp cơ sở xác định và bộ khởi tạo trên các tập con khó không gian, hãy dừng Phase 3 và không ẩn giấu thất bại sau sự mượt mà thị giác hoặc lựa chọn oracle. Nếu P3-G8 vượt qua, tiến hành sang Phase 4 và thêm âm vận học/pha dưới dạng một đóng góp có thể đo lường riêng biệt.

---

## 30. Báo cáo triển khai và đánh giá độ sẵn sàng (03-08-2026)

### 30.1 Triển khai bổ sung đã giao

Phase 3 được triển khai dưới dạng một gói biệt lập mới, `phase3_posterior/`. Không tệp nguồn DexAvatar fitting, expert, Phase 1, hoặc Phase 2 nào bị sửa đổi. Các hàm cache/rotation/render/evaluation ổn định của Phase 2 được nhập ở dạng chỉ đọc.

Các thành phần được triển khai:

- thừa kế/xác thực YAML fail-closed, đầu ra chỉ-thêm, băm đầy đủ git/môi trường/đầu vào, checkpoint RNG đầy đủ, EMA, cosine warm-up, thực thi BF16, tích lũy độ dốc, xén độ dốc và mức trần 4 công nhân;
- chỉ số đoạn Phase 2 bất biến, kiểm tra phân chia giấy phép/nguồn/người ký, từ chối đường dẫn SGNify/đánh giá tác giả bị cấm, SHA-256 theo từng đoạn, báo cáo Q0/Q1/Q2, sidecar quan hệ và kiểm toán cache P3-G0 chính thức;
- bộ thích ứng ma trận/6D `SO(3)` 51 khớp với các định nghĩa vùng;
- đồ thị thân/tay 32 nút cố định, các cạnh ứng viên cố định, đặc trưng cạnh 16 chiều, phép biến đổi tọa độ thân/cổ tay, trễ tiếp xúc, sự duy trì, thứ tự chiều sâu và đầu tiếp xúc;
- giao diện prior đóng băng rõ ràng và bộ kiểm toán hợp đồng DPoser-X fail-closed; tuyến mặc định được đánh dấu trung thực `from_scratch/no_pretrained_prior`;
- mô hình điểm số thời gian-quan hệ 43.215.691 tham số với tám khối rộng 384, attention không gian và thời gian riêng cho thân/trái/phải, trao đổi xuyên bộ phận chỉ cho cổ tay/nhóm, độ lệch thời gian tương đối học được cắt gọt, điều kiện hóa quan hệ và phép chiếu điểm số thặng dư khởi tạo bằng 0;
- nhiễu sub-VP liên tục tương thích DPoser, khớp điểm số vùng bằng nhau, tổn thất trắc địa/chuyển động bị che, tổn thất focal tiếp xúc/duy trì và chiều sâu, và trọng số mục tiêu đặc thù nguồn;
- hỗn hợp biến dạng chính xác 20/12/12/10/10/10/8/6/6/6, bùng nổ 4/8/16 khung hình, phân chia điều kiện/bằng chứng xác định và 10% classifier-free dropout;
- các CLI huấn luyện R2 relation, R3-R6 diffusion và R7 listwise-selector với cấu hình giai đoạn bổ sung và khởi tạo ràng buộc băm;
- lấy mẫu K-giả thuyết xác định với nhiễu ứng viên độc lập, đồng nhất ứng viên 0, neo quan sát theo trọng số U0, độ bao phủ chuỗi dài chính xác, nhiễu chồng phủ chia sẻ, căn chỉnh bán cầu quaternion và hòa trộn Hann;
- suy luận bộ chọn không dùng GT, xuất PKL kết quả SMPL-X mỏ neo nguồn chuẩn, dự phòng giá trị hữu hạn, dựng hình, đánh giá bản kê khai chung nghiêm ngặt và các hàm quyết định fail-closed từ P3-G0 đến P3-G8; và
- 13 kiểm thử Phase 3 bao phủ các phép xoay, thứ tự/đặc trưng đồ thị, trễ tiếp xúc, từ chối rò rỉ, mặt nạ biến dạng/bằng chứng, tính đồng nhất điểm số-prior, độ dốc, các ứng viên đa dạng xác định, cô lập API bộ chọn và ranh giới cổng.

Các tệp chính bao gồm:

| Khu vực | Các tệp |
|---|---|
| Hợp đồng/nguồn gốc | `config.py`, `provenance.py`, `training.py`, `README.md` |
| Dữ liệu | `data/cache_schema.py`, `data/build_phase3_index.py`, `data/audit_phase3_cache.py`, `data/build_relation_targets.py`, `data/quality_filter.py`, `data/dataset.py`, `data/corruptions.py`, `data/evidence_split.py` |
| Hình học | `geometry/state_adapter.py`, `geometry/relation_anchors.py`, `geometry/contact.py` |
| Mô hình | `models/dposer_adapter.py`, `models/relation_graph.py`, `models/contact_head.py`, `models/temporal_score.py`, `models/relational_diffusion.py`, `models/evidence_selector.py` |
| Mục tiêu/Lấy mẫu | `losses/diffusion.py`, `losses/geometry.py`, `losses/relation.py`, `losses/selector.py`, `sample.py` |
| Thực thi | `train_relation.py`, `train_diffusion.py`, `train_selector.py`, `infer.py`, `render.py`, `evaluate.py`, `gates.py` |
| Điểm khởi đầu đóng băng | `configs/data_sources_v1.yaml`, `configs/rdp_base.yaml`, `configs/rdp_r2_relation.yaml` đến `configs/rdp_r7_selector.yaml` |

### 30.2 Bằng chứng xác minh

Các kiểm tra toàn bộ repo yêu cầu đã vượt qua:

```text
ruff check phase2_refiner phase3_posterior                         PASS
pytest -q phase2_refiner/tests phase3_posterior/tests              69 passed
python -m compileall -q phase2_refiner phase3_posterior            PASS
git diff --check                                                   PASS
```

Tất cả các cấu hình đề xuất đều được giải quyết và xác thực. Mô hình bắt đầu đóng băng chứa 43.215.691 tham số có thể huấn luyện, gần với mức chặn dưới khoảng 45M được đề xuất mà không tính giáo viên DPoser-X đóng băng bên ngoài.

Việc nạp DPoser đã được thử nghiệm cố ý với hợp đồng mẫu đã check-in.
Artifact:
`outputs/phase3_gates/g1/implementation_dposer_contract_audit.json`, SHA-256
`5090c7404f7a039d53a5fe1f70757e5b5038f5cca198fae42e520e6fc474a537`.
Nó trả về chính xác `passed: false`: sub-VP khớp, trong khi trọng số toàn thân 51 khớp chính thức, ánh xạ biểu diễn 6D, băm chuẩn hóa và hồ sơ giấy phép đều vắng mặt. Nó chọn `teacher_distillation_or_from_scratch` và không đại diện sai cho checkpoint ký hiệu chỉ-thân địa phương.

### 30.3 Quyết định đánh giá: GO kỹ thuật, các cổng khoa học vẫn NO-GO/CHỜ ĐỊNH (PENDING)

Triển khai bổ sung là **ĐƯỢC PHÉP để bắt đầu cụ thể hóa dữ liệu R0 và chuẩn bị thử nghiệm bên ngoài**. Vẫn **CHƯA ĐƯỢC PHÉP để khởi chạy chuỗi huấn luyện R2-R8 cấp bài báo đầy đủ**, vì các điều kiện tiên quyết theo thứ tự chưa vượt qua.

| Cổng | Trạng thái hiện tại | Lý do chính xác |
|---|---|---|
| P3-G0 hợp đồng/dữ liệu | **NO-GO** | ID giấy phép nguồn thử nghiệm được đánh dấu rõ ràng `LOCAL_REVIEW_REQUIRED`; danh tính người ký How2Sign vắng mặt trong siêu dữ liệu cache hiện tại; cache Phase 3 chỉ-thêm và kiểm toán thủ công 300 đoạn chưa hoàn thành |
| P3-G1 prior tiền huấn luyện | **NO-GO** | kiểm toán chính thức ở trên: 1/5 kiểm tra tương thích vượt qua; trọng số toàn thân/bộ chuẩn hóa/giấy phép chính thức bị thiếu |
| P3-G2 quan hệ/tiếp xúc | **CHỜ ĐỊNH** | mã đã sẵn sàng, nhưng không có huấn luyện R2 hoặc số liệu định lượng biệt lập nguồn |
| P3-G3 không gian bị che | **CHỜ ĐỊNH** | bị chặn bởi G0/G1/G2 theo thứ tự; không có chỉ số phục hồi giải mã |
| P3-G4 posterior thời gian | **CHỜ ĐỊNH** | bị chặn bởi G3 |
| P3-G5 thích ứng ký hiệu | **CHỜ ĐỊNH** | bị chặn bởi G4 và mục tiêu ký hiệu cấp bài báo |
| P3-G6 posterior quan sát | **CHỜ ĐỊNH** | bị chặn bởi G5 |
| P3-G7 bộ chọn K | **CHỜ ĐỊNH** | bị chặn cho đến khi một posterior đóng băng chứng minh khoảng cách oracle hữu ích khác 0 |
| P3-G8 benchmark tác giả | **ĐÃ KHÓA / CHƯA MỞ** | Lane-L vẫn giữ nguyên cho việc tinh chỉnh và phải đóng cho đến khi G0-G7 vượt qua |

Không có mô hình Phase 3 nào được huấn luyện và không có chỉ số độ chính xác/phục hồi/tiếp xúc nào được tuyên bố trong báo cáo triển khai này. Thành công của kiểm thử chỉ xác minh hợp đồng phần mềm và tính tương thích ngược; nó không phải là một kết quả khoa học GO.

### 30.4 Hành động chính xác tiếp theo

Không nhảy trực tiếp sang huấn luyện. Đầu tiên thay thế các định danh giấy phép giữ chỗ, thêm ID người ký How2Sign có thể kiểm toán hoặc xây dựng một phân chia ký hiệu chứng minh được biệt lập người ký, cài đặt và băm checkpoint/bộ chuẩn hóa toàn thân DPoser-X chính thức, và hoàn thành báo cáo tiếp xúc/mục tiêu thủ công 300 đoạn. Sau đó xây dựng `cache/phase3/v1`, chạy các kiểm toán P3-G0 và P3-G1 chính thức, và dừng lại nếu một trong hai vẫn là NO-GO. Chỉ sau khi cả hai vượt qua mới khởi chạy R2 trong tmux với đầu ra `logs/phase3/` chỉ-thêm và mức trần 4 luồng CPU.

---

## 31. Giải quyết điểm chặn R0 và báo cáo khởi chạy R2 (03-08-2026)

Mục này thay thế trạng thái R0 trong Mục 30.3. Công việc vẫn mang tính bổ sung: không có cache Phase 1/Phase 2, phương pháp cũ, đầu vào đánh giá của tác giả, hoặc artifact Lane-L nào bị sửa đổi hoặc đọc để tinh chỉnh.

### 31.1 Các sửa lỗi chính xác

- Bộ lấy mẫu xác định giờ đây tích hợp chính xác ODE dòng xác suất sub-VP, bao gồm chiết khấu sub-VP và hệ số điểm số một phần hai yêu cầu.
- Mỏ neo quan sát được điều chỉnh tỷ lệ theo bước tích hợp tuyệt đối, vì vậy việc thay đổi số bước lấy mẫu không còn làm thay đổi sức mạnh mỏ neo theo cấu trúc.
- Mặt nạ điều kiện hóa 51 khớp được ánh xạ tới 32 nút quan hệ cố định. Mọi cạnh có điểm đầu cuối bị ẩn đều bị hủy hợp lệ và đưa về 0 trước khi điều kiện hóa đồ thị, và các cạnh bị ẩn bị loại trừ khỏi giám sát tiếp xúc/chiều sâu.
- Đồ thị quan hệ hiện có một hợp đồng bộ tích lũy độ chính xác hỗn hợp rõ ràng. Một kiểm thử hồi quy autocast CUDA BF16 bao phủ thất bại tìm thấy khi khởi chạy.
- Bộ thích ứng chỉ-đọc Phase 3 khởi tạo tất cả các thuộc tính tập dữ liệu Phase 2 hiện tại; một kiểm thử nhúng phân mục chỉ số thực đã vượt qua trước khi khởi chạy lại.

### 31.2 Dữ liệu, người ký và kết quả hình học

Danh tính người ký How2Sign được trích xuất từ trường người ký cuối cùng trong tên đoạn chính thức. Vì các bản kê khai huấn luyện/kiểm định/hiệu chỉnh chính thức trước đó chia sẻ người ký, Phase 3 sử dụng các phân chia thành phần người ký mà không làm thay đổi Phase 2:

| Phân chia Phase 3 | ID người ký | Các đoạn |
|---|---:|---:|
| huấn luyện | 3, 5, 8 | 10.643 |
| kiểm định | 1, 2 | 754 |
| hiệu chỉnh | 4, 9, 11 | 420 |

Phân chia có 2.242 nhóm video nguồn, không chồng phủ nhóm nguồn và không chồng phủ người ký. Cache chỉ-thêm cuối cùng chứa 14.142 đoạn huấn luyện, 1.312 đoạn kiểm định và 420 đoạn hiệu chỉnh (tổng cộng 15.874).

Đầu vào và nhãn quan hệ sử dụng các nhà cung cấp riêng biệt. Tư thế khởi tạo và mục tiêu ARCTIC và How2Sign được giải mã độc lập thông qua mô hình SMPL-X trung tính đóng băng. InterHand sử dụng các khớp 3D tọa độ thế giới chính thức, giữ nguyên vị trí hai tay thực tế thay vì đặt các phép xoay MANO trên một thân SMPL-X trung tính. Tiếp xúc sử dụng khoảng cách tâm giải mã trừ đi bán kính đại diện giải phẫu cố định được ghi nhận; các bán kính này chỉ dán nhãn tiếp xúc ứng viên và không phải là mục tiêu tái tạo.

| Nguồn | Độ bao phủ nút/keypoint | Độ bao phủ thân | Độ bao phủ cổ tay/tay | Độ bao phủ cạnh | Nhãn tiếp xúc hợp lệ | Tỷ lệ tiếp xúc dương |
|---|---:|---:|---:|---:|---:|---:|
| ARCTIC | 100,00% | 100,00% | 100,00% | 100,00% | 95,06% | 0,4892% |
| InterHand2.6M | 75,00% | 20,00% | 100,00% | 23,46% | 20,99% | 7,1087% |
| How2Sign | 100,00% | 100,00% | 100,00% | 100,00% | 95,06% | 1,5358% |

Độ bao phủ thân của InterHand cố ý giới hạn ở hai cổ tay được quan sát; các chú thích chính thức của nó không chứa thân, và không có mục tiêu thân nào được tạo ra giả mạo.

### 31.3 Kiểm toán thị giác và P3-G0 chính thức

Một cuộc kiểm toán 300 đoạn xác định đã lấy mẫu 100 đoạn cho mỗi nguồn và kiểm tra bốn khung hình mỗi đoạn trong cả chiếu XY và XZ. Người kiểm duyệt là Codex, được chủ sở hữu dự án ủy quyền. Mọi hình ảnh bằng chứng và sidecar quan hệ đều được ràng buộc băm trong `cache/phase3/v1/manual_quality_300.json`.

- đoạn đã kiểm duyệt: **300**;
- thất bại thảm họa: **0**;
- tỷ lệ thất bại: **0.0000**, yêu cầu `< 0.10`;
- băm bằng chứng bị thiếu/thay đổi: **0**.

Lần chạy chính thức đầu tiên được lưu giữ là `outputs/phase3_gates/g0/r0_cache_audit_failed_float_boundary.json`. Điểm chặn duy nhất của nó là `0.1999999999999958 < 0.20` do tích lũy số thực dấu phẩy động. Nhập so sánh được sửa bằng độ dung sai `1e-9` mà không đổi yêu cầu 0.20. Kiểm toán lặp lại báo cáo:

| Kiểm tra P3-G0 | Kết quả |
|---|---|
| sidecar quan hệ hoàn chỉnh và các băm | GO |
| biệt lập nguồn/người ký/video | GO |
| quét nguồn tác giả/SGNify bị cấm | GO |
| bằng chứng giấy phép đã ghi | GO |
| hợp đồng độ bao phủ quan hệ | GO |
| xem xét 300 đoạn và thất bại `<10%` | GO (0,00%) |
| số điểm chặn | **0** |

Quyết định chính thức: **P3-G0 GO**.

### 31.4 Tuyến R1 và phụ thuộc artifact R7

Hợp đồng DPoser-X bên ngoài vẫn không sẵn có, vì vậy P3-G1 không được dán nhãn lại thành GO prior tiền huấn luyện. Tuyến dự phòng `from_scratch` đã khai báo được sử dụng và không có tuyên bố prior tiền huấn luyện nào được đưa ra. Tiền huấn luyện quan hệ R2 không tiêu thụ checkpoint DPoser.

`outputs/phase3_training/rdp_r6_observation_seed42/selector_train.npz` không thể được tạo trước R6: các hàng bằng chứng của nó phải được lấy mẫu từ posterior R6 đóng băng, và việc tạo thư mục đầu ra R6 sớm cũng sẽ vi phạm hợp đồng chạy chỉ-thêm. `phase3_posterior.data.build_selector_features` giờ đã được triển khai để tạo artifact sau R6. Một giữ chỗ hậu phương ngẫu nhiên hoặc tổng hợp đã không được bịa đặt và không phải là điểm chặn R0.

### 31.5 Khởi chạy huấn luyện R2

Giáo trình R2 đóng băng là 30.000 bước warm-up tổng quát trên ARCTIC + InterHand, tiếp theo là 20.000 bước thích ứng chung. Giai đoạn kết hợp giữ lại khoảng 25% các đoạn tổng quát. Batch vật lý là 8, tích lũy độ dốc là 8, tốc độ học AdamW là `2e-4`, EMA là `0.9999` và một `last.pt` nguyên tử được ghi mỗi 1.000 bước optimizer.

Ba lần khởi chạy fail-closed đã dừng trước bước optimizer đầu tiên: một lần lộ thuộc tính bộ thích ứng Phase 2 cũ và hai lần kiểm tra CUDA liên tiếp đã lộ và giải quyết hoàn toàn sự bất đồng kiểu bộ tích lũy BF16. Các thư mục đầu ra và nhật ký của chúng được lưu giữ với tên `failed_*` rõ ràng. Sau các sửa lỗi và bộ kiểm thử hồi quy hoàn chỉnh, lần khởi chạy được chấp nhận là:

- phiên tmux: `phase3_r2_relation`;
- nhật ký: `logs/phase3/rdp_r2_relation_seed42_v4.txt`;
- đầu ra: `outputs/phase3_training/rdp_r2_relation_seed42`;
- giai đoạn ban đầu: `generic_warmup`;
- tổn thất tổng thể/tiếp xúc/duy trì/chiều sâu bước 1: **0.590228 / 0.124783 / 0.157754 / 1.007812**;
- tổn thất tổng thể/tiếp xúc/duy trì/chiều sâu bước 200: **0.179023 / 0.013691 / 0.017824 / 0.396484**;
- cấp phát GPU Phase 3 quan sát được: khoảng **680 MiB**;
- sử dụng CPU Phase 3 khi khởi chạy: khoảng **187%**, dưới mức trần 500%.

Kiểm định sau các sửa lỗi cuối cùng:

```text
ruff check phase2_refiner phase3_posterior                    PASS
pytest -q phase2_refiner/tests phase3_posterior/tests         85 passed
python -m compileall -q phase2_refiner phase3_posterior       PASS
git diff --check                                               PASS
```

### 31.6 Các băm artifact chính

| Artifact | SHA-256 |
|---|---|
| `cache/phase3/v1/manifest.json` | `fa71eb2f82b49689c1c62d611e9b4edac05b84ceeda9b04469790eaec4196581` |
| `cache/phase3/how2sign_signer_splits_v1/report.json` | `60545a8188875c2e73a4d55115d1b7f60437b8833fed0b118e114208bec30782` |
| `cache/phase3/v1/manual_quality_300.json` | `aca58ce9d9e0219eb4f08df1730fa4b3f91913cd99bffb26c120134173ebfb30` |
| `outputs/phase3_gates/g0/r0_cache_audit.json` | `f8a05cd439e79914aff7569ede9152861fb010bb8c109bcf2b55730b8a403df5` |
| `outputs/phase3_gates/g0/decision.json` | `f56784b35c6ad01274598bddcd802f4ef9a91a4e28eb5a8168b172ae56ce6682` |
| Bằng chứng giấy phép dữ liệu Phase 3 | `0bf83eb4b9d7c4d5dfd661211da1da2fda5deb249128d4764a6c5bc58163239f` |
| Cấu hình R2 | `10f022a1c23a624442b886ea0a1b31471a12a84a8ab15d4a7884ef18f7d9ee07` |

P3-G2 và các cổng độ chính xác sau đó vẫn là **CHỜ ĐỊNH**. Tối ưu hóa bước 1 và P3-G0 GO là các kết quả độ sẵn sàng, không phải tuyên bố rằng độ chính xác quan hệ/tiếp xúc hoặc Phase 3 đầy đủ đã vượt qua.

---

## 32. Hoàn thành R2 và kết quả P3-G2 fail-closed (04-08-2026)

### 32.1 Artifact huấn luyện đã hoàn thành

Lần chạy quan hệ/tiếp xúc R2 đã hoàn thành ngân sách 50.000 cập nhật đóng băng và thoát bình thường. Nó đã sử dụng warm-up 30.000 bước đóng băng ARCTIC+InterHand tiếp theo là 20.000 bước thích ứng chung; nó không sử dụng Lane-L. Trọng số có thể triển khai là trạng thái EMA của checkpoint, phù hợp với `phase3_posterior.training.load_weights`.

| Mục | Giá trị |
|---|---|
| checkpoint | `outputs/phase3_training/rdp_r2_relation_seed42/best.pt` |
| cập nhật hoàn thành | 50.000 |
| tổn thất huấn luyện cuối cùng được ghi | 0.049974 |
| tổn thất tiếp xúc / duy trì / chiều sâu cuối | 0.004957 / 0.003900 / 0.108398 |
| SHA-256 checkpoint | `03bb7bff28a27c44c7745117f8a22943b46f728c3f5eb869ae0f9325b50a10b4` |
| SHA-256 cấu hình đóng băng | `10f022a1c23a624442b886ea0a1b31471a12a84a8ab15d4a7884ef18f7d9ee07` |

### 32.2 Phép đo kiểm định biệt lập nguồn

Trạng thái EMA đã được đánh giá xác định trên tất cả 1.312 đoạn trong bản kê khai kiểm định biệt lập nguồn/người ký bất biến (511 ARCTIC, 47 InterHand2.6M và 754 How2Sign), với cửa sổ 32 khung hình và ngưỡng tiếp xúc 0,5 đã khai báo trước. Đây là sự đánh giá các sidecar quan hệ được dẫn xuất từ hình học, không phải là một benchmark lỗi lưới cuối cùng.

| Tập con kiểm định | Độ chính xác tiếp xúc | Độ gợi nhớ tiếp xúc | F1 tiếp xúc | Độ chính xác thứ tự chiều sâu không mơ hồ |
|---|---:|---:|---:|---:|
| tất cả các cạnh tiếp xúc hợp lệ | 0.6373 | 0.7599 | **0.6932** | **0.9873** |
| ARCTIC | 0.5532 | 0.9251 | 0.6924 | 0.9973 |
| InterHand2.6M | 0.9746 | 0.6738 | 0.7967 | 0.9924 |
| How2Sign, tất cả cạnh hợp lệ | 0.6442 | 0.7448 | 0.6908 | 0.9805 |
| các cạnh tay--thân thể How2Sign | 0.5372 | 0.4469 | **0.4879** | n/a |

Số lượng ma trận nhầm lẫn tiếp xúc trên tất cả các cạnh là TP=30.201, FP=17.189 và FN=9.541. Đối với tập con tay--thân thể How2Sign bắt buộc, chúng là TP=101, FP=87 và FN=125.

### 32.3 Quyết định chính thức: P3-G2 NO-GO

**NO-GO.** Điều kiện F1 tiếp xúc tổng thể vượt qua (0.6932 >= 0.65), và điều kiện thứ tự chiều sâu đề xuất vượt qua (98.73% >= 80%). Tuy nhiên, F1 tiếp xúc tay--thân thể ký hiệu bắt buộc là 0.4879, dưới yêu cầu 0.60 một khoảng 0.1121.

Ngoài ra, lần triển khai R2 đầu tiên này không huấn luyện/đánh giá bộ so sánh MLP chỉ-hình học yêu cầu, MAE khoảng cách quan hệ, phân tách không-duy-trì/trượt tiếp xúc, hoặc phép đo an toàn vùng tái tạo đóng băng. Nó cũng lưu trạng thái cuối cùng làm `best.pt` thay vì chọn nó so với điểm kiểm định. Những phép đo bị thiếu đó phải được triển khai trước khi một quyết định P3-G2 chính thức có thể đọc được bằng máy có thể được phát ra; chúng không được đại diện là đã vượt qua. Theo chính sách Giai đoạn R2, không bắt đầu R3 từ checkpoint này. Giữ lại các đặc trưng hình học tương đối an toàn của nó cho một thử nghiệm R2 đã sửa chữa, và giữ năng lượng tiếp xúc bị tắt cho đến khi cổng tiếp xúc ký hiệu và tất cả phép đo so sánh/an toàn yêu cầu vượt qua.

---

## 33. Thực thi R2 v2 đã sửa chữa (04-08-2026)

### 33.1 Nguyên nhân gốc và các sửa lỗi bổ sung

Lần chạy R2 đầu tiên không thể sửa bằng cách tinh chỉnh ngưỡng. Triển khai của nó thiếu các đặc trưng quan hệ mục tiêu liên tục, để kênh khoảng cách bề mặt ở mức 0, làm pha giãng các điểm dương tay--thân thể ký hiệu cực kỳ thưa thớt, và không triển khai bộ so sánh chỉ-hình học của đề xuất, phân tách duy trì, kiểm định định kỳ hoặc lựa chọn checkpoint tốt nhất. Cache, checkpoint và cấu hình v1 ban đầu vẫn được lưu giữ.

Tuyến đã sửa chữa bổ sung giới thiệu:

- schema quan hệ v2 với các đặc trưng cạnh mục tiêu độc lập và bộ khởi tạo đã giải mã;
- đặc trưng khoảng cách bề mặt giải phẫu cố định khác 0 và một thặng dư khoảng cách học được;
- hỗn hợp chung `55% How2Sign / 30% ARCTIC / 15% InterHand` đóng băng;
- phân tầng đoạn tiếp xúc dương 35% bên trong phân bổ How2Sign, được biện minh bởi tỷ lệ đoạn dương tự nhiên đo được khoảng 9,3%;
- cân bằng dương focal và trọng số cạnh tay--thân thể ký hiệu cố định;
- một MLP chỉ-hình học được huấn luyện ở cùng ngân sách cập nhật;
- một thử nghiệm phân tách đồ thị không-duy-trì được khởi tạo giống hệt nhau;
- kiểm định xác định mỗi 2.000 cập nhật và lựa chọn checkpoint tốt nhất EMA;
- kiểm tra cổng chiều sâu, tính sẵn có của trượt, an toàn chỉ-quan-hệ và mức tăng khoảng cách rõ ràng; và
- một lượt chạy thử hai bước end-to-end đã thực thi tối ưu hóa, kiểm định và tuần tự hóa checkpoint trước lượt chạy dài.

Không có dữ liệu Lane-L nào được mở hoặc sử dụng cho những thay đổi này.

### 33.2 Kiểm toán cache đã sửa chữa

`cache/phase3/r2_relation_targets_v2` chứa 15.874 đoạn: 14.142 huấn luyện, 1.312 kiểm định và 420 hiệu chỉnh. Kiểm toán fail-closed tại `outputs/phase3_gates/g0/r2_relation_targets_v2_audit.json` báo cáo:

| Kiểm tra | Kết quả |
|---|---:|
| số điểm chặn | **0** |
| các đoạn với schema quan hệ v2 | 15.874 / 15.874 |
| nhà cung cấp mục tiêu độc lập How2Sign | 11.817 / 11.817 |
| các giá trị mục tiêu liên tục hữu hạn | 745.950.384 |
| khung hình-cạnh hợp lệ / dương tay--thân thể How2Sign | 2.757 / 22.688.640 |
| tỷ lệ dương tay--thân thể How2Sign | 0.01215% |

Các băm chính:

| Artifact | SHA-256 |
|---|---|
| bản kê khai cache đã sửa | `b166eae34f4d68528d4b30a578afc28cc1f5a211374096112bfc87f27843cfca` |
| kiểm toán cache đã sửa | `adbcbef71167807e83e01f3767f5561eab17e4356304680065fe5a1d35fef165` |
| cấu hình v2b được chấp nhận | `4609e27c9330b6ec0c4492211f212c21cb30a25e7f3e627b970d95deb6e7a9f4` |

### 33.3 Lượt chạy huấn luyện được chấp nhận và kiểm định sớm

Lần khởi chạy 100 bước ban đầu được lưu giữ là `rdp_r2_relation_corrected_v2_seed42_superseded_effective_batch64` sau khi phân tích tiết lộ rằng tích lũy kế thừa tạo ra effective batch 64 chứ không phải 32 theo yêu cầu của đề xuất. Điều này đã được sửa trước checkpoint đầu tiên, không cần quan sát Lane-L hay thay đổi một cổng số học.

Lượt chạy được chấp nhận là:

- tmux: `phase3_r2_relation_v2b`;
- nhật ký: `logs/phase3/rdp_r2_relation_corrected_v2b_seed42.txt`;
- đầu ra: `outputs/phase3_training/rdp_r2_relation_corrected_v2b_seed42`;
- physical batch / tích lũy / effective batch: `8 / 4 / 32`;
- giáo trình: 30.000 cập nhật warm-up tổng quát, sau đó 20.000 cập nhật thích ứng chung;
- sử dụng CPU: khoảng 200%, dưới mức trần 500%;
- baseline thử nghiệm: **87 passed**, với lint và biên dịch Phase 3 vượt qua.

Ở cập nhật 2.000, khi vẫn nằm trong warm-up chỉ-tổng-quát, kiểm định bên ngoài đầu tiên báo cáo độ chính xác thứ tự chiều sâu 85,52%, F1 tiếp xúc 0,0, F1 tay--thân thể ký hiệu 0,0, mức tăng MAE quan hệ -64,81%, và so sánh trượt không sẵn có vì đồ thị EMA không có tiếp xúc dương thực sự. Đây là một **ảnh chụp ảnh NO-GO sớm** dự kiến, không phải quyết định cuối cùng. Giáo trình dương ký hiệu không bắt đầu cho đến cập nhật 30.001. Lượt chạy tiếp tục trong tmux; P3-G2 vẫn **CHỜ ĐỊNH/NO-GO cho đến khi một checkpoint hoàn chỉnh vượt qua mọi điều kiện chính thức**.

---

## 34. Hoàn thành R2 v2b đã sửa và quyết định P3-G2 chính thức (04-08-2026)

Lượt chạy 50.000 cập nhật đã sửa đổi hoàn thành bình thường. Quy tắc lựa chọn khai báo trước đã chọn checkpoint EMA ở cập nhật 36.000 thay vì cập nhật cuối cùng. Đánh giá sử dụng bản kê khai kiểm định biệt lập nguồn/người ký v2 bất biến và ngưỡng tiếp xúc 0,5 cố định; Lane-L không được mở.

| Điều kiện P3-G2 | Kết quả checkpoint tốt nhất | Yêu cầu | Quyết định |
|---|---:|---:|---|
| mức tăng MAE khoảng cách quan hệ so với MLP chỉ-hình-học | **15.61%** | >=10% | GO |
| F1 tiếp xúc tổng thể | **0.7049** | >=0.65 | GO |
| F1 tiếp xúc tay--thân thể How2Sign | **0.4667** | >=0.60 | **NO-GO** |
| độ chính xác thứ tự chiều sâu | **98.43%** | >=80% | GO |
| mức giảm trượt so với phân tách không-duy-trì | **0.72%** | >=15% | **NO-GO** |
| suy giảm tái tạo chỉ-quan-hệ | **0.00%** | <=1% | GO |
| bộ so sánh trượt sẵn có | true | true | GO |

Số lượng ma trận nhầm lẫn tiếp xúc của đồ thị được chọn là TP=29.392, FP=14.254 và FN=10.350 tổng thể. Trên tập con tay--thân thể How2Sign yêu cầu, chúng là TP=91, FP=73 và FN=135. Thất bại chủ yếu là do độ gợi nhớ tiếp xúc ký hiệu không đủ, và sự duy trì làm thay đổi độ trượt tiếp xúc dự đoán chỉ 0,72%, xa mức yêu cầu 15%.

**Quyết định P3-G2 chính thức: NO-GO.** Đồ thị quan hệ đã sửa đổi được giữ lại như một thành phần hình học tương đối hữu ích vì nó vượt qua các điều kiện khoảng cách và tiếp xúc tổng thể. Năng lượng tiếp xúc bị tắt cho các giai đoạn posterior sau: bằng chứng tiếp xúc ký hiệu và duy trì không biện minh cho việc bật nó. Không bắt đầu R3--R8 dưới dạng tiến triển Phase 3 được tuyên bố từ checkpoint này.

| Artifact | SHA-256 |
|---|---|
| `best.pt` được chọn (bước 36.000) | `69852e0d88a166bff65326ae22eb6359aada658458384b5f3c879709131d33b9` |
| `last.pt` cuối cùng (bước 50.000) | `5ff34cf359795e0918a85c490f0d6993ae0f1cfd2fb116e78af33130c56affaa` |
| đánh giá biệt lập nguồn | `0854bf70b8c345f840f6219e7e096b14af9da1437cecfefb99fbdcb4783600cb` |
| quyết định G2 chính thức | `5cceb65aca11d026d057d4d172dd435a81de71edc9aac7840d35ab6b986c18e8` |

---

## 35. Tiến triển R3 chỉ-hình-học Option-A (04-08-2026)

### 35.1 Quyết định dự phòng đóng băng

P3-G2 vẫn giữ nguyên **NO-GO**. Các điều kiện thất bại không đổi: F1 tiếp xúc tay--thân thể How2Sign là 0.4667 so với yêu cầu 0.60, và độ trượt tiếp xúc dự đoán cải thiện 0.72% so với 15% yêu cầu. Checkpoint cập nhật 36.000 được chọn chỉ được sử dụng như một bộ trích xuất đặc trưng chiều sâu/hình học tương đối đóng băng. Dự đoán tiếp xúc và duy trì của nó không được chấp nhận phía hạ nguồn.

ID đường ống chẩn đoán là `R2_geometry_only_R3_progression`. Cấu hình dự phòng chia sẻ của nó bắt buộc tất cả những điều sau cho các thế hệ con R3--R8:

```yaml
fallback:
  mode: geometry_only
  contact_energy_enabled: false
  force_coupling_enabled: false
  persistence_constraints_enabled: false
model:
  contact_energy_enabled: false
  freeze_relation_backbone: true
loss:
  contact: 0.0
  persistence: 0.0
```

Mô hình loại bỏ các logit tiếp xúc và duy trì khỏi hợp đồng đầu ra dự phòng của nó, trong khi giữ lại các dự đoán `relation_token`, các token cạnh, khoảng cách và thứ tự chiều sâu. Điều này khiến cho việc vô tình sử dụng năng lượng tiếp xúc hoặc sự duy trì sẽ fail closed thay vì chỉ phụ thuộc vào trọng số vô hướng bằng 0. Bộ lấy mẫu không chứa bước liên kết lực (force-coupling) hay hút tiếp xúc (contact-attraction).

### 35.2 Dữ liệu R3 và khởi tạo

R3 chỉ sử dụng các nguồn Tầng A/B sạch. Bản kê khai bổ sung `cache/phase3/r3_geometry_only_tier_ab_v1` chứa 3.499 đoạn huấn luyện và 558 đoạn kiểm định từ ARCTIC và InterHand2.6M. Danh tính nguồn, người ký và nhóm nguồn giữ nguyên biệt lập. Mục tiêu giả Tầng C How2Sign bị loại trừ khỏi huấn luyện không gian bị che và Lane-L vẫn chưa mở.

Khởi tạo nghiêm ngặt tải tất cả 34 tensor khung xương quan hệ từ `outputs/phase3_training/rdp_r2_relation_corrected_v2b_seed42/best.pt`. Lượt chạy thử GPU hai bước xác nhận tất cả 34 tensor giữ nguyên chính xác bit sau khi tối ưu hóa R3.

Lần khởi chạy đầu tiên làm lộ một tổn thất chuyển động phụ trợ nhiễu cao không trọng số trước khi checkpoint được ghi. Nó được lưu giữ là `rdp_r3_spatial_geometry_only_seed42_superseded_unweighted_aux`. Triển khai đã được sửa để áp dụng trọng số hình học phụ trợ cắt gọt SNR của đề xuất với `gamma=5`; tổn thất thử nghiệm đã sửa giảm từ 100,39 xuống 20,02. Một bản thử nghiệm batch-4 đã sửa bắt đầu với tổn thất tổng thể hữu hạn 2.3401, nhưng phân tích cho thấy thông lượng thấp không cần thiết. Nó đã được dừng trước checkpoint đầu tiên và lưu giữ là `rdp_r3_spatial_geometry_only_v2_seed42_superseded_batch4`. Lượt chạy được chấp nhận chỉ thay đổi sự phân tách batch vật lý/tích lũy từ 4/8 sang 8/4, giữ nguyên effective batch 32 và mọi thiết lập optimizer, loss, model, data và gate. Lượt chạy thử batch-8 hoàn thành với tổn thất hữu hạn 1.4247, và lượt chạy dài được chấp nhận bắt đầu với tổn thất hữu hạn 4.5623.

### 35.3 Kiểm tra trước, khởi chạy và các băm

Kiểm tra trước đã sửa báo cáo **GO với 0 điểm chặn** cho hợp đồng thực thi dự phòng. Baseline hồi quy là **90 passed**, với lint, biên dịch và kiểm tra khoảng trắng được phạm vi Phase 3 vượt qua.

| Mục | Giá trị |
|---|---|
| tmux | `phase3_r3_geometry_only_v3` |
| nhật ký | `logs/phase3/rdp_r3_spatial_geometry_only_v3_seed42.txt` |
| đầu ra | `outputs/phase3_training/rdp_r3_spatial_geometry_only_v3_seed42` |
| cập nhật tối đa | 75.000 |
| physical / accumulated / effective batch | 8 / 4 / 32 |
| SHA-256 checkpoint quan hệ | `69852e0d88a166bff65326ae22eb6359aada658458384b5f3c879709131d33b9` |
| SHA-256 cấu hình cơ sở dự phòng | `afb3dd0ab17f4f11dbf5daa64d56c3febb9c884bf427fd408e6f4d0a946a43f2` |
| SHA-256 cấu hình R3 | `e1d3a61efe955e004b70f1d7dfcad1ca2ff672f6e0938bd8fee3b44e96d43feb` |
| SHA-256 bản kê khai Tầng A/B | `926935184b46c5a922e52bce05d5b6d7170013035a4f693a0681aa171d886a0d` |
| SHA-256 kiểm tra trước được chấp nhận | `0bc6fde6f94d54da4efecf890cab65f2a90f3e9522d89684c32354bc526bd7df` |

Huấn luyện R3 là một tiến triển dự phòng chẩn đoán được ủy quyền, không phải là sự đảo ngược quyết định P3-G2. P3-G3 vẫn chờ định cho đến khi đánh giá an toàn tập sạch và phục hồi bị che giải mã hoàn thành.

Tiến trình tmux được chấp nhận đã xác nhận sống qua cập nhật 200. Tại ảnh chụp đó, tổn thất score đã giảm từ 1.0064 ở cập nhật 1 xuống 0.5053; năng lượng tiếp xúc là false và khung xương quan hệ bị đóng băng trong mọi bản ghi. Tiến trình sử dụng khoảng 1.5 GiB VRAM và khoảng 150% CPU tổng hợp, dưới mức trần 500% CPU. Các checkpoint phục hồi nguyên tử định kỳ bắt đầu từ cập nhật 1.000.

### 35.4 Tồn đọng huấn luyện lại R2 tương lai riêng biệt

Một lượt chạy R2 tập trung vào tiếp xúc tương lai được giữ riêng biệt khỏi dự phòng đóng băng này. Trước lượt chạy đó, hãy thu thập hoặc đánh giá kép các nhãn tiếp xúc tay--thân thể ký hiệu, hard-mine các mẫu âm tính giả và mẫu âm tính gần tiếp xúc gây nhầm lẫn, sử dụng giáo trình tiếp xúc-dương đã khai báo trước, và huấn luyện một đầu thời gian đặc thù cho duy trì so với thử nghiệm phân tách không-duy-trì cùng vị trí. Nó phải lặp lại quyết định P3-G2 biệt lập nguồn hoàn chỉnh; không kết quả R3 nào có thể dán nhãn lại hồi tố cho NO-GO tiếp xúc R2 hiện tại.

---

## 36. Sửa đổi đánh giá chính thức R3 và huấn luyện lại điều kiện v4b (05-08-2026)

### 36.1 Kết quả v3 đã hoàn thành và quyết định P3-G3 chính thức

Lượt chạy v3 chỉ-hình-học đã hoàn thành tất cả 75.000 cập nhật mà không gặp lỗi số. Checkpoint cuối cùng của nó được đánh giá trên tất cả 558 đoạn kiểm định Tầng A/B bất biến bằng cách sử dụng lấy mẫu sub-VP có điều kiện 30 bước, biến dạng 35 độ xác định, mặt nạ thân trên/tay/ngón tay/cổ tay cố định và các đỉnh vùng SMPL-X được giải mã. Bộ đánh giá được sửa đổi kẹp chặt mọi khớp khởi tạo không bị biến dạng, bao gồm cả các tổ tiên thân thể không có nhãn trong các mẫu InterHand chỉ có tay.

| Điều kiện P3-G3 | Kết quả v3 | Yêu cầu | Quyết định |
|---|---:|---:|---|
| phục hồi thân trên | **-105.99%** | >=30% | **NO-GO** |
| phục hồi mặt nạ tồi nhất tay trái | **-177.32%** | >=30% | **NO-GO** |
| phục hồi mặt nạ tồi nhất tay phải | **-239.48%** | >=30% | **NO-GO** |
| suy giảm tập sạch tối đa | **0.00%** | <1% | GO |
| độ bao phủ kiểm định | **558 / 558 đoạn** | 558 / 558 | GO |

**Quyết định P3-G3 chính thức cho v3: NO-GO.** R4 tiếp tục bị chặn. Các giá trị âm có nghĩa là mẫu kết quả đã làm tăng, thay vì loại bỏ, lỗi vùng được tiêm vào. Việc giữ lại tập sạch vượt qua vì các khớp được quan sát đầy đủ được phục hồi chính xác sau khi tích hợp ngược.

### 36.2 Nguyên nhân gốc

`joint_valid` trong các cache này mô tả các mục tiêu vị trí khớp 3D được giải mã tùy chọn; nó không phải tính hợp lệ của quan sát tư thế. Nó là 0% cho thân trên, tay trái và tay phải trên toàn bộ 558 đoạn phân chia kiểm định. Bộ huấn luyện v3 ban đầu sử dụng trường này để xây dựng mặt nạ điều kiện hóa, vì vậy mọi token quan sát đều bằng 0 và mọi cạnh quan hệ đều bị loại bỏ. Do đó, mô hình v3 75.000 cập nhật đã học một mô hình điểm số không điều kiện mặc dù ở cấu hình không gian bị che.

Hai sửa đổi chính xác bổ sung được yêu cầu:

1. khuếch tán ngược giờ chuyển mặt nạ khớp vào mạng điểm số, loại bỏ các cạnh quan hệ chạm vào các điểm đầu ẩn, đi theo đường dẫn nhiễu tiến cố định cho các khớp được quan sát và phục hồi các khớp đó chính xác ở cuối; và
2. các mẫu chỉ có tay giữ lại tất cả các xoay bộ khởi tạo không bị biến dạng làm điều kiện hóa, ngay cả khi giám sát mục tiêu không có sẵn, ngăn các tổ tiên thân ngẫu nhiên làm dịch chuyển các đỉnh tay sạch.

Bộ đánh giá fail-close trên độ bao phủ không hoàn chỉnh và tiêm các biến dạng xác định trước khi tính toán phục hồi, tránh baseline lỗi bằng 0 không xác định dẫn đến từ các mục tiêu đồng nhất Tầng A/B sạch.

### 36.3 Lựa chọn checkpoint dựa trên kiểm định

Huấn luyện R3 tương lai giờ đây chạy kiểm định EMA xác định mỗi 2.500 cập nhật, ghi một `validation_<step>.json` bất biến, và cập nhật `best.pt` chỉ khi điểm vùng bằng nhau khai báo trước cải thiện. `last.pt` không còn được sao chép mù quáng sang `best.pt`. Tám lần kiểm định liên tiếp không cải thiện sẽ kích hoạt dừng sớm. Bộ chọn trong quá trình huấn luyện sử dụng một đại diện SO(3) bị che cố định cho việc xếp hạng checkpoint tiết kiệm; quyết định P3-G3 cuối cùng vẫn yêu cầu đánh giá đỉnh giải mã hoàn chỉnh ở trên.

### 36.4 Lượt chạy sửa chữa v4b được chấp nhận

Bản thử nghiệm khởi động ấm điều kiện đầu tiên đã làm lộ các trọng số phép chiếu quan sát/quan hệ ngẫu nhiên ngủ yên: tổn thất bước 1 của nó là 72,63 và nó đã bị dừng trước checkpoint. Nó được lưu giữ là `rdp_r3_spatial_geometry_only_v4_seed42_superseded_uncalibrated_conditioning`. Đối với v4b, chỉ hai ma trận trọng số phép chiếu đó được khởi tạo về 0; các bias của chúng và prior v3 đã học hoàn chỉnh giữ nguyên không đổi. Điều này khiến cập nhật 0 tái tạo chính xác điểm số không điều kiện ổn định trong khi cho phép điều kiện hóa học tập. Bước thử nghiệm đã hiệu chỉnh có tổng tổn thất 0,1085 thay vì 72,63.

Lượt chạy được chấp nhận đang hoạt động với:

| Mục | Giá trị |
|---|---|
| tmux | `phase3_r3_geometry_only_v4b` |
| nhật ký | `logs/phase3/rdp_r3_spatial_geometry_only_v4b_seed42.txt` |
| đầu ra | `outputs/phase3_training/rdp_r3_spatial_geometry_only_v4b_seed42` |
| khởi tạo | prior hình học v3 EMA cộng với đồ thị R2 bước 36.000 đóng băng chính xác |
| tổng tổn thất / tổn thất điểm số cập nhật 1 | `0.12046 / 0.01263` |
| năng lượng tiếp xúc | tắt |
| khung xương quan hệ | đóng băng |
| bộ kiểm thử hồi quy | **31 passed**, lint và biên dịch vượt qua |
| kiểm tra trước cuối cùng | **GO, 0 điểm chặn, 16/16 kiểm tra** |

P3-G3 vẫn **NO-GO/CHỜ ĐỊNH** cho đến khi checkpoint được chọn bởi kiểm định v4b vượt qua đánh giá giải mã 558 đoạn hoàn chỉnh. Không bắt đầu R4 từ v3 hoặc từ một checkpoint v4b chưa đánh giá.

| Artifact | SHA-256 |
|---|---|
| v3 `best.pt` | `f3825c40e7cc00bd318cccaef2b6eaae9efeca40a43402be2068b88b6aec6e14` |
| đánh giá G3 v3 hoàn chỉnh được sửa | `f28328d35b62e749cb0bd8d28e327c2839ad7c370f4086a55087d743b4dafd05` |
| quyết định G3 v3 chính thức được sửa | `fdb65ddd6be2c2475b59ac5f024a30aa92a9f528e0626e9523dc18f13db5128d` |
| cấu hình v4b được chấp nhận | `c06eff40e7767902164817cb92a002a5f31de6057932121af562aaf50372b003` |
| kiểm tra trước v4b được chấp nhận | `cd3b5660159658f232be85a5f483cb64ff6f48aeebaaffdfb4fb36347148c13b` |

### 36.5 Hoàn thành v4b và quyết định P3-G3 chính thức

Lượt chạy v4b được sửa chữa đã dừng ở cập nhật 30.000 theo quy tắc kiên nhẫn khai báo trước. Điểm số lựa chọn checkpoint SO(3) cải thiện qua cập nhật 10.000, sau đó không cải thiện trong 8 lần kiểm định 2.500 cập nhật liên tiếp. Do đó `best.pt` là checkpoint EMA ở cập nhật 10.000; `last.pt` là trạng thái dừng ở cập nhật 30.000.

| Mục kiểm định | Kết quả |
|---|---:|
| cập nhật tốt nhất | **10.000** |
| điểm lựa chọn tốt nhất | **2.46289** |
| phục hồi đại diện thân trên cập nhật 10k | -64.58% |
| phục hồi đại diện tay trái cập nhật 10k | -158.10% |
| phục hồi đại diện tay phải cập nhật 10k | -141.18% |
| kiểm định không cải thiện khi dừng | 8 |

Checkpoint được chọn đã được đánh giá với quy trình giải mã SMPL-X 30 bước, 558 đoạn bất biến hoàn chỉnh:

| Điều kiện P3-G3 | Kết quả v4b | Yêu cầu | Quyết định |
|---|---:|---:|---|
| phục hồi thân trên | **-52.81%** | >=30% | **NO-GO** |
| phục hồi mặt nạ tồi nhất tay trái | **-171.05%** | >=30% | **NO-GO** |
| phục hồi mặt nạ tồi nhất tay phải | **-158.42%** | >=30% | **NO-GO** |
| suy giảm tập sạch tối đa | **0.00%** | <1% | GO |
| độ bao phủ kiểm định | **558 / 558 đoạn** | 558 / 558 | GO |

Kết quả giải mã theo cấp mặt nạ là:

| Mặt nạ | Biến dạng ban đầu | Dự đoán v4b | Phục hồi |
|---|---:|---:|---:|
| thân trên | 186.31 mm | 284.69 mm | -52.81% |
| toàn bộ tay trái | 9.98 mm | 27.05 mm | -171.05% |
| chuỗi ngón tay trái | 2.05 mm | 4.64 mm | -126.44% |
| đính kèm cổ tay trái | 25.51 mm | 44.13 mm | -73.03% |
| toàn bộ tay phải | 10.10 mm | 26.11 mm | -158.42% |
| chuỗi ngón tay phải | 2.01 mm | 4.47 mm | -122.15% |
| đính kèm cổ tay phải | 25.33 mm | 44.26 mm | -74.78% |

**Quyết định P3-G3 chính thức cho v4b: NO-GO.** Huấn luyện có điều kiện đã giảm đáng kể một số lỗi dự đoán giải mã so với v3, đáng chú ý nhất là thân trên, toàn bộ tay phải, cả hai trường hợp đính kèm cổ tay và cả hai chuỗi ngón tay. Tuy nhiên, nó vẫn chưa phục hồi tư thế được tiêm vào ở bất kỳ vùng yêu cầu nào. Thất bại còn lại không phải là vấn đề độ bao phủ hay an toàn tập sạch. Đó là thất bại chất lượng posterior: prior không gian bằng 0 hiện tại và đường dẫn dòng xác suất Gaussian-sang-tư-thế tạo ra các mẫu tồi tệ hơn bộ khởi tạo bị biến dạng nhẹ 35 độ. Không tiến hành sang R4. Thử nghiệm R3 tiếp theo phải giảm thiểu rủi ro ánh xạ prior không gian/score và một cầu nối có điều kiện căn giữa theo bộ khởi tạo trên kiểm định Tầng A/B bên ngoài mà không làm thay đổi cổng.

| Artifact | SHA-256 |
|---|---|
| v4b `best.pt` được chọn (cập nhật 10.000) | `30c312674214ac4f32d25b5d1012600e52689d9c31862c0715463cc6649d75b4` |
| v4b `last.pt` bị dừng (cập nhật 30.000) | `948c76d1f8a570fca158c9dec511f053c98326e0416b7be58d83705397a6f8ba` |
| nhật ký huấn luyện chỉ-thêm v4b | `f1bdf3f5942b5b153df4c70d8d4ac05bfd4880917b8fbdb4167dc4ef1c797ee2` |
| đánh giá G3 giải mã hoàn chỉnh v4b | `105023d0d79af2e2f3d3c0ad29acafc1cbd1f086ffdc65188fdb2109fd066203` |
| quyết định G3 chính thức v4b | `2c32bb5cc63d930e4f0ff1de88261c317379421c631a18f8a014e9d869876b2a` |

---

## 37. Hoàn thành thặng dư điều kiện trung tâm bộ khởi tạo R3 (v5, 05-08-2026)

### 37.1 Nguyên nhân gốc và ranh giới sửa chữa

Baseline cổng v4b và posterior đã học đã không giải quyết cùng một độ khó. Bộ đánh giá P3-G3 làm nhiễu các phép xoay bộ khởi tạo được chọn tối đa 35 độ, vì vậy đầu vào của nó đã là một ước lượng gần mục tiêu mạnh mẽ. Ngược lại, v4b đã xóa các phép xoay, vận tốc và gia tốc bị biến dạng và khởi tạo các khớp đó từ nhiễu Gaussian. Do đó nó đã cố gắng hoàn thành không điều kiện các vùng khó nhất và cấu trúc khó có thể đánh bại bộ khởi tạo bị biến dạng nhẹ. Tinh chỉnh bộ lấy mẫu không thể sửa chữa sự không phù hợp thông tin này.

V5 triển khai sự hoàn thành thặng dư có điều kiện mà không làm lộ các mục tiêu sạch. Bộ đánh giá và bộ huấn luyện trước tiên tiêm cùng một biến dạng SO(3) bị chặn vào bộ khởi tạo, tính toán lại các đặc trưng xoay/chuyển động của nó, và chỉ làm lộ các đặc trưng bị biến dạng đó thông qua một phép chiếu `corruption_observation` riêng biệt. Đường dẫn quan sát bình thường vẫn bị giới hạn ở các khớp đáng tin cậy, các cạnh quan hệ chạm vào điểm đầu ẩn vẫn bị che, và hai mặt nạ phải biệt lập. Các khớp khởi tạo không bị biến dạng nhưng không hợp lệ với mục tiêu vẫn đáng tin cậy để các mẫu chỉ có tay giữ lại tổ tiên thân thể. Phép chiếu mới được khởi tạo bằng 0; do đó mô hình v5 tái tạo chính xác posterior v4b được chọn trước khi học gợi ý thặng dư. Một đợt warm-up 5.000 cập nhật chỉ dành cho đầu chiếu ngăn prior hình học tiền huấn luyện bị trôi dạt trong khi đường dẫn đầu vào đó được hiệu chỉnh.

Sửa chữa này không làm yếu bất kỳ cổng nào hay sử dụng Lane-L. Huấn luyện và lựa chọn vẫn giữ nguyên trên phân chia ARCTIC/InterHand Tầng A/B bất biến. Năng lượng tiếp xúc, tổn thất tiếp xúc, liên kết lực, rằng buộc duy trì và tổn thất duy trì tiếp tục bị tắt; mô hình quan hệ R2 bước 36.000 vẫn là một bộ trích xuất đặc trưng hình học/chiều sâu đóng băng.

### 37.2 Xác minh và khởi chạy

Triển khai vượt qua Ruff, biên dịch bytecode và **34/34 kiểm thử Phase 3**. Độ bao phủ hồi quy mới xác minh tính biệt lập mặt nạ biến dạng/điều kiện hóa, hành vi CFG dropout, khởi tạo bằng 0, lấy mẫu hữu hạn và dòng độ dốc khác 0 tới phép chiếu gợi ý. Lượt chạy thử GPU hai cập nhật hoàn thành với các tổn thất hữu hạn và đường dẫn tối ưu hóa chỉ-gợi-ý hoạt động.

Kiểm tra trước fail-closed v5 báo cáo **GO với 0 điểm chặn và 16/16 kiểm tra vượt qua**: 3.499 đoạn huấn luyện Tầng A/B, 558 đoạn kiểm định biệt lập nguồn/người ký/nhóm nguồn, khởi tạo quan hệ đóng băng nghiêm ngặt, lựa chọn checkpoint dựa trên kiểm định và hợp đồng dự phòng chỉ-hình-học hoàn chỉnh.

| Mục | Giá trị |
|---|---|
| tmux | `phase3_r3_geometry_only_v5` |
| nhật ký chỉ-thêm | `logs/phase3/rdp_r3_spatial_geometry_only_v5_seed42.txt` |
| đầu ra | `outputs/phase3_training/rdp_r3_spatial_geometry_only_v5_seed42` |
| khởi tạo | checkpoint v4b cập nhật 10.000 được chọn |
| bộ trích xuất hình học đóng băng | checkpoint R2 bước 36.000 |
| warm-up chỉ-gợi-ý | 5.000 cập nhật |
| tổng tổn thất / tổn thất điểm số cập nhật 1 | `0.09333 / 0.01236` |
| tỷ lệ gợi ý cập nhật 1 | `0.09804` |
| năng lượng tiếp xúc / khung xương quan hệ | tắt / đóng băng |
| mức trần CPU | 4 luồng |

| Artifact | SHA-256 |
|---|---|
| cấu hình v5 | `587d998dafed8bcb7f6061a534bff82b5fcd30a7e3396d4b6ee8847dfde6b256` |
| kiểm tra trước v5 | `17c40b6c76dacfb25da1c3057bb9dcc6133b7fd5499b244c171ca65bd00b371f` |
| checkpoint khởi tạo v4b | `30c312674214ac4f32d25b5d1012600e52689d9c31862c0715463cc6649d75b4` |
| checkpoint hình học R2 đóng băng | `69852e0d88a166bff65326ae22eb6359aada658458384b5f3c879709131d33b9` |
| quyết định P3-G2 đóng băng | `5cceb65aca11d026d057d4d172dd435a81de71edc9aac7840d35ab6b986c18e8` |

P3-G3 giữ nguyên hình thức **NO-GO** cho đến khi checkpoint v5 được chọn bởi kiểm định vượt qua đánh giá giải mã 558 đoạn hoàn chỉnh: ít nhất 30% phục hồi ở thân trên và cả hai tay trường hợp tồi nhất, suy giảm tập sạch dưới 1%, và độ bao phủ đầy đủ. R4 tiếp tục bị chặn. Lần khởi chạy này là một sửa chữa nguyên nhân gốc, không phải tuyên bố rằng cổng số học đã vượt qua.

### 37.3 Huấn luyện v5 hoàn thành và kết quả P3-G3 chính thức

V5 dừng sớm ở cập nhật 70.000 sau 8 lần kiểm định liên tiếp không có sự cải thiện. Bộ chọn kiểm định bất biến đã chọn checkpoint EMA ở cập nhật 50.000, trước khi có các suy giảm sau đó. Đại diện SO(3) 10 bước của nó đã phục hồi 56,03% thân trên, 44,97% tay trái và 44,78% tay phải, với điểm chọn 0.642568 và không có suy giảm tập sạch.

Checkpoint được chọn sau đó được đánh giá một lần với tập kiểm định Tầng A/B 558 đoạn bất biến hoàn chỉnh, 7 mặt nạ chính thức, 30 bước ngược, seed 3042 và các đỉnh vùng SMPL-X giải mã. Lane-L không được mở.

| Điều kiện P3-G3 | Kết quả v5 | Yêu cầu | Quyết định |
|---|---:|---:|---|
| phục hồi thân trên | **65.88%** | >=30% | **GO** |
| phục hồi mặt nạ tồi nhất tay trái | **48.29%** | >=30% | **GO** |
| phục hồi mặt nạ tồi nhất tay phải | **48.55%** | >=30% | **GO** |
| suy giảm tập sạch tối đa | **0.00%** | <1% | **GO** |
| độ bao phủ kiểm định | **558 / 558 đoạn** | 558 / 558 | **GO** |

Kết quả mặt nạ giải mã hoàn chỉnh là:

| Mặt nạ | Biến dạng ban đầu | Dự đoán v5 | Phục hồi |
|---|---:|---:|---:|
| thân trên | 187.12 mm | 63.84 mm | **65.88%** |
| toàn bộ tay trái | 9.95 mm | 5.08 mm | **48.91%** |
| chuỗi ngón tay trái | 2.04 mm | 1.05 mm | **48.29%** |
| đính kèm cổ tay trái | 25.72 mm | 11.65 mm | **54.69%** |
| toàn bộ tay phải | 10.04 mm | 5.16 mm | **48.55%** |
| chuỗi ngón tay phải | 2.06 mm | 1.04 mm | **49.33%** |
| đính kèm cổ tay phải | 25.64 mm | 11.88 mm | **53.68%** |

**Quyết định P3-G3 chính thức: GO.** Điều này giải quyết điểm chặn phục hồi không gian bị che. R4 giờ có thể bắt đầu dưới cùng ranh giới dự phòng chỉ-hình-học: năng lượng tiếp xúc, tổn thất tiếp xúc, liên kết lực và rằng buộc duy trì vẫn bị tắt. Quyết định tiếp xúc P3-G2 giữ nguyên NO-GO và không bị ghi đè bởi kết quả này.

| Artifact | SHA-256 |
|---|---|
| v5 `best.pt` được chọn (cập nhật 50.000) | `9c871f259be4be3b8c4f1d3dfe368a175a8b50c760626c230dc15c3a3a1b3fc3` |
| v5 `last.pt` bị dừng (cập nhật 70.000) | `6885963402d20b19d5817fd726f6a50408c7006d4e28b62b5ff73b7e5461e5b1` |
| nhật ký huấn luyện chỉ-thêm v5 | `1ed3d00e5910d8c67773c7f9e86b768c71943a1883e3bca7ef2b941861e5ab95` |
| nhật ký đánh giá chính thức v5 | `4d2d2dac76b0b9d58881574124cec6bf8a368d0862d3b2fe402ce86350c84b11` |
| đánh giá P3-G3 giải mã đầy đủ v5 | `e493ec07b1706a053cd9058bae4702f8931b61aa8679b10805d6a197268cb475` |
| quyết định P3-G3 độc lập v5 | `c920917ed4cfe37c97cb2e6b0271739b0c6bb5f8f0da618bd00d670ba6049fa4` |

---

## 38. Chiến lược phục hồi tiếp xúc thời gian P3-G2 (v3, 06-08-2026)

### 38.1 Điểm bắt đầu đóng băng và chẩn đoán thất bại

Đây là một nhánh huấn luyện lại tiếp xúc riêng biệt. Nó không sửa đổi hay làm mất hiệu lực checkpoint R3 chỉ-hình-học GO. Quyết định nguồn P3-G2 giữ nguyên NO-GO với F1 tiếp xúc tổng thể 0.7049, F1 tay--thân thể ký hiệu 0.4667 và chỉ cải thiện 0.72% trượt. Checkpoint v2b cập nhật 36.000 là sự khởi tạo bất biến.

Các thất bại quan sát được có ba nguyên nhân trực tiếp:

1. **Mất cân bằng cực đoan trong miền.** Huấn luyện How2Sign chứa 2.506 khung hình-cạnh tay--thân thể dương trong số 20.434.560 khung hình hợp lệ (0,0123%). Trọng số miền 4x cũ nhân các cạnh ký hiệu dương và âm bằng nhau do đó không cân bằng bài toán phân loại này. Kiểm định chỉ có 226 điểm dương.
2. **Duy trì độc lập theo khung hình.** Đồ thị v2b xử lý mọi khung hình một cách độc lập. Đầu duy trì của nó không có trạng thái thời gian, mặc dù sự duy trì được định nghĩa qua các khung hình liền kề.
3. **Mất kết nối giữa mục tiêu và suy luận.** Sự duy trì được giám sát trên tất cả các cạnh hợp lệ, nơi lớp không-tiếp-xúc tầm thường chiếm ưu thế, và logit của nó không bao giờ được dùng để chọn tiếp xúc. Độ trượt chỉ có thể cải thiện gián tiếp qua các đặc trưng chia sẻ. Có điều kiện trên một tiếp xúc tay--thân thể ký hiệu thực sự, 49,1% nhãn huấn luyện là duy trì, cung cấp một mục tiêu thời gian lành mạnh hơn nhiều.

Quét ngưỡng trên mô hình cũ chỉ được lưu giữ làm chẩn đoán. V3 không tinh chỉnh ngưỡng trên tập kiểm định chính thức: ngưỡng được đóng băng ở mức 0,5 trước lượt chạy mới.

### 38.2 Kiến trúc v3 bổ sung và các tổn thất

Khung xương quan hệ hình học/chiều sâu được tải từ v2b và đóng băng vĩnh viễn. Khoảng cách, chiều sâu, token cạnh và token quan hệ của nó bỏ qua bộ thích ứng tiếp xúc mới, vì vậy mức tăng MAE khoảng cách 15.61% và độ chính xác chiều sâu 98.43% đã vượt qua không thể bị trôi dạt trong quá trình phục hồi tiếp xúc.

Mỗi cạnh cố định nhận một GRU hai chiều trên chuỗi 64 khung hình của nó. Phép chiếu thặng dư khởi tạo bằng 0 khiến bộ thích ứng thời gian trở thành đồng nhất chính xác ở cập nhật 0. Một đầu tiếp xúc/duy trì được sao chép sau đó được tối ưu hóa với:

- tổn thất focal tiếp xúc tổng thể ban đầu;
- tổn thất tay--thân thể ký hiệu được chuẩn hóa riêng chứa tất cả điểm dương và tối đa 8 điểm âm tồi nhất cho mỗi điểm dương;
- tổn thất focal duy trì có điều kiện tiếp xúc, chỉ được đánh giá trên tiếp xúc thực sự; và
- chấm điểm nhận biết duy trì rõ ràng
  `guided_contact_logit = contact_logit + 2 * persistence_logit`.

Bộ so sánh không-duy-trì có cùng kiến trúc thời gian, cùng khởi tạo v2b đóng băng, lấy mẫu, optimizer và các tổn thất tiếp xúc. Khác biệt duy nhất của nó là tổn thất duy trì bằng 0 và trọng số hợp nhất bằng 0. Điều này làm cho thử nghiệm phân tách trượt mang tính nguyên nhân hơn là so sánh các checkpoint không liên quan.

### 38.3 Dữ liệu, giáo trình và các siêu tham số đóng băng

Cache quan hệ v2 biệt lập nguồn/người ký/nhóm nguồn được tái sử dụng mà không biến đổi. Lane-L và dữ liệu đánh giá tác giả vẫn bị cấm. Huấn luyện lấy mẫu 70% How2Sign, 20% ARCTIC và 10% InterHand; 65% số đoạn How2Sign được lấy mẫu là tiếp xúc-dương. Điều này thay đổi tần suất lấy mẫu nhưng không bao giờ dán nhãn lại một cạnh.

| Tham số | Giá trị đóng băng |
|---|---:|
| cập nhật tối đa | 20.000 |
| physical / accumulated / effective batch | 8 / 4 / 32 |
| tốc độ học / weight decay | 1e-4 / 0.01 |
| chiều rộng ẩn thời gian | 128 hai chiều |
| trọng số tổn thất tiếp xúc ký hiệu | 4.0 |
| âm tính cứng cho mỗi dương tính ký hiệu | 8 |
| trọng số duy trì có điều kiện | 1.0 |
| trọng số hợp nhất duy trì | 2.0 |
| ngưỡng tiếp xúc | 0.5, đóng băng |
| khoảng thời gian kiểm định / độ kiên nhẫn | 1.000 / 8 lần kiểm định |
| công nhân CPU | 4 |

Lựa chọn checkpoint duy trì fail closed trên vectơ P3-G2 hoàn chỉnh. GO chính thức vẫn yêu cầu mức tăng MAE quan hệ >=10%, F1 tiếp xúc tổng thể >=0.65, F1 tay--thân thể ký hiệu >=0.60, độ chính xác chiều sâu >=0.80, mức tăng trượt >=15%, một bộ so sánh không-duy-trì hợp lệ và suy giảm vùng <=1%. Không chỉ số riêng lẻ nào có thể thay thế cho quyết định này.

### 38.4 Xác minh và độ sẵn sàng khởi chạy

Ruff, biên dịch và **36/36 kiểm thử Phase 3** vượt qua. Các kiểm thử bao gồm khởi tạo đồng nhất chính xác, đầu ra hình học đóng băng, độ dốc tiếp xúc phân tầng hữu hạn và độ dốc duy trì có điều kiện tiếp xúc. Lượt chạy thử GPU hai cập nhật đã hoàn thành một kiểm định 1.312 đoạn đầy đủ hai lần. Chẩn đoán cập nhật 2 của nó giữ lại mức tăng MAE quan hệ 15.61%, độ chính xác chiều sâu 98.43%, suy giảm tái tạo bằng 0 và đo được phân tách trượt 5.32% trước khi có huấn luyện thời gian ý nghĩa.

Kiểm tra trước phục hồi fail-closed báo cáo **GO, 13/13 kiểm tra, 0 điểm chặn**, với 14.142 đoạn huấn luyện, 1.312 đoạn kiểm định, 2.506 khung hình-cạnh dương tay--thân thể ký hiệu huấn luyện và 226 kiểm định, khởi tạo cập nhật 36.000 đã khóa, danh tính biệt lập, ngưỡng đóng băng và không có đường dẫn Lane-L.

| Artifact | Giá trị |
|---|---|
| cấu hình | `phase3_posterior/configs/rdp_r2_contact_recovery_v3.yaml` |
| đầu ra | `outputs/phase3_training/rdp_r2_contact_recovery_v3_seed42` |
| tmux | `phase3_r2_contact_recovery_v3` |
| nhật ký | `logs/phase3/rdp_r2_contact_recovery_v3_seed42.txt` |
| SHA-256 cấu hình | `14ed63e8346cb31961f2aaaab2442ba6d41fa0128908939db077b2e2d2a912bc` |
| SHA-256 v2b đóng băng | `69852e0d88a166bff65326ae22eb6359aada658458384b5f3c879709131d33b9` |
| SHA-256 quyết định P3-G2 nguồn | `5cceb65aca11d026d057d4d172dd435a81de71edc9aac7840d35ab6b986c18e8` |

Mục này đóng băng chiến lược phục hồi trước bất kỳ kết quả kiểm định chính thức v3 nào. P3-G2 vẫn giữ nguyên NO-GO cho đến khi quyết định số học hoàn chỉnh vượt qua từng điều kiện.

### 38.5 Kết quả thực thi V3 và V4

V3 đã xác nhận giả thuyết duy trì thời gian nhưng không xác nhận giả thuyết phân loại tiếp xúc ký hiệu. Checkpoint cập nhật 7.000 được chọn đã vượt qua trượt với mức tăng 20,73% và giữ lại F1 tổng thể 0.6614, nhưng F1 ký hiệu chỉ là 0.4644. Các cập nhật sau đó làm tăng trượt trong khi giảm cả chất lượng tiếp xúc tổng thể và ký hiệu. Lượt chạy đã được dừng sau cập nhật 8.000 thay vì chi tiêu ngân sách còn lại cho sự đánh đổi sai lầm.

V4 cung cấp cho tiếp xúc một bản sao có thể huấn luyện của bộ mã hóa quan hệ trong khi giữ lại nhà cung cấp hình học/chiều sâu riêng biệt đóng băng. Kiểm tra trước fail-closed của nó đã vượt qua 15/15 kiểm tra sau khi bắt và sửa cờ `requires_grad=False` bị sao chép. Bản thử nghiệm không hợp lệ được lưu giữ là `rdp_r2_contact_recovery_v4_smoke_superseded_frozen_encoder`. Lượt chạy thử được chấp nhận và **37/37 kiểm thử** đã vượt qua.

V4 đã chọn cập nhật 1.500. Đánh giá chính thức biệt lập nguồn hoàn chỉnh là:

| Điều kiện P3-G2 | Kết quả V4 | Yêu cầu | Quyết định |
|---|---:|---:|---|
| mức tăng MAE khoảng cách quan hệ | **15.61%** | >=10% | GO |
| F1 tiếp xúc tổng thể | **0.6537** | >=0.65 | GO |
| F1 tiếp xúc tay--thân thể ký hiệu | **0.4673** | >=0.60 | **NO-GO** |
| độ chính xác thứ tự chiều sâu | **98.43%** | >=80% | GO |
| mức tăng trượt so với không-duy-trì | **23.06%** | >=15% | GO |
| bộ so sánh trượt sẵn có | true | true | GO |
| suy giảm tái tạo tối đa | **0.00%** | <=1% | GO |

Số lượng ma trận nhầm lẫn ký hiệu là TP=132, FP=207, FN=94: độ chính xác 0.3894 và độ gợi nhớ 0.5841. **Quyết định P3-G2 chính thức V4: NO-GO**, với tiếp xúc ký hiệu là điều kiện thất bại duy nhất. Năng lượng tiếp xúc tiếp tục bị tắt. Kết quả R3 chỉ-hình-học GO vẫn giữ nguyên hợp lệ và không đổi.

### 38.6 Điểm chặn miền bằng chứng

Việc tinh chỉnh dung lượng hoặc ngưỡng hơn nữa không được biện minh trên cache hiện tại. Một đợt quét ngưỡng chẩn đoán của mô hình thời gian được chọn không thể vượt quá F1 ký hiệu 0.49. Hai cạnh ký hiệu ưu thế có ROC-AUC khoảng cách bộ khởi tạo khoảng 0.994, nhưng sự mất cân bằng lớp cực đoan của chúng để lại độ chính xác trung bình chỉ 0.58 và 0.47. Một probe thời gian 49 đặc trưng phi tuyến được huấn luyện trên tất cả 2.506 điểm dương ký hiệu đạt F1 huấn luyện 0.958, sau đó sụp đổ trên các người ký kiểm định chưa thấy về F1 0.268 (217 TP, 1.178 FP, 9 FN). Đây là bằng chứng mạnh mẽ về sự dịch chuyển mục tiêu/bằng chứng xuyên người ký, không phải vấn đề optimizer hay ngưỡng.

Các mục tiêu ký hiệu được dẫn xuất từ hình học hiện tại cũng quá thưa thớt cho một tuyên bố tiếp xúc cấp bài báo: chỉ 2.506 khung hình-cạnh huấn luyện dương và 226 khung hình-cạnh kiểm định dương, tập trung chủ yếu vào hai cạnh đại diện đầu ngón tay--ngực. Phân chia hiệu chỉnh 420 đoạn chỉ có 25 khung hình-cạnh tay--thân thể ký hiệu dương. Nó không thể hỗ trợ hiệu chỉnh ổn định theo ngưỡng hoặc đặc thù theo cạnh. Reusing kiểm định chính thức để chọn các giá trị này sẽ là rò rỉ.

### 38.7 Chiến lược tập trung dữ liệu bắt buộc cho P3-G2 GO hợp lý

Nỗ lực tiếp theo là **R2 sign-contact target v3**, và nó phải dừng lại ở mỗi cổng thứ tự bên dưới thay vì khởi chạy một mô hình khác trên các nhãn hiện tại.

1. **D0: kiểm toán tiếp xúc và chú thích.** Xây dựng ít nhất 300 đoạn How2Sign/PHOENIX biệt lập nguồn/người ký, phân tầng cố ý trên các trường hợp tay--mặt, tay--thân, hai tay, âm tính gần tiếp xúc, chuyển tiếp nhanh, bị che và không tiếp xúc. Đánh giá kép ít nhất 10%; yêu cầu lỗi mục tiêu thảm họa dưới 10% và sự đồng thuận giữa người đánh giá (Cohen's kappa >=0.75). Ghi lại khởi phát, duy trì, giải phóng, vùng thân thể được tiếp xúc và khả năng nhìn thấy. **NO-GO** nếu đồng thuận/hỗ trợ thất bại.
2. **D1: mục tiêu bề mặt lưới.** Thay thế các đại diện hình cầu khớp cố định bằng khoảng cách từ đỉnh tay gần nhất đến bề mặt thân thể, định danh bộ phận thân thể, căn chỉnh pháp tuyến bề mặt, vận tốc tiếp tuyến và trễ khởi phát/giải phóng 12/20-mm đóng băng. Sử dụng bằng chứng video để từ chối các tiếp xúc giả chỉ-chiều-sâu. Giữ nguyên các sidecar v2 ban đầu; cụ thể hóa một cache mới và kiểm toán từng băm.
3. **D2: phân chia biệt lập người ký được hỗ trợ.** Yêu cầu ít nhất 2.000 khung hình-cạnh tiếp xúc ký hiệu huấn luyện dương, 500 hiệu chỉnh và 1.000 kiểm định dương, với ít nhất 50 đoạn dương cho mỗi nhóm người ký đại diện. Đóng băng hiệu chỉnh và kiểm định trước các thử nghiệm đặc trưng/mô hình. **NO-GO** nếu bất kỳ mức tối thiểu hỗ trợ dương nào thất bại.
4. **D3: đặc trưng quan sát đóng băng.** Chỉ sử dụng bằng chứng Phase-1 có thể triển khai: hình học tay WiLoR/HaMeR, hình học thân SMPLer-X, độ tin cậy 2D/thân Sapiens, nhúng cắt xén tay/thân sẵn có địa phương, độ tin cậy điểm đầu cuối, khoảng cách bề mặt, vận tốc tiếp cận và khả năng nhìn thấy. Không sử dụng đầu ra R3 hoặc GT khi suy luận, tránh thứ tự R2-sang-R3 vòng quanh.
5. **D4: probe đầy đủ đặc trưng.** Huấn luyện một probe tiếp xúc nhỏ chỉ trên tập huấn luyện và đóng băng ngưỡng của nó trên tập hiệu chỉnh. Nó phải đạt F1 ký hiệu >=0.65 cùng độ chính xác và độ gợi nhớ đều >=0.60 trên tập kiểm định biệt lập người ký nguyên bản. **NO-GO** ở đây có nghĩa là cải thiện mục tiêu/đặc trưng; không khởi chạy đồ thị quan hệ.
6. **D5: huấn luyện lại quan hệ/tiếp xúc.** Tiền huấn luyện hình học trên ARCTIC/InterHand, sau đó chạy thích ứng cân bằng ký hiệu với lấy mẫu đoạn tiếp xúc-dương, âm tính cứng theo vùng, điều khoản xếp hạng AUPRC/listwise, sự duy trì có điều kiện tiếp xúc và bộ so sánh không-duy-trì đồng nhất. Giữ hình học/chiều sâu được bỏ qua và đóng băng. Chọn các checkpoint trên tập hiệu chỉnh, không bao giờ trên tập kiểm định chính thức.
7. **D6: P3-G2 chính thức.** Đóng băng ngưỡng, hợp nhất và checkpoint; chạy cổng hoàn chỉnh một lần trên tập kiểm định biệt lập người ký nguyên bản. Yêu cầu từng điều kiện số học, sau đó lặp lại seed 123 và 456 làm kiểm toán độ ổn định. Năng lượng tiếp xúc chỉ có thể được bật phía hạ nguồn nếu tất cả điều kiện chính thức vượt qua và độ lệch chuẩn seed F1 ký hiệu dưới 0.03.

Chiến lược này không chặn tiến triển R4 chỉ-hình-học, nhưng nó chặn bất kỳ tuyên bố nào rằng năng lượng tiếp xúc là an toàn. Hành động đúng đắn ngay lập tức là xây dựng mục tiêu/bằng chứng thông qua D0--D4, không phải tinh chỉnh siêu tham số bổ sung trên tập kiểm định chính thức 226 điểm dương.

| Artifact | SHA-256 |
|---|---|
| cấu hình v3 | `14ed63e8346cb31961f2aaaab2442ba6d41fa0128908939db077b2e2d2a912bc` |
| kiểm tra trước v3 | `cead29ed0f6baab79a017aeef423b6a899bf55c73f404bb02ad1be60acb01856` |
| checkpoint v3 được chọn | `ba50500c7b0d4f7403040dc913d47a042867e0e47926de48a6ca393593e821ee` |
| cấu hình v4 | `f5d2e7582158b2abc9a93c9a7bf9ce2487447a68840c8ea7bbe35eaa7b009a31` |
| kiểm tra trước v4 | `d0e3ecfb7fa659e03b69ae58a8750c216e5ee7a3e6df776c66baabf3e640ad91` |
| checkpoint v4 được chọn | `5021cc6295780e72e1348467707fa00c00c3c1b84f9a9c60e4fd7af9463fd754` |
| đánh giá chính thức v4 | `1ed880906175ebdceb2e6d8b82e8f9250a4b72088161a0ea64450717104dc90f` |
| quyết định P3-G2 chính thức v4 | `47eb07d253801d76bddb96845bf8cedcb4e51278f100fa6584aa9496383c7909` |

---

## 39. Khôi phục nhánh quan sát và kiểm toán người ký 10 niêm phong (06-08-2026)

### 39.1 Nguyên nhân gốc và sửa chữa dữ liệu bổ sung

Thất bại V4 không phải do dung lượng mạng không đủ hay do ngưỡng 0.5 đóng băng. Các đợt quét ngưỡng của mô hình cũ vẫn ở dưới F1 tiếp xúc ký hiệu 0.49, trong khi một probe chẩn đoán dung lượng cao đạt F1 huấn luyện 0.958 và chỉ 0.268 trên các người ký chưa thấy. Phần phân chia huấn luyện How2Sign ban đầu chỉ chứa thành phần người ký kết nối `{3, 5, 8}`. Người ký 1 và 2 chia sẻ các nhóm video nguồn, vì vậy coi họ như các nhóm nguồn độc lập cũng sẽ vi phạm hợp đồng biệt lập nguồn yêu cầu.

Tín hiệu có thể triển khai bị thiếu khỏi mô hình tiếp xúc thời gian là thặng dư 2D được quan sát trừ đi được chiếu của Phase-2. Một đợt kiểm toán độ đầy đủ chỉ-huấn-luyện đã định lượng hiệu ứng trước khi xây dựng mô hình:

| Bằng chứng chẩn đoán | F1 ký hiệu người ký chưa thấy |
|---|---:|
| chỉ đặc trưng quan hệ | 0.4459 |
| + chuyển động 2D/độ tin cậy được quan sát | 0.5475 |
| + thặng dư chiếu lại theo vùng | 0.5726 |

Thặng dư có ích nhưng tự nó không đủ. Do đó nó được đưa vào thông qua một nhánh đồ thị ngữ cảnh chứa đựng: cơ sở V4 vượt qua được đóng băng, nhánh mới khởi tạo bằng 0, và một thặng dư bằng 0 chính xác giữ nguyên các logit cũ. V6 xác định rằng giá trị EMA cũ 0.9999 đã trễ nhánh chứa đựng một cách nghiêm trọng (ở cập nhật 1.000, live delta norm 0.0774 so với EMA delta norm 0.00268). V7 chỉ thay đổi cơ chế phục hồi này thành EMA 0.99 và huấn luyện nhánh quan sát trong khi giữ khung xương hình học/chiều sâu và đường dẫn tiếp xúc cơ sở bị đóng băng. Việc chứa đựng chỉ tay-thân V8 được triển khai và thử nghiệm khói như một khoản dự trữ, nhưng cố ý **không được huấn luyện** sau khi V7 vượt qua cổng phát triển.

Mọi thay đổi đều mang tính bổ sung. Các cache, config, checkpoint và phương pháp Phase 2/3 legacy không bị ghi đè. Lane-L và tập đánh giá 1.493 khung hình của tác giả không bị đọc, huấn luyện, hay dùng cho lựa chọn.

### 39.2 Mở rộng biệt lập nguồn/người ký và xây dựng kiểm thử niêm phong

Người ký kiểm thử How2Sign chính thức 10 được trích xuất độc lập. Trong số 247 đoạn hợp lệ (7.904 khung hình), 220 đoạn vượt qua sự tinh chỉnh mục tiêu thời gian và tinh chế chiếu lại; 27 đoạn bị từ chối bởi các quy tắc chất lượng fail-closed hiện tại. Phân chia cuối cùng là:

| Phân chia | Các người ký | Các đoạn quan hệ | Vai trò |
|---|---|---:|---|
| huấn luyện | 3, 4, 5, 8, 9, 11 cộng với ARCTIC/InterHand tổng quát | 14.562 | fitting |
| kiểm định phát triển | 1, 2 cộng với ARCTIC/InterHand tổng quát | 1.312 | lựa chọn |
| kiểm thử How2Sign niêm phong | chỉ 10 | 220 | kiểm toán chuyển giao một lần |

Tập niêm phong chứa 38 đoạn tiếp xúc-dương và 123 khung hình-cạnh tay--thân thể dương trong số 422.400 khung hình-cạnh hợp lệ. Các nhóm người ký và nguồn huấn luyện, phát triển và kiểm thử là biệt lập. Bản kê khai kiểm thử đã được băm trước khi đánh giá. Nó được mở chính xác một lần sau khi checkpoint V7 cập nhật 5.000, chính sách EMA, hợp nhất điểm số, ngưỡng và mã quyết định P3-G2 chính thức đều đã được đóng băng.

### 39.3 Kết quả phát triển V5--V7

V5 cho phép bằng chứng mới cập nhật bộ mã hóa tiếp xúc đầy đủ. Nó không cải thiện đặc trưng chuyển giao lẫn việc chứa đựng và vẫn là NO-GO. V6 đóng băng cơ sở nhưng đã bị dừng sau khi xác nhận trễ EMA; nó mang tính chẩn đoán và không có tuyên bố chính thức. V7 là thử nghiệm nhánh chứa đựng được chấp nhận.

| Điều kiện P3-G2 | V5 | V7 phát triển | Yêu cầu |
|---|---:|---:|---:|
| mức tăng MAE khoảng cách quan hệ | 15.61% | **15.61%** | >=10% |
| F1 tiếp xúc tổng thể | 0.6501 | **0.6619** | >=0.65 |
| F1 tay--thân thể ký hiệu | 0.4783 | **0.6032** | >=0.60 |
| độ chính xác thứ tự chiều sâu | 98.43% | **98.43%** | >=80% |
| mức tăng trượt | 24.08% | **26.79%** | >=15% |
| suy giảm tái tạo tối đa | 0.00% | **0.00%** | <=1% |

Kiểm tra trước V7 fail-closed có 0 điểm chặn, và quyết định phát triển hoàn chỉnh đã vượt qua mọi điều kiện. Điều này được ghi nhận là **P3-G2 phát triển GO**, không phải GO chuyển giao đầy đủ, vì tập phân chia kiểm định cũ đã được sử dụng trong quá trình điều tra phục hồi.

### 39.4 Kết quả một lần người ký 10 niêm phong

Đánh giá người ký 10 bất biến đã hoàn thành tất cả 220 đoạn. Cổng chính thức của nó sử dụng `contact_logits + 2.0 * persistence_logits`, chính xác như đã đóng băng trước kiểm thử.

| Điều kiện P3-G2 | Kết quả niêm phong | Yêu cầu | Quyết định |
|---|---:|---:|---|
| mức tăng MAE khoảng cách quan hệ | **-3.81%** | >=10% | **NO-GO** |
| F1 tiếp xúc tổng thể | **0.6188** | >=0.65 | **NO-GO** |
| F1 tay--thân thể ký hiệu | **0.5669** | >=0.60 | **NO-GO** |
| độ chính xác thứ tự chiều sâu | **97.45%** | >=80% | GO |
| mức tăng trượt so với không-duy-trì | **13.92%** | >=15% | **NO-GO** |
| bộ so sánh trượt sẵn có | true | true | GO |
| suy giảm tái tạo tối đa | **0.00%** | <=1% | GO |
| tái tạo chỉ-quan-hệ không đổi | true | true | GO |

Chẩn đoán chỉ-tiếp-xúc (tắt hợp nhất duy trì) đạt F1 ký hiệu 0.6108, nhưng nó không phải là điểm số cổng khai báo trước và không thể thay thế nó sau khi quan sát kiểm thử. MAE tay--tay đồ thị là 0.03331 m so với 0.03209 m cho MLP hình học đóng băng, giải thích cho mức tăng quan hệ âm. Cùng với việc giảm tiếp xúc và trượt từ phát triển sang kiểm thử, điều này xác định một sự dịch chuyển hình học người ký/miền và thất bại chuyển giao hợp nhất duy trì hơn là một bug optimizer còn lại.

**Trạng thái P3-G2 cuối cùng: chuyển giao niêm phong NO-GO.** Năng lượng tiếp xúc và tất cả sự thu hút, duy trì và rằng buộc liên kết lực do tiếp xúc thúc đẩy tiếp tục bị tắt. Kết quả P3-G3 GO chỉ-hình-học trước đó vẫn giữ nguyên hợp lệ và có thể tiếp tục phía hạ nguồn theo dự phòng đã tài liệu hóa.

### 39.5 Thử nghiệm phục hồi hợp lệ tiếp theo

Người ký 10 giờ đây đã bị tiêu thụ và không bao giờ được trở thành một tập tinh chỉnh hay lựa chọn checkpoint. Một đợt thử tiếp theo hợp lý phải tạo ra một quy trình phát triển/kiểm thử mới trước khi huấn luyện:

1. cụ thể hóa các quan sát thân/tay PHOENIX và các mục tiêu tiếp xúc bề mặt lưới, sau đó phân chia theo người ký và thành phần video nguồn kết nối;
2. dự trữ một thành phần làm đợt kiểm toán cuối cùng niêm phong mới và chỉ sử dụng các thành phần PHOENIX còn lại cho độ đầy đủ đặc trưng và lựa chọn checkpoint;
3. chuẩn hóa các thặng dư 2D theo tỷ lệ thân và cắt xén camera, tăng cường nhiễu camera/cắt xén, và yêu cầu các probe chỉ-huấn-luyện vượt qua F1 >=0.65 với cả độ chính xác và độ gợi nhớ >=0.60 trên thành phần phát triển mới;
4. tiền huấn luyện các nhánh quan sát/duy trì chứa đựng qua How2Sign và PHOENIX, giữ hình học quan hệ được bỏ qua và đóng băng; chọn hợp nhất và ngưỡng chỉ trên tập phát triển;
5. yêu cầu ba seed phát triển vượt qua vectơ P3-G2 hoàn chỉnh trước khi mở thành phần niêm phong mới một lần.

Không có huấn luyện V8 hay tinh chỉnh V7 sau kiểm thử nào được khởi chạy trong chu kỳ này. Việc dừng này là cố ý: một lần chạy khác được chọn từ phản hồi người ký 10 có thể tạo ra một số trên ngưỡng nhưng sẽ không còn là một kết quả chuyển giao hợp lệ.

### 39.6 Xác minh và các artifact bất biến

Ruff vượt qua cho `phase2_refiner` và `phase3_posterior`; cả hai gói đều biên dịch. Kiểm thử vượt qua **67/67** cho Phase 2 và **42/42** cho Phase 3.

| Artifact | SHA-256 |
|---|---|
| cấu hình V7 | `84a93772a47b74ecd84ea83e30ee21ccd548678aa93c881adff336f477f91d56` |
| checkpoint V7 được chọn (cập nhật 5.000) | `da398ab1aa6399c38705d14d6559150d152a55a2495729292c149aee7d3a840a` |
| kiểm tra trước V7 | `eac48e2ba041b1478bbc9fd5ba6ade3b4a93a6cadf200814a25f281eed4a761b` |
| đánh giá phát triển V7 | `569ab1575787fb0400f1376372c1466334431b9abcff62472951023966be9b2c` |
| quyết định phát triển V7 | `d3eb3927a037d8794b17f965f76b31ab2f4655a6b2dc99673985c77b74bbb5ba` |
| bản kê khai huấn luyện mở rộng | `eeda81b36c79eafa6dcb6c7fecac89380c2bb7b44ba548166b7e18801ca5c4ca` |
| bản kê khai phát triển mở rộng | `6827f7148a4e28ca35b735e075e08ddff49692cc4267da6e0e81b55e8cd46ef6` |
| bản kê khai người ký 10 niêm phong | `17921e781ccd586a198f47399a120d43c608222172d916848b7d2370a93f3fb7` |
| bản kê khai cache quan hệ người ký 10 | `056ec93ef42b2ed8f1e46fc1cdf4fa3b80442748aa26a6ddbc078af5e7eada17` |
| báo cáo chiếu lại người ký 10 | `908bee45f972e9f5d42d857b510bf461464e05ebb1592341d786fb8b6fcc5723` |
| đánh giá niêm phong | `cfc02fd8d829dde6a1628cdff3f20b92595ae5fb66de970a19511826d3df45d7` |
| quyết định chính thức niêm phong | `ebb46861f75c7b2a729b7e0cd64bd1e394aed9bd26f349dac8934bb676f2a9e6` |
| nhật ký chỉ-thêm niêm phong | `9438b389e3cd17cc9aa4962de4cc9ceaa5e1b55786927ec52e1e0a3df4b708cc` |
