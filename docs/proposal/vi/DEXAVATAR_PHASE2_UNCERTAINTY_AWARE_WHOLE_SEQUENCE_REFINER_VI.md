# Kế hoạch Xây dựng Phase 2: Bộ Tinh chỉnh Toàn Chuỗi Nhận thức Độ không Chắc chắn (Uncertainty-Aware Whole-Sequence Refiner)

- **Dự án:** DexAvatar / Chương trình giảm thiểu rủi ro SignPosterior4D
- **Giai đoạn:** Chỉ Phase 2
- **Tên phương pháp:** `UAWSR` (Uncertainty-Aware Whole-Sequence Refiner - Bộ tinh chỉnh toàn chuỗi nhận thức độ không chắc chắn)
- **Ngày:** 22 tháng 7 năm 2026
- **Mục tiêu chính:** Xác định liệu một mô hình chuỗi hai chiều, xác định (deterministic, bidirectional sequence model) có thể cải thiện bộ khởi tạo đóng đóng đóng (frozen initializer) mạnh mẽ của DexAvatar bằng cách sửa chữa các thất bại tạm thời về thân, cổ tay và bàn tay mà không làm phẳng mịn quá mức (oversmoothing) các chuyển động ký hiệu hợp lệ hay không.

---

## 1. Quyết định điều hành

Xây dựng Phase 2 dưới dạng một **bộ tinh chỉnh thặng dư xác định độc lập (standalone deterministic residual refiner) hoạt động trên các quan sát đóng băng đã được lưu vào bộ nhớ đệm (cached frozen observations)**. Mô hình sẽ tiêu thụ một đoạn video từ ký hiệu cô lập (isolated-sign clip) hoàn chỉnh, lập luận đồng thời trên thân trên, cả hai cổ tay và cả hai bàn tay, ước tính mức độ tin cậy của mỗi quan sát, và xuất ra một chuỗi tham số và lưới (mesh) SMPL-X hoàn chỉnh.

**Không** thêm khuếch tán (diffusion), lấy mẫu đa giả thuyết (multi-hypothesis sampling), ngữ âm học (phonology), tiếp xúc học được (learned contact), hoặc dự đoán pha (phase prediction) trong giai đoạn này. Các cơ chế đó thuộc về giai đoạn sau khi bộ tinh chỉnh xác định chứng minh được rằng tín hiệu thời gian có ích thực sự tồn tại vượt xa các chuyên gia theo từng khung hình (framewise experts) mạnh mẽ hơn.

Chuỗi Phase 2 bắt buộc phải tuân theo:

```text
Đoạn RGB clip
  -> các quan sát đóng băng từ SMPLer-X/NLF, WiLoR và Sapiens
  -> bộ nhớ đệm quan sát phân phiên bản với mặt nạ dữ liệu thiếu
  -> chuẩn hóa tọa độ và góc xoay (canonicalization)
  -> bộ tinh chỉnh khớp Thân-Cổ tay-Bàn tay hai chiều xác định
  -> hợp nhất thặng dư dựa trên cổng độ không chắc chắn trên SO(3)
  -> giải mã SMPL-X khả vi (differentiable SMPL-X decoding)
  -> tinh chỉnh chuỗi tùy chọn chỉ dựa trên quan sát ngắn
  -> kết quả các file PKL + lưới mesh
  -> manifest đánh giá theo phong cách tác giả đã khóa và các bài kiểm tra chẩn đoán
```

Giai đoạn này chỉ được coi là thành công nếu sự cải thiện diễn ra trên không gian, được hỗ trợ về mặt thống kê, mạnh mẽ trước thất bại của quan sát, và đạt được trên cùng một quần thể khung hình exact như bộ khởi tạo của nó. Tốc độ thấp hơn hoặc giật (jerk) thấp hơn đơn thuần không được tính là thành công.

---

## 2. Các thực tế repository ràng buộc thiết kế

### 2.1 Tham chiếu đã công bố và các đo lường địa phương không thể thay thế cho nhau

Tham chiếu DexAvatar đã công bố là:

| Phương pháp | UBody(-F) | LHand | RHand |
|---|---:|---:|---:|
| DexAvatar (đã công bố) | 30.13 | 13.53 | 13.08 |

Giữ đây làm tham chiếu bên ngoài. Không tuyên bố ngang hàng (parity) hoặc cải thiện từ kết quả tổng hợp địa phương cho đến khi việc căn chỉnh và giao thức khung hình (frame protocol) được đối soát hoàn toàn.

Kiểm toán repository trong `docs/dexavatar_diagnosis/E0_PHASE_REPORT.md` phát hiện luồng địa phương kiểu tác giả tạo ra:

- 57 ký hiệu;
- 1,493 cặp GT/dự đoán thứ tự, không phải 2,872 khung hình như trong bài báo phát biểu;
- 1,163 cặp LHand trên 42 ký hiệu vì các ký hiệu class-0 bỏ qua bàn tay trái;
- 1,493 cặp RHand trên tất cả 57 ký hiệu; và
- mặt nạ tác giả chính xác gồm 7,279 / 778 / 778 đỉnh tương ứng cho UBody(-F), LHand, và RHand.

Manifest so sánh địa phương bất biến là `probes/results/phase0/frame_manifest.csv`. Phase 2 phải sử dụng cùng quần thể này cho bộ khởi tạo và kết quả đã tinh chỉnh. Một phương pháp thiếu khung hình sẽ thất bại ở cổng độ bao phủ (coverage gate); nó không được phép so sánh bằng cách âm thầm cắt ngắn cả hai danh sách với `min(...)`.

Bộ đánh giá địa phương căn trung tâm tọa độ dịch chuyển (translation-center) một cách độc lập cho từng vùng được đánh giá. Báo cáo điều này dưới dạng **repository-local author-style regional TR-V2V** cho đến khi tính tương đương với giao thức chính thức dự kiến được chứng minh.

### 2.2 Các artifact hiện có hữu ích

Cây thư mục hiện tại đã cung cấp:

- các khung hình nguồn tại `data/frames/<sign>/`;
- ground truth SGNify chỉ chứa mesh tại `data/smplx_gt/<sign>/`;
- các asset đánh giá của tác giả tại `data/evaluation_from_author/data/data/`;
- tái tạo tương thích gốc/HaMeR trong `outputs/method_hamer/`;
- các thử nghiệm WiLoR và NLF mạnh hơn trong `outputs/method_nlf_wilor/`;
- phát hiện 2D toàn thân Sapiens trong các file `sapiens.pkl` theo từng ký hiệu;
- các PKL khởi tạo SMPLer-X hoặc NLF SMPL-X;
- đầu ra thô của WiLoR trong `wilor/wilor.pkl` và bản xuất tương thích HaMeR trong `hamer/hamer.pkl`;
- các PKL kết quả đã fitting chứa tư thế thân 63-D và tư thế trục-góc bàn tay 45-D; và
- một GPU NVIDIA RTX 5880 Ada 48 GB duy nhất, đủ cho mô hình xác định được đề xuất.

Hợp đồng đầu vào chính phải là **bộ khởi tạo đóng băng mạnh hơn của Phase 1**, chứ không phải mặc định giả định `method_hamer`. `method_hamer` vẫn là A0, tham chiếu tương thích DexAvatar lịch sử. Chỉ chọn bộ khởi tạo Phase 1 sau khi đánh giá trên manifest chung xác nhận hình học và độ bao phủ của nó.

### 2.3 Prototype thời gian hiện tại không được phép mở rộng thành phương pháp chính

`dexavatar_fitting/smplifyx/fit_temporal_window.py` có ích như một baseline làm mịn đơn giản / thất bại, nhưng nó không phải là Phase 2 vì:

- nó tối ưu hóa các mã ẩn VAE tách biệt theo từng khung hình thay vì học một phép hiệu chỉnh chuỗi;
- áp dụng phạt tốc độ, gia tốc và giật (jerk) chỉ đối với tư thế thân;
- không tinh chỉnh thời gian cho cả hai bàn tay;
- khởi tạo mọi mã ẩn tại 0 thay vì chuỗi đã được fit;
- lấy trung bình trực tiếp các vector trục-góc, điều này không phải là lấy trung bình góc xoay hợp lệ;
- không dựng độ không chắc chắn đặc thù cho từng quan sát;
- kế thừa lọc khung hình loại bỏ các khung hình thiếu phát hiện bàn tay;
- sử dụng một tập hợp con giản lược của các số hạng fitting ban đầu;
- ghi các dictionary kết quả nhưng không hoàn thành hợp đồng render lưới chuẩn; và
- đã tạo ra 0 PKL kết quả và 0 mesh dưới `outputs/output_wilor_temporal/` trong checkout hiện tại.

Giữ nó làm baseline `B2: prototype cửa sổ thời gian hiện có`. Triển khai UAWSR trong một module mới để dữ liệu, huấn luyện và hành vi suy luận của nó có thể kiểm thử độc lập.

### 2.4 Xử lý quan sát hiện tại phá hủy thông tin cần thiết cho độ không chắc chắn

Hành vi của `data_parser.py` hiện tại không phù hợp với cache mới:

- các khung hình không có phát hiện bàn tay yêu cầu sẽ bị loại bỏ;
- đối với ký hiệu hai tay, cả hai bàn tay đều phải được phát hiện;
- độ tin cậy điểm đặc trưng bàn tay của WiLoR bị thay thế bằng `1` sau khi chèn;
- một quan sát một tay bị thiếu có thể được sao chép từ khung hình trước đó;
- sự vắng mặt của chuyên gia và các quan sát được sao chép không được lộ ra cho hàm tổn thất; và
- tùy chọn fitting "nhận thức độ không chắc chắn" hiện tại thu phóng các số hạng bàn tay chỉ bằng độ tin cậy cổ tay của Sapiens.

Phase 2 phải giữ lại mọi khung hình đã lên lịch và đại diện cho sự vắng mặt một cách rõ ràng. Việc thiếu bàn tay chính là trường hợp mà mô hình toàn chuỗi được thiết kế để giải quyết.

### 2.5 Tập dữ liệu huấn luyện ký hiệu hiện đang staged không phải là một tập dữ liệu chuỗi

`data/body_data/sign_v1` chỉ chứa 1,449 / 181 / 182 tư thế thân 63-D theo từng khung hình. Việc staging How2Sign địa phương chứa khoảng 100 đoạn ngắn khoảng 10 khung hình. Staging PHOENIX hiện tại chỉ có phần train split và các nhóm trích xuất ngắn. Các asset này hữu ích cho các bài kiểm tra khói (smoke tests) và khởi tạo không gian, nhưng không đủ bằng chứng cho một bộ tinh chỉnh chung thân-bàn tay 32–64 khung hình.

Thu thập hoặc xây dựng một tập dữ liệu chuỗi rời rạc về nguồn gốc (source-disjoint sequence corpus) do đó là một cổng dữ liệu rõ ràng của Phase 2 (data gate), không phải là một cải tiến tùy chọn.

---

## 3. Phạm vi và mục tiêu loại trừ của Phase 2

### 3.1 Trong phạm vi (In scope)

- các chuyên gia theo khung hình đóng băng, đã được lưu đệm;
- nạp chuỗi tất cả các khung hình với thông tin thiếu rõ ràng;
- trạng thái thống nhất thân, cổ tay, và hai bàn tay;
- suy luận toàn bộ clip hai chiều hoặc cửa sổ được đệm;
- hiệu chỉnh góc xoay thặng dư xác định (deterministic residual rotation correction);
- baseline độ tin cậy heuristic cố định (fixed heuristic reliability baseline);
- độ tin cậy dị sai học được (learned heteroscedastic reliability) sau khi vượt qua các cổng hiệu chuẩn;
- biến dạng do lỗi bộ ước lượng thực tế và che khuất bùng nổ (burst-occlusion corruption);
- tổn thất duy trì chuyển động và khớp/đỉnh/đầu ngón tay/lòng bàn tay;
- xuất PKL SMPL-X và mesh hợp lệ;
- đánh giá TR-V2V, thời gian, sự nhiễu loạn và hiệu chuẩn trên manifest chung; và
- tự động lùi về (fallback) bộ khởi tạo khi bộ tinh chỉnh thất bại ở bài kiểm tra an toàn lúc suy luận.

### 3.2 Ngoài phạm vi một cách rõ ràng (Out of scope)

- lấy mẫu chuỗi dựa trên khuếch tán (diffusion) hoặc dựa trên score;
- lựa chọn ứng viên best-of-K hoặc oracle;
- điều kiện hóa theo gloss, HamNoSys, hoặc ngữ âm học;
- các lớp pha ký hiệu học được;
- dự đoán tiếp xúc dày đặc (dense contact prediction);
- thay đổi hoặc fine-tune WiLoR/SMPLer-X/NLF cho kết quả chính;
- huấn luyện trên các lưới đánh giá SGNify;
- chọn siêu tham số từ benchmark 57 ký hiệu;
- tuyên bố rằng độ đo địa phương hiện tại tái tạo lại chỉ số đã công bố `30.13 / 13.53 / 13.08`; và
- sửa đổi `evaluation/evaluate_new_fitting_local.py` để làm cho kết quả trông đẹp hơn.

### 3.3 Giả thuyết khoa học

> Khi cho cùng các quan sát đóng băng như một bộ khởi tạo framewise mạnh mẽ, một bộ tinh chỉnh hai chiều thân-cổ tay-hai bàn tay được huấn luyện trên các thất bại ước lượng thực tế sẽ giảm lỗi tái tạo không gian vì các quan sát rõ ràng trước và sau một thất bại sẽ ràng buộc khoảng thời gian mơ hồ. Độ tin cậy rõ ràng sẽ cải thiện sự hiệu chỉnh này bằng cách ngăn các quan sát tự tin nhưng không nhất quán chi phối chuỗi.

Giả thuyết không (null hypothesis) là bộ khởi tạo đã chứa tất cả thông tin có thể phục hồi và mô hình hóa thời gian chỉ thay đổi độ mượt. Kế hoạch thử nghiệm được thiết kế để chấp nhận giả thuyết không đó và dừng lại trước Phase 3 nếu cần thiết.

---

## 4. Hợp đồng hệ thống (System contract)

### 4.1 Đầu vào

Với mỗi clip và khung hình `t` đã được lên lịch, lưu bộ nhớ đệm:

1. **Danh tính khung hình**
   - `clip_id`, số khung hình nguồn, đường dẫn, nhãn thời gian, tốc độ lấy mẫu;
   - chiều rộng và chiều cao ảnh gốc;
   - SHA-256 của khung hình và mỗi checkpoint/config chuyên gia.

2. **SMPL-X của bộ khởi tạo**
   - `global_orient [3]`, `body_pose [63]`;
   - `left_hand_pose [45]`, `right_hand_pose [45]`;
   - `betas [10]`, `transl [3]`, thông số nội camera;
   - tên chuyên gia nguồn và liệu một trường có bị thay thế từ chuyên gia khác hay không.

3. **Quan sát Sapiens**
   - 133 khớp 2D và độ tin cậy gốc;
   - mặt nạ tính hợp lệ và trong khung hình.

4. **Quan sát WiLoR cho mọi bàn tay được phát hiện**
   - Khớp 2D và 3D, khớp nối MANO, giá trị camera toàn cục/crop;
   - tính chất thuận tay (handedness), tâm/kích thước hộp khung (box), độ tin cậy bộ phát hiện, và việc cắt xén crop;
   - thông tin ứng viên trùng lặp thay vì chỉ ứng viên đầu tiên cho mỗi bên.

5. **Quan sát dẫn xuất**
   - Khớp/đỉnh SMPL-X từ bộ khởi tạo đóng băng;
   - khớp bàn tay 3D tương đối với cổ tay trong một hệ tọa độ được ghi nhận tài liệu;
   - đầu ngón tay, tâm MCP, tâm lòng bàn tay, pháp tuyến lòng bàn tay;
   - thặng dư chiếu lại 2D (reprojection residuals);
   - biến đổi xoay và vị trí khớp giữa các khung hình (innovations); và
   - mặt nạ chuyên gia hiện diện, bị sao chép, nội suy, và bị thiếu.

Các đặc trưng RGB hoặc crop là thử nghiệm tùy chọn của Phase 2.5. Mô hình Phase 2 đầu tiên được chấp nhận phải chứng minh giá trị bằng hình học đã lưu đệm, vì điều này cô lập việc suy luận chuỗi khỏi một bộ mã hóa hình ảnh mới được huấn luyện.

### 4.2 Đầu ra

Với mỗi khung hình đầu vào, ghi:

- các trường và hình dạng PKL kết quả DexAvatar chuẩn;
- `refiner_delta_rotvec` cho 51 khớp được tinh chỉnh;
- `uncertainty` và `fallback_mask` theo nhóm thân/bàn tay trái/bàn tay phải;
- một OBJ SMPL-X 10,475 đỉnh hợp lệ;
- JSON chẩn đoán theo từng khung hình; và
- một báo cáo suy luận cấp clip chứa độ bao phủ, thời gian chạy, NaNs, lùi an toàn, và các hash bộ nhớ đệm/mô hình.

PKL kết quả thông thường phải duy trì khả năng tiêu thụ bởi bộ renderer hiện có:

| Trường | Hình dạng yêu cầu | Hành vi Phase 2 |
|---|---:|---|
| `body_pose` | `(1, 63)` | được tinh chỉnh cho các khớp thân trên; giữ nguyên cho các khớp thân dưới |
| `left_hand_pose` | `(1, 45)` | được tinh chỉnh hoặc giữ an toàn từ bộ khởi tạo |
| `right_hand_pose` | `(1, 45)` | được tinh chỉnh hoặc giữ an toàn từ bộ khởi tạo |
| `betas` | `(1, 10)` | một giá trị clip dùng chung mạnh mẽ |
| `global_orient` | `(1, 3)` | đóng băng trong mô hình đầu tiên; ablation hiệu chỉnh tần số thấp tùy chọn |
| `transl` | `(1, 3)` | đóng băng trong mô hình đầu tiên |
| các trường khuôn mặt | hình dạng hiện có | được sao chép không đổi |

Không để cập nhật dịch chuyển (translation update) khai thác việc căn trung tâm theo vùng của bộ đánh giá. Sự cải thiện tư thế phải tự đứng vững trên chính nó.

### 4.3 Bố cục bộ nhớ đệm (Cache layout)

Sử dụng bố cục phân phiên bản, chỉ thêm (append-only):

```text
cache/phase2/v1/
  manifest.json
  splits/{train,val,test}.json
  clips/<source>/<clip_id>.npz
  diagnostics/<source>/<clip_id>.json
```

Mỗi thay đổi schema bộ nhớ đệm sẽ tăng số phiên bản. Không bao giờ ghi đè lên bộ nhớ đệm tại chỗ. Manifest ghi lại giấy phép nguồn, phân tách nguồn, người ký, ngôn ngữ, hash commit/checkpoint chuyên gia, đơn vị, tọa độ, FPS, và lệnh tiền xử lý.

---

## 5. Trạng thái chuẩn hóa và chính sách hệ tọa độ

### 5.1 Biểu diễn góc xoay

Các file SMPL-X sử dụng trục-góc (axis-angle), nhưng mạng thần kinh không nên hồi quy hoặc lấy trung bình các giá trị trục-góc thô.

Đối với mỗi trong số 21 khớp thân và 30 khớp bàn tay:

1. chuyển đổi trục-góc của bộ khởi tạo thành ma trận xoay;
2. mã hóa ma trận dưới dạng đầu vào xoay 6D liên tục;
3. dự đoán một thặng dư cục bộ `delta_omega in R^3`;
4. kẹp (clamp) biên độ thặng dư bằng một biên khả vi; và
5. hợp nhất các góc xoay trên đa thức (manifold):

$$
R^{out}_{t,j} = \exp(\alpha_{t,j}\,\Delta\omega_{t,j})R^{init}_{t,j}.
$$

`alpha` là một cổng tin cậy quan sát/tái tạo trong `[0,1]`. Chỉ chuyển đổi sang trục-góc khi xuất file hoặc gọi SMPL-X.

Đối với các cửa sổ chồng lặp, hòa trộn (blend) các ma trận xoay bằng cách căn chỉnh bán cầu quaternion (quaternion hemisphere alignment) sau đó lấy trung bình có trọng số đã chuẩn hóa, hoặc lấy trung bình trắc địa (geodesic averaging). Không bao giờ lấy trung bình trực tiếp các vector trục-góc.

### 5.2 Tập hợp khớp được tinh chỉnh

Mạng nhận tất cả 51 khớp tư thế nhưng chỉ thay đổi:

- spine1, spine2, spine3;
- cổ và cả hai xương đòn (collars);
- cả hai vai, khuỷu tay, và cổ tay;
- tất cả 15 khớp của mỗi bàn tay.

Đầu có thể là ngữ cảnh đầu vào nhưng ban đầu bị đóng băng. Hông, đầu gối, cổ chân và bàn chân được sao chép từ bộ khởi tạo. Điều này tập trung năng lực vào ba vùng được đánh giá và ngăn chặn sự trôi lệch thân dưới không liên quan.

### 5.3 Các hệ tọa độ

Duy trì ba hệ tọa độ rõ ràng:

- **camera frame:** phép chiếu 2D và đầu ra camera chuyên gia;
- **root/torso frame:** vị trí khớp thân và ngữ cảnh chéo tay;
- **wrist-local frame:** khớp nối bàn tay và hình học lòng bàn tay.

Lưu trữ một phép biến đổi đồng nhất (homogeneous transform) cho mọi chuyển đổi và thêm các bài kiểm tra khứ hồi (round-trip tests). WiLoR 3D không được giả định là tương thích metric với SMPL-X chỉ vì cả hai đều dùng XYZ. Ước tính tỷ lệ cục bộ cổ tay từ chiều dài xương MANO và sử dụng hình học tương đối cổ tay trừ khi có sẵn một sự căn chỉnh tọa độ camera đã được kiểm toán.

Mô hình sẽ nhìn thấy cả hình học bàn tay cục bộ cổ tay và các đặc trưng lòng bàn tay/cổ tay trong khung thân. Điều này cho phép khớp nối ngón tay duy trì ổn định về tỷ lệ trong khi hướng cánh tay và lòng bàn tay vẫn phối hợp nhịp nhàng.

### 5.4 Hình dạng dùng chung (Shared shape)

Tính toán một hình dạng clip từ trung vị mạnh mẽ (robust median) của các betas bộ khởi tạo đóng băng, loại bỏ các điểm ngoại lệ theo khung hình với một ngưỡng đã khai báo trước, và đóng băng nó. Việc học hiệu chỉnh hình dạng nằm ngoài phạm vi vì benchmark chứa một người ký và hình dạng có thể làm lẫn lộn mức tăng tư thế.

---

## 6. Độ không chắc chắn của quan sát

### 6.1 Các đặc trưng độ tin cậy (Reliability features)

Đối với mỗi khung hình và khớp thân/bàn tay, xây dựng các đặc trưng độ tin cậy từ thông tin có sẵn mà không cần GT của benchmark:

- độ tin cậy và sự hiện diện của bộ phát hiện;
- độ tin cậy điểm đặc trưng 2D;
- tỷ lệ hộp khung (bounding-box) và phần diện tích nằm ngoài hình ảnh;
- các phát hiện bàn tay trùng lặp hoặc xung đột tính thuận tay;
- sự bất đồng 2D giữa WiLoR và Sapiens tại cổ tay/đầu ngón tay;
- thặng dư chiếu lại từ bộ khởi tạo tới điểm đặc trưng;
- sự không nhất quán về chiều dài xương;
- pháp tuyến lòng bàn tay và biến đổi góc xoay trắc địa từ các láng giềng;
- gia tốc tương đối so với trung vị cục bộ mạnh mẽ;
- điểm mờ (blur score) và độ phân giải crop, nếu được tính toán xác định; và
- liệu giá trị là gốc, bị sao chép, nội suy, hay bị thiếu.

Bộ xuất WiLoR hiện tại không giữ lại độ tin cậy phát hiện YOLO trong bản ghi bàn tay thô. Hãy thêm nó vào bộ nhớ đệm/bộ xuất mới trước khi huấn luyện đầu độ tin cậy học được. Không tái tạo lại nó sau đó từ một giá trị không có sẵn.

### 6.2 Hai chế độ độ không chắc chắn

Triển khai độ không chắc chắn theo hai chế độ có thứ tự.

#### U0: Độ tin cậy cố định (fixed reliability)

Sử dụng một ánh xạ xác định được ghi nhận tài liệu từ các đặc trưng trên thành trọng số. Đây là một baseline và cách bảo đảm an toàn (fail-safe), không phải là một tuyên bố độ không chắc chắn được hiệu chuẩn. U0 cho phép mô hình thời gian được kiểm thử trước khi tồn tại một tập hiệu chuẩn đáng tin cậy.

#### U1: Độ tin cậy dị sai học được (learned heteroscedastic reliability)

Một MLP nhỏ dự đoán log phương sai đặc thù theo nhóm cho:

- xoay thân/cánh tay;
- khớp nối bàn tay trái;
- khớp nối bàn tay phải;
- quan sát 2D; và
- quan sát 3D cục bộ cổ tay.

Ràng buộc log phương sai vào một khoảng an toàn về mặt số học và sử dụng nó trong cả cơ chế attention và các hàm tổn thất quan sát. Đối với thặng dư `r`:

$$
L_{NLL} = \frac{\rho(r)}{2\sigma^2} + \frac{1}{2}\log\sigma^2.
$$

Chặn gradient (stop gradients) từ bộ tinh chỉnh vào một đặc trưng độ tin cậy khi đặc trưng đó được tính từ chính dự đoán của bộ tinh chỉnh; nếu không hệ thống có thể hạ tổn thất bằng cách tuyên bố các lỗi của nó là không chắc chắn.

### 6.3 Cổng hiệu chuẩn (Calibration gate)

U1 chỉ thay thế U0 nếu, trên một tập hiệu chuẩn rời rạc về nguồn và người ký:

- tương quan Spearman giữa độ không chắc chắn/lỗi đạt ít nhất `0.35` cho thân và mỗi bàn tay;
- AUROC để phát hiện 10% lỗi tồi tệ nhất đạt ít nhất `0.75` cho cả hai bàn tay và `0.70` cho thân;
- rủi ro giảm đơn điệu khi 10%, 20%, và 30% các quan sát không chắc chắn nhất bị loại bỏ;
- NLL đã hiệu chuẩn cải thiện so với U0 chỉ dùng độ tin cậy của bộ phát hiện; và
- U1 cải thiện việc tái tạo trên các clip kiểm tra bị làm nhiễu mà không suy giảm tập sạch quá 1% ở bất kỳ vùng nào.

Nếu bất kỳ điều kiện nào thất bại, phát hành U0 và báo cáo độ không chắc chắn học được như một kết quả tiêu cực. Không gọi U0 là đã được hiệu chuẩn.

---

## 7. Kiến trúc bộ tinh chỉnh xác định

### 7.1 Mô hình đề xuất

Sử dụng một Transformer không gian-thời gian luôn phiên trên `T x 51` joint tokens:

- cửa sổ huấn luyện tối đa: 64 khung hình;
- kích thước ẩn (hidden size): 256;
- 6 khối luôn phiên (alternating blocks);
- 8 đầu attention;
- tỷ lệ MLP: 4;
- mã hóa vị trí thời gian tương đối;
- embedding khớp/bên/loại học được;
- bias attention độ không chắc chắn và tính hợp lệ rõ ràng; và
- khoảng 15–30 triệu tham số có thể huấn luyện.

Mỗi khối chứa:

1. spatial attention giữa 51 khớp trong một khung hình;
2. temporal attention cho mỗi khớp qua tất cả các khung hình hợp lệ;
3. các token tóm tắt chéo phần (cross-part) cho thân, cánh tay trái, cánh tay phải, bàn tay trái, và bàn tay right; và
4. một cập nhật thặng dư feed-forward.

Phân tích này là khả thi trên GPU 48 GB hiện có và vẫn cho phép sự phối hợp giữa cổ tay/cẳng tay, bàn tay trái/phải, và thân. Sử dụng mặt nạ đệm để các clip gồm 12–48 khung hình được đánh giá có thể được xử lý dưới dạng các chuỗi hoàn chỉnh.

### 7.2 Đầu vào cho mỗi joint token

- xoay của bộ khởi tạo ở dạng 6D;
- vị trí khớp 3D tương đối với thân hoặc cổ tay của bộ khởi tạo;
- vị trí 2D khớp tương ứng được chuẩn hóa theo kích thước ảnh;
- khớp 3D chuyên gia khi có sẵn;
- sai số hữu hạn tốc độ và gia tốc;
- độ tin cậy quan sát, log phương sai cố định/học được;
- mặt nạ hiển thị, trong khung hình, và bị thiếu;
- danh tính chuyên gia và tính thuận tay; và
- ngữ cảnh nhóm như pháp tuyến lòng bàn tay, kích thước crop, và sự bất đồng cổ tay.

Không bao giờ mã hóa một giá trị thiếu dưới dạng số 0 mà không có một mặt nạ hợp lệ riêng biệt.

### 7.3 Đầu ra

Mô hình dự đoán:

- các vector xoay thặng dư 3D cho các khớp được chọn;
- một cổng áp dụng thặng dư `alpha` cho mỗi khớp/khung hình;
- vị trí khớp thân/cổ tay/bàn tay đã sửa dưới dạng đầu ra phụ phụ; và
- log phương sai U1 khi độ không chắc chắn học được được bật.

Khởi tạo lớp thặng dư cuối cùng bằng 0, làm cho mạng ban đầu là một hàm đồng nhất (identity function). Điều này làm giảm mạnh rủi ro huấn luyện sớm phá hủy một bộ khởi tạo mạnh.

### 7.4 Tại sao tinh chỉnh thặng dư được ưu tiên ở đây

Hồi quy chuỗi tuyệt đối sẽ cần học lại các tư thế hợp lệ, danh tính, quy ước camera và điểm mạnh của chuyên gia. Hợp thành thặng dư làm cho giả thuyết hẹp hơn: sửa các khung hình và khớp không nhất quán với toàn bộ clip trong khi giữ lại các quan sát rõ ràng.

---

## 8. Chiến lược dữ liệu huấn luyện

### 8.1 Các tầng dữ liệu (Data tiers)

Sử dụng ba tầng, mỗi tầng có một mục đích riêng biệt.

| Tầng | Mục đích | Nguồn phù hợp | Thuộc tính yêu cầu |
|---|---|---|---|
| A: chuyển động sạch | học cấu trúc thời gian thân-bàn tay | SignAvatars có bản quyền/chuỗi ký hiệu SMPL-X khác; chuỗi toàn thân chất lượng cao | SMPL-X hoàn chỉnh có thứ tự với nguồn gốc rõ ràng |
| B: bàn tay chất lượng cao | học khớp nối và động lực học bàn tay tương tác | InterHand2.6M, ARCTIC, WHIM hoặc dữ liệu MANO có bản quyền khác | hình học bàn tay đáng tin cậy và danh tính thời gian |
| C: thặng dư chuyên gia thực | học bài toán hiệu chỉnh thực tế | video có GT/nhãn giả sạch sau khi chạy đúng các chuyên gia đóng băng | các chuỗi quan sát/mục tiêu ghép cặp |

Chuyển động bàn tay-vật thể chung có thể tiền huấn luyện động lực học bàn tay nhưng phải giảm trọng số trong quá trình thích ứng ký hiệu. Nó không thay thế dữ liệu chuỗi ký hiệu.

### 8.2 Cổng sẵn sàng dữ liệu (Data readiness gate)

Không bắt đầu lần chạy huấn luyện chính cho đến khi tất cả các điều sau là đúng:

- có sẵn ít nhất 10,000 clip ký hiệu không chồng lặp hoặc riêng biệt về nguồn, hoặc ít nhất 250,000 khung hình ký hiệu hợp lệ giữ nguyên ranh giới clip;
- ít nhất 80% các clip huấn luyện được chọn chứa 16 khung hình liên tiếp trở lên;
- trường tư thế thân và cả hai bàn tay 45-D tồn tại cho ít nhất 70% khung hình huấn luyện, với sự thiếu hụt được giữ lại cho phần còn lại;
- tập train/validation/test rời rạc về nguồn, video, và người ký nơi tồn tại ID người ký;
- không có khung hình đánh giá SGNify hoặc mục tiêu dẫn xuất nào xuất hiện trong huấn luyện hoặc validation;
- giấy phép dữ liệu và checkpoint được ghi nhận; và
- kiểm toán thủ công 100 chuỗi được lấy mẫu ngẫu nhiên thấy ít nhất dưới 10% thất bại mục tiêu giả nghiêm trọng.

Nếu cổng này thất bại, giới hạn công việc ở các thử nghiệm tính khả thi làm nhiễu tổng hợp và không diễn giải kết quả như một bộ tinh chỉnh toàn ký hiệu đã được huấn luyện.

### 8.3 Xây dựng mục tiêu (Target construction)

Sử dụng mục tiêu mạnh nhất có sẵn cho mỗi tầng:

- tham số SMPL-X/MANO thực sự nơi được cung cấp;
- tham số từ multi-view hoặc mocap nơi được cấp phép;
- nếu không, một giáo viên offline đóng băng được tinh chỉnh với bằng chứng đa khung hình và lọc chất lượng.

Không huấn luyện một mô hình để ánh xạ một bộ ước lượng framewise quay lại bản sao giống hệt chính nó. Các cặp như vậy dạy tính đồng nhất và không thể thiết lập khả năng hiệu chỉnh. Khi các mục tiêu giả đến từ cùng một họ bộ ước lượng, yêu cầu một tín hiệu tinh chỉnh độc lập như multi-view, chú giải bàn tay mạnh hơn, hoặc sự đồng thuận với một chuyên gia thứ hai.

### 8.4 Bộ nhớ đệm chuyên gia chính xác (Exact-expert cache)

Chạy cùng các phiên bản chuyên gia đóng băng được sử dụng lúc suy luận trên các video huấn luyện Tầng C. Lưu trữ các thất bại thực sự của chúng. Nhiễu Gaussian ngẫu nhiên là bổ sung; nó không thay thế cho thặng dư của bộ ước lượng.

### 8.5 Giáo trình làm nhiễu (Corruption curriculum)

Áp dụng nhiễu loạn vào các quan sát, không bao giờ vào các mục tiêu sạch:

- che (mask) một chuỗi ngón tay trong 2–8 khung hình;
- che một bàn tay hoàn chỉnh trong 4, 8, và 16 khung hình;
- che cả hai bàn tay ngắn ngắt trong quá trình tương tác;
- đưa vào lỗi xoay cổ tay từ 10–45 độ;
- chèn các thặng dư WiLoR và bộ ước lượng thân được lấy mẫu thực nghiệm;
- tráo đổi một giả thuyết bàn tay hoặc lật tính thuận tay với xác suất thấp;
- làm nhiễu hướng lòng bàn tay độc lập với khớp nối;
- giảm độ tin cậy 2D hoặc làm hỏng crop gần ranh giới ảnh;
- đưa vào nhiễu bùng nổ (burst) thay vì chỉ nhiễu khung hình độc lập; và
- để 25–35% các batch hoàn toàn sạch.

Các batch sạch là cần thiết để dạy hành vi đồng nhất (identity behavior) kỳ vọng khi bộ khởi tạo đã chính xác.

---

## 9. Các hàm tổn thất (Losses)

Sử dụng một hàm tổn thất cân bằng đo lường cả độ trung thực của tư thế và hình học ngữ nghĩa của bàn tay:

$$
L =
\lambda_R L_{rot}
+ \lambda_J L_{joint}
+ \lambda_V L_{region-vertex}
+ \lambda_F L_{fingertip}
+ \lambda_P L_{palm}
+ \lambda_O L_{obs-NLL}
+ \lambda_M L_{motion}
+ \lambda_A L_{anchor}
+ \lambda_B L_{biomech}.
$$

### 9.1 Góc xoay và hình học

- `L_rot`: tổn thất xoay trắc địa trên các khớp thân và bàn tay được chọn;
- `L_joint`: lỗi khớp thân tương đối với thân và khớp bàn tay tương đối với cổ tay;
- `L_region-vertex`: tổn thất đỉnh UBody(-F), LHand, và RHand cân bằng qua SMPL-X;
- `L_fingertip`: trọng số bổ sung cho mười đầu ngón tay; và
- `L_palm`: lỗi trắc địa pháp tuyến lòng bàn tay cộng với hướng từ cổ tay đến MCP.

Chuẩn hóa tổn thất các vùng một cách độc lập để 7,279 đỉnh thân trên không áp đảo mỗi bàn tay 778 đỉnh.

### 9.2 Khả năng của quan sát (Observation likelihood)

`L_obs-NLL` so sánh chuỗi được tinh chỉnh với các quan sát 2D/3D hợp lệ được lưu đệm bằng độ tin cậy U0 hoặc U1. Các quan sát thiếu đóng góp không trực tiếp vào số hạng dữ liệu nhưng vẫn đủ điều kiện để suy luận chuỗi.

### 9.3 Duy trì chuyển động (Motion preservation)

Không giảm thiểu tốc độ về 0. Khớp với tốc độ và gia tốc của mục tiêu sạch:

$$
L_{motion} =
\|\Delta J^{out}-\Delta J^{target}\|_1
+ 0.5\|\Delta^2 J^{out}-\Delta^2 J^{target}\|_1.
$$

Tăng trọng số các khung hình chuyển tiếp bằng biên độ chuyển động mục tiêu trong khi huấn luyện. Điều này phạt hiện tượng giật (jitter) và làm mịn quá mức (oversmoothing) một cách đối xứng.

### 9.4 Neo giữ quan sát đáng tin cậy (Reliable-observation anchor)

Đối với các quan sát có độ không chắc chắn thấp, phạt các hiệu chỉnh không cần thiết. Trọng số neo giữ giảm mượt mà theo độ không chắc chắn. Điều này ngăn mô hình thay đổi các bàn tay WiLoR rõ ràng chỉ để thỏa mãn một prior chuyển động.

### 9.5 An toàn sinh cơ học (Biomechanical safety)

Sử dụng các số hạng giới hạn khớp mềm và tính hợp lệ của mesh chỉ như các số hạng điều hòa an toàn (safety regularizers). Chúng không được chi phối tổn thất mục tiêu bàn tay. Việc kẹp cứng (hard clamping) chỉ xảy ra ở lớp an toàn cuối cùng và phải được ghi log.

---

## 10. Giáo trình huấn luyện

### 10.1 Cấu hình tối ưu hóa ban đầu

Sử dụng cấu hình này làm cấu hình tái tạo đầu tiên, sau đó thay đổi từng yếu tố một trên tập validation bên ngoài:

| Mục | Giá trị khởi đầu |
|---|---|
| optimizer | AdamW |
| learning rate | `2e-4` cho UAWSR; `1e-4` cho độ tin cậy U1 |
| weight decay | `0.05`, ngoại trừ tham số norm và bias |
| schedule | 5% linear warm-up, sau đó cosine decay |
| precision | BF16 khi được hỗ trợ; FP32 cho tổn thất hình học SMPL-X nếu cần |
| physical batch | 8 cửa sổ lên tới 64 khung hình |
| gradient accumulation | 4, cho effective batch 32 |
| gradient clipping | global norm `1.0` |
| dropout | `0.1` |
| training length | tối đa 100,000 updates với early stopping |
| model averaging | EMA `0.999`, được đánh giá song song với trọng số thô |
| seeds | ba seed cố định cho các thử nghiệm được chấp nhận |
| checkpoint selection | điểm số validation bên ngoài duy nhất $S_{val}$ |

Gộp các clip theo độ dài để giảm đệm padding. Lấy mẫu ở cấp độ clip, sau đó cân bằng nguồn và nội dung một hand / hai hand; không bao giờ lấy mẫu đều từ các khung hình, điều này sẽ khiến các video dài chi phối. Lưu các checkpoint `last`, `best`, và định kỳ với optimizer, scheduler, scaler, RNG, hash đệm, và config đã resolve.

Chọn `best` bằng một điểm số validation bên ngoài được khai báo trước:

$$
S_{val} = \frac{1}{3}\sum_{r \in \{U,L,R\}}
\frac{E_r^{model}}{E_r^{initializer}}
+ 0.5\sum_r \max\left(0,
\frac{E_r^{model}}{E_r^{initializer}}-1.01\right).
$$

Điều này xử lý ba vùng bình đẳng và phạt một checkpoint đánh đổi một vùng. Không chọn checkpoint từ SGNify Lane L.

### Giai đoạn T0: Hợp đồng và tính đồng nhất

- overfit 4–8 clip ngắn;
- xác minh đầu ra khởi tạo bằng 0 tái tạo chính xác bộ khởi tạo trước khi huấn luyện;
- xác minh tổn thất đạt gần 0 khi đầu vào bằng mục tiêu;
- xác minh padding không thay đổi đầu ra các khung hình hợp lệ; và
- xác minh chuyển đổi SO(3) và xuất khứ hồi.

**Go:** các mesh được xuất tái tạo các mesh đầu vào trong vòng `0.01 mm` lỗi đỉnh trung bình khi tắt thặng dư.

**No-go:** dừng lại và sửa sự không khớp về tọa độ, hình dạng, thứ tự khớp, hoặc renderer.

### Giai đoạn T1: Phục hồi nhiễu tổng hợp

- huấn luyện trên các chuỗi chuyển động sạch với nhiễu bùng nổ (burst corruption);
- chỉ sử dụng U0;
- không có đặc trưng RGB và không có tối ưu hóa cuối cùng.

**Go:** phục hồi ít nhất 30% lỗi đỉnh được chèn cho trường hợp thiếu bàn tay 4/8/16 khung hình, trong khi các đầu vào sạch suy giảm ít hơn 2% ở mọi vùng.

**No-go:** đơn giản hóa mô hình, kiểm tra chất lượng mục tiêu và quy mô nhiễu, không bắt đầu huấn luyện thặng dư thực tế.

### Giai đoạn T2: Học thặng dư thực tế

- chạy các chuyên gia đóng băng chính xác trên các video Tầng C;
- huấn luyện hiệu chỉnh thặng dư từ quan sát sang sạch;
- trộn 50% thặng dư thực, 25% nhiễu bùng nổ tổng hợp, và 25% batch sạch ban đầu;
- tinh chỉnh hỗn hợp chỉ trên validation bên ngoài.

**Go:** cải thiện lỗi vùng trọng số trên validation sạch bên ngoài ít nhất 3%, không có vùng nào tệ hơn quá 1%, và cải thiện tập hợp thất bại được định nghĩa trước ít nhất 8%.

**No-go:** xác định xem thất bại đến từ chuyển đổi tọa độ, nhiễu mục tiêu, hay sự thiếu hụt ngữ cảnh tương lai có thể phục hồi. Chưa thêm độ không chắc chắn học được.

### Giai đoạn T3: Thích ứng ký hiệu (Sign adaptation)

- fine-tune trên các chuỗi ký hiệu rời rạc về nguồn;
- giữ lại 20–30% batch bàn tay chung/chất lượng cao để tránh trôi lệch bàn tay nghiêm trọng;
- sử dụng learning rate thấp hơn và dừng sớm (early stopping) trên validation bên ngoài.

**Go:** validation ký hiệu cải thiện so với T2 và lỗi chuyển tiếp không suy giảm.

**No-go:** giữ T2 làm Phase 2 và báo cáo rằng các mục tiêu giả ký hiệu hiện có không giúp ích.

### Giai đoạn T4: Độ không chắc chắn học được

- huấn luyện đầu độ tin cậy trên một phân tách hiệu chuẩn rời rạc;
- đóng băng bộ tinh chỉnh trước, sau đó fine-tune chung ngắn nếu hiệu chuẩn vẫn hợp lệ;
- hiệu chuẩn thang đo nhiệt độ/phương sai chỉ được fit trên tập hiệu chuẩn.

**Go:** vượt qua mọi tiêu chí hiệu chuẩn trong Phần 6.3 và đánh bại U0 trên tập hợp thất bại.

**No-go:** phát hành U0.

### Giai đoạn T5: Tinh chỉnh chuỗi ngắn tùy chọn (Optional short sequence optimization)

Bắt đầu từ đầu ra UAWSR trực tiếp, chạy tối đa 10–20 bước Adam trên toàn bộ clip bằng cách sử dụng khả năng quan sát hợp lệ, neo đầu ra đáng tin cậy, và sinh cơ học mềm. Hình dạng, dịch chuyển toàn cục, và khuôn mặt vẫn bị đóng băng.

**Go:** validation trên manifest chung cải thiện ít nhất 0.2 mm ở một vùng mà không có sự suy giảm vùng nào quá 0.1 mm và thời gian chạy vẫn chấp nhận được.

**No-go:** sử dụng đầu ra UAWSR trực tiếp làm phương pháp Phase 2 cuối cùng.

---

## 11. Thuật toán suy luận

Đối với một ký hiệu cô lập:

1. liệt kê tất cả các khung hình đã lên lịch từ clip, không chỉ các khung hình có bàn tay thành công;
2. tải hoặc tính toán bộ nhớ đệm quan sát đóng băng;
3. xác minh hash checkpoint/config và metadata tọa độ;
4. chọn bộ khởi tạo Phase 1 và xây dựng hình dạng clip dùng chung;
5. chuyển đổi tất cả các góc xoay sang ma trận/6D và dẫn xuất các đặc trưng khớp, lòng bàn tay, và thời gian;
6. tính toán độ tin cậy U0 hoặc phương sai U1 đã hiệu chuẩn;
7. đệm toàn bộ clip lên 64 khung hình và chạy suy luận hai chiều một lần;
8. đối với các clip dài hơn 64, sử dụng 50% chồng lặp và hòa trộn trắc địa;
9. hợp nhất các góc xoay thặng dư có biên với bộ khởi tạo;
10. chạy tinh chỉnh T5 tùy chọn chỉ khi được bật bởi một config đóng băng;
11. áp dụng kiểm tra an toàn lúc suy luận và lùi an toàn theo nhóm;
12. giải mã SMPL-X, lưu các PKL chuẩn, render mesh, và ghi chẩn đoán; và
13. khẳng định rằng các ID khung hình đầu ra khớp chính xác với manifest đầu vào đã khóa.

Đối với benchmark đã kiểm toán, độ dài clip là 12–48 khung hình ghép cặp với trung vị 25, vì vậy một mô hình 64 khung hình có thể xử lý mọi ký hiệu dưới dạng một chuỗi hoàn chỉnh.

### 11.1 An toàn và lùi về (Safety and fallback)

Lùi về bộ khởi tạo đóng băng cho một nhóm thân hoặc bàn tay nếu xảy ra bất kỳ điều nào sau đây:

- NaN/Inf trong tham số, đỉnh, hoặc độ không chắc chắn;
- góc thặng dư vượt quá biên validation đóng băng;
- lỗi điểm đặc trưng đáng tin cậy được chiếu kém đi vượt quá độ dung lượng khai báo trước;
- chiều dài xương hoặc hình học lòng bàn tay trở nên không hợp lệ;
- mesh đầu ra thay đổi topolgy; hoặc
- độ không chắc chắn nằm ngoài phạm vi huấn luyện đã hiệu chuẩn.

Lùi về diễn ra theo từng nhóm nếu có thể, vì vậy một hiệu chỉnh bàn tay trái thất bại không bỏ đi một hiệu chỉnh bàn tay phải hoặc thân hữu ích. Mọi trường hợp lùi về đều được đếm và báo cáo. Nhiều hơn 1% lùi về nhóm-khung hình trên validation sạch là một no-go cho bản phát hành.

---

## 12. Thiết kế đánh giá

### 12.1 Các luồng đánh giá (Evaluation lanes)

Duy trì hai luồng được đặt tên rõ ràng.

#### Luồng L: Phát triển địa phương đã khóa (Locked local development)

- sử dụng `probes/results/phase0/frame_manifest.csv`;
- yêu cầu 1,493 cặp UBody/RHand và 1,163 cặp LHand;
- sử dụng mặt nạ tác giả chính xác và hành vi class-0;
- báo cáo repository-local author-style regional TR-V2V;
- sử dụng cùng ID khung hình dự đoán cho bộ khởi tạo và bộ tinh chỉnh; và
- không bao giờ sử dụng `min(GT, prediction)` làm chính sách xử lý đầu ra bị thiếu.

#### Luồng O: So sánh chính thức/đã công bố (Official/published comparison)

Chỉ mở sau khi sự khác biệt 2,872 so với 1,493 và định nghĩa căn chỉnh được giải quyết. Dòng `30.13 / 13.53 / 13.08` thuộc về đây. Cho đến lúc đó, không trộn lẫn các giá trị L và O trong một bảng xếp hạng.

### 12.2 Các baseline bắt buộc

| ID | Cấu hình | Câu hỏi |
|---|---|---|
| A0 | `outputs/method_hamer` | tham chiếu tương thích DexAvatar địa phương lịch sử |
| A1 | bộ khởi tạo đóng băng mạnh hơn Phase 1 được chọn | bao nhiêu phần đến từ việc thay thế/hợp nhất chuyên gia? |
| B1 | bộ khởi tạo + làm mịn Gaussian/Savitzky/tốc độ | việc làm mịn đơn giản có giải thích được mức tăng không? |
| B2 | prototype cửa sổ thời gian hiện có, nếu làm cho chạy được mà không đổi phương pháp | liệu việc tối ưu hóa tham số đơn thuần có giúp ích không? |
| P2.0 | UAWSR, không có đặc trưng độ không chắc chắn | liệu suy luận thặng dư toàn chuỗi có giúp ích không? |
| P2.1 | UAWSR + độ tin cậy cố định U0 | liệu xử lý độ tin cậy/thiếu hụt rõ ràng có giúp ích không? |
| P2.2 | UAWSR + U1 đã hiệu chuẩn | liệu độ không chắc chắn học được có thêm giá trị không? |
| P2.3 | P2.2 + tối ưu hóa T5 tùy chọn | liệu tinh chỉnh quan sát cuối cùng có thêm giá trị không? |

### 12.3 Các thử nghiệm loại trừ bắt buộc (Ablations)

- ngữ cảnh nhân quả/chỉ quá khứ so với hai chiều;
- 8, 16, 32, và 64 khung hình;
- chỉ thân, chỉ bàn tay, và tinh chỉnh chung thân-bàn tay;
- không có đặc trưng cổ tay/lòng bàn tay;
- không có nhiễu bùng nổ;
- quan sát thiếu điền 0 so với mặt nạ thiếu rõ ràng;
- U0 so với chỉ độ tin cậy bộ phát hiện so với U1;
- khớp chuyển động mục tiêu so với làm mịn tốc độ 0;
- chỉ bộ khởi tạo so với các dải biên độ hiệu chỉnh thặng dư;
- toàn bộ chuỗi so với hòa trộn cửa sổ trượt; và
- đầu ra trực tiếp so với tối ưu hóa T5.

### 12.4 Các chỉ số (Metrics)

Chính:

- TR-V2V theo vùng phong cách tác giả địa phương của UBody(-F), LHand, và RHand;
- khác biệt ghép cặp theo từng ký hiệu;
- khoảng tin cậy 95% bootstrap gom nhóm theo ký hiệu; và
- số lượng độ bao phủ/thất bại chính xác.

Chẩn đoán:

- lỗi cổ tay, đầu ngón tay, và pháp tuyến lòng bàn tay trên tập dữ liệu bên ngoài có GT tương thích;
- MPJVE, lỗi gia tốc, và giật (jerk);
- các tập hợp con sạch, mờ, che khuất, thiếu, và bất đồng;
- các tập hợp con chuyển tiếp/tốc độ cao so với tốc độ thấp;
- NLL độ không chắc chắn, độ bao phủ, đường cong rủi ro-độ bao phủ, tương quan Spearman, và AUROC decile tồi nhất;
- phần trăm và nguyên nhân lùi an toàn lúc suy luận; và
- thời gian chạy, bộ nhớ đỉnh, số lượng tham số, và kích thước bộ nhớ đệm.

Các chỉ số thời gian không bao giờ thay thế TR-V2V không gian trong một quyết định tiến/dừng.

### 12.5 Đơn vị thống kê

Bootstrap theo ký hiệu/clip, không theo từng đỉnh hoặc khung hình riêng lẻ. Các khung hình trong một ký hiệu có sự tương quan. Báo cáo trung bình, trung vị, và thay đổi theo từng ký hiệu ở decile tồi nhất bên cạnh tổng hợp đỉnh-khung hình được gộp chung.

---

## 13. Chiến lược tiến/dừng tổng thể (Master go/no-go strategy)

### Cổng G0: Khóa bộ đánh giá và độ bao phủ

**Go khi:** A0 và A1 được render và đánh giá trên một manifest bất biến; ID khung hình, hash, topology, đơn vị, mặt nạ, và sự căn chỉnh được ghi nhận; A1 không thiếu đầu ra.

**No-go khi:** các giá trị tổng hợp thay đổi theo thứ tự file, sự cắt ngắn, tính sẵn có của khung hình, hoặc sự căn chỉnh vùng. Dừng mô hình hóa và sửa hợp đồng bộ đánh giá.

### Cổng G1: Chất lượng bộ khởi tạo Phase 1

**Go khi:** bộ khởi tạo mạnh hơn được chọn có gán nối cổ tay/cẳng tay hợp lệ và cải thiện hoặc duy trì baseline trên manifest chung mà không bị suy giảm vùng đáng kể.

**No-go khi:** chuyển đổi WiLoR/MANO, tính thuận tay, camera, tỷ lệ, hoặc hướng cổ tay không nhất quán. Sửa Phase 1 trước khi huấn luyện thời gian.

### Cổng G2: Độ sẵn sàng của dữ liệu

**Go khi:** Phần 8.2 vượt qua và bộ nhớ đệm quan sát tái tạo xác định.

**No-go khi:** chỉ có tập `sign_v1` framewise hiện tại và các đoạn ngắn địa phương. Chỉ chạy các thử nghiệm tính khả thi; không khởi chạy mô hình chính.

### Cổng G3: Khả năng phục hồi tổng hợp

**Go khi:** T1 phục hồi ít nhất 30% lỗi được chèn và duy trì các clip sạch trong vòng 2% mỗi vùng.

**No-go khi:** mô hình không thể giải quyết các đợt bùng nổ thiếu được kiểm soát. Debug biểu diễn và mục tiêu; cấm thêm sự phức tạp.

### Cổng G4: Giá trị validation thực tế

**Go khi:** trên validation bên ngoài rời rạc về nguồn, lỗi vùng có trọng số cải thiện ít nhất 3%, không có vùng nào tệ hơn quá 1%, và tập hợp khó được định nghĩa trước cải thiện ít nhất 8%.

**No-go khi:** chỉ có các chỉ số độ mượt cải thiện. Xem xét lại căn chỉnh thời gian, chất lượng mục tiêu giả, và tổn thất chuyển động mục tiêu. Không thêm khuếch tán.

### Cổng G5: Tính hợp lệ của độ không chắc chắn

**Go khi:** U1 vượt qua mọi điều kiện hiệu chuẩn và cải thiện tái tạo tập khó so với U0.

**No-go khi:** độ tin cậy chưa được hiệu chuẩn hoặc chỉ theo dõi độ tin cậy bộ phát hiện. Giữ lại U0.

### Cổng G6: Benchmark địa phương đã khóa

Phase 2 chỉ được chấp nhận để chuyển tiếp nếu, so với bộ khởi tạo A1 được chọn trên Luồng L:

- tất cả ba vùng đều có độ bao phủ đầy đủ, giống hệt nhau;
- không có vùng nào suy giảm hơn `0.20 mm` lỗi gộp;
- ít nhất hai trong ba vùng cải thiện với 95% CI gom nhóm theo ký hiệu loại trừ số 0;
- mức cải thiện tương đối có trọng số bằng nhau trên ba vùng đạt ít nhất `3%`;
- tập hợp che khuất/thiếu/bất đồng được định nghĩa trước cải thiện ít nhất `8%`;
- các khung hình sạch có độ không chắc chắn thấp suy giảm ít hơn `1%` ở mọi vùng;
- ít hơn 1% nhóm-khung hình kích hoạt lùi an toàn; và
- kết quả có thể tái tạo qua ba seed, với độ lệch chuẩn vùng dưới `0.20 mm`.

Nếu G6 thất bại, Phase 2 không bào chữa cho relational diffusion hoặc ngữ âm học. Giữ lại phương pháp Phase 1 hình học tốt nhất và báo cáo kết quả thời gian một cách trung thực.

### Cổng G7: So sánh chính thức

**Go khi:** một giao thức được xác minh độc lập tái tạo hoặc giải thích baseline đã công bố và sử dụng quần thể khung hình chính thức chung. Chỉ khi đó mới so sánh Phase 2 với `30.13 / 13.53 / 13.08`.

**No-go khi:** các giao thức đã công bố và địa phương vẫn không thể đối soát. Chỉ báo cáo Luồng L và tránh ngôn ngữ SOTA.

---

## 14. Bảng chuyển hướng theo lỗi quan sát được

| Lỗi quan sát được | Nguyên nhân có khả năng nhất | Chuyển hướng bắt buộc |
|---|---|---|
| việc thiếu tổng hợp không được phục hồi | bug biểu diễn/mô hình hoặc ngữ cảnh thời gian yếu | overfit 1 clip, kiểm tra mặt nạ và hợp thành góc xoay |
| các khung hình sạch trở nên tệ hơn | đầu thặng dư không duy trì tính đồng nhất | output zero-init, tăng cường neo đáng tin cậy, thêm batch sạch |
| bàn tay cải thiện nhưng thân trên tệ đi | gắn nối cổ tay/cẳng tay hoặc mất cân bằng tổn thất | đóng băng thân, giảm cập nhật cross-part, cân bằng lại tổn thất vùng |
| thân trên cải thiện nhưng bàn tay tệ đi | sự áp đảo của đỉnh thân | chuẩn hóa theo từng vùng, tăng tổn thất đầu ngón tay/lòng bàn tay |
| giật thấp nhưng TR-V2V không đổi/tệ hơn | làm mịn quá mức (oversmoothing) | tốc độ/gia tốc mục tiêu, giảm context prior, so sánh B1 |
| các khung hình thiếu biến mất | kế thừa lọc từ `data_parser.py` | xây dựng lại cache loader; sự thiếu hụt phải rõ ràng |
| hành vi trái/phải bất đối xứng | chuyển đổi tính thuận tay hoặc quần thể đánh giá | kiểm toán hướng bên và báo cáo quần thể class-0 riêng biệt |
| độ không chắc chắn tăng ở mọi nơi | lối tắt phình phương sai (variance inflation) | stop-gradient feedback, phương sai bounds, calibration split |
| độ không chắc chắn không có tương quan lỗi | thiếu nhãn/đặc trưng độ tin cậy | giữ U0; thêm score bộ phát hiện/dữ liệu bất đồng |
| validation cải thiện nhưng SGNify thì không | domain shift hoặc không khớp giao thức | kiểm tra tập con từng ký hiệu; không tune trên SGNify |
| kết quả biến động giữa các lần chạy | đệm/thứ tự/huấn luyện không xác định | khóa manifest, seed workers, xuất xác định |
| tối ưu hóa T5 xóa bỏ chuyển động | năng lượng quan sát áp đảo mục tiêu thời gian | tắt T5 hoặc giảm số bước; mô hình trực tiếp vẫn là chính |

---

## 15. Bố cục triển khai

Tạo một package độc lập mới thay vì thêm các nhánh khắp `fitting.py`:

```text
phase2_refiner/
  README.md
  configs/
    uawsr_u0.yaml
    uawsr_u1.yaml
  data/
    cache_schema.py
    build_observation_cache.py
    build_sequence_index.py
    corruptions.py
    dataset.py
  geometry/
    rotations.py
    coordinates.py
    palm.py
    smplx_decode.py
  models/
    embeddings.py
    reliability.py
    spatial_temporal_refiner.py
    heads.py
  losses/
    geometry.py
    motion.py
    uncertainty.py
  train.py
  calibrate.py
  infer.py
  render.py
  evaluate.py
  tests/
```

Gợi ý các artifact được tạo ra:

```text
outputs/phase2_<experiment>/
  <sign>/smplifyx/results/*.pkl
  <sign>/smplifyx/meshes/*.obj
  <sign>/phase2_diagnostics/*.json
  run_manifest.json
  per_frame.csv
  per_sign.csv
  summary.csv
```

Không ghi đè lên `outputs/method_hamer`, `outputs/method_nlf_wilor`, hoặc bất kỳ đầu ra Phase 1 nào.

### 15.1 Hợp đồng lệnh tối thiểu

```bash
python -m phase2_refiner.data.build_observation_cache \
  --frames data/frames \
  --initializer outputs/<phase1_method> \
  --out cache/phase2/v1

python -m phase2_refiner.train \
  --config phase2_refiner/configs/uawsr_u0.yaml

python -m phase2_refiner.infer \
  --config phase2_refiner/configs/uawsr_u0.yaml \
  --cache cache/phase2/v1 \
  --output outputs/phase2_uawsr_u0

python -m phase2_refiner.evaluate \
  --manifest probes/results/phase0/frame_manifest.csv \
  --baseline outputs/<phase1_method> \
  --prediction outputs/phase2_uawsr_u0
```

Mỗi lệnh ghi lại config đã resolve, git SHA, phiên bản các thư viện phụ thuộc, hash checkpoint, hash manifest đầu vào, và random seeds.

---

## 16. Các bài kiểm thử bắt buộc trước khi chạy đầy đủ

### Bài kiểm thử đơn vị (Unit tests)

- chuyển đổi khứ hồi: trục-góc -> ma trận -> 6D -> ma trận -> trục-góc;
- quy ước MANO trái/phải trên một tư thế bất đối xứng đã biết;
- khứ hồi: cục bộ cổ tay -> thân -> camera -> thân;
- dấu pháp tuyến lòng bàn tay và thứ tự khớp đầu ngón tay;
- quan sát thiếu tạo ra các token hữu hạn và tổn thất dữ liệu trực tiếp bằng 0;
- tính bất biến đệm (padding invariance);
- mặt nạ nhân quả so với mặt nạ hai chiều;
- hòa trộn góc xoay chồng lặp qua ranh giới `pi`;
- betas dùng chung qua mọi khung hình đầu ra;
- schema PKL kết quả và topology mesh; và
- bộ đánh giá từ chối một khung hình thiếu hoặc trùng lặp.

### Bài kiểm thử tích hợp (Integration tests)

- luồng đồng nhất trên 1 ký hiệu tái tạo tất cả mesh trong vòng `0.01 mm`;
- một đợt thiếu bàn tay 8 khung hình bị che cố ý được lấp đầy mà không làm thay đổi quá mức các khung hình bàn tay đối diện không bị che;
- các ký hiệu class-0 một tay giữ lại bên thụ động trong quá trình tái tạo mặc dù LHand bị bỏ qua trong độ đo của tác giả;
- suy luận tạo ra cùng một hash đầu ra khi lặp lại với cùng seed/config; và
- một batch 64 khung hình vừa trong bộ nhớ GPU với ít nhất 20% dải an toàn (headroom).

### Bài kiểm thử Red-team

- ứng viên tính thuận tay sai;
- hai phát hiện cho cùng một bên;
- không có bàn tay nào trong các khung hình đầu tiên;
- thiếu bàn tay hoàn toàn trong toàn bộ clip;
- khoảng trống số khung hình;
- cắt xén crop;
- trục-góc cực đoan gần `pi`;
- đầu vào chuyên gia bị NaN;
- độ phân giải hình ảnh không nhất quán; và
- một thư mục đầu ra chứa các mesh thừa cũ.

---

## 17. Các cột mốc và thứ tự dự kiến

| Cột mốc | Công việc | Artifact đầu ra |
|---|---|---|
| M0 | khóa manifest Luồng L và chọn A1 | manifest đánh giá và baseline đã ký |
| M1 | bộ nhớ đệm quan sát với mặt nạ thiếu tất cả khung hình | bộ nhớ đệm v1 cộng báo cáo kiểm toán |
| M2 | luồng đồng nhất hình học/SO(3)/SMPL-X | bài kiểm tra tích hợp tính đồng nhất |
| M3 | khung UAWSR và nhiễu tổng hợp | báo cáo T1 và checkpoint |
| M4 | tập dữ liệu thặng dư thực chuyên gia chính xác | manifest Tầng C và kiểm toán chất lượng |
| M5 | thặng dư thực + thích ứng ký hiệu | các mô hình P2.0/P2.1 và báo cáo validation |
| M6 | nghiên cứu hiệu chuẩn | báo cáo quyết định U0/U1 |
| M7 | hoàn thành benchmark và ablations | bảng Luồng L, CIs, chẩn đoán |
| M8 | T5 tùy chọn và đóng rắn bản phát hành | package Phase 2 cuối cùng |

Thứ tự thực tế trên 1 GPU là 10–14 tuần nếu dữ liệu chuỗi đã được cấp phép và truy cập được. Thu thập dữ liệu, tạo mục tiêu giả, hoặc làm rõ giấy phép nằm ngoài ước tính đó và là một phụ thuộc cứng chứ không phải khoảng trống tiến độ ẩn.

---

## 18. Định nghĩa hoàn thành (Definition of done)

Phase 2 hoàn thành khi:

1. bộ khởi tạo mạnh hơn được chọn và UAWSR chia sẻ một manifest đánh giá bất biến;
2. tất cả các khung hình đã lên lịch, bao gồm cả các thất bại chuyên gia, đi qua mô hình;
3. mô hình tinh chỉnh chung thân trên, cổ tay, và cả hai bàn tay 45-D bằng cách hợp thành SO(3) hợp lệ;
4. U0 có sẵn và U1 chỉ được sử dụng nếu được hiệu chuẩn;
5. các PKL kết quả và mesh 10,475 đỉnh được xuất cho mọi khung hình;
6. các kiểm thử tính đồng nhất, sự thiếu hụt, góc xoay, padding, và đầu ra xác định vượt qua;
7. bảng baseline/ablation đầy đủ được đánh giá với các khoảng tin cậy gom nhóm theo ký hiệu;
8. các tiêu chí G6 chính vượt qua, hoặc một quyết định dừng (no-go) được ghi nhận tài liệu;
9. các điểm số DexAvatar đã công bố không bị trộn lẫn với các điểm số giao thức địa phương chưa giải quyết; và
10. mọi nguồn dữ liệu, chuyên gia, bộ nhớ đệm, config, checkpoint, và đầu ra đều có thể truy xuất nguồn gốc bằng hash.

---

## 19. Khuyến nghị cuối cùng

Phase 2 có xác suất thành công cao nhất được thu hẹp một cách có chủ đích so với đề xuất SignPosterior4D đầy đủ:

> **Các quan sát đóng băng mạnh + sự thiếu hụt rõ ràng + trạng thái thống nhất thân/cổ tay/hai bàn tay đúng tọa độ + tinh chỉnh thặng dư hai chiều xác định + mô hình độ tin cậy qua cổng hiệu chuẩn.**

Giai đoạn này cần trả lời một câu hỏi quyết định trước khi thử nghiệm nghiên cứu phức tạp hơn: **liệu ngữ cảnh toàn bộ ký hiệu có thể sửa chữa các thất bại không gian mà một bộ khởi tạo framewise mạnh không thể sửa hay không?**

Nếu câu trả lời vượt qua G6, hãy tiếp tục sang một nghiên cứu nhỏ về mối quan hệ/tiếp xúc và chỉ sau đó mới tới khuếch tán hoặc điều kiện hóa ngôn ngữ học có cấu trúc. Nếu không, dừng leo thang độ phức tạp của mô hình và quay lại hình học quan sát, chất lượng mục tiêu, và giao thức đánh giá.

---

## 20. Trạng thái triển khai

- **Ngày triển khai:** 22 tháng 7 năm 2026
- **Trạng thái code:** slice dọc đầu tiên có thể thực thi của Phase 2 đã hoàn thành
- **Trạng thái nghiên cứu:** chưa được huấn luyện cho nhiệm vụ cuối cùng; chưa có tuyên bố về độ chính xác hay SOTA

Việc triển khai đã được cố ý thêm vào dưới dạng package `phase2_refiner/` độc lập. Không có file nào dưới `dexavatar_fitting/`, `methods/`, `runners/`, `evaluation/`, hoặc thư mục đầu ra hiện có bị thay đổi. Các phương pháp hiện có vẫn tái sử dụng được và đóng vai trò làm các bộ khởi tạo/baseline chỉ đọc.

### 20.1 Các file đã triển khai

| File | Trách nhiệm đã hoàn thành |
|---|---|
| `phase2_refiner/__init__.py` | point nhập package/phát hành phiên bản |
| `phase2_refiner/config.py` | tải cấu hình YAML |
| `phase2_refiner/data/cache_schema.py` | hợp đồng NPZ schema-v2 tương thích ngược với thời gian, hash, biến đổi tọa độ, hình học 2D/3D có mặt nạ, độ tin cậy, và mục tiêu tùy chọn |
| `phase2_refiner/data/build_observation_cache.py` | chuyển đổi chỉ đọc các artifact kết quả/Sapiens/HaMeR-WiLoR hiện có thành đệm theo từng ký hiệu |
| `phase2_refiner/data/build_sequence_index.py` | manifest train/validation/test/calibration rõ ràng rời rạc về nguồn/người ký |
| `phase2_refiner/data/providers.py` | giao diện trung lập với nhà cung cấp cho các tập dữ liệu quan sát và mục tiêu bên ngoài sau này |
| `phase2_refiner/data/dataset.py` | nạp toàn chuỗi có đệm/cửa sổ, xây dựng đặc trưng, và gom batch |
| `phase2_refiner/data/corruptions.py` | loại bỏ liên tục thân/bàn tay trái/bàn tay phải và nhiễu loạn SO(3) |
| `phase2_refiner/geometry/rotations.py` | các thao tác trục-góc, ma trận, 6D, quaternion, trắc địa, thặng dư có biên, và hợp thành khả vi |
| `phase2_refiner/geometry/coordinates.py` | phép biến đổi đồng nhất đã xác minh và các chuyến đi khứ hồi tọa độ |
| `phase2_refiner/geometry/palm.py` | thứ tự bàn tay cố định, đầu ngón tay, tâm lòng bàn tay, và pháp tuyến nhất quán hướng bên |
| `phase2_refiner/geometry/smplx_decode.py` | hook giải mã chuỗi khả vi cho giám sát hình học/đỉnh |
| `phase2_refiner/models/embeddings.py` | embedding token khớp/nhóm/thời gian/độ tin cậy |
| `phase2_refiner/models/reliability.py` | độ tin cậy U0 cố định và U1 xoay/2D/3D học được |
| `phase2_refiner/models/heads.py` | đầu xoay, cổng, và vị trí khớp an toàn về tính đồng nhất |
| `phase2_refiner/models/pretrained.py` | hook khởi tạo spatial-prior tương thích đã kiểm toán |
| `phase2_refiner/models/spatial_temporal_refiner.py` | Transformer không gian/thời gian/nhóm phân tích 51 khớp với chế độ hai chiều hoặc nhân quả |
| `phase2_refiner/losses/sequence.py` | tổn thất xoay, tốc độ mục tiêu, gia tốc mục tiêu, neo đáng tin cậy, và phương sai dị sai |
| `phase2_refiner/train.py` | huấn luyện AdamW, giáo trình bùng nổ, tích lũy/cắt gradient, validation, và ghi checkpoint |
| `phase2_refiner/infer.py` | suy luận toàn bộ clip và cửa sổ chồng lặp, hòa trộn trắc địa/quaternion, lùi an toàn, xuất PKL, và chẩn đoán |
| `phase2_refiner/render.py` | render mesh SMPL-X tiêu chuẩn và tân mạn neo nguồn |
| `phase2_refiner/evaluate.py` | TR-V2V theo vùng manifest chung nghiêm ngặt, kiểm tra độ bao phủ/topology, đầu ra từng khung hình/ký hiệu, và bootstrap CI |
| `phase2_refiner/calibrate.py` | co giãn phương sai, NLL, Spearman, AUROC decile tồi nhất, và kiểm toán rủi ro-độ bao phủ |
| `phase2_refiner/configs/uawsr_u0.yaml` | cấu hình khởi đầu độ tin cậy cố định |
| `phase2_refiner/configs/uawsr_u1.yaml` | cấu hình khởi đầu độ không chắc chắn học được |
| `phase2_refiner/README.md` | các lệnh cache/train/infer/evaluate/calibrate có thể chạy và hợp đồng không phá hủy |
| `phase2_refiner/tests/` | kiểm thử xoay, đệm, mô hình, nhiễu loạn, gradient huấn luyện, lùi an toàn, hiệu chuẩn, và bộ đánh giá nghiêm ngặt |

### 20.2 Hành vi đã triển khai

- Các bộ khởi tạo hiện có chỉ được đọc. Lệnh đệm và suy luận từ chối ghi đè theo mặc định.
- Bộ nhớ đệm giữ lại độ tin cậy rõ ràng, sự hiện diện chuyên gia, sự thiếu hụt, kích thước crop chuẩn hóa, cắt xén crop, biến đổi thời gian, phát hiện bên trùng lặp, khớp 2D chuẩn hóa, và đường dẫn kết quả nguồn cho mọi khung hình bộ khởi tạo đã lên lịch.
- Trạng thái chứa tất cả 21 xoay thân và cả hai bàn tay 15 khớp. Mặt nạ đầu ra mặc định tinh chỉnh 12 khớp thân trên/cánh tay cộng với tất cả 30 khớp bàn tay và đóng băng các khớp thân dưới.
- Thặng dư được khởi tạo bằng 0, giới hạn ở 25 độ cho thân và 35 độ cho bàn tay, qua cổng, và hợp thành trái trên SO(3).
- Các clip lên tới 64 khung hình chạy trong một lần suy luận hai chiều. Các clip dài hơn sử dụng chồng lặp 50%, trọng số Hann, căn chỉnh bán cầu quaternion, và hòa trộn quaternion đã chuẩn hóa thay vì lấy trung bình trục-góc.
- Các đầu ra không hữu hạn hoặc vượt giới hạn sẽ lùi về độc lập cho thân, bàn tay trái, hoặc bàn tay phải và được đếm trong chẩn đoán.
- PKL đầu ra giữ nguyên các hình dạng DexAvatar hiện có. Chẩn đoán được lưu riêng để không làm hỏng các renderer cũ.
- Render tân mạn neo nguồn (source-anchored rendering) bù đắp cho sự bất khớp quan sát được giữa một số PKL cũ đã lưu và mesh đã lưu của chúng: nó áp dụng dịch chuyển đỉnh (refined-minus-initializer) cùng mô hình vào mesh gốc.
- Bộ đánh giá mới từ chối độ bao phủ không đầy đủ và các mesh thừa cũ thay vì sử dụng cắt ngắn `min(GT, prediction)`.

### 20.3 Validation đã hoàn thành

| Kiểm tra | Kết quả |
|---|---|
| Biên dịch Python | tất cả các module `phase2_refiner` đều biên dịch thành công |
| Kiểm thử tự động | **19 passed** sau các thay đổi căn chỉnh thiết kế/schema-v2 |
| Chuyến đi khứ hồi ma trận xoay/6D/trục-góc | lỗi ma trận tối đa dưới `5e-7` trong bài kiểm toán độc lập |
| Bộ nhớ đệm 1 ký hiệu | Ablehnen: 14 khung hình, trạng thái tư thế `(14, 51, 3)`, trạng thái quan sát `(14, 51, 8)` |
| Bộ nhớ đệm địa phương đầy đủ | **57 ký hiệu và 1,493 khung hình**, khớp với quần thể manifest `method_hamer` đã kiểm toán |
| Suy luận tính đồng nhất đầy đủ | xuất được **1,493/1,493** PKL kết quả |
| Bảo toàn tham số đồng nhất | chênh lệch tối đa trên thân, cả hai tay, hình dạng, gốc, dịch chuyển, hàm, và biểu cảm: **0.0** |
| Hành vi an toàn đồng nhất | **0** lần lùi về nhóm-khung hình trên 57 ký hiệu |
| Đánh giá tính đồng nhất manifest đầy đủ nghiêm ngặt | dự đoán bằng baseline tại **29.907 / 13.573 / 12.927 mm** (thân trên / trái / phải); paired deltas là **0.0 mm** với khoảng bootstrap `[0.0, 0.0] mm` cho mọi vùng |
| Xuất mesh 1 ký hiệu | 14/14 mesh, 10,475 đỉnh, topology mặt không đổi |
| Lỗi mesh đồng nhất neo nguồn | trung bình khoảng `4.1e-6 mm` và tối đa `8.7e-6 mm` trên các khung hình được kiểm tra |
| Smoke test huấn luyện | 1 lần cập nhật optimizer UAWSR đầy đủ hoàn thành và ghi checkpoint 31 MB |
| Smoke test suy luận checkpoint | xuất 14/14 PKL; đường dẫn thặng dư học được là khác 0 và có biên |

Tất cả bộ nhớ đệm, checkpoint, và dự đoán thử nghiệm đều được ghi dưới các thư mục tạm `/tmp/dexavatar_phase2_*`. Không có đầu ra thử nghiệm nào bị trộn lẫn với phương pháp hiện có.

### 20.4 Phát hiện tương thích quan trọng

Một lần chạy forward SMPL-X mới từ một PKL kết quả `method_hamer` hiện có đã khác với mesh đã lưu của chính nó khoảng `0.33–0.47 mm` trung bình trên hai khung hình được kiểm tra, với các điểm cực đại cục bộ lớn hơn, mặc dù các mảng tham số được xuất là giống hệt nhau. Các nguyên nhân có thể bao gồm trạng thái renderer/mô hình lịch sử hoặc sự trôi lệch PKL/mesh đã lưu.

Do đó, việc render lại trực tiếp mọi bộ khởi tạo sẽ làm thay đổi baseline trước khi Phase 2 đưa ra dự đoán. Bộ renderer tân mạn neo nguồn được triển khai sẽ tránh được sự lộn xộn đó:

$$
V^{phase2} = V^{saved-init} +
\left(F_{SMPLX}(X^{phase2}) - F_{SMPLX}(X^{init})\right).
$$

Điều này làm cho thặng dư bằng 0 tái tạo chính xác mesh baseline đã lưu trong khi vẫn áp dụng dịch chuyển đỉnh phụ thuộc tư thế học được. Nó nên tiếp tục là một ablation rõ ràng so với việc render mới trực tiếp.

### 20.5 Các cổng vẫn mở

Triển khai này **không** có nghĩa là Phase 2 đã vượt qua các cổng nghiên cứu của nó.

- **G0 vẫn mở một phần:** code và kiểm tra độ bao phủ đã tồn tại, nhưng sự khác biệt giao thức chính thức 1,493 so với 2,872 vẫn chưa được giải quyết.
- **G1 vẫn mở:** bộ khởi tạo Phase 1 mạnh hơn cuối cùng chưa được chọn trên một manifest chung hoàn chỉnh. Đầu ra `method_nlf_wilor` hiện tại chưa có độ bao phủ 1,493 khung hình đã kiểm toán.
- **G2 là no-go cho huấn luyện chính:** tập hợp mục tiêu toàn chuỗi có bản quyền, rời rạc về nguồn/người ký bắt buộc chưa được chuẩn bị. Dữ liệu `sign_v1` được staged hiện tại vẫn ở dạng framewise và chỉ cho thân.
- **G3–G6 vẫn mở:** checkpoint thử nghiệm 1 bước không phải là mô hình đã huấn luyện và không bao giờ được đánh giá hay báo cáo như độ chính xác của Phase 2.
- Hiệu chuẩn U1 yêu cầu một tập dữ liệu thặng dư rời rạc. Tiện ích hiệu chuẩn được triển khai đã sẵn sàng, nhưng chưa có mô hình U1 đã hiệu chuẩn nào tồn tại.
- Phiên bản bộ nhớ đệm đầu tiên cung cấp các đặc trưng góc xoay, 2D, và độ tin cậy. Các quan sát 3D thân/bàn tay đầy đủ tương thích metric đã kiểm toán, hình học lòng bàn tay, và các đặc trưng RGB/crop vẫn là các bản sửa đổi bộ nhớ đệm tiếp theo.
- Lùi an toàn dựa trên chiếu lại nhóm học được và kiểm tra sinh cơ học vẫn sẽ được thêm vào sau khi có sẵn các quan sát 3D hợp lệ về tọa độ; an toàn hiện tại bao gồm các thất bại về mặt số học và giới hạn góc xoay.

### 20.6 Thứ tự thực thi được ủy quyền tiếp theo

1. hoàn thành G0 và chọn bộ khởi tạo A1 có độ bao phủ đầy đủ;
2. xây dựng `cache/phase2/<a1>_v1` mà không ghi đè A1;
3. thu thập/chuẩn bị một tập mục tiêu chuỗi có bản quyền và vượt qua G2;
4. chạy T0 đồng nhất và T1 phục hồi tổng hợp với 3 seed;
5. dừng lại nếu G3 thất bại;
6. xây dựng các cặp thặng dư thực chuyên gia chính xác và chạy T2/P2.0;
7. so sánh P2.0, U0/P2.1, và làm mịn đơn giản dưới bộ đánh giá nghiêm ngặt;
8. chỉ huấn luyện/hiệu chuẩn U1 sau khi tinh chỉnh xác định vượt qua G4; và
9. render và đánh giá Luồng L chỉ sau khi xác minh độ bao phủ đầy đủ giống hệt nhau.

---

## 21. Báo cáo hoàn thành — 22 Tháng 7, 2026

Triển khai Phase 2 hoàn thành dưới dạng một **nền tảng có thể thực thi độc lập, không phá hủy**. Package `phase2_refiner/` mới được thêm chứa việc xây dựng bộ nhớ đệm, tinh chỉnh chuỗi, huấn luyện, suy luận, render, đánh giá nghiêm ngặt, hiệu chuẩn, cấu hình, tài liệu, và các kiểm thử. Các phương pháp DexAvatar hiện có, code của chúng, thư mục đầu ra của chúng, và bộ đánh giá được cung cấp vẫn giữ nguyên và có thể tái sử dụng như trước.

### 21.1 Các file đã thêm

- `phase2_refiner/data/`: bộ nhớ đệm quan sát phân phiên bản, dataset chuỗi, và giáo trình làm nhiễu tổng hợp;
- `phase2_refiner/geometry/rotations.py`: chuyển đổi SO(3) hợp lệ, biên thặng dư, và hợp thành;
- `phase2_refiner/models/spatial_temporal_refiner.py`: bộ tinh chỉnh thân/cổ tay/hai bàn tay toàn chuỗi 51 khớp;
- `phase2_refiner/losses/sequence.py` và `phase2_refiner/train.py`: các hàm mục tiêu và vòng lặp huấn luyện;
- `phase2_refiner/infer.py` và `phase2_refiner/render.py`: đầu ra PKL tương thích, suy luận cửa sổ chồng lặp, lùi an toàn, và render mesh tân mạn neo nguồn;
- `phase2_refiner/evaluate.py` và `phase2_refiner/calibrate.py`: đánh giá TR-V2V manifest khóa nghiêm ngặt và kiểm toán độ không chắc chắn; và
- `phase2_refiner/tests/`, `phase2_refiner/configs/`, và `phase2_refiner/README.md`: độ bao phủ kiểm thử, cấu hình khởi đầu U0/U1, và các lệnh.

### 21.2 Validation triển khai cuối cùng

| Mục validation | Kết quả cuối cùng |
|---|---|
| Kiểm tra tĩnh | `ruff check phase2_refiner`: passed; biên dịch Python: passed |
| Kiểm thử tự động | **19 passed** |
| Độ bao phủ đệm đã khóa | **57 ký hiệu, 1,493 khung hình** |
| Đầu ra đồng nhất | **1,493/1,493** PKL; chênh lệch tham số tối đa **0.0** |
| An toàn đồng nhất | **0** lần lùi về nhóm thân/trái/phải |
| Tương thích mesh đồng nhất | 10,475 đỉnh và topology không đổi; lỗi neo nguồn khoảng `4.1e-6 mm` trung bình trên các khung hình được kiểm tra |
| Đánh giá nghiêm ngặt manifest đầy đủ | Tính đồng nhất Phase 2 bằng `method_hamer`: **29.907 / 13.573 / 12.927 mm** cho thân trên / tay trái / tay phải |
| So sánh ghép cặp | **0.0 mm** delta ở mọi vùng; khoảng bootstrap 95% là `[0.0, 0.0] mm` |
| Luồng huấn luyện | 1 lần cập nhật optimizer hoàn chỉnh và smoke test suy luận checkpoint đã vượt qua |

Đánh giá nghiêm ngặt sử dụng manifest 1,493 khung hình địa phương đã kiểm toán. Điểm số của nó phải giữ riêng biệt với kết quả `30.13 / 13.53 / 13.08` đã trích dẫn trước đó cho đến khi sự khác biệt giao thức 1,493 so với 2,872 được giải quyết.

### 21.3 Trạng thái Go/No-Go cuối cùng

- **Tính toàn vẹn triển khai: GO.** Luồng tính đồng nhất tái tạo chính xác baseline và các artifact cũ không bị ghi đè.
- **Tuyên bố huấn luyện/nghiên cứu: NO-GO.** Chưa báo cáo kết quả chất lượng Phase 2: sự đối soát giao thức G0, việc chọn bộ khởi tạo mạnh hơn G1, và dữ liệu giám sát chuỗi G2 vẫn được yêu cầu.
- **Độ không chắc chắn U1: NO-GO.** Hiệu chuẩn đã được triển khai nhưng yêu cầu một tập dữ liệu thặng dư thực rời rạc trước khi có thể bật.

---

## 22. Hiệu chỉnh triển khai căn chỉnh theo đề xuất — 24 Tháng 7, 2026

Một lần review code thứ hai đã so sánh `phase2_refiner/` từng dòng một với kế hoạch xây dựng này và giai đoạn Phase 2 của `DEXAVATAR_METHOD_SOTA_ASTAR_PROPOSAL.md`. Triển khai đã được hiệu chỉnh để loại bỏ các điểm không khớp về mặt kỹ thuật mà không phụ thuộc vào việc thu thập tập dữ liệu bên ngoài.

### 22.1 Các điểm không khớp thiết kế đã được sửa chữa

- Cache schema v2 hiện ghi lại số khung hình, nhãn thời gian, FPS, kích thước ảnh, hash ảnh/nguồn, biến đổi tọa độ, hình dạng dùng chung, tính hợp lệ 2D/3D rõ ràng, hình học thân/cục bộ cổ tay, pháp tuyến lòng bàn tay, độ tin cậy U0 xác định, các khớp mục tiêu tùy chọn, và nguồn gốc nhà cung cấp. Bộ nhớ đệm thử nghiệm schema-v1 vẫn đọc được.
- Việc xây dựng đệm có thể tiêu thụ một lịch trình CSV đã khóa và thất bại khi thiếu khung hình bộ khởi tạo thay vì âm thầm chỉ liệt kê các PKL kết quả có sẵn. Bộ nhớ đệm chỉ thêm vào và từ chối `--overwrite`.
- Lệnh chỉ số phân tách rõ ràng từ chối sự chồng lặp nguồn hoặc người ký giữa các phân tách train, validation, test, và calibration. It không tự sáng tác ra các phân tách khoa học khi metadata không có sẵn.
- Các tích hợp dataset/expert sau này sử dụng hợp đồng `ObservationProvider` và `TargetProvider` trung lập. Không có phụ thuộc đặc thù dataset nào được thêm vào trong lần hiệu chỉnh này.
- Mỗi joint token tăng từ 28 lên 43 chiều và hiện bao gồm tính hợp lệ rõ ràng, độ tin cậy cố định, hình học thân, hình học cục bộ cổ tay, ngữ cảnh lòng bàn tay, và thời gian khoảng trống khung hình chuẩn hóa bên cạnh các đặc trưng góc xoay và 2D.
- Bộ tinh chỉnh hiện sử dụng vị trí thời gian tương đối học được, bằng chứng không gian/thời gian/nhóm có trọng số độ tin cậy, phương sai quan sát xoay và 2D/3D học được riêng biệt, vị trí khớp phụ trợ, và pháp tuyến lòng bàn tay dẫn xuất. Thặng dư bằng 0 vẫn giữ là một đồng nhất tư thế chính xác.
- Nhiễu tổng hợp hiện bao gồm các đợt bùng nổ thân trên, một tay, cả hai tay, chuỗi ngón tay, gắn nối cổ tay, mất điểm đặc trưng, cắt xén crop, và tráo đổi tính thuận tay trong khi vẫn giữ 35% batch sạch theo cấu hình.
- Giao diện tổn thất hiện bao phủ góc xoay, đỉnh vùng cân bằng, khớp, đầu ngón tay, pháp tuyến lòng bàn tay, khả năng quan sát, tốc độ/gia tốc mục tiêu với trọng số chuyển tiếp, neo giữ đáng tin cậy, sinh cơ học mềm, và độ không chắc chắn dị sai. Các số hạng hình học kích hoạt khi một nhà cung cấp sau đó cung cấp mục tiêu hợp lệ của họ.
- Huấn luyện hiện hỗ trợ BF16, EMA, không weight decay trên các tensor norm/bias/embedding, dừng sớm, checkpoint phục hồi định kỳ, khôi phục trạng thái đầy đủ an toàn, phục hồi optimizer/scheduler/RNG, và khởi tạo spatial-prior tùy chọn đã kiểm toán. U1 hỗ trợ giai đoạn warm-up chỉ cho độ không chắc chắn.
- Suy luận U1 yêu cầu một báo cáo hiệu chuẩn rời rạc đã vượt qua cổng hiệu chuẩn. Hiệu chuẩn phương sai được áp dụng trước khi tính trọng số độ tin cậy, và các hash hiệu chuẩn/config/checkpoint/cache được ghi vào run manifest.
- Suy luận thực hiện preflight đầu ra, từ chối ghi vào các PKL bộ khởi tạo đóng băng, từ chối các file kết quả cũ, xác minh chính xác ID khung hình đầu ra, sử dụng giới hạn an toàn được cấu hình trong checkpoint, lưu mặt nạ lùi an toàn theo nhóm, và làm rõ hành vi ghi đè mesh.
- Đánh giá hiện từ chối các hàng manifest trùng lặp, tôn trọng cả hai cờ đánh giá bàn tay, ghi lại hash manifest, và báo cáo trung vị và deltas decile tồi nhất theo ký hiệu bên cạnh các khoảng bootstrap gom nhóm.

### 22.2 Validation sau hiệu chỉnh

| Kiểm tra | Kết quả |
|---|---|
| Định dạng/Phân tích tĩnh | `ruff format` sạch; `ruff check` passed |
| Kiểm thử tự động | **19 passed** |
| Biên dịch Python | tất cả các module Phase 2 passed |
| Cache-v2 bộ khởi tạo thực | Ablehnen: 14 khung hình, `(14, 51, 43)` tokens, các trường timing/hash/transform hợp lệ |
| Khả năng tương thích tính đồng nhất | 14/14 PKL; chênh lệch tham số tối đa từ `method_hamer`: **0.0** |
| Training/checkpoint smoke | một cập nhật optimizer BF16/EMA, các checkpoint trạng thái đầy đủ best/last, và suy luận an toàn `weights_only=True` đã vượt qua |
| Resume smoke | khôi phục optimizer/scheduler/mô hình/EMA và CPU/CUDA/data-loader RNG đã vượt qua |
| Suy luận checkpoint | 14/14 đầu ra với nguồn gốc và chẩn đoán được xuất |

### 22.3 Ranh giới cố ý để lại cho các nhà cung cấp sau

Code hiện đã sẵn sàng để nhận dữ liệu chuỗi bên ngoài, nhưng nhà cung cấp vẫn phải cung cấp các mục tiêu được cấp phép, ID nguồn/người ký, các khớp 3D tương thích metric, biến đổi tọa độ, mặt nạ chất lượng mục tiêu, và — nơi có sẵn — giám sát đỉnh hoặc lòng bàn tay. Việc chuyển giao DPoser-X cũng yêu cầu một adapter khóa/biểu diễn đặc thù checkpoint trước khi hook nạp tương thích trung lập có thể nhận nó.

Do đó, bản sửa đổi này thay đổi trạng thái **căn chỉnh kỹ thuật** thành **GO**, trong khi G0–G2 và huấn luyện khoa học đầy đủ vẫn được quản lý bởi các cổng Go/No-Go hiện có. Các artifact SGNify và đánh giá của tác giả vẫn bị cấm làm mục tiêu huấn luyện.

---

## 23. Tích hợp dữ liệu địa phương, kế hoạch thực thi tối thượng, và lần chạy T1 thực tế — 24 Tháng 7, 2026

Các tập dữ liệu địa phương mới được cung cấp đã được kiểm toán ở cấp độ mảng/annotation-schema trước khi huấn luyện. Điều này thay đổi G2 từ "nhà cung cấp không có sẵn" thành một quyết định dữ liệu một phần được đo lường rõ ràng; nó **không** làm cho cổng dữ liệu Phase 2 đầy đủ đi qua.

### 23.1 Vai trò đã kiểm toán của từng nguồn địa phương

| Nguồn địa phương | Nội dung đã kiểm toán | Vai trò Phase 2 được ủy quyền | Ràng buộc huấn luyện |
|---|---|---|---|
| `data/InterHand2.6M/annotations/train` | 18,718 khung hình được chú giải MANO trước khi lọc độ dài chuỗi; pose `(48)`, shape `(10)`, translation `(3)`, metadata multi-view, và 42 khớp thế giới | Tiền huấn luyện T1 khớp nối bàn tay và động lực bàn tay tương tác Tầng B | chỉ dùng tập train chính thức; thân và cổ tay rõ ràng không được giám sát |
| `data/InterHand2.6M/annotations/val` | 2,736 khung hình chú giải MANO | validation chính thức rời rạc về nguồn cho Tầng B | chỉ dùng cho checkpoint validation |
| `data/InterHand2.6M/annotations/test` | các chú giải test chính thức | test giữ nguyên sau này | không bao giờ được đọc bởi cache builder hoặc lần chạy huấn luyện |
| `data/signbposer_data/train` | 1,449 tư thế thân với hình dạng `(63)` cộng với metadata nguồn/chỉ số và trọng số mẫu theo từng tư thế | khởi động nhiễu không gian chỉ cho thân | không tuyên bố về thời gian: các bản ghi được cung cấp không có danh tính hay thứ tự chuỗi |
| `outputs/shared/<sign>` và các đầu ra phương pháp hiện có | 57 ký hiệu cô lập và các artifact Phase 1/baseline đóng băng | xây dựng bộ khởi tạo chỉ đọc, suy luận, và so sánh Luồng L | không bao giờ dùng làm mục tiêu huấn luyện sạch; các ký hiệu hiện tại thuộc về benchmark |
| `data/smplx_gt` và `data/evaluation_from_author` | Ground truth SGNify/benchmark địa phương và asset đánh giá | chỉ dùng cho đánh giá nghiêm ngặt | bị cấm khỏi train, validation, hiệu chuẩn, và chọn checkpoint |

Adapter InterHand sử dụng thứ tự khớp nối MANO `index, middle, pinky, ring, thumb`, mỗi thứ tự MCP→PIP→DIP. Các khớp thế giới được chuyển đổi từ milimét sang mét cục bộ cổ tay. Hướng bàn tay toàn cục MANO không được âm thầm xử lý như một mục tiêu cổ tay thân SMPL-X; nó vẫn nằm ngoài mặt nạ bàn tay 15 khớp được giám sát. Hình dạng MANO được giữ lại dưới dạng nguồn gốc nhưng không bị dán nhãn sai thành betas SMPL-X.

### 23.2 Triển khai mới đã hoàn thành

| File | Trách nhiệm hoàn thành trong bản sửa đổi này |
|---|---|
| `phase2_refiner/data/build_interhand_cache.py` | bộ đọc luồng cho các file COCO gần GB, chọn metadata chuỗi/camera xác định, giữ nguyên phân tách chính thức, các clip không chồng lặp, chuyển đổi MANO/khớp, mặt nạ chỉ bàn tay, nguồn gốc, đầu ra chỉ thêm, và preflight dung lượng đĩa |
| `phase2_refiner/data/build_signbposer_cache.py` | adapter ngân hàng tư thế chỉ thân 1 khung hình từ chối sáng tạo tính liên tục thời gian một cách rõ ràng |
| `phase2_refiner/data/audit_training_cache.py` | xác minh cache/schema, quét chồng lặp phân tách và nguồn bị cấm, thống kê độ bao phủ mục tiêu, và báo cáo G2/T1 Go/No-Go có thể thực thi |
| `phase2_refiner/data/cache_schema.py` | hợp đồng `target_rotation_valid[T,51]` tương thích ngược |
| `phase2_refiner/data/dataset.py` | các batch mục tiêu một phần có đệm và lắp ráp quan sát khớp thân/cục bộ cổ tay chính xác |
| `phase2_refiner/data/corruptions.py` | chỉ làm nhiễu các vùng/khung hình có mục tiêu hợp lệ; ngăn việc làm nhiễu vùng không được giám sát trở thành một nhiệm vụ sai lệch |
| `phase2_refiner/losses/sequence.py` | mặt nạ mục tiêu một phần áp dụng cho các số hạng xoay, chuyển động, neo, sinh cơ học, và độ không chắc chắn |
| `phase2_refiner/geometry/rotations.py` và `losses/geometry.py` | FP32, các tổn thất góc an toàn với đồng nhất chính xác có gradient hữu hạn dưới BF16 |
| `phase2_refiner/train.py` | nhiễu nhận thức mục tiêu, validation làm nhiễu xác định, báo cáo lỗi và phục hồi được chèn/thặng dư, validation sạch, và hủy lập tức khi tổn thất/gradient không hữu hạn |
| `phase2_refiner/evaluate_t1_recovery.py` | báo cáo phục hồi 4/8/16 khung hình xác định sau huấn luyện; từ chối cho qua G3 từ một proxy chỉ xoay mà không có bảo toàn đỉnh vùng được giải mã |
| `phase2_refiner/configs/uawsr_t1_interhand.yaml` | cấu hình khả thi T1 Tầng B đóng băng 5,000 bước |

Các phương pháp DexAvatar hiện có, thư mục baseline, code bộ đánh giá, và các file đầu ra Phase 1 không bị chỉnh sửa. Việc tạo bộ nhớ đệm chỉ ghi các đường dẫn mới dưới `cache/phase2/`; huấn luyện chỉ ghi các đường dẫn `outputs/phase2_training/` mới.

### 23.3 Bộ nhớ đệm được cụ thể hóa và các cổng được đo lường

Các bộ nhớ đệm chỉ thêm là:

- `cache/phase2/interhand_t1_v1`: **1,148 clip train / 16,096 khung hình** và **47 clip validation / 2,736 khung hình**, chiếm khoảng 38 MB;
- `cache/phase2/signbposer_spatial_v1`: **1,449 mẫu 1 khung hình chỉ cho thân**, chiếm khoảng 18 MB; và
- `cache/phase2/interhand_t1_v1/readiness_report.json`: đo lường độ sẵn sàng bất biến và hash manifest.

| Mục cổng | Kết quả đo lường | Quyết định |
|---|---:|---|
| sự chồng lặp clip-ID train/validation | 0 | GO |
| vi phạm nguồn `smplx_gt` / đánh giá tác giả bị cấm | 0 | GO |
| sự không khớp phân tách chính thức | 0 | GO |
| số clip / khung hình InterHand train | 1,148 / 16,096 | dưới 10,000 clip và 250,000 khung hình: NO-GO cho G2 |
| clip train có ít nhất 16 khung hình | 148 / 1,148 = 12.89% | dưới 80%: NO-GO cho G2 |
| khung hình mục tiêu đầy đủ thân + cả hai tay | 0% | dưới 70%: NO-GO cho G2 |
| khung hình mục tiêu tay trái / tay phải | 10,481 / 10,663 | GO cho tính khả thi T1 chỉ bàn tay |

Do đó:

- **G2 huấn luyện Phase 2 chính/đầy đủ: NO-GO.** Tuyên bố chất lượng end-to-end đầy đủ bị cấm với dữ liệu hiện có.
- **T1 tính khả thi làm nhiễu bàn tay tổng hợp Tầng B: GO.** Đây là lần chạy huấn luyện được ủy quyền khoa học duy nhất được bắt đầu bây giờ.
- **Tiền huấn luyện thời gian SignBPoser: NO-GO.** Nó có thể khởi tạo hành vi không gian của thân, nhưng 1,449 tư thế xáo trộn của nó không thể huấn luyện động lực học toàn chuỗi.

### 23.4 Kế hoạch thực thi dàn dựng tối thượng

1. **D0 — Hợp đồng dữ liệu bất biến.** Giữ các vai trò train/val/test InterHand chính thức; phân phiên bản mọi bộ nhớ đệm; ghi lại giấy phép, hash, ID nguồn/người ký/video, đơn vị, thứ tự khớp, biến đổi tọa độ, và mặt nạ mục tiêu. Yêu cầu kiểm toán nguồn bị cấm và phân tách rời rạc vượt qua trên mỗi lần rebuild.
2. **T0 — Hợp đồng tính đồng nhất/số học.** Yêu cầu đồng nhất tham số thặng dư bằng 0 chính xác, bất biến đệm, gradient FP32/BF16 hữu hạn, khứ hồi mesh dưới `0.01 mm`, và hủy lập tức khi có bất kỳ giá trị không hữu hạn nào.
3. **S0 — Khởi động không gian thân tùy chọn.** Làm nhiễu các tư thế thân SignBPoser cá thể bằng nhiễu SO(3) 1 khung hình và chỉ giám sát 12 khớp thân được tinh chỉnh. Chỉ chuyển các trọng số không gian tương thích nếu việc phục hồi tư thế thân giữ nguyên cải thiện và tính đồng nhất sạch được bảo toàn. Không bao giờ gọi đây là huấn luyện chuỗi.
4. **T1-B — Phục hồi tổng hợp InterHand.** Huấn luyện U0 trên các clip InterHand train chính thức; validate trên các clip val chính thức với các đợt bùng nổ xoay bàn tay/ngón tay 4–16 khung hình xác định. Việc chọn checkpoint sử dụng tổng validation bị nhiễu, trong khi validation sạch được báo cáo riêng.
5. **Quyết định G3.** GO chỉ khi việc phục hồi lỗi được chèn đạt ít nhất 30% cho các trường hợp 4/8/16 khung hình và sự suy giảm vùng sạch vẫn ở dưới 2%. Nếu không hãy dừng lại, kiểm tra ánh xạ/quy mô nhiễu, và không thêm độ không chắc chắn.
6. **Thu thập Tầng A/C cho G2.** Thêm các chuỗi SMPL-X ký hiệu hoàn chỉnh được cấp phép, có thứ tự và chạy các chuyên gia Phase 1 đóng băng chính xác trên các khung hình RGB của chúng. Yêu cầu ≥10,000 clip ký hiệu riêng biệt nguồn hoặc ≥250,000 khung hình hợp lệ, ≥80% clip có độ dài ≥16, ≥70% các trường thân/hai tay hoàn chỉnh, người ký và video rời rạc, và kiểm toán chất lượng mục tiêu 100 chuỗi dưới 10% lỗi nghiêm trọng.
7. **T2 — Học thặng dư chuyên gia thực tế.** Trộn 50% thặng dư chuyên gia chính xác, 25% bùng nổ tổng hợp, và 25% batch sạch; cân bằng nguồn và nội dung một hand/hai hand ở cấp clip. GO yêu cầu cải thiện validation bên ngoài ≥3%, không có vùng nào tệ hơn >1%, và ≥8% trên tập khó đóng băng.
8. **T3 — Thích ứng ký hiệu.** Fine-tune trên các chuỗi ký hiệu rời rạc về nguồn ở learning rate thấp hơn trong khi giữ lại 20–30% dữ liệu bàn tay chất lượng cao chung. Từ chối thích ứng nếu lỗi chuyển tiếp hoặc bất kỳ tiêu chí vùng nào suy giảm.
9. **T4 — Độ không chắc chắn.** Chỉ huấn luyện U1 sau khi T2/T3 xác định đi qua. Fit co giãn phương sai trên một phân tách hiệu chuẩn rời rạc; giữ lại U0 trừ khi tất cả các cổng NLL/xếp hạng/rủi ro-độ bao phủ đi qua.
10. **T5 và Luồng L.** Chỉ chạy tối ưu hóa chuỗi 10–20 bước tùy chọn sau khi ablation đóng băng bên ngoài của nó đi qua. Sau đó suy luận trên manifest A1 chính xác, áp dụng lùi an toàn theo nhóm, render mesh tân mạn neo nguồn, và đánh giá tất cả các vùng với độ bao phủ đầy đủ giống hệt nhau và 3 seed. G6/G7 vẫn là các cổng chấp nhận cuối cùng.

### 23.5 Khởi chạy huấn luyện và trạng thái trực tiếp

Lần thử đầu tiên được lưu giữ cố ý tại `logs/phase2/t1_interhand_seed42_20260724.txt`. Nó làm lộ các gradient NaN ở bước 10 và đã dừng; không có checkpoint nào từ lần chạy đó là hợp lệ. Nguyên nhân là đạo hàm của lỗi góc norm bằng 0 dưới các mục tiêu đồng nhất BF16. Sau khi sửa số học và thêm bảo vệ giá trị hữu hạn, **21 kiểm thử tự động passed**.

Một lần preflight cache thực 25 bước trong tmux đã hoàn thành với huấn luyện, validation, và checkpoint hữu hạn:

- tmux session: `phase2_t1_preflight_run2_20260724` (hoàn thành);
- log: `logs/phase2/t1_interhand_preflight_run2_20260724.txt`;
- output: `outputs/phase2_training/t1_interhand_preflight_run2`;
- lỗi xoay được chèn / thặng dư validation bị nhiễu: `0.312508 / 0.312511 rad`;
- sự phục hồi trước huấn luyện có ý nghĩa: khoảng `0%`, đúng như kỳ vọng; và
- lỗi xoay validation sạch: khoảng `1.18e-5 rad`.

Lần chạy T1 5,000 bước được ủy quyền hiện đang hoạt động:

- tmux session: `phase2_t1_interhand_run2_20260724`;
- command/config: `phase2_refiner/configs/uawsr_t1_interhand.yaml` với output override `outputs/phase2_training/t1_interhand_seed42_run2`;
- text log chỉ thêm vào: `logs/phase2/t1_interhand_seed42_run2_20260724.txt`; và
- xác minh khởi chạy: hữu hạn qua lần validation bước 250 đầu tiên với khoảng 3.9 GB bộ nhớ GPU được cấp phát; validation bị nhiễu cải thiện từ khoảng `0%` phục hồi tại preflight 25 bước lên **1.06%** tại bước 250, trong khi lỗi xoay sạch vẫn giữ khoảng `5.51e-4 rad`. Đây là tiến triển, không phải cổng G3 đã đi qua.

Theo dõi mà không đính kèm:

```bash
tail -f logs/phase2/t1_interhand_seed42_run2_20260724.txt
tmux capture-pane -pt phase2_t1_interhand_run2_20260724:0 -S -100
```

Lần chạy này là một **thử nghiệm tính khả thi trực tiếp, không phải là kết quả độ chính xác đã hoàn thành**. Quyết định G3 của nó chỉ được nối thêm sau khi checkpoint cuối cùng và việc đánh giá phục hồi 4/8/16 khung hình được khai báo trước hoàn thành.

---

## 24. Độ đo T1 đã hoàn thành — 24 Tháng 7, 2026

Lần chạy tính khả thi Tầng B InterHand seed-42 đã hoàn thành tất cả **5,000 bước optimizer** mà không có tổn thất hay gradient không hữu hạn nào sau khi sửa gradient đồng nhất. Nó đã ghi `last.pt`, các checkpoint phục hồi định kỳ 500 bước, và checkpoint validation tốt nhất. Tmux session huấn luyện đã thoát bình thường.

### 24.1 Chọn checkpoint và validation tổng hợp

Tổng validation bị nhiễu được khai báo trước đã chọn bước **4,000** là tốt nhất:

| Chỉ số | Tốt nhất bước 4,000 | Cuối cùng bước 5,000 |
|---|---:|---:|
| tổng validation bị nhiễu | **0.062359** | 0.062665 |
| lỗi xoay được chèn | 0.312508 rad | 0.312508 rad |
| lỗi xoay thặng dư | **0.151483 rad** | 0.152361 rad |
| sự phục hồi tổng hợp | **51.53%** | 51.25% |
| tổn thất xoay bị nhiễu | **0.020548 rad** | 0.020669 rad |
| tổn thất tốc độ bị nhiễu | **0.149738** | 0.150927 |
| tổn thất gia tốc bị nhiễu | **0.039898** | 0.040159 |
| lỗi xoay sạch | — | **1.616e-5 rad** |
| tổng sạch | — | **0.000344** |

Hash checkpoint:

- best step-4,000 SHA-256: `f3b4ea84b54d1d042200a068ccace78544bc8acb026442113ec145097259aca1`;
- last step-5,000 SHA-256: `49fc4d50dfcf319e855c94d69a761a59d1529b1330cbd706ea4c40ac3c6d6a74`.

### 24.2 Đánh giá phục hồi EMA 4/8/16 khung hình chính xác

`phase2_refiner.evaluate_t1_recovery` đã đánh giá checkpoint EMA tốt nhất trên bộ nhớ đệm validation InterHand chính thức đầy đủ với các seed nhiễu cố định.

| Thời lượng bùng nổ | Lỗi được chèn | Lỗi thặng dư | Sự phục hồi | Cổng xoay 30% |
|---:|---:|---:|---:|---|
| 4 khung hình | 0.307224 rad | 0.150729 rad | **50.94%** | GO |
| 8 khung hình | 0.307798 rad | 0.151754 rad | **50.70%** | GO |
| 16 khung hình | 0.305799 rad | 0.156842 rad | **48.71%** | GO |

Các chỉ số đầu vào sạch cho cùng checkpoint EMA tốt nhất là:

- xoay: `9.662e-6 rad`;
- tốc độ: `2.426e-4`;
- gia tốc: `3.869e-5`;
- lỗi khớp/đầu ngón tay trong biểu diễn phụ trợ cục bộ cổ tay: `2.956e-5 / 2.845e-5 m`;
- tổng sạch: `0.000269`.

Báo cáo đọc được bởi máy là `outputs/phase2_training/t1_interhand_seed42_run2/t1_recovery.json`, với SHA-256 `d16bba48cc4bc01182df7e5e1420d29c3431d8c8125539415211d4122c4ec5a6`.
Log văn bản hoàn chỉnh là:

- `logs/phase2/t1_interhand_seed42_run2_20260724.txt`; và
- `logs/phase2/t1_interhand_seed42_recovery_20260724.txt`.

### 24.3 Diễn giải Go/No-Go cuối cùng

- **Proxy phục hồi xoay T1: GO.** Mọi thời lượng cố định đều vượt mức giảm 30% bắt buộc, bao gồm cả đợt bùng nổ 16 khung hình khó nhất.
- **Hành vi số học và tính đồng nhất sạch: GO.** Huấn luyện kết thúc mà không có NaN sau khi sửa, và trôi lệch xoay sạch vào khoảng `9.7e-6 rad`.
- **G3 như được định nghĩa trong đề xuất: PENDING / NO-GO để chuyển tiếp.** Cổng chính thức yêu cầu phục hồi lỗi đỉnh vùng được giải mã và sự suy giảm đỉnh vùng sạch dưới 2%. Mục tiêu Tầng B hiện tại là khớp nối MANO một phần không có mục tiêu thân/cổ tay SMPL-X hoàn chỉnh, vì vậy một proxy chỉ xoay không thể hoàn thành cổng một cách trung thực.
- **G2/Phase 2 end-to-end đầy đủ: NO-GO.** Lần chạy này không thay thế tập hợp ký hiệu rời rạc nguồn bị thiếu, các cặp thặng dư thực chuyên gia chính xác, hoặc yêu cầu 3 seed. Nó không được báo cáo như độ chính xác cuối cùng của benchmark DexAvatar.

Hành động đúng tiếp theo là lưu giữ checkpoint này làm ứng viên khởi tạo bàn tay Tầng B, thêm các chuỗi ký hiệu Tầng A/C hoàn chỉnh, chạy các chuyên gia Phase 1 chính xác trên các chuỗi đó, và sau đó lặp lại T1/T2 với các cổng đỉnh vùng được giải mã và 3 seed cố định.

---

## 25. Khắc phục G3 chính thức và chạy lại dữ liệu mở rộng — 24 Tháng 7, 2026

Phần này ghi lại việc triển khai và chạy lại được yêu cầu cho các quyết định G3 và G2 chính thức còn lại. Nó thay thế bất kỳ diễn giải nào chỉ dựa trên proxy xoay trong Phần 24.

### 25.1 Kết quả đỉnh được giải mã InterHand

`phase2_refiner.evaluate_t1_vertices` đã giải mã checkpoint EMA InterHand tốt nhất qua mô hình SMPL-X địa phương và đo lường các vùng bàn tay tương thích MANO 778 đỉnh tính bằng milimét.

| Bùng nổ | Tay trái được chèn → thặng dư | Trái phục hồi | Tay phải được chèn → thặng dư | Phải phục hồi |
|---:|---:|---:|---:|---:|
| 4 | 8.5195 → 6.1668 mm | **27.62%** | 9.0993 → 5.5590 mm | **38.91%** |
| 8 | 7.3305 → 5.2198 mm | **28.79%** | 7.8371 → 5.0559 mm | **35.49%** |
| 16 | 8.1066 → 5.9960 mm | **26.04%** | 9.1771 → 6.3045 mm | **31.30%** |

Trôi lệch đỉnh trung bình sạch là `0.01981 mm` cho bàn tay trái và `0.01922 mm` cho bàn tay phải, chỉ khoảng `0.21–0.27%` lỗi được chèn tương ứng. Do đó bảo toàn sạch đi qua, bàn tay phải vượt 30% ở mọi thời lượng, và bàn tay trái thất bại 30% ở mọi thời lượng. **G3 chính thức vẫn là NO-GO; proxy chỉ xoay đã che giấu sự bất đối xứng vùng này.**

Báo cáo đọc được bởi máy là `outputs/phase2_training/t1_interhand_seed42_run2/t1_vertex_recovery.json` (SHA-256 `21e4cc19a26aa09f6b3f32f6e9ee92f2aca76e7141084c81ef8f8965295a5c27`).

### 25.2 Nguồn địa phương hoàn chỉnh mới được kiểm toán

File nén địa phương `data/ARCTIC/downloads/raw_seqs.zip` hợp lệ và chứa 301 chuỗi SMPL-X có thứ tự / 218,273 khung hình trên 9 đối tượng có sẵn. Mỗi chuỗi cung cấp tư thế thân `(T,63)`, cả hai bàn tay `(T,45)`, hướng gốc, dịch chuyển, và xoay khuôn mặt. Adapter mới đọc trực tiếp từ ZIP và không bao giờ giải nén hay đọc các mục tiêu SGNify.

Gán bộ nhớ đệm rời rạc nguồn là:

- đối tượng train `s01,s02,s04,s05,s06,s07`: 201 chuỗi nguồn, **2,351 clip / 146,781 khung hình giữ lại**;
- đối tượng validation `s08`: 42 chuỗi nguồn, **511 clip / 31,822 khung hình giữ lại**; và
- đối tượng `s09,s10`: giữ nguyên để sử dụng kiểm tra sau này.

Tất cả các clip được cụ thể hóa đều ít nhất 16 khung hình và tất cả các khung hình được giữ lại chứa mục tiêu tư thế thân và cả hai tay hoàn chỉnh. Bộ nhớ đệm chiếm khoảng 285 MB và không có vi phạm nguồn bị cấm. Báo cáo độ sẵn sàng của nó vẫn là **G2 NO-GO** vì 2,351 clip / 146,781 khung hình huấn luyện nằm dưới ngưỡng khối lượng của đề xuất và ARCTIC là chuyển động bàn tay-vật thể chung, không phải chuyển động ngôn ngữ ký hiệu rời rạc nguồn.

### 25.3 Triển khai khắc phục

Các thành phần mới hoặc được sửa đổi là:

- `phase2_refiner/data/build_arctic_cache.py`: bộ đọc ZIP chỉ thêm, các đoạn giữ nguyên chuỗi, phân tách đối tượng rõ ràng, mục tiêu 51 khớp hoàn chỉnh, và nguồn gốc miền chung;
- `phase2_refiner/evaluate_t1_vertices.py`: phục hồi đỉnh vùng giải mã 4/8/16 khung hình chính xác, tỷ lệ bảo toàn sạch trên nhiễu, phát hiện vùng thiếu, và quyết định G3 nghiêm ngặt;
- `phase2_refiner/configs/uawsr_t1_arctic_geometry.yaml`: nhiễu toàn bộ thân/hai tay và huấn luyện đỉnh vùng cân bằng khả vi;
- `phase2_refiner/geometry/rotations.py`: chuyển đổi ma trận→quaternion→trục-góc gradient hữu hạn cho giải mã SMPL-X;
- `phase2_refiner/train.py`: trọng số decoder đóng băng, giải mã mục tiêu không có gradient, bảo vệ gradient hữu hạn, và override batch/accumulation có thể tái tạo; và
- `phase2_refiner/tests/test_rotations.py`: kiểm thử hồi quy gradient decoder.

Lần preflight hình học đầu tiên làm lộ gradient chuyển đổi không hữu hạn và đã dừng trước khi cập nhật optimizer. Sau khi sửa, **22 kiểm thử passed** và cả preflight hình học GPU batch-1 lẫn batch-8 đều hoàn thành với tổn thất, gradient, và checkpoint hữu hạn.

### 25.4 Lần chạy lại hoàn chỉnh vùng đang hoạt động

Checkpoint EMA InterHand tốt nhất khởi tạo một lần chạy ARCTIC T1 51 khớp hoàn chỉnh với tổn thất đỉnh SMPL-X thân trên/bàn tay trái/bàn tay phải cân bằng. Lần chạy được ủy quyền đang hoạt động với:

- tmux session: `phase2_t1_arctic_geometry_seed42_20260724`;
- config: `phase2_refiner/configs/uawsr_t1_arctic_geometry.yaml`;
- output: `outputs/phase2_training/t1_arctic_geometry_seed42`;
- text log: `logs/phase2/t1_arctic_geometry_seed42_20260724.txt`;
- optimizer batch: 8 clip 64 khung hình hoàn chỉnh, accumulation 1;
- trạng thái khởi chạy: hữu hạn qua bước 30; và
- trạng thái GPU khi huấn luyện: cấp phát khoảng 29 GB tổng cộng và 100% sử dụng trên RTX 5880 Ada 49 GB.

Lần chạy này có thể đóng G3 vùng hoàn chỉnh chung nếu đánh giá vùng giải mã chính xác vượt qua. Nó không thể, bất kể bộ nhớ GPU, biến chuyển động ARCTIC chung thành tập dữ liệu ký hiệu Tầng A/C bị thiếu; do đó G2 đầy đủ vẫn giữ là NO-GO cho đến khi dữ liệu đó được cung cấp hoặc một tập ký hiệu địa phương được cấp phép riêng vượt qua kiểm toán thực thi tương tự.

---

## 26. Lần chạy ARCTIC T1 đã hoàn thành và quyết định G3 chính thức — 24 Tháng 7, 2026

Phần 25.4 không còn là placeholder lần chạy đang hoạt động. Lần chạy hình học thân/hai tay hoàn chỉnh được ủy quyền đã hoàn thành tất cả **5,000 bước optimizer** mà không có tổn thất không hữu hạn, hủy gradient, vết traceback, hay lỗi hết bộ nhớ. Nó sử dụng bộ nhớ đệm ARCTIC rời rạc đối tượng, trọng số EMA, huấn luyện BF16, effective batch 8, accumulation 1, và checkpoint InterHand tốt nhất làm khởi tạo không gian tương thích. Các phương pháp DexAvatar hiện có và submodule `sapiens` không bị sửa đổi bởi lần chạy.

### 26.1 Kết quả huấn luyện

Bước cuối cùng cũng là checkpoint validation tốt nhất:

| Mục | Bước 500 | Bước tốt nhất 5,000 |
|---|---:|---:|
| lỗi xoay được chèn validation cố định | 0.304090 rad | 0.304090 rad |
| lỗi xoay thặng dư | 0.140643 rad | **0.087226 rad** |
| sự phục hồi | 53.75% | **71.32%** |
| tổn thất đỉnh giải mã cân bằng | 4.398 mm | **2.310 mm** |
| tổng tổn thất validation | 0.022852 | **0.012971** |

Checkpoint: `outputs/phase2_training/t1_arctic_geometry_seed42/best.pt`, bước 5,000, EMA, SHA-256 `29f081b96b942e9651484a0f52155b2ab28d6ce2dff65ac3bfd1b0a3f9bfa9c2`. Cấu hình đã resolve ghi lại các override CLI có hiệu lực và hash manifest train/validation bất biến.

### 26.2 Kết quả xoay FP32 thời lượng cố định chính xác

Đánh giá sau huấn luyện chính thức sử dụng FP32 ngay cả khi huấn luyện sử dụng BF16. Điều này ngăn việc định lượng ma trận xoay giảm độ chính xác bị đếm thành suy giảm sạch do mô hình gây ra.

| Bùng nổ | Được chèn | Thặng dư | Sự phục hồi | Cổng proxy 30% |
|---:|---:|---:|---:|:---:|
| 4 khung hình | 0.302967 rad | 0.074240 rad | **75.50%** | GO |
| 8 khung hình | 0.305548 rad | 0.080485 rad | **73.66%** | GO |
| 16 khung hình | 0.305432 rad | 0.100833 rad | **66.99%** | GO |

Lỗi xoay sạch là `2.633e-6 rad`. Báo cáo máy đọc được là `outputs/phase2_training/t1_arctic_geometry_seed42/t1_recovery_fp32.json` (SHA-256 `789457d971378ca0f320138165894a02651014ae77fa8152fc3eadf8cf16cd44`).

### 26.3 Kết quả đỉnh vùng giải mã chính xác

Bộ đánh giá chính thức giải mã SMPL-X và đo lường 7,279 đỉnh thân trên và 778 đỉnh cho mỗi bàn tay. Cả ba vùng đều có mục tiêu hoàn chỉnh và quần thể khung hình bị nhiễu không rỗng ở mọi thời lượng.

| Bùng nổ | Thân trên được chèn → thặng dư (phục hồi) | Bàn tay trái được chèn → thặng dư (phục hồi) | Bàn tay phải được chèn → thặng dư (phục hồi) |
|---:|---:|---:|---:|
| 4 | 101.6667 → 20.9825 mm (**79.36%**) | 11.6372 → 3.2522 mm (**72.05%**) | 11.6602 → 3.1141 mm (**73.29%**) |
| 8 | 92.3241 → 19.8518 mm (**78.50%**) | 12.0146 → 3.1989 mm (**73.38%**) | 12.6043 → 3.3114 mm (**73.73%**) |
| 16 | 91.9098 → 21.1719 mm (**76.96%**) | 12.2632 → 4.1272 mm (**66.34%**) | 11.3235 → 3.6046 mm (**68.17%**) |

Trôi lệch đỉnh trung bình sạch là **0.001656 mm** cho thân trên, **0.002696 mm** cho bàn tay trái, và **0.001681 mm** cho bàn tay phải. Tùy theo thời lượng, đây chỉ là `0.0016–0.0018%`, `0.0220–0.0232%`, và `0.0133–0.0148%` lỗi được chèn tương ứng, tất cả đều nằm dưới giới hạn 2%.

**G3 vùng hoàn chỉnh chung chính thức: GO.** Mọi sự phục hồi theo vùng 4/8/16 khung hình đều trên 30%, mọi tỷ lệ sạch trên nhiễu đều dưới 2%, và thân trên cùng cả hai bàn tay đều có sẵn. Điều này thay thế kết quả InterHand-only G3 NO-GO trong Phần 25.1; nó không thay thế cổng G2 miền ký hiệu riêng biệt.

Báo cáo đọc được bởi máy: `outputs/phase2_training/t1_arctic_geometry_seed42/t1_vertex_recovery_fp32.json` (SHA-256 `db6ab28db7a10bb9b484baadff21e44314c7bb7cde2591bd7b9c5f3521e8d55d`).

### 26.4 Quyết định G2 và Phase 2 đầy đủ

Chạy lại đã khép lại G3 về cơ bản, nhưng **G2/Phase 2 ngôn ngữ ký hiệu đầy đủ vẫn là NO-GO**. Kiểm toán có thể thực thi hiện yêu cầu khối lượng ngưỡng phải thuộc về miền ký hiệu một cách rõ ràng chứ không cho phép một tập hợp chung lớn vô tình vượt qua. Kết quả hiện tại là:

| Kiểm tra G2 | Kết quả |
|---|---:|
| loại trừ rò rỉ/tính toàn vẹn | GO |
| các clip ít nhất 16 khung hình | 100%, GO |
| thân hoàn chỉnh và cả hai tay | 100%, GO |
| tổng khối lượng train | 2,351 clip / 146,781 khung hình, NO-GO |
| khối lượng train miền ký hiệu | **0 clip / 0 khung hình, NO-GO** |

`cache/phase2/arctic_t1_v1/readiness_report.json` có SHA-256 `7c985da12653f26cd130850171179d2bead45c9216f29e3e9fd49d9f6c116e2f`.
Do đó, không có lần chạy Tầng C quan sát sang sạch, hiệu chuẩn độ không chắc chắn, hay tuyên bố benchmark ký hiệu đã khóa nào bị thêu dệt. Phase 2 đầy đủ chỉ có thể bắt đầu sau khi một tập ký hiệu có thứ tự, được cấp phép, rời rạc về nguồn/người ký đạt ít nhất 10,000 clip hoặc 250,000 khung hình và vượt qua cùng một kiểm toán tính đầy đủ và rò rỉ.

### 26.5 Các file và bằng chứng khả năng tái tạo hoàn thành trong lần chạy lại này

- `phase2_refiner/configs/uawsr_t1_arctic_geometry.yaml`: mô hình T1 vùng hoàn chỉnh, nhiễu, hình học, và công thức huấn luyện;
- `phase2_refiner/data/build_arctic_cache.py`: adapter ZIP ARCTIC không phá hủy và phân tách đối tượng;
- `phase2_refiner/data/audit_training_cache.py`: kế toán miền chuyển động và cổng khối lượng G2 miền ký hiệu rõ ràng;
- `phase2_refiner/evaluate_t1_recovery.py`: proxy xoay 4/8/16 xác định với chế độ chính thức FP32 chính xác;
- `phase2_refiner/evaluate_t1_vertices.py`: đánh giá G3 thân trên và hai tay được giải mã xác định với chế độ chính thức FP32 chính xác;
- `phase2_refiner/geometry/rotations.py`, `phase2_refiner/train.py`, và `phase2_refiner/tests/test_rotations.py`: gradient decoder hữu hạn, hình học mục tiêu đóng băng, override effective-batch có thể tái tạo, và độ bao phủ hồi quy; và
- `phase2_refiner/README.md`: các lệnh cache, audit, training, và evaluation hiện tại cộng với các hợp đồng cổng miền ký hiệu và FP32.

Log được bổ sung tại `logs/phase2/t1_arctic_geometry_seed42_20260724.txt` và `logs/phase2/t1_arctic_formal_eval_20260724.txt`. Xác minh cuối cùng sau khi triển khai và báo cáo: `ruff check phase2_refiner` passed, `ruff format --check phase2_refiner` passed, tất cả **22 kiểm thử passed**, `compileall` passed, và `git diff --check` passed.

---

## 27. Lần chạy Phase 2 miền ngôn ngữ ký hiệu How2Sign — Khởi chạy 24 Tháng 7, 2026

Các asset How2Sign và PHOENIX mới được cung cấp đã được kiểm toán trước khi bắt đầu một tuyên bố huấn luyện mới. How2Sign được chọn vì nó cung cấp **31,047 clip train chính thức ghép cặp / 5,053,093 khung hình pose 2D có thứ tự**, 97.85% số clip có ít nhất 16 khung hình, và 2,192 video nguồn train của nó không trùng lặp với 115 video nguồn dev chính thức. PHOENIX không được sử dụng cho lần chạy này vì việc trích xuất địa phương hiện chỉ chứa 822 tư thế chỉ thân được gom lại và một dàn dựng giáo viên 3 khung hình mỗi clip thưa thớt; nó không thể giám sát một bộ tinh chỉnh toàn chuỗi thân/hai tay hoàn chỉnh.

Các file tư thế How2Sign chứa các đường vết 2D 133 điểm thay vì các góc xoay mục tiêu SMPL-X. Do đó chúng không bị xử lý sai thành nhãn 3D sạch. Giai đoạn giáo viên mới có thể phục hồi giải mã các video nguồn trực tiếp trong bộ nhớ, sử dụng các đường vết toàn thân được cung cấp cho các crop người ký xác định, và chạy mô hình SMPLer-X H32 đóng băng để tạo ra các mục tiêu 3D giả thân/tay trái/tay phải hoàn chỉnh có thứ tự. It không bao giờ sao chép các khung hình nguồn, đọc phân tách test chính thức, hay đọc các mục tiêu SGNify.

Đo lường Preflight:

| FP32 teacher batch | Clips mỗi lần gọi | Cấp phát GPU đỉnh | Thấu lượng (Throughput) | Kết quả |
|---:|---:|---:|---:|:---:|
| 32 khung hình | 1 | 6.63 GiB | 0.365 clips/s | hữu hạn |
| 128 khung hình | 4 | 8.91 GiB | 0.600 clips/s | hữu hạn |
| 256 khung hình | 8 | **15.13 GiB** | **0.652 clips/s preflight; khoảng 0.84 clips/s duy trì** | được chọn |
| 512 khung hình | 16 | 27.58 GiB | 0.606 clips/s | hữu hạn nhưng chậm hơn |

MMCV ROIAlign được cài đặt từ chối các đầu vào hỗn hợp FP16/FP32, nên AMP không an toàn không được dùng. Lần gọi FP32 256 khung hình được chọn nhanh hơn lần gọi 512 khung hình và đạt 100% GPU utilization trong khi giữ lại hơn 30 GiB khoảng trống vật lý.

Mục tiêu trích xuất đang hoạt động là 11,000 clip train chính thức và 1,200 clip dev rời rạc nguồn, mỗi clip có 32 khung hình có thứ tự được lấy mẫu đều. Điều này cung cấp một đệm thất bại trên cả hai ngưỡng khối lượng G2. Lưu trữ giáo viên được dự đoán nhỏ gọn dưới 1 GiB, điều quan trọng vì hệ thống file workspace chỉ có khoảng 29 GiB trống khi khởi chạy.

Triển khai mới cho lần chạy này:

- `phase2_refiner/data/extract_how2sign_teacher.py`: chọn phân tách chính thức xác định, giải mã video trong bộ nhớ, crop từ đường vết, suy luận H32 đóng băng theo batch, đầu ra từng clip nguyên tử, phục hồi, log thất bại, và log GPU/ETA;
- `phase2_refiner/data/build_how2sign_cache.py`: ánh xạ quan sát COCO-WholeBody sang 51-token, mục tiêu SMPL-X giả hoàn chỉnh, bộ lọc mục tiêu thảm họa tự động, gom nhóm video nguồn 11 ký tự chính xác, bộ nhớ đệm chỉ thêm, và hỗn hợp giữ lại ARCTIC khoảng 20%;
- `phase2_refiner/data/audit_training_cache.py`: trùng lặp nhóm nguồn train/dev, khối lượng miền chuyển động, kế toán loại mục tiêu, và cổng chất lượng mục tiêu giả;
- `phase2_refiner/configs/uawsr_t1_how2sign_geometry.yaml`: thích ứng ký hiệu khởi tạo ARCTIC tốc độ thấp hơn với tổn thất hình học vùng đầy đủ; và
- `phase2_refiner/scripts/run_how2sign_phase2_after_teacher.sh`: cache tự động, G2 nghiêm ngặt, preflight GPU batch, huấn luyện 5,000 bước, và đánh giá xoay/đỉnh giải mã FP32 chính xác sau khi trích xuất giáo viên.

Các session và log đang hoạt động:

- teacher session: `phase2_how2sign_teacher_v1_20260724`;
- downstream train/eval session: `phase2_how2sign_train_eval_v1_20260724`;
- teacher log: `logs/phase2/how2sign_teacher_v1_20260724.txt`; và
- pipeline/training log: `logs/phase2/how2sign_phase2_pipeline_20260724.txt`.

Tại checkpoint khởi chạy được ghi lại, việc trích xuất đã đạt 288/11,000 clip train với 0 thất bại. Xác minh tĩnh sau triển khai báo cáo `ruff check` passed, `compileall` passed, `git diff --check` passed, và tất cả **25 kiểm thử passed**. Các kết quả khoa học G2/G3 sẽ được nối vào chỉ từ các báo cáo đệm, checkpoint, và báo cáo chính thức đọc được bởi máy đã hoàn thành.

### 27.1 Khởi động lại tăng tốc trích xuất — 24 Tháng 7, 2026

Bộ trích xuất ban đầu đã được dừng tại một ranh giới nguyên tử với **776/11,000** clip train hoàn thành và 0 đầu ra hỏng/thất bại. Các file đó và `selection.json` xác định được giữ nguyên; tiến trình tăng tốc tiếp tục việc chọn lựa giống hệt tại clip 777 thay vì ghi đè công việc trước đó.

`extract_how2sign_teacher.py` hiện sử dụng 5 bộ giải mã OpenCV độc lập với threading lồng nhau của OpenCV bị tắt. Nó cũng giữ chính xác một nhóm clip được prefetch: các CPU worker giải mã và crop nhóm tiếp theo trong khi SMPLer-X xử lý nhóm hiện tại trên GPU. Ràng buộc 1 nhóm tránh tăng trưởng bộ nhớ host vô hạn, và các lần ghi NPZ nén từng clip vẫn là các thao tác file-tạm-cộng-đổi-tên-nguyên-từ.

Thiết lập ổn định cuối cùng là 20 clip / 640 khung hình mỗi lần gọi FP32 teacher, 5 bộ giải mã, và 1 nhóm prefetch. Các lần gọi đơn lớn hơn bị từ chối bằng đo lường chứ không phải triển khai: 768, 992, và 1,024 khung hình thất bại trong deconvolution box-head SMPLer-X với `CUDNN_STATUS_NOT_SUPPORTED`; trường hợp 1,024 đạt 34,896 MiB trước khi thất bại. Cấu hình chồng lặp được chọn thay vào đó chiếm **34,798 MiB VRAM** ở trạng thái ổn định trong khi PyTorch báo cáo 21.57 GiB cấp phát tensor đỉnh trực tiếp. Các cửa sổ giải mã đo được khoảng 421–496% CPU, bao gồm cả snapshot 469% trên job sản xuất được phục hồi.

Preflight 3 nhóm đã hoàn thành 60/60 clip không có thất bại tại **1.224 clips/s**. Trên lần chạy sản xuất được khôi phục, tốc độ cuộn đạt **1.453 clips/s tại 956/11,000**, so với 0.833 clips/s trước khi tạm dừng: khoảng **thấu lượng cao hơn 74.4%**. ETA trích xuất train được báo cáo do đó giảm từ 3.41 giờ xuống 1.92 giờ tại checkpoint đó. Bộ theo dõi cache, G2 nghiêm ngặt, huấn luyện, và đánh giá chính thức phía hạ nguồn đã tạm dừng trong lúc tráo đổi extractor và chỉ tiếp tục sau khi tmux session teacher thay thế được xác nhận sống. Không có chỉ số huấn luyện nào được tuyên bố từ việc tối ưu hóa trích xuất này.

### 27.2 Phục hồi SIGKILL và khởi động lại an toàn bộ nhớ host

Tiến trình batch-640 sau đó bị kill từ bên ngoài sau khi hoàn thành nguyên tử **1,416/11,000** clip train. Không có ngoại lệ Python, CUDA, giải mã, hay đầu ra nào và không có NPZ một phần nào còn lại. Bản ghi OOM kernel không thể đọc được bởi account này, nhưng bằng chứng thời gian cho thấy áp lực bộ nhớ host: extractor giữ khoảng 14 GiB RSS, toàn bộ 8 GiB swap bị tiêu thụ, và các workload máy khác giữ thêm vài GiB. Điều này được ghi nhận như một sự gián đoạn runtime, không phải là thất bại mô hình và không phải là kết quả trích xuất hoàn thành.

Bộ theo dõi hạ nguồn trước đây dùng việc biến mất của tmux session teacher làm điều kiện hoàn thành của nó. Nó do đó cố gắng xây dựng cache tại 1,416 clip và đã thất bại đúng ở mức tối thiểu 10,000 clip của cache. Bộ theo dõi hiện cổng duy nhất trên số lượng đầu ra nguyên tử chính xác (11,000 train và 1,200 validation). Nếu tmux session teacher biến mất sớm, nó ghi log `stalled` và tiếp tục chờ; nó không thể tiến tới cache/huấn luyện trên một tập teacher bị cắt ngắn.

Preflight máy dùng chung của thiết lập thay thế — 384 khung hình / 12 clip, 4 decoder, và 1 nhóm prefetch — hoàn thành 24/24 clip không lỗi tại 0.882 clips/s, 21.36 GiB cấp phát GPU trực tiếp đỉnh, và khoảng 11.7 GiB RSS đỉnh. Cả hai session tmux đã được tạo lại và trích xuất tiếp tục từ clip 1,417 mà không ghi lại các đầu ra xác định hiện có. Điều này giảm áp lực bộ nhớ host và chấp nhận thấu lượng thấp hơn như một đánh đổi ổn định bắt buộc trong khi các workload máy khác vẫn hoạt động.

---

## 28. Lần chạy How2Sign T1 đã hoàn thành và quyết định Phase 2 hiện tại — 26 Tháng 7, 2026

Pipeline được khôi phục đã hoàn thành trích xuất teacher, xây dựng đệm, kiểm toán độ sẵn sàng tự động, 5,000 bước huấn luyện, và cả hai đánh giá T1 FP32 chính thức vào ngày 25 tháng 7 năm 2026. Không có tiến trình Phase 2 nào còn hoạt động.

Kết quả này phải được đặt tên chính xác: nó là một **mô hình phục hồi nhiễu tổng hợp T1 xác định được huấn luyện trên các chuỗi mục tiêu giả miền ký hiệu**. Nó chưa phải là mô hình Phase 2 nhận thức độ không chắc chắn quan sát-sang-sạch hoàn chỉnh. Cấu hình đã resolve có `predict_uncertainty: false`, `uncertainty_weight: 0.0`, và `observation_weight: 0.0`.

### 28.1 Các artifact dữ liệu và huấn luyện đã hoàn thành

| Mục | Kết quả đã hoàn thành |
|---|---:|
| tập How2Sign teacher train | 11,000 clip / 352,000 khung hình có thứ tự |
| tập How2Sign teacher validation | 1,200 clip / 38,400 khung hình có thứ tự |
| nhóm nguồn train / validation | 2,130 / 114 |
| trùng lặp clip hoặc nhóm nguồn train-validation | 0 / 0 |
| mục tiêu thân và cả hai tay hoàn chỉnh | 100% |
| thất bại mục tiêu thảm họa tự động | 0 / 11,000 train; 0 / 1,200 validation |
| tối ưu hóa | 5,000 bước, batch 48, BF16, seed 42, EMA |
| phục hồi xoay validation cố định cuối cùng | 72.28% |
| lỗi xoay thặng dư validation cố định cuối cùng | 0.084592 rad |

Báo cáo độ sẵn sàng tự động ghi `GO: full Phase 2 training` cho các kiểm tra đã triển khai của nó. Báo cáo đó thiết lập khối lượng, độ dài, tính đầy đủ, tính toàn vẹn phân tách, và màn lọc khung hình thảm họa tự động. Bản thân nó **không** chứng minh các yêu cầu riêng biệt của Phần 8.2 về một giấy phép tập dữ liệu được ghi nhận và việc kiểm toán thị giác thủ công trên 100 chuỗi mục tiêu giả được lấy mẫu ngẫu nhiên; hai mục bằng chứng đó vẫn còn mở.

### 28.2 Phục hồi xoay tổng hợp FP32 chính thức

| Đợt bùng nổ nhiễu | Lỗi được chèn | Lỗi thặng dư | Sự phục hồi | Ngưỡng T1 30% |
|---:|---:|---:|---:|:---:|
| 4 khung hình | 0.305079 rad | 0.081961 rad | **73.13%** | GO |
| 8 khung hình | 0.305644 rad | 0.082710 rad | **72.94%** | GO |
| 16 khung hình | 0.305359 rad | 0.086519 rad | **71.67%** | GO |

Lỗi xoay sạch là `1.0661e-5 rad`. Báo cáo chỉ-xoay cố ý để G3 ở trạng thái pending cho đến khi hình học vùng được giải mã được kiểm tra; báo cáo đỉnh chính thức sau đây cung cấp bằng chứng bắt buộc đó.

### 28.3 Phục hồi đỉnh vùng giải mã FP32 chính thức

| Bùng nổ | Thân trên được chèn → thặng dư (phục hồi) | Bàn tay trái được chèn → thặng dư (phục hồi) | Bàn tay phải được chèn → thặng dư (phục hồi) |
|---:|---:|---:|---:|
| 4 | 86.4102 → 18.0603 mm (**79.10%**) | 12.0199 → 5.6508 mm (**52.99%**) | 11.8223 → 6.0529 mm (**48.80%**) |
| 8 | 90.5142 → 19.0841 mm (**78.92%**) | 11.9027 → 5.7665 mm (**51.55%**) | 11.7596 → 6.0595 mm (**48.47%**) |
| 16 | 83.1817 → 18.6267 mm (**77.61%**) | 12.1194 → 6.1771 mm (**49.03%**) | 12.2049 → 6.6317 mm (**45.66%**) |

Trôi lệch giải mã sạch là **0.000652 mm** cho thân trên, **0.001713 mm** cho bàn tay trái, và **0.001259 mm** cho bàn tay phải. Mọi sự phục hồi theo vùng đều trên 30%, cả ba vùng đều hiện diện, và mọi tỷ lệ sạch trên nhiễu đều nằm xa dưới 2%. Do đó **G3 chính thức là GO**.

### 28.4 Trạng thái cổng tổng thể sau lần chạy đã hoàn thành

| Cổng | Quyết định hiện tại | Bằng chứng hoặc Yếu tố chặn |
|---|:---:|---|
| G0 khóa bộ đánh giá và độ bao phủ | PENDING | A0 và A1 được chọn vẫn thiếu 1 hợp đồng Luồng L độ bao phủ đầy đủ bất biến; sự khác biệt giao thức 1,493 so với 2,872 chưa giải quyết. |
| G1 chất lượng bộ khởi tạo Phase 1 | PENDING | Chưa có A1 độ bao phủ đầy đủ cuối cùng nào được chọn và làm sạch trên manifest chung. |
| G2 độ sẵn sàng dữ liệu | CONDITIONAL GO | Các kiểm tra thực thi pass tại 11,000 clip ký hiệu / 352,000 khung hình không trùng lặp phân tách; bằng chứng giấy phép tập dữ liệu và kiểm toán thủ công 100 chuỗi bắt buộc chưa được ghi nhận. |
| G3 khả năng phục hồi tổng hợp | **GO** | Tất cả sự phục hồi thân trên và hai tay 4/8/16 khung hình đều vượt 30%, với trôi lệch sạch dưới 2%. |
| G4 giá trị validation thực tế | NO-GO / NOT RUN | Không có thử nghiệm thặng dư thực Tầng C chuyên gia đóng băng chính xác nào cho thấy cải thiện trọng số bên ngoài ≥3%, suy giảm vùng ≤1%, và tăng tập khó ≥8%. |
| G5 tính hợp lệ độ không chắc chắn | NO-GO / NOT RUN | U1 bị tắt; không có kết quả Spearman, AUROC, rủi ro-độ bao phủ, NLL, hoặc tái tạo U1-vs-U0 nào tồn tại. |
| G6 benchmark địa phương đã khóa | NO-GO / NOT RUN | Không có đánh giá PKL/mesh Luồng L A1-vs-refiner độ bao phủ đầy đủ, CI gom nhóm, kết quả tập khó, tỷ lệ lùi an toàn, hay kết quả 3 seed nào tồn tại. |
| G7 so sánh chính thức | NO-GO | Sự khác biệt giao thức chính thức/địa phương chưa giải quyết, nên việc so sánh với `30.13 / 13.53 / 13.08` vẫn bị cấm. |

**Quyết định:** lần chạy này là **GO để tiến hành từ T1 sang T2 học thặng dư thực tế**. Nó là **NO-GO cho việc tuyên bố rằng Phase 2 nhận thức độ không chắc chắn đầy đủ đã hoàn thành, được chấp nhận, tốt hơn bộ khởi tạo Phase 1 được chọn, hoặc tốt hơn kết quả DexAvatar đã công bố**. Không có cổng định lượng đã thực thi nào thất bại; toàn bộ quyết định bị chặn bởi các giai đoạn và đánh giá bắt buộc chưa được chạy. Công việc bắt buộc tiếp theo là khóa manifest G0/G1, hoàn thành nghiêm ngặt hai mục bằng chứng G2 còn lại, T2/G4, sau đó U1/G5, tiếp theo là đánh giá G6 Luồng L 3 seed.

### 28.5 Các artifact khả năng tái tạo

- checkpoint: `outputs/phase2_training/t1_how2sign_geometry_seed42/best.pt`, SHA-256 `c86a95a7e900dda02a8f8ebc1bbe0ef36c656e4186ec4ef24507da65286b1b9e`;
- resolved configuration: `outputs/phase2_training/t1_how2sign_geometry_seed42/resolved_config.json`, SHA-256 `f34c091eebe630e0def86d1d60883c8c963d4be5580e070f7990168632ce80b5`;
- rotation report: `outputs/phase2_training/t1_how2sign_geometry_seed42/t1_recovery_fp32.json`, SHA-256 `a873190e9779e77bf87225d031caec21f9476756b2caf2a9c1700f9a1d9f6fa3`;
- decoded-vertex report: `outputs/phase2_training/t1_how2sign_geometry_seed42/t1_vertex_recovery_fp32.json`, SHA-256 `d93566febfedf8b95c823bf3bdb1c89d164938eaeee7fc257d45b56d13cbccec`;
- readiness report: `cache/phase2/how2sign_t1_v1/readiness_report.json`, SHA-256 `d383210d9d2806c89d308f9b69b5891af32f46b8c7cf0acd3ab87a0a76e4add5`; và
- append-only pipeline log: `logs/phase2/how2sign_phase2_pipeline_20260724.txt`, SHA-256 `ca5ecb71ff0392d1b87a80a9ceb6433eb0d7082ab7a73443d7ed4531d4fde119`.

---

## 29. Triển khai khắc phục cổng và chẩn đoán Lane-L đã thực thi — 26 Tháng 7, 2026

Phần này ghi lại việc điều tra nguyên nhân gốc rễ, khắc phục mã nguồn, và các thử nghiệm Go/No-Go mới được thực thi. Nó thay thế trạng thái G0/G1/G6 trong Phần 28.4. It không thay thế kết quả G3 chính thức trong Phần 28.

### 29.1 Các nguyên nhân gốc rễ được tìm thấy

1. **Không có thặng dư thực tế nào tồn tại trong bộ nhớ đệm How2Sign hiện tại.** Tất cả 11,000 clip train và 1,200 clip validation đều dùng `init_axis_angle == target_axis_angle`. Các mục tiêu và bộ khởi tạo đều là tư thế giáo viên giả SMPLer-X H32 giống nhau. Điều này đúng cho nhiễu tổng hợp T1, nhưng không thể huấn luyện hay validate hiệu chỉnh quan sát sang sạch T2.
2. **Trainer đã không triển khai thành phần batch T2 được chỉ định.** Nó đã áp dụng nhiễu ngẫu nhiên cho 1 luồng đệm, thay vì lấy mẫu rõ ràng 50% thặng dư thực, 25% tổng hợp từ sạch, và 25% ví dụ đồng nhất sạch.
3. **Lệnh hiệu chuẩn U1 ban đầu có thể tự thông qua.** Nó so sánh NLL trước và sau khi co giãn vô hướng của chính U1. It không yêu cầu một bộ so sánh U0 độ tin cậy bộ phát hiện tương ứng, nguồn gốc rời rạc về nguồn/người ký thực tế, mức tăng tái tạo bị nhiễu, hoặc giới hạn suy giảm sạch theo vùng.
4. **Không có quyết định G4/G6 có thể thực thi nào tồn tại.** Các tóm tắt vùng và CI đã được tạo ra, nhưng sự kết hợp đầy đủ của các ngưỡng đề xuất không được thực thi bằng máy. Đặc biệt, 1 seed có thể bị nhầm lẫn thành một kết quả khả năng tái tạo.
5. **Các ứng viên Phase-1 mạnh hơn không hoàn chỉnh.** Hầu hết chứa 1,450 trong số 1,493 khung hình đã khóa. Việc so sánh các quần thể bản địa của chúng sẽ làm thay đổi benchmark thay vì so sánh các phương pháp.
6. **Mô hình T1 đã học được việc phục hồi bị kiểm soát, không phải sự hiệu chỉnh các lỗi A1 tự nhiên.** Trên Luồng L sạch, nó hoạt động gần như đồng nhất. Điều này giải thích tại sao G3 mạnh trong khi chẩn đoán G6 mới chỉ có 0.089% mức tăng cân bằng.

Nguyên nhân thứ nhất và thứ sáu là các yếu tố chặn dữ liệu/khoa học, không phải vấn đề dung lượng GPU. Thêm bộ nhớ GPU không thể tạo ra các mục tiêu độc lập hoặc các thặng dư chuyên gia đóng băng thực tế.

### 29.2 Khắc phục mã nguồn

Các bổ sung sau đây được cô lập dưới `phase2_refiner` và không chỉnh sửa các phương pháp DexAvatar lịch sử hay đầu ra của chúng:

- `data/build_locked_fallback_view.py`: tạo một view symlink độ bao phủ đầy đủ chính xác, chỉ chọn một khung hình chính khi cả PKL và mesh của nó đều tồn tại và nếu không sẽ chọn cả hai artifact từ A0; ghi lại mọi lựa chọn và hash;
- `data/corruptions.py` và `train.py`: triển khai hỗn hợp 50/25/25 thực/tổng hợp/sạch T2 rõ ràng và giữ các đặc trưng xoay, tốc độ, và gia tốc nhất quán sau khi thay thế bộ khởi tạo bằng một mục tiêu sạch;
- `data/audit_real_residual_cache.py`: kiểm toán fail-closed yêu cầu loại mục tiêu độc lập, các nhà cung cấp bộ khởi tạo/mục tiêu riêng biệt, các nhóm nguồn, và một quần thể thặng dư thực có thể đo lường được;
- `data/build_observation_cache.py`: từ chối các gốc bộ khởi tạo/mục tiêu giống hệt nhau và truyền các trường nguồn gốc thặng dư thực bắt buộc;
- `evaluate_uncertainty.py` và `calibrate.py`: xuất các thặng dư sạch và bị nhiễu khớp cặp U0/U1, yêu cầu kiểm toán thặng dư thực vượt qua, so sánh NLL U1 đã hiệu chuẩn với U0 độ tin cậy bộ phát hiện, và thực thi các điều kiện tái tạo theo vùng;
- `evaluate_lane_diagnostics.py`: đóng băng và đánh giá tập hợp độ khó quan sát Luồng L, tập hợp sạch độ không chắc chắn thấp, và lùi an toàn nhóm-khung hình;
- `gates.py`: thực thi bằng máy G4 và mọi điều kiện số học G6 và từ chối cho qua khả năng tái tạo với ít hơn chính xác 3 seed;
- `configs/uawsr_t2_real_residual.yaml` và `configs/uawsr_u1_real_residual.yaml`: các cấu hình khởi đầu T2 và U1 căn chỉnh theo đề xuất; và
- `data/build_manual_audit_queue.py`: hàng đợi xem xét thủ công 100 chuỗi xác định, phân tầng theo nhóm nguồn.

`README.md` hiện tài liệu hóa các lệnh T2, U1, A1-lock, và gate được thực thi. Kết quả validation sau các thay đổi này là **31 kiểm thử passed**, Python `compileall` passed, và `git diff --check` passed.

### 29.3 Quét bộ khởi tạo đã khóa G0/G1

Manifest Luồng L bất biến chứa 1,493 khung hình và 57 ký hiệu, SHA-256 `ed76c077aeb9ece61eb860183bfad6e4aeef9a236a27238c313903414996fd2c`.
Mọi ứng viên không hoàn chỉnh đều được đánh giá dưới dạng lai không phá hủy: sử dụng 1,450 khung hình có sẵn của nó và A0 cho 43 khung hình còn thiếu. Do đó mọi hàng bên dưới đều sử dụng cùng manifest, topology, đơn vị, mặt nạ vùng, và sự căn chỉnh giống hệt nhau.

| Ứng viên | Thay đổi thân trên | Thay đổi tay trái | Thay đổi tay phải | Mức tăng tương đối vùng bình đẳng | Quyết định |
|---|---:|---:|---:|---:|:---:|
| Biomech + A0 fallback | −0.279 mm | −0.744 mm | −0.777 mm | 4.143% | hợp lệ |
| Direct + A0 fallback | +0.363 mm | −0.764 mm | −0.805 mm | 3.547% | loại: suy giảm thân trên |
| **Ensemble + A0 fallback** | **−0.373 mm** | **−0.749 mm** | **−0.814 mm** | **4.353%** | **được chọn làm A1** |
| Hand2D + A0 fallback | −0.336 mm | −0.808 mm | −0.764 mm | 4.329% | hợp lệ, điểm cân bằng thấp hơn |
| NLF-WiLoR + A0 fallback | +4.246 mm | +6.592 mm | +4.774 mm | −33.230% | loại |
| WiLoR + A0 fallback | −0.286 mm | −0.763 mm | −0.813 mm | 4.290% | hợp lệ, điểm cân bằng thấp hơn |

Các chỉ số A0 đã khóa là `29.907413 / 13.573462 / 12.927137 mm`. Các chỉ số A1 được chọn là **`29.534720 / 12.824893 / 12.112852 mm`**. Các khoảng delta 95% gom nhóm theo ký hiệu của nó là `[-0.612, -0.176]`, `[-1.108, -0.525]`, và `[-1.076, -0.574] mm`; cả ba đều loại trừ số 0 theo hướng cải thiện. View này có 1,450 khung hình ensemble và 43 lùi an toàn A0 nguyên tử (2.8801%), với 1,493/1,493 PKL và mesh. Topology SMPL-X làm cho bề mặt cổ tay/cẳng tay được kết nối, trong khi các lỗi thân trên và cả hai tay đều cải thiện.

Do đó **Luồng-L G0 là GO và G1 là GO**, với `method_ensemble + method_hamer fallback` được đóng băng làm A1. Câu hỏi quần thể chính thức riêng biệt vẫn nằm trong G7: A0 địa phương nghiêm ngặt về mặt số học gần với `30.13 / 13.53 / 13.08`, nhưng sự khác biệt 1,493 so với 2,872 khung hình chưa được đối soát độc lập.

### 29.4 Cập nhật bằng chứng G2

Trang dự án How2Sign chính thức ghi lại tập dữ liệu dưới dạng CC BY-NC 4.0. Bản ghi đọc được bởi máy là `docs/proposal/evidence/HOW2SIGN_LICENSE_RECORD.json`, SHA-256 `bdaf1c2d5fd05f998845b6faf46b067fe4b3551dc2ecb810655b89a752c21cd1`.
Nó là GO chỉ cho nghiên cứu phi thương mại với ghi nhận tác giả, link giấy phép, và công khai các thay đổi.

Hàng đợi xem xét thủ công xác định hiện tồn tại tại `outputs/phase2_gates/g2/how2sign_manual_audit_100.csv`, SHA-256 `d1245eaf302d5cd9ddff5267e65be33f0eaef519e82ac2e0e7b77992eb5548bc`.
Nó chứa 100 clip từ 100 nhóm nguồn riêng biệt. Tất cả các ô đánh giá vẫn là `PENDING`; không có tên hay quyết định người đánh giá người thật nào bị làm giả. Do đó **G2 vẫn là CONDITIONAL GO**, với chính xác một hành động bằng chứng còn lại: một người đánh giá được đặt tên phải hoàn thành việc kiểm toán thị giác 100 chuỗi và quan sát ít hơn 10 thất bại thảm họa.

### 29.5 Kiểm toán thặng dư T2/G4 chính thức

Kiểm toán fail-closed mới đã được thực thi trên tất cả các bộ nhớ đệm How2Sign hiện có:

| Phân tách | Clip / Khung hình | Khung hình thặng dư thực có thể đo lường | Loại mục tiêu độc lập | Nguồn gốc nhà cung cấp riêng biệt |
|---|---:|---:|:---:|:---:|
| train | 11,000 / 352,000 | **0 (0.0%)** | fail trên 11,000 | fail trên 11,000 |
| validation | 1,200 / 38,400 | **0 (0.0%)** | fail trên 1,200 | fail trên 1,200 |

Artifact: `outputs/phase2_gates/g4/how2sign_real_residual_audit.json`, SHA-256 `290aced40884b1efee1637e221a6405d6fc1cc2b3f47ff1066cd4f7e62336fa4`.

Điều này thiết lập chính thức **G4 NO-GO** với dữ liệu hiện tại. Bắt đầu T2 trên các cặp này sẽ huấn luyện mô hình tái tạo lại chính giáo viên của nó và không phải là một nỗ lực hợp lệ tại cổng đề xuất. Do đó huấn luyện U1 không được khởi chạy: chiến lược có thứ tự cấm rõ ràng T4 trước khi T2 xác định vượt qua G4. **G5 vẫn là NO-GO / không chạy đúng như thiết kế.**

### 29.6 Chẩn đoán Luồng-L GPU đã thực thi

Checkpoint T1 seed-42 đã được chạy trên bộ nhớ đệm A1 được chọn, render thành 1,493 mesh neo nguồn trên GPU, và đánh giá so với A1. Lần chạy sử dụng một session tmux và log chỉ thêm `outputs/phase2_gates/logs/lane_l_seed42_infer_render.txt`, SHA-256 `8721d84559ff5b7796d48b7ab5ccc03c66bc218af1032aa34f0fd1978af6fbfc`.
Renderer đạt khoảng 95% GPU utilization; công việc CPU được giới hạn ở 4 core.

| Vùng | A1 được chọn | T1 refiner | Thay đổi gộp | Delta trung bình theo ký hiệu 95% CI |
|---|---:|---:|---:|---:|
| Thân trên | 29.534720 | 29.531215 | −0.003505 mm | [−0.008547, +0.000361] |
| Tay trái | 12.824893 | 12.812379 | −0.012514 mm | [−0.033478, +0.003990] |
| Tay phải | 12.112852 | 12.093706 | −0.019146 mm | [−0.036076, +0.012084] |

Mức tăng tương đối vùng bình đẳng chỉ là **0.0892%**, so với cổng 3%. Zero vùng cải thiện với một CI gom nhóm loại trừ số 0. Lùi an toàn tốt: 0/4,479 nhóm-khung hình (0%). Trong tập hợp khó v1 đóng băng, mức tăng tay trái/tay phải là 2.985% và 4.408%, đều dưới 8%; bộ nhớ đệm A1 có sẵn không có khung hình thân nào thỏa mãn định nghĩa khó được khai báo trước đó. Suy giảm tay sạch là khoảng 0.00020% và 0.00035%, nhưng tương tự không có tập hợp thân sạch hợp lệ dưới định nghĩa đóng băng.

| Điều kiện G6 | Chẩn đoán Seed-42 |
|---|:---:|
| độ bao phủ 1,493 khung hình giống hệt | GO |
| không có vùng gộp nào tệ hơn >0.20 mm | GO |
| ít nhất 2 vùng cải thiện với CI loại trừ 0 | **NO-GO (0 vùng)** |
| mức tăng vùng bình đẳng ≥3% | **NO-GO (0.0892%)** |
| mức tăng tập hợp khó ≥8% | **NO-GO** |
| suy giảm sạch <1% ở mọi vùng | **NO-GO: tập hợp thân không có sẵn** |
| lùi an toàn <1% | GO (0%) |
| 3 seed, độ lệch chuẩn vùng <0.20 mm | **NO-GO: chỉ 1 seed** |

Quyết định thực thi là **G6 NO-GO**. Huấn luyện thêm 2 seed T1 sẽ không sửa được các kiểm tra kích thước hiệu ứng, độ có ý nghĩa, và tập khó đã thất bại, do đó chiến lược Go/No-Go dừng nhánh đắt đỏ đó thay vì tiêu tốn thời gian GPU cho một kết quả không thể đi qua.

### 29.7 Quyết định tổng thể hiện tại

| Cổng | Quyết định sau hiệu chỉnh | Lý do |
|---|:---:|---|
| G0 khóa bộ đánh giá/độ bao phủ Luồng-L | **GO** | hợp đồng A0/A1 1,493 khung hình bất biến, hash, topology, mặt nạ, và độ bao phủ đầy đủ đã ghi |
| G1 chất lượng bộ khởi tạo | **GO** | A1 lai được chọn cải thiện tất cả các vùng với tất cả các CI gom nhóm dưới 0 |
| G2 độ sẵn sàng dữ liệu | CONDITIONAL GO | khối lượng/tính toàn vẹn/giấy phép đã ghi; kiểm toán người thật 100 chuỗi đặt tên còn pending |
| G3 phục hồi tổng hợp | **GO** | phục hồi góc xoay và đỉnh vùng giải mã 4/8/16 khung hình chính thức passed |
| G4 giá trị validation thực | **NO-GO** | bộ nhớ đệm hiện tại chứa 0 khung hình thặng dư thực và không có ghép cặp mục tiêu/nhà cung cấp độc lập |
| G5 tính hợp lệ độ không chắc chắn | **NO-GO / chưa chạy** | cổng có thứ tự cấm U1 trước G4; bộ công cụ fail-closed hiện đã sẵn sàng |
| G6 benchmark địa phương đã khóa | **NO-GO** | mức tăng 0.0892%, 0 vùng có ý nghĩa, tập khó dưới ngưỡng, 1 seed |
| G7 so sánh chính thức | **NO-GO** | quần thể giao thức 1,493 so với 2,872 chưa giải quyết |

**Quyết định cuối cùng: Phase 2 đầy đủ chưa GO.** Triển khai hiện đã được căn chỉnh và chiến lược có thể thực thi địa phương đã hoàn thành, nhưng thử nghiệm thất bại một cách trung thực tại G4 và G6. Con đường hợp lệ tiếp theo không phải là huấn luyện T1 thêm: tạo các cặp Tầng C bằng cách chạy các chuyên gia A1 đóng băng chính xác trên các video rời rạc nguồn và sử dụng một mục tiêu sạch độc lập (GT, fit multi-view, hoặc giáo viên đa khung hình được tinh chỉnh độc lập), hoàn thành kiểm toán 100 chuỗi được đặt tên, chạy T2 với hỗn hợp 50/25/25 được triển khai, và yêu cầu G4 đi qua. Chỉ khi đó mới huấn luyện/hiệu chuẩn U1 và lặp lại G6 với 3 seed. Cho đến khi dữ liệu đó được cung cấp, việc giữ lại A1 hình học được chọn là quyết định phát hành được đề xuất bắt buộc.

### 29.8 các hash khả năng tái tạo

- G0 A0 summary: `c02f1ccb77bbd6ce5e6e1546ef9156a666dd1069c6ee7166aa087d242bda3d96`;
- selected A1 evaluation: `74d3042dc872a9cf5bb87d5c6f1dff25950537d99aecadde14320f024f7180a6`;
- selected A1 locked-view manifest: `cd9d52da521da5ea4cc50b3c249ff44c2f26e93380836691e9e286af96c4cb1c`;
- selected A1 cache manifest: `507dfad5c13a148cf9cd967104c98ff1cb750e22f0c1e198c0a1a181eb239933`;
- seed-42 inference manifest: `28277e07d294e83ca4cab67ad9519115bf112d84f45bff700acd5101d7f299d4`;
- seed-42 strict evaluation: `fca18bb162e94e79e33db951c9eeca7c1edc086bf0af0d54a90601543dfd9483`;
- seed-42 subset/fallback diagnostics: `bb3ad955dbc06b5fde353cb85a3543ca6227c8ee71c11b080aac1b7818926f5d`; và
- executable G6 decision: `00dea60a16e7e1f135c74011d69a36c1434d61fb4fe77a45c9340c86ce33a243`.

---

## 30. Thử nghiệm khắc phục Full-GO đã thực thi (26-27/07/2026)

Phần này thay thế trạng thái pending trong các Phần 29.4--29.7 cho nhánh thời gian 2D How2Sign mới. Nó ghi lại việc huấn luyện GPU thực tế và đánh giá đã khóa; không có chỉ số nào bên dưới là dự đoán hay sáng tác thủ công. DexAvatar gốc và tất cả các triển khai Phase 1/Phase 2 trước đó vẫn nguyên vẹn. Hành vi mới là opt-in thông qua cấu hình v6 và cờ `use_reprojection_skip`.

### 30.1 Quyết định thị giác G2 được ủy quyền bởi owner

Theo yêu cầu của owner dự án, Codex đã kiểm tra hàng đợi 100 clip xác định thay vì yêu cầu owner đánh giá các video. Bốn khung hình nguồn/giáo viên mỗi clip được render, sắp xếp thành 10 tấm hình liên tiếp (contact sheets), và xem xét cho các trường hợp tráo đổi đối tượng, vỡ thân thể, đứt gãy tư thế lớn, và các thất bại mục tiêu giả thảm họa khác.

| Mục kiểm toán | Kết quả |
|---|---:|
| Clip / nhóm nguồn riêng biệt | 100 / 100 |
| Khung hình kiểm tra mỗi clip | 4 |
| Thất bại thảm họa | 0 |
| Tỷ lệ thất bại thảm họa | 0.0% |
| Tỷ lệ yêu cầu | <10% |
| Quyết định AI được ủy quyền bởi owner | **GO** |

Báo cáo đọc được bởi máy là `outputs/phase2_gates/g2/how2sign_ai_visual_audit_100_report.json`, SHA-256 `9506f2825267696513e2a969985c4aebb6b70a8a1d7eeb0a0fa5326668a259da`.
Bản ghi này chứng nhận việc lọc thảm họa; nó không tuyên bố sự chính xác ở mức milimét hay ngón tay tinh tế. Nếu đề xuất được hiểu là yêu cầu một người thật bên ngoài được đặt tên thay vì một AI reviewer được ủy quyền bởi owner, G2 vẫn ở trạng thái conditional cho chữ ký bên ngoài đó.

### 30.2 Mục tiêu thời gian 2D độc lập và triển khai thặng dư

Bộ nhớ đệm How2Sign trước đó có tư thế bộ khởi tạo/mục tiêu giống hệt nhau. Pipeline mới tạo ra một mục tiêu huấn luyện thực sự khác 0 bằng cách kết hợp các đường vết 2D How2Sign có thứ tự với tinh chỉnh chùm thời gian (temporal bundle adjustment), một neo bộ khởi tạo, và các hiệu chỉnh tư thế có giới hạn. Bug ánh xạ đường vết tay trái/phải và sự không khớp tọa độ How2Sign `[0,1]` so với Lane `[-1,1]` đã được sửa trước khi huấn luyện. Thặng dư chiếu lại 102 chiều theo khung hình được lưu trữ không bị cắt và chỉ co giãn trong đường dẫn đầu vào mô hình.

| Phân tách | Clip | Khung hình | Nhóm nguồn | Chồng lặp với train |
|---|---:|---:|---:|---:|
| Train | 10,822 | 346,304 | 2,128 | -- |
| Validation | 498 | 15,936 | 57 | 0 |
| Calibration | 497 | 15,904 | 57 | 0 |

Tất cả các clip có 32 khung hình và mục tiêu thân/hai tay hoàn chỉnh. Việc fit mục tiêu giảm lỗi chiếu lại 2D 37.692% tổng thể: 4.169% thân, 56.886% tay trái, và 55.196% tay phải. Hiệu chỉnh mục tiêu trung bình là 5.001 độ và tối đa là 15.807 độ. Artifact G2 được kiểm toán lại là `outputs/phase2_gates/g2/how2sign_2d_temporal_reprojection_g2.json`, SHA-256 `f621a3f1e28f99d2c1f165d1f5a554213a6bf4e7b876b08c3f710662e97290ec`.

Việc khắc phục phía mô hình là an toàn về tính đồng nhất:

- schema v3 thêm `reprojection_residual_2d` tùy chọn trong khi vẫn giữ schema v1/v2 đọc được;
- phép chiếu 43 kênh đã tiền huấn luyện được sao chép vào mô hình 45 kênh và chỉ 2 tóm tắt độ tin cậy mới bắt đầu từ 0;
- các hàng hỗn hợp tổng hợp/sạch xóa các thặng dư đệm cũ sau khi thay thế bộ khởi tạo;
- một đường tắt (skip) chiếu lại chéo khớp khởi tạo bằng 0 tùy chọn ánh xạ tất cả 102 tọa độ thặng dư tới 153 hiệu chỉnh tư thế;
- validation sử dụng điểm số đề xuất các vùng bình đẳng chính xác, hình phạt suy giảm vùng >1%, và so sánh rõ ràng giữa trọng số thô và EMA.

Một linear probe rời rạc nguồn đã dự đoán các hiệu chỉnh mục tiêu từ tín hiệu chiếu lại với mức giảm 69% MSE tương đối và độ tương đồng cosine 0.83. Điều này thiết lập tính sẵn có của tín hiệu trước lần chạy huấn luyện v6.

### 30.3 Huấn luyện T2 v6 và G4 proxy

T2 được huấn luyện trên GPU trong tmux với CPU thread được giới hạn ở 4. Checkpoint EMA được chọn là bước 1,500:

- checkpoint: `outputs/phase2_training/t2_how2sign_2d_temporal_reprojection_v6_seed42/best.pt`, SHA-256 `8c4e8c011fd69e51b6bc492012f1eb1667384cb095b2996a14935b0a26d8a482`;
- log huấn luyện chỉ thêm: `outputs/phase2_gates/logs/t2_how2sign_2d_temporal_reprojection_v6_seed42.txt`, SHA-256 `926fff2c74736a2fcc98b3dd159aff0f26873c701ac70382bd1077d48b6d01fc`.

Đánh giá validation đầy đủ độc lập tạo ra:

| Vùng | Lỗi bộ khởi tạo | Lỗi T2 | Mức tăng tương đối | Clip-bootstrap delta 95% CI |
|---|---:|---:|---:|---:|
| Thân trên | 2.5610 deg | 2.2599 deg | 11.76% | [−0.3052, −0.2968] deg |
| Tay trái | 7.4778 deg | 6.4641 deg | 13.56% | [−1.0332, −0.9937] deg |
| Tay phải | 7.3657 deg | 6.4226 deg | 12.80% | [−0.9596, −0.9260] deg |

Mức tăng các vùng bình đẳng là **12.71%**, mức tăng tập khó là **10.88%**, tất cả 15,936 khung hình được bao phủ, và lùi an toàn là 0%. Do đó mọi tiêu chí số học G4 đi qua. Artifact đánh giá: `outputs/phase2_gates/g4/how2sign_reprojection_v6_checkpoint_eval.json`, SHA-256 `ab7ccd7cb2d3d33be47091dbb15cc39ac344a313d8725cbe1512749d0e49a13f`.

Đây vẫn là **proxy G4 GO, formal G4 NO-GO**. Mọi mẫu How2Sign sử dụng `SMPLer-X H32` đóng băng làm bộ khởi tạo, không phải stack `method_ensemble + method_hamer fallback` Lane A1 chính xác đã chọn. Kiểm toán fail-closed loại tất cả 10,822/498/497 clip train/validation/calibration vì lý do nguồn gốc duy nhất này. Kiểm toán: `outputs/phase2_gates/g4/how2sign_2d_temporal_formal_exact_a1_audit.json`, SHA-256 `f714b9aa99a3dcca0163cbe3d58841991110d50a8dabb4de5bb591711914ac6d`; quyết định chính thức: `outputs/phase2_gates/g4/how2sign_reprojection_v6_formal_g4_decision.json`, SHA-256 `12aa7dcb42ee67f05ca7fbeca18795ef298bebbd8bb75f80e4f4e330cacfc8ba`.

### 30.4 Kết quả độ không chắc chắn U1/G5

U1 đã được huấn luyện như một chẩn đoán thông qua giai đoạn warm-up bộ tinh chỉnh đóng băng và dừng ở bước 1,000 trước khi fine-tune chung vì cổng tái tạo của nó thất bại. Checkpoint SHA-256 là `f01692261a9cc9f1a7b209c1da5856425e8e03ac6df0d32cc11ec8bb4f8b975b`; log SHA-256 là `29a70d2e78c5b7b7f27d39c3733afa3944969acd6739279bd7e9a37f3fc28f06`.

Đầu độ không chắc chắn chứa tín hiệu xếp hạng hữu ích: Spearman tổng thể là 0.7807, AUC decile tồi nhất là 0.7916, rủi ro là đơn điệu, và NLL đã hiệu chuẩn cải thiện từ −0.8684 xuống −1.6347. It vẫn thất bại bản phát hành:

- AUC decile tồi nhất tay trái/tay phải là 0.7139/0.7313, dưới ngưỡng bàn tay 0.75;
- tái tạo sạch U1 là 0.09654 so với U0 0.09155;
- tái tạo nhiễu U1 là 0.10443 so với U0 0.10346;
- mọi kiểm tra tái tạo theo vùng đều thất bại, và nguồn gốc thặng dư vẫn là proxy H32 chứ không phải A1 chính xác.

Quyết định thực thi là **G5 NO-GO; giữ lại U0 xác định và không phát hành U1**. Báo cáo: `outputs/phase2_gates/g5/how2sign_u1_v6_calibration_report.json`, SHA-256 `6cd5e6cb918bc2d1736004205dc40752b795fc15cded1afb9a78a0e76ef94cf6`.

### 30.5 Chuyển giao Luồng-L đã khóa và G6

Checkpoint T2 EMA đã được suy luận và render trên bộ nhớ đệm A1 Luồng-L 1,493 khung hình bất biến, sau đó đánh giá so với cùng GT, mặt nạ, topology, và manifest như G0/G1.

| Vùng | A1 đóng băng | T2 v6 | Delta gộp | Mean delta theo ký hiệu 95% CI |
|---|---:|---:|---:|---:|
| Thân trên | 29.534720 | 29.380139 | −0.154581 mm | [−0.228104, −0.023595] mm |
| Tay trái | 12.824893 | 12.691886 | −0.133007 mm | [−0.269451, −0.002396] mm |
| Tay phải | 12.112852 | 12.277672 | **+0.164819 mm** | **[+0.085230, +0.297501] mm** |

Thân và tay trái cải thiện có ý nghĩa, nhưng tay phải suy giảm có ý nghĩa. Mức tăng tương đối vùng bình đẳng chỉ là **0.0666%**, xa dưới 3%. Trên các tập hợp khó đóng băng, mức tăng tay trái/tay phải là −6.15%/−3.76% (đều là suy giảm); tập hợp tay phải sạch suy giảm 1.51%, trên giới hạn an toàn 1%. Độ bao phủ đầy đủ và lùi an toàn nhóm-khung hình vẫn giữ 0%.

| Điều kiện G6 | Kết quả |
|---|:---:|
| độ bao phủ 1,493 khung hình giống hệt | GO |
| không có suy giảm vùng gộp nào >0.20 mm | GO (tay phải +0.1648 mm) |
| ít nhất 2 vùng cải thiện có CI có ý nghĩa | GO (thân, tay trái) |
| mức tăng vùng bình đẳng ít nhất 3% | **NO-GO (0.0666%)** |
| mức tăng tập hợp khó ít nhất 8% | **NO-GO** |
| suy giảm sạch dưới 1% | **NO-GO (tay phải 1.51%)** |
| lùi an toàn dưới 1% | GO (0%) |
| chính xác 3 seed | **NO-GO (dừng sớm sau seed 42)** |

Chiến lược khai báo trước dừng 2 seed còn lại vì lần chạy được khóa đầu tiên nằm dưới mục tiêu hiệu ứng hai bậc độ lớn và thất bại cả kiểm tra tập khó lẫn an toàn sạch. Huấn luyện thêm các seed chỉ để tìm kiếm một kết quả Luồng thuận lợi cũng sẽ biến tập đánh giá được khóa thành tập tinh chỉnh.

Artifacts:

- đánh giá nghiêm ngặt: `outputs/phase2_gates/g6_reprojection_v6/seed42/summary.json`, SHA-256 `d17033d0ba52d8eea20fd6c6c3271875df76f4e809dc370fdc8b90c53e429867`;
- chẩn đoán tập hợp: `outputs/phase2_gates/g6_reprojection_v6/seed42/diagnostics.json`, SHA-256 `6dae98fe8d27a1db9a8d555927172e78e573f8dd81becded61dbb4309fafe0ff`;
- quyết định: `outputs/phase2_gates/g6_reprojection_v6/decision.json`, SHA-256 `ff9f6a5cbb555a61833451812376ba47b5b6280c13060101bc890fa56696a9f2`;
- log suy luận/render: `outputs/phase2_gates/logs/lane_l_reprojection_v6_seed42_infer_render.txt`, SHA-256 `d0b6c4f0db4aa44c30c4f9223a4a2f2b4577a6f209dd555b373b93fc1a924abf`.

### 30.6 Đối soát quần thể G7

Sự khác biệt 2,872 so với 1,493 hiện đã được giải thích chính xác. Cộng tổng các khoảng phân đoạn nguồn loại trừ điểm cuối của 57 ký hiệu cho ra 2,872. Bộ đánh giá được phát hành gấp đôi biên giới phân đoạn, nhưng các mesh GT được phát hành có nhịp (cadence) bằng 4; điều này tạo ra đúng manifest 1,493 hàng được dùng ở địa phương. Mỗi ký hiệu khớp với lựa chọn đã phát hành. Các mesh trung gian chính thức cần thiết để đánh giá tất cả 2,872 khung hình bài báo không có trong dữ liệu được cung cấp, do đó kết quả Luồng-L địa phương không thể được dán nhãn là kết quả quần thể đầy đủ của bài báo. G7 vẫn là **NO-GO cho một so sánh bài báo chính thức**, trong khi giao thức 1,493 khung hình được phát hành hoàn toàn có thể tái tạo. Báo cáo: `outputs/phase2_gates/g7/protocol_reconciliation_v1.json`, SHA-256 `922376b8ffdc4fc712d6f2a69f96121f8b7bfb701ef82793899a7611383761e9`.

### 30.7 Quyết định nguyên nhân gốc rễ và trạng thái cuối cùng

| Cổng | Quyết định hiện tại | Lý do dựa trên bằng chứng |
|---|:---:|---|
| G0 | **GO** | bộ đánh giá đóng băng, topology, mặt nạ, quần thể tác giả và độ bao phủ |
| G1 | **GO** | A1 được chọn cải thiện tất cả 3 vùng |
| G2 | **GO (được ủy quyền bởi owner)** | kiểm toán thị giác AI 100/100, 0 thất bại thảm họa; lưu ý người thật bên ngoài ở trên |
| G3 | **GO** | các kiểm thử phục hồi tổng hợp đi qua |
| G4 | **proxy GO / formal NO-GO** | mức tăng mục tiêu độc lập 12.71%, nhưng bộ khởi tạo là H32 chứ không phải A1 đóng băng chính xác |
| G5 | **NO-GO** | U1 làm tệ hơn việc tái tạo và trượt cả hai ngưỡng AUC bàn tay |
| G6 | **NO-GO** | mức tăng 0.0666%; các tập khó và an toàn tay phải sạch thất bại |
| G7 | **NO-GO official / GO Luồng-L đã phát hành** | số lượng được giải thích, nhưng 2,872 mesh GT trung gian vắng mặt |

**Phase 2 đầy đủ vẫn là NO-GO.** Việc khắc phục thiết lập rằng kiến trúc và tín hiệu huấn luyện thời gian 2D hoạt động trong miền (in-domain), nhưng nó cũng cô lập yếu tố chặn chính: sự không khớp miền bộ khởi tạo. Phép hiệu chỉnh thặng dư được huấn luyện trên H32 không chuyển giao an toàn sang ensemble A1 mạnh hơn, đặc biệt cho tay phải. Điều này không thể sửa một cách trung thực bằng cách tinh chỉnh trên Luồng-L sau khi đã quan sát kết quả của nó.

Con đường ngắn nhất hợp lệ để đạt full GO hiện đã cụ thể:

1. tạo các đệm How2Sign/PHOENIX rời rạc nguồn bằng cách chạy các chuyên gia A1 đóng băng chính xác và chính sách lùi an toàn, giữ nguyên sự bất đồng chuyên gia và thặng dư 2D thô;
2. xây dựng các mục tiêu 2D-temporal/multi-view độc lập hoặc mocap cho cùng các khởi tạo A1 đó và vượt qua kiểm toán nguồn gốc chính xác fail-closed;
3. huấn luyện lại T2 với đường dẫn v6 đã triển khai và vượt qua G4 trên một tập validation giữ lại không phải là Luồng-L;
4. chỉ huấn luyện lại U1 sau khi T2 vượt qua, yêu cầu AUC mỗi tay ít nhất 0.75 và không có suy giảm tái tạo U0;
5. đóng băng một lần, sau đó chạy 3 seed G6 trên Luồng-L; không tinh chỉnh kiến trúc, quy mô, hay các cổng vùng từ phản hồi Luồng-L;
6. lấy các mesh GT chính thức trung gian bị thiếu nếu yêu cầu so sánh bài báo 2,872 khung hình.

### 30.8 Các file được triển khai trong bản khắc phục này

Các module có thể thực thi mới:

- `phase2_refiner/data/refine_how2sign_targets.py`;
- `phase2_refiner/data/add_reprojection_residuals.py`;
- `phase2_refiner/data/render_how2sign_audit.py`;
- `phase2_refiner/data/complete_visual_audit.py`;
- `phase2_refiner/evaluate_residual_checkpoint.py`;
- `phase2_refiner/audit_official_protocol.py`;
- `phase2_refiner/configs/uawsr_t2_how2sign_2d_temporal.yaml`;
- `phase2_refiner/configs/uawsr_u1_how2sign_2d_temporal.yaml`;
- `phase2_refiner/tests/test_train.py`.

Các module được cập nhật:

- `phase2_refiner/config.py`, `train.py`, `infer.py`, `calibrate.py`, và `evaluate_uncertainty.py`;
- `phase2_refiner/models/pretrained.py` và `models/spatial_temporal_refiner.py`;
- `phase2_refiner/data/build_how2sign_cache.py`, `cache_schema.py`, `corruptions.py`, `dataset.py`, và `audit_real_residual_cache.py`;
- `phase2_refiner/tests/test_cache.py`, `test_corruptions.py`, `test_how2sign.py`, và `test_model.py`;
- `phase2_refiner/README.md` và báo cáo đề xuất này.

Tất cả các cấu hình cũ mặc định tắt đường dẫn skip mới, và tính tương thích schema/checkpoint lịch sử được bao phủ bởi các kiểm thử. Bộ kiểm thử địa phương cuối cùng pass 41 kiểm thử.

---

## 31. Khắc phục nguyên nhân gốc rễ, kiểm thử an toàn T5, và quyết định U1 v7 — 01 Tháng 8, 2026

Phần này ghi lại các thử nghiệm cuối cùng được yêu cầu sau Phần 30 và thay thế bảng trạng thái hiện tại của nó. Chủ sở hữu dự án đã chọn bộ đánh giá được tác giả phát hành và `data/evaluation_from_author` làm hợp đồng đánh giá chuẩn. Do đó **G7 là GO trên quần thể 57 ký hiệu, 1,493 khung hình của tác giả**. Quần thể 2,872 khung hình bài báo không có sẵn nằm ngoài phạm vi và không phải là yếu tố chặn cho dự án này.

### 31.1 Các nguyên nhân gốc rễ của các quyết định NO-GO còn lại

Kiểm toán đã cô lập 5 nguyên nhân độc lập thay vì một thất bại huấn luyện chung:

1. **Không tương thích bộ khởi tạo chính thức:** tất cả 10,822/498/497 clip train/validation/calibration chứa bộ khởi tạo H32 đóng băng, không phải stack ensemble/fallback Lane-L A1 chính xác được chọn bởi G1. Các mục tiêu rời rạc nguồn và thặng dư 2D là thực, nhưng bit nguồn gốc chính thức phải giữ là false.
2. **Trôi lệch miền thặng dư lớn:** tỷ lệ thặng dư trung vị Lane-A1 so với How2Sign là 2.35172 cho thân trên, 0.32744 cho bàn tay trái, và 0.34763 cho bàn tay phải. Mỗi vùng đều nằm ngoài phạm vi `[0.5, 2.0]` được khai báo trước. Điều này giải thích tại sao một phép hiệu chỉnh học được từ H32 lại chuyển giao kém sang bộ khởi tạo A1 mạnh hơn.
3. **Phản hồi U1 trong quá trình warm-up:** đầu phương sai mới khởi tạo đã ngay lập tức chỉnh sửa độ tin cậy attention. Các dự đoán ban đầu xấp xỉ ngẫu nhiên của nó đã thay đổi việc tái tạo trước khi đầu được hiệu chuẩn. U1 v7 học độ không chắc chắn mà không phản hồi nó quay lại đường dẫn tinh chỉnh xác định.
4. **Không khớp mục tiêu/cổng U1:** U1 cũ tối ưu hóa NLL dị sai nhưng G5 đánh giá xếp hạng decile tồi nhất. V7 thêm một tổn thất xếp hạng decile tồi nhất theo vùng, bao gồm các số hạng riêng cho thân, tay trái, và tay phải.
5. **Đặc trưng nhiễu cũ và quần thể chẩn đoán:** nhiễu tổng hợp đã thay đổi các quan sát/bộ khởi tạo mà không xóa thặng dư chiếu lại được đệm trong mỗi chế độ bị ảnh hưởng. Chẩn đoán Lane cũng đếm các khớp thân không hoạt động, tạo ra các tập hợp khó/sạch rỗng hoặc gây hiểu lầm. Nhiễu hiện làm vô hiệu hóa các kênh thặng dư cũ, và hợp đồng chẩn đoán v2 chỉ đánh giá quần thể `refine_mask` bất biến.

Việc nạp đệm A1 chính thức hiện ở trạng thái fail-closed. It xác minh ID stack G1 đóng băng, hash manifest Lane, hash đánh giá G1, hash của tất cả 5 thành phần stack, và mọi hash PKL kết quả theo từng khung hình trước khi cụ thể hóa. Bộ nhớ đệm H32 hiện tại do đó bị loại bỏ thay vì bị dán nhãn sai thành A1. Kiểm toán chính thức: `outputs/phase2_gates/g4/how2sign_formal_exact_a1_audit_v2.json`, SHA-256 `83e2f555e9322184f15fe5f4736066460098de52cd8e2dd0ce90dc93dce7cb0c`.
Kiểm toán trôi lệch miền: `outputs/phase2_gates/g6_reprojection_domain_shift_v1.json`, SHA-256 `362c04782d8085f9628d8f1e4178de9142d94f5ee8f8e945e1caef5654927971`.

### 31.2 Chiến lược an toàn chỉ dựa trên quan sát T5 được triển khai

Chiến lược lùi an toàn T5 của đề xuất hiện có thể thực thi end-to-end. Cho mỗi chuỗi hoàn chỉnh, nó chỉ tối ưu hóa một delta tư thế có biên trong 15 bước Adam (tối đa cứng 20), sử dụng các quan sát 2D đóng băng, camera tập dữ liệu chính xác, neo có trọng số độ tin cậy, tốc độ, và gia tốc. It không bao giờ đọc một mục tiêu huấn luyện hay GT benchmark. Các nhóm thân/trái/phải được chọn độc lập dựa trên việc cải thiện chiếu lại, theo sau là kiểm tra an toàn thứ hai so với bộ khởi tạo A1 gốc. Mỗi nhóm bị từ chối được đếm là lùi an toàn.

Trên tập validation How2Sign bên ngoài, T5 vượt qua mọi điều kiện **số học** G4:

| Đo lường G4 | Kết quả |
|---|---:|
| mức tăng tương đối các vùng bình đẳng | **33.1391%** |
| mức tăng tương đối tập hợp khó | **30.4181%** |
| mức tăng tương đối thân trên | **11.2553%** |
| mức tăng tương đối tay trái | **43.8149%** |
| mức tăng tương đối tay phải | **44.3470%** |
| độ bao phủ / suy giảm vùng | đầy đủ / không có |

Đây là **proxy G4 GO, formal G4 NO-GO**, chỉ vì bộ khởi tạo là H32 chứ không phải A1 chính xác. Artifacts:

- `outputs/phase2_gates/g4/how2sign_reprojection_v6_t5_checkpoint_eval.json`, SHA-256 `a652caa8c7a64bb471be6e3ce8fc25de3a04d61cdc44797730fc98582b0dc33e`;
- `outputs/phase2_gates/g4/how2sign_reprojection_v6_t5_formal_g4_decision.json`, SHA-256 `7264ab49c2d465d75f32c4f097fe7b4076baa600f70aec8e4485c5aeb74ed9bd`.

Các thiết lập sau đó được đóng băng và chạy 1 lần trên toàn bộ 57 ký hiệu Lane và tất cả 1,493 khung hình đánh giá của tác giả. Không có tinh chỉnh Lane nào được thực hiện.

| Vùng | A1 (mm) | T5 (mm) | T5 − A1 |
|---|---:|---:|---:|
| thân trên | 29.534720 | 29.436036 | −0.098685 |
| tay trái | 12.824893 | 12.835032 | +0.010140 |
| tay phải | 12.112852 | 12.359398 | **+0.246545** |

Kết quả G6 có thể thực thi là **NO-GO**: mức tăng tương đối các vùng bình đẳng là −0.5934%, suy giảm tay phải vượt quá 0.20 mm, mức tăng tập hợp khó gộp là −5.2511%, suy giảm tay phải sạch là 1.7700%, và lùi an toàn là 1,368/4,479 nhóm-khung hình (30.5425%). Chỉ có 0/57 clip thân, 7/57 clip tay trái, và 9/57 clip tay phải được chấp nhận. Theo giao thức đã khóa, 2 seed còn lại không được chạy sau thất bại seed-42 quyết định này, và T5 bị tắt cho bản phát hành.

Artifacts:

- chỉ số nghiêm ngặt: `outputs/phase2_gates/g6_reprojection_v6_t5/seed42/summary.json`, SHA-256 `0376b8aaf7871ea5f3838df5059ccef0fa6ca6169f701c6606764fd255d8b4ce`;
- chẩn đoán: `outputs/phase2_gates/g6_reprojection_v6_t5/seed42/diagnostics.json`, SHA-256 `87436c4940ce9bbe78f1edde9385582da763e32e2d15746dd702e1440c2cf843`;
- quyết định: `outputs/phase2_gates/g6_reprojection_v6_t5/decision_seed42.json`, SHA-256 `966bbde45fe964db11a623ece5462e90a7858daa58e05c98a00d56eb7f14e925`.

### 31.3 Huấn luyện U1 v7 và khắc phục G5

U1 v7 đã được huấn luyện thực sự trên GPU trong 3,000 bước optimizer với batch 24, gradient accumulation 2 (effective batch 48), BF16, 4 CPU thread, 1,000 bước warm-up đầu độ không chắc chắn, và fine-tune chung sau đó. It được khởi tạo từ checkpoint không gian-thời gian T2 v6 đã huấn luyện. Checkpoint EMA được chọn là bước 3,000 với điểm số chọn lọc validation bị nhiễu 0.661517, cải thiện từ 0.725525 tại bước 2,500.

Trên phân tách hiệu chuẩn rời rạc, 667,968 quan sát khớp ghép cặp U0/U1 cho ra:

| Đo lường G5 | Kết quả | Ngưỡng |
|---|---:|---:|
| Spearman tổng thể | **0.817074** | ≥0.35 |
| AUC decile tồi nhất tổng thể | **0.802250** | ≥0.70 |
| AUC thân | **0.939934** | ≥0.70 |
| AUC tay trái | **0.753737** | ≥0.75 |
| AUC tay phải | **0.770316** | ≥0.75 |
| Lỗi nhiễu U0 → U1 | 0.104108 → **0.083074** | cải thiện |
| Lỗi sạch U0 → U1 | 0.091552 → **0.050310** | suy giảm ≤1% |
| rủi ro đơn điệu / tự hiệu chuẩn / NLL U1-vs-U0 | **GO / GO / GO** | tất cả GO |

Do đó mọi điều kiện số học G5 hiện đã đi qua. Báo cáo máy vẫn là `passed: false` chỉ vì `source_and_signer_disjoint_real_residual=false`, vì G5 chính thức fail-closed bổ sung yêu cầu bộ khởi tạo A1 chính xác. Quyết định đúng là **proxy G5 GO, formal G5 NO-GO**; U1 phải giữ ở trạng thái tắt trong suy luận Lane được phát hành cho đến khi có sẵn đệm A1 chính xác bên ngoài.

Artifacts:

- checkpoint được chọn: `outputs/phase2_training/u1_how2sign_2d_temporal_reprojection_v7_seed42/best.pt`, SHA-256 `edc8a035225e246530b80e4482876b921ab1d69ef8211d1afa8c66f063198570`;
- log huấn luyện: `outputs/phase2_gates/logs/u1_how2sign_2d_temporal_reprojection_v7_seed42.txt`, SHA-256 `f8d3d77939d0c3bfb1d93228303edeb2f6de1cfd8a7c88398d2119fb61e72bd0`;
- thặng dư hiệu chuẩn: `outputs/phase2_gates/g5/how2sign_u1_v7_calibration_residuals.npz`, SHA-256 `72f517981a6be2d1379d8ffe273799a650a359c33dfbf74afc741327a93bfeaa`;
- báo cáo hiệu chuẩn: `outputs/phase2_gates/g5/how2sign_u1_v7_calibration_report.json`, SHA-256 `2d0663fa33a307e2e638f20604e75f891b87b71018be67fcd99eb86c96841bca`.

### 31.4 Quyết định cổng thay thế

| Cổng | Quyết định sau hiệu chỉnh | Bằng chứng |
|---|:---:|---|
| G0 | **GO** | bộ đánh giá đóng băng, topology, mặt nạ, quần thể tác giả, và độ bao phủ |
| G1 | **GO** | stack ensemble/fallback A1 hoàn chỉnh cải thiện cả 3 vùng |
| G2 | **GO (được ủy quyền bởi owner)** | kiểm toán thị giác tự động 100/100 với 0 thất bại thảm họa |
| G3 | **GO** | khả năng phục hồi tổng hợp chính thức passed |
| G4 | **proxy GO / formal NO-GO** | tất cả kiểm tra T5 số học pass; bộ khởi tạo bên ngoài không phải là A1 chính xác |
| G5 | **proxy GO / formal NO-GO** | mọi kiểm tra U1 v7 số học pass; bộ khởi tạo bên ngoài không phải là A1 chính xác |
| G6 | **NO-GO** | T2 trực tiếp và T5 chỉ dựa trên quan sát đều thất bại chuyển giao Lane đã khóa |
| G7 | **GO theo phạm vi dự án** | bộ đánh giá 57 ký hiệu/1,493 khung hình của tác giả là chuẩn; 2,872 nằm ngoài phạm vi |

**Phase 2 đầy đủ vẫn là NO-GO vì G4/G5 thiếu nguồn gốc A1 chính xác và G6 thất bại — không phải vì G7.** Lỗi triển khai U1 đã được sửa và chiến lược lùi an toàn T5 đã được kiệt xuất. Con đường duy nhất hợp lệ về mặt khoa học để đạt full GO là thêm các đầu ra nhà cung cấp A1 chính xác bên ngoài, cụ thể hóa chúng với nạp đệm ràng buộc hash mới, làm phong phú thặng dư thực của chúng, huấn luyện lại T2/U1 trên miền đó, đóng băng 1 lần, và sau đó lặp lại G4/G5 tiếp theo là chuỗi G6 được khóa. Luồng-L không được dùng để tinh chỉnh một kiến trúc hay ngưỡng khác.

### 31.5 Các file được triển khai hoặc cập nhật trong bản khắc phục này

Các file mới:

- `phase2_refiner/t5_optimize.py`;
- `phase2_refiner/data/materialize_exact_a1_cache.py`;
- `phase2_refiner/data/audit_reprojection_domain_shift.py`;
- `phase2_refiner/configs/exact_a1_provenance_v1.json`;
- `phase2_refiner/configs/uawsr_t2_how2sign_2d_temporal_t5.yaml`;
- `phase2_refiner/configs/uawsr_u1_how2sign_2d_temporal_v7.yaml`;
- `phase2_refiner/tests/test_t5.py`.

Các file cập nhật:

- `phase2_refiner/config.py`, `infer.py`, `evaluate_residual_checkpoint.py`, và `evaluate_lane_diagnostics.py`;
- `phase2_refiner/models/spatial_temporal_refiner.py`;
- `phase2_refiner/losses/sequence.py` và `losses/uncertainty.py`;
- `phase2_refiner/data/add_reprojection_residuals.py`, `data/audit_real_residual_cache.py`, và `data/corruptions.py`;
- `phase2_refiner/tests/test_cache.py`, `test_calibrate.py`, `test_corruptions.py`, và `test_model.py`;
- `phase2_refiner/README.md` và báo cáo này.

Tính tương thích ngược được bảo toàn: các cấu hình lịch sử giữ lại phản hồi độ không chắc chắn trừ khi tắt rõ ràng, T5 mặc định off, và không có phương pháp hay đầu ra trước đó nào bị ghi đè. Xác minh pass **48/48 kiểm thử Phase 2**, các kiểm tra Ruff đầy đủ, và biên dịch byte Python. Artifact phạm vi dự án G7 là `outputs/phase2_gates/g7/project_scope_author_1493_v1.json`, SHA-256 `733612b1a44aee3beeb6da2c59a3b2a3ed276ffc9cca73fe05d4c869a7cd432f`.
