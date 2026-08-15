# Đề xuất Nghiên cứu Lấy Phương pháp làm Trung tâm cho Phục dựng Ngôn ngữ Ký hiệu 3D Đạt Chuẩn SOTA

**Tài liệu ý tưởng (Working paper concept):** **SignPosterior4D: Phonology- and Interaction-Conditioned Whole-Sequence Posterior Reconstruction** *(Phục dựng Hậu phương Toàn Chuỗi Điều kiện hóa theo Âm tiết học và Tương tác)*

**Tập chuẩn chính (Primary benchmark):** SGNify, sử dụng giao thức TR-V2V chuẩn được áp dụng bởi SGNify và DexAvatar

**Đầu ra mục tiêu:** Chuyển động cơ thể trên SMPL-X (không gồm khuôn mặt) và hai bàn tay được phục dựng từ video ngôn ngữ ký hiệu đơn nhãn (monocular video)

**Mục tiêu nghiên cứu:** một phương pháp có tính khả thi kỹ thuật cao, công khai, có thể tái lập, phù hợp để gửi tới các hội nghị thị giác máy tính hạng A* (A*-level vision conference)

**Ngày kiểm định (Audit date):** 13 tháng 7 năm 2026

---

## 1. Phạm vi và quyết định nghiên cứu

Tài liệu này coi việc đánh giá TR-V2V trên SGNify là một tập chuẩn (benchmark) cố định. Tài liệu **không** đề xuất việc thay đổi, chỉ trích hay tối ưu hóa quanh bộ đánh giá (evaluator). Nhiệm vụ là xây dựng một phương pháp phục dựng thực sự tốt hơn giúp hạ thấp:

- TR-V2V phần thân trên (không gồm khuôn mặt);
- TR-V2V bàn tay trái; và
- TR-V2V bàn tay phải.

Phân tích dựa trên:

1. [Bài báo DexAvatar WACV 2026 và phụ lục](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html);
2. [Kho lưu trữ DexAvatar chính thức](https://github.com/kaustesseract/DexAvatar);
3. Mã nguồn gốc nguyên bản chứa trong lịch sử kho lưu trữ tại commit `7e97916`, thay vì các phần bổ sung thử nghiệm sau này như NLF, WiLoR, DPoser-X hay các thành phần khác trong nhánh làm việc hiện tại; và
4. [Bài báo SGNify](https://openaccess.thecvf.com/content/CVPR2023/html/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.html), nơi giới thiệu giao thức benchmark này.

### Khuyến nghị trung tâm

**Không** đưa ra đóng góp chính của bài báo dưới dạng “chúng tôi thay thế HaMeR bằng bộ ước lượng bàn tay mới hơn” hoặc “chúng tôi thay thế VAE bằng diffusion.” Đây là các điểm mốc cơ bản (baseline) hữu ích, nhưng không còn đủ tính mới mang tính đột phá.

Bài báo đề xuất nên tái cấu trúc bài toán phục dựng thành **suy luận phân bố hậu phương toàn chuỗi (whole-sequence posterior inference) dưới một mô hình chuyển động ngôn ngữ ký hiệu có cấu trúc**:

> Cho trước các quan sát thiếu chắc chắn về cơ thể, bàn tay, hình ảnh và điểm đặc trưng (keypoint), suy luận ra chuỗi SMPL-X phối hợp có xác suất cao nhất trong khi vẫn bảo toàn dạng bàn tay (handshape), hướng lòng bàn tay (palm orientation), vị trí (location), pha chuyển động (movement phase), tính đối xứng/ưu thế (symmetry/dominance) và quan hệ tiếp xúc (contact relations) của ký hiệu.

Phương pháp được đề xuất, với tên gọi tạm thời là **SignPosterior4D (SP4D)**, sở hữu một ý tưởng chủ đạo nhất quán:

> Một phân bố tiền phương diffusion quan hệ (relational diffusion prior) điều kiện hóa theo âm tiết học giúp phục dựng đồng thời phần thân trên, cổ tay và cả hai bàn tay trên toàn bộ chuỗi ký hiệu, trong khi một mô hình độ không đảm bảo (uncertainty model) sẽ quyết định khi nào nên tin tưởng quan sát thị giác và khi nào nên suy luận thông qua ngữ cảnh thời gian và giữa các bộ phận.

Hướng đi này đánh trực tiếp vào những hạn chế phương pháp luận lớn nhất của DexAvatar và vẫn giữ được sự biệt lập rõ rệt so với các nghiên cứu gần đây về phục hồi toàn cơ thể theo thời gian tổng quát, ghép nối tay-thân (hand-body stitching) và diffusion tư thế tổng quát.

---

## 2. Mục tiêu cạnh tranh

### 2.1 Kết quả đã xuất bản và báo cáo mới

| Phương pháp | Thân trên | Bàn tay trái | Bàn tay phải | Ý tưởng chính |
|---|---:|---:|---:|---|
| SGNify | 55.63 | 19.22 | 17.50 | Ràng buộc đối xứng bàn tay và bất biến tư thế |
| Neural Sign Actors | 46.42 | 16.17 | 15.23 | Gãn nhãn SMPL-X tinh chỉnh cho sản xuất ký hiệu |
| EVA* | 40.38 | 13.73 | 13.68 | Nhiều nguồn giả giám sát off-the-shelf |
| DexAvatar | **30.13** | **13.53** | **13.08** | VAE cơ thể/tay tĩnh đặc thù cho ký hiệu cộng với khớp nối (fitting) |
| Tamaththul3D preprint | 29.28 | 10.65 | 8.90 | Chuyển đổi WiLoR sang SMPL-X, dóng hàng hình học cổ tay/cẳng tay, tinh chỉnh vai |

Mục tiêu chính thức của DexAvatar là `30.13 / 13.53 / 13.08`. Tuy nhiên, bản [preprint Tamaththul3D v2](https://arxiv.org/html/2605.05367v2) tháng 6 năm 2026 báo cáo chỉ số `29.28 / 10.65 / 8.90` trên SGNify. Hãy coi đây là kết quả hiện đại mạnh nhất cần tái lập với bộ cài đặt TR-V2V cố định của dự án, sau đó đặt mục tiêu vượt qua **cả** Tamaththul3D lẫn DexAvatar.

### 2.2 Các cổng thành công (Success Gates) đề xuất

Đây là các mục tiêu kỹ thuật, không phải là kết quả cam kết.

| Cổng | Thân trên | Bàn tay trái | Bàn tay phải | Ý nghĩa |
|---|---:|---:|---:|---|
| Tái lập DexAvatar | xấp xỉ 30.13 | xấp xỉ 13.53 | xấp xỉ 13.08 | Xác nhận đường ống (pipeline) gốc |
| Baseline hiện đại mạnh | < 29.3 | < 10.7 | < 8.9 | Cạnh tranh với kết quả cao nhất hiện được báo cáo |
| Mục tiêu đáng để viết bài | **< 27.5** | **< 9.5** | **< 8.0** | Cải thiện rõ rệt với không gian phát triển vượt khỏi việc thay thế bộ ước lượng |
| Mục tiêu đột phá (Stretch target) | < 26.5 | < 9.0 | < 7.5 | Kết quả SOTA vượt trội nếu được hỗ trợ bởi thực nghiệm tổng quát hóa |

Một sự chênh lệch số học nhỏ là chưa đủ cho một bài nộp cấp A*. Mục tiêu là một phương pháp có mức cải thiện lớn nhất chính tại các khoảng nhòe chuyển động (blur), che khuất (occlusion), tương tác bàn tay và các chuyển tiếp nhanh—và các phân tích triệt tiêu (ablation) chứng minh được lý do tại sao.

---

## 3. Bản chất hoạt động thực sự của DexAvatar gốc

DexAvatar là một đường ống tối ưu hóa (optimization pipeline), không phải là một mạng phục dựng video end-to-end.

### 3.1 Đường ống quan sát và khởi tạo

Đối với mỗi khung hình video, bản phát hành gốc thu được:

- Các tham số cơ thể, camera, hình dạng và bàn tay ban đầu trên SMPL-X từ SMPLer-X;
- Khớp 2D toàn cơ thể từ Sapiens;
- Ảnh cắt bàn tay, khớp bàn tay 2D/3D và ước lượng MANO từ HaMeR; và
- Lớp ký hiệu một tay/hai tay từ bộ phân loại của SGNify.

Sau đó, nó tối ưu hóa các biến ẩn (latent variables) số chiều thấp được giải mã bởi SignBPoser và SignHPoser và render ra lưới SMPL-X.

### 3.2 Các phân bố tiền phương đã học (Learned priors)

DexAvatar huấn luyện hai tiền phương VAE 3 lớp:

- **SignBPoser**, một không gian ẩn cơ thể 33 chiều được huấn luyện bằng các chuỗi giả SMPL-X đã qua lọc trích xuất từ video ký hiệu; và
- **SignHPoser**, một không gian ẩn bàn tay 23 chiều được huấn luyện từ găng tay/hệ thống Vicon thu thập thao tác đánh vần ngón tay (fingerspelling) của 8 người ký hiệu.

Các tiền phương này là các phân bố tư thế theo từng khung hình (frame-pose distributions). Chúng không phải là mô hình chuỗi và không biểu diễn đồng thời cơ thể, cổ tay, tay trái và tay phải.

### 3.3 Hàm mục tiêu khớp nối gốc

Bài báo thể hiện tổn thất khớp nối (fitting loss) dưới dạng:

$$
\mathcal{L}_{\text{Dex}} =
\mathcal{L}_{\text{joint}}
+ \lambda_b \mathcal{L}_{\text{B-prior}}
+ \lambda_h \mathcal{L}_{\text{H-prior}}
+ \lambda_{\text{pen}} \mathcal{L}_{\text{pen}}
+ \lambda_t \mathcal{L}_{\text{temp}}
+ \lambda_{bb} \mathcal{L}_{\text{body-bio}}
+ \lambda_{hb} \mathcal{L}_{\text{hand-bio}}.
$$

Trong ảnh chụp mã nguồn gốc (snapshot):

- Bộ tối ưu chủ yếu cập nhật các biến ẩn của SignBPoser và SignHPoser;
- Biến ẩn cơ thể và bàn tay bắt đầu từ giá trị 0 ở mỗi khung hình;
- Các tư thế được giải mã bị neo chặt vào các tham số khởi tạo từ SMPLer-X và HaMeR;
- Thành phần 3D khớp tay hiện có chỉ so sánh **tọa độ chiều sâu** tương đối so với cổ tay và được chuẩn hóa độc lập, thay vì tọa độ XYZ mét đầy đủ; trong file YAML phát hành, toàn bộ `data_3d_weights` bị đặt về 0; và
- Thành phần thời gian là một hình phạt bậc nhất lên tư thế cơ thể so với khung hình được xử lý liền trước.

Các đường dẫn gốc liên quan là [fitting.py](../dexavatar_fitting/smplifyx/fitting.py), [fit_single_frame.py](../dexavatar_fitting/smplifyx/fit_single_frame.py), [main.py](../dexavatar_fitting/smplifyx/main.py), và [fit_smplx_vposer_x.yaml](../dexavatar_fitting/cfg_files/fit_smplx_vposer_x.yaml). Vì các tệp hiện tại chứa các thử nghiệm sau này, hãy dùng lệnh `git show 7e97916:<path>` khi kiểm tra bản cài đặt đã phát hành.

### 3.4 Lý do DexAvatar tạo ra bước tiến lớn

Nhận thức cốt lõi của DexAvatar vẫn hoàn toàn đúng: một phân bố tư thế chuyên biệt cho ngôn ngữ ký hiệu hiệu quả hơn nhiều so với một tiền phương chuyển động hàng ngày tổng quát. Việc lọc dữ liệu của họ đã giúp giảm sai số cơ thể từ `34.06` xuống `30.28`, và việc huấn luyện tiền phương bàn tay đã sửa giúp giảm 3 sai số báo cáo từ `31.34 / 14.19 / 13.92` xuống `30.17 / 13.55 / 13.06` trước khi áp dụng thành phần sinh học cơ học cuối cùng.

Bài báo tiếp theo nên bảo tồn nhận thức đặc thù ký hiệu này trong khi thay thế tiền phương độc lập, tĩnh, định tính bằng một mô hình phối hợp, theo thời gian và nhận biết độ không đảm bảo.

---

## 4. Các điểm yếu phương pháp giới hạn trực tiếp chỉ số TR-V2V

Phần này chỉ phân tích các điểm yếu phục dựng. Chúng được sắp xếp theo mức độ ảnh hưởng dự kiến tới 3 điểm số benchmark.

### 4.1 Phương pháp không phục dựng ký hiệu dưới dạng một chuỗi

DexAvatar xử lý từng khung hình một cách nối tiếp và dùng tư thế cơ thể trước đó làm tham chiếu làm mượt cục bộ. Nó không bao giờ thấy các khung hình tương lai khi giải quyết khung hình hiện tại.

Hệ quả:

- Một khung hình bị nhòe không thể mượn thông tin từ một khung hình rõ nét sau đó 5 khung hình;
- Một bàn tay tạm thời bị che bởi tay kia không thể suy luận được từ cấu hình đầu vào và đầu ra một cách đồng thời;
- Sai số có thể lan truyền theo chiều thời gian từ khung hình này sang khung hình khác;
- Việc làm mượt bậc nhất trừng phạt cả các chuyển tiếp nhanh hợp lệ; và
- Cử động ngón tay không nhận được ràng buộc toàn chuỗi học được tương đương.

Đây là điểm yếu quan trọng nhất đối với một bài toán xử lý video.

### 4.2 Cơ thể và bàn tay độc lập về mặt thống kê

SignBPoser và SignHPoser được huấn luyện và giải mã riêng biệt. Hai bàn tay cũng sử dụng các vectơ ẩn độc lập.

Nhưng chuyển động ký hiệu là sự phối hợp:

- Hướng cổ tay giới hạn sự xoay hợp lý của cẳng tay;
- Quỹ đạo cánh tay dự đoán bàn tay sẽ xuất hiện ở đâu và như thế nào;
- Các ký hiệu đối xứng có sự tương quan giữa dạng và chuyển động bàn tay trái/phải;
- Tư thế bàn tay thụ động phụ thuộc vào bàn tay chủ động và hình học tiếp xúc; và
- Quan hệ tay-mặt, tay-thân và tay-tay tồn tại kéo dài qua thời gian.

Một tiền phương độc lập có thể tạo ra các bộ phận hợp lý về mặt giải phẫu nhưng lại bất nhất khi kết hợp với nhau. Điều này ảnh hưởng đến TR-V2V thân trên thông qua sai số cánh tay/cổ tay và TR-V2V bàn tay thông qua sai hướng lòng bàn tay và khớp ngón tay.

### 4.3 SignHPoser không thể học được sự phối hợp cổ tay – cẳng tay

Phụ lục bài báo nêu rõ rằng các phép xoay cổ tay từ dữ liệu MANUS không thể chuyển giao tin cậy do sự không tương thích về T-pose và bone-roll của bộ xương. Kết quả là SignHPoser chỉ học khớp ngón tay chứ không học được mối quan hệ hướng bàn tay toàn cục so với cổ tay và cẳng tay.

Đây là một hạn chế về mặt cấu trúc, không đơn thuần là thiếu dữ liệu. Hướng lòng bàn tay chính xác có ý nghĩa quan trọng về mặt ngữ âm học và đóng góp rất lớn vào sai số đỉnh (vertex error) ngay cả khi các góc ngón tay trông có vẻ hợp lý.

### 4.4 Giám sát bàn tay bỏ qua hầu hết thông tin 3D độ đo (metric 3D)

Mã nguồn khớp nối gốc cài đặt một thành phần khớp bàn tay chỉ chọn kênh chiều sâu, biến nó thành tương đối so với cổ tay và chuẩn hóa dự đoán lẫn quan sát một cách độc lập. Cấu hình phát hành đặt trọng số lập lịch của thành phần này bằng 0, do đó giám sát bàn tay thực tế đến từ các khớp 2D và các neo tham số tư thế HaMeR mạnh. Ngay cả khi được bật, thành phần này cũng chỉ cung cấp một tín hiệu yếu về cấu hình 3D thực tế của bàn tay.

Tối ưu hóa không khai thác đầy đủ:

- Hình học xương XYZ độ đo;
- Mặt phẳng lòng bàn tay và pháp tuyến lòng bàn tay;
- Hướng từ cổ tay đến khớp MCP;
- Vị trí đầu ngón tay nhất quán theo tỷ lệ (scale-consistent);
- Tư thế bàn tay trái/phải tương đối; hoặc
- Vị trí bàn tay so với cơ thể.

Một phương pháp hiện đại nên tiếp nhận các giả thuyết 3D đầy đủ và biểu diễn độ không đảm bảo của chúng thay vì tiêu giảm chúng thành chiều sâu chuẩn hóa.

### 4.5 Bộ tối ưu bị buộc chặt vào một bộ khởi tạo có thể sai

Các tư thế cơ thể và bàn tay được giải mã bị giám sát trực tiếp bởi các tham số SMPLer-X và HaMeR, với trọng số khởi tạo lớn ở tất cả các giai đoạn khớp nối. Khi một bộ khởi tạo thất bại dưới điều kiện mờ hoặc che khuất, tiền phương học được bị yêu cầu phải duy trì vị trí gần với sai số đó.

Do đó hệ thống hoạt động giống như việc khử nhiễu có ràng buộc (constrained denoising) hơn là suy luận hậu phương thực sự. Nó không có cơ chế để duy trì nhiều giả thuyết hợp lý và chọn ra một giả thuyết bằng cách dùng bằng chứng từ nơi khác trong video.

### 4.6 Các tiền phương tư thế là các VAE tĩnh đơn thức (unimodal)

Một VAE Gaussian nhỏ thì hiệu quả, nhưng nó có xu hướng trung bình hóa các cấu hình mơ hồ. Điều này đặc biệt gây hại cho:

- Độ gập ngón tay có phép chiếu 2D tương tự nhau;
- Sự mơ hồ giữa hướng lòng bàn tay hướng vào trong so với hướng ra ngoài;
- Bàn tay bị bắt chéo hoặc bị che khuất;
- Sự đối ứng ngón cái (thumb opposition); và
- Các dạng bàn tay hiếm nhưng hợp lệ.

Giải pháp không chỉ là cái tên “diffusion”. Bước tiến hữu ích là một **phân bố chuỗi có điều kiện** có thể biểu diễn nhiều chuyển động hợp lý và sử dụng các quan sát video để thu gọn phân bố hậu phương.

### 4.7 Tính nhất quán theo thời gian không nhận biết được pha (phase aware)

Ngôn ngữ ký hiệu bao gồm các điểm dừng (holds), chuyển tiếp (transitions), lặp lại (repetitions) và các chuyển động hướng nhanh. Một hình phạt vận tốc bằng 0 hoặc đạo hàm đồng nhất sẽ đối xử với các pha có ý nghĩa khác nhau này như cùng một quá trình.

Nó có thể:

- Làm mượt quá mức một sự thay đổi nhanh giữa các dạng bàn tay;
- Xóa bỏ đỉnh (apex) của một chuyển động;
- Giữ lại sự rung lắc (jitter) trong một điểm dừng cố ý nếu thành phần dữ liệu mạnh; và
- Đưa vào độ trễ (lag) vì chỉ sử dụng chuyển động trong quá khứ.

Một mô hình nhận biết pha nên cứng vững trong các điểm dừng và linh hoạt trong các chuyển tiếp có ý nghĩa.

### 4.8 Quyết định ký hiệu cứng bỏ qua chuyển động yếu nhưng hữu ích

Đối với một ký hiệu một tay được dự đoán, quá trình khớp nối gốc sẽ tắt cánh tay và bàn tay không ưu thế. Trong ký hiệu thực tế, bên không ưu thế vẫn có thể thể hiện sự điều chỉnh tư thế, chuyển động chuẩn bị hoặc sự ổn định.

Một biến ưu thế mang tính xác suất sẽ tốt hơn một công tắc cứng. Nó có thể chuẩn hóa mạnh mẽ phía thụ động mà không ép nó phải giữ nguyên không đổi.

### 4.9 Tiếp xúc được coi là tránh va chạm chứ không phải quan hệ ký hiệu

Tổn thất xuyên thấu (interpenetration loss) ngăn chặn hình học không hợp lệ, nhưng ngôn ngữ ký hiệu thực tế thường xuyên chứa các tiếp xúc cố ý. Việc chỉ tránh va chạm không thể cho hệ thống biết:

- Bàn tay nào ở phía trước;
- Đầu ngón tay nào chạm vào bề mặt nào;
- Khi nào tiếp xúc bắt đầu và kết thúc;
- Liệu tiếp xúc có nên duy trì qua một điểm dừng hay không; hoặc
- Hai bàn tay di chuyển cùng nhau như thế nào sau khi tiếp xúc.

Tiếp xúc phải là một trạng thái quan hệ được dự đoán, không chỉ là một thành phần đẩy lùi.

### 4.10 Độ tin cậy quan sát chưa đầy đủ

DexAvatar trọng số các điểm đặc trưng 2D theo độ tin cậy của bộ phát hiện, nhưng nó không ước lượng độ không đảm bảo được hiệu chuẩn (calibrated uncertainty) cho tư thế SMPL-X, khớp MANO, hướng lòng bàn tay hoặc tính nhất quán thời gian.

Chỉ riêng độ tin cậy của bộ phát hiện là không đủ: một bộ ước lượng bàn tay có thể tự tin sai dưới sự mơ hồ đối gương, che khuất nghiêm trọng hoặc ảnh cắt kém. Sự bất đồng giữa các bộ ước lượng và sự bất nhất theo thời gian cung cấp các tín hiệu độ không đảm bảo bổ sung giá trị.

### 4.11 Dữ liệu huấn luyện chỉ bao phủ tư thế chứ không bao phủ toàn bộ quá trình thất bại phục dựng

Tiền phương thấy các mẫu tư thế, nhưng không được huấn luyện một cách tường minh để sửa chữa các lỗi bộc phát (burst errors) tạo ra bởi các bộ ước lượng hình ảnh dưới điều kiện mờ, che khuất, cắt mép và tương tác tay. Nhiễu Gaussian ngẫu nhiên không phải là sự thay thế thỏa đáng cho thất bại thực tế của bộ ước lượng.

Mô hình mới nên được huấn luyện với các sai số còn lại (residuals) tạo ra bởi chính các bộ khởi tạo công khai được dùng lúc suy luận, cộng với các mặt nạ bộc phát giả lập và độ mờ. Điều này giúp dóng hàng quá trình huấn luyện với bài toán ngược thực tế.

---

## 5. Lý do các nâng cấp hiển nhiên là chưa đủ cho bài báo

### 5.1 Thay thế WiLoR là một baseline bắt buộc, không phải tính mới

[WiLoR](https://github.com/rolpotamias/WiLoR) là một bộ định vị và phục dựng bàn tay công khai mạnh tại CVPR 2025. Tamaththul3D đã cho thấy việc thay thế các tham số bàn tay và dóng hàng cổ tay có thể giảm đáng kể sai số bàn tay báo cáo. Do đó:

- Sử dụng WiLoR làm bộ khởi tạo mạnh;
- Tái lập việc chuyển đổi trực tiếp MANO sang SMPL-X;
- Tái lập baseline dóng hàng cổ tay/cẳng tay vi phân được; nhưng
- Đừng trình bày riêng điều này như một bài báo mới.

### 5.2 Dung hợp cơ thể–bàn tay theo thời gian tổng quát đã có người làm

Bản preprint [DanceHMR](https://arxiv.org/html/2605.18102) tháng 5 năm 2026 đã dung hợp đồng thời quan sát cơ thể và bàn tay trước khi mô hình hóa thời gian, bổ sung tăng cường cận cảnh, giám sát nhận biết độ hiển thị và huấn luyện tập trung vào đầu ngón tay. Một "Transformer thời gian cho cơ thể và bàn tay" tổng quát sẽ bị trùng lặp rất nhiều.

Điểm biệt lập của chúng ta phải đặc thù cho ký hiệu và rõ ràng: âm tiết học cấu thành (compositional phonology), động lực học pha ký hiệu, tiếp xúc quan lệ, lấy mẫu hậu phương được hiệu chuẩn và sự bảo toàn ngữ nghĩa được kiểm định.

### 5.3 Diffusion toàn cơ thể tổng quát đã có người làm

[DPoser-X](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_DPoser-X_Diffusion_Model_as_Robust_3D_Whole-body_Human_Pose_Prior_ICCV_2025_paper.html) đã sử dụng masked diffusion để mô hình hóa sự phụ thuộc giữa các bộ phận toàn cơ thể, trong khi [FUSION](https://arxiv.org/abs/2601.03959) mô hình hóa chuyển động chung cơ thể và bàn tay. Việc thay thế SignBPoser bằng một trong hai tiền phương này là hữu ích, nhưng không đủ tính mới.

Đóng góp đề xuất là một **phân bố hậu phương ký hiệu (sign posterior)**, không phải tiền phương chuyển động tổng quát: nó điều kiện hóa dựa trên cấu trúc âm tiết học suy luận được và các quan sát quan hệ tay–tay/tay–cơ thể tường minh và được tối ưu hóa so với các quan sát video thiếu chắc chắn.

### 5.4 Làm mượt sau (Post-hoc smoothing) đã có người làm và có thể hại ký hiệu

Tamaththul3D sử dụng làm mượt vận tốc, gia tốc và jerk. Các phương pháp bàn tay theo thời gian và DanceHMR cũng xử lý sự rung lắc. Làm mượt đồng nhất không giải quyết được sự mơ hồ và có thể xóa mất khớp động nhanh có ý nghĩa.

Bài báo thay vào đó nên học khi nào một khung hình thuộc về điểm dừng, chuyển tiếp, lặp lại hoặc khoảng tiếp xúc và điều chỉnh động lực học hậu phương cho phù hợp.

### 5.5 Giới hạn sinh học cơ học mạnh hơn còn ít tiềm năng phát triển

Thành phần bàn tay sinh học cơ học cuối cùng của DexAvatar cải thiện bàn tay trái một cách nhỏ lẻ và làm tệ đi một chút ở bàn tay phải. Giới hạn khớp cứng hơn có thể cải thiện tính hợp lý về mặt thị giác mà không làm giảm TR-V2V. Hãy dùng sinh học cơ học như một thành phần chuẩn hóa an toàn (safety regularizer), không phải hướng nghiên cứu trung tâm.

---

## 6. Phương pháp đề xuất: SignPosterior4D

### 6.1 Phát biểu bài toán

Cho trước một chuỗi RGB đơn nhãn:

$$
\mathbf{I}_{1:T} = \{I_1, \ldots, I_T\},
$$

khôi phục một chuỗi SMPL-X nhất quán theo thời gian:

$$
\mathbf{X}_{1:T} = \{X_1, \ldots, X_T\}
$$

biểu diễn tư thế thân trên, hướng cả hai cổ tay, khớp cả hai bàn tay, hình dạng cơ thể dùng chung và các tham số camera.

Thay vì giải quyết từng khung hình độc lập, hãy suy luận:

$$
p(\mathbf{X}_{1:T} \mid \mathbf{O}_{1:T}, \mathbf{Z}_{\text{ph}}, \mathbf{R}_{1:T}),
$$

trong đó:

- $\mathbf{O}_{1:T}$ là các quan sát thiếu chắc chắn từ RGB, điểm đặc trưng 2D, bộ ước lượng cơ thể và bộ ước lượng bàn tay;
- $\mathbf{Z}_{\text{ph}}$ là biểu diễn âm tiết học xác suất được suy luận từ video; và
- $\mathbf{R}_{1:T}$ chứa các trạng thái quan hệ tay–tay và tay–cơ thể.

Sự chuyển dịch quan trọng là từ **khớp một tư thế định tính vào các ước lượng nhiễu** sang **suy luận một phân bố trên toàn bộ ký hiệu phối hợp và chọn ra mẫu được hỗ trợ tốt nhất bởi video**.

### 6.2 Biểu diễn trạng thái chuỗi

Sử dụng phép xoay 6D liên tục bên trong mô hình và chỉ chuyển sang axis-angle khi gọi SMPL-X. Định nghĩa mỗi khung hình là:

$$
X_t = [
\theta_t^{\text{torso}},
\theta_t^{\text{arms}},
R_{t,w}^{L}, R_{t,w}^{R},
\theta_{t,h}^{L}, \theta_{t,h}^{R},
\beta,
\pi_t
].
$$

Chuẩn hóa khuyến nghị:

- Phép xoay thân trên và vị trí khớp trong hệ tọa độ lấy thân người (torso) làm trung tâm;
- Khớp bàn tay trong hệ tọa độ lấy cổ tay làm trung tâm;
- Hướng lòng bàn tay được giữ tường minh so với cẳng tay và thân người;
- Hình dạng dùng chung $\beta$ trên toàn chuỗi; và
- Camera chỉ xuất hiện trong mô hình quan sát, không nằm bên trong tiền phương chuyển động ký hiệu.

Sự tách biệt này ngăn biến đổi camera tiêu tốn dung lượng của tiền phương chuyển động trong khi vẫn giữ được hướng cổ tay cần thiết cho các đỉnh lưới chính xác.

### 6.3 Ngân hàng quan sát (Observation bank)

Bản cài đặt ban đầu nên sử dụng các chuyên gia công khai được đóng đóng băng (frozen public experts):

- **Cơ thể:** SMPLer-X là baseline bắt buộc; tùy chọn SMPLest-X hoặc Hand4Whole++ như một giả thuyết bổ sung;
- **Bàn tay:** WiLoR làm giả thuyết MANO chính; tùy chọn HaMeR, khớp NLF, mẫu MaskHand hoặc OmniHands để tăng tính đa dạng;
- **Bằng chứng 2D:** Khớp toàn cơ thể Sapiens hoặc RTMPose và khớp bàn tay độ phân giải cao;
- **Vẻ ngoài (Appearance):** Đặc trưng toàn cơ thể cộng với đặc trưng cắt riêng tay trái/phải; và
- **Chuyển động (Motion):** Các vệt đặc trưng ngắn hạn hoặc luồng thị giác (optical flow) quanh cổ tay, đầu ngón tay và các mép tay nhìn thấy được.

Cấu hình bắt buộc chỉ được phụ thuộc vào mã nguồn và checkpoint công khai—tối thiểu là SMPLer-X và WiLoR công khai. Các chuyên gia bổ sung như MaskHand hoặc OmniHands nên là tùy chọn trừ khi có sẵn trọng số và giấp phép tương thích. Bài báo và bản phát hành phải làm cho cấu hình chuyên gia đơn công khai tái lập hoàn toàn; các tập hợp chuyên gia tùy chọn thuộc về một ablation riêng.

Đối với mỗi khớp và khung hình, dựng một token quan sát:

$$
O_{t,j} = [\hat{x}_{t,j}^{2D}, \hat{x}_{t,j}^{3D},
\hat{R}_{t,j}, f_{t,j}, c_{t,j}, v_{t,j}, d_{t,j}],
$$

chứa ước lượng, đặc trưng hình ảnh, độ tin cậy của bộ phát hiện, độ hiển thị và sự bất đồng giữa các chuyên gia.

Phương pháp không nên nối các ước lượng một cách mù quáng. Một đầu độ tin cậy (reliability head) học được sẽ dự đoán độ không đảm bảo dị sai (heteroscedastic uncertainty):

$$
(\mu_{t,j}, \log \sigma^2_{t,j}) = g_{\omega}(O_{t,j}, O_{t-1:t+1,j}).
$$

Huấn luyện đầu này trên các tập dữ liệu có giám sát 3D và các nhiễu thực tế. Lúc suy luận, độ không đảm bảo cao sẽ giảm ảnh hưởng của quan sát và để tiền phương ký hiệu cùng các khung hình lân cận lấp đầy khoảng trống.

### 6.4 Biến ẩn âm tiết học cấu thành (Compositional phonology latent)

Biểu diễn âm tiết học là điểm biệt lập chính đặc thù ký hiệu so với HMR video tổng quát. Nó không được yêu cầu nhãn từ (gloss) đã biết tại thời điểm kiểm thử.

Suy luận một biến ẩn mềm:

$$
Z_{\text{ph}} = [
H^L, H^R,
O^L, O^R,
L^L, L^R,
M^L, M^R,
S, D, C
],
$$

trong đó:

- $H$: các token dạng bàn tay (handshape);
- $O$: hướng lòng bàn tay và ngón tay;
- $L$: vị trí trong không gian ký hiệu lấy thân làm trung tâm;
- $M$: loại và hướng chuyển động;
- $S$: quan hệ đối xứng;
- $D$: xác suất ưu thế/hoạt động; và
- $C$: loại tiếp xúc tay-tay hoặc tay-cơ thể.

#### Cách thu thập giám sát

Sử dụng 3 nguồn bổ trợ cho nhau:

1. **Nhãn trích xuất hình học.** Trích xuất pháp tuyến lòng bàn tay, cụm dạng bàn tay, hướng quỹ đạo, sự đối xứng, tính hoạt động và tiếp xúc trực tiếp từ các chuỗi 3D tinh lọc.
2. **Giám sát HamNoSys.** Tập dữ liệu [SignAvatars](https://signavatars.github.io/) bao gồm gợi ý HamNoSys, từ và câu cho một phần trong 70.000 chuỗi của nó và có thể cung cấp cấu trúc âm tiết học tường minh.
3. **Tự giám sát từ video.** Dự đoán các thuộc tính từ chuỗi RGB, phục dựng chuỗi 3D, và yêu cầu bộ trích xuất hình học định tính áp dụng lên bản phục dựng phải khôi phục lại đúng các thuộc tính đó.

Điều này tạo ra một vòng lặp ngữ nghĩa – động học (semantic–kinematic cycle):

$$
\hat{Z}_{\text{ph}} = q_{\phi}(I_{1:T}),
\qquad
\tilde{Z}_{\text{ph}} = F_{\text{geom}}(X_{1:T}),
$$

$$
\mathcal{L}_{\text{ph-cycle}} =
D(\hat{Z}_{\text{ph}}, \tilde{Z}_{\text{ph}}).
$$

Khác với sự lựa chọn 6 nhóm cứng của SGNify, biểu diễn này mang tính cấu thành, xác suất và nhận biết thời gian. Nó có thể biểu thị độ không đảm bảo—ví dụ `0.65 đối xứng` thay vì ép buộc một quyết định nhị phân.

### 6.5 Đồ thị quan hệ tay – cơ thể

Dựng một đồ thị quan hệ nhỏ tại mỗi khung hình. Các nút biểu diễn:

- Cả hai cổ tay;
- Tâm và pháp tuyến lòng bàn tay;
- Đầu ngón tay và các khớp MCP;
- Đầu, cằm, ngực, vai và bắp tay; và
- Các neo tiếp xúc dày đặc tùy chọn trên SMPL-X.

Các cạnh mã hóa:

- Vectơ từ cổ tay đến cổ tay;
- Khoảng cách giữa các đầu ngón tay và giữa hai lòng bàn tay;
- Khoảng cách có dấu tay-mặt và tay-thân;
- Hướng lòng bàn tay tương đối;
- Thứ tự trước/sau; và
- Xác suất và sự duy trì tiếp xúc.

Các quan hệ này ổn định hơn dưới sự dịch chuyển toàn cục và mang nhiều thông tin hơn dưới sự che khuất so với hai tư thế bàn tay độc lập. Một bộ mã hóa quan hệ tạo ra $R_t$, điều kiện hóa cả tiền phương và bước tinh chỉnh cuối cùng.

Đối với một cạnh tiếp xúc được dự đoán $(a,b)$, sử dụng khoảng cách mục tiêu mềm thay vì chỉ đẩy lùi thuần túy:

$$
\mathcal{L}_{\text{contact}} =
\sum_{t,a,b} c_{t,a,b}
\rho\left(d(V_{t,a}, V_{t,b}) - \delta_{a,b}\right)
+ \lambda_{\text{slip}} c_{t,a,b}c_{t-1,a,b}
\left\|\Delta r_{t,a,b}\right\|_1.
$$

Giữ một thành phần chống xuyên thấu riêng biệt cho các cặp đỉnh không tiếp xúc.

### 6.6 Phân bố tiền phương masked diffusion điều kiện hóa theo âm tiết học

Huấn luyện một bộ khử nhiễu không-thời gian phân cấp trên các chuỗi ký hiệu hoàn chỉnh. Một kiến trúc thực tế là:

1. **Các khối đồ thị nội bộ bộ phận (intra-part graph blocks)** cho thân/cánh tay, tay trái và tay phải;
2. **Chú ý quan hệ giữa các bộ phận (cross-part relational attention)** thông qua các token cổ tay và tiếp xúc;
3. **Chú ý thời gian hai chiều (bidirectional temporal attention)** trên toàn bộ ký hiệu trung tâm; và
4. **Chú ý chéo (cross-attention) đến các token âm tiết học mềm và quan sát thiếu chắc chắn**.

Tại bước diffusion $\tau$:

$$
\hat{\epsilon} =
\epsilon_{\theta}(X^{\tau}_{1:T}, \tau,
Z_{\text{ph}}, R_{1:T}, O_{1:T}).
$$

Quá trình nhiễu hóa khi huấn luyện phải giống với các thất bại phục dựng:

- Che các ngón tay riêng lẻ;
- Che toàn bộ một bàn tay trong một chuỗi khung hình bộc phát;
- Gây nhiễu hướng cổ tay;
- Đưa vào các sai số còn lại thực tế của SMPLer-X, HaMeR và WiLoR;
- Hoán đổi hoặc phản chiếu các giả thuyết bàn tay mơ hồ;
- Giả lập độ mờ hình ảnh và mất điểm đặc trưng;
- Cắt xén bàn tay tại biên hình ảnh; và
- Gây nhiễu cánh tay và bàn tay đồng thời để tái tạo các trường hợp hỏng cổ tay.

Sử dụng huấn luyện mặt nạ hỗn hợp (mixed masked training) để các tập dữ liệu chỉ có bàn tay chất lượng cao có thể cải thiện việc mô hình ngón tay mà không đòi hỏi nhãn toàn cơ thể, tuân theo nguyên lý hữu ích đã được chứng minh bởi DPoser-X. SignAvatars sau đó sẽ thích ứng mô hình tổng quát này vào động lực học ký hiệu và sự phối hợp tay-cơ thể.

### 6.7 Động lực học nhận biết pha (Phase-aware dynamics)

Dự đoán phân bố theo từng khung hình trên các trạng thái `hold` (dừng), `transition` (chuyển tiếp), `repetition` (lặp lại), và `contact-transition` (chuyển tiếp tiếp xúc):

$$
p_t = \operatorname{softmax}(h_{\text{phase}}(O_{1:T}, Z_{\text{ph}})).
$$

Sử dụng pha để điều khiển động lực học:

$$
\mathcal{L}_{\text{phase-dyn}} =
\sum_t p_t^{\text{hold}}\|\Delta X_t\|_1
+ p_t^{\text{transition}}
\rho(\Delta^2 X_t)
+ p_t^{\text{contact}}
\|\Delta R_t^{\text{contact}}\|_1.
$$

Do đó, mô hình cực kỳ ổn định trong khoảng dừng, mượt mà nhưng nhạy bén trong chuyển tiếp, và ổn định về mặt quan hệ trong suốt quá trình tiếp xúc duy trì. Nó tránh được thất bại trung tâm của việc làm mượt đồng nhất: coi vận tốc có ý nghĩa là nhiễu.

### 6.8 Lấy mẫu hậu phương và tinh chỉnh cuối cùng

Lúc suy luận:

1. Chạy các chuyên gia quan sát được đóng đóng băng;
2. Chuyển đổi tất cả các ứng viên sang chuẩn hóa chung SMPL-X/thân người;
3. Dự đoán độ không đảm bảo quan sát, âm tiết học, pha và tiếp xúc;
4. Lấy mẫu $K$ giả thuyết chuỗi hoàn chỉnh từ phân bố hậu phương diffusion có điều kiện;
5. Tinh chỉnh mỗi giả thuyết với các tổn thất quan sát và quan hệ SMPL-X vi phân được; và
6. Chọn giả thuyết có khả năng tương thích cao nhất với bằng chứng video.

Năng lượng cuối cùng có thể viết là:

$$
E(X) =
\lambda_{\text{obs}}E_{\text{uncertain-obs}}
+ \lambda_{2D}E_{\text{reproj}}
+ \lambda_{\text{img}}E_{\text{image}}
+ \lambda_{\text{rel}}E_{\text{relation}}
+ \lambda_{\text{ph}}E_{\text{ph-cycle}}
+ \lambda_{\text{dyn}}E_{\text{phase-dyn}}
+ \lambda_{\text{bio}}E_{\text{biomech}}
+ \lambda_{\text{pen}}E_{\text{penetration}}.
$$

Đối với các quan sát 3D dị sai:

$$
E_{\text{uncertain-obs}} =
\sum_{t,j,e}
\frac{\rho(X_{t,j} - \mu^{e}_{t,j})}{2(\sigma^{e}_{t,j})^2}
+ \frac{1}{2}\log (\sigma^{e}_{t,j})^2,
$$

trong đó $e$ đánh chỉ số các chuyên gia quan sát. Điều này cho phép phương pháp sử dụng bằng chứng WiLoR mạnh trên bàn tay phải rõ ràng trong khi bác bỏ một dự đoán tự tin nhưng bất nhất theo thời gian đối với bàn tay trái bị che.

Chỉ sử dụng bằng chứng video cho việc chọn mẫu—không bao giờ dùng nhãn chuẩn (ground truth) của tập chuẩn. Một kiểm thử lựa chọn hữu ích là dự đoán quan sát bị ẩn: tạm thời giữ lại một tập con các điểm đặc trưng hoặc đặc trưng tin cậy và chọn giả thuyết dự đoán chúng tốt nhất.

Đối với benchmark SGNify ký hiệu cô lập, mô hình có thể xử lý từng đoạn ký hiệu được cung cấp như một chuỗi. Đối với ký hiệu liên tục, chạy các cửa sổ thời gian chồng lấp và hợp nhất chúng với độ tin cậy hậu phương của khung hình dùng chung; một token ranh giới/pha học được có thể ngăn việc phân đoạn từ thủ công trở thành một giả định suy luận.

### 6.9 Hai chế độ suy luận

Nghiên cứu nên đưa ra hai chế độ từ cùng một mô hình.

#### SP4D-Fast

- Một đường khử nhiễu định tính;
- Một bước tinh chỉnh SMPL-X ngắn;
- Phù hợp cho việc gán nhãn tập dữ liệu và các nghiên cứu định tính diện rộng.

#### SP4D-Best

- 4–8 giả thuyết hậu phương chỉ tại những nơi độ không đảm bảo cao;
- Ngữ cảnh toàn bộ ký hiệu hai chiều;
- Tinh chỉnh vi phân dài hơn;
- Dùng cho bảng kết quả TR-V2V chính.

Điều này làm cho bài báo có ích vượt ra ngoài tập chuẩn trong khi vẫn cho phép cấu hình độ chính xác cao nhất thể hiện được giới hạn trên của phương pháp.

---

## 7. Lý do phương pháp đề xuất nên giảm từng chỉ số lỗi

| Thất bại của DexAvatar | Cơ chế của SP4D | Chỉ số dự kiến được cải thiện |
|---|---|---|
| Bàn tay sai dưới độ mờ | Điền khuyết thời gian hai chiều cộng với token dạng bàn tay | Bàn tay trái/phải |
| Hướng lòng bàn tay sai | Trạng thái lòng bàn tay/cẳng tay tường minh và quan sát XYZ đầy đủ | Bàn tay và thân trên |
| Cổ tay bị hỏng sau khi thay tay | Trạng thái cổ tay–cẳng tay chung và tinh chỉnh động học | Thân trên và bàn tay |
| Bàn tay tương tác bị che khuất | Đồ thị quan hệ, trạng thái tiếp xúc, hậu phương đa giả thuyết | Cả hai bàn tay |
| Rung lắc trong điểm dừng dạng bàn tay | Động lực học nhận biết điểm dừng | Cả hai bàn tay |
| Chuyển tiếp nhanh bị làm mượt quá mức | Động lực học nhận biết chuyển tiếp | Cả ba |
| Bộ khởi tạo xấu nhưng tự tin | Độ không đảm bảo được hiệu chuẩn và sự bất đồng chuyên gia | Cả ba |
| Đoán tay trái/phải độc lập | Chú ý chéo giữa hai tay điều kiện hóa theo đối xứng/ưu thế | Cả hai bàn tay |
| Lỗi cánh tay quanh khu vực tiếp xúc tay | Quan hệ tay-cơ thể lan truyền ngược qua chuỗi cánh tay | Thân trên |
| Tư thế ký hiệu hiếm bị kéo về trung bình VAE | Tiền phương có điều kiện đa thức (multimodal) | Cả ba |

Thứ tự cải thiện dự kiến là:

1. Một chuyên gia bàn tay hiện đại mạnh tạo ra sự cắt giảm lớn đầu tiên trong TR-V2V bàn tay;
2. Tích hợp XYZ đầy đủ và cổ tay/cẳng tay ngăn việc mất đi mức cải thiện đó khi chèn bàn tay vào SMPL-X;
3. Suy luận hậu phương thời gian sửa chữa các khung hình nơi chuyên gia bị sai;
4. Âm tiết học ngăn ngừa dạng bàn tay hoặc hướng mượt mà theo thời gian nhưng sai về mặt ngữ nghĩa; và
5. Lập luận quan hệ cải thiện các trường hợp hai tay/tiếp xúc và đưa ràng buộc cổ tay tốt hơn vào thân trên.

---

## 8. Chiến lược huấn luyện

### 8.1 Nguồn dữ liệu

#### Chuyển động toàn cơ thể đặc thù ký hiệu

- **SignAvatars:** 70.000 chuỗi, 153 người ký hiệu và 8,34 triệu khung hình SMPL-X với nhiều loại gợi ý. Sử dụng tập con chất lượng cao nhất sau khi lọc tự động.
- **Chú giải công khai từ How2Sign:** Hữu ích cho ASL liên tục và đồng cấu âm (co-articulation) nếu giấp phép và bản phát hành của họ cho phép biểu diễn được chọn.
- **Dữ liệu/checkpoint SignBPoser của DexAvatar:** Hữu ích làm giáo viên hoặc khởi tạo, nhưng phương pháp mới không nên phụ thuộc vào dữ liệu mocap bàn tay riêng tư không có sẵn.

#### Bàn tay chất lượng cao và sự phối hợp

- InterHand2.6M cho bàn tay tương tác;
- ARCTIC cho khớp bàn tay chính xác và ngữ cảnh cơ thể;
- DexYCB hoặc HanCo cho hình học bàn tay nhìn thấy được;
- WHIM từ WiLoR cho bàn tay trong tự nhiên (in-the-wild);
- AssemblyHands-X nếu giấp phép và bản phát hành cho phép; và
- Chuyển động tổng hợp tương thích MANO được render với các bản cắt video ký hiệu thực tế.

#### Chuyển động toàn cơ thể tổng quát

- Motion-X và các tập dữ liệu SMPL-X khác có thể huấn luyện trước cấu trúc cơ thể-bàn tay;
- Sử dụng chúng với xác suất lấy mẫu thấp hơn trong quá trình thích ứng ký hiệu để các cử chỉ hàng ngày không lấn át động lực học ký hiệu.

Tập chuẩn định lượng SGNify phải hoàn toàn chỉ dùng cho đánh giá.

### 8.2 Lọc chất lượng

SignAvatars rất lớn nhưng được gán nhãn tự động. Tránh huấn luyện một bộ khử nhiễu lặp lại các lỗi gán nhãn.

Giữ lại hoặc tăng trọng số cho các khung hình/chuỗi có:

- Sự đồng thuận giữa hai bộ ước lượng cơ thể/bàn tay;
- Sai số chiếu lại 2D thấp;
- Danh tính và hình dạng ổn định;
- Phép xoay khớp hợp lệ và không bị sụp lưới (mesh collapse);
- Chiều dài xương bàn tay hợp lý;
- Hướng lòng bàn tay nhất quán theo thời gian; và
- Sự khớp nhau giữa hình ảnh và bóng lưới (silhouette agreement) nếu có.

Chỉ sử dụng các chuỗi chất lượng thấp hơn còn lại với sự giám sát được trọng số theo độ tin cậy hoặc dưới dạng đầu vào nhiễu kết hợp với mục tiêu được tinh chỉnh từ giáo viên.

### 8.3 Chương trình huấn luyện 4 giai đoạn

#### Giai đoạn A: Huấn luyện trước chuyển động có cấu trúc tổng quát

Huấn luyện các khối nội bộ bộ phận và giữa các bộ phận với dữ liệu hỗn hợp toàn cơ thể và chỉ bàn tay. Sử dụng huấn luyện phần bị che để các nhãn thiếu không trở thành tư thế bằng 0.

#### Giai đoạn B: Thích ứng ngôn ngữ ký hiệu

Huấn luyện trên SignAvatars với tỷ lệ lấy mẫu cao hơn cho tương tác hai tay, khớp nối nhanh, ảnh cắt thân trên và các dạng bàn tay hiếm. Bổ sung giám sát âm tiết học và pha trích xuất từ hình học.

#### Giai đoạn C: Huấn luyện nén nhiễu phục dựng (Reconstruction-corruption training)

Chạy đúng các chuyên gia quan sát dự định trên video huấn luyện. Học cách ánh xạ các phân bố sai số thực tế của họ sang các chuỗi tinh chỉnh. Bổ sung che bộc phát, mờ, mất ảnh cắt, hoán đổi tay và nhiễu cổ tay.

#### Giai đoạn D: Hiệu chuẩn hậu phương và bằng chứng

Huấn luyện độ không đảm bảo và lựa chọn đa giả thuyết trên các người ký hiệu và tập dữ liệu độc lập. Hiệu chuẩn phương sai dự đoán bằng negative log likelihood và đường cong độ bao phủ, không chỉ dùng sai số hồi quy.

### 8.4 Các tổn thất trong quá trình huấn luyện

Tổn thất huấn luyện hoàn chỉnh có thể là:

$$
\mathcal{L}_{\text{train}} =
\lambda_{\epsilon}\mathcal{L}_{\text{diff}}
+ \lambda_v\mathcal{L}_{\text{vertex}}
+ \lambda_j\mathcal{L}_{\text{joint}}
+ \lambda_r\mathcal{L}_{\text{rotation}}
+ \lambda_{tip}\mathcal{L}_{\text{fingertip}}
+ \lambda_{rel}\mathcal{L}_{\text{relation}}
+ \lambda_c\mathcal{L}_{\text{contact}}
+ \lambda_p\mathcal{L}_{\text{phonology}}
+ \lambda_{cyc}\mathcal{L}_{\text{ph-cycle}}
+ \lambda_{phase}\mathcal{L}_{\text{phase}}
+ \lambda_u\mathcal{L}_{\text{uncertainty}}.
$$

Các lựa chọn cài đặt quan trọng:

- Giám sát các đỉnh và khớp đầu ngón tay, không chỉ riêng góc xoay;
- Cung cấp cho hướng lòng bàn tay một tổn thất xoay trắc địa (geodesic rotation loss) tường minh;
- Cân bằng tay trái và tay phải thay vì để số lượng đỉnh cơ thể áp đảo;
- Lấy mẫu theo ký hiệu thay vì theo khung hình để bảo tồn các chuỗi hiếm;
- Chỉ phản chiếu chuỗi với phép biến đổi thuận tay và ưu thế chính xác; và
- Giữ hình dạng dùng chung trên toàn chuỗi.

---

## 9. Lộ trình phát triển thực tiễn

Phương pháp đầy đủ nên đạt được thông qua các baseline trung gian có thể xuất bản. Mỗi giai đoạn phải sử dụng cùng đầu vào benchmark và script TR-V2V.

### Pha 0: Tái lập bản phát hành gốc

Mục tiêu: tái lập DexAvatar từ commit `7e97916` và lưu trữ các lưới cùng điểm số của nó.

Sản phẩm bàn giao:

- Các cấu hình gốc chính xác;
- Bảng sai số theo từng ký hiệu;
- Dự đoán theo từng khung hình cho các so sánh thống kê theo cặp; và
- Bộ sưu tập các thất bại định tính được nhóm theo độ mờ, che khuất, tiếp xúc và chuyển động nhanh.

### Pha 1: Baseline không mới nhưng mạnh nhất

Xây dựng:

- Cơ thể SMPLer-X;
- Bàn tay WiLoR;
- Chuyển đổi MANO sang SMPL-X chính xác;
- Dóng hàng cổ tay/cẳng tay vi phân được;
- Tinh chỉnh quan sát XYZ và 2D đầy đủ; và
- Làm mượt hai chiều đơn giản.

Điều này giúp xác định liệu mã nguồn có thể đạt tới vùng hoạt động hiện đại `xấp xỉ 29 / 10.6 / 8.9` hay không. Bước này là thiết yếu, nhưng chưa phải là phương pháp cuối cùng.

### Pha 2: Bộ tinh chỉnh toàn chuỗi nhận biết độ không đảm bảo

Thêm:

- Cửa sổ chuỗi hoàn chỉnh;
- Các token quan sát và đầu độ tin cậy;
- Nhiễu hỏng bộc phát thực tế;
- Trạng thái chung cơ thể/cổ tay/bàn tay; và
- Khử nhiễu thời gian định tính.

Đây là cách nhanh nhất để kiểm tra xem suy luận thời gian có giảm TR-V2V vượt khỏi bộ khởi tạo mạnh hay không.

### Pha 3: Phân bố hậu phương diffusion quan hệ

Thay thế khử nhiễu định tính bằng masked diffusion và thêm:

- Đồ thị giữa hai tay và tay-cơ thể;
- Dự đoán tiếp xúc;
- Suy luận $K$-giả thuyết; và
- Lựa chọn mẫu dựa trên bằng chứng.

Điều này sẽ tạo ra mức cải thiện lớn nhất trên các bàn tay bị che và tương tác.

### Pha 4: Âm tiết học và pha ký hiệu

Thêm:

- Biểu diễn dạng bàn tay/hướng/vị trí/chuyển động mềm;
- Tổn thất vòng lặp ngữ nghĩa–động học;
- Tính đối xứng và ưu thế xác suất;
- Pha dừng/chuyển tiếp/lặp lại/접촉; và
- Động lực học được điều khiển bởi pha.

Đây là đóng góp đặc thù ký hiệu định hình nên bài báo.

### Pha 5: Hiệu năng và phát hành

Đóng gói hậu phương tốt nhất thành SP4D-Fast, ghi chép lại toàn bộ tiền xử lý, phát hành các mô hình và dự đoán đã huấn luyện trước, và cung cấp một lệnh benchmark duy nhất có thể tái lập.

---

## 10. Nghiên cứu triệt tiêu (Ablation study) bắt buộc

Bảng chính nên báo cáo cả 3 vùng TR-V2V tiêu chuẩn cho mọi dòng.

| Dòng | Cấu hình | Câu hỏi khoa học |
|---|---|---|
| A0 | DexAvatar gốc | Tham chiếu đã xuất bản |
| A1 | Cơ thể hiện đại + WiLoR | Bao nhiêu phần cải thiện đến từ quan sát mới hơn? |
| A2 | A1 + tích hợp cổ tay/cẳng tay + XYZ đầy đủ | Phép dung hợp đúng hình học có giúp ích không? |
| A3 | A2 + bộ tinh chỉnh hai chiều định tính | Ngữ cảnh toàn chuỗi có giúp ích không? |
| A4 | A3 + trọng số độ không đảm bảo | Mô hình có thể bác bỏ các quan sát xấu không? |
| A5 | A4 + đồ thị quan hệ/tiếp xúc | Tương tác có cải thiện bàn tay và cánh tay không? |
| A6 | A5 + hậu phương diffusion, K=1 | Tiền phương sinh có giúp ích khi không có best-of-K? |
| A7 | A6 + K giả thuyết và lựa chọn bằng chứng | Việc mô hình hóa sự mơ hồ có giúp ích không? |
| A8 | A7 + điều kiện hóa âm tiết học | Phương pháp có thực sự nhận biết ký hiệu không? |
| A9 | A8 + động lực học nhận biết pha | Nó có bảo toàn chuyển động trong khi ổn định điểm dừng không? |

Các nghiên cứu triệt tiêu có kiểm soát bổ sung:

- Không có ngữ cảnh tương lai so với ngữ cảnh hai chiều;
- Tiền phương chuyển động tổng quát so với tiền phương thích ứng ký hiệu;
- Lớp SGNify cứng so với âm tiết học cấu thành mềm;
- Không có token dạng bàn tay, không có token hướng, không có token chuyển động, không có token quan hệ;
- Không huấn luyện với nhiễu hỏng bộc phát;
- Bộ ước lượng đơn so với quan sát đa chuyên gia;
- Độ không đảm bảo dự đoán so với chỉ dùng độ tin cậy của bộ phát hiện;
- `K = 1, 2, 4, 8` giả thuyết;
- Làm mượt đồng nhất so với động lực học nhận biết pha;
- Không tiếp xúc, chỉ va chạm, và tiếp xúc dự đoán;
- Ngữ cảnh thời gian 16, 32, 64 khung hình và toàn bộ ký hiệu; và
- Đường cong quy mô dữ liệu SignAvatars tại 10%, 25%, 50%, và 100%.

Không bao giờ chỉ báo cáo cấu hình đầy đủ. Phản biện phải thấy được rằng âm tiết học, tương tác và suy luận hậu phương đều đóng góp giá trị vượt khỏi một bộ ước lượng bàn tay mạnh hơn.

---

## 11. Gói đánh giá dành cho một bài báo cấp A*

### 11.1 Kết quả định lượng chính

- TR-V2V SGNify tiêu chuẩn trên tất cả 2.872 khung hình trung tâm;
- Thân trên không gồm mặt, bàn tay trái, và bàn tay phải;
- So sánh với DexAvatar gốc và mọi baseline công khai có thể tái lập;
- Chênh lệch theo cặp từng ký hiệu và khoảng tin cậy 95% bootstrap; và
- Sai số ký hiệu trung bình, trung vị và 10% tồi nhất trong phụ lục.

### 11.2 Các tập con chẩn đoán

Tạo các chú giải mà không làm thay đổi chỉ số benchmark:

- Một tay so với hai tay;
- Đối xứng so với bất đối xứng;
- Tiếp xúc so với không tiếp xúc;
- Tiếp xúc tay-tay so với tay-cơ thể;
- Nhìn thấy so với bị che một phần so với bị che nghiêm trọng;
- Mờ thấp so với mờ cao;
- Điểm dừng so với chuyển tiếp;
- Vận tốc cổ tay/đầu ngón tay chậm so với nhanh; và
- Sự bất đồng giữa bộ ước lượng bên trái/bên phải.

Báo cáo cùng chỉ số TR-V2V trong từng tập con. Phương pháp đề xuất nên cho thấy mức cải thiện tương đối lớn nhất trên các trường hợp che khuất nghiêm trọng, tiếp xúc và chuyển tiếp.

### 11.3 Khả năng tổng quát hóa trên nhiều tập dữ liệu

Benchmark SGNify khá nhỏ. Một bài báo mạnh cũng nên kiểm thử:

- UBody hoặc ARCTIC cho phục dựng toàn cơ thể/bàn tay;
- InterHand2.6M cho phục dựng bàn tay tương tác;
- How2Sign, PHOENIX-2014T, WLASL, và MM-WLAuslan cho đánh giá định tính và tính nhất quán 2D nơi không có 3D ground truth; và
- Ít nhất một người ký hiệu/ngôn ngữ bị giữ lại không dùng trong huấn luyện tiền phương ký hiệu.

Không chỉnh phương pháp riêng cho từng ngôn ngữ. Các thuộc tính cấu thành nên được chuyển giao ngay cả khi từ vựng ký hiệu không chuyển giao.

### 11.4 Kiểm thử độ bền vững (Robustness tests)

Áp dụng các nhiễu có kiểm soát vào các khung hình có ground truth hợp lệ:

- Độ mờ Gaussian và mờ chuyển động;
- Che khuất bàn tay 10%, 25%, và 40%;
- Mất hoàn toàn bàn tay trong 4, 8, và 16 khung hình liên tiếp;
- Cắt xén ảnh;
- Nhiễu điểm đặc trưng 2D;
- Giả thuyết bàn tay sai được chèn vào một chuyên gia; và
- Giảm tốc độ khung hình (frame-rate reduction).

Vẽ đồ thị TR-V2V theo mức độ nghiêm trọng của nhiễu. Điều này kiểm thử trực tiếp các khẳng định về phân bố hậu phương và độ không đảm bảo.

### 11.5 Các chỉ số thời gian và ngữ nghĩa

TR-V2V vẫn là benchmark chính, nhưng các khẳng định về thời gian và âm tiết học của bài báo cần thêm bằng chứng:

- MPJVE, sai số gia tốc và jerk;
- Sai số quỹ đạo đầu ngón tay;
- Sai số trắc địa pháp tuyến lòng bàn tay;
- Precision/recall tiếp xúc và sự trượt tiếp xúc;
- Độ chính xác thuộc tính dạng bàn tay/hướng/vị trí;
- Độ chính xác nhận dạng ký hiệu từ chuyển động 3D phục dựng; và
- Nghiên cứu cảm nhận với người ký hiệu Điếc đo lường tính định danh và tính tự nhiên của ký hiệu.

Đánh giá ngữ nghĩa rất quan trọng: một lưới mượt mà hơn không tự động là một ký hiệu chính xác hơn.

### 11.6 Hiệu năng tính toán

Báo cáo:

- Thời gian tiền xử lý của chuyên gia;
- Thời gian chạy SP4D-Fast và SP4D-Best trên mỗi khung hình/ký hiệu;
- Bộ nhớ GPU;
- Số bước diffusion và số giả thuyết; và
- Sự đánh đổi giữa độ chính xác và thời gian chạy.

Tamaththul3D nhấn mạnh rõ ràng về thời gian chạy, vì vậy độ chính xác không thể là điểm so sánh duy nhất.

---

## 12. Tính mới cho xuất bản và định vị bài báo

### 12.1 Hướng tên bài báo khuyến nghị

**SignPosterior4D: Phonology-Conditioned Relational Diffusion for 3D Sign Language Reconstruction**

Các tên thay thế:

- **Beyond Framewise Priors: Whole-Sequence Posterior Inference for 3D Signing Avatars**
- **PhonoSign4D: Uncertainty-Aware 4D Hand–Body Reconstruction from Monocular Signing Video**
- **RelSign4D: Interaction- and Phase-Aware 3D Sign Reconstruction**

### 12.2 Luận điểm của bài báo

> Các phương pháp phục dựng ký hiệu hiện tại sử dụng các ràng buộc hoặc tiền phương tư thế đặc thù cho ký hiệu nhưng xử lý các quan sát thiếu chắc chắn, sự phối hợp cơ thể–bàn tay và thời gian một cách chưa đầy đủ. Chúng tôi giới thiệu một phân bố hậu phương có cấu trúc kết hợp âm tiết học ký hiệu cấu thành với chuyển động 4D quan hệ, cho phép các khung hình nhìn thấy và các bộ phận cơ thể phối hợp giải quyết sự mơ hồ của khớp bàn tay bị che mà không làm mượt quá mức các chuyển tiếp có ý nghĩa.

### 12.3 Các đóng góp có thể bảo vệ được

1. **Một phân bố hậu phương ký hiệu cấu thành.** Tiền phương phục dựng đầu tiên được điều kiện hóa đồng thời trên dạng bàn tay, hướng, vị trí, chuyển động, tính đối xứng/ưu thế và tiếp xúc suy luận được mà không đòi hỏi nhãn từ ký hiệu lúc suy luận.
2. **Suy luận toàn chuỗi quan hệ.** Một mô hình masked diffusion biểu diễn đồng thời thân trên, hướng cổ tay, cả hai bàn tay và tiếp xúc tay–cơ thể trên các ký hiệu hoàn chỉnh.
3. **Phục dựng nhận biết độ không đảm bảo và pha.** Dung hợp chuyên gia được hiệu chuẩn và động lực học nhận biết điểm dừng/chuyển tiếp giúp điền khuyết các vị trí bị che mà không áp chế chuyển động có ý nghĩa về mặt ngôn ngữ.
4. **Một nghiên cứu phục dựng nghiêm ngặt.** Standard TR-V2V cộng với các đánh giá qua tập dữ liệu, nhiễu hỏng, thời gian, tiếp xúc và ngữ nghĩa.

Nếu phương pháp đầy đủ quá lớn cho một bài báo, hãy giữ lại đóng góp 1–3 và coi việc nén mô hình hiệu năng cao là công việc tương lai.

### 12.4 Phân biệt rõ ràng với các nghiên cứu lân cận

| Nghiên cứu lân cận | Đã đóng góp | SP4D phải đóng góp vượt khỏi đó |
|---|---|---|
| SGNify | Các lớp đối xứng cứng và bất biến tư thế | Âm tiết học cấu thành mềm, pha học được, hậu phương quan hệ |
| DexAvatar | VAE cơ thể/tay tĩnh đặc thù ký hiệu | Phân bố chuỗi thống nhất cơ thể–cổ tay–bàn tay |
| Tamaththul3D | Chuyển đổi WiLoR và dóng hàng cánh tay hình học | Giải tỏa mơ hồ thời gian học được và ngữ nghĩa ký hiệu |
| Hand4Whole++ | Điều kiện hóa đặc trưng và dóng hàng cứng đơn khung hình | Phục dựng xác suất toàn bộ ký hiệu |
| DanceHMR | Hồi quy toàn cơ thể nhận biết bàn tay theo thời gian tổng quát | Âm tiết học ký hiệu, đồ thị tiếp xúc, hậu phương nhận biết pha, các giả thuyết |
| DPoser-X | Tiền phương masked diffusion toàn cơ thể tổng quát | Bài toán ngược thời gian và quan hệ có điều kiện ký hiệu |
| FUSION | Diffusion chuyển động cơ thể và bàn tay tổng quát | Phục dựng có điều kiện video và cấu trúc ký hiệu |
| MaskHand | Khôi phục bàn tay đơn ảnh mang tính xác suất | Suy luận chuỗi phối hợp hai bàn tay/cơ thể |
| OmniHands | Bàn tay 4D tương tác nhận biết quan hệ | Ngữ nghĩa ký hiệu tường minh và toàn bộ thân trên SMPL-X |
| PAD-Hand | Diffusion chuyển động bàn tay nhận biết vật lý | Âm tiết học ký hiệu, quan hệ hai tay/cơ thể, hậu phương RGB |

### 12.5 Các khẳng định cần tránh

Không khẳng định:

- “Phục dựng toàn cơ thể bàn tay theo thời gian đầu tiên”;
- “Tiền phương diffusion đầu tiên cho cơ thể và bàn tay”;
- “Phục dựng bàn tay đa giả thuyết đầu tiên”;
- “Tích hợp tay–cơ thể đầu tiên”;
- “Tiền phương ngôn ngữ học đầu tiên cho phục dựng ký hiệu”; hoặc
- “SOTA” trước khi mọi phương pháp công khai được chạy dưới cùng một giao thức cố định.

Giao điểm mới hẹp hơn và mạnh mẽ hơn: **suy luận phân bố hậu phương quan hệ điều kiện hóa theo âm tiết học cấu thành cho phục dựng 3D toàn bộ ký hiệu**.

---

## 13. Rủi ro từ phản biện và cách trung hòa

### Rủi ro 1: “Mức cải thiện chỉ đến từ WiLoR.”

Phản hồi bắt buộc: trình bày A1/A2, sau đó là các mức giảm nhất quán bổ sung từ phân bố hậu phương thời gian, các quan hệ, âm tiết học và pha. Đưa vào cả khởi tạo HaMeR và WiLoR để chứng minh sự tinh chỉnh không phụ thuộc bộ khởi tạo.

### Rủi ro 2: “Đây là DanceHMR được thích ứng cho ký hiệu.”

Phản hồi bắt buộc: chứng minh các phân tích triệt tiêu về âm tiết học cấu thành, pha ký hiệu, đồ thị tiếp xúc và hậu phương đa giả thuyết. Đánh giá sự bảo toàn thuộc tính và nhận dạng ký hiệu, điều mà HMR tổng quát không hướng tới.

### Rủi ro 3: “Tiền phương ký hiệu ghi nhớ các ký hiệu SGNify.”

Phản hồi bắt buộc: giữ SGNify hoàn toàn cho đánh giá, loại bỏ lặp chuỗi huấn luyện theo nguồn/từ/người ký hiệu, đánh giá trên người ký hiệu và ngôn ngữ chưa từng thấy, và công bố hash các tập chia.

### Rủi ro 4: “Dữ liệu giả 3D lớn chỉ lặp lại thiên kiến của giáo viên.”

Phản hồi bắt buộc: lọc chất lượng với nhiều chuyên gia, sử dụng các tập dữ liệu bàn tay 3D thực tế, trình bày kết quả theo chất lượng mục tiêu huấn luyện, và chứng minh sự cải thiện so với mọi giáo viên.

### Rủi ro 5: “Diffusion làm phương pháp chậm và phức tạp.”

Phản hồi bắt buộc: báo cáo SP4D-Fast, chỉ kích hoạt lấy mẫu dựa trên độ không đảm bảo ở các khoảng khó, và nén hậu phương đa mẫu thành một bộ tinh chỉnh định tính.

### Rủi ro 6: “Độ mượt thời gian cải thiện vẻ ngoài chứ không cải thiện độ chính xác.”

Phản hồi bắt buộc: báo cáo TR-V2V, sai số quỹ đạo đầu ngón tay, các tập con điểm dừng/chuyển tiếp, và triệt tiêu làm mượt đồng nhất. Phương pháp phải giảm sai số không gian cũng như giảm rung lắc.

### Rủi ro 7: “Nhãn âm tiết học bị nhiễu hoặc đặc thù cho ngôn ngữ.”

Phản hồi bắt buộc: sử dụng các thuộc tính liên tục trích xuất từ hình học, báo cáo độ hiệu chuẩn/độ chính xác thuộc tính, và kiểm thử qua nhiều ngôn ngữ. Coi HamNoSys là thông tin huấn luyện đặc quyền, không phải đầu vào suy luận bắt buộc.

### Rủi ro 8: “Ràng buộc tiếp xúc ép buộc các tương tác sai.”

Phản hồi bắt buộc: làm cho tiếp xúc mang tính xác suất và được trọng số theo độ không đảm bảo, triệt tiêu tiếp xúc hoàn hảo (oracle)/dự đoán/không tiếp xúc, và tắt năng lượng tiếp xúc khi phân bố hậu phương của nó bị phân tán.

---

## 14. Các thực nghiệm đầu tiên được đề xuất

Thứ tự tạo ra bằng chứng nhanh nhất là:

1. Tái lập các lưới và điểm số DexAvatar gốc;
2. Xây dựng một baseline SMPLer-X + WiLoR + dóng hàng cổ tay sạch sé;
3. Thay thế giám sát bàn tay chỉ có chiều sâu chuẩn hóa bằng quan sát XYZ đầy đủ, pháp tuyến lòng bàn tay và đầu ngón tay được trọng số theo độ tin cậy;
4. Tối ưu hóa các cửa sổ thời gian hai chiều 32–64 khung hình trong khi dùng chung hình dạng;
5. Huấn luyện một bộ tinh chỉnh thời gian định tính từ nhiễu-sang-sạch trên SignAvatars;
6. Bổ sung dự đoán độ tin cậy và đo lường sự cải thiện đặc biệt trên các khung hình có bộ khởi tạo xấu;
7. Bổ sung đồ thị quan hệ và đầu tiếp xúc;
8. Chuyển đổi bộ tinh chỉnh thành một phân bố hậu phương masked conditional diffusion;
9. Bổ sung điều kiện hóa âm tiết học mềm và pha; và
10. Chạy bảng triệt tiêu hoàn chỉnh trước khi tinh chỉnh cho con số headline cuối cùng.

### Các điểm kiểm tra Go/No-go

- Nếu bước 2 không thể tiếp cận kết quả bàn tay hiện đại mạnh nhất, hãy sửa phép chuyển đổi tọa độ MANO/SMPL-X trước khi huấn luyện bất kỳ mô hình mới nào.
- Nếu bước 4 giảm rung lắc nhưng không giảm TR-V2V, hãy kiểm tra sự dóng hàng thời gian và việc làm mượt quá mức trước khi thêm diffusion.
- Nếu bộ tinh chỉnh định tính không thể cải thiện các giáo viên đóng băng của nó trên dữ liệu kiểm định có giám sát, sự phức tạp hậu phương thêm vào sẽ không giải quyết được bài toán dữ liệu.
- Nếu âm tiết học không mang lại mức tăng trên các tập con bị che/mờ, hãy cải thiện giám sát thuộc tính thay vì giấu nó bên trong mô hình đầy đủ.
- Nếu đồ thị quan hệ chỉ giúp ích cho các chỉ số tiếp xúc nhưng làm hại TR-V2V, hãy dùng độ không đảm bảo tiếp xúc để chặn năng lượng của nó.

---

## 15. Bài báo khả thi tối thiểu (MVP) so với bài báo đầy đủ

### Bài báo mạnh khả thi tối thiểu (MVP)

Phiên bản nhỏ nhất nhất quán có thể xuất bản là:

- Khởi tạo cơ thể và bàn tay công khai mạnh;
- Chuẩn hóa chung thân trên/cổ tay/bàn tay;
- Masked sequence diffusion nhận biết độ không đảm bảo;
- Điều kiện hóa dạng bàn tay/hướng/vị trí/chuyển động mềm;
- Tinh chỉnh hậu phương nhận biết pha; và
- Standard TR-V2V cộng với đánh giá độ bền vững và ngữ nghĩa.

Tiếp xúc có thể là một mô-đun phụ trợ nếu thời gian cài đặt có hạn.

### Bài báo đầy đủ

Phiên bản đầy đủ bổ sung thêm:

- Đồ thị tay–tay/tay–cơ thể tường minh;
- Dự đoán tiếp xúc và thứ tự chiều sâu;
- Lấy mẫu đa giả thuyết chọn lọc theo bằng chứng;
- Khả năng tổng quát hóa qua nhiều ngôn ngữ;
- Nghiên cứu cảm nhận với người ký hiệu Điếc; và
- Suy luận nhanh được nén (distilled fast inference).

Phiên bản đầy đủ phù hợp hơn cho CVPR/ICCV/ECCV vì nó cung cấp cả đóng góp kỹ thuật mạnh hơn và câu chuyện đánh giá rộng hơn.

---

## 16. Khuyến nghị cuối cùng

Con đường có giá trị cao nhất không phải là tiếp tục điều chỉnh tăng dần tổn thất LBFGS theo từng khung hình của DexAvatar. Sử dụng các quan sát và hạ tầng SMPL-X của nó để thiết lập baseline, sau đó chuyển trọng tâm nghiên cứu sang **phục dựng hậu phương toàn bộ ký hiệu (whole-sign posterior reconstruction)**.

Hệ thống cuối cùng được khuyến nghị là:

> **Các quan sát công khai hiện đại + độ không đảm bảo được hiệu chuẩn + trạng thái thống nhất cơ thể/cổ tay/hai bàn tay + masked diffusion quan hệ + âm tiết học suy luận + tinh chỉnh hậu phương nhận biết pha.**

Hướng đi này có một cơ chế thực tế để vượt qua cả kết quả DexAvatar gốc `30.13 / 13.53 / 13.08` và baseline tích hợp bàn tay hiện đại mạnh hơn. Quan trọng hơn, nó hỗ trợ một khẳng định khoa học cấp A* vẫn giữ nguyên giá trị ngay cả sau khi các bộ ước lượng off-the-shelf cải thiện: quá trình phục dựng được dẫn dắt bởi cấu trúc ngôn ngữ học cấu thành và cấu trúc quan hệ của ký hiệu, chứ không đơn thuần bởi bộ hồi quy theo từng khung hình nào mới nhất.

---

## 17. Nguồn tài liệu nghiên cứu chính

### Phục dựng ký hiệu và dữ liệu

- [DexAvatar, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html)
- [Kho lưu trữ DexAvatar chính thức](https://github.com/kaustesseract/DexAvatar)
- [SGNify, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.html)
- [SignAvatars, ECCV 2024 dự án và dữ liệu](https://signavatars.github.io/)
- [Neural Sign Actors, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Baltatzis_Neural_Sign_Actors_A_Diffusion_Model_for_3D_Sign_Language_CVPR_2024_paper.html)
- [Tamaththul3D v2, preprint tháng 6 năm 2026](https://arxiv.org/html/2605.05367v2)

### Khởi tạo toàn cơ thể và bàn tay

- [Kho lưu trữ chính thức SMPLer-X](https://github.com/caizhongang/SMPLer-X)
- [Kho lưu trữ chính thức SMPLest-X](https://github.com/MotrixLab/SMPLest-X)
- [Kho lưu trữ chính thức WiLoR](https://github.com/rolpotamias/WiLoR)
- [HaMeR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Pavlakos_Reconstructing_Hands_in_3D_with_Transformers_CVPR_2024_paper.html)
- [Hand4Whole++, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Moon_Enhancing_Hands_in_3D_Whole-Body_Pose_Estimation_with_Conditional_Hands_CVPR_2026_paper.html)
- [Dự án chính thức OmniHands](https://omnihand.github.io/)

### Tiền phương thời gian và sinh (Temporal and generative priors)

- [DanceHMR, preprint tháng 5 năm 2026](https://arxiv.org/html/2605.18102)
- [DPoser-X, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Lu_DPoser-X_Diffusion_Model_as_Robust_3D_Whole-body_Human_Pose_Prior_ICCV_2025_paper.html)
- [FUSION full-body motion prior](https://arxiv.org/abs/2601.03959)
- [MaskHand, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Saleem_MaskHand_Generative_Masked_Modeling_for_Robust_Hand_Mesh_Reconstruction_in_ICCV_2025_paper.html)
- [Pose-Guided Temporal Enhancement, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fan_Pose-Guided_Temporal_Enhancement_for_Robust_Low-Resolution_Hand_Reconstruction_CVPR_2025_paper.html)
- [Dyn-HaMR, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html)
- [PAD-Hand, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ismayilzada_PAD-Hand_Physics-Aware_Diffusion_for_Hand_Motion_Recovery_CVPR_2026_paper.html)

---

## 18. Chiến lược huấn luyện được xác thực: Pretrained so với Huấn luyện từ đầu

Phần này kiểm định các mô-đun có thể huấn luyện được đề xuất ở trên và phân tách:

1. Các thành phần nên dùng mô hình pretrained công khai;
2. Các thành phần nên giữ đóng băng (frozen);
3. Các thành phần đòi hỏi huấn luyện mới đặc thù cho DexAvatar; và
4. Các mô hình hữu ích về mặt kỹ thuật nhưng có rủi ro về độ sẵn có hoặc giấp phép.

Kết luận chính là:

> **Không huấn luyện toàn bộ hệ thống SP4D từ khởi tạo ngẫu nhiên. Sử dụng các biểu diễn nhận thức, tư thế không gian và video ký hiệu đã pretrained, trong khi chỉ huấn luyện các thành phần độ không đảm bảo, quan hệ, pha, điều kiện hóa và lựa chọn giả thuyết cho nhiệm vụ này.**

Đây vừa là con đường có xác suất cao nhất để đạt độ chính xác phục dựng mạnh, vừa là thiết kế khoa học sạch nhất. Việc huấn luyện tất cả các mạng xương sống (backbones) từ số 0 sẽ đòi hỏi nhiều dữ liệu sạch và tài nguyên tính toán hơn đáng kể trong khi làm cho nó khó xác định hơn liệu mức tăng đến từ công thức hậu phương đề xuất hay từ việc học biểu diễn cơ bản.

### 18.1 Bảng quyết định cấp mô-đun

| Thành phần đề xuất | Quyết định khởi tạo | Nguồn khuyến nghị | Những gì nên huấn luyện được |
|---|---|---|---|
| Chuyên gia quan sát cơ thể/tay từ ảnh | Dùng pretrained và đóng băng | Hand4Whole++ | Không tinh chỉnh bộ ước lượng trong thực nghiệm chính; chỉ huấn luyện các bộ thích ứng (adapters) hạ nguồn |
| Tiền phương tư thế SMPL-X không gian | Chuyển giao và thích ứng chọn lọc | Checkpoint toàn cơ thể DPoser-X | Các adapter thời gian, phép chiếu điều kiện, và tùy chọn các khối diffusion phía trên ở learning rate thấp |
| Tiền phương diffusion thời gian toàn cơ thể | Chuyển giao nếu quyền hạn cho phép | FUSION; nếu không dùng DPoser-X cộng với mạng thời gian mới | Các lớp thời gian đặc thù ký hiệu và các mô-đun điều kiện hóa |
| Biểu diễn video ký hiệu | Chuyển giao và thích ứng chọn lọc | SHuBERT | Các đầu âm tiết học và pha mới; tùy chọn LoRA hoặc tinh chỉnh các lớp cuối |
| Đặc trưng vẻ ngoài/hình ảnh | Dùng pretrained và đóng băng ban đầu | Luồng thị giác SHuBERT hoặc một bộ mã hóa ảnh tự giám sát công khai khác | Chỉ các phép chiếu đặc trưng nhỏ |
| Độ tin cậy và phương sai quan sát | Huấn luyện cho hệ thống này | Không có đầu pretrained tương thích trực tiếp | Đầu độ tin cậy hoàn chỉnh và hiệu chuẩn sau huấn luyện |
| Các đầu thuộc tính âm tiết học | Huấn luyện cho nhiệm vụ này | Đặc trưng SHuBERT cộng với chú giải ký hiệu | Các đầu dạng bàn tay, hướng, vị trí, chuyển động, đối xứng, ưu thế và tiếp xúc |
| Đầu pha ký hiệu | Huấn luyện cho nhiệm vụ này | Đặc trưng ký hiệu/thời gian dùng chung | Bộ phân loại điểm dừng, chuyển tiếp, lặp lại và chuyển tiếp tiếp xúc |
| Đồ thị quan hệ tay–tay/tay–cơ thể | Khởi tạo hỗn hợp | Đặc trưng tư thế DPoser-X/FUSION | Các cạnh quan hệ, các đầu tiếp xúc/thứ tự chiều sâu và cập nhật đồ thị |
| Bộ lựa chọn bằng chứng đa giả thuyết | Huấn luyện cho hệ thống này | Không có mô hình có thể chuyển giao trực tiếp | Bộ chấm điểm/xếp hạng ứng viên hoàn chỉnh |
| Tinh chỉnh SMPL-X cuối cùng | Bắt đầu với tối ưu hóa | Hình học vi phân được SMPL-X | Ban đầu không cần mạng học bổ sung |

### 18.2 Mô hình quan sát chính: Đóng băng Hand4Whole++

Bộ khởi tạo chính thực tế mạnh nhất là Hand4Whole++ thay vì kết hợp độc lập SMPLer-X và WiLoR. Hand4Whole++ đã xử lý sự tích hợp cổ tay, bàn tay và thân trên một cách nhất quán và tạo ra đầu ra tương thích SMPL-X.

Khuyến nghị sử dụng:

- Chạy checkpoint đã phát hành làm chuyên gia quan sát theo khung hình chính;
- Giữ lại các tín hiệu độ tin cậy nguyên bản của nó và các đặc trưng bàn tay/cơ thể trung gian nếu có;
- Giữ đóng băng bộ ước lượng trong thực nghiệm SP4D chính;
- Huấn luyện SP4D để sửa các thất bại ở cấp độ chuỗi của nó thay vì âm thầm thay đổi bộ ước lượng hình ảnh; và
- Giữ lại đường ống dóng hàng SMPLer-X + WiLoR gốc làm một baseline bắt buộc.

Việc đóng băng mô hình quan sát rất quan trọng về mặt khoa học. Nó cho phép bài báo khẳng định rằng sự cải thiện đến từ việc phục dựng hậu phương đề xuất, mô hình độ không đảm bảo và lập luận ngôn ngữ/quan hệ chứ không phải từ một front-end mới được huấn luyện lại.

Các tài nguyên đã phát hành của WiLoR sử dụng một giấp phép hạn chế phi thương mại, không phái sinh. Do đó:

- Không tinh chỉnh hoặc phân phối lại trọng số WiLoR đã sửa đổi;
- Chỉ sử dụng WiLoR thông qua một đường dẫn suy luận đóng băng được cho phép;
- Báo cáo chính xác checkpoint và giấp phép trong bài báo và kho lưu trữ; và
- Cho phản hồi làm rõ bằng văn bản trước bất kỳ việc sử dụng nào không được trang bị rõ ràng bởi các điều khoản phát hành.

Các tài nguyên chính thức liên quan:

- [Kho lưu trữ chính thức và checkpoint Hand4Whole++](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE)
- [Kho lưu trữ chính thức và thông tin giấp phép WiLoR](https://github.com/rolpotamias/WiLoR)

### 18.3 Tiền phương tư thế không gian: Khởi tạo từ DPoser-X

DPoser-X là điểm khởi đầu công khai tương thích nhất cho tiền phương tư thế SMPL-X đề xuất. Nó mô hình hóa tư thế toàn cơ thể trong khi khai thác cả dữ liệu huấn luyện toàn cơ thể lẫn bộ phận, làm cho nó phù hợp hơn một bộ khử nhiễu không gian ngẫu nhiên cho việc hoàn thiện cơ thể, cổ tay và bàn tay.

Chiến lược kiến trúc khuyến nghị:

1. Tải checkpoint toàn cơ thể DPoser-X công khai;
2. Giữ lại biểu diễn khử nhiễu không gian theo từng khung hình của nó;
3. Bổ sung chú ý thời gian hoặc các khối adapter thời gian;
4. Bổ sung các phép chiếu cho độ không đảm bảo quan sát, âm tiết học, pha và các đặc trưng quan hệ;
5. Ban đầu đóng băng hoặc giảm trọng số cập nhật đáng kể cho các khối không gian pretrained;
6. Huấn luyện các mô-đun mới trên dữ liệu chuyển động và bàn tay tổng quát; và
7. Đồng thích ứng các khối không gian sau này trong quá trình tinh chỉnh đặc thù ký hiệu.

Nếu kiến trúc thời gian cuối cùng quá khác biệt cho việc chuyển giao trọng số trực tiếp, hãy dùng DPoser-X làm giáo viên:

- Khử nhiễu các tư thế SMPL-X bị hỏng bằng DPoser-X;
- Cất nén điểm tư thế sạch (clean-pose score) hoặc bản phục dựng của nó vào mô hình mới;
- Bảo tồn tiền phương không gian của nó trong khi học phân bố hậu phương thời gian mới; và
- So sánh sự khởi tạo trọng số với việc chắt lọc từ giáo viên (teacher distillation) trong một ablation.

Kho lưu trữ hiện chứa một checkpoint DPoser-X được tinh chỉnh ký hiệu cục bộ tại:

`DPoser-X/checkpoints/dposer/sign/sign_body_ft/last.ckpt`

Checkpoint đó hữu ích cho việc tích hợp và so sánh sơ bộ, nhưng tập huấn luyện phân đoạn cục bộ quá nhỏ để thiết lập kết quả bài báo cuối cùng. Mô hình bài báo nên được thích ứng bằng một tập hợp chuỗi đã qua lọc chất lượng lớn hơn đáng kể và được đánh giá so với checkpoint công khai gốc.

Tài nguyên liên quan:

- [Bài báo DPoser-X ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Lu_DPoser-X_Diffusion_Model_as_Robust_3D_Whole-body_Human_Pose_Prior_ICCV_2025_paper.pdf)
- [Kho lưu trữ chính thức DPoser-X](https://github.com/careless-lu/DPoser)
- [Trọng số pretrained công khai DPoser-X](https://huggingface.co/Moon-bow/DPoser-X)

Mã nguồn DPoser-X và trọng số phát hành không sử dụng các giấp phép giống hệt nhau. Hãy ghi lại cả giấp phép mã nguồn và giấp phép trọng số mô hình trong tài liệu phát hành.

### 18.4 Tiền phương thời gian: FUSION mạnh về kỹ thuật nhưng đòi hỏi cổng giấp phép

FUSION là một sự khởi tạo thời gian đặc biệt phù hợp vì nó là một tiền phương diffusion chuyển động cơ thể và bàn tay thống nhất. Về mặt kỹ thuật, đó là bước khởi động nóng thời gian tốt hơn việc học một Transformer chuyển động toàn cơ thể hoàn chỉnh từ số 0.

Tuy nhiên, giấp phép được công bố của nó hạn chế việc sửa đổi và phân phối lại phần mềm/mô hình đã phát hành. Do đó, một checkpoint SP4D tinh chỉnh thu được từ FUSION có thể không thể công bố nếu không có sự cho phép tường minh.

Sử dụng chiến lược hai luồng:

#### Luồng A: Khởi tạo FUSION được chấp thuận quyền hạn

- Liên hệ với các tác giả trước khi biến FUSION thành một phụ thuộc quan trọng;
- Yêu cầu sự cho phép tường minh để tinh chỉnh checkpoint;
- Yêu cầu sự cho phép để phát hành các trọng số hoặc adapter phái sinh;
- Lưu giữ bằng chứng về các điều khoản được cấp; và
- Báo cáo kết quả khởi tạo từ FUSION riêng biệt với con số có thể phân phối lại hoàn toàn.

#### Luồng B: Mô hình thời gian độc lập

- Khởi tạo các thành phần không gian từ DPoser-X;
- Bổ sung và huấn luyện các adapter thời gian hoặc một temporal Transformer;
- Huấn luyện trước các mô-đun thời gian mới trên các chuỗi cơ thể/tay tổng quát;
- Thực hiện thích ứng chuỗi đặc thù cho ký hiệu; và
- Phát hành mô hình này làm cấu hình tái lập chính.

FUSION cũng có thể được đánh giá như một giáo viên đóng băng hoặc baseline khi được phép. Dự án không được phụ thuộc vào việc xin phép muộn trong lịch trình xuất bản.

Tài nguyên liên quan:

- [Kho lưu trữ chính thức FUSION](https://github.com/enesduran/FUSION)
- [Giấp phép dự án FUSION](https://fusion.is.tue.mpg.de/license.html)

### 18.5 Âm tiết học và pha: Chuyển giao biểu diễn SHuBERT, huấn luyện các đầu mới

Không nên huấn luyện một bộ mã hóa video ký hiệu từ số 0. SHuBERT cung cấp các đặc trưng tự giám sát đặc thù ký hiệu được học từ một tập video ngôn ngữ ký hiệu lớn và sử dụng các luồng bàn tay, mặt và thân trên dóng hàng tốt với các biến điều kiện hóa đề xuất.

Khuyên dùng:

- Khởi tạo bộ mã hóa thị giác/ký hiệu từ trọng số SHuBERT công khai;
- Đóng băng các lớp sớm trong giai đoạn đầu đặc thù ký hiệu;
- Huấn luyện các đầu thuộc tính và pha mới;
- Sau đó áp dụng thích ứng lớp muộn với learning rate thấp hoặc LoRA nếu kiểm định cải thiện;
- Sử dụng các xác suất mềm thu được, không phải nhãn ngôn ngữ học cứng, làm các điều kiện SP4D; và
- Che các thuộc tính không tin cậy hoặc ngoài phạm vi bằng cách dùng độ tin cậy dự đoán.

SHuBERT không phải mô hình hình học 3D và không nên thay thế tiền phương tư thế. Nó nên cung cấp các đặc trưng ký hiệu ngữ cảnh giúp giải tỏa mơ hồ cho quan sát tư thế.

Các đầu mới nên dự đoán:

- Dạng bàn tay;
- Hướng lòng bàn tay/ngón tay;
- Vị trí tương đối so với cơ thể;
- Loại và hướng chuyển động;
- Cấu trúc một tay so với hai tay;
- Đối xứng và ưu thế;
- Khả năng tiếp xúc; và
- Các pha dừng, chuyển tiếp, lặp lại và chuyển tiếp tiếp xúc.

Chú giải HamNoSys không nên được coi là nhãn pha chính xác ở cấp độ khung hình. Chúng mô tả cấu trúc ký hiệu, nhưng ranh giới pha thời gian nên thu được từ:

- Vận tốc bàn tay và cổ tay;
- Sự thay đổi gia tốc và hướng;
- Thời điểm bắt đầu/kết thúc tiếp xúc;
- Cấu trúc lặp lại;
- Sự dóng hàng thời gian tới ranh giới ký hiệu; và
- Một tập con được duyệt thủ công.

Tài nguyên liên quan:

- [Kho lưu trữ chính thức và trọng số công khai SHuBERT](https://github.com/ShesterG/SHuBERT)
- [Dự án SignAvatars](https://signavatars.github.io/)
- [Kho lưu trữ chính thức SignAvatars](https://github.com/ZhengdiYu/SignAvatars)
- [ASL-LEX 2.0](https://asl-lex.org/about.html)
- [WLASL-LEX, ACL 2022](https://aclanthology.org/2022.acl-short.49/)

Các tài nguyên từ vựng riêng cho ASL là nguồn giám sát bổ sung hữu ích nhưng không được là nguồn ngôn ngữ duy nhất khi đánh giá DGS hoặc sự tổng quát hóa qua các ngôn ngữ.

### 18.6 Các mô-đun bắt buộc phải huấn luyện cho hệ thống đề xuất

#### 18.6.1 Độ tin cậy quan sát và độ không đảm bảo dị sai

Đầu độ không đảm bảo phải được huấn luyện trên chính các bộ ước lượng và quá trình nhiễu hỏng chính xác mà SP4D sử dụng. Các bộ dự đoán độ tin cậy tổng quát sẽ không được hiệu chuẩn cho Hand4Whole++, WiLoR, SMPLer-X hoặc các thất bại đặc trưng của chúng.

Ví dụ huấn luyện nên bao gồm:

- Độ mờ chuyển động và độ phân giải không gian thấp;
- Che khuất bàn tay một phần và toàn bộ;
- Bàn tay bị che bởi mặt và che bởi thân;
- Hoán đổi tay trái/phải;
- Hoán đổi danh tính bàn tay tương tác;
- Gắn cổ tay sai;
- Chiều sâu bàn tay sai;
- Khớp ngón tay không hợp lý;
- Bỏ sót phát hiện;
- Nhảy vọt thời gian; và
- Sự bất đồng giữa các chuyên gia quan sát.

Sử dụng cả nhiễu hỏng tổng hợp và các sai số còn lại thực tế của bộ ước lượng trên dữ liệu kiểm định 3D sạch. Hiệu chuẩn phương sai dự đoán sau khi huấn luyện bằng một tập chia hiệu chuẩn giữ riêng. Báo cáo expected calibration error, negative log likelihood, đường cong sai số-theo-độ tin cậy và độ chính xác chọn lọc khi các quan sát độ tin cậy thấp bị từ chối.

#### 18.6.2 Đồ thị quan hệ và tiếp xúc

Các đặc trưng tư thế cấp thấp hơn có thể được khởi tạo từ DPoser-X hoặc tiền phương thời gian, nhưng các cạnh quan hệ và dự đoán tiếp xúc nên được học riêng cho ngôn ngữ ký hiệu.

Dữ liệu pretraining khuyến nghị:

- InterHand2.6M cho khớp ngón tay và danh tính bàn tay tương tác;
- ARCTIC cho các chuỗi đồng bộ toàn cơ thể, hai tay và giàu tiếp xúc; và
- Contact4D cho độ hiển thị tiếp xúc và học đặc trưng nhạy cảm với tiếp xúc.

Tài nguyên liên quan:

- [Dự án chính thức InterHand2.6M](https://mks0601.github.io/InterHand2.6M/)
- [Dự án chính thức ARCTIC](https://arctic.is.tue.mpg.de/)
- [Contact4D, 3DV 2026](https://openreview.net/forum?id=5DPvfQtAjm)

Các tập dữ liệu này nhấn mạnh chuyển động tay-tay hoặc tay-vật thể và không giải quyết trực tiếp tiếp xúc tay-mặt, tay-thân hoặc tiếp xúc phụ thuộc ngôn ngữ của ngôn ngữ ký hiệu. Hãy tinh chỉnh đồ thị tiếp xúc bằng các tiếp xúc ký hiệu trích xuất từ hình học và một tập con ký hiệu được xác minh thủ công.

Tiếp xúc phải giữ tính xác suất. Năng lượng tiếp xúc nên được làm yếu đi hoặc tắt khi:

- Xác suất tiếp xúc thấp;
- Phân bố hậu phương trên vị trí tiếp xúc bị phân tán;
- Bằng chứng hình ảnh mâu thuẫn với tiếp xúc; hoặc
- Ước lượng ứng viên đòi hỏi sự biến dạng cơ thể hoặc ngón tay không hợp lý.

#### 18.6.3 Bộ lựa chọn bằng chứng đa giả thuyết

Bộ lựa chọn bằng chứng phải được huấn luyện sau khi mô hình sinh tạo ra các ứng viên có ý nghĩa. Nó cần học hành vi của chính các mẫu hậu phương SP4D và ngân hàng quan sát chính xác được dùng lúc suy luận.

Huấn luyện bộ lựa chọn bằng cách dùng:

- Sai số tư thế có giám sát trên dữ liệu huấn luyện và kiểm định bên ngoài;
- Dự đoán quan sát bị ẩn;
- Tính nhất quán thời gian;
- Khả năng xuất hiện quan sát đã hiệu chuẩn;
- Tính hợp lý về mặt vật lý;
- Tính nhất quán quan hệ; và
- Sự đồng thuận tiếp xúc nhận biết độ không đảm bảo.

Không tinh chỉnh bộ lựa chọn trên ground truth của tập kiểm thử SGNify. Bộ lựa chọn nên chấm điểm các ứng viên được tạo ra độc lập và không được trở thành một oracle gián tiếp đặc thù cho benchmark.

#### 18.6.4 Tinh chỉnh cuối cùng

Bắt đầu với tối ưu hóa vi phân được trên SMPL-X thay vì một mạng học lớn khác. Tối ưu hóa:

- Các quan sát 2D/3D tin cậy;
- Tính nhất quán hậu phương;
- Gắn nối bàn tay/cơ thể;
- Động lực học thời gian;
- Các thành phần mềm quan hệ/tiếp xúc;
- Giới hạn khớp; và
- Hình dạng dùng chung.

Chỉ thay thế bộ tối ưu này bằng một bộ tinh chỉnh học được nếu việc đo hiệu năng (profiling) chứng minh rằng thời gian chạy là hạn chế quyết định tới việc xuất bản.

### 18.7 Chiến lược dữ liệu và chất lượng giám sát

SignAvatars phù hợp cho việc thích ứng ký hiệu quy mô lớn, nhưng các chú giải 3D phục dựng của nó là giả ground truth (pseudo-ground truth) chứ không phải chuyển động thu thập từ cảm biến (marker-based mocap) sạch đồng nhất. Huấn luyện hoàn toàn trên các nhãn này có thể tái lặp các thiên kiến của giáo viên gán nhãn.

Sử dụng 3 tầng giám sát:

#### Tầng 1: Pretraining hình học và chuyển động sạch

- Các miền huấn luyện DPoser-X công khai nơi giấp phép cho phép;
- InterHand2.6M;
- ARCTIC;
- Dữ liệu chuyển động SMPL-X chất lượng cao có sẵn; và
- Các chuỗi chuyển động nội bộ hoặc bản quyền được xác minh cẩn thận.

#### Tầng 2: Thích ứng đặc thù ký hiệu lớn

- SignAvatars với lọc chất lượng;
- How2Sign nơi dữ liệu và chú giải đòi hỏi được cấp phép;
- Dữ liệu huấn luyện DexAvatar;
- Giả nhãn tách biệt người ký hiệu tạo bởi các chuyên gia đóng băng mạnh; và
- Các tài nguyên âm tiết học/từ vựng tương thích với từng ngôn ngữ ký hiệu.

#### Tầng 3: Kiểm định ký hiệu được xác minh thủ công

Tạo một tập hợp nhỏ nhưng chất lượng cao chứa:

- Tư thế bàn tay và cổ tay chính xác;
- Vị trí bàn tay tương đối so với cơ thể;
- Danh tính trái/phải;
- Tiếp xúc tay–tay và tay–cơ thể;
- Thứ tự chiều sâu;
- Ranh giới pha; và
- Nhãn thất bại rõ ràng cho các chuyên gia quan sát.

Tập hợp này là cần thiết để:

- Hiệu chuẩn độ không đảm bảo;
- Đo lường thiên kiến của giáo viên;
- Kiểm định các dự đoán tiếp xúc và pha;
- Chọn checkpoint mô hình;
- Chẩn đoán xem âm tiết học có giúp ích cho các khung hình thực sự khó không; và
- Ngăn chất lượng giả nhãn trở thành mục tiêu đánh giá ẩn.

Tất cả các tập chia nên tách biệt người ký hiệu và nguồn nếu có thể. SGNify phải giữ hoàn toàn chỉ cho đánh giá.

### 18.8 Chương trình huấn luyện đã sửa đổi

#### Giai đoạn 0: Kiểm định bộ đánh giá và tọa độ

1. Tái lập chính xác giao thức đánh giá DexAvatar chính thức.
2. Xác minh đơn vị, hướng toàn cục, tọa độ camera, quy ước khớp SMPL-X, ánh xạ MANO sang SMPL-X, và dóng hàng thời gian.
3. Báo cáo các chỉ số theo khung hình và tổng hợp trên cùng tập khung hình với các nghiên cứu trước.
4. Đóng băng bộ đánh giá trước khi phát triển phương pháp.

Điều kiện Go/No-go: không huấn luyện SP4D cho đến khi baseline phát hành và bộ đánh giá cục bộ khớp nhau trong khoảng dung sai có thể giải thích.

#### Giai đoạn 1: Baseline quan sát đóng băng mạnh nhất

1. Chạy checkpoint Hand4Whole++ đã phát hành.
2. Giữ lại baseline dóng hàng SMPLer-X + WiLoR.
3. Đánh giá cả hai dưới giao thức chính xác.
4. Phân loại các thất bại theo độ mờ, che khuất, tương tác, gắn cổ tay, chiều sâu và sự mất ổn định thời gian.

Điều kiện Go/No-go: nếu bộ khởi tạo hiện đại không tạo ra một baseline cạnh tranh, hãy sửa tiền xử lý và chuyển đổi tọa độ trước khi huấn luyện các mô hình hạ nguồn.

#### Giai đoạn 2: Bộ tinh chỉnh thời gian pretrained định tính

1. Khởi tạo tiền phương không gian từ DPoser-X.
2. Bổ sung các adapter thời gian và điều kiện hóa quan sát XYZ đầy đủ.
3. Đưa vào các ràng buộc chuỗi cổ tay, pháp tuyến lòng bàn tay, đầu ngón tay và hình dạng dùng chung.
4. Huấn luyện phục dựng từ nhiễu-sang-sạch trước khi đưa vào lấy mẫu diffusion.

Điều kiện Go/No-go: mô hình định tính phải cải thiện giáo viên đóng băng trên dữ liệu kiểm định có giám sát bên ngoài, không chỉ trên tập huấn luyện giả nhãn của nó.

#### Giai đoạn 3: Độ không đảm bảo được hiệu chuẩn

1. Tạo các ví dụ nhiễu hỏng và sai số còn lại của bộ ước lượng thực tế.
2. Huấn luyện đầu độ tin cậy/phương sai.
3. Hiệu chuẩn trên các chuỗi thực tế được giữ riêng.
4. Xác minh sự cải thiện đặc biệt trên các tập con có độ tin cậy thấp và bị lỗi.

Điều kiện Go/No-go: độ không đảm bảo dự đoán phải tương quan với sai số thực tế và cải thiện sự phục dựng khi được dùng làm trọng số.

#### Giai đoạn 4: Điều kiện hóa quan hệ và tiếp xúc

1. Huấn luyện trước các đặc trưng tương tác trên InterHand2.6M và ARCTIC.
2. Bổ sung các đầu quan hệ/tiếp xúc xác suất.
3. Thích ứng với các quan hệ tay–tay và tay–cơ thể trong ngôn ngữ ký hiệu.
4. Đo độ chính xác tiếp xúc riêng biệt với TR-V2V.

Điều kiện Go/No-go: điều kiện hóa quan hệ không được làm giảm độ chính xác hình học thông qua các tiếp xúc sai bị ép buộc.

#### Giai đoạn 5: Masked diffusion posterior

1. Chuyển đổi bộ tinh chỉnh định tính thành công thành masked conditional diffusion.
2. Chỉ sử dụng khởi tạo FUSION nếu có được quyền hạn.
3. Nếu không, giữ lại khởi tạo không gian DPoser-X và các khối thời gian được huấn luyện độc lập.
4. Bắt đầu với 1 mẫu hậu phương trong quá trình phát triển mô hình.

Điều kiện Go/No-go: diffusion phải cải thiện các tập con khó/mơ hồ hoặc likelihood đã hiệu chuẩn, chứ không chỉ tạo ra chuyển động trông mượt mà hơn.

#### Giai đoạn 6: Âm tiết học và pha

1. Khởi tạo bộ mã hóa ký hiệu từ SHuBERT.
2. Huấn luyện các đầu âm tiết học mềm và pha.
3. Điều kiện hóa SP4D bằng xác suất và mặt nạ độ tin cậy.
4. Đánh giá sự tăng trưởng trên các tập con bị che, mờ, tương tác nặng và qua nhiều ngôn ngữ.

Điều kiện Go/No-go: âm tiết học phải cải thiện hình học hoặc tính chính xác ngữ nghĩa trên các trường hợp khó mục tiêu. Nếu không, hãy giữ nó làm giám sát phụ trợ thay vì một đóng góp headline.

#### Giai đoạn 7: Lựa chọn đa giả thuyết

1. Tạo một tập ứng viên nhỏ, như `K = 2` hoặc `K = 4`.
2. Huấn luyện bộ lựa chọn bằng chứng trên các chuỗi huấn luyện/kiểm định giữ riêng.
3. So sánh lựa chọn với tư thế trung bình, likelihood cao nhất, và lựa chọn hoàn hảo (oracle).
4. Chỉ tăng `K` nếu bộ lựa chọn lấp đầy được một phần đáng kể khoảng cách tới oracle.

### 18.9 Các triệt tiêu khởi tạo và chuyển giao bắt buộc

Bảng triệt tiêu cuối cùng nên bao gồm:

1. Khởi tạo không gian ngẫu nhiên so với khởi tạo DPoser-X;
2. Khởi tạo DPoser-X so với chắt lọc giáo viên DPoser-X;
3. Luồng thời gian DPoser-X so với khởi tạo FUSION, nếu có sẵn về mặt pháp lý;
4. Bộ mã hóa ký hiệu ngẫu nhiên so với SHuBERT đóng băng so với SHuBERT thích ứng;
5. Hand4Whole++ đóng băng so với baseline SMPLer-X + WiLoR cũ hơn;
6. Độ tin cậy quan sát cố định so với độ không đảm bảo học được;
7. Độ không đảm bảo trước và sau khi hiệu chuẩn;
8. Không có đồ thị quan hệ so với chỉ quan hệ so với quan hệ cộng tiếp xúc;
9. Không có âm tiết học so với âm tiết học mềm dự đoán;
10. Không có pha so với pha động học so với pha học được;
11. Bộ tinh chỉnh định tính so với phân bố hậu phương diffusion;
12. `K = 1`, `K > 1` được chọn, và `K > 1` hoàn hảo (oracle); và
13. Bật/tắt tinh chỉnh tối ưu hóa.

Các triệt tiêu này là cần thiết để chứng minh rằng kết quả bài báo không bị giải thích đơn thuần bởi một checkpoint quan sát công khai mạnh hơn.

### 18.10 Cấu hình tái lập chính và cấu hình nghiên cứu tùy chọn

#### Cấu hình tái lập chính

Mục tiêu phát hành ưu tiên là:

> **Các quan sát Hand4Whole++ đóng băng + Khởi tạo không gian DPoser-X + Các adapter thời gian huấn luyện độc lập + Độ không đảm bảo được hiệu chuẩn + Các mô-đun quan hệ/pha/âm tiết học đặc thù ký hiệu + Tinh chỉnh dựa trên bằng chứng.**

Luồng này giảm thiểu sự phụ thuộc vào các mô hình có giấp phép cấm phân phối các trọng số phái sinh.

#### Cấu hình hiệu năng cao nhất tùy chọn

Nếu có được chấp thuận bằng văn bản:

> **Các quan sát Hand4Whole++ đóng băng + Khởi tạo thời gian FUSION + Đặc trưng ký hiệu SHuBERT + Các mô-đun độ không đảm bảo, quan hệ, pha và lựa chọn SP4D đề xuất.**

Báo cáo cấu hình này riêng biệt nếu checkpoint của nó không thể phát hành dưới cùng điều khoản với mô hình chính.

Các chuyên gia tùy chọn OmniHands, SAM 3D Body, SMPLest-X hoặc các chuyên gia khác nên xuất hiện như các giả thuyết bổ sung hoặc triệt tiêu thay vì các phần bắt buộc của phương pháp trung tâm. Các mô hình không có checkpoint công khai được xác minh hoặc giấp phép rõ ràng không được trở thành các phụ thuộc quan trọng.

### 18.11 Khẳng định cấp A* và kiểm soát phạm vi

Các mô hình pretrained nên được đối xử như các nền tảng có kiểm soát, không phải là đóng góp chính của bài báo. Khẳng định trung tâm nên là:

> **Suy luận phân bố hậu phương điều kiện hóa theo ký hiệu và hiệu chuẩn độ không đảm bảo tạo ra bản phục dựng toàn cơ thể nhất quán theo thời gian và quan hệ dưới các quan sát bàn tay mơ hồ.**

Các đóng góp có thể bảo vệ được mạnh nhất là:

1. Mô hình hóa quan sát dị sai đặc thù cho từng bộ ước lượng;
2. Suy luận toàn chuỗi bị che có điều kiện hóa theo âm tiết học;
3. Lập luận quan hệ tay–tay và tay–cơ thể mang tính xác suất tường minh;
4. Động lực học thời gian nhận biết pha; và
5. Lựa chọn dựa trên bằng chứng từ một phân bố hậu phương được hiệu chuẩn.

Bài báo nên tránh trình bày một tập hợp lớn các mô hình công khai làm tính mới của mình. Sử dụng một mô hình quan sát đóng băng chính, một tiền phương hình học/chuyển động, và một bộ mã hóa ký hiệu trong phương pháp chính. Đưa các chuyên gia bổ sung vào các thực nghiệm phụ trợ.

Kết quả tại một hội nghị A* không thể được đảm bảo chỉ bằng việc lựa chọn mô hình. Nghiên cứu sẽ đòi hỏi thêm:

- Tái lập benchmark chính xác và minh bạch;
- Các baseline hiện đại mạnh;
- Kiểm định sạch tách biệt người ký hiệu;
- Các phân tích triệt tiêu mục tiêu chuyên sâu;
- Đánh giá độ bền vững và độ hiệu chuẩn;
- Đánh giá ngữ nghĩa và cảm nhận ngoài sai số hình học;
- Báo cáo thời gian chạy và khả năng tái lập; và
- Các điều khoản phát hành cho phép các nhà nghiên cứu khác đánh giá cấu hình chính.

### 18.12 Quyết định cài đặt cuối cùng

Quyết định khuyến nghị cho kho lưu trữ này là:

- **Không huấn luyện mạng xương sống quan sát từ số 0;**
- **Không huấn luyện tiền phương SMPL-X không gian từ số 0;**
- **Không huấn luyện biểu diễn video ký hiệu từ số 0;**
- **Huấn luyện các adapter thời gian và đường dẫn điều kiện hóa SP4D;**
- **Huấn luyện các đầu độ không đảm bảo, âm tiết học, pha, quan hệ/tiếp xúc và lựa chọn bằng chứng;**
- **Chỉ sử dụng FUSION sau khi có cổng chấp thuận và phát hành rõ ràng;**
- **Coi checkpoint DPoser-X tinh chỉnh ký hiệu nhỏ hiện tại là checkpoint tích hợp chứ không phải mô hình nghiên cứu cuối cùng;** và
- **Xây dựng một tập kiểm định ký hiệu được xác minh thủ công trước khi mở rộng toàn bộ phân bố hậu phương.**

Chiến lược này mang lại sự cân bằng tốt nhất giữa độ chính xác có thể đạt được, tính mới khoa học, khả năng tái lập, sự an toàn về giấp phép và bằng chứng được kỳ vọng trong một bài nộp cạnh tranh tại CVPR/ICCV/ECCV.
