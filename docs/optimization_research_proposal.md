# Đề xuất Hướng Nghiên Cứu: Tối Ưu Hóa Thích Ứng Khử Nhiễu (Langevin Dynamics) thay thế L-BFGS cho DexAvatar
Tài liệu này chi tiết hóa hướng nghiên cứu khoa học nhằm thay thế thuật toán tối ưu hóa L-BFGS truyền thống trong DexAvatar bằng các thuật toán tối ưu ngẫu nhiên và thích ứng mới, đặc biệt là **Joint-Specific Adaptive Langevin Dynamics (JS-ALD)**, nhằm vượt qua các giới hạn của SOTA hiện tại và hướng tới công bố tại các hội nghị Computer Vision hàng đầu (CVPR/ICCV/ECCV/WACV).
---
## 1. Vấn đề của bộ tối ưu hóa L-BFGS hiện tại trong DexAvatar
Trong pipeline DexAvatar gốc, quá trình khớp mesh SMPL-X sử dụng thuật toán **L-BFGS với Strong Wolfe Line Search** để tối ưu hóa không gian ẩn của VAE (SignBPoser và SignHPoser). Phương pháp này đối mặt với các nút thắt cổ chai lớn:
* **Sự không liên tục của hàm Loss (Non-smooth Loss Landscape):** Các hàm loss va chạm vật lý (Interpenetration/Collision Loss) và các giới hạn sinh học khớp (Biomechanical joint limits) có biên gradient rất sắc nhọn và không liên tục. L-BFGS giả định hàm loss trơn tuột nên rất dễ bị lỗi dòng tìm kiếm (Line Search Failure) hoặc dừng sớm (premature convergence) khi chạm vào các ranh giới này.
* **Mắc kẹt ở cực tiểu cục bộ (Local Minima trap):** L-BFGS là thuật toán tất định (deterministic). Khi điểm khóa 2D từ Sapiens bị nhiễu (do nhòe chuyển động) hoặc bị che khuất, L-BFGS không thể thoát khỏi các bẫy cục bộ để tìm ra cấu hình 3D chính xác.
* **Bị giới hạn bởi bộ nén ẩn VAE (Latent Bottleneck):** Việc ép tối ưu hóa trong không gian ẩn $33$-chiều (thân) và $23$-chiều (tay) của VAE làm triệt tiêu các chuyển động ngón tay chi tiết và độc đáo của ngôn ngữ ký hiệu (hiện tượng Mode Collapse).
---
## 2. Giải pháp Đề xuất: Joint-Specific Adaptive Langevin Dynamics (JS-ALD)
Ý tưởng cốt lõi là thay thế L-BFGS bằng **Động lực học Langevin thích ứng theo từng khớp xương (JS-ALD)**, tối ưu trực tiếp trên không gian góc khớp (Rotation Space) và được dẫn dắt bởi DPoser-X (thân) cùng SignHPoser VAE Projection (tay).
```
 3 Khối Khởi tạo thô (Sapiens 2D + Confidence C_i, SMPLer-X Body, HaMeR/WiLoR Hands)
                                        │
                                        ▼
                  Khởi tạo tư thế ban đầu ở bước t = T: θ^(T)
                                        │
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ Vòng lặp tối ưu Langevin Step t (Khử nhiễu thích ứng từ t = T về 0):         │
 │                                                                             │
 │ 1. Prior Score: Lấy gradient từ DPoser-X (thân) & Hand VAE (tay)            │
 │ 2. Data Gradient: Lấy gradient của E_data (2D keypoints + 3D hand match)    │
 │ 3. Adaptive Update cho từng khớp i theo độ tự tin C_i:                      │
 │    θ_i^(t-1) = θ_i^(t) + α_i * (g_prior_i + λ_i(C_i)*g_data_i) + σ_i(C_i)*ε │
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │
                                        ▼
                              Mesh 3D tối ưu cuối cùng
```
### 2.1. Công thức Toán học thích ứng (Adaptive Langevin Formulation)
Tối ưu hóa Langevin Dynamics truyền thống cập nhật toàn bộ khớp với cùng một bước nhảy và nhiễu ngẫu nhiên. Trong ngôn ngữ ký hiệu, điều này không hợp lý vì các khớp thân (lưng, cổ) rất ổn định và rõ ràng, trong khi khớp tay (cổ tay, ngón tay) di chuyển cực nhanh và dễ bị che khuất.
Với mỗi khớp $i$ tại bước khử nhiễu $t$ (đi từ $T$ về $0$), ta cập nhật tư thế khớp $\theta_i$ theo công thức:
$$\theta_i^{(t-1)} = \theta_i^{(t)} + \alpha_t^i \left[ g_{prior}^i + \lambda_i(C_i) \cdot g_{data}^i \right] + \sigma_t^i(C_i) \cdot \epsilon_i$$
Trong đó:
* **$\theta_i^{(t)}$:** Góc xoay khớp $i$ (dạng axis-angle $3$ chiều) ở bước tối ưu $t$.
* **$\alpha_t^i$:** Tốc độ học (step size) thích ứng cho từng khớp (khớp ngón tay nhỏ cần bước nhảy nhỏ hơn khớp vai).
* **$\epsilon_i \sim \mathcal{N}(0, I)$:** Nhiễu Gauss ngẫu nhiên để thuật toán "rung lắc" thoát khỏi các cực tiểu cục bộ của ảnh 2D nhiễu.
* **$C_i \in [0, 1]$:** Độ tự tin (confidence score) của điểm khóa 2D khớp $i$ do Sapiens cung cấp.
* **$\lambda_i(C_i)$ (Trọng số dữ liệu thích ứng):**
  $$\lambda_i(C_i) = \lambda_{base} \cdot C_i$$
  Nếu khớp rõ ràng ($C_i \approx 1$), lực kéo từ ảnh 2D cực mạnh. Nếu khớp bị che khuất ($C_i \approx 0$), ta bỏ qua hoàn toàn ảnh 2D để tránh bị kéo lệch mesh.
* **$\sigma_t^i(C_i)$ (Hệ số nhiễu thích ứng):**
  $$\sigma_t^i(C_i) = \sigma_{base} \cdot (1 - C_i) \cdot \gamma_t$$
  Nếu khớp bị che khuất ($C_i$ thấp), ta tăng cường nhiễu ngẫu nhiên để thuật toán tự do khám phá các tư thế hợp lệ từ Prior. Nếu khớp rõ ràng, nhiễu được triệt tiêu về $0$. ($\gamma_t$ là hệ số suy giảm nhiễu theo thời gian).
### 2.2. Cơ chế xử lý Priors trong JS-ALD
* **Với Thân (Body):**
  Khớp thân được tối ưu trực tiếp trên không gian góc khớp $63$-chiều, dẫn dắt bởi Score function từ **DPoser-X** (mô hình khuếch tán):
  $$g_{prior\_body} = \nabla_{\theta_{body}} \log p(\theta_{body})$$
* **Với Tay (Hand):**
  Chúng ta vẫn giữ mô hình VAE (**SignHPoser**) nhưng không tối ưu hóa trên latent space $23$ chiều của nó. Ta tối ưu trực tiếp trên không gian khớp tay $45$-chiều, nhưng ở mỗi bước lặp Langevin, ta áp dụng một toán tử chiếu **VAE Projection Operator**:
  $$\theta_{hand\_projected}^{(t)} = \text{Decoder}(\text{Encoder}(\theta_{hand}^{(t)}))$$
  Toán tử này hoạt động như một bộ lọc giải phẫu học, lập tức kéo các ngón tay bị uốn éo dị dạng (do nhiễu ngẫu nhiên của Langevin) trở lại phân phối hình học bàn tay ký hợp lệ.
---
## 3. Các thuật toán tối ưu hóa thay thế khác (Dành cho Literature Review & So sánh)
Để bài báo có độ dày nghiên cứu, ngoài JS-ALD, bạn nên đưa vào so sánh thực nghiệm (ablation study) các thuật toán tối ưu hóa hiện đại sau:
### 3.1. AdamW kết hợp Cosine Annealing (Tối ưu hóa Gradient Descent thích ứng)
* **Ý tưởng:** Áp dụng AdamW trực tiếp vào tối ưu hóa tham số mesh (test-time optimization).
* **Cơ chế:** Sử dụng moment động lượng để lướt qua các điểm yên ngựa (saddle points) và tự thích ứng tốc độ học theo độ dốc gradient.
* **Cách căn chỉnh (Tuning):**
  * Tốc độ học khởi đầu: $LR = 10^{-2}$.
  * Sử dụng **Cosine Annealing Warm Restarts** scheduler: Cho phép LR dao động hình sin để thuật toán định kỳ nhảy ra khỏi cực tiểu cục bộ trong các pha tối ưu hóa đầu tiên.
  * Phù hợp tối ưu khi có loss va chạm tay-ngực phức tạp vì AdamW không nhạy cảm với các vùng không khả vi liên tục như L-BFGS.
### 3.2. Vùng tin cậy Levenberg-Marquardt (LM Trust-Region)
* **Ý tưởng:** Coi bài toán khớp mesh là tối ưu hóa bình phương tối thiểu phi tuyến (Non-linear Least Squares).
* **Cơ chế:** Định nghĩa một vùng bán kính tin cậy xung quanh bước nhảy. Nếu hàm loss giảm đúng như dự đoán của mô hình bậc hai, ta mở rộng vùng tin cậy; ngược lại, ta thu hẹp vùng tin cậy.
* **Cách căn chỉnh (Tuning):**
  * Thiết lập ma trận trọng số Jacobian để cân bằng gradient giữa các khớp lớn (vai) và khớp nhỏ (ngón tay).
  * Thích hợp nhất cho giai đoạn tinh chỉnh cuối cùng (Stage 3) vì nó đảm bảo tính hội tụ toán học cực kỳ ổn định.
---
## 4. Chiến lược Căn chỉnh Tối ưu (Calibration and Tuning Strategy)
Để các bộ tối ưu trên đạt hiệu quả tốt nhất, cần áp dụng cơ chế tối ưu phân rã **Block Coordinate Descent (BCD)**:
1. **Phân rã tham số (Decoupling):** Chia quá trình tối ưu thành các nhóm độc lập chạy luân phiên:
   * *Nhóm 1:* Định vị toàn cục (Camera và Global Orient).
   * *Nhóm 2:* Cấu trúc cơ thể (Body Pose và Betas).
   * *Nhóm 3:* Chi tiết cử chỉ (Left Hand, Right Hand, Expression).
2. **Học vị riêng biệt (Joint-specific Learning Rates):**
   * Khớp vai/lưng: LR lớn ($10^{-2}$) vì biên độ di chuyển rộng.
   * Khớp ngón tay: LR nhỏ ($10^{-3}$ đến $5 \cdot 10^{-4}$) để tránh biến dạng cục bộ.
3. **Cân bằng trọng số loss động (Dynamic Loss Balancing):**
   * Sử dụng cơ chế cân bằng trọng số tự động dựa trên độ bất định homoscedastic (Homoscedastic Uncertainty Weighting) để tự điều chỉnh tỷ lệ giữa $E_{2D}$, $E_{3D\_hand}$, $E_{collision}$ và $E_{prior}$ theo từng bước lặp, tránh việc một hàm loss lấn át hoàn toàn các hàm loss khác.
---
## 5. Kế hoạch Thực nghiệm & Đánh giá (Evaluation Roadmap)
Để chứng minh tính hiệu quả của phương pháp tối ưu hóa mới so với DexAvatar gốc (L-BFGS):
* **Dataset sử dụng:** SGNify Dataset (bao gồm ảnh monocular và ground truth 3D từ hệ thống motion capture).
* **Các Metric so sánh:**
  * **TR-V2V (Translation-Removed Vertex-to-Vertex):** Đo bằng mm trên 3 vùng độc lập (Upper Body, Left Hand, Right Hand). Kỳ vọng JS-ALD giảm sai số tay từ $10\%$ đến $15\%$.
  * **MPVPE (Mean Per-Vertex Position Error):** Đánh giá độ khớp hình học tổng thể.
  * **Jitter Metric:** Đo độ mượt mà chuyển động xuyên suốt các frame (chứng minh việc thêm nhiễu thích ứng của Langevin không làm mất đi tính ổn định liên tục của video).
* **Ablation Studies cần thực hiện:**
  * So sánh L-BFGS vs AdamW vs Langevin Dynamics truyền thống vs JS-ALD đề xuất.
  * Đánh giá hiệu quả của VAE Projection Operator đối với ngón tay.
  * Khảo sát độ nhạy của độ tự tin $C_i$ lấy từ Sapiens đối với hệ số nhiễu $\sigma_t^i$.
