# Vượt SOTA DexAvatar trên TR‑V2V (Table‑1) sau khi thay HAMER → WiLoR

> Mục tiêu: đề xuất các hướng method mới (journal‑friendly) dựa trên pipeline DexAvatar gốc + hand model WiLoR để **cải thiện TR‑V2V (mm)** ở 3 vùng: **UBody(-F), LHand, RHand** trên split **57 signs** (protocol Table‑1 trong paper DexAvatar).
>
> Tài liệu này tập trung vào *ý tưởng phương pháp + cách tích hợp vào codebase hiện tại + kế hoạch ablation/eval*. Không nhằm tái mô tả lại paper.

---

## 0) Tóm tắt pipeline hiện tại (để bám điểm can thiệp)

Pipeline DexAvatar (gốc) có 4 stage chính:

1. **Sapiens**: 2D keypoints (COCO-WholeBody 133)
2. **SMPLer‑X**: init SMPL‑X params (body + camera + hand pose thô)
3. **Hand specialist**: trước là HAMER, nay là **WiLoR** (MANO + 2D/3D hand keypoints)
4. **SMPLify‑X fitting**: tối ưu SMPL‑X per frame với loss gồm 2D reprojection + priors + hand 3D prior + init‑hand prior

Điểm ghép hiện tại (trong fitting):
- 2D hand keypoints từ hand model thay cho Sapiens ở slot tay
- hand_pose từ hand model thay cho SMPLer‑X init
- 3D hand keypoints dùng làm prior (wrist‑relative + normalization)

=> Muốn vượt SOTA: phải **cải thiện hand articulation (LHand/RHand)** *mà không làm degrade body* và (quan trọng) giảm lỗi ở đoạn khó như occlusion, motion blur, one‑handed sign.

---

## 1) Những nguyên nhân khiến TR‑V2V còn cao (các “failure modes” thường gặp)

1. **Hand‑body inconsistency**: hand pose tốt nhưng không khớp forearm/wrist orientation của SMPL‑X → lỗi tay tăng hoặc body tăng.
2. **Single‑frame fitting** (hoặc temporal yếu): jitter frame‑to‑frame → TR‑V2V trung bình tăng, nhất là hands.
3. **Uncertainty không được mô hình hóa**: khi detector/hand model fail, pipeline vẫn “tin” 2D/3D hand, kéo fit sai.
4. **Scale/camera mismatch**: hand 3D prior wrist‑relative giúp ổn, nhưng vẫn có mismatch về rotation / handedness / coordinate.
5. **Contact/interaction constraints thiếu**: bàn tay gần mặt/torso hoặc self‑contact (tay chạm tay) không được ràng buộc → lỗi vùng upper body và hands.
6. **One‑handed sign ambiguity**: chọn active hand sai hoặc propagate sai ở frame không detect.

---

## 2) Method ideas (có novelty rõ, dễ publish) — theo nhóm

### A) Uncertainty‑Aware Hand Fusion (UAHF): dùng độ tin cậy để cân loss tay

**Ý tưởng**: WiLoR (và detector) có confidence (detection conf, reprojection conf, keypoint conf). Ta dùng nó để:
- down‑weight 2D hand keypoint loss khi occluded/blur
- down‑weight 3D hand prior khi geometry bất thường
- tránh “kéo sai” SMPL‑X.

**Tích hợp**:
- Xuất thêm metadata confidence từ WiLoR exporter (nếu có) vào `hamer.pkl`-compatible dict (vd thêm `pred_conf` hoặc `kp_conf`).
- Trong fitting loss, thay `hand_joints_weights` cố định bằng `hand_joints_weights * conf(frame,hand)`.

**Ablations**:
- baseline WiLoR (no uncertainty)
- + detection confidence only
- + per‑joint confidence
- + robust loss schedule (Huber/Geman) theo conf

**Rủi ro**: nếu conf không calibration tốt, có thể bỏ qua tay quá nhiều → tay “drift”.

---

### B) Temporal Hand‑Body Consistency (THBC): ràng buộc thời gian cho pose và vertices

**Ý tưởng**: Table‑1 metric trung bình trên frames, nên giảm jitter đem lại lợi lớn. Thay vì fit mỗi frame độc lập, thêm temporal smoothing có chọn lọc.

**Các biến thể**:
1. **Temporal pose smoothness**: L2/L1 trên axis‑angle của body_pose + hand_pose giữa t và t‑1.
2. **Temporal vertex smoothness**: smoothing trên subset vertices (hands/upper body) theo AR(1).
3. **Keyframe + tracking**: fit keyframes mạnh, frames giữa dùng warm‑start + penalty nhỏ.

**Tích hợp**:
- thêm chế độ “sequence fitting” trong `smplifyx/main.py` để fit theo chunk timeline (theo `segment.json`).
- warm‑start parameters từ frame trước (đã có prev_* cho 1‑hand; mở rộng cho toàn bộ pose/cam).

**Ablations**:
- no temporal
- + warm‑start only
- + temporal pose loss
- + temporal vertex loss

**Rủi ro**: over‑smooth làm mất chuyển động nhanh của sign (cần robust schedule / adaptive weight theo motion).

---

### C) Hand‑Arm Kinematic Coupling (HAKC): khớp wrist/forearm orientation giữa MANO và SMPL‑X

**Ý tưởng**: WiLoR/MANO chỉ mô hình hóa bàn tay, nhưng SMPL‑X cần liên kết wrist & forearm. Thêm loss ràng buộc:
- hướng trục cổ tay (wrist) của SMPL‑X khớp với wrist orientation suy ra từ MANO/hand joints
- đồng bộ bone direction (wrist→index MCP) thay vì pose trực tiếp.

**Tích hợp**:
- compute wrist frame từ hand keypoints 3D (WiLoR) và từ SMPL‑X joints hiện tại
- thêm penalty rotation alignment / directional cosine

**Ablations**:
- baseline
- + direction alignment
- + full frame alignment (3 axis)

**Rủi ro**: coordinate mismatch (left/right flip) phải nhất quán.

---

### D) Contact‑Aware Fitting: ràng buộc self‑contact (hand↔torso/hand)

**Ý tưởng**: trong sign language, tay thường gần torso/face hoặc chạm tay. Nếu không có contact constraints, fitting dễ “xuyên mesh” hoặc offset.

**Các thành phần**:
- **penetration penalty** (đã có interpenetration nhưng thường global): làm mạnh hơn cho hand‑torso
- **soft contact loss**: khi khoảng cách hand‑torso trong 2D nhỏ và depth prior hợp lý → khuyến khích gần nhưng không xuyên.

**Tích hợp**:
- dùng region indices (hands/upper body) để tính distance field nhanh
- chỉ bật loss khi tín hiệu 2D gần contact (heuristic)

**Ablations**:
- baseline
- + hand‑torso anti‑penetration
- + conditional soft contact

**Rủi ro**: contact heuristic sai → kéo tay vào torso.

---

### E) Multi‑Cue Hand Robustness: kết hợp 2D (Sapiens) + 2D/3D (WiLoR) thay vì overwrite cứng

**Ý tưởng**: hiện tại pipeline overwrite keypoints tay bằng WiLoR. Đôi khi Sapiens 2D ổn hơn khi WiLoR fail. Thay overwrite bằng fusion:

- `kp2d_hand = w * kp2d_wilor + (1-w) * kp2d_sapiens`
- `w` từ uncertainty (A) hoặc từ reprojection consistency.

**Tích hợp**:
- trong data_parser: giữ lại Sapiens hand kps, blend thay vì replace.

**Ablations**:
- overwrite
- blend with fixed w
- adaptive w by conf

**Rủi ro**: nếu 2 nguồn coordinate mismatch, cần normalize kỹ.

---

### F) Self‑Training / Test‑Time Adaptation (TTA) cho hand model (WiLoR) hoặc detector

**Ý tưởng**: video sign có domain shift (lighting, pose, blur). TTA có thể giảm lỗi hand → trực tiếp giảm LHand/RHand.

**Nhẹ (khả thi)**:
- optimize camera/scale/offset head của WiLoR trên clip bằng photometric / 2D reprojection consistency
- hoặc fine‑tune detector threshold per clip.

**Ablations**:
- no TTA
- TTA on last layer only
- TTA + early stop by validation heuristic

**Rủi ro**: compute tăng; cần guard để không degrade.

---

## 3) Ưu tiên roadmap (đề xuất để “vượt SOTA” nhanh nhất)

### Phase 1 (1–2 tuần): “Low‑risk, high‑gain”
1. **Temporal warm‑start + temporal pose smoothing nhẹ** (B)
2. **Uncertainty gating cho hand losses** (A)
3. **2D fusion thay overwrite cứng** (E)

Kỳ vọng: giảm jitter → giảm TR‑V2V trung bình đáng kể cho LHand/RHand, không hại UBody.

### Phase 2 (2–4 tuần): “Novel constraints”
4. **Hand‑arm coupling** (C)
5. **Contact‑aware constraints** (D)

Kỳ vọng: cải thiện cả hands lẫn UBody(-F) trong các sign có contact.

### Phase 3 (tuỳ chọn): “Domain adaptation”
6. **TTA / self‑training** (F)

---

## 4) Đề xuất thiết kế thí nghiệm & báo cáo (journal‑friendly)

### A) Bộ chỉ số
- Primary: TR‑V2V mm (UBody(-F), LHand, RHand) trên 57 signs (Table‑1).
- Secondary: per‑sign breakdown (script eval đã in), plus jitter statistics (temporal variance) nếu làm temporal.

### B) Ablation table (gợi ý format)
- Baseline: DexAvatar(HAMER)
- Baseline: DexAvatar(WiLoR)
- +A (uncertainty)
- +B (temporal)
- +C (kinematic coupling)
- +D (contact)
- +E (2D fusion)

### C) Error analysis định tính
- show frames/mesh ở các sign khó: occlusion, one‑hand, contact.
- plot per‑frame TR‑V2V time series để chứng minh temporal smoothness.

---

## 5) Notes gắn với codebase hiện tại

Bạn đã có:
- WiLoR exporter tạo `hamer.pkl` compatibility
- region indices files (hands, upper body minus face)
- evaluator TR‑V2V

Các điểm can thiệp cụ thể trong code:
- `dexavatar_fitting/smplifyx/data_parser.py`: nơi fuse/overwrite 2D hand + inject hand_pose/hand3D
- `dexavatar_fitting/smplifyx/fitting.py`: nơi thêm weight schedule / uncertainty / temporal loss (nếu sequence fitting)
- `dexavatar_fitting/smplifyx/main.py`: orchestration level (fit theo segment) + warm start

---

## 6) Checklist để “vượt SOTA” mà vẫn fair

1. **Giữ protocol Table‑1**: 57 signs + regions + TR‑V2V definition đúng.
2. **Report compute**: nếu thêm temporal/TTA, báo thời gian inference.
3. **Reproducibility**: lưu config + manifest indices + seed.
4. **Ablation rõ**: chứng minh gain đến từ method, không phải tuning ngẫu nhiên.

---

## 7) Next actions (thực dụng)

1. Chạy baseline numbers:
   - DexAvatar(HAMER) vs DexAvatar(WiLoR)
2. Thêm nhanh temporal warm‑start + smoothing (B) và uncertainty gating (A)
3. Chạy lại full 57 signs, so sánh improvement theo từng vùng

Nếu bạn muốn, mình có thể viết thêm một file `docs/experimental_plan.md` với:
- bảng ablation template
- danh sách config knobs
- lệnh chạy batch + naming convention cho output folders
