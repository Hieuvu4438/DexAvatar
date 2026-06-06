# DexAvatar — Phân Tích Chuyên Sâu & Research Gaps

## A. KIẾN TRÚC HIỆN TẠI

### Pipeline (5 Stage):

```
Video → [S1] Sapiens (133 keypoints) → [S2] SMPLer-X (initial SMPL-X) → [S3] HaMeR (3D hands) 
     → [S4] DexAvatar Fitting (SignBPoser + SignHPoser + SMPL-X optimization) → Output Mesh
```

**Core insight:** Đây là optimization-based method, KHÔNG phải regression/learning-based. Mỗi frame được fit độc lập qua LBFGS optimizer, dùng 2D keypoints làm observation và 3D priors làm regularization.

### Loss Function (7 terms):

| Term | Vai trò |
|---|---|
| `L_joint` | 2D reprojection error (Sapiens body + HaMeR hands) |
| `L_bprior` | SignBPoser latent regularization + supervision từ SMPLer-X |
| `L_hprior` | SignHPoser latent regularization + supervision từ HaMeR |
| `L_pen` | Interpenetration prevention (self-collision) |
| `L_temp` | Temporal smoothness (current vs previous frame) |
| `L_bbiomech` | Body biomechanical constraints (6 joints, 3 Euler angles) |
| `L_hbiomech` | Hand biomechanical constraints (15 joints, 3 Euler angles) |

### Priors Architecture:
- **SignBPoser**: VAE, encoder-decoder 3 linear layers, latentD=33, input=21 body joints × 9D rotation matrix, trained on SignAvatars (filtered How2Sign)
- **SignHPoser**: VAE, identical architecture, latentD=23, input=15 hand joints × 9D, trained on mocap (8 signers, 93 words, Vicon+Manus gloves)

---

## B. RESEARCH GAPS — PHÂN TÍCH CHI TIẾT

### Gap 1: `[CRITICAL]` Phụ thuộc hoàn toàn vào pseudo-ground truth

**Vấn đề:** DexAvatar dùng Sapiens, SMPLer-X, HaMeR làm initialization. Cả 3 đều là off-the-shelf models được train trên dữ liệu general-purpose (COCO, Human3.6M, etc.). Khi cả 3 cùng fail (motion blur nặng, self-occlusion lớn), DexAvatar không có cơ chế fallback. Đây là single-point-of-failure.

**Cơ hội:** Thay thế pipeline khởi tạo bằng **end-to-end learning** hoặc **diffusion-based initialization** được train trực tiếp trên sign language data. Có thể dùng transformer-based model nhận video frames làm input, output trực tiếp SMPL-X parameters, bỏ qua hoàn toàn Sapiens + SMPLer-X + HaMeR.

**Tiềm năng impact:** Rất cao — giảm độ phức tạp pipeline 4 stage → 1 stage, tăng robustness với edge cases.

---

### Gap 2: `[CRITICAL]` Temporal modeling quá yếu

**Vấn đề:** `L_temp = ψ(θ_b − θ_b^pre)` chỉ là L2 đơn giản giữa frame hiện tại và frame trước. Đây là **Markov assumption bậc 1** — không capture được:
- Long-range temporal dependencies (sign kéo dài 2-5 giây)
- Motion dynamics (vận tốc, gia tốc của tay)
- Transition patterns giữa các phonemes
- Co-articulation effects (bàn tay biến đổi dần giữa các signs)

**Cơ hội:** Thay thế `L_temp` bằng:
- **Temporal transformer** hoặc **LSTM/GRU** module học motion dynamics từ sequence
- **Motion VAE prior** (tương tự VPoser nhưng cho whole sequence thay vì single pose)
- **Score-based / Diffusion motion prior** cho trajectory optimization

**Tiềm năng impact:** Cao — motion jitter và inconsistency giữa các frame là vấn đề lớn trong sign language reconstruction.

---

### Gap 3: `[CRITICAL]` SignBPoser trained trên pseudo-GT từ 1 dataset

**Vấn đề:** SignBPoser dùng dữ liệu từ SignAvatars (derived from How2Sign) — bản thân đây là pseudo-ground truth từ OSX, đã có bias. Pipeline: How2Sign → OSX → SignAvatars → SignBPoser, qua 3 tầng gián tiếp. Thêm vào đó, chỉ filter bằng biomechanical constraints đơn giản (Euler angle ranges).

**Cơ hội:**
- Train SignBPoser trực tiếp từ **mocap data thật** (như đã làm với SignHPoser)
- Dùng **adversarial training** hoặc **cycle consistency** để học body pose manifold không cần paired data
- **Multi-dataset training**: kết hợp How2Sign + SGNify + custom mocap

**Tiềm năng impact:** Trung bình-cao — body pose ít phức tạp hơn hand pose trong signing, nhưng vẫn quan trọng.

---

### Gap 4: `[CRITICAL]` Thiếu contact modeling giữa 2 tay và tay-thân

**Vấn đề:** `L_pen` chỉ ngăn mesh penetration (hình học), không model **contact physics**:
- Khi 2 tay chạm vào nhau khi ký (fingerspelling, classifiers)
- Khi tay chạm vào thân/ngực (body-anchored signs)
- Không phân biệt được "chạm nhẹ" vs "đè mạnh"

**Cơ hội:**
- Thêm **contact loss**:鼓励/penalize specific hand-hand, hand-body contacts dựa trên sign class
- **Contact prediction network**: dự đoán contact map từ sign class hoặc video context
- **Physics-based refinement**: dùng differentiable physics simulator cho hand-hand interaction

**Tiềm năng impact:** Cao — contact là yếu tố phonological quan trọng trong sign language.

---

### Gap 5: `[MEDIUM]` Hand Decision Maker thô sơ

**Vấn đề:** Classifier phân biệt one-handed vs two-handed signs bằng cách đọc `signs.txt` (mapping cứng). Không handle được:
- Signs có cả 2 tay nhưng 1 tay dominant hơn (asymmetrical signs)
- Frame-level variation (có frame dùng 1 tay, frame khác dùng 2 tay)
- Non-manual signals (body shift, head tilt, facial expression) ảnh hưởng đến hand choice

**Cơ hội:**
- **Temporal hand activity detection**: dùng transformer học frame-level hand activation
- **Weakly-supervised learning**: từ video không có annotation hand choice
- **Multi-task learning**: joint predict sign class + hand choice + pose

---

### Gap 6: `[MEDIUM]` Non-manual features bị bỏ qua

**Vấn đề:** Facial expression, head pose, jaw pose, eye gaze được model hóa trong SMPL-X nhưng DexAvatar đặt weight rất thấp (`expr_weights: [0, 0.5e1, 0.5e1]`, jaw được set = 0 ở phase 1). Lí do: không có supervision tốt cho face trong sign language video.

**Cơ hội:**
- **Face-specific prior**: tương tự SignBPoser/SignHPoser nhưng cho facial expressions trong signing (grammatical markers, mouth morphemes)
- **Multi-modal supervision**: dùng text/gloss để constrain facial grammar
- **Lip reading integration**: kết hợp mouth shape với linguistic content

**Tiềm năng impact:** Trung bình — quan trọng cho linguistic completeness nhưng metrics hiện tại không đánh giá face.

---

### Gap 7: `[MEDIUM]` Dataset bottleneck

**Vấn đề:** SGNify chỉ có 57 signs (~2,872 frames evaluation). SignHPoser mocap: 8 signers, 93 words (fingerspelling). Quá nhỏ để:
- Train deep learning models end-to-end
- Cover diversity of signing styles (dialects, ages, proficiency levels)
- Generalize to continuous signing (hiện tại là isolated signs)

**Cơ hội:**
- **Synthetic data generation**: dùng SignBPoser + SignHPoser để generate synthetic 3D signing data
- **Domain adaptation**: adapt từ large-scale 2D sign datasets (WLASL, AUTSL, How2Sign) sang 3D
- **Self-supervised learning**: dùng cycle consistency giữa 2D→3D→2D

---

### Gap 8: `[MEDIUM]` Per-frame optimization chậm

**Vấn đề:** LBFGS optimization per-frame, 30 iterations, trên RTX 4090. Không thể real-time. Mỗi frame chạy độc lập → không tận dụng được shared context.

**Cơ hội:**
- **Amortized inference**: train network predict SMPL-X params trực tiếp, dùng DexAvatar làm refinement
- **Test-time optimization với warm start**: dùng prediction từ frame trước
- **Knowledge distillation**: train student network từ DexAvatar outputs

---

### Gap 9: `[LOW-MEDIUM]` Không có evaluation trên continuous signing

**Vấn đề:** Chỉ evaluate trên isolated signs (SGNify). Không test trên continuous signing videos có transitions, co-articulation, sentence-level context.

**Cơ hội:** Build benchmark cho continuous sign language 3D reconstruction. Dùng dataset như How2Sign (có continuous signing + 2D poses).

---

### Gap 10: `[LOW]` Lower body bị ignore hoàn toàn

**Vấn đề:** `joint_weights[:, 11:23] = 0` — lower body joints bị disable. Với upper-body-focused signing thì OK, nhưng một số ngôn ngữ ký hiệu dùng lower body (body shift, leg movement cho role shift).

**Cơ hội:** Thêm tùy chọn full-body reconstruction cho các ngôn ngữ ký hiệu có yếu tố chân.

---

## C. ĐỀ XUẤT HƯỚNG NGHIÊN CỨU MỚI

### Hướng 1 (Khả thi cao nhất): **End-to-End Sign Language 3D Reconstruction với Diffusion Prior**

**Ý tưởng:** Thay pipeline 4-stage bằng:
1. Video encoder (ViT/Swin) → feature sequence
2. Transformer decoder → SMPL-X parameter sequence
3. **Sign Diffusion Prior**: diffusion model train trên sign language motion data, dùng làm prior trong optimization hoặc sampling

**Ưu điểm:**
- Single model, end-to-end differentiable
- Diffusion prior mạnh hơn VAE prior (cover multi-modal distribution tốt hơn)
- Tận dụng được progress gần đây của diffusion models
- Có thể train với ít dữ liệu hơn nhờ pre-trained diffusion prior

**Novelty:** Diffusion-based sign language motion prior chưa có ai làm. Kết hợp diffusion + biomechanical constraints + contact modeling.

---

### Hướng 2 (Impact cao): **Contact-Aware Sign Language Reconstruction**

**Ý tưởng:** Focus vào contact modeling:
1. **ContactNet**: predict per-frame hand-hand, hand-body contact maps
2. **Contact-guided optimization**: dùng contact predictions làm additional constraints
3. **Physics-informed refinement**: differentiable contact dynamics

**Novelty:** Contact modeling cho sign language 3D reconstruction là unexplored area. Liên quan mật thiết đến phonological structure của sign.

---

### Hướng 3 (Dài hạn nhất): **Cross-Lingual 3D Sign Prior**

**Ý tưởng:** Train SignBPoser/SignHPoser trên **nhiều ngôn ngữ ký hiệu** (DGS, ASL, Auslan, etc.) để học **universal sign pose manifold**. Sau đó fine-tune hoặc zero-shot cho ngôn ngữ mới.

**Novelty:** Cross-lingual transfer trong 3D sign language processing chưa được khám phá.

---

### Hướng 4 (Practical nhất): **Boosting DexAvatar với Better Initialization & Temporal Model**

**Ý tưởng:** Giữ optimization framework nhưng cải tiến:
1. **Video-based initialization**: dùng temporal model (VideoMAE, TimeSFormer) predict SMPL-X params từ sequence
2. **Motion VAE prior**: thay `L_temp` đơn giản bằng learned motion prior
3. **Adaptive hand choice**: attention-based hand activation thay vì hard classifier

**Risk thấp nhất** vì build trên framework có sẵn, dễ compare với baseline DexAvatar.

---

## D. SO SÁNH CÁC HƯỚNG

| Tiêu chí | Hướng 1 (Diffusion) | Hướng 2 (Contact) | Hướng 3 (Cross-Lingual) | Hướng 4 (Boosting) |
|---|---|---|---|---|
| **Độ mới (novelty)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Khả thi** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Impact** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Dữ liệu cần** | Nhiều | Trung bình | Rất nhiều | Ít |
| **Thời gian** | 6-8 tháng | 4-6 tháng | 12+ tháng | 3-4 tháng |
| **Phù hợp target** | WACV/ICCV | CVPR/ECCV | TPAMI/IJCV | WACV/3DV |

**Khuyến nghị:** Nếu đây là dự án đầu tiên trong lĩnh vực này, nên bắt đầu với **Hướng 4** (lowest risk, nhanh ra kết quả) sau đó mở rộng sang **Hướng 1 hoặc 2** cho paper chất lượng cao hơn.

---

## E. LITERATURE REVIEW — TOÀN CẢNH SIGN LANGUAGE 3D RECONSTRUCTION

### E.1 Phân Loại Bài Toán Liên Quan

Bài toán 3D Sign Language Reconstruction nằm ở giao điểm của 4 lĩnh vực:

```
                    ┌─────────────────────────┐
                    │  3D Human Pose/Mesh      │
                    │  Estimation              │
                    │  (SMPL-X, MANO, FLAME)   │
                    └───────────┬─────────────┘
                                │
    ┌───────────────────────────┼───────────────────────────┐
    │                           │                           │
    ▼                           ▼                           ▼
┌───────────────┐   ┌───────────────────────┐   ┌──────────────────┐
│ Sign Language │   │  Sign Language 3D     │   │ Sign Language    │
│ Recognition   │   │  Reconstruction       │   │ Production (SLP)  │
│ (SLR)         │   │  (DexAvatar, SGNify)  │   │ (Neural Sign     │
│               │   │                       │   │  Actors, etc.)   │
└───────────────┘   └───────────────────────┘   └──────────────────┘
```

- **SLR (Recognition):** Video → label/gloss. Hiểu sign là gì.
- **3D Reconstruction:** Video → 3D mesh/pose. Khôi phục hình dạng 3D.
- **SLP (Production):** Text/gloss → 3D pose/video. Sinh ra sign từ ngôn ngữ.

DexAvatar nằm ở nhánh **3D Reconstruction**, nhưng có thể hưởng lợi từ SLP (dùng generative models làm prior) và SLR (dùng recognition features làm supervision).

---

### E.2 Các Paper Trực Tiếp Về 3D Sign Language Reconstruction

#### E.2.1 SGNify — CVPR 2023 ⭐ NỀN TẢNG
**Reconstructing Signing Avatars from Video Using Linguistic Priors**
- **Tác giả:** Maria-Paola Forte, Peter Kulits, Chun-Hao P. Huang, Vasileios Choutas, Dimitrios Tzionas, Katherine J. Kuchenbecker, Michael J. Black (MPI-IS)
- **Venue:** CVPR 2023
- **Tóm tắt:** Paper đầu tiên giải quyết bài toán 3D sign language reconstruction từ monocular video. Dùng optimization-based pipeline với linguistic priors. Tạo dataset SGNify (mocap ground truth từ Vicon+Manus gloves, 57 signs).
- **Link:** [CVPR 2023 Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.html)
- **Relation to DexAvatar:** Đây là paper tiền nhiệm trực tiếp. DexAvatar = SGNify + SignBPoser + SignHPoser + biomechanical constraints.

#### E.2.2 DexAvatar — WACV 2026 ⭐ HIỆN TẠI
**DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors**
- **Tác giả:** Kaustubh Kundu, Hrishav Bakul Barua, Lucy Robertson-Bell, Zhixi Cai, Kalin Stefanov (Monash/KAUST)
- **Venue:** WACV 2026
- **Tóm tắt:** Cải tiến SGNify bằng learned hand & body priors (SignBPoser, SignHPoser). 35.11% improvement over SGNify trên SGNify benchmark.
- **Link:** [arXiv 2512.21054](https://arxiv.org/abs/2512.21054)

#### E.2.3 SignAvatar — FG 2024
**Sign Language 3D Motion Reconstruction and Generation**
- **Tác giả:** Lu Dong, Lipisha Chaudhary, Fei Xu, Xiao Wang, Mason Lary, Ifeoma Nwogu (U Buffalo)
- **Venue:** IEEE FG 2024
- **Tóm tắt:** Transformer-based CVAE cho cả reconstruction và generation từ video/text/CLIP. Tạo ASL3DWord dataset với 3D joint rotations (body, hands, face).
- **Link:** [arXiv 2405.07974](https://arxiv.org/abs/2405.07974)
- **Relation to DexAvatar:** Combined reconstruction + generation. CVAE approach trái ngược với optimization approach của DexAvatar. Dataset ASL3DWord có thể dùng để bổ sung training data.

#### E.2.4 FESLAR — MSc Thesis 2025
**Frame-Efficient Sign Language Avatar Reconstruction**
- **Tác giả:** Rabea Ahmed (U Windsor)
- **Tóm tắt:** Build trên SGNify, chỉ xử lý các frame quan trọng về mặt ngôn ngữ, dùng FILM interpolation cho các frame còn lại. Giảm 86% computation.
- **Relation to DexAvatar:** Keyframe selection approach có thể áp dụng để tăng tốc DexAvatar.

#### E.2.5 Independent SLR with 3D Body, Hands, and Face — ICASSP 2021
- **Tác giả:** Agelos Kratimenos, Georgios Pavlakos, Petros Maragos (NTUA / UC Berkeley)
- **Venue:** ICASSP 2021
- **Tóm tắt:** Paper đầu tiên dùng SMPL-X cho Sign Language Recognition. Chứng minh 3D holistic (body+hands+face) > RGB/2D skeletons.
- **Link:** [arXiv 2012.05698](https://arxiv.org/abs/2012.05698)
- **Relation to DexAvatar:** Cung cấp bằng chứng nền tảng rằng body+hands+face reconstruction quan trọng cho sign language tasks.

---

### E.3 Các Dataset Lớn Về 3D Sign Language

#### E.3.1 SignAvatars — ECCV 2024 ⭐
**A Large-scale 3D Sign Language Holistic Motion Dataset and Benchmark**
- **Tác giả:** Zhengdi Yu, Shaoli Huang, Yongkang Cheng, Tolga Birdal (Imperial / Tencent AI Lab)
- **Venue:** ECCV 2024
- **Scale:** 70,000 videos, 153 signers, 8.34M frames
- **Annotations:** SMPL-X meshes, MANO hands, FLAME face, 2D/3D keypoints, HamNoSys, text, words
- **Link:** [Project Page](https://signavatars.github.io/)
- **Relation to DexAvatar:**
  - Đây LÀ dataset SignAvatars mà SignBPoser được train trên đó (filtered từ How2Sign qua OSX)
  - Có thể dùng để train SignBPoser/SignHPoser mạnh hơn với nhiều data hơn
  - Benchmark cho continuous signing mà DexAvatar hiện chưa evaluate

#### E.3.2 MC-TRISLAN — LREC 2022
**A Large 3D Motion Capture Sign Language Data-set**
- **Tác giả:** Jan Jedlicka, Zdenek Krnoul et al.
- **Venue:** LREC 2022 (SignLang Workshop)
- **Scale:** Professional motion capture, continuous signing
- **Link:** [ACL Anthology](https://aclanthology.org/2022.signlang-1.14/)
- **Relation to DexAvatar:** Mocap chất lượng cao (không phải pseudo-GT), có thể dùng làm evaluation benchmark.

#### E.3.3 How2Sign — CVPR 2021
**A Large-scale Multimodal Dataset for Continuous American Sign Language**
- **Tác giả:** Amanda Duarte et al.
- **Venue:** CVPR 2021
- **Scale:** 80+ hours video, speech, transcripts, depth
- **Link:** [Project Page](https://how2sign.github.io/)
- **Relation to DexAvatar:** Panoptic studio subset cung cấp 3D pose quality cao. Là nguồn dữ liệu cho SignAvatars → SignBPoser.

---

### E.4 Sign Language Production (SLP) — Generative Models Có Thể Dùng Làm Prior

#### E.4.1 Neural Sign Actors — CVPR 2024 ⭐
**A Diffusion Model for 3D Sign Language Production from Text**
- **Tác giả:** Vasileios Baltatzis, Rolandos Alexandros Potamias, Evangelos Ververas, Guanxiong Sun, Jiankang Deng, Stefanos Zafeiriou (Imperial College)
- **Venue:** CVPR 2024
- **Tóm tắt:** Diffusion model sinh 3D SMPL-X sequences từ text. Dùng anatomically-informed GNN trên SMPL-X skeleton. Paper đầu tiên đưa diffusion vào SLP 3D.
- **Link:** [arXiv 2312.02702](https://arxiv.org/abs/2312.02702)
- **Relation to DexAvatar:** 
  - SMPL-X GNN encoder có thể dùng làm structural prior cho DexAvatar
  - Diffusion model có thể inverse thành reconstruction prior (diffusion posterior sampling)
  - Cùng nhóm với WiLoR, Signs as Tokens, MaDiS (Imperial College/InsightFace group)

#### E.4.2 Signs as Tokens (SOKE) — ICCV 2025 ⭐
**A Retrieval-Enhanced Multilingual Sign Language Generator**
- **Tác giả:** Ronglai Zuo, Rolandos Alexandros Potamias, Evangelos Ververas, Jiankang Deng, Stefanos Zafeiriou (Imperial College)
- **Venue:** ICCV 2025
- **Tóm tắt:** Decoupled Tokenizer (DETO) với 3 VQ-VAEs riêng biệt cho upper body, left hand, right hand. Autoregressive multilingual generator dùng mBART. Hỗ trợ ASL + CSL.
- **Link:** [arXiv 2411.17799](https://arxiv.org/abs/2411.17799)
- **Relation to DexAvatar:**
  - Decoupled body/hand tokenization → có thể áp dụng để tách riêng body prior, left hand prior, right hand prior
  - SMPL-X annotation pipeline cho CSL-Daily có thể bổ sung training data
  - Cùng research group với WiLoR, MaDiS, Neural Sign Actors

#### E.4.3 MaDiS — arXiv 2026
**Taming Masked Diffusion Language Models for Sign Language Generation**
- **Tác giả:** Ronglai Zuo et al. (Imperial College)
- **Venue:** arXiv Jan 2026
- **Tóm tắt:** Masked-diffusion language model cho SLP. Tri-level cross-modal pretraining, mixture-of-parts embedding. 40% higher throughput.
- **Link:** [arXiv 2601.19577](https://arxiv.org/abs/2601.19577)
- **Relation to DexAvatar:** Mixture-of-parts embedding có thể cải tiến body part decomposition trong optimization.

#### E.4.4 SignSparK — arXiv 2026
**Efficient Multilingual Sign Language Production via Sparse Keyframe Learning**
- **Tác giả:** Jianhe Low, Alexandre Symeonidis-Herzig, Richard Bowden et al. (U Surrey)
- **Venue:** arXiv Mar 2026
- **Tóm tắt:** Conditional Flow Matching (CFM) trong SMPL-X + MANO space. FAST segmentation cho keyframe extraction. Hỗ trợ 4 sign languages. 3D Gaussian Splatting rendering.
- **Link:** [arXiv 2603.10446](https://arxiv.org/abs/2603.10446)
- **Relation to DexAvatar:**
  - CFM trong SMPL-X space → có thể dùng làm motion prior thay cho VAE
  - FAST keyframe extractor → chọn frame quan trọng để tối ưu

#### E.4.5 M3T — arXiv 2026 ⭐ QUAN TRỌNG NHẤT CHO DEXAVATAR
**Discrete Multi-Modal Motion Tokens for Sign Language Production**
- **Tác giả:** Alexandre Symeonidis-Herzig, Jianhe Low, Ozge Mercanoglu Sincan, Richard Bowden (U Surrey)
- **Venue:** arXiv Mar 2026
- **Tóm tắt:** Giới thiệu **SMPL-FX** = FLAME expression + SMPL-X body. Tokenization riêng biệt cho body, hands, face qua modality-specific FSQ-VAEs. Autoregressive transformer trên multi-modal motion vocabulary. 58.3% accuracy trên NMFs-CSL (signs chỉ khác nhau bởi non-manual features).
- **Link:** [arXiv 2603.23617](https://arxiv.org/abs/2603.23617)
- **Relation to DexAvatar:**
  - **SMPL-FX là hướng nâng cấp trực tiếp cho SMPL-X trong DexAvatar**
  - DexAvatar hiện bỏ qua facial expression (weight thấp, jaw=0 ở phase 1)
  - Non-manual features (mouth morphemes, eyebrow, gaze) critical cho sign language grammar
  - Đây là paper CÓ THỂ ĐÓNG GÓP TRỰC TIẾP NHẤT cho DexAvatar

#### E.4.6 SignAligner — arXiv 2025
**Harmonizing Complementary Pose Modalities for Coherent Sign Language Generation**
- **Tác giả:** Xu Wang, Shengeng Tang, Lechao Cheng, Feng Li, Shuo Wang, Richang Hong (HFUT)
- **Venue:** arXiv Jun 2025
- **Tóm tắt:** 3-stage pipeline: co-generation → online correction → video synthesis. Tạo PHOENIX14T+ dataset với Pose + HaMeR + SMPLer-X annotations.
- **Link:** [arXiv 2506.11621](https://arxiv.org/abs/2506.11621)
- **Relation to DexAvatar:**
  - PHOENIX14T+ dataset có HaMeR + SMPLer-X annotations → dùng được cho DexAvatar evaluation
  - Multi-modal alignment strategy → có thể dùng làm joint hand-body prior

#### E.4.7 Discrete to Continuous (Sign-D2C) — CVPR 2025
**Generating Smooth Transition Poses from Sign Language Observation**
- **Tác giả:** Shengeng Tang, Jiayi He, Lechao Cheng, Jingjing Wu, Dan Guo, Richang Hong (HFUT)
- **Venue:** CVPR 2025
- **Tóm tắt:** Conditional diffusion model synthesize smooth transition frames. Linear interpolation + diffusion refinement.
- **Link:** [arXiv 2411.16810](https://arxiv.org/abs/2411.16810)
- **Relation to DexAvatar:** Transition generation có thể dùng làm temporal post-processing hoặc temporal prior term.

#### E.4.8 SignLLM — ICCV 2025 Workshop
**Sign Languages Production Large Language Models**
- **Tác giả:** Sen Fang et al.
- **Venue:** ICCV 2025 Workshop (CV4A11y)
- **Tóm tắt:** Multilingual SLP (8 sign languages). Reinforcement learning với Priority Learning Channel. Output 3D skeletal keypoints (50 keypoints = 8 upper-body + 21 per hand).
- **Relation to DexAvatar:** Multilingual approach. 3D skeleton có thể dùng làm intermediate supervision.

#### E.4.9 Các SLP Papers Khác
| Paper | Venue | Nội dung chính |
|---|---|---|
| Progressive Transformers | ECCV 2020 | Transformer đầu tiên cho SLP, counter mechanism |
| Mixed SIGNals | ICCV 2021 | Motion primitives decomposition |
| Signing at Scale | CVPR 2022 | Co-articulation modeling |
| T2S-GPT | ACL 2024 | Dynamic VQ-VAE + GPT autoregressive |
| SignDiff | arXiv 2023 | Dual-condition diffusion (text + skeleton) |
| Latent Motion Transformer | WACV 2024 | Latent motion space learning |
| SignGen | ECCV 2024 | End-to-end text-to-video latent diffusion |

---

### E.5 Hand/Body Pose Estimation — Công Nghệ Nền

#### E.5.1 WiLoR — CVPR 2025
**End-to-end 3D Hand Localization and Reconstruction in-the-wild**
- **Tác giả:** Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, Stefanos Zafeiriou (Imperial College)
- **Venue:** CVPR 2025
- **Tóm tắt:** Real-time hand detector (2M+ in-the-wild images) + transformer-based 3D hand reconstruction. Robust to occlusion, diverse lighting, gloved hands.
- **Link:** [arXiv 2409.12259](https://arxiv.org/abs/2409.12259)
- **Relation to DexAvatar:**
  - Có thể **thay thế HaMeR** làm hand initialization tốt hơn
  - Robust trong in-the-wild conditions (sign language videos thường có motion blur, occlusion)
  - Cùng nhóm Imperial College/InsightFace

#### E.5.2 HaMeR — CVPR 2024
**Reconstructing Hands in 3D with Transformers**
- **Tác giả:** Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, Jitendra Malik (UC Berkeley / UMich / NYU)
- **Venue:** CVPR 2024
- **Tóm tắt:** Fully transformer-based hand mesh recovery. MANO parameters. Đang được dùng trong DexAvatar pipeline.
- **Link:** [arXiv 2312.05251](https://arxiv.org/abs/2312.05251)
- **Relation to DexAvatar:** Đang là stage 3 trong pipeline. Có thể được thay thế bởi WiLoR.

#### E.5.3 SMPLer-X — CVPR 2024
**Scaling Up Expressive Human Pose and Shape Estimation**
- **Tác giả:** Zhongang Cai et al. (Sensetime / NTU)
- **Venue:** CVPR 2024
- **Tóm tắt:** Large-scale pretraining cho SMPL-X estimation. Đang dùng trong DexAvatar.
- **Relation to DexAvatar:** Đang là stage 2 trong pipeline.

---

### E.6 VERIFICATION: ĐÁNH GIÁ 10 PAPER USER YÊU CẦU

| # | Paper | Venue | Hướng | Liên quan DexAvatar? | Đóng góp được? |
|---|-------|-------|-------|---------------------|----------------|
| 1 | **SignAvatars** | ECCV 2024 | Dataset | ✅ RẤT CAO — Là dataset train SignBPoser, có thể mở rộng training/evaluation | ✅ **HIGH** — Thêm data, benchmark continuous signing |
| 2 | **WiLoR** | CVPR 2025 | Hand Reconstr. | ✅ CAO — Hand initialization tốt hơn, robust hơn HaMeR | ✅ **HIGH** — Thay HaMeR = WiLoR, tăng robustness |
| 3 | **Discrete to Continuous** | CVPR 2025 | Generation | ⚠️ TRUNG BÌNH — Transition generation, không phải reconstruction | ⚠️ **MEDIUM** — Temporal post-processing |
| 4 | **SignViP** | NeurIPS 2025 | Video Generation | ❌ THẤP — Video generation, inverse direction | ❌ **LOW** — Khác hướng |
| 5 | **MaDiS** | arXiv 2026 | SLP | ⚠️ TRUNG BÌNH — Masked diffusion, mixture-of-parts | ⚠️ **MEDIUM** — Part embedding decomposition |
| 6 | **Signs as Tokens (SOKE)** | ICCV 2025 | SLP | ✅ CAO — Decoupled VQ-VAEs, SMPL-X annotations | ✅ **HIGH** — Decoupled body/hand priors |
| 7 | **SignAligner** | arXiv 2025 | SLP | ⚠️ TRUNG BÌNH — Dataset PHOENIX14T+ có HaMeR+SMPLer-X | ⚠️ **MEDIUM** — Evaluation data |
| 8 | **SignSparK** | arXiv 2026 | SLP | ⚠️ TRUNG BÌNH — CFM trong SMPL-X space, keyframe extraction | ⚠️ **MEDIUM** — Motion prior, speed-up |
| 9 | **M3T** | arXiv 2026 | SLP | ✅ RẤT CAO — SMPL-FX representation, non-manual features | ✅ **VERY HIGH** — Nâng cấp SMPL-X → SMPL-FX |
| 10 | **Neural Sign Actors** | CVPR 2024 | SLP | ⚠️ TRUNG BÌNH — Diffusion SLP, GNN prior | ⚠️ **MEDIUM** — GNN structural prior |

---

### E.7 PHÂN TÍCH RESEARCH CLUSTERS

Các paper thuộc về **3 research groups chính**:

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  🏛️ IMPERIAL COLLEGE / INSIGHTFACE GROUP                        │
│     (Zuo, Potamias, Deng, Zafeiriou)                             │
│                                                                  │
│     WiLoR ──► Signs as Tokens ──► MaDiS ──► Neural Sign Actors  │
│     (hands)     (VQ-VAE SLP)       (masked diff)  (diff SLP)     │
│                                                                  │
│     📌 Focus: Hand reconstruction + Text-to-3D Sign Generation   │
│     📌 Strengths: SMPL-X generation, robust hand detection       │
│     📌 Dùng được: WiLoR thay HaMeR, VQ-VAE priors               │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🏛️ MPI-IS / MPG GROUP (Tübingen)                               │
│     (Forte, Choutas, Tzionas, Black)                             │
│                                                                  │
│     SGNify ──► SignAvatars ──► DexAvatar (collab Monash/KAUST)  │
│                                                                  │
│     📌 Focus: 3D Reconstruction từ video                         │
│     📌 Strengths: SMPL-X optimization, mocap evaluation          │
│     📌 Core papers: SGNify (CVPR 2023), SignAvatars (ECCV 2024)  │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🏛️ SURREY GROUP                                                │
│     (Saunders, Camgoz, Bowden, Low, Symeonidis-Herzig)           │
│                                                                  │
│     Progressive Trans. → Mixed SIGNals → SignSparK → M3T         │
│     (ECCV 2020)         (ICCV 2021)      (2026)      (2026)      │
│                                                                  │
│     📌 Focus: SLP từ text/gloss, multilingual, representations   │
│     📌 Strengths: SMPL-FX, keyframe extraction, flow matching    │
│     📌 Dùng được: M3T-SMPL-FX upgrade, SignSparK motion prior    │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🏛️ HFUT / HEFEI GROUP                                          │
│     (Tang, He, Cheng, Guo, Hong)                                 │
│                                                                  │
│     Sign-D2C ──► SignAligner                                     │
│                                                                  │
│     📌 Focus: Chinese Sign Language, diffusion for transitions   │
│     📌 Strengths: PHOENIX14T+ dataset                            │
│     📌 Dùng được: Evaluation data, temporal smoothing            │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### E.8 ĐỀ XUẤT GHÉP PAPER ĐỂ CẢI TIẾN DEXAVATAR

#### Chiến lược A: Nâng cấp representation (Short-term, Impact: CAO)

```
SMPL-X (hiện tại) ──► SMPL-FX (từ M3T)
                         │
                         ├── FLAME expression (face)
                         ├── MANO hands (giữ nguyên)
                         └── SMPL-X body (giữ nguyên)
```

**Papers cần tích hợp:**
- **M3T** → SMPL-FX representation
- **WiLoR** → Hand initialization tốt hơn HaMeR

**Lợi ích:** Fix được Gap 6 (Non-manual features), tăng hand robustness

#### Chiến lược B: Học prior mạnh hơn (Medium-term, Impact: RẤT CAO)

```
SignBPoser (VAE, latentD=33)
    │
    ▼
Decoupled VQ-VAE priors (từ Signs as Tokens)
    │
    ├── Body codebook (riêng)
    ├── Left Hand codebook (riêng)
    └── Right Hand codebook (riêng)

SignHPoser (VAE, latentD=23)
    │
    ▼
Conditional Flow Matching prior (từ SignSparK)
    │
    └── Motion-level (không chỉ pose-level)
```

**Papers cần tích hợp:**
- **Signs as Tokens** → Decoupled VQ-VAE priors
- **SignSparK** → CFM motion prior
- **SignAvatars** → Training data (70K videos)

**Lợi ích:** Fix Gap 1, 2, 3, 4

#### Chiến lược C: End-to-end với Diffusion (Long-term, Impact: RẤT CAO)

```
Pipeline 5-stage (hiện tại)
    │
    ▼
End-to-End Diffusion-based Reconstruction
    │
    ├── Video Encoder (ViT/Swin)
    ├── Diffusion Prior (train trên SignAvatars)
    └── SMPL-FX Decoder
```

**Papers cần tích hợp:**
- **Neural Sign Actors** → Diffusion architecture template
- **SignAvatars** → Training data
- **M3T** → SMPL-FX output representation
- **Discrete to Continuous** → Temporal smoothness

**Lợi ích:** Fix toàn bộ Gap 1-9

---

### E.9 KEY TAKEAWAYS

1. **DexAvatar không đơn độc:** Có ~40 paper liên quan trong hệ sinh thái sign language processing (reconstruction + recognition + production)

2. **Phân biệt rõ research clusters:** 
   - Imperial College làm **generation** (text→3D)
   - MPI-IS/Monash làm **reconstruction** (video→3D)
   - Surrey làm **production** (text→pose)
   - HFUT làm **Chinese SL generation**

3. **Paper có thể đóng góp trực tiếp nhất:**
   - **M3T** (SMPL-FX representation) — nâng cấp output format
   - **WiLoR** (Hand reconstruction) — thay thế HaMeR
   - **Signs as Tokens** (Decoupled VQ-VAE) — học prior tốt hơn
   - **SignAvatars** (Dataset) — train với nhiều data hơn

4. **Paper không phù hợp để ghép:**
   - SignViP, SignGen: video generation (khác hướng hoàn toàn)
   - Các SLP paper thuần túy (Progressive Transformers, T2S-GPT...): chỉ liên quan gián tiếp qua motion priors

5. **Cơ hội novelty:** 
   - **SMPL-FX cho reconstruction** — chưa ai làm (M3T mới làm cho SLP)
   - **CFM motion prior cho sign reconstruction** — chưa ai làm
   - **Decoupled body/hand VQ-VAE priors** — chưa ai áp dụng cho reconstruction
