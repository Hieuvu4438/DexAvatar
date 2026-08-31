# DexAvatar — Step 2: Scientific Bottleneck Prioritization and Targeted Literature Review

**Audit date:** 2026-08-25 (Asia/Bangkok)  
**Scope lock:** static scientific audit only; the official SGNify evaluator is treated as locked infrastructure; no SGNify test ground truth was used; no score reported below is reproduced.  
**Evidence labels:** `[VERIFIED]` = directly supported by an inspected primary source; `[INFERENCE]` = evidence-based interpretation not established by the authors; `[UNRESOLVED]` = available sources do not decide the point.  
**Assessment labels:** all 0–5 viability scores and the final ranking are *expert assessments*, not measured results.

---

## 1. Executive conclusion

`[VERIFIED]` DexAvatar itself identifies fast motion, motion blur, hand–hand occlusion and hand–body occlusion as recurring sources of unreliable hand evidence; its released fitting path uses frame-wise HaMeR hand targets but propagates only the preceding **body** pose, not a temporal hand state ([DexAvatar PDF, pp. 3–7, Secs. 1, 3.1–3.4, Fig. 2, Eqs. 9–12](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html); repository commit `a0dfd427f60f5811aadb35c8657b3856d47f56b5`, `dexavatar_fitting/smplifyx/fitting.py:430–499,527–662`, `main.py:226–330`). The audited evaluator removes one centroid translation independently for UBody(-F), LHand and RHand, but preserves vertex errors due to finger articulation, wrist/root rotation, scale and shape. Thus temporal recovery that changes those quantities has direct metric leverage; trajectory-only/global-translation gains largely do not.

`[VERIFIED]` The strongest public ecosystem for this failure family is now materially broader than at DexAvatar publication: HandFlow estimates 16-frame MANO pose/translation conditioned on HaMeR tokens, 2D joints and confidence; HaPTIC adapts HaMeR over 8-frame clips; Dyn-HaMR tracks two hands and uses a hand-motion prior; HMP, Deformer and PAD-Hand supply complementary evidence about motion priors, temporal fusion and physical plausibility. However, these systems differ sharply in handedness, single- versus two-hand processing, missing-detection behavior, output frames and availability of training code. None is a verified drop-in SMPL-X solution.

`[VERIFIED]` A second, distinct bottleneck is kinematic inconsistency between whole-body and hand estimates. DexAvatar freezes the upstream SMPL-X body chain while fitting separate hand targets. Hand4Whole++ provides direct evidence that hand observations can improve wrist/upper-body-chain recovery and that hard-copying a hand estimator’s root rotation can degrade accuracy; its released final mesh nevertheless scatters aligned MANO vertices into SMPL-X vertex slots, so literal reuse would not preserve a parameter-regenerable unified SMPL-X result.

**PRIMARY BOTTLENECK — expert decision:** **temporal bimanual hand-state recovery under unreliable monocular evidence** (blur, self/hand–body occlusion, missing/low-confidence detections), including handedness and identity continuity. This has the strongest verified DexAvatar failure evidence, affects both hand metrics and hand/arm vertices in UBody(-F), and has the most viable public research substrate.

**BACKUP BOTTLENECK — expert decision:** **whole-body–hand kinematic inconsistency across the shoulder–elbow–wrist–finger chain** caused by independently estimated body and hand observations. It has high UBody(-F)/hand leverage and strong recent evidence, but stricter representation and parameter-consistency risks.

No claim is made that either choice will improve DexAvatar without controlled experiments.

---

## 2. Bottleneck classification

### 2.1 Complete disposition of dossier B01–B21

| Class | Dossier IDs | Bottlenecks | Rationale and evidence status |
|---|---:|---|---|
| **A. MODEL-ADDRESSABLE** | B01–B12, B19 | Motion blur; hand–hand occlusion; hand–body occlusion/contact ambiguity; noisy/missing keypoints; unreliable 2D keypoints; depth ambiguity; SMPLer-X initialization/target error; HaMeR target error; handedness/detection-order ambiguity; one-hand class/side error; temporal drift/stale predecessor; prior over-regularization; missing/low-confidence first active-hand observation | `[VERIFIED]` These arise in observations, estimators, priors, temporal state or optimization and can change recovered rotations/shape/vertices. Only this class enters the research ranking. |
| **B. EVALUATION ARTIFACT** | B15–B17 | Positional frame pairing; class-dependent masks/weighting; independent centroid alignment/translation blindness | `[VERIFIED]` These are evaluator/protocol properties, not reconstruction-model contributions. They are excluded from ranking and treated as immutable. |
| **C. GROUND-TRUTH LIMITATION** | B13–B14 | Potentially inaccurate/implausible SGNify hand GT; anatomical plausibility–vertex-error mismatch | `[VERIFIED]` DexAvatar supplementary reports implausible hand GT examples and metric/plausibility tension. These cannot be solved by exploiting or changing the official evaluator. |
| **D. REPRODUCIBILITY/ENGINEERING** | B18, B20, B21 | Saved-parameter/OBJ mismatch; paper–code anatomy-penalty mismatch; lexical re-sort after numeric ordering | `[VERIFIED]` These concern artifact identity, released implementation consistency or file ordering. They require reproducibility controls, not a scientific reconstruction direction. |

### 2.2 Per-bottleneck traceability

| Bottleneck | Dossier IDs | Root cause | DexAvatar stage | Affected vertices | Affected TR-V2V region | Evidence status |
|---|---:|---|---|---|---|---|
| Fast motion / motion blur | B01 | Image evidence is temporally smeared | RGB → Sapiens/HaMeR observations | Mostly fingers/wrist; potentially arms via 2D joints | LHand, RHand; UBody(-F) where included | `[VERIFIED]` Paper/supplementary failure discussion; exact per-frame frequency `[UNRESOLVED]`. |
| Hand–hand self-occlusion | B02 | One hand hides parts of the other | Detection, crop, 2D/3D hand estimation | Occluded fingers and wrist | Both hand regions; UBody(-F) | `[VERIFIED]` Paper figures/discussion; causal size on score `[UNRESOLVED]`. |
| Hand–body occlusion/contact ambiguity | B03 | Body appearance and depth obscure hand evidence | Observation and fitting | Hand plus nearby torso/arm | Hand regions and UBody(-F) | `[VERIFIED]` Paper; whether contact constraints would lower TR-V2V is `[UNRESOLVED]`. |
| Noisy or absent keypoints | B04, B05 | Low-confidence Sapiens joints or missed detections | 2D supervision | Body chain, wrist, fingers through fitted pose | All reported regions | `[VERIFIED]` Code thresholds/masks and paper motivation. |
| Monocular depth ambiguity | B06 | Several 3D states project similarly | Reprojection/fitting | Rotated limbs/fingers, body depth and scale | All regions after centering except pure translation | `[VERIFIED]` General ambiguity and DexAvatar use of priors/3D targets; precise dominant joints `[UNRESOLVED]`. |
| Whole-body initialization/target error | B07 | SMPLer-X pseudo-target is inaccurate and upstream variables are fixed | Initialization and loss target | Shoulder–elbow–wrist chain, body, shape | Mainly UBody(-F), indirectly hands | `[VERIFIED]` Fixed-variable code path; error prevalence `[UNRESOLVED]`. |
| Hand target error | B08 | Frame-wise HaMeR estimate is inaccurate under sign imagery | Hand supervision | Wrist/root and 15 finger joints | LHand/RHand, included hand vertices in UBody(-F) | `[VERIFIED]` HaMeR is target source and paper reports hard cases; target-specific error magnitude `[UNRESOLVED]`. |
| Handedness / detection-order ambiguity | B09 | Multi-detection association does not guarantee temporal identity | HaMeR target assembly | Entire wrong/swapped hand | Both hand regions; UBody(-F) | `[VERIFIED]` Released ordering behavior; observed swap rate `[UNRESOLVED]`. |
| One-hand class/active-side error | B10 | Static class label plus wrist-speed heuristic can choose wrong active hand | Sign decision/masking | Chosen hand and excluded hand | Class-0 RHand and UBody(-F); LHand not evaluated | `[VERIFIED]` Code; empirical error rate `[UNRESOLVED]`. |
| Temporal drift / stale predecessor | B11 | Serial fitting passes previous body state; missing predecessor can be stale; no hand state is propagated | Fitting sequence | Body/arm; hands only indirectly | Mainly UBody(-F); possible hand effect through wrist chain | `[VERIFIED]` Code path; score contribution `[UNRESOLVED]`. |
| Prior over-regularization | B12 | Pose manifold or fixed target may suppress rare sign articulations | Latent optimization/priors | Body and fingers | All regions | `[VERIFIED]` Objective contains strong target/prior terms; over-regularization as actual test cause is `[INFERENCE]`. |
| Missing first active-hand evidence | B19 | First usable crop/keypoint for a side is absent/weak, so no reliable temporal anchor exists | Detection → target initialization | Entire active hand | Corresponding hand; UBody(-F) | `[VERIFIED]` Missing-detection behavior exists; first-frame impact is `[INFERENCE]`. |
| GT anatomical error | B13 | Mocap/registration GT can be implausible at hands | Benchmark target | Hand vertices | LHand/RHand | `[VERIFIED]` Supplementary examples; per-sign incidence `[UNRESOLVED]`. |
| Plausibility–TR-V2V mismatch | B14 | Euclidean GT distance does not encode anatomy/contact | Evaluation interpretation | Mostly hands | LHand/RHand | `[VERIFIED]` Reported ablation/examples; no license to alter evaluator. |
| Positional pairing | B15 | Sorted predicted/GT lists are zipped rather than joined by frame ID | Evaluator | Potentially all vertices | All regions | `[VERIFIED]` Evaluator; prohibited as research target. |
| Class-dependent weighting | B16 | Class-0 excludes LHand and removes left-hand vertices from UBody | Evaluator | Left hand | LHand and UBody(-F) | `[VERIFIED]`; locked. |
| Region-wise translation blindness | B17 | Each region is independently centroid-aligned | Evaluator | Relative offsets | All three regions | `[VERIFIED]`; locked. |
| Parameter/OBJ identity risk | B18 | Exported mesh may not be regenerable from saved parameters | Export/reproducibility | All | All | `[VERIFIED]` dossier code audit; scientific effect `[UNRESOLVED]`. |
| Anatomy mismatch paper/code | B20 | Hand-biomechanics term claimed in Eq. 12 is absent from active fitting total; body biomechanics remains | Objective/release | Hands/body | All | `[VERIFIED]`; classified as release-consistency issue. |
| Lexical re-sort | B21 | Numeric frame order is later replaced by lexical filename sort | Evaluator/file handling | Potentially all | All | `[VERIFIED]`; locked engineering artifact. |

---

## 3. Consolidated model-addressable bottlenecks

| Consolidated bottleneck | Dossier IDs | Observation → failure → mesh error → official TR-V2V | Metric-aware assessment |
|---|---:|---|---|
| **M1. Degraded and ambiguous hand evidence** | B01–B06, B08, B19 | Blur/occlusion/missing or low-confidence evidence → unstable 2D joints/HaMeR pose and unresolved depth → wrong wrist/finger rotations, hand shape/scale → non-rigid residual remains after per-hand centering | `[VERIFIED]` Direct LHand/RHand leverage; hand vertices also contribute to UBody(-F), except the evaluator’s class-0 left-hand exclusion. Pure crop/global translation gains do not help. |
| **M2. Temporal hand identity and state discontinuity** | B09–B11, B19 | Independent detections and no fitted hand state across frames → side swaps, jumps, failure to bridge missing observations → articulation/root-rotation errors over many vertex-frames | `[VERIFIED]` DexAvatar lacks hand temporal loss/state propagation. `[INFERENCE]` Vertex-frame weighting makes persistent failures more consequential than isolated ones, but size is unmeasured. |
| **M3. Inconsistent whole-body and hand kinematics** | B07, B08 (and part of B06) | SMPLer-X body target and HaMeR hand target are obtained independently while upstream body/global variables are fixed → shoulder/elbow/wrist/finger chain disagrees → limb/wrist rotations and hand articulation remain wrong | `[VERIFIED]` Affects UBody(-F) and hand metrics; relative hand-to-body translation is weakened by independent centering, but wrist/root rotation and deformations remain measurable. |
| **M4. Prior/regularization bias** | B12 | Generic prior/strong pseudo-target → rare sign pose is pulled toward a common manifold → anatomically smoother but lexically wrong rotations → vertex error can rise | `[INFERENCE]` Plausibility alone is not sufficient for official TR-V2V; DexAvatar’s own biomechanical ablations show that a plausible constraint can help one region and hurt another. |

**Causal grouping used for ranking.** M1 and M2 are treated as one primary scientific family because the useful temporal signal is specifically needed when a hand observation is unreliable. M3 remains separable: it concerns cross-chain consistency even for a sharp, confidently detected frame. M4 is not selected because evidence that it is the dominant DexAvatar error is weaker and SGNify’s hand GT can disagree with anatomical plausibility.

---

## 4. Literature-search protocol

### 4.1 Search date, sources and queries

Searches were performed through 2026-08-25. Technical claims were accepted only from primary sources: CVF/OpenAccess or publisher pages, arXiv manuscripts, author project pages and official repositories. Repository status means the exact public state inspected on 2026-08-25, not a historical promise.

Representative query families were:

- `temporal monocular hand mesh recovery MANO video occlusion confidence 2023 2024 2025 2026`;
- `two hand reconstruction video interacting hands MANO temporal tracking`;
- `motion blur hand mesh reconstruction MANO`;
- `whole body SMPL-X hand conditioned wrist pose recovery Hand4Whole`;
- `temporal SMPL-X reconstruction expressive whole body video`;
- `hand motion prior MANO latent sequence reconstruction`;
- `physics-aware diffusion hand motion recovery MANO`;
- `contact aware hand body reconstruction SMPL-X`;
- `uncertainty confidence aware 3D hand pose mesh`;
- `VLM correct SMPL-X MANO pose parameters`;
- exact-title follow-ups for paper, supplementary, project, GitHub, checkpoint and license.

Sources searched: CVF Open Access, arXiv, ACM/publisher/project pages where present, GitHub official organizations/users, and repository files/commit history. Google-like web search was used only for discovery; it is not cited as evidence.

### 4.2 Inclusion and exclusion

**Included** candidates had a direct path to at least one model-addressable bottleneck and produced or optimized SMPL-X/MANO parameters, joints or a compatible hand mesh. A serious candidate required enough main-method detail to identify its observation, state representation and temporal behavior. Public feasibility required a real repository, usable inference path and checkpoint—not merely a project page.

**Excluded** were sign recognition/generation, text-to-motion, action recognition, rendering-only avatars, NeRF/Gaussian reconstruction without pose recovery, generic mesh/image editing, 2D-only methods without a defined 3D path, and alternative topology with no SMPL-X/MANO-compatible output. Evaluator exploits, test-GT adaptation and MANO-mesh stitching were excluded by scope.

### 4.3 Search yield

`[VERIFIED]` The screened pool contained **30 unique records** after title/project deduplication. **15** were retained in the verified landscape and **6** received deep paper–repository audit. The other **15** were excluded or used only as negative controls. Representative exclusions were HaWoR (its main advantage is world-coordinate hand trajectory), Hand4DGS and HandAvatar (rendering/appearance rather than pose recovery), UniPose (language-conditioned human pose editing without verified MANO/SMPL-X hand correction), TokenHMR/WHAM (body-centric or translation-centric), EasyHOI/UniHOPE/LatentHOI/HOIGPT (object-centric objectives), FoundHand (not a parameter-regenerable MANO/SMPL-X recovery path), and multiple sign-recognition or sign-generation systems. `[INFERENCE]` Some excluded works may remain useful scientific context, but they cannot outrank candidates on the user’s feasibility and official-metric criteria.

---

## 5. Verified paper landscape

The table is intentionally capped at the 15 closest papers. “Training needed” answers whether the released checkpoint can be assessed without first retraining; it does **not** authorize or request an experiment. “Compatible” means representationally related, not plug-and-play equivalence between MANO and the SMPL-X hand chain.

| Paper, authors, primary URLs | Venue/year | Problem; input → output | Temporal? | SMPL-X/MANO compatibility | Public code / checkpoint / training needed | Target bottleneck | Technical and repository evidence |
|---|---|---|---|---|---|---|---|
| **[HandFlow: Fully Generative 4D Hand Recovery with Flow Matching](https://arxiv.org/abs/2607.11221)** — Mingxi Xu, Bowen Duan, Yi Gu, Zhengyang Shen, Renjing Xu, Yutao Yue. [Official repository](https://github.com/mxxu00/HandFlow) | arXiv 2026; manuscript metadata mentions TOG, publisher status `[UNRESOLVED]` | Monocular single-hand crop sequence + HaMeR tokens + 21 keypoints/confidence → MANO shape, 48-D axis-angle pose and translation over 16 frames | Yes; whole-window | Direct MANO intermediate; left/right and wrist-frame conversion still required | `[VERIFIED]` Inference, checkpoint, MIT code license; training/preprocessing absent. No retraining for released inference; exact retraining unavailable | M1/M2 | Paper Sec. 3 and appendix; repo commit `67fa7df536db233408fe6270ca5d2de28d5959c3`, `README.md:76–102,154,179–185`, `utils/online_hamer.py:190–255`. |
| **[StableHand: Quality-Aware Flow Matching for World-Space Dual-Hand Motion Estimation from Egocentric Video](https://arxiv.org/abs/2605.18553)** — Huajian Zeng, Chaohua Yao, Yuantai Zhang, Jiaqi Yang, Rolandos A. Potamias, Xingxing Zuo. [Official repository](https://github.com/huajian-zeng/stablehand) | arXiv 2026, “in submission” | Egocentric video + per-frame WiLoR observations/learned quality → dual-hand wrist trajectory and finger articulation | Yes; bimanual | MANO-family hand state `[VERIFIED]`; SMPL-X chain mapping `[UNRESOLVED]` | `[VERIFIED]` Placeholder only: one commit, assets/README, “code after acceptance”; no checkpoint. Training required but unavailable | M1/M2 | Paper Secs. 3–4; repository README lines 162–175 as accessed 2026-08-25. Frontier evidence only, not a feasibility leader. |
| **[HaPTIC: Predicting 4D Hand Trajectory from Monocular Videos](https://arxiv.org/abs/2501.08329)** — Yufei Ye, Yao Feng, Omid Taheri, Haiwen Feng, Shubham Tulsiani, Michael J. Black. [Official repository](https://github.com/JudyYe/haptic) | 3DV 2026 | Monocular video clips → per-frame MANO pose/shape plus camera/global trajectory | Yes; 8-frame clips | Direct MANO intermediate; sides processed as separate sequences | `[VERIFIED]` Training, demo, preprocessed labels and checkpoint public; raw datasets and MANO licensed separately. No retraining for inference | M1/M2, partly B06 | Paper Sec. 3 and appendix; repo commit `f9362c1bdf2c1ea2bfa695be2d4e6f362371e7df`, `README.md:171–237`, `demo.py:65–125`, configs `MODEL.NUM_FRAMES=8`. |
| **[Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator (Hand4Whole++)](https://openaccess.thecvf.com/content/CVPR2026/html/Moon_Enhancing_Hands_in_3D_Whole-Body_Pose_Estimation_with_Conditional_Hands_CVPR_2026_paper.html)** — Gyeongsik Moon. [Official repository](https://github.com/mks0601/Hand4Whole-plus-plus) | CVPR 2026 | Single RGB image → whole-body SMPL-X parameters plus an aligned, vertex-scattered MANO/SMPL-X visualization mesh | No | Whole-body SMPL-X backbone; released final vertices are not guaranteed to be regenerated by its saved SMPL-X parameters | `[VERIFIED]` Full training/inference/checkpoint, MIT code; licensed SMPL-X/MANO assets required | M3 | Paper Secs. 3.2–3.4, Fig. 2, Table 2; repo commit `f81d35ddd2b74206c40142243eb62b6d64ce0d65`, `main/model.py:58–93,200–222`, `common/nets/wilor.py:79–124`. |
| **[PAD-Hand: Physics-Aware Diffusion for Hand Motion Recovery](https://openaccess.thecvf.com/content/CVPR2026/html/Ismayilzada_PAD-Hand_Physics-Aware_Diffusion_for_Hand_Motion_Recovery_CVPR_2026_paper.html)** — Elkhan Ismayilzada, Yufei Zhang, Zijun Cui. [Official repository](https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026) | CVPR 2026 Highlight | Video → refined MANO pose sequence and per-joint/time physics variance | Yes; 16-frame windows | MANO intermediate; output uses ZXY Euler in released demo and must not be confused with SMPL-X local axis-angle | `[VERIFIED]` Demo/checkpoint public; no training/evaluation code and no root license found. No retraining for demo, exact retraining unavailable | M1/M4 | Paper Sec. 3, Tables 4–5 and supplementary; repo commit `ca9ed97bc671199c25cf569d3b1de0e6f7937251`, `demo.py:28–37,79–139`, `wilor_inference.py:67–120`. |
| **[Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html)** — Zhengdi Yu, Stefanos Zafeiriou, Tolga Birdal. [Official repository](https://github.com/ZhengdiYu/Dyn-HaMR) | CVPR 2025 Highlight | Monocular video → two separately parameterized MANO hands, global orientation/translation and world trajectories | Yes; true two-hand tracking/optimization | MANO intermediate; unified SMPL-X body/wrist-frame integration not provided | `[VERIFIED]` Code, checkpoint and MIT license public; MANO asset license required. Motion prior optional and disabled in inspected default config | M1/M2/B06 | Paper Sec. 3 and supplementary; repo commit `fa9cd7412c205fd15ee4139c8caacf79bf6167e6`, `README.md:218–239`, `config/hamr_config.yaml:1–30`. |
| **[WiLoR: End-to-end 3D Hand Localization and Reconstruction in-the-wild](https://openaccess.thecvf.com/content/CVPR2025/html/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.html)** — Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, Stefanos Zafeiriou. [Official repository](https://github.com/rolpotamias/WiLoR) | CVPR 2025 | Single RGB image → detections, MANO matrices/shape/camera and meshes for multiple hands | No temporal model | MANO; model is right-hand canonical. Left crops are flipped; raw left rotation matrices are not directly SMPL-X-local rotations | `[VERIFIED]` Code/models/data public; code license and non-commercial/no-derivatives model terms plus MANO license apply | M1/B08/B09 | Paper Sec. 3 and supplementary; repo commit `fcb911312a38fa8badd30d9656a167485d61b8f9`, `README.md:50–57,86–87`, `demo.py:80–107`, `wilor/datasets/vitdet_dataset.py:61–81`. |
| **[HMP: Hand Motion Priors for Pose and Shape Estimation from Video](https://openaccess.thecvf.com/content/WACV2024/html/Duran_HMP_Hand_Motion_Priors_for_Pose_and_Shape_Estimation_From_WACV_2024_paper.html)** — Enes Duran, Muhammed Kocabas, Vasileios Choutas, Zicong Fan, Michael J. Black. [Official repository](https://github.com/enesduran/HMP) | WACV 2024 | Monocular video observations → optimized MANO motion/shape using an AMASS-trained prior | Yes; optimization | Direct MANO intermediate; single-hand/right-canonical assumptions require audit | `[VERIFIED]` Code, trained prior and training instructions public; MANO/AMASS licenses apply | M1/M2/M4 | Paper Sec. 3 and supplementary; repo commit `35d799f76b2b2bc1d1e945117b021014b099e7e6`, `README.md:30–66`. |
| **[Deformer: Dynamic Fusion Transformer for Robust Hand Pose Estimation](https://openaccess.thecvf.com/content/ICCV2023/html/Fu_Deformer_Dynamic_Fusion_Transformer_for_Robust_Hand_Pose_Estimation_ICCV_2023_paper.html)** — Qichen Fu, Xingyu Liu, Ran Xu, Juan Carlos Niebles, Kris M. Kitani. [Official repository](https://github.com/fuqichen1998/Deformer) | ICCV 2023 | Short hand video → MANO-compatible mesh/pose using neighboring-frame deformation and dynamic fusion | Yes; reported `T=7` | MANO/778-vertex hand output; no unified SMPL-X mapping | `[VERIFIED]` Training/evaluation code and Apache-2.0 license; README documents resume path but no official checkpoint link was found | M1/M2 | Paper Sec. 3 and supplementary; repo commit `64dbedbff2417b5fa2881e72705a6eeb1f88b514`, `README.md:25–62`. |
| **[Recovering 3D Hand Mesh Sequence from a Single Blurry Image: A New Dataset and Temporal Unfolding](https://arxiv.org/abs/2303.15417)** — Yeounguk Oh, JoonKyu Park, Jaeha Kim, Gyeongsik Moon, Kyoung Mu Lee. [Official repository](https://github.com/JaehaKim97/BlurHand_RELEASE) | CVPR 2023 | One blurry image → past/current/future MANO mesh sequence represented by temporal unfolding | Temporal states inferred from one image; not video aggregation | MANO | `[VERIFIED]` Code repository public; checkpoint availability in current README `[UNRESOLVED]`; training data BlurHand released under its terms | B01 specifically | Paper Sec. 3; repo inspected at commit `19864229065f7c52238155df933da1fc0e95f1e9`, README title/citation. |
| **[HandOS: 3D Hand Reconstruction in One Stage](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_HandOS_3D_Hand_Reconstruction_in_One_Stage_CVPR_2025_paper.html)** — Xingyu Chen, Zhuheng Song, Xiaoke Jiang, Yaoqing Hu, Junzhi Yu, Lei Zhang. Claimed [repository URL](https://github.com/idea-research/HandOS) | CVPR 2025 | Single image → joint hand detection, handedness-free 2D/3D keypoints, vertices and camera | No | MANO/hand-mesh compatible; exact SMPL-X convention path absent | `[VERIFIED]` Paper exists; claimed repository returned 404 on 2026-08-25. No code/checkpoint verified | B09 and M1 | Paper Sec. 3; repository status is direct access result, not a claim of nonexistence elsewhere. |
| **[ACR: Attention Collaboration-Based Regressor for Arbitrary Two-Hand Reconstruction](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_ACR_Attention_Collaboration-Based_Regressor_for_Arbitrary_Two-Hand_Reconstruction_CVPR_2023_paper.html)** — Zhengdi Yu, Shaoli Huang, Chen Fang, Toby P. Breckon, Jue Wang. [Official repository](https://github.com/ZhengdiYu/Arbitrary-Hands-3D-Reconstruction) | CVPR 2023 | Single image → one or two MANO hands with cross-hand/part attention | No | MANO; no body-chain output | `[VERIFIED]` Code, checkpoint path and MIT license public; MANO license required | B02/B09 | Paper Sec. 3 and supplementary; repository README. Useful two-hand evidence but no temporal identity. |
| **[MaskHand: Generative Masked Modeling for Robust Hand Mesh Reconstruction in the Wild](https://openaccess.thecvf.com/content/ICCV2025/html/Saleem_MaskHand_Generative_Masked_Modeling_for_Robust_Hand_Mesh_Reconstruction_in_ICCV_2025_paper.html)** — Muhammad Usama Saleem, Ekkasit Pinyoanuntapong, Mayur J. Patel, Hongfei Xue, Ahmed Helmy, Srijan Das, Pu Wang. [Project page](https://m-usamasaleem.github.io/publication/MaskHand/MaskHand.html) | ICCV 2025 | Single image + 2D pose/image context → confidence-sampled VQ-MANO pose tokens/mesh | No | Direct VQ-MANO output; token-to-SMPL-X conventions require mapping | `[UNRESOLVED]` No official code repository/checkpoint was found from the paper/project page by access date | M1/B06 | Paper Sec. 3 and supplementary. Strong uncertainty framing; weak public feasibility. |
| **[MeMaHand: Exploiting Mesh-Mano Interaction for Single Image Two-Hand Reconstruction](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_MeMaHand_Exploiting_Mesh-Mano_Interaction_for_Single_Image_Two-Hand_Reconstruction_CVPR_2023_paper.html)** — Congyi Wang, Feida Zhu, Shilei Wen. Paper page; official repo `NOT FOUND` | CVPR 2023 | Single image → two hand meshes and MANO parameters with inter-/intra-hand attention | No | Direct two-hand MANO plus non-parametric mesh; unified SMPL-X must use the parameter branch, not replace topology | `[UNRESOLVED]` No official public repo/checkpoint verified | B02/M1 | Paper Sec. 3 and supplementary. Relevant technical evidence; low reproducibility. |
| **[TempCLR: Reconstructing Hands via Time-Coherent Contrastive Learning](https://arxiv.org/abs/2209.00489)** — Andrea Ziani, Zicong Fan, Muhammed Kocabas, Sammy Christen, Otmar Hilliges. [Official repository](https://github.com/eth-ait/tempclr) | 3DV 2022 | Labeled images + unlabeled videos for representation learning → frame-wise MANO/hand reconstruction | Temporal at training; standard frame inference | MANO-based ExPose hand branch | `[VERIFIED]` Training/inference code and pretrained model public; licensed MANO/data required | M1/M2 | Paper Sec. 3; repository README explicitly exposes training and inference. Older foundational candidate. |

### 5.1 Landscape-level findings

- `[VERIFIED]` Only StableHand and Dyn-HaMR among the retained set explicitly formulate **two-hand temporal** recovery. StableHand’s code is unavailable; Dyn-HaMR’s two hands remain separate MANO chains.
- `[VERIFIED]` HandFlow, HaPTIC, HMP, Deformer and PAD-Hand are temporally relevant but their released inference paths process one side or one hand state at a time. Running both sides does not by itself create cross-hand reasoning or identity association.
- `[VERIFIED]` Hand4Whole++ is the only retained work with direct recent evidence that hand-specific evidence can modify the SMPL-X whole-body chain. Its released visualization mesh and saved parameter vector are not equivalent representations.
- `[VERIFIED]` WiLoR and HaMeR-style systems use a right-hand canonical model. Correct left-hand use requires more than visually mirroring vertices: crop parity, pose parity, pose mean, joint order and local/global wrist frames matter.
- `[INFERENCE]` SignAvatars can plausibly supply sign-domain SMPL-X/hand pseudo-targets, and unlabeled How2Sign/Neural Sign Actors video can plausibly support domain adaptation or temporal observation learning; the retained papers do not verify that such adaptation preserves official SGNify TR-V2V. Generic hand datasets remain auxiliary rather than sign-domain evidence.

---

## 6. Deep audit of the six strongest candidates

### 6.1 HandFlow — confidence-aware temporal MANO recovery

**What is verified.** The paper represents a 16-frame state with one 10-D MANO shape vector, per-frame 48-D axis-angle pose (global/root plus 15 finger joints) and 3-D translation. Conditioning combines frozen HaMeR image tokens with 21 2D joints and confidence; continuous masking substitutes unreliable conditioning and a full-window flow model reasons non-autoregressively across time (paper Sec. 3, Eqs. 1–9 and appendix). The released checkpoint and video inference are real, but the README explicitly withholds training, preprocessing and evaluation code (`README.md:154,179–185`). The online inference selects one target side and mirrors a left-side video into the right-hand canonical input (`utils/online_hamer.py:190–255`). Therefore it is not a bimanual tracker.

**DexAvatar and metric connection.** `[VERIFIED]` It reuses HaMeR appearance features, which overlap DexAvatar’s per-frame hand observation. The new information is temporal context, confidence-conditioned masking and a learned motion distribution—not an independent camera view or independent detector. `[INFERENCE]` This can target articulation/root-rotation failures that survive hand centering; its large paper gains in world-space translation are not directly transferable evidence for TR-V2V. Shape consistency can affect centered vertices, although whether a sequence-constant MANO shape maps cleanly to DexAvatar’s fixed SMPL-X shape is `[UNRESOLVED]`.

**Feasibility boundary.** `[VERIFIED]` Checkpoint inference is public under an MIT repository license. Exact retraining is not reproducible from the release, and paper training uses right-hand-centric DexYCB/HOT3D processing. Left/right parity, MANO-to-SMPL-X joint ordering and root/local wrist conversion are not supplied for DexAvatar. A literal replacement would also risk double-counting the same HaMeR failure.

**Falsifier.** The hypothesis is weakened if, after removing region centroids, HandFlow’s gains concentrate in translation/acceleration while finger/root-rotation residuals on low-confidence sign frames do not improve; it is also weakened by side-conversion errors or loss of fast sign articulation.

### 6.2 HaPTIC — multi-frame HaMeR adaptation

**What is verified.** HaPTIC repurposes HaMeR for eight-frame clips using cross-view self-attention and global cross-attention, predicting MANO pose/shape and a coherent trajectory (paper Sec. 3, Fig. 2). Its official repository provides a checkpoint, demo, training entry point and preprocessed labels; raw H2O, DexYCB, ARCTIC, HO3D and InterHand images remain separately obtained (`README.md:217–237`). The demo uses overlapping clips, saves `hand_pose`, `global_orient` and `betas`, and the data path separates left/right sequences. A side can be omitted if fewer than half of its frames are valid (`nnutils/det_utils.py`, sequence-validity branch); left crops are mirrored into canonical space.

**DexAvatar and metric connection.** `[VERIFIED]` The visual backbone/observation family is again HaMeR, so observation dependence and double-counting are real. `[VERIFIED]` The paper’s strongest contribution is global trajectory, while local pose remains comparable to the image estimator and can be slightly worse in some reported settings. Since official TR-V2V separately centers the hand region, trajectory improvement alone has low leverage. `[INFERENCE]` Multi-frame attention can still improve visible articulation under short occlusions, but the paper does not establish this on sign language or SGNify.

**Feasibility boundary.** `[VERIFIED]` This is more retrainable than HandFlow. It does not jointly reason over two hands or the SMPL-X arm chain, and its skip behavior under long missing detections matters directly for B19.

**Falsifier.** The hypothesis that HaPTIC is useful for DexAvatar is refuted if its centered hand-vertex/root-rotation error is no better than frame-wise HaMeR, even when trajectory and reprojection improve.

### 6.3 Hand4Whole++ — hand-conditioned whole-body kinematics

**What is verified.** Hand4Whole++ combines a frozen SMPLer-X-L32 whole-body estimator with frozen WiLoR features. CHAM modulates the whole-body feature stream so that the SMPL-X branch predicts wrist orientation coherently with the upper body; separate hand articulation/shape are rigidly aligned from a canonical hand using wrist and MCP anchors (paper Secs. 3.2–3.4). Table 2 shows that directly copying the hand estimator’s root rotation performs worse than conditional modulation, while CHAM improves hand and whole-body measures. This is direct evidence against hard-overwriting a SMPL-X wrist with a MANO root.

**Representation audit.** `[VERIFIED]` The released `main/model.py:58–93` rigidly aligns the hand mesh and scatters MANO vertices into indexed SMPL-X vertex slots. It also exports SMPL-X parameter fields (`main/model.py:200–222`), but those parameters do not encode the scattered final vertex replacement. Thus the displayed/exported final mesh is not guaranteed to be regenerated from a single SMPL-X parameter vector. The left-hand WiLoR wrapper explicitly flips crops and applies axis-angle parity/pose-mean corrections (`common/nets/wilor.py:79–124`); this is valuable convention evidence but not proof of compatibility with DexAvatar’s SMPL-X layer version.

**DexAvatar and metric connection.** `[VERIFIED]` DexAvatar already uses a SMPLer-X whole-body estimate and HaMeR hand targets, but it does **not** feed hand evidence back into an optimizable shoulder–elbow–wrist body chain. Hand4Whole++ therefore supplies cross-source kinematic information, not merely a hand-backbone replacement. Wrist/root and upper-limb rotations survive regional centering and can affect UBody(-F), LHand and RHand. `[INFERENCE]` Its per-image design cannot solve temporal swaps or missing detections.

**Feasibility boundary and falsifier.** Code, weights and training are public under MIT, with licensed body assets. The strict unified-SMPL-X constraint rules out treating the vertex-scattered mesh as the final representation. The bottleneck hypothesis is weakened if improvements disappear when evaluation uses vertices regenerated solely from coherent SMPL-X parameters, or if sign-domain hand features fail to change wrist/upper-chain error.

### 6.4 Dyn-HaMR — true two-hand tracking and optimization

**What is verified.** Dyn-HaMR reconstructs two interacting hands from monocular video with separate per-hand MANO local pose (15×3 axis-angle), global wrist orientation, shape and translation; it uses camera-motion estimation, multi-stage optimization and an interacting-hand/HMP-based infilling prior (paper Sec. 3, Figs. 2–3). The repository provides code, checkpoints and an MIT license. The inspected default config sets the motion-prior path off, while the README documents an explicit option and 128-frame chunks (`README.md:218–239`; `config/hamr_config.yaml`). It is the strongest retained *public* evidence that temporal hand identity and two-hand occlusion can be treated jointly.

**DexAvatar and metric connection.** `[VERIFIED]` DexAvatar has neither temporal hand identity nor interacting-hand motion optimization. Dyn-HaMR adds cross-hand and temporal information. However, much of Dyn-HaMR’s scientific emphasis is world trajectory and dynamic-camera recovery; independent region centering removes pure translation and weakens relative hand placement. Its local articulation, global wrist rotation and shape still affect TR-V2V. `[UNRESOLVED]` The fraction of its gains attributable to these surviving terms is not reported under an SGNify-like metric.

**Feasibility boundary.** The output remains two separate MANO chains with no body. The repository does not define MANO-to-SMPL-X local wrist composition, and optional-prior/default behavior must not be conflated. SLAM/camera components add integration risk with little direct metric leverage.

**Falsifier.** If centered per-hand articulation/rotation is unchanged relative to its frame initializer, or if identity gains occur only in world-space association without changing paired SGNify frames, its official-TR-V2V hypothesis fails.

### 6.5 PAD-Hand — physics-aware single-hand motion refinement

**What is verified.** PAD-Hand formulates 16-frame MANO motion refinement through conditional diffusion, with a MeshCNN–Transformer and Euler–Lagrange residuals treated as virtual observations; it also estimates per-joint/per-time physics variance (paper Sec. 3). The release includes a demo and checkpoint but not training/evaluation code or a root license. The demo advances by non-overlapping 16-frame windows, requires complete window observations, converts output using ZXY Euler angles and processes one selected hand. Its WiLoR wrapper chooses a right hand if present, otherwise the first detection (`wilor_inference.py:67–120`). It is therefore neither a robust missing-detection sequence path nor a bimanual model.

**DexAvatar and metric connection.** `[VERIFIED]` DexAvatar already includes body biomechanics but its active released objective lacks the paper-claimed hand-biomechanics fitting term. PAD adds temporal physics uncertainty, but it depends on another image estimator and mainly changes MANO motion. Paper ablations show that the data-driven refinement accounts for most accuracy gain, whereas the physics component adds much smaller metric changes. `[INFERENCE]` Physics can improve plausibility yet increase SGNify vertex error where GT is anatomically imperfect, reproducing the B14 tension.

**Feasibility boundary and falsifier.** Checkpoint inference is assessable; exact retraining and license permissions are not. Its released Euler convention and root orientation cannot be inserted as an SMPL-X wrist overwrite. The candidate is falsified for this task if gains vanish after centering, complete-window coverage is low, or physics improves plausibility/variance without reducing parameter-regenerated hand vertices.

### 6.6 WiLoR — high-fidelity per-frame hand observation

**What is verified.** WiLoR jointly detects multiple hands and predicts MANO rotation matrices, shape and weak-perspective camera from one image. It has no temporal component; the authors’ smooth-video demonstrations remain frame-wise inference (paper Sec. 3 and CVPR page). Public code, models and data are available, subject to repository/model terms and the MANO license.

**Handedness audit.** `[VERIFIED]` The release stores only `MANO_RIGHT`; left image crops are flipped into right canonical space (`wilor/datasets/vitdet_dataset.py:61–81`). The public demo mirrors left vertices/joints/camera for rendering but does not expose a fully verified, SMPL-X-local left rotation conversion (`demo.py:80–107`). Hand4Whole++’s wrapper performs additional axis-angle parity and pose-mean treatment. This proves that “both output MANO” is insufficient to assume parameter compatibility.

**DexAvatar and metric connection.** WiLoR could change finger articulation/root/shape, all visible after centering, but it adds no temporal identity and no body-chain information. It would primarily replace HaMeR with another estimator trained from closely related hand data—a high risk of being engineering substitution rather than a new scientific contribution. Two detections in one image do not imply interacting-hand reasoning.

**Falsifier.** The replacement hypothesis fails if sign-domain centered hand error or low-confidence coverage is not better than HaMeR, or if left-hand parity/detection-order errors offset any per-frame gain.

### 6.7 Mandatory critical-question matrix

| Critical question | HandFlow | HaPTIC | Hand4Whole++ | Dyn-HaMR | PAD-Hand | WiLoR |
|---|---|---|---|---|---|---|
| 1. DexAvatar already has something similar? | HaMeR observation yes; hand temporal model no | HaMeR yes; multi-frame backbone no | Separate body/hand targets yes; hand-conditioned body chain no | Frame hands yes; temporal bimanual tracking no | Generic priors/biomechanics yes; temporal physics diffusion no | Frame hand estimator yes; this is mainly an alternative |
| 2. New information or estimator replacement? | Temporal confidence + motion distribution; not a new view | Multi-frame image context; same estimator family | Hand features condition the body chain | Temporal, cross-hand and camera/trajectory state | Temporal physical prior/variance; upstream WiLoR unchanged | Mostly estimator/detector replacement |
| 3. Same observation dependency? | Frozen HaMeR tokens and keypoints overlap strongly | HaMeR ancestry overlaps strongly | SMPLer-X family overlaps body; WiLoR is a separate hand source | Depends on per-frame hand initializers plus video/SLAM | Depends on WiLoR sequence | RGB crop/detector only; independent model but same pixels |
| 4. Double-counting risk? | High if treated as independent supervision | High | Moderate: two pretrained streams, explicitly coupled | Moderate | High relative to WiLoR target | N/A as replacement; high if naively added beside HaMeR |
| 5. Body chain, wrist or fingers? | Root/wrist + 15 finger joints; no body chain | Root/wrist + fingers + trajectory; no body chain | Wrist and upper chain plus fingers/shape | Global wrist + fingers; no body | Root/hand generalized coordinates; no body chain | Wrist/root + fingers/shape; no body |
| 6. Truly two-hand? | No; single selected side | No; sides are separate tracks | Two hand crops in one frame, but not temporal interacting-hand reconstruction | Yes, two interacting hands | No; one selected hand | Multiple frame detections, no cross-hand reconstruction model |
| 7. Temporal identity/handedness tracking? | Side locked for a single sequence; no cross-hand ID | Separate side sequences; no joint identity model | No temporal tracking | Yes, explicit temporal two-hand tracks | No robust identity; complete-window single hand | No temporal component |
| 8. Exact output | MANO β10, θ48 axis-angle, translation | MANO hand pose/global orient/betas and trajectory | SMPL-X params plus vertex-scattered aligned MANO final mesh | Per-hand MANO local AA, global wrist AA, β, translation | MANO motion; released demo uses ZXY Euler for rotations | MANO rotation matrices, β, camera, vertices/joints |
| 9. Could centered TR-V2V decrease? | `[INFERENCE]` Yes through articulation/root/shape; translation gain removed | `[INFERENCE]` Only if local pose improves; trajectory alone no | `[INFERENCE]` Yes through wrist/upper chain/fingers | `[INFERENCE]` Yes through local pose/rotation, not pure world trajectory | `[INFERENCE]` Possible through pose; plausibility alone insufficient | `[INFERENCE]` Possible per frame; no persistence benefit |
| 10. Non-public retraining dependency? | Exact retraining unavailable | Training code/labels public; raw datasets licensed | Training public; body/hand assets licensed | Inference public; prior/data assets licensed | Training unavailable | Inference/checkpoints public; original large training data/model terms apply |
| 11. Novel research gap or replacement? | Sign-specific temporal uncertainty remains a gap | Adaptation alone risks incremental work | Sign-specific cross-chain consistency is a clear gap | Bimanual sign adaptation is a gap, but literal port is broad/complex | Physics-only sign refinement has uncertain novelty/metric value | Replacement alone is weak novelty |
| 12. Evidence that would refute benefit | No centered local-pose gain on unreliable frames | Local pose equal/worse after centering | No gain from parameter-regenerated SMPL-X or sign wrists | Gains only in world trajectory/association | Plausibility improves but vertex error/coverage does not | No sign-domain centered gain or left-side instability |

---

## 7. Viability ranking

### 7.1 Expert-assessment scorecard (0–5)

These scores are qualitative expert judgments, **not measurements and not an additive decision rule**. For criteria 1–8, 5 is most favorable. For **implementation risk**, 5 means highest risk.

Abbreviations: **BE** = DexAvatar bottleneck evidence; **TL** = official TR-V2V leverage; **TD** = technical directness; **CM** = SMPL-X/MANO compatibility; **PR** = public reproducibility; **DF** = data feasibility; **IA** = integration isolation; **NP** = research novelty potential; **RK** = implementation risk.

| Candidate / technique | BE | TL | TD | CM | PR | DF | IA | NP | RK | Assessment rationale |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| HandFlow-style confidence-aware temporal MANO recovery | 5 | 4 | 5 | 3 | 3 | 2 | 4 | 4 | 4 | Directly targets unreliable observations and local hand state. Checkpoint/inference exist, but training and exact left/SMPL-X conversion do not. |
| Hand4Whole++-style hand-conditioned whole-body kinematics | 4 | 5 | 5 | 3 | 5 | 4 | 3 | 4 | 4 | Direct wrist/upper-chain evidence and complete release; strict parameter-regeneration constraint invalidates literal vertex scatter and raises isolation risk. |
| HaPTIC multi-frame HaMeR adaptation | 5 | 3 | 4 | 3 | 4 | 3 | 4 | 3 | 4 | Strong public release and temporal observation path, but headline trajectory gains are largely removed by centering and sides are independent. |
| Dyn-HaMR interacting two-hand temporal optimization | 5 | 3 | 4 | 2 | 4 | 3 | 2 | 4 | 5 | True two-hand/occlusion reasoning; separate MANO chains, camera/SLAM scope and uncertain centered local-pose gain make integration broad and risky. |
| HMP latent hand-motion prior | 5 | 4 | 4 | 3 | 4 | 3 | 3 | 2 | 4 | Public optimization prior with occlusion evidence; DexAvatar already has learned pose priors, so novelty depends on the temporal sign-specific question, not prior replacement. |
| Deformer temporal dynamic fusion | 5 | 4 | 4 | 2 | 2 | 3 | 4 | 3 | 4 | Direct blur/occlusion evidence and local mesh leverage; no verified official checkpoint and non-unified hand mesh pathway reduce feasibility. |
| PAD-Hand physics-aware refinement | 4 | 3 | 3 | 3 | 2 | 2 | 4 | 3 | 4 | Checkpoint/demo exist, but complete-window single-hand behavior, Euler convention, no training release and plausibility–GT mismatch constrain value. |
| StableHand quality-aware bimanual recovery | 5 | 4 | 5 | 3 | 0 | 1 | 3 | 5 | 5 | Scientifically closest frontier to M1/M2, but repository is an explicit placeholder; not a public feasible basis at access date. |
| WiLoR per-frame estimator replacement | 4 | 4 | 3 | 2 | 5 | 4 | 4 | 1 | 3 | Excellent artifacts and possible per-frame hand gain, but no temporal/cross-body information and substantial left-canonicalization risk; low research novelty alone. |

### 7.2 Final ranking

| Rank | Candidate/technique | Target bottleneck | TR-V2V linkage | Compatibility | Public artifacts | Main risk | Confidence |
|---:|---|---|---|---|---|---|---|
| **1** | Confidence-aware temporal hand-state recovery, evidenced most directly by HandFlow | M1+M2 | Finger articulation, root/wrist rotation and shape survive hand centering; affects LHand/RHand and included hand vertices in UBody(-F) | MANO intermediate only; conversion unresolved | Inference + checkpoint; no training | Same HaMeR observation, one-side canonicalization, unavailable retraining | `[INFERENCE]` **medium-high** |
| **2** | Hand-conditioned whole-body kinematic correction, evidenced by Hand4Whole++ | M3 | Wrist/upper-chain/finger errors survive centering; potential leverage across all three regions | SMPL-X branch exists, but released final mesh is vertex-scattered | Full code, checkpoint, training, MIT | Parameter/mesh nonequivalence; single-frame design | `[INFERENCE]` **medium-high** |
| **3** | Multi-frame HaMeR adaptation (HaPTIC) | M1+M2 | Only local pose/root/shape gains matter; trajectory does not | MANO intermediate; separate sides | Full training/inference/checkpoint; no root license found | Paper emphasizes trajectory; long missing tracks may be skipped | `[INFERENCE]` **medium** |
| **4** | Interacting two-hand tracking/optimization (Dyn-HaMR) | M1+M2 | Local rotations/articulation matter; world translation largely removed | Two separate MANO hands, no body chain | Code/checkpoint/MIT; optional prior | High scope and convention risk; uncertain default/prior path | `[INFERENCE]` **medium** |
| **5** | Latent hand-motion prior (HMP) | M1+M2+M4 | Can alter local pose through occlusion, which survives centering | MANO intermediate | Prior/checkpoints/training public | Potential sign articulation bias; limited novelty as prior substitution | `[INFERENCE]` **medium** |
| **6** | Dynamic neighboring-frame fusion (Deformer) | M1+M2 | Direct local hand-vertex effect | MANO mesh family, no body | Train/eval code and license; no verified checkpoint | Retraining burden and representation path | `[INFERENCE]` **medium-low** |
| **7** | Physics-aware hand-motion refinement (PAD-Hand) | M4 and part of M1 | Pose changes can matter; plausibility/variance alone cannot | MANO; released ZXY Euler conversion | Demo/checkpoint only | Missing windows, one hand, no training/license, GT mismatch | `[INFERENCE]` **medium-low** |
| **8** | Quality-aware dual-hand recovery (StableHand) | M1+M2 | Articulation channels have leverage; translation channel does not | MANO-family | Placeholder only | No code/checkpoint/training | `[VERIFIED]` artifact status; viability **low** |
| **9** | WiLoR-only estimator replacement | B08/B09 | Per-frame articulation/root/shape can matter | Right-canonical MANO; left conversion nontrivial | Strongest complete per-frame release | No temporal identity/new information; engineering-only contribution risk | `[INFERENCE]` **medium** |

**Why this is not a mechanical sum.** Hand4Whole++ has the strongest release and nominal metric leverage, but its literal output violates the parameter-regenerable unified-SMPL-X constraint and it does not address the most frequently evidenced temporal/visibility failures. HandFlow ranks first despite incomplete training release because the *bottleneck family*—not a literal package—is the decision target, and several independent public systems support the same family. StableHand scores high scientifically yet ranks low operationally because the user requires a public, feasible basis. WiLoR scores high on public artifacts but low on novelty/directness: replacing HaMeR does not itself explain how unreliable temporal evidence is recovered.

---

## 8. Primary bottleneck decision

### 8.1 Exact problem statement

**PRIMARY BOTTLENECK:** **Temporal bimanual hand-state recovery under unreliable monocular evidence, with correct handedness and temporal identity.**

The problem is that fast sign motion, blur, hand–hand/hand–body occlusion, missing or low-confidence detections and monocular ambiguity can corrupt the frame-wise hand observation. DexAvatar does not maintain or optimize a temporal state for either hand and does not jointly associate the two hands across time. The resulting errors of interest are wrong 15-joint finger articulation, wrong local/global wrist orientation, inconsistent hand shape/scale and persistent side/identity mistakes—not pure global translation.

### 8.2 DexAvatar evidence

- `[VERIFIED]` Paper/supplementary identifies blur, fast motion, self/hand-body occlusion and unreliable keypoints as practical difficulties (PDF pp. 1–7, Secs. 1 and 3; qualitative/failure figures in supplementary pp. 15–21).
- `[VERIFIED]` Released fitting uses HaMeR-derived per-frame hand targets; HaMeR 3D-hand loss is configured with weight zero, and hand latents are optimized toward those targets (`fit_smplx_vposer_x.yaml:54–119`; `fitting.py:527–662`).
- `[VERIFIED]` The only temporal term/state uses the preceding **body pose** (`fitting.py:430–499`; `main.py:226–330`; `fit_single_frame.py:728–730`). No hand temporal state or hand temporal loss appears in the active total.
- `[VERIFIED]` Multi-detection order and one-hand heuristic leave handedness/side failure paths (`data_parser.py:176–203`). Their empirical frequency on the 57 signs is `[UNRESOLVED]`.

### 8.3 Affected official metric regions

- **LHand:** direct effect on all included left-hand vertex-frames except class-0 signs, for which the evaluator skips LHand entirely.
- **RHand:** direct effect on right-hand articulation/root/shape on all evaluated classes.
- **UBody(-F):** hand vertices contribute for two-handed signs; class-0 removes left-hand vertices. Arm/wrist vertices can also change if the recovered state influences the unified chain, but such influence is not automatic.
- `[VERIFIED]` Per-region centering removes one translation vector only. It does not remove finger articulation, root rotation, scale or shape. `[INFERENCE]` Temporal continuity has metric value only insofar as it improves these states, rather than merely smoothing a wrong pose.

### 8.4 Why prioritized over the alternatives

1. **Evidence strength:** M1/M2 pools nine directly related dossier IDs and contains the paper’s own dominant visual failure cases. M3 is real but its empirical dominance is not established; M4 is still weaker and confounded by imperfect hand GT.
2. **Metric leverage:** local hand rotations and deformation remain after centering and affect the evaluator’s vertex-frame samples across L/R hands. Translation-only gains are explicitly discounted; the exact evaluated sample count remains unresolved because the SGNify files are absent.
3. **Public research substrate:** HandFlow supplies checkpointed confidence-aware sequence inference; HaPTIC supplies full multi-frame HaMeR training/inference; Dyn-HaMR supplies public two-hand tracking/optimization; HMP supplies a public hand-motion prior; Deformer supplies training code for blur/occlusion-aware temporal fusion. No single package satisfies all constraints, but the underlying scientific question is supported from several independent directions.
4. **Isolation:** the contribution can in principle be tested at the hand-observation/temporal-state boundary while keeping evaluator, SMPL-X topology and test protocol fixed. This is an `[INFERENCE]` about experimental separability, not an implementation plan.
5. **Novelty potential:** `[INFERENCE]` Existing public temporal hand methods are chiefly generic/egocentric, single-hand or world-trajectory-oriented; sign-specific high-speed bimanual articulation under a unified body model remains insufficiently addressed. Merely swapping a per-frame estimator would not meet this bar.

### 8.5 Technical building-block evidence, without method formulation

- HandFlow demonstrates that confidence-aware full-window MANO-state recovery can condition on corrupted/missing observations.
- HaPTIC demonstrates that a strong image hand transformer can be adapted to short temporal clips with public training code.
- Dyn-HaMR demonstrates explicit temporal identities for two interacting hands and motion-prior infilling.
- HMP demonstrates latent optimization with a hand-motion prior under occlusion.
- Deformer demonstrates direct temporal fusion targeted at blur/occlusion and fingertip-heavy errors.
- BlurHand demonstrates that motion-blur formation carries information about temporal hand states, but its single-blurry-image formulation is not itself the selected direction.

These are verified capabilities of separate papers; they are **not** proposed here as a combined architecture.

### 8.6 What remains unknown and what could falsify the decision

`[UNRESOLVED]` There is no released per-frame error decomposition of DexAvatar on SGNify by blur, occlusion, detector confidence, handedness swap or sign speed. `[UNRESOLVED]` It is unknown whether temporal candidate gains persist after SMPL-X parameter conversion and independent centering. `[UNRESOLVED]` There is no verified sign-domain comparison of HaMeR, WiLoR, HandFlow or HaPTIC using the locked metric.

The primary choice should be rejected later if a diagnostic shows that (a) low-confidence/occluded frames are not materially worse after centering; (b) most DexAvatar hand error comes from a stable MANO–SMPL-X convention/shape bias rather than time-varying observations; (c) temporal systems improve only world translation or acceleration, not centered articulation/root/shape; or (d) handedness/identity errors are negligible and single-frame targets already dominate the achievable error floor.

---

## 9. Backup bottleneck decision

### 9.1 Exact problem statement

**BACKUP BOTTLENECK:** **Whole-body–hand kinematic inconsistency across the shoulder–elbow–wrist–finger chain caused by independently estimated body and hand observations.**

### 9.2 Evidence and metric regions

`[VERIFIED]` DexAvatar initializes/supervises the body from SMPLer-X and the hands from HaMeR, then freezes global orientation, body shape, expression, translation/camera and other upstream variables while optimizing body and hand pose latents. A wrong body target can therefore constrain wrist placement/orientation even when the hand crop predicts plausible fingers. `[VERIFIED]` Hand4Whole++ independently establishes that hand features can improve wrist/upper-chain whole-body recovery and that direct copying of the hand root is inferior to a body-aware wrist estimate (CVPR 2026 paper Secs. 3.2–3.4, Table 2).

Wrist/upper-limb rotations change UBody(-F) after centering; wrist/root and finger rotations change LHand/RHand. Pure attachment translation between a hand and body is weakened because hands are centered independently, so this bottleneck is prioritized for rotational/kinematic errors rather than absolute hand placement.

### 9.3 Why backup, not primary

The official-metric leverage is high and the Hand4Whole++ release is unusually complete. However: (i) DexAvatar’s per-sign incidence of body–hand disagreement is not measured; (ii) Hand4Whole++ is single-frame and therefore does not address the best-verified blur/occlusion/identity family; and (iii) its released final mesh uses aligned MANO vertex scatter, which is incompatible with the requirement that the final mesh be regenerated from one SMPL-X parameterization. The scientific bottleneck remains viable; the literal released output path does not.

### 9.4 Evidence base, unknowns and falsifier

Technical evidence comes primarily from Hand4Whole++; WiLoR supplies the hand-specific features/convention audit, while DexAvatar supplies the fixed-chain failure opportunity. `[UNRESOLVED]` It is unknown how often SGNify’s error resides in shoulder/elbow/wrist versus fingers, whether a coherent SMPL-X parameter vector retains Hand4Whole++’s reported gains, and whether sign-domain crops provide enough wrist evidence under blur.

Reject the backup choice if parameter-regenerated SMPL-X evaluation shows no wrist/upper-chain gain, if DexAvatar’s body-chain errors are already small relative to finger articulation, or if improvements occur only in relative hand attachment translation removed by region centering.

---

## 10. Rejected directions

| Rejected direction | Decision | Specific reason |
|---|---|---|
| Positional frame pairing, missing-frame shift, class masks, unused `--central`, hard-coded paths, aggregation/alignment changes | **Excluded** | `[VERIFIED]` B15–B17/B21 are locked evaluation or engineering artifacts. Changing them is not a reconstruction contribution and is prohibited by scope. |
| Optimizing only global camera/world translation or choosing SLAM/world-trajectory recovery as the primary contribution | **Rejected as primary** | `[VERIFIED]` One centroid translation is removed independently for every evaluated region/frame. Only associated changes to rotation, articulation, scale or shape could help. |
| Face reconstruction | **Rejected** | `[VERIFIED]` UBody(-F) removes face vertices; no official face metric is reported. |
| Contact/collision/biomechanics alone | **Rejected as primary** | `[VERIFIED]` DexAvatar already has collision/body biomechanics; released hand-biomechanics behavior contradicts the paper. `[VERIFIED]` Plausibility and SGNify vertex GT can disagree. There is no evidence that contact-only refinement is the dominant centered-error source. |
| WiLoR-only or HandOS-only per-frame replacement | **Rejected as scientific direction** | WiLoR has strong public artifacts but adds no temporal identity or body-chain information; HandOS’s claimed repo was inaccessible. A new backbone alone has weak novelty and unverified sign-domain advantage. |
| Literal Hand4Whole++ final vertex scatter | **Rejected** | `[VERIFIED]` The final displayed mesh is modified by scattering aligned MANO vertices and is not guaranteed to be regenerated from its saved SMPL-X parameters. This violates the final-representation constraint. Its scientific cross-chain evidence remains relevant. |
| Directly overwriting SMPL-X wrist/root with MANO global orientation | **Rejected** | `[VERIFIED]` Hand4Whole++ reports worse performance for direct root copying; MANO and SMPL-X local/global frames, pose means and parity differ. |
| Stitching two MANO meshes into a body or averaging estimator vertices/rotation matrices | **Rejected** | Explicitly prohibited; it would break unified SMPL-X parameterization and lacks a valid rotation-manifold interpretation. |
| PAD-Hand as turnkey temporal solution | **Rejected** | `[VERIFIED]` Released demo is one-hand, complete-window, non-overlapping, Euler-converted and missing training/license artifacts. Physics gains need not lower imperfect-GT vertex error. |
| StableHand as primary public basis | **Rejected at this date** | `[VERIFIED]` It is a scientifically direct bimanual candidate but its official repository is a one-commit placeholder promising future code; no checkpoint or inference path exists. |
| VLM visual critique/correction | **Rejected** | `[VERIFIED]` The screened primary literature, including UniPose, did not establish a VLM that directly and verifiably corrects SMPL-X/MANO hand parameters for this task. Language critique or planning is not a parameter correction signal. |
| Alternative mesh topology, generic mesh generation, NeRF/Gaussian avatar rendering | **Excluded** | No parameter-regenerable unified SMPL-X output and/or no pose-recovery contribution. |
| Sign recognition, text-to-sign/text-to-motion generation, action recognition | **Excluded** | These do not recover the observed signer’s frame-aligned SMPL-X mesh. |
| Training, tuning, calibration, early stopping or checkpoint selection on SGNify test GT | **Excluded** | Violates the test-only evaluation constraint and invalidates scientific conclusions. |
| Optimizing the one-hand heuristic as the main direction | **Not selected** | `[VERIFIED]` B10 is real, but its failure frequency is unknown and official class-0 masking narrows leverage: LHand is skipped and UBody removes left-hand vertices. It remains a diagnostic variable, not the leading bottleneck. |
| Generic prior replacement without sign-domain evidence | **Not selected** | DexAvatar already uses learned body/hand pose priors. Replacing a prior without isolating temporal unreliable-observation recovery risks over-regularizing rare sign articulation and offers limited novelty. |

No rejected item is being proposed as an alternate method.

---

## 11. Unresolved questions for the next prompt

Only questions that must be resolved before method formulation are listed.

1. **Where is the centered error concentrated?** Required evidence: a non-test/development-set decomposition by fingers, palm, wrist, forearm and upper arm, using the locked vertex masks and alignment. This decides whether M1/M2 or M3 is truly dominant.
2. **How strongly do blur, occlusion and confidence predict error?** Required evidence: frame-quality annotations or detector-confidence logs paired with development-set centered errors; no SGNify test GT may be used for tuning.
3. **How frequent are side swaps, duplicate associations and one-hand side mistakes?** Required evidence: a hand-track/active-side audit on permissible development video with explicit left/right IDs.
4. **Do temporal candidates improve local pose after centering?** Required evidence: candidate-reported or separately validated decomposition into root rotation, 15-joint articulation, shape/scale and translation; global-trajectory improvement alone is insufficient.
5. **What is the exact HaMeR/MANO ↔ DexAvatar SMPL-X convention?** Required files/confirmation: MANO and SMPL-X model versions, joint ordering, pose means, coordinate axes, crop parity, left/right conversion, and whether wrist orientation is local to the forearm or global to camera.
6. **Can Hand4Whole++’s reported benefit survive parameter regeneration?** Required evidence: a comparison between its vertex-scattered output and vertices regenerated solely from the exported coherent SMPL-X parameter vector, on non-test data.
7. **Which upstream DexAvatar variables may change without changing the task?** Required author/protocol confirmation: whether body-chain pose, shape and wrist parent rotations can be reconsidered while retaining the official representation and fair baseline.
8. **What public sign-domain supervision is legally and technically usable?** Required manifests/licenses: exact SignAvatars SMPL-X annotations/splits; How2Sign and Neural Sign Actors video/pose availability; signer/clip split rules; no use of SGNify test meshes.
9. **How should sequence boundaries and missing detections be represented?** Required dataset metadata: clip boundaries, frame cadence, side-presence masks, dropped frames and one-/two-hand class labels. This is necessary to compare 8-, 16-, 128-frame and whole-window claims without assuming uniform coverage.
10. **Is fixed SMPL-X shape a significant hand-error floor?** Required evidence: development-set subject/sequence shape stability and a topology-consistent sensitivity analysis; pure hand centering does not eliminate scale/shape errors.
11. **What is the clean ablation boundary?** Required baseline outputs: frozen per-frame HaMeR/SMPLer-X observations, exact fitted parameters and deterministic frame manifest so that temporal observation recovery and body–hand kinematic consistency can be evaluated separately.
12. **Does SGNify GT plausibility confound the chosen diagnostic subsets?** Required evidence: blind anatomical QA flags kept separate from test-score tuning; the flags may interpret failures but must not alter official evaluation.

---

## 12. Source manifest

### 12.1 Baseline sources

| Source | Exact identity and access | Coverage |
|---|---|---|
| Step-1 dossier | `DexAvatar_Baseline_and_TR-V2V_Evaluation_Dossier(1).md` supplied as workspace copy `DexAvatar_Baseline_and_TR-V2V_Evaluation_Dossier.md`; SHA-256 `715e36be9bf892386f78fa2833c981b4e6485b74cca4a07afa685e8d179d44b5` | Read in full, lines 1–804. Bottleneck register lines 626–648; metric specification lines 294–345; source manifest lines 686–787. |
| DexAvatar paper + supplementary | “DexAvatar: 3D Sign Language Reconstruction with Hand and Body Pose Priors,” Kundu et al.; supplied 21-page PDF; [CVF paper page](https://openaccess.thecvf.com/content/WACV2026/html/Kundu_DexAvatar_3D_Sign_Language_Reconstruction_with_Hand_and_Body_Pose_WACV_2026_paper.html) | PDF pp. 1–8 main paper and pp. 9–18 entire supplementary read in full; pp. 19–21 references checked. Text extraction plus rendered-page inspection inherited from Step 1. |
| Official DexAvatar repository | [kaustesseract/DexAvatar](https://github.com/kaustesseract/DexAvatar), branch `main`, commit `a0dfd427f60f5811aadb35c8657b3856d47f56b5`, clean tree, accessed 2026-08-25 | Static reinspection focused on pipeline, observations, temporal state and loss activation. Full/partial file coverage below. |
| Supplied evaluator | Workspace attachment `evaluate_new_fitting(2).py`; `evaluate_new_fitting(1).py` **NOT SUPPLIED / NOT INSPECTED** | `(2)` read in full in Step 1. Key ranges reused: L152–203 alignment functions; L231–280 frame selection/discovery; L283–461 pairing, masks and aggregation; L467–589 CLI/assets. Evaluator treated as locked. |

### 12.2 DexAvatar repository file coverage

**Read in full** at commit `a0dfd427…`: `README.md`; `run_dexavatar.py`; `Full_running_command.sh`; `M3_mean_shape_smplerx.py`; `scripts/config*.sh`; `scripts/S1_sapiens_extract.sh`; `scripts/S1_smplerx_extract.sh`; `scripts/M3.5_hamer_extract.sh`; `scripts/M4_smplifyx_pose.sh`; `scripts/env_install.sh`; `scripts/bug_fix_dexavatar.sh`; `data/signs.txt`; `data/segment.json`; `dexavatar_fitting/script.py`; both fitting YAMLs; `dexavatar_fitting/smplifyx/{main.py,cmd_parser.py,fit_single_frame.py,fitting.py,data_parser.py,prior.py,body_constants.py,camera.py,test_bposer.py,test_hposer.py,utils.py}`; optimizer factory/LBFGS wrapper; `assets/mapping_func.py`; `hamer/demo.py`; `SMPLer-X/main/{script_smplerx.py,inference.py,config.py}`; `SMPLer-X/main/config/config_smpler_x_h32.py`; relevant licenses/requirements.

**Read in part:** `rewrite_body_model.py` constructor/reset and SMPL-X forward; `assets/joint_mapping.py` name/mapping regions; `SMPLer-X/main/SMPLer_X.py` rotation/output sections; vendored HaMeR, collision and renderer interfaces at called entry points. **NOT INSPECTED:** missing Sapiens Gitlink content; missing SignBPoser/SignHPoser external folders/checkpoints; license-gated SMPL-X assets; SGNify meshes/masks and unreleased prior training data. No content of a missing item is inferred.

### 12.3 Deep-audit papers and supplementary

All were accessed 2026-08-25.

| Paper | Sections/pages inspected | Supplementary status |
|---|---|---|
| [HandFlow](https://arxiv.org/abs/2607.11221) | Main method Sec. 3, experiments Sec. 4, limitations and all arXiv appendices covering state, windowing, conditioning, datasets and release | Appendices in arXiv v1 inspected; no separate publisher supplement verified. Venue beyond arXiv/in-submission metadata remains `[UNRESOLVED]`. |
| [HaPTIC](https://arxiv.org/abs/2501.08329) | Secs. 3–5, architecture/data/trajectory-vs-local-pose tables, limitations and appendices | ArXiv appendices inspected; 3DV 2026 status cross-checked on author publication page. |
| [Hand4Whole++](https://openaccess.thecvf.com/content/CVPR2026/html/Moon_Enhancing_Hands_in_3D_Whole-Body_Pose_Estimation_with_Conditional_Hands_CVPR_2026_paper.html) | Main Secs. 3.1–3.4, Figs. 1–3, Tables 1–5; representation/ablation sections | CVF supplementary inspected for architecture, training/data and additional ablations. |
| [PAD-Hand](https://openaccess.thecvf.com/content/CVPR2026/html/Ismayilzada_PAD-Hand_Physics-Aware_Diffusion_for_Hand_Motion_Recovery_CVPR_2026_paper.html) | Main Secs. 3–4, Eqs. 1–14, Tables 1–5 | CVF supplementary inspected for physical formulation, sequence representation and ablations. |
| [Dyn-HaMR](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html) | Main Secs. 3–4, Figs. 2–3, representation/optimization/ablation tables | CVF supplementary inspected for per-hand variables, initialization, stages and prior. |
| [WiLoR](https://openaccess.thecvf.com/content/CVPR2025/html/Potamias_WiLoR_End-to-end_3D_Hand_Localization_and_Reconstruction_in-the-wild_CVPR_2025_paper.html) | Main paper Secs. 3–4 and all reported output conventions/benchmarks | CVF supplementary read; local PDFs retained as `step2_work/papers/WiLoR_CVPR2025*.pdf`. |

### 12.4 Landscape papers screened

The following primary sources were inspected at least through method, experiments relevant to the bottleneck, supplementary where linked, and official artifact status: [StableHand](https://arxiv.org/abs/2605.18553); [HMP](https://openaccess.thecvf.com/content/WACV2024/html/Duran_HMP_Hand_Motion_Priors_for_Pose_and_Shape_Estimation_From_WACV_2024_paper.html); [Deformer](https://openaccess.thecvf.com/content/ICCV2023/html/Fu_Deformer_Dynamic_Fusion_Transformer_for_Robust_Hand_Pose_Estimation_ICCV_2023_paper.html); [BlurHand](https://arxiv.org/abs/2303.15417); [HandOS](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_HandOS_3D_Hand_Reconstruction_in_One_Stage_CVPR_2025_paper.html); [ACR](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_ACR_Attention_Collaboration-Based_Regressor_for_Arbitrary_Two-Hand_Reconstruction_CVPR_2023_paper.html); [MaskHand](https://openaccess.thecvf.com/content/ICCV2025/html/Saleem_MaskHand_Generative_Masked_Modeling_for_Robust_Hand_Mesh_Reconstruction_in_ICCV_2025_paper.html); [MeMaHand](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_MeMaHand_Exploiting_Mesh-Mano_Interaction_for_Single_Image_Two-Hand_Reconstruction_CVPR_2023_paper.html); [TempCLR](https://arxiv.org/abs/2209.00489). Deformer and HMP main/supplementary PDFs were retained locally. These nine were not treated as equivalent to the six line-by-line deep audits.

### 12.5 Repository manifest

All hashes are exact HEADs inspected or resolved on 2026-08-25.

| Repository | Branch / commit | Inspected artifacts | Availability conclusion |
|---|---|---|---|
| [mxxu00/HandFlow](https://github.com/mxxu00/HandFlow) | `main` / `67fa7df536db233408fe6270ca5d2de28d5959c3` | README, configs, inference entry points, online HaMeR wrapper, download scripts, license | Inference + checkpoint **AVAILABLE**; training/preprocessing **MISSING**. |
| [JudyYe/haptic](https://github.com/JudyYe/haptic) | `main` / `f9362c1bdf2c1ea2bfa695be2d4e6f362371e7df` | README, `demo.py`, `train.py`, configs, sequence/detection utilities | Training/inference/checkpoint **AVAILABLE**; raw data/MANO **REQUIRES LICENSE**; root license **NOT FOUND**. |
| [mks0601/Hand4Whole-plus-plus](https://github.com/mks0601/Hand4Whole-plus-plus) | `main` / `f81d35ddd2b74206c40142243eb62b6d64ce0d65` | README, demo/train, `main/model.py`, WiLoR wrapper, configs, license | Full code/checkpoint **AVAILABLE**; SMPL-X/MANO assets **REQUIRE LICENSE**; final vertex/parameter equivalence not satisfied. |
| [ZhengdiYu/Dyn-HaMR](https://github.com/ZhengdiYu/Dyn-HaMR) | `main` / `fa9cd7412c205fd15ee4139c8caacf79bf6167e6` | README, configs, inference/optimization/prior entry paths, license | Code/checkpoints **AVAILABLE**; MANO/assets licensed; optional-prior default documented above. |
| [DominoAI-Lab/PAD-Hand-CVPR-2026](https://github.com/DominoAI-Lab/PAD-Hand-CVPR-2026) | `main` / `ca9ed97bc671199c25cf569d3b1de0e6f7937251` | README, demo, MANO preprocessing, WiLoR inference, requirements | Demo/checkpoint **PARTIALLY AVAILABLE**; training/eval/license **MISSING/UNVERIFIED**. |
| [rolpotamias/WiLoR](https://github.com/rolpotamias/WiLoR) | `main` / `fcb911312a38fa8badd30d9656a167485d61b8f9` | README, demo, dataset/crop code, model interfaces, license | Code/models/data **AVAILABLE** under stated non-commercial/model terms; MANO licensed. |
| [enesduran/HMP](https://github.com/enesduran/HMP) | `main` / `35d799f76b2b2bc1d1e945117b021014b099e7e6` | README, prior training/optimization entry paths, license | Code/checkpoints/training instructions **AVAILABLE**; data/assets licensed. |
| [fuqichen1998/Deformer](https://github.com/fuqichen1998/Deformer) | `main` / `64dbedbff2417b5fa2881e72705a6eeb1f88b514` | README, model, train/eval, license | Training/evaluation **AVAILABLE**; official checkpoint **NOT VERIFIED**. |
| [JaehaKim97/BlurHand_RELEASE](https://github.com/JaehaKim97/BlurHand_RELEASE) | `master` / `19864229065f7c52238155df933da1fc0e95f1e9` | README and release structure | Code **AVAILABLE**; current checkpoint/license status **UNVERIFIED**. |
| [huajian-zeng/stablehand](https://github.com/huajian-zeng/stablehand) | `main` / `6d38cc3cf31bc467b39da94970a3deba9e73a314` | One-commit README/assets | **PLACEHOLDER**; code and checkpoint **MISSING**. |
| [eth-ait/tempclr](https://github.com/eth-ait/tempclr) | `main` / `c62d0e17e451a952ad65e0a1321cb13a5467f6cc` | README and public tree | Training/inference/model **AVAILABLE**; MANO/data licensed. |
| [ZhengdiYu/Arbitrary-Hands-3D-Reconstruction](https://github.com/ZhengdiYu/Arbitrary-Hands-3D-Reconstruction) | `main` / `a4e462628e58e2ef2e28556f63858b0dc7c576bc` | README/public tree | Code/checkpoint path/MIT **AVAILABLE**; MANO licensed. |
| [idea-research/HandOS](https://github.com/idea-research/HandOS) | N/A | Direct URL access | **404 / NOT INSPECTED**; no public code claim accepted. |
| MaskHand official repository | `NOT FOUND` | Paper/project links | Code/checkpoint **UNVERIFIED**. |
| MeMaHand official repository | `NOT FOUND` | Paper/project search | Code/checkpoint **UNVERIFIED**. |

### 12.6 Quality-control record

- No final method, architecture, objective, loss, module composition, code, pseudocode or implementation plan is included.
- No candidate score is described as measured; no SGNify score is predicted or reproduced.
- Paper claims, released-code behavior and evaluator behavior remain separate.
- Official TR-V2V linkage explicitly respects independent region centering, class-0 masks and vertex-frame weighting.
- Evaluator artifacts, test-GT use, topology replacement, vertex/rotation averaging and wrist hard-overwrite are excluded.
- Important factual statements carry `[VERIFIED]`, `[INFERENCE]` or `[UNRESOLVED]`; absence of a repository/checkpoint is not silently converted into availability.

STEP 2 COMPLETE — PRIMARY AND BACKUP SCIENTIFIC BOTTLENECKS SELECTED; READY FOR EXTERNAL REVIEW BEFORE METHOD FORMULATION.
