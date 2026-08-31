# DexAvatar Step 3 — Feasibility, Compatibility and Novelty Gate

**Audit date:** 2026-08-26 (Asia/Bangkok)  
**Mode:** paper, supplementary, official documentation, and static-source inspection only. No repository, checkpoint, dataset, or evaluator was executed.  
**Evidence labels:** `VERIFIED`, `INFERRED`, `UNKNOWN`, `CONTRADICTED`.  
**Metric-benefit labels:** `PRESERVED BY TR-V2V`, `PARTIALLY PRESERVED`, `REMOVED BY CENTROID ALIGNMENT`, `UNKNOWN`.

The Step 2 PRIMARY is treated below as a hypothesis to audit, not a premise. The official SGNify evaluator—including frames, vertex sets, independent region-centroid alignment, class-`0` rules, and vertex-frame aggregation—is locked. “Public” means present on an official source at the access date, not executed or reproduced.

## 1. Executive verdict

**Decision: `GO WITH SCOPE REDUCTION`.**

- `[VERIFIED]` The part of the PRIMARY with direct official-metric leverage is recovery of the **15 local finger-joint rotations of each physical hand**, with stable left/right identity. Local articulation changes centered hand geometry and survives LHand/RHand centroid alignment. Pure translation and most camera/world-trajectory improvement are `REMOVED BY CENTROID ALIGNMENT` ([Step 1 dossier, §§2, 6 and 8](./upload/DexAvatar_Baseline_and_TR-V2V_Evaluation_Dossier(2).md); evaluator `transl_point_error`, lines 159–169, and aggregation, lines 432–461).
- `[VERIFIED]` DexAvatar has no learned or optimized temporal hand-state model. `[CONTRADICTED]` The stronger claim that it has no hand-state propagation at all is false: class-`0` branches can reuse a one-frame cache of the active hand’s rotations/keypoints. This cache is neither a sequence prior nor joint two-hand recovery ([DexAvatar `data_parser.py`, lines 120–130 and 445–650](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/data_parser.py#L445-L542)).
- `[VERIFIED]` HandFlow is the strongest available substrate for the reduced state: it performs non-causal full-window generative MANO-state inference, conditions on visual/skeletal evidence and confidence, releases an inference checkpoint, and reports root-aligned/local as well as world-space results ([HandFlow, §3 and Tables 1–2](https://arxiv.org/html/2607.11221)). `[VERIFIED]` Its confidence ablation does **not** isolate local-articulation improvement; the authors say confidence chiefly stabilizes translation ([HandFlow, Table 4 and §4.5](https://arxiv.org/html/2607.11221#S4.SS5)).
- `[VERIFIED]` None of the five candidates supplies all of: a learned joint-bimanual local model, exact left/right-to-SMPL-X conversion, a released retrainable pipeline, and a final mesh reproducible from one SMPL-X parameter vector. HandFlow/HMP are single-hand and right-canonical; HaPTIC processes sides separately; Dyn-HaMR tracks both sides but uses per-hand HMP latents and emphasizes world trajectory; Hand4Whole++ scatters aligned MANO vertices into its final mesh.
- `[INFERRED]` PRIMARY retains greater direct leverage on LHand/RHand than BACKUP because local-finger errors move many evaluated hand vertices. BACKUP is metric-aligned—especially wrist/forearm consistency—but its strongest audited evidence is per-frame, and Hand4Whole++’s literal mesh path violates unified SMPL-X parameterization.

**Reduced PRIMARY:** estimate two persistent-side sequences of 15 SMPL-X-compatible local finger rotations under missing/low-confidence monocular evidence, while holding DexAvatar body pose, wrist/root orientation, SMPL-X shape, camera, translation, upstream per-frame evidence, and evaluator fixed. Two side-indexed streams are **not** called learned joint-bimanual modeling.

**Technical base:** HandFlow whole-window generative MANO-state inference.  
**Optional auxiliary mechanism:** Dyn-HaMR fixed side-indexed track identity and observation-validity contract only—not its SLAM, world trajectory, or full optimizer.

## 2. Corrections and validation of Step 2

| Step 2 claim | Primary-source check | Status | Gate consequence |
|---|---|---|---|
| Blur, hand–hand/hand–body occlusion, missed detections, and unreliable 2D evidence are DexAvatar failure channels. | Paper/supplementary and the Step 1 bottleneck register identify these channels; parsing removes missing upstream results and uses detector confidence/side logic ([DexAvatar paper, failure discussion; Step 1 B01–B08](./upload/DexAvatar_Baseline_and_TR-V2V_Evaluation_Dossier(2).md)). | `[VERIFIED]` | The regime is real; its frequency in leakage-free sign validation remains `[UNKNOWN]`.
| DexAvatar has no temporal hand propagation. | Class-`0` code caches/reuses previous hand state; no temporal latent/hand sequence objective is optimized. | Absolute claim `[CONTRADICTED]`; “no learned/optimized temporal hand model” `[VERIFIED]`. | Baseline must retain the released cache behavior.
| Temporal hand recovery directly affects official TR-V2V. | Articulation/rotation/shape survive centering; pure translation does not. | `[VERIFIED]` only for metric-preserved state. | Candidate trajectory claims must be separated from local pose.
| HandFlow’s confidence masking is proven to correct local fingers. | §3.2 defines confidence masking, but Table 4 changes MPJPE far more than PA-MPJPE; authors attribute most effect to translation. Random masking improves temporal metrics but not reported MPJPE. | `[CONTRADICTED]` as an isolated local-pose claim; overall HandFlow local signal `[VERIFIED]`. | Confidence alone is neither sufficient evidence nor novelty.
| HandFlow is bimanual. | Paper/release operate one MANO stream; DexYCB protocol uses right-hand crops; README asks users to mirror left videos. | `[CONTRADICTED]` | Use only as a single-hand temporal substrate; side identity is external and explicit.
| HaPTIC materially improves local 3D pose beyond HaMeR. | HO3D Appendix Table 6 reports HaPTIC PA-MPJPE/PA-MPVPE 8.0/8.1 versus HaMeR 7.7/7.9; its clear advantages are trajectory and occluded-2D alignment ([HaPTIC Appendix Table 6](https://arxiv.org/html/2501.08329)). | `[CONTRADICTED]` as a general local-3D claim. | Not selected as metric-preserved base.
| Dyn-HaMR has one learned joint-bimanual prior. | It maintains two side tracks and jointly optimizes interaction constraints, but HMP latents are per hand; coupling comes through optimization, shared camera/scale, biomechanics, and penetration ([Dyn-HaMR §§3.1–3.3](https://arxiv.org/html/2412.12861)). | `[CONTRADICTED]` for the learned-prior wording; identity continuity `[VERIFIED]`. | Only the side/validity contract is retained.
| HMP is a reusable bimanual prior. | HMP learns a right-hand local prior and reflects left motions; project warns about close/interacting-hand crop/keypoint failure ([HMP supplementary; `amass.py`, lines 96–139](https://github.com/enesduran/HMP/blob/35d799f76b2b2bc1d1e945117b021014b099e7e6/src/datasets/amass.py#L96-L139)). | Bimanual claim `[CONTRADICTED]`; local single-hand prior `[VERIFIED]`. | Strong nearest prior, not the selected base.
| Hand4Whole++ final mesh is regenerated by its reported SMPL-X parameters alone. | It predicts SMPL-X parameters but also rigidly aligns a MANO hand, scatters vertices, and smooths the seam ([paper §3.3](https://arxiv.org/html/2603.14726); [`main/model.py`, lines 42–126, 160–222](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE/blob/f81d35ddd2b74206c40142243eb62b6d64ce0d65/main/model.py#L42-L126)). | `[CONTRADICTED]` for final mesh; CHAM wrist evidence `[VERIFIED]`. | Useful for BACKUP evidence, inadmissible as literal final output.
| SignAvatars is turnkey paired real RGB–mocap SMPL-X supervision. | Official parameters are pseudo fits; annotations and upstream videos have separate access. Repository exposes frame arrays and validity, not mocap GT ([paper §3](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/653_ECCV_2024_paper.php); [format](https://github.com/ZhengdiYu/SignAvatars#dataset-format)). | `[CONTRADICTED]` | Pairing, license, split, and pseudo-label quality must be audited rather than assumed.

Additional metric facts retained from Step 1:

- `[VERIFIED]` Class-`0` signs omit LHand and remove left-hand vertices from UBody(-F), so left-only corrections have no official contribution on those excluded terms (evaluator lines 380–395).
- `[VERIFIED]` Relative hand-to-body displacement is `PARTIALLY PRESERVED`: it affects composite UBody(-F), but independent LHand/RHand centering removes it from isolated hand regions.
- `[VERIFIED]` Official aggregation is vertex-frame weighted, not mean-per-sign. Temporal smoothness is `UNKNOWN` unless it improves centered geometry on evaluated frames.
- `[INFERRED]` PRIMARY has the highest potential hand-region leverage, but evidence does not prove that unreliable evidence explains most of DexAvatar’s aggregate error.
- `[VERIFIED]` BACKUP is metric-aligned: Hand4Whole++ Table 2 shows direct wrist copying can worsen body and hand errors, while body-context conditioning improves them ([Table 2](https://arxiv.org/html/2603.14726#S4.SS2)).

## 3. Metric-relevant state and coordinate contract

### 3.1 Locked DexAvatar contract

`[VERIFIED]` DexAvatar optimizes a 33-D SignBPoser body latent and one/two 23-D SignHPoser hand latents, decoded to axis-angle pose. Its final optimizer list excludes global orientation, SMPL-X betas, expression, jaw/eyes, translation, and camera ([`fit_single_frame.py`, lines 223–243 and 476–503](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/fit_single_frame.py#L476-L503)). `[VERIFIED]` HaMeR’s 15 hand rotation matrices become 15 Rodrigues vectors; the parser flips left y/z axis-angle signs and does not install HaMeR root/global orientation into the SMPL-X wrist ([`data_parser.py`, lines 397–425](https://github.com/kaustesseract/DexAvatar/blob/a0dfd427f60f5811aadb35c8657b3856d47f56b5/dexavatar_fitting/smplifyx/data_parser.py#L397-L425)).

The reduced contract therefore accepts only a side-labelled 15×3 **local finger** rotation block per physical hand, in DexAvatar’s joint order, pose-mean convention, handed local frames, and axis-angle layout. SMPL-X wrist/body chain stays unchanged; candidate root, translation, camera, MANO β, and vertices are excluded. This is a representation contract, not a fusion rule.

### 3.2 State table

| State | DexAvatar / candidate facts | Coordinate and mapping rule | Unified SMPL-X mapping | Metric fate | Status |
|---|---|---|---|---|---|
| Left/right identity | Dex routes 2D by `is_right`, but reads non-class-`0` rotations by fixed list positions; class-`0` uses Sapiens wrist motion and cache. HandFlow/HMP are right-canonical; HaPTIC has side crops; Dyn carries side+track ID; Hand4Whole++ has explicit side handling. | Physical side label must persist across time; mirror parity must not change identity. | **Direct** only after track is fixed before writing `left_hand_pose`/`right_hand_pose`. | Side swaps alter articulation: `PRESERVED BY TR-V2V`; class-`0` masks still apply. | Contract `[VERIFIED]`; Dex list-order identity risk `[INFERRED]`.
| Left 15 local rotations | Dex accepts 15×3 axis-angle with left parity handling. Candidate outputs are MANO-local in differing interfaces. | Must verify right-canonical→left parity, pose mean, joint order, and parent frames. | Dimensionally direct; exact HandFlow-left conversion `[UNKNOWN]`. No invented map accepted. | `PRESERVED BY TR-V2V` where left vertices are evaluated. | Dimensions `[VERIFIED]`; convention `[UNKNOWN]`.
| Right 15 local rotations | All hand candidates expose a right/local finger state; HandFlow θ48 contains root3+finger45. | Remove root3; confirm MANO/SMPL-X local order and pose mean. | Conditional-direct; no vertex stitching. | `PRESERVED BY TR-V2V` for RHand and UBody(-F). | State `[VERIFIED]`; cross-model convention partly `[UNKNOWN]`.
| Wrist/root orientation | Dex wrist is in SMPL-X body pose; body global orientation is separate/fixed. Candidates often output MANO root/global orientation. | MANO root/global and SMPL-X wrist-local have different parents. | **Excluded** from reduced PRIMARY; no hard overwrite. | Rotation `PRESERVED BY TR-V2V`; translated root location mostly removed in isolated hands. | `[VERIFIED]`.
| Local vs global rotation | Dex finger rotations are local to SMPL-X chain; candidates separate local pose/root/world to differing degrees. | Axis-angle equality does not imply equal coordinate frame. | Only verified local finger block is admissible. | Local articulation/root rotation `PRESERVED BY TR-V2V`; translation removed. | `[VERIFIED]`.
| MANO β vs SMPL-X β | Dex SMPL-X β is unified and fixed. HandFlow predicts β10; HaPTIC/Dyn/HMP estimate MANO β; Hand4Whole++ uses MANO hand shape. | These are different learned shape spaces despite similar names/dimensions. | Direct copy incompatible; candidate β excluded. | Shape/scale is `PRESERVED BY TR-V2V`, but candidate benefit is out of scope. | `[VERIFIED]`.
| Translation/camera/root trajectory | Dex final hand fitting fixes them. HandFlow/HaPTIC/Dyn/HMP predict or optimize them; Dyn adds SLAM. | Crop camera, hand camera translation, SMPL-X translation, and world trajectory are distinct. | Excluded. | Pure translation `REMOVED BY CENTROID ALIGNMENT`; relative body–hand placement `PARTIALLY PRESERVED`. | `[VERIFIED]`.
| Confidence/uncertainty | Dex uses detector scores for routing/side logic, not a calibrated 45-D posterior. HandFlow uses confidence masking; Dyn/HMP use validity/masks; HaPTIC valid fraction/interpolation. | Detector confidence is conditioning metadata, not an SMPL-X state. | Conditioning only. | `UNKNOWN` unless it changes centered local geometry; HandFlow shows mainly translation effect. | Mechanisms `[VERIFIED]`; local benefit `[UNKNOWN]`.
| Missing-frame state | Dex removes frames lacking required upstream estimates and sometimes reuses class-`0` cache. Candidates mask, interpolate, or infill with different semantics. | Missingness must remain side- and timestamp-specific; deletion changes temporal adjacency. | Track metadata, not SMPL-X parameter. | Indirectly `PRESERVED BY TR-V2V` only when recovered local geometry is evaluated. | Dex behavior `[VERIFIED]`; equivalence across candidates `[UNKNOWN]`.
| Timestamp/FPS | Dex orders retained file names; no explicit FPS enters its hand prior. HandFlow T=16; HaPTIC T=8; HMP/Dyn use longer/chunked sequences. | File index is not physical time; skipped frames alter velocities. | Requires preserved source indices; plug-compatible sampling `[UNKNOWN]`. | `UNKNOWN`—timing affects inference, not directly scored. | Window sizes `[VERIFIED]`.
| Shoulder–elbow–wrist–finger chain | Shoulder/elbow/wrist are body pose; fingers are hand pose; one SMPL-X forward pass creates mesh. Hand-only candidates omit body chain; Hand4Whole++ explicitly conditions wrist. | Wrist-local is relative to forearm, unlike MANO root. | Frozen in reduced PRIMARY; chain correction belongs to BACKUP. | Rotations `PRESERVED BY TR-V2V`; displacement `PARTIALLY PRESERVED`. | `[VERIFIED]`.

### 3.3 Metric-preservation summary

| Candidate-side change | Classification | Reason |
|---|---|---|
| Correct 15 local finger rotations with wrist fixed | `PRESERVED BY TR-V2V` | Changes centered hand shape and many evaluated vertices.
| Correct wrist/root rotation | `PRESERVED BY TR-V2V` | Centroid subtraction cannot remove rotation.
| Correct hand shape/scale | `PRESERVED BY TR-V2V` | Centering does not remove deformation, although MANO β transfer is inadmissible.
| Correct only global/camera-space translation | `REMOVED BY CENTROID ALIGNMENT` | Each frame/region is centered independently.
| Correct wrist-to-torso displacement | `PARTIALLY PRESERVED` | Retained in UBody(-F), removed from independently centered hand metrics.
| Make motion smoother without changing centered per-frame geometry | `UNKNOWN` | The evaluator contains no temporal-smoothness term.

## 4. Candidate artifact and representation matrix

All repository states below were inspected on 2026-08-26.

| Candidate | Input → output | Temporal / hand scope | Missingness and identity | Public artifacts | License / training feasibility | SMPL-X compatibility | Metric benefit | Gate role |
|---|---|---|---|---|---|---|---|---|
| **HandFlow** — Xu et al., *HandFlow: Fully Generative 4D Hand Recovery with Flow Matching* (arXiv 2026; metadata says TOG) | Monocular hand video; frozen HaMeR visual tokens, 21 keypoint rays and confidence → β10, per-frame θ48 axis-angle, translation3 ([§3](https://arxiv.org/html/2607.11221)). | T=16, non-causal whole-window flow with overlapping windows; single hand/right-canonical, not joint bimanual. | Continuous confidence mask + random masking; no two-hand identity model. | `[VERIFIED]` Inference/demo, checkpoint and normalization public. `[CONTRADICTED]` README marks training, full preprocessing and evaluation TODO despite one helper ([repo](https://github.com/mxxu00/HandFlow/tree/67fa7df536db233408fe6270ca5d2de28d5959c3)). | MIT code; CC BY-NC-SA paper; MANO gated. Checkpoint usable; exact retraining not established. | Finger45 conditionally usable; exact left parity/mean/order `[UNKNOWN]`. Root/β/translation/mesh excluded. | Local pose `PRESERVED BY TR-V2V`; translation `REMOVED BY CENTROID ALIGNMENT`; isolated confidence gain `UNKNOWN`. | **Selected technical base, reduced scope.**
| **HaPTIC** — Ye et al., *Predicting 4D Hand Trajectory from Monocular Videos* | Eight hand crops + full frames, adapted HaMeR attention → MANO pose/shape, camera/trajectory ([§3.3](https://arxiv.org/html/2501.08329#S3.SS3)). | T=8, one-frame overlap; sides are separate right-canonical streams. | Requires ≥50% valid frames/side; missing boxes interpolated. | Training/inference, checkpoint, preprocessing labels public ([README](https://github.com/JudyYe/haptic/blob/f9362c1bdf2c1ea2bfa695be2d4e6f362371e7df/README.md)). | Root license not found `[UNKNOWN]`; paper reports 1M iterations on 8 H100s. | Released left path flips crop then mirrors vertices/joints after right-MANO; verified left-local SMPL-X export absent ([`seq2clip.py` L97–224](https://github.com/JudyYe/haptic/blob/f9362c1bdf2c1ea2bfa695be2d4e6f362371e7df/haptic/datasets/seq2clip.py#L97-L224); [`demo.py` L342–394](https://github.com/JudyYe/haptic/blob/f9362c1bdf2c1ea2bfa695be2d4e6f362371e7df/demo.py#L342-L394)). | Trajectory `REMOVED BY CENTROID ALIGNMENT`; local pose `PRESERVED BY TR-V2V` in principle, but no HO3D gain over HaMeR. | Rejected as base.
| **Dyn-HaMR** — Yu et al., CVPR 2025 | RGB + multi-detector tracks + SLAM + HaMeR/HMP → per-hand local pose, β, root/translation, world motion ([§3](https://arxiv.org/html/2412.12861)). | Two side tracks; chunked temporal optimization; per-hand HMP latents plus interaction constraints, not one learned bimanual latent. | Explicit side/track, validity, interpolation and infilling. | Fitting/preprocessing public; many dependencies. Release config has `run_prior:false` ([README](https://github.com/ZhengdiYu/Dyn-HaMR/blob/fa9cd7412c205fd15ee4139c8caacf79bf6167e6/README.md); [`config.yaml`](https://github.com/ZhengdiYu/Dyn-HaMR/blob/fa9cd7412c205fd15ee4139c8caacf79bf6167e6/dyn-hamr/confs/config.yaml#L1-L58)). | MIT; gated MANO/BMC/dependencies. | Local rotations conceptually mappable; full output is MANO/world, not unified SMPL-X. Discrete side/validity contract is representation-neutral. | Local `PRESERVED BY TR-V2V`; world trajectory `REMOVED BY CENTROID ALIGNMENT`; interaction placement `PARTIALLY PRESERVED`. | **Optional auxiliary contract only.**
| **HMP** — Duran et al., WACV 2024 | Image/keypoint estimates + learned motion prior → optimized MANO local pose, root, translation, shape ([paper](https://openaccess.thecvf.com/content/WACV2024/html/Duran_HMP_Hand_Motion_Priors_for_Pose_and_Shape_Estimation_From_WACV_2024_paper.html)). | Non-autoregressive T=128; right-hand prior, left reflected; single-hand. | Masks occluded/missing observation terms; latent infilling. | Full training/fitting/preprocessing public; checkpoint after registration ([README](https://github.com/enesduran/HMP/blob/35d799f76b2b2bc1d1e945117b021014b099e7e6/README.md)). | Non-commercial research license; MANO/SMPL-X gated. | Local 6-D rotations can yield local finger rotations after verified convention; root/shape excluded. Left transform documented. | Local `PRESERVED BY TR-V2V`; translation `REMOVED BY CENTROID ALIGNMENT`; smoothing alone `UNKNOWN`. | Strong nearest prior; not base.
| **Hand4Whole++** — Moon, CVPR 2026 | Full image; frozen whole-body + hand features → SMPL-X parameters plus post-alignment/scattered hand vertices ([§3](https://arxiv.org/html/2603.14726)). | Single-frame; both hands in body context; no temporal identity. | No temporal missing state; explicit side crop/parity. | Train/demo/checkpoints public ([README](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE/blob/f81d35ddd2b74206c40142243eb62b6d64ce0d65/README.md)). | MIT; SMPL-X/MANO gated. | CHAM wrist is SMPL-X; final enhanced mesh is not regenerated by SMPL-X params alone because of MANO scatter/smoothing. | Wrist/body chain `PRESERVED BY TR-V2V`; literal vertex path inadmissible. | BACKUP evidence only.

### 4.1 Deep candidate findings

**HandFlow.** `[VERIFIED]` State dimension is (10+51T): shared β10, T×θ48 (root3+finger45 axis-angle), and T×translation3. A dual-stream flow model denoises full-window latent and visual/skeletal condition tokens; long videos use velocity-blended overlapping windows ([§3, Eqs. 1 and 7–9](https://arxiv.org/html/2607.11221)). `[VERIFIED]` DexYCB Table 1 reports RA-MPJPE 8.12 and PA-MPJPE 3.88 with occlusion-bin PA errors 4.21/4.35/4.85; HOT3D Table 2 reports PA-MPJPE 5.49. These are reported, not reproduced, and RA/PA alignment is not SGNify centroid-V2V. They support only a local signal beyond raw trajectory. `[VERIFIED]` Table 4 says no-confidence mainly harms translation; random masking trades small per-frame accuracy for temporal/world coherence. `[CONTRADICTED]` Handedness documentation is not closed: README says mirror left video; demo exposes `--side left`; invalid sides default right; MANO utility feeds θ48 directly to selected layer without a visible complete parity/pose-mean conversion ([`online_hamer.py` L195–255, 373–392](https://github.com/mxxu00/HandFlow/blob/67fa7df536db233408fe6270ca5d2de28d5959c3/utils/online_hamer.py#L195-L255); [`mano_utils.py` L14–117](https://github.com/mxxu00/HandFlow/blob/67fa7df536db233408fe6270ca5d2de28d5959c3/utils/mano_utils.py#L14-L117)). It is viable only as a fixed checkpoint substrate with left semantics closed before method formulation.

**HaPTIC.** `[VERIFIED]` It adds cross-view temporal attention and global-context attention to an image transformer over eight frames, training on five video and ten image datasets ([§§3.3–4.1](https://arxiv.org/html/2501.08329)). Its main advantage is coherent trajectory. HO3D local PA results are slightly worse than HaMeR, while HInt occluded-joint PCK improves. `[VERIFIED]` The release flips left crops into right-canonical input; demo mirrors resulting vertices/joints, but does not establish an authoritative left-local SMPL-X parameter vector. Code/checkpoint/training are public, but root license is `[UNKNOWN]` and exact retraining is costly.

**Dyn-HaMR.** `[VERIFIED]` It explicitly represents per-hand local pose, β, root orientation, translation, and handedness; tracks left/right and infills invalid spans ([§§3.1–3.3](https://arxiv.org/html/2412.12861)). Local evidence is real: InterHand Table 1 gives 7.94/8.15 full versus 8.98/9.25 without Stage III and 9.84/10.13 HaMeR; H2O Table 6 gives MPJPE 22.5 versus 24.1 without generative infilling. These are reported, not SGNify-reproduced. `[CONTRADICTED]` Release behavior differs in places: prior off by default and loaded keypoint confidence reset to one after interpolation ([`dataset.py` L213–305](https://github.com/ZhengdiYu/Dyn-HaMR/blob/fa9cd7412c205fd15ee4139c8caacf79bf6167e6/dyn-hamr/data/dataset.py#L213-L305)). Its tracking submodule was absent from the inspected shallow tree. The full system is rejected because much of its contribution is world trajectory.

**HMP.** `[VERIFIED]` HMP learns a non-autoregressive NeMF prior from AMASS hand-bearing subsets, modeling local finger motion; root/translation/shape are optimized separately. Supplementary gives T=128, local 6-D rotations and a 1024-D latent. Left motion is reflected to right convention by x-position/translation flips and y/z axis-angle sign changes ([`amass.py` L96–139](https://github.com/enesduran/HMP/blob/35d799f76b2b2bc1d1e945117b021014b099e7e6/src/datasets/amass.py#L96-L139)). It reports local/root-aligned and occlusion benefits and has the most complete training release among candidates, but is single-hand, not image-conditioned, and warns of interacting-hand upstream failures.

**Hand4Whole++.** `[VERIFIED]` CHAM lets hand features condition the whole-body stream’s SMPL-X wrist; Table 2 shows direct wrist copying worsens body/hand error whereas CHAM improves it. `[VERIFIED]` Its final hand path creates a canonical MANO mesh, rigidly aligns via wrist+four MCP correspondences, scatters vertices, then smooths seams. Thus topology can look unified while parameterization is not. Its wrapper explicitly flips left crops, mirrors vertices/translation, changes y/z axis-angle signs and applies side-specific pose means ([`wilor.py` L79–124](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE/blob/f81d35ddd2b74206c40142243eb62b6d64ce0d65/common/nets/wilor.py#L79-L124)). This supports BACKUP and the need for explicit parity, but is not temporal and cannot be adopted literally.

## 5. Information-independence/double-counting audit

| Candidate | Evidence shared with DexAvatar | New information/bias | What it can plausibly correct | What it may only smooth/repeat | Double-counting assessment |
|---|---|---|---|---|---|
| HandFlow | Default frontend is frozen HaMeR crop/features/keypoints/confidence, overlapping DexAvatar’s HaMeR evidence. | Learned full-window MANO distribution from DexYCB/HOT3D; temporal bidirectional attention; missing-condition prior. | `[INFERRED]` Local ambiguity when neighboring frames reveal complementary geometry. | Systematic HaMeR side/orientation error; confidence effect dominated by translation. | High upstream correlation, meaningful new temporal prior. HaMeR and HandFlow are not independent votes.
| HaPTIC | HaMeR backbone and same local-pose recipe. | Temporal attention plus full-frame context and global trajectory supervision. | Trajectory/depth and some occluded 2D alignment. | Shared local HaMeR finger error; local HO3D quality not improved. | Very high local double-counting; new evidence is mainly metric-removed trajectory.
| Dyn-HaMR | HaMeR is one initializer. | Multiple trackers, explicit identity, HMP prior, interaction constraints, SLAM. | Local interaction plausibility, invalid-span infilling, track continuity. | Wrong initial track/pose, long-horizon drift; world-only gain. | Medium; more independent cues but many coupled components.
| HMP | Image/keypoint observations can overlap upstream evidence. | Mocap-derived local motion manifold independent of HaMeR pseudo-targets. | Local pose during occlusion. | Smooth but wrong mode in interacting signs; no body/cross-hand evidence. | Low prior correlation, high observation correlation.
| Hand4Whole++ | SMPLer-X-family whole-body source overlaps DexAvatar body initialization. | Hand features condition body/wrist; hand-only estimator is a distinct per-frame source. | Wrist/forearm consistency. | Per-frame occlusion; direct wrist/vertex transfer can worsen errors. | Medium; useful only for BACKUP here.

`[VERIFIED]` HandFlow adds temporal inductive bias but not independent monocular evidence by default. `[INFERRED]` The scientific intervention must concern how reliability/missingness changes the local state across time, not count agreement with HaMeR twice. `[VERIFIED]` HMP is the most independent motion prior, but Dyn-HaMR already combines per-hand HMP infilling with two-hand tracking; a simple HMP+track composition is inside prior art.

## 6. Training-data feasibility matrix

SGNify test GT is excluded from every training/tuning/model-selection role. No dataset bytes were opened.

| Dataset | RGB/video vs motion | Parametric supervision | Side/visibility/camera/time | Split/access/license | Feasible role | Status |
|---|---|---|---|---|---|---|
| **SignAvatars** (ECCV 2024) | 70K sequences, 153 signers, 8.34M frames/117h; RGB belongs to source datasets and must be reacquired ([project](https://signavatars.github.io/)). | Per-frame pseudo-SMPL-X, including 45-D hand blocks, intrinsics, 2D/predicted 2D, `left_valid`/`right_valid`; not mocap ([format](https://github.com/ZhengdiYu/SignAvatars#dataset-format)). | Side validity and camera fields; heterogeneous source FPS; uniform occlusion labels absent. | Annotation request/non-commercial; upstream licenses. Unified signer-disjoint provenance `[UNKNOWN]`. | Potential paired sign adaptation only after lawful video/annotation alignment; correlated pseudo-label risk. | Size/format `[VERIFIED]`; turnkey pairing/splits `[UNKNOWN]`.
| **How2Sign** (CVPR 2021) | >80h frontal/side synchronized RGB, depth and 2D; ~3h Panoptic detailed 3D ([official site](https://how2sign.github.io/)). | No native SMPL-X/MANO GT. | ~30fps, 11 signers, calibration/depth in subsets; no uniform MANO visibility. | CC BY-NC 4.0. Original split has signer-overlap/duplicate risks documented by paper. | Sign RGB conditioning/domain adaptation; not direct finger-rotation supervision alone. | `[VERIFIED]`.
| **Neural Sign Actors curated data** (CVPR 2024) | Project describes ~35K curated How2Sign sequences ([project](https://baltatzisv.github.io/neural-sign-actors/)). | Pseudo-SMPL-X fitted with whole-body/hand cues and temporal regularization; not mocap. | Derived timing; released visibility/camera/side-validity schema `[UNKNOWN]`. | Data link exists; repository, explicit license and authoritative split manifest not found. | Pseudo sign motion prior only until pairing/terms are verified. | Method/corpus `[VERIFIED]`; artifact contract `[UNKNOWN]`.
| **DexYCB** | 1,000 multiview RGB-D object-grasp sequences (~582K frames, 10 subjects) ([official](https://dex-ycb.github.io/)). | Calibrated 3D/MANO; HandFlow uses s0 right-hand crops. | Temporal/calibrated; object occlusion; selected use right-hand. | CC BY-NC 4.0; official s0. | Strong generic right-hand local/occlusion supervision; weak for left, two-hand, sign dynamics. | `[VERIFIED]`.
| **HOT3D** | ~833 min, ~1.5M multiview frames/3.7M images, head-mounted interaction ([official](https://facebookresearch.github.io/hot3d/)). | Accurate MANO pose/shape and objects. | 30fps, 19 subjects, both hands, calibrated views. | Registration/license; test GT private; HandFlow uses participant-held-out train subset. | Strong generic both-hand temporal supervision; domain differs from signing. | High-level `[VERIFIED]`; exact HandFlow split/preprocessing `[UNKNOWN]`.
| **ARCTIC** | ~2.1M synchronized RGB, 8 exo+1 ego, bimanual object interaction ([repo](https://github.com/zc-alexfan/arctic)). | SMPL-X body, MANO hands, object capture/fits. | Both hands, calibration, clips, subject split. | Gated/non-commercial and model licenses. | Bimanual/body-context auxiliary; not sign-specific. | `[VERIFIED]`.
| **InterHand2.6M** | Multiview RGB at 5/30fps, single/interacting hands ([official](https://mks0601.github.io/InterHand2.6M/)). | 3D joints + fitted MANO, calibrated/world coordinates. | Explicit sides and interaction. | Official splits; CC BY-NC 4.0; repo archived but readable. | Strong generic bimanual identity/local pose. | `[VERIFIED]`.
| **HaPTIC five-video mix** | ARCTIC-EGO/EXO, DexYCB, H2O, HO3D, InterHand plus ten image datasets ([§4.1, Appendix B](https://arxiv.org/html/2501.08329#S4.SS1)). | Heterogeneous MANO/3D/world supervision. | Mixed sides/cameras/FPS; sampling augmented up to 6fps. | Multiple licenses; no unified global identity audit. | Sufficient for a large generic temporal estimator; costly and not sign-domain. | Composition `[VERIFIED]`; cross-source leakage `[UNKNOWN]`.
| **AMASS GRAB/TCDHands/SAMP used by HMP** | Motion/parametric capture, no paired in-the-wild RGB. | Mocap-derived local hand/body parameters. | T=128 clips; no image confidence/occlusion. | Dataset registrations/licenses. | Motion prior only, not image-conditioned confidence learning. | `[VERIFIED]`.

Data conclusion: `[VERIFIED]` a released HandFlow checkpoint satisfies checkpoint-level feasibility without exact retraining. `[INFERRED]` Generic public/gated resources suffice to study local temporal pose and identity without test GT, but do not establish sign generalization. `[UNKNOWN]` No audited sign resource combines real RGB, mocap-quality both-hand SMPL-X/MANO, explicit visibility/confidence, calibration, and signer-disjoint splits in one documented package.

## 7. Fatal blockers vs non-fatal uncertainties

### 7.1 Feasibility gate

| Required condition | Original broad PRIMARY | Reduced PRIMARY | Evidence-based verdict |
|---|---|---|---|
| 1. Released checkpoint or realistically retrainable base | Pass at checkpoint level. | **PASS:** HandFlow inference checkpoint; no exact-retraining claim. | `[VERIFIED]` checkpoint, incomplete training release.
| 2. Valid unified SMPL-X output | Fail if root, MANO β, trajectory or candidate mesh are included. | **CONDITIONAL PASS:** only 15 local rotations/side; exact left convention must close. | Right state `[VERIFIED]`; left map `[UNKNOWN]`.
| 3. Honest both-hand identity | Fail for “one learned joint-bimanual model.” | **PASS WITH LIMIT:** two persistent independent side streams; no joint-model claim. | Dyn identity contract `[VERIFIED]`; exact crossings partly `[UNKNOWN]`.
| 4. Main benefit survives centering | Mixed because candidates emphasize world trajectory. | **PASS:** intervention restricted to local fingers. | `[VERIFIED]` metric linkage.
| 5. Clean isolation from DexAvatar | Too many coupled states in full candidate systems. | **PASS conceptually:** body/wrist/shape/camera/evaluator held fixed. | `[INFERRED]` isolation boundary, not implementation plan.
| 6. No SGNify test GT needed | Pass. | **PASS.** | `[VERIFIED]` alternative artifacts exist.
| 7. No fatal unreleased dependency | Fail if HandFlow retraining is prerequisite. | **CONDITIONAL PASS:** fixed checkpoint only; conversion evidence still required. | `[VERIFIED]` release limits.

### 7.2 Blockers removed from scope

- `[VERIFIED]` Joint learned bimanual claim: fatal because selected base is single-hand.
- `[VERIFIED]` Candidate MANO root/wrist: fatal because it is not SMPL-X wrist-local and hard overwrite is forbidden.
- `[VERIFIED]` MANO β transfer: fatal because MANO/SMPL-X shape bases differ.
- `[VERIFIED]` Global/camera/world trajectory as main contribution: fatal because isolated hand metrics center translation.
- `[VERIFIED]` MANO vertex scatter/stitch: fatal because final vertices are not regenerated by one SMPL-X parameter vector.
- `[VERIFIED]` Exact HandFlow retraining as prerequisite: fatal because training/end-to-end preprocessing/evaluation are not released.

### 7.3 Remaining uncertainties

| Uncertainty | Why non-fatal now | Becomes fatal if… |
|---|---|---|
| HandFlow left parity, pose mean, joint order, root/finger split | Right state is explicit and README acknowledges mirror-to-right; other official code shows a side-aware transform is possible in principle. | No authoritative conversion yields a left local state consistent with SMPL-X forward kinematics.
| Local gain specifically under sign blur/occlusion | Overall root-aligned/occlusion-bin results motivate a test, not success. | Only translation/smoothness changes while centered local vertices do not.
| Missing HandFlow training code | Fixed checkpoint can be a substrate. | Later contribution requires retraining/modifying the unreleased denoiser.
| Track identity through crossings/missing spans | Dyn provides a side/validity contract. | Identity cannot persist without GT or side swaps.
| SignAvatars pairing/splits | Generic checkpoint/data keep feasibility alive. | Sign-supervised research becomes essential but lawful paired signer-safe data are unavailable.
| Object-manipulation→sign domain gap | It is the falsifiable scientific uncertainty. | Checkpoint’s local prior is inappropriate and yields no centered benefit.
| Fraction of Dex error caused by unreliable evidence | Gate only needs a regime-specific question. | Regime is negligible in non-test sign data.

## 8. Novelty boundary against nearest prior works

### 8.1 What already exists

- `[VERIFIED]` **HandFlow** already provides conditional generative full-window MANO recovery, confidence masking, and overlapping inference. Replacing DexAvatar’s HaMeR estimate with its checkpoint is engineering integration, not novelty ([HandFlow §§3–4](https://arxiv.org/html/2607.11221)).
- `[VERIFIED]` **HMP** already provides latent optimization of a right-canonical local motion prior for unreliable/occluded monocular observations. Adding the same prior independently to both hands is not novel ([HMP §§3–4](https://openaccess.thecvf.com/content/WACV2024/html/Duran_HMP_Hand_Motion_Priors_for_Pose_and_Shape_Estimation_From_WACV_2024_paper.html)).
- `[VERIFIED]` **Dyn-HaMR** already combines two side tracks, per-hand HMP infilling, interaction constraints, and multi-stage fitting. “HMP twice + tracking + penetration” is inside the nearest-prior envelope ([Dyn-HaMR §3](https://arxiv.org/html/2412.12861#S3)).
- `[VERIFIED]` **HaPTIC** already adapts a strong image hand estimator temporally with full-frame context and mixed video/image training. A temporal attention wrapper around HaMeR alone is not novel ([HaPTIC §§3.3–3.4](https://arxiv.org/html/2501.08329#S3.SS3)).
- `[VERIFIED]` **Hand4Whole++** already conditions whole-body wrist prediction on hand features and shows direct wrist copying is unsafe. A hand-estimator wrist overwrite is non-novel and empirically risky ([Hand4Whole++ Table 2](https://arxiv.org/html/2603.14726#S4.SS2)).

### 8.2 Directions that do not clear the gate

| Direction | Classification | Rejection reason |
|---|---|---|
| Substitute HandFlow for HaMeR | Engineering integration | Existing checkpoint already performs the complete estimator task.
| Run a right-canonical estimator separately on native right and mirrored left | Engineering integration | Two single-hand streams are not learned joint-bimanual dynamics and do not establish identity through crossings.
| Add ordinary temporal smoothing | Engineering integration | Can improve acceleration/appearance while every centered frame remains geometrically wrong.
| Fine-tune on sign videos without a new causal mechanism | Domain adaptation | Dataset name is not novelty; pseudo labels may repeat upstream errors.
| Combine HandFlow and HMP as two priors | Module aggregation | Their motion-prior roles overlap; Dyn-HaMR already contains HMP-based two-track infilling.
| Copy MANO root or scatter MANO vertices into SMPL-X | Incompatible integration | Violates local/global wrist semantics or one regenerable SMPL-X parameterization.
| Improve SLAM/world trajectory for hand metrics | Metric-misaligned integration | Pure translation is `REMOVED BY CENTROID ALIGNMENT`.

### 8.3 Conditional research gap

`[INFERRED]` No audited release demonstrates a **side-stable temporal intervention whose controlled output is only the two metric-preserved local SMPL-X finger states while the whole-body wrist chain remains fixed**, evaluated through region-centered SMPL-X vertices in sign-language motion. The works cover subsets: HandFlow has uncertainty-aware generation but no closed left/unified-SMPL-X contract or two-hand identity; Dyn-HaMR has identity/interaction optimization but emphasizes MANO/world motion; HMP has a local prior but is right-only; HaPTIC emphasizes trajectory; Hand4Whole++ has body–wrist context without time and changes final vertices outside one parameter vector.

`[INFERRED]` The material boundary is **state-selective uncertainty**, not simply the sign dataset. HandFlow denoises pose, root translation, and shape together, while its confidence ablation attributes most benefit to translation. Official TR-V2V removes translation but retains local articulation. None of the five releases establishes that reliability/missingness selectively improves the two persistent-side local finger states without importing hand-only root, shape, or mesh changes into the SMPL-X chain.

This is not yet a novelty claim for a method. Applying the selected base alone fails novelty. A later contribution would need a genuinely new temporal, two-side-identity-aware uncertainty or state-consistency intervention within this boundary. Step 3 intentionally defines neither its architecture nor objective.

`[UNKNOWN]` Publication novelty cannot be certified by absence in a targeted five-candidate audit. The gap is nevertheless sufficiently precise to support one falsifiable scientific question; if the next formulation reduces to module substitution, the direction must be rejected.

## 9. Exactly one falsifiable research hypothesis

> **Trong các frame ký hiệu có quan sát tay bị thiếu hoặc có độ tin cậy thấp nhưng còn ngữ cảnh temporal lân cận, sử dụng suy luận temporal sinh có conditioning theo trạng thái confidence/missing để ước lượng, cho từng side-track vật lý cố định, hai chuỗi 15 local finger-joint rotations tương thích SMPL-X sẽ giảm centroid-aligned LHand/RHand TR-V2V và phần lỗi của các hand vertices còn được giữ trong UBody(-F) so với DexAvatar, khi giữ cố định body pose, wrist/root orientation, SMPL-X shape, camera, translation, upstream per-frame observations và official evaluator.**

The intervention is the temporally conditioned estimate of the two side-indexed 45-D local states. The comparator is released DexAvatar, including class-`0` cache behavior. It is falsified if centered hand-region error is not lower in the stated unreliable-observation regime when the listed baseline states/evaluator are fixed—even if visual smoothness, world trajectory, or translation improves. `[INFERRED]` This is metric-aligned and testable; it predicts no numerical effect size.

## 10. Technical base and optional auxiliary mechanism

**Technical base: HandFlow whole-window generative MANO-state inference.**

`[VERIFIED]` It is selected because it uniquely combines a released checkpoint, full-window non-causal inference, explicit unreliable-observation conditioning, and reported local/root-aligned evidence in addition to trajectory evidence. Only its finger45 state is admitted. MANO β, root/global orientation, translation, camera/world motion, and vertices are excluded. Selection is checkpoint/substrate-level and does not claim exact retraining.

**Optional auxiliary mechanism: Dyn-HaMR fixed side-indexed track identity and observation-validity contract.**

`[VERIFIED]` Dyn-HaMR binds observations to persistent left/right tracks and distinguishes visible/invalid spans before per-hand infilling ([§3.1 and Appendix A](https://arxiv.org/html/2412.12861#S3.SS1)). Only this discrete contract is selected. Its SLAM, world trajectory, full optimizer, HMP latent, biomechanics, penetration, root, camera, and MANO mesh are excluded. It supplies no second pose to average and does not turn two streams into joint-bimanual modeling.

Candidate rejection rationale:

- **HaPTIC:** artifacts are comparatively complete, but demonstrated gains are trajectory/2D-heavy; local HO3D PA pose is slightly worse than HaMeR, and left local-SMPL-X export is unverified.
- **Dyn-HaMR as base:** local ablations and identity are useful, but full-system contribution is world-space MANO+SLAM with many coupled, metric-removed components and paper/release mismatches.
- **HMP:** strongest reproducible local prior, but right-canonical, single-hand, not image-conditioned, and already used in Dyn-HaMR’s two-track prior-art envelope.
- **Hand4Whole++:** strongest BACKUP evidence, but per-frame and its literal final mesh uses inadmissible MANO vertex scatter.

No second auxiliary is selected. No pose/rotation/vertex averaging, hard wrist overwrite, or mesh stitching is part of this substrate decision.

## 11. Final decision: GO WITH SCOPE REDUCTION

**`GO WITH SCOPE REDUCTION`**

`[INFERRED]` PRIMARY passes only after removing states and claims unsupported by artifacts. It does not pass as broad joint-bimanual world-space recovery. BACKUP is not activated, but remains fallback if authoritative left-local conversion or centered-local benefit cannot be established without test GT.

| Retained scope | Removed scope | Reason |
|---|---|---|
| Persistent physical left/right labels | One learned joint-bimanual-model claim | No selected base supplies it.
| 15 verified local finger rotations per side | MANO root/global orientation | Not SMPL-X wrist-local; no hard overwrite.
| Temporal context around missing/low-confidence observations | Pure smoothing | Smoothness alone is not official metric improvement.
| Explicit valid/missing state per side | Global/camera/world trajectory and SLAM | Main benefit removed by centering.
| Frozen body pose, wrist chain, SMPL-X β, camera, translation, upstream observations | MANO β, candidate vertices, mesh stitch/scatter | Preserve one regenerable SMPL-X parameter vector.
| Locked UBody(-F), LHand/RHand and class-`0` rules | Any evaluator/frame/mask/alignment/aggregation change | Infrastructure is locked.
| Released artifacts and non-test data | SGNify test GT for training/tuning/calibration/selection | Test GT is evaluation-only.

Gate interpretation:

- `[VERIFIED]` **Feasibility:** a fixed checkpoint-level base exists; exact retraining does not.
- `[INFERRED]` **Compatibility:** right local state is identifiable; left has a plausible but not authoritative conversion path. Compatibility is conditional.
- `[VERIFIED]` **Metric leverage:** local articulation survives centering; pure translation does not.
- `[INFERRED]` **Isolation:** changing only side-indexed local fingers while freezing other states defines a clean causal boundary.
- `[INFERRED]` **Novelty:** a narrow gap remains, but base substitution alone is not novel.
- `[UNKNOWN]` **Outcome:** no score was reproduced or forecast.

## 12. Unresolved evidence required before method formulation

These are evidence requirements, not an implementation or experiment plan.

1. **Authoritative HandFlow handedness contract.** `[UNKNOWN]` Official documentation/source or author confirmation is needed for checkpoint canonical side, left mirroring, inverse parity, pose means, and equivalence of README mirroring to demo `--side left`.
2. **Exact 45-D order/frame mapping.** `[UNKNOWN]` A source-level table is needed for HandFlow’s 15 MANO joints versus DexAvatar SMPL-X hand blocks, including local parents, flat order, axis-angle convention, and pose mean. Equal dimensions are insufficient.
3. **Checkpoint identity.** `[UNKNOWN]` An immutable model card/hash is needed to link checkpoint and normalization statistics to paper split, frontend, T=16, and side assumptions.
4. **Local—not translation-only—robustness.** `[UNKNOWN]` Primary evidence is needed that separates centered/local finger error from root translation in missing, blurred, hand–hand-occluded, and hand–body-occluded intervals. Current confidence ablation chiefly validates translation.
5. **Non-test sign-regime prevalence.** `[UNKNOWN]` A provenance-safe non-test source is needed to establish how often sign sequences exhibit missing detections, low confidence, side ambiguity, and occlusion. This report does not assume dominance in SGNify.
6. **SignAvatars pairing/split provenance.** `[UNKNOWN]` Official manifests are needed for RGB joins, FPS/timestamps, camera convention, validity semantics, and signer-disjoint splits that do not inherit How2Sign leakage.
7. **Dyn-HaMR identity parity.** `[UNKNOWN]` The exact tracker/submodule version, crossing side-swap policy, and confidence handling after interpolation remain unverified.
8. **Licenses.** `[UNKNOWN]` HaPTIC repository terms, HandFlow checkpoint terms, and combined MANO/SMPL-X/sign-dataset permissions must be explicit for the intended setting.
9. **Novelty beyond replacement.** `[UNKNOWN]` The future intervention must be distinguishable from HandFlow cmask/full-window flow, HMP infilling, Dyn per-side prior optimization, and ordinary domain adaptation; otherwise this gate collapses to no-go.
10. **Backup evidence if conditional gate fails.** `[VERIFIED]` Hand4Whole++ supports body-conditioned wrist relevance and rejects direct copying; `[UNKNOWN]` a temporal, fully SMPL-X-parametric shoulder–elbow–wrist–finger treatment is still absent. BACKUP is not formulated here.

## 13. Source manifest

### 13.1 Supplied artifacts and integrity

| Artifact | Inspection record |
|---|---|
| `DexAvatar_Baseline_and_TR-V2V_Evaluation_Dossier(2).md` | SHA-256 `715e36be9bf892386f78fa2833c981b4e6485b74cca4a07afa685e8d179d44b5`; read in full.
| `DexAvatar_Step2_Bottleneck_Prioritization_and_Targeted_Literature_Review(2).md` | SHA-256 `f80d840fd5b4d1595bdbd9e67fffac795571bb8e82a74119aefced8bc6f58a9f`; read in full and re-audited.
| `Đã dán markdown (1)(1).md` (Step 3 specification) | SHA-256 `abfbd23e0e84aa11ccc8ae898e19d057a97727d2826436c1efb26fe78f6fc1f8`; read in full.
| `Kundu et al. - 2025 - DexAvatar 3D Sign Language Reconstruction with Hand and Body Pose Priors(1).pdf` | Main+supplementary, PDF pages 1–21, read in full during Step 1; relevant method/failure claims rechecked against its indexed dossier.
| `evaluate_new_fitting(2).py` | Static inspection only: `transl_point_error` L159–169; topology L356–370; class/regions L380–395; aggregation L432–461; CLI/assets L479–567. Not executed.

### 13.2 Repositories

All accessed **2026-08-26**.

| Repository | Branch / commit | Files inspected |
|---|---|---|
| [DexAvatar](https://github.com/kaustesseract/DexAvatar) | `main` / `a0dfd427f60f5811aadb35c8657b3856d47f56b5` | `README.md` full; `data_parser.py` partial L120–203, 249–346, 397–425, 445–650; `main.py` L129–162; `fit_single_frame.py` L223–243, 376–503, 611–660. Full Step 1 source manifest covers all baseline entry/config/fitting/prior/export files.
| [HandFlow](https://github.com/mxxu00/HandFlow) | `main` / `67fa7df536db233408fe6270ca5d2de28d5959c3` | `README.md` full; `scripts/demo.py` relevant CLI/window/side path; `utils/online_hamer.py` L195–255, 373–392; `utils/mano_utils.py` relevant full path; preprocessing helper presence. Training/full data builder/evaluation unavailable.
| [HaPTIC](https://github.com/JudyYe/haptic) | `main` / `f9362c1bdf2c1ea2bfa695be2d4e6f362371e7df` | `README.md` full; `haptic/datasets/seq2clip.py` L97–224; `demo.py` L342–394; root license inventory.
| [Dyn-HaMR](https://github.com/ZhengdiYu/Dyn-HaMR) | `main` / `fa9cd7412c205fd15ee4139c8caacf79bf6167e6` | `README.md` full; `dyn-hamr/confs/config.yaml` full; `dyn-hamr/data/dataset.py` L213–305; dependency manifest. Missing tracking submodule **NOT INSPECTED**.
| [HMP](https://github.com/enesduran/HMP) | `main` / `35d799f76b2b2bc1d1e945117b021014b099e7e6` | `README.md`+license full; `src/datasets/amass.py` L96–139; documented training/fitting/preprocessing paths.
| [Hand4Whole++](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE) | `main` / `f81d35ddd2b74206c40142243eb62b6d64ce0d65` | `README.md`+license full; `main/model.py` L42–126, 160–222; `common/nets/wilor.py` L79–124.

### 13.3 Papers and supplementary

1. Mingxi Xu, Bowen Duan, Yi Gu, Zhengyang Shen, Renjing Xu, Yutao Yue. [*HandFlow: Fully Generative 4D Hand Recovery with Flow Matching*](https://arxiv.org/html/2607.11221), arXiv:2607.11221v1, 2026. Main §§1–5, Tables 1–4, Appendices A–G. ArXiv says “Journal: TOG”; publisher version `[UNKNOWN]`.
2. Yufei Ye, Yao Feng, Omid Taheri, Haiwen Feng, Shubham Tulsiani, Michael J. Black. [*Predicting 4D Hand Trajectory from Monocular Videos*](https://arxiv.org/html/2501.08329), arXiv 2025 (HaPTIC). Main §§3–4, Appendices A–C, Tables 5–6.
3. Zhengdi Yu, Stefanos Zafeiriou, Tolga Birdal. [*Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera*](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_Dyn-HaMR_Recovering_4D_Interacting_Hand_Motion_from_a_Dynamic_Camera_CVPR_2025_paper.html), CVPR 2025; [supplementary HTML](https://arxiv.org/html/2412.12861). Main §§3–5 and Appendices A–C.
4. Enes Duran, Muhammed Kocabas, Vasileios Choutas, Zicong Fan, Michael J. Black. [*HMP: Hand Motion Priors for Pose and Shape Estimation from Video*](https://openaccess.thecvf.com/content/WACV2024/html/Duran_HMP_Hand_Motion_Priors_for_Pose_and_Shape_Estimation_From_WACV_2024_paper.html), WACV 2024. Main and supplementary read for representation/training/handedness/evaluation.
5. Gyeongsik Moon. [*Enhancing Hands in 3D Whole-Body Pose Estimation with Conditional Hands Modulator*](https://openaccess.thecvf.com/content/CVPR2026/html/Moon_Enhancing_Hands_in_3D_Whole-Body_Pose_Estimation_with_Conditional_Hands_CVPR_2026_paper.html), CVPR 2026; [supplementary HTML](https://arxiv.org/html/2603.14726). Main §§3–5, Table 2, supplementary.

### 13.4 Dataset/model documentation

- [SignAvatars project](https://signavatars.github.io/), [ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/653_ECCV_2024_paper.php), [repository](https://github.com/ZhengdiYu/SignAvatars).
- [How2Sign](https://how2sign.github.io/); [Neural Sign Actors](https://baltatzisv.github.io/neural-sign-actors/) and [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/html/Baltatzis_Neural_Sign_Actors_A_Diffusion_Model_for_3D_Sign_Language_CVPR_2024_paper.html).
- [DexYCB](https://dex-ycb.github.io/), [HOT3D](https://facebookresearch.github.io/hot3d/), [ARCTIC](https://github.com/zc-alexfan/arctic), [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/), [AMASS](https://amass.is.tue.mpg.de/).
- All pages accessed 2026-08-26. MANO/SMPL-X assets remain separately licensed and were not opened/executed.

### 13.5 Not inspected / not executed

- SGNify test GT: **NOT INSPECTED** and unused.
- Candidate checkpoints, runtime outputs, training logs, datasets, evaluator: **NOT EXECUTED**.
- HandFlow complete training, end-to-end data construction, official evaluation: **NOT AVAILABLE / NOT INSPECTED** at audited commit.
- Dyn-HaMR missing tracker submodule: **NOT INSPECTED**.
- HaPTIC root license: **NOT FOUND**; reuse terms `[UNKNOWN]`.
- Neural Sign Actors repository, schema, explicit license: **NOT FOUND / NOT INSPECTED**.
- Every number is a reported primary-paper result, not a reproduced result.

### 13.6 Integrity check

- No evaluator, frame, region, alignment, aggregation, or test-GT exploit is used.
- Translation/world trajectory is separated from centered local geometry.
- No MANO mesh is treated as final SMPL-X; no wrist overwrite or rotation/vertex averaging is proposed.
- Independent side streams are not called learned joint-bimanual modeling.
- Fatal conventions are exposed as unknown, not guessed.
- No architecture, complete objective, code, pseudocode, implementation plan, experiment plan, score forecast, or reproduction claim is included.

STEP 3 COMPLETE — GO WITH SCOPE REDUCTION; REVISED PRIMARY HYPOTHESIS READY FOR REVIEW BEFORE METHOD FORMULATION.
