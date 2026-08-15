# Phase 3 — Failure-Mode and Research-Gap Analysis

**Target:** monocular RGB video → accurate signing SMPL-X reconstruction  
**Primary benchmark:** SGNify TR-V2V — upper body excluding face, left hand, right hand  
**Evidence cut-off:** 10 August 2026  
**Scope constraint:** this report diagnoses failures and states falsifiable research hypotheses. It does **not** propose an architecture or final method.

---

## Executive conclusion

### What is established

- **FACT:** DexAvatar reports 30.13 / 13.53 / 13.08 mm TR-V2V for upper body excluding face / left hand / right hand on SGNify. It remains the lowest result found under an explicitly documented translation-only protocol for the upper body. SOKE reports lower hand errors, 10.55 / 8.94 mm, on a nearly inherited SGNify table, but does not restate the translation-only alignment. Thus the residual error is real, but the exact current hand leaderboard is not protocol-secure.
- **FACT:** DexAvatar's released fitting is driven by Sapiens and HaMeR observations, SMPLer-X/HaMeR initialization, sign-specific body/hand latent priors, a previous-frame body term, range-of-motion terms, and collision repulsion. It has no explicit visibility state, calibrated observation uncertainty, intended-contact label, or hand temporal term. Its released normalized HaMeR 3D hand-depth term has zero weight.
- **FACT:** its primary quantitative benchmark contains 57 isolated DGS signs and one signer; robustness to blur and self-occlusion is qualitative; temporal fidelity, contact correctness, penetration, biomechanics, and cross-signer generalization are not quantitatively evaluated.
- **FACT:** the paper's own body-prior ablation shows mild over-regularization: `BPf+bio` is worse than `BPf` on all four reported body regions. Filtering/correcting prior data helps, showing that training-label quality matters.
- **FACT:** SGNify ground truth is a personalized SMPL-X mesh fitted by MoSh++ to Vicon markers, not a direct per-vertex scan at every video frame. DexAvatar reports occasional collapsed fingers and irregular knuckle spacing in these reference meshes. This is evidence of local annotation defects, not evidence that all SGNify ground truth is inaccurate.
- **FACT:** SGNify's supplement includes TUCH-derived penetration and contact-normal terms. DexAvatar's released objective does not preserve an explicit intended-contact state; its collision term is repulsive. Neither line of work reports sign-specific contact precision/recall.

### What is not established

- The literature does **not** identify what fraction of DexAvatar's remaining error comes from 2D observations, initialization, wrist orientation, fingers, camera, priors, or optimization. Most components lack causal ablations.
- Aggregate TR-V2V does **not** establish robustness to occlusion or blur, temporal fidelity, contact correctness, biomechanical validity, semantic preservation, or uncertainty calibration.
- Selected qualitative examples do **not** establish that DexAvatar is robust to motion blur or occlusion.
- The evidence does **not** justify calling SGNify ground truth globally inaccurate, nor does it justify treating it as exact surface truth.

### Causal diagnosis

The most defensible error chain is:

> image degradation or visibility loss → biased/missing 2D and initialization evidence → a deterministic, strongly regularized latent fit selects one depth/interaction mode → fixed camera and short-memory constraints limit correction → region-averaged per-frame metrics hide when and why the result fails.

This chain is an **INFERENCE** from the released dependencies and missing ablations. It must be tested through controlled observation, visibility, camera, prior, and temporal interventions before a method is designed.

### Top-five gaps

1. **Occlusion gap** — no sign-specific, visibility-stratified 3D evidence for hand–hand, hand–body, or long-duration occlusion.
2. **Uncertainty gap** — direct methods return one pose despite depth and articulation ambiguity; detector confidence is not calibrated 3D uncertainty.
3. **Evaluation gap** — the leaderboard and target metric are not protocol-stable and do not measure timing, contact, robustness, or semantic damage.
4. **Temporal gap** — local smoothness is measured more readily than motion fidelity; rapid hand motion and recovery after missing evidence remain untested.
5. **Interaction/contact gap** — intended touch, surface/depth ordering, persistence, and separation timing are neither reconstructed nor scored directly.

These are gaps, not proposed mechanisms. The evidence does not yet prove that any one will yield the largest aggregate TR-V2V improvement.

---

## 1. Epistemic and scoring conventions

### Labels

- **FACT:** explicitly reported in a paper, supplement, official code, or benchmark definition.
- **EVIDENCE:** one or more facts that bear on a causal claim but do not prove it.
- **INFERENCE:** the most plausible explanation consistent with current evidence.
- **HYPOTHESIS:** a prospective, falsifiable claim.
- **SPECULATION:** a possibility with weak or no direct evidence. Speculation is not used to rank a gap highly.
- **NOT MEASURED:** the required experiment or annotation was not found.

### Gap scores

Each gap is scored from 1 (low) to 5 (high) on six axes. Scores are ordinal judgments, not measured probabilities.

- **E:** evidence that the gap exists.
- **R:** plausible share of remaining error that is unresolved by current methods.
- **M:** expected impact on the requested geometric/robustness metrics if the gap is resolved.
- **N:** novelty potential in sign-specific 3D reconstruction, after accounting for adjacent prior art.
- **F:** implementation and evaluation feasibility.
- **P:** publication potential if supported by decisive experiments.

The reported total is an equal-weight sum out of 30. Ties are broken by stronger direct sign evidence and then feasibility. Because causal error shares have not been measured, differences of one point should not be over-interpreted.

---

## 2. The 19 requested failure modes: evidence and causal diagnosis

### 2.1 Observed phenomenon → evidence → cause → affected module → missing information

| ID | Failure | Observed phenomenon | Evidence and status | Likely technical cause | Affected module(s) | Information currently missing |
|---:|---|---|---|---|---|---|
| F1 | Inaccurate 2D keypoints | Fingertips, wrists, or arms can be misplaced, missing, merged, or assigned to the wrong hand, especially when the hands are small, blurred, or overlap. A pipeline-wide error contribution is **not measured**. | **FACT:** DexAvatar directly fits Sapiens and HaMeR 2D observations and uses their confidence. Some frames with missing HaMeR/SMPLer-X files are dropped. No detector ablation is reported. **EVIDENCE:** replacing older hand estimation with WiLoR accounts for most of SOKE/Tamaththul3D's published hand improvement. | Finite crop resolution; blur; self-occlusion; left/right identity ambiguity; detector domain shift; confidence that is not calibrated to 3D error. Strong initialization and reprojection weights can preserve a bad target. | Sapiens/HaMeR observations; hand detector/crop; SMPLer-X/HaMeR initialization; reprojection loss; hand decision logic. | Ground-truth 2D keypoints with visibility; per-joint detector error; confidence calibration; causal oracle-2D ablation; error by crop size and overlap. |
| F2 | Depth ambiguity | Distinct 3D hand, elbow, or torso configurations can produce nearly the same frontal 2D projection; front/back ordering can be wrong while reprojection is low. Direct failure frequency is **not measured**. | **FACT:** monocular fitting is under-constrained. DexAvatar computes a wrist-relative HaMeR 3D term but its released weight is zero. Camera and global variables are largely fixed. **EVIDENCE:** ProHMR, ScoreHMR, MaskHand, and related work explicitly represent multiple plausible 3D solutions. | Perspective projection removes depth; weak-perspective/full-perspective parameter coupling; correlated kinematic compensation; a unimodal deterministic fit selects a prior-favoured mode rather than representing alternatives. | Reprojection objective; body/hand priors; camera; optimizer; initialization. | Depth-axis joint error; posterior/multimodal diagnostics; oracle multi-view or mocap-depth ablation; calibration of competing 3D hypotheses. |
| F3 | Wrist orientation | Fingers may look locally plausible while the global hand orientation, forearm twist, or wrist attachment is inconsistent with the body. | **FACT:** SignHPoser models local finger joints, not the global wrist. Its supplement says wrist rotations could not be transferred accurately during retargeting because SMPL-X and MANUS T-poses/bone rolls differ. **EVIDENCE:** Hand4Whole++ identifies naive specialist-hand/whole-body wrist integration as anatomically problematic; Tamaththul3D reports qualitative kinematic improvement from wrist alignment. DexAvatar has no wrist-specific metric. | The wrist is estimated through the body pathway while fingers come from a separate hand pathway; rotation conventions and bone rolls differ; 2D keypoints weakly constrain axial forearm rotation. | SMPLer-X body initialization; SignBPoser arm/wrist pose; MANO-to-SMPL-X integration; camera; kinematic chain. | Wrist geodesic error; forearm twist error; hand-normal/orientation error; a factorial oracle-wrist versus oracle-finger intervention. |
| F4 | Finger articulation | Residual hand error remains roughly 9–14 mm in the strongest SGNify-table results; visually plausible compact hands can still have wrong individual joint angles or spacing. | **FACT:** DexAvatar reports 13.53/13.08 mm; SOKE reports 10.55/8.94 mm under an unverified alignment convention. SignHPoser is trained on fingerspelling motion from eight signers and corrects implausible configurations. No per-finger or handshape-stratified result is reported. | Small image footprint; self-occlusion; coupled MANO/SMPL-X pose space; limited handshape coverage; prior pulls rare lexical configurations toward common training modes; 2D keypoints omit surface and depth detail. | Hand crop/estimator; SignHPoser; MANO/SMPL-X conversion; hand reprojection and initialization losses. | Per-joint and phalanx-level errors; handshape/phonological strata; pixel-size and visibility strata; oracle-finger intervention with wrist held fixed. |
| F5 | Hand–hand occlusion | When hands overlap, the hidden fingers and relative depth/identity may be wrong even when the visible silhouette appears plausible. DexAvatar shows selected qualitative occlusion cases, not a failure rate. | **FACT:** DexAvatar has no explicit visibility or occlusion state. **EVIDENCE:** Tamaththul3D discloses severe inter-hand occlusion failures in WiLoR; IntagHand, MaskHand, and Dyn-HaMR show that interaction/occlusion needs dedicated treatment in adjacent hand reconstruction. **NOT MEASURED:** SGNify TR-V2V by inter-hand overlap. | Evidence from the hidden hand disappears; two independent hand crops lose relative geometry; detector identity can swap; the prior may invent a common pose without knowing which surfaces are visible. | Hand detection and assignment; HaMeR/WiLoR-class initialization; SignHPoser; interaction/collision objective; temporal propagation. | Visibility masks; overlap ratio; hand identity through time; depth ordering; occlusion-duration labels; error conditioned on these variables. |
| F6 | Hand–body occlusion | A hand crossing the torso, face, neck, or opposite arm may be reconstructed on the wrong side of the surface, with wrong location or a missing hand. This is not quantitatively isolated. | **FACT:** direct methods use detector confidence or fallback rather than a latent hand–body visibility model. DexAvatar's released objective has collision repulsion but no intended hand–body contact label. **EVIDENCE:** sign papers identify hand–body interaction as important; TUCH shows contact supervision is useful for general HMR. | The body both occludes the hand and supplies a plausible contact surface; 2D overlap alone does not determine whether the hand is in front, touching, or behind. Repulsion cannot infer intended contact or visible-side ordering. | Body/hand observations; depth/camera; collision term; body and hand priors; initialization. | Body-part segmentation and depth order; true contact labels; hand–body signed distance; error by occluding body region and visibility fraction. |
| F7 | Long-duration occlusion | During multiple consecutive missing/ambiguous frames, a deterministic fit can drift to a prior mode, freeze, swap identity, or recover with a discontinuity. No direct sign benchmark measures this. | **FACT:** DexAvatar has previous-frame regularization only for body pose and no hand temporal term. One-hand detection may fall back to the previous detection; frames missing required estimates can be dropped. **EVIDENCE:** GLAMR and Dyn-HaMR explicitly evaluate/generate motion through occlusion in adjacent tasks. | First-order memory contains no long-horizon evidence; accumulated bias and mode collapse grow with gap length; no explicit recovery or uncertainty mechanism identifies when the trajectory is unsupported. | Detection/fallback; temporal term; hand/body priors; optimizer; frame selection. | Contiguous-gap length; visibility through the gap; drift and recovery latency; post-gap identity and phase continuity; real long-occlusion 3D ground truth. |
| F8 | Motion blur | Fast fingers and wrists lose edge/keypoint evidence; direct methods may output a plausible but wrong hand or fail to detect it. DexAvatar's claimed robustness is qualitative. | **FACT:** SGNify identifies motion blur as a core sign-capture problem. DexAvatar shows three selected blur examples/supplementary video without 3D robustness numbers or reported blur magnitude. **NOT MEASURED:** error-versus-blur curves. | Temporal exposure integrates multiple poses; single-frame estimators see a non-physical average; detector confidence may not reflect directional blur; priors prefer static/common shapes. | RGB observations; hand detector/crop; Sapiens/HaMeR; initialization; priors. | Camera exposure and motion kernel; blur severity; true intra-exposure motion; calibrated confidence; controlled same-GT blurred inputs. |
| F9 | Camera estimation | Incorrect focal length, scale, principal point, or camera orientation can be compensated by body depth/shape/pose, biasing upper-body and wrist geometry. Its share of DexAvatar error is unknown. | **FACT:** DexAvatar initializes camera intrinsics from SMPLer-X and keeps the camera matrix fixed in released fitting; a camera-initialization loss is constructed but not run. TR-V2V removes translation only, so rotation, scale, and articulation compensations remain. | Monocular camera–body ambiguity; cropped upper-body input; weak/full-perspective mismatch; fixed initialization prevents later evidence from correcting camera bias. | SMPLer-X initialization; projection; global orientation/translation; shape; body pose. | Calibrated SGNify camera parameters used in fitting; focal/rotation error; raw versus aligned metric decomposition; oracle-camera ablation. |
| F10 | Temporal inconsistency | Framewise hands can jitter or change articulation abruptly; conversely, strong smoothing can delay or attenuate true rapid motion. Aggregate per-frame TR-V2V can miss both. | **FACT:** DexAvatar uses a fixed-weight previous-body-pose Geman–McClure term and no final hand temporal term, sequence window, velocity, or acceleration model. **FACT:** SGNify's discussion states that per-frame V2V misses temporally important errors. **EVIDENCE:** Tamaththul3D reduces jerk by post-hoc smoothing without GT dynamic-fidelity evidence. | Independent/locally coupled fitting; noisy observations; fixed smoothness across slow and fast phases; no distinction between jitter and genuine high-frequency sign motion. | Temporal regularizer; framewise hand latent; optimizer schedule; initialization propagation. | GT velocity/acceleration; phase/event timing; spectral attenuation; lag; hand and wrist jitter separately; recovery after occlusion. |
| F11 | Contact ordering | Two distinct questions are unresolved: which surface is in front immediately around contact, and whether the temporal sequence is approach → touch → separate in the correct order. No direct paper scores either. | **FACT:** DexAvatar has no intended-contact state. **FACT:** SGNify includes self-contact/interpenetration terms, but no sign-specific contact event labels or contact-order metric. **INFERENCE:** 2D overlap plus repulsion cannot uniquely establish front/back or touch timing. | Monocular depth ambiguity; lack of surface-pair identity; no contact-state persistence; local optimization can switch depth modes or create/destroy contact at the wrong frame. | Depth/camera; body/hand priors; collision/contact terms; temporal model. | Contacting surface pairs; signed depth order; onset/offset frame; contact duration; approach/separation velocity; multi-view confirmation. |
| F12 | Interpenetration | Fingers can penetrate palms, the opposite hand, or body; a repulsive penalty can also separate a true contact. DexAvatar provides no quantitative penetration result. | **FACT:** released DexAvatar uses BVH-based collision/penetration repulsion. **FACT:** no penetration volume/depth or contact-recall ablation is reported. **EVIDENCE:** DexAvatar's supplement alleges finger–palm penetration in some SGNify reconstructions, but selected examples do not provide prevalence. | Approximate collision pairs; optimizer trade-off against strong 2D/init terms; repulsion lacks intended-contact identity; discrete collision geometry and local minima. | Collision loss; mesh geometry; optimizer weights; hand/body pose priors. | Maximum penetration depth/volume; collision pair types; false separation of true contacts; sensitivity to collision weight and initialization. |
| F13 | One-handed/two-handed ambiguity | The wrong active hand can be optimized while evidence for the relevant hand is suppressed; a passive hand may be geometrically or linguistically important despite little motion. | **FACT:** the release uses a precomputed clip class; class 0 is one-handed. The active side is selected by average Sapiens wrist motion. Near-ties under a 1.2 ratio become ambiguous and the downstream Boolean routes them to the left branch. Zeroing observations does not fully disable the shared body latent or strong initialization. No decision-maker ablation is reported. | Clip-level discrete categorization; motion magnitude is an imperfect proxy for dominance; ambiguous transitions and passive-hand roles; shared latent couples both arms. | Sign class mapping; hand decision maker; observation weights; body latent; optimizer variable selection. | Decision accuracy; frequency of near-ties; per-frame interaction state; passive-hand semantic role; error under oracle versus corrupted class/side labels. |
| F14 | Signer variation | Pose scale, limb proportions, handedness, articulation range, speed, coarticulation, and personal signing style can shift a sample away from learned priors. Cross-signer geometric performance is unknown. | **FACT:** SGNify's primary quantitative set has one signer. SignHPoser uses eight signers; the SignBPoser data scale/splits are not fully reported. DexAvatar lists broader signer/style coverage as future work. | Limited prior support; fixed/global range-of-motion assumptions; per-clip mean shape; signer-specific camera/crop and appearance; under-represented left-handed or atypical mobility patterns. | Shape estimation; SignBPoser/SignHPoser; detector; biomechanical constraints; temporal model. | Multi-signer accurate SMPL-X ground truth; signer demographics/handedness; leave-one-signer-out error; personalization versus universal-prior decomposition. |
| F15 | Domain shift | Performance may degrade across sign languages, cameras, clothing, skin tones, resolutions, continuous signing, and in-the-wild backgrounds. The degree is unknown. | **FACT:** priors/observations draw on ASL/Auslan/general HMR sources, quantitative evaluation is one DGS setup, and MM-WLAuslan evidence is qualitative. Cross-language exposure is suggestive but not a controlled generalization test. | Dataset-specific pose frequency, crop and camera statistics, linguistic inventories, background/appearance shift, pseudo-label pipeline bias, isolated-to-continuous motion shift. | All off-the-shelf observations; sign priors; hand decision logic; temporal assumptions; evaluation protocol. | Matched cross-domain GT; within-domain versus cross-domain comparison; worst-group results; covariate labels; continuous-sign evaluation. |
| F16 | Prior over-regularization | Rare or fast valid poses can be pulled toward the learned manifold, initialization, previous frame, or nominal joint ranges; results may look plausible but be wrong. | **FACT:** `BPf+bio` is worse than `BPf` on every reported body region; authors describe mild over-regularization. Released fitting uses strong initialization matching and fixed body-temporal weighting. No weight sweep in final fitting is reported. | Training distribution is narrower than valid signing; Gaussian latent penalty favours common modes; hard/soft range limits may reject signer-specific or rare configurations; fixed weights ignore evidence quality. | SignBPoser/SignHPoser latent losses; ROM/biomechanics; initialization loss; temporal loss. | Error versus latent likelihood/pose rarity; performance across prior weights; per-joint limit violations; fast/rare sign strata; oracle-observation interaction. |
| F17 | Noisy pseudo-ground truth | A learned prior or annotator can reproduce systematic errors from its fitted training labels, even when training loss is low. | **FACT:** SignBPoser is trained on pseudo-SMPL-X derived from SignAvatars/How2Sign. Filtering rotations improves downstream upper-body TR-V2V (`BPu` 34.06 → `BPf` 30.28 mm for UBody excluding face). **EVIDENCE:** this is direct evidence that label curation changes downstream geometry, but it does not identify all remaining noise. | Detector/fitter bias becomes the target distribution; filtering removes invalid samples but may also remove rare valid poses; no independent 3D truth distinguishes noise from diversity. | Body-prior training data; filtering; prior latent; downstream fitting. | Independent mocap validation of pseudo labels; label uncertainty; annotator disagreement; coverage lost by filtering; paired training with raw/filtered/high-quality labels. |
| F18 | Inaccurate SGNify ground truth | Some reference hands may have collapsed fingers or irregular knuckle spacing, so a plausible prediction can receive a worse vertex score. The prevalence and magnitude are unknown. | **FACT:** SGNify fits a personalized SMPL-X mesh to Vicon markers using MoSh++; it notes that reflective markers can influence contact-heavy motion and excludes face from evaluation because 27 markers are insufficient for face detail. **FACT:** DexAvatar reports selected implausible hand-reference examples. **EVIDENCE:** MoSh++ is validated against 4D scans in AMASS, but that does not quantify SGNify's per-frame finger error. | Sparse/noisy marker fitting; marker placement and soft-tissue motion; hand marker coverage; SMPL-X model and optimization limits; contact altered by physical markers; synchronization/retargeting error. | Benchmark annotation pipeline; personalized shape; MoSh++ pose fit; metric reference. | Held-out marker error; multi-view silhouette/landmark residual; repeated fitting uncertainty; per-finger reference quality; independent scan or expert audit; whether rankings change after excluding uncertain frames. |
| F19 | Limitations of TR-V2V | A low mean per-frame vertex error can coexist with wrong temporal evolution, contact, semantics, or rare catastrophic frames. Translation centering removes global translation error and the same-topology reference may itself be imperfect. | **FACT:** TR-V2V centres prediction and GT per frame and averages corresponding region vertices. SGNify explicitly says per-frame V2V is not ideal for sign language and gives an example where a few inaccurate frames damage recognition despite small overall error. Later papers mix TR-V2V, generic MPVPE, and PA-MPVPE labels. | Per-frame averaging ignores temporal order; means hide tails/signer/sign strata; no visibility/contact/semantic conditioning; alignment choices remove different errors; correlated vertices do not equal perceptual importance. | Evaluator, frame manifest, vertex subsets, aggregation and SOTA claims. | Public evaluator/manifest; confidence intervals/per-sign distributions; metric-to-recognition correlation; temporal/contact/occlusion metrics; raw/root/TR/PA decomposition. |

---

## 3. Falsifiable diagnostic hypotheses and decisive experiments

These hypotheses do not prescribe an architecture. Each asks whether a suspected source actually explains residual error. “Refute” denotes a concrete result that would make the hypothesis scientifically unattractive; thresholds should be pre-registered before viewing test results.

| ID | Falsifiable hypothesis | Metric expected to improve if true | Confirming/refuting experiment |
|---:|---|---|---|
| H1 | Detector/2D-observation error explains a substantial share of residual hand error, concentrated in low-resolution, blurred, and occluded joints. | Hand TR-V2V/MPVPE; wrist/finger MPJPE; occluded-joint PCK; robustness slope. | Project mocap joints into the synchronized RGB camera and run a 2×2 intervention: detector versus oracle 2D, and detector versus oracle initialization, with all other terms frozen. Stratify by visibility/crop size/blur. **Refute** if oracle 2D gives no statistically reliable reduction and the 95% upper bound is below 1 mm hand TR-V2V in every hard stratum. |
| H2 | Monocular depth ambiguity, not only 2D localization noise, is a major residual source during overlap and near-frontal articulation. | Depth-axis MPJPE; wrist-relative depth error; hand/body TR-V2V; depth-order accuracy; uncertainty calibration. | Hold 2D observations fixed and compare monocular fits with synchronized multi-view/mocap depth constraints; repeat fitting from diverse initial depth modes. **Refute** if oracle depth/multi-view evidence changes neither geometry nor mode variance by a practically meaningful pre-registered margin. |
| H3 | Wrist/forearm orientation error contributes independently of local finger error and propagates to both hand and upper-body vertices. | Wrist geodesic error; forearm twist; hand-normal error; hand and UBody TR-V2V. | Perform a 2×2 oracle intervention on wrist orientation and local finger articulation using GT SMPL-X parameters. Report isolated and interaction effects. **Refute** if correcting the wrist reduces neither hand nor upper-body error and wrist error is uncorrelated with both. |
| H4 | Residual hand error is dominated by specific finger/handshape strata rather than being uniform over the hand. | Per-finger MPJPE/MPVPE; fingertip error; handshape-conditioned TR-V2V; worst-decile error. | Partition signs by handshape and per-finger visibility; compute segment-level error and repeat the oracle-wrist/oracle-finger factorial. **Refute** if errors are statistically homogeneous across strata and oracle fingers yield negligible gain. |
| H5 | Inter-hand overlap causes a super-additive error increase beyond ordinary low keypoint confidence. | Visibility-stratified L/R hand TR-V2V; identity-switch rate; depth-order accuracy; contact F1. | Derive overlap/visibility masks from calibrated multi-view data or manual annotation; match non-overlap frames on crop size, speed, and confidence. Compare errors and run synthetic controlled occlusions. **Refute** if matched overlap has no residual effect and no duration interaction. |
| H6 | Hand–body overlap creates a distinct depth/contact failure not explained by generic hand visibility alone. | Hand TR-V2V; hand-to-body location error; signed-distance/contact F1; body-part depth-order accuracy. | Label occluding body part, visibility, and contact; compare matched hand–hand, hand–body, and no-overlap frames. Use multi-view evidence to establish side/order. **Refute** if hand–body category adds no error after visibility and speed are controlled. |
| H7 | Error and recovery discontinuity grow nonlinearly with consecutive missing-evidence duration. | Error-versus-gap AUC; end-of-gap TR-V2V; recovery latency; identity switches; velocity/acceleration error. | Mask hand observations for contiguous gaps of 1, 3, 5, 10, and 20 frames while retaining the same GT; test real occlusions separately. **Refute** if error remains flat with gap length and post-gap recovery is immediate and accurate. |
| H8 | Motion blur degrades reconstruction mainly through observation failure, with directional blur more damaging than equal-energy Gaussian noise. | TR-V2V degradation curve; keypoint error; detection recall; calibration error; high-speed-stratum MPJPE. | Apply physically parameterized motion kernels to SGNify frames while preserving GT, plus matched Gaussian corruption. Where possible, collect synchronized short-/long-exposure views. **Refute** if hand error and detection/calibration do not worsen with blur magnitude or direction after controlling for image energy. |
| H9 | Fixed camera error contributes materially to upper-body and wrist geometry despite translation alignment. | Calibrated camera rotation/focal error; raw and TR UBody MPVPE; shoulder/elbow/wrist MPJPE; reprojection. | Run identical fits with SMPLer-X camera, optimized camera, calibrated SGNify camera, and controlled camera perturbations. Keep shape and observations fixed. **Refute** if oracle calibration changes UBody TR-V2V by less than a pre-registered negligible margin and produces no pose compensation change. |
| H10 | Current smoothness reduces jitter but attenuates or delays genuine fast signing motion, especially in hands. | GT velocity/acceleration error; jerk; phase lag; event timing; spectral amplitude ratio; per-frame TR-V2V. | Evaluate no/weak/default/strong temporal weights on mocap sequences, binned by true angular speed; measure both jitter and fidelity. **Refute** if stronger smoothness lowers jitter without increasing lag, acceleration error, or high-frequency attenuation. |
| H11 | Contact transitions have elevated depth/order and timing error even when aggregate hand TR-V2V is modest. | Contact-pair F1; onset/offset error; duration error; signed depth-order accuracy; contact-localized TR-V2V. | Annotate surface-pair contact and approach/touch/separate phases from multi-view/mocap geometry; compare contact transitions to speed/visibility-matched non-contact frames. **Refute** if no transition-localized error or ordering failure remains after matching. |
| H12 | Collision repulsion reduces penetration but causes a measurable precision–recall trade-off with valid contact. | Penetration volume/max depth; collision count; contact precision/recall; TR-V2V around contact. | Sweep collision weight from zero through default while freezing all other terms; compute penetration and intended-contact metrics. **Refute** if penetration falls without any loss of true-contact recall or increase in geometric error across the full sweep. |
| H13 | The handedness/class heuristic causes a concentrated tail of large left/right errors in ambiguous or low-motion one-handed signs. | Handedness/active-side accuracy; L/R hand TR-V2V; worst-sign and worst-frame error; passive-hand error. | Compare released decisions with expert/oracle labels; run counterfactual correct, swapped, ambiguous, and two-hand settings on identical clips. **Refute** if oracle decisions do not reduce tail error and corrupted decisions do not worsen it. |
| H14 | Prior and fitting accuracy varies substantially across signers beyond what shape normalization explains. | Cross-signer TR-V2V/MPJPE; worst-signer error; joint-limit violations; calibration; personalization gain. | Capture or obtain multi-signer synchronized 3D data; use leave-one-signer-out priors and mixed-effects analysis controlling for speed, sign, shape, and visibility. **Refute** if between-signer variance is small and held-out performance matches within-signer performance. |
| H15 | Domain shift from controlled isolated DGS to continuous/in-the-wild and cross-language signing produces a systematic geometric loss. | Cross-domain TR-V2V/MPJPE; worst-domain error; detection recall; calibration; temporal/contact metrics. | Evaluate the same frozen pipeline on matched accurate-3D corpora across camera/language/continuousness, or a controlled capture varying one domain factor at a time. **Refute** if confidence intervals show practical equivalence across domains. |
| H16 | Strong priors and fixed regularization improve common poses but harm rare, fast, or signer-specific valid poses. | TR-V2V by pose rarity/speed; latent reconstruction error; lag; joint-limit violation; worst-decile error. | Sweep prior, initialization, temporal, and ROM weights one factor at a time and jointly; define pose rarity before test evaluation using training-set density. **Refute** if performance improves monotonically for both common and rare/fast strata with no tail penalty. |
| H17 | Pseudo-label bias learned by SignBPoser survives filtering and limits downstream upper-body accuracy. | Prior reconstruction MPJPE/MPVPE on independent mocap; downstream UBody TR-V2V; error correlation between pseudo labels and predictions. | Train otherwise identical priors on raw pseudo labels, filtered pseudo labels, and a smaller independently captured set; use the same fitter and held-out signer/signs. **Refute** if independent label quality neither changes downstream error nor predicts its spatial pattern. |
| H18 | Local SGNify GT defects materially change method rankings on hand geometry or biomechanics. | Held-out marker/multi-view error; reference uncertainty; ranking stability; plausible-pose violation rate; hand TR-V2V. | Audit all evaluated frames using held-out markers, multi-view silhouettes/keypoints, repeated MoSh++ fits, and blinded expert geometric checks. Re-rank methods with uncertain frames excluded/downweighted. **Refute** if reference uncertainty is small, rare, and rankings/paired differences remain stable. |
| H19 | TR-V2V alone is weakly aligned with perceptual/linguistic correctness and hides catastrophic temporal/contact errors. | Correlation with expert recognition/naturalness; temporal event error; contact F1; worst-decile/per-sign error; ranking stability across alignments. | Release one frame/vertex/evaluator manifest; compute raw, root/TR, and PA metrics plus temporal/contact/occlusion strata; conduct blinded proficient-signer evaluation. **Refute** if TR-V2V strongly predicts expert judgments, captures tails, and yields stable rankings across protocols. No reconstruction metric is expected to improve merely by testing this hypothesis; measurement validity should improve. |

---

## 4. Cross-failure attribution plan

The 19 hypotheses are not independent. A minimum causal study should separate the following factors before interpreting a new model result:

| Factorial block | Controlled variables | Primary question |
|---|---|---|
| Observation × initialization | detector/oracle 2D × detector/oracle initialization | Is the fitter limited by observations, initialization, or both? |
| Wrist × fingers | predicted/oracle wrist × predicted/oracle fingers | Which hand component drives local versus global error? |
| Visibility × duration × speed | visible/occluded × gap length × GT angular speed | Does failure arise from missing evidence, dynamics, or their interaction? |
| Camera × shape | estimated/calibrated camera × estimated/personalized shape | Is body error being absorbed by camera or anthropometrics? |
| Prior × data quality | generic/sign prior × raw/filtered/independent labels | Is improvement due to sign specificity or label curation? |
| Collision × true contact | loss weight × annotated contact state | Does repulsion prevent penetration while destroying intended touch? |
| Metric × reference quality | TR/raw/PA/temporal/contact × clean/uncertain GT | Does the leaderboard reflect reconstruction quality or protocol/reference choices? |

All comparisons should be paired on the identical frame manifest and report per-sign distributions, bootstrap confidence intervals, and effect sizes. This is experimental design, not an architecture proposal.

---

## 5. Research-gap classification and ranking

| Rank | Gap category | Failure IDs | Gap statement | E | R | M | N | F | P | Total /30 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **Occlusion gap** | F5–F7, F8 | Direct sign methods are not evaluated with 3D visibility strata and do not explicitly distinguish hand–hand, hand–body, or long-gap missing evidence. | 5 | 5 | 5 | 4 | 3 | 5 | **27** |
| 2 | **Uncertainty gap** | F1–F8, F13, F18 | A single deterministic SMPL-X estimate is returned even when depth, identity, and articulation are ambiguous; detector score is not calibrated 3D uncertainty. | 5 | 4 | 4 | 4 | 4 | 5 | **26** |
| 3 | **Evaluation gap** | F18–F19, F8–F12 | The protocol is not stable across later papers, the exact evaluator/manifest is unclear, and target metrics omit temporal, contact, robustness, and semantic failure. | 5 | 5 | 3 | 4 | 4 | 5 | **26** |
| 4 | **Temporal gap** | F7, F8, F10–F11 | Direct methods use framewise or short-memory smoothing; fast-motion fidelity, long-gap recovery, hand dynamics, and contact event timing are unmeasured. | 5 | 4 | 4 | 3 | 4 | 5 | **25** |
| 5 | **Interaction/contact gap** | F5–F6, F11–F12 | Intended touch, surface-pair identity, depth order, persistence, and separation timing are absent from direct evaluation and incompletely represented in fitting. | 5 | 4 | 3 | 5 | 3 | 5 | **25** |
| 6 | **Observation gap** | F1, F5–F9 | The causal ceiling imposed by 2D observations, crop quality, camera, and initialization has never been isolated on the sign benchmark. | 5 | 5 | 5 | 1 | 5 | 3 | **24** |
| 7 | **Optimization gap** | F2, F9, F12, F16 | Fixed variables, strong initialization, local minima, and non-adaptive weights may prevent observations from correcting the initial mode; component contributions are not ablated. | 5 | 4 | 4 | 2 | 5 | 3 | **23** |
| 8 | **Dataset gap** | F7–F8, F11, F14–F18 | Accurate SMPL-X signing ground truth lacks signer/language/domain scale, visibility/contact annotations, controlled blur, and independent label-quality estimates. | 5 | 4 | 4 | 3 | 1 | 4 | **21** |
| 9 | **Generalization gap** | F14–F15 | One-signer isolated-sign results cannot establish cross-signer, cross-language, continuous, camera, or in-the-wild geometric generalization. | 4 | 4 | 4 | 3 | 2 | 4 | **21** |
| 10 | **Representation gap** | F2–F6, F11 | Separate deterministic body/hand latents do not encode visibility, relative interaction state, multimodal depth, or explicit surface-pair relationships. Adjacent representations already address parts of this problem. | 4 | 4 | 4 | 2 | 3 | 3 | **20** |
| 11 | **Linguistic gap** | F4, F11, F13, F15, F19 | Geometric errors are not stratified by handshape, orientation, location, movement, dominance, or meaning, so metric gains are not linked to sign intelligibility. | 4 | 3 | 2 | 4 | 2 | 4 | **19** |
| 12 | **Biomechanical gap** | F3–F4, F12, F16, F18 | Plausibility is claimed but not measured with common joint, palm, penetration, or contact metrics; generic biomechanical constraints are already established adjacent prior art. | 4 | 3 | 2 | 2 | 4 | 3 | **18** |

### Why observation is not in the top five

Observation quality is likely high-impact and easy to test, but “use a better keypoint/hand estimator” is already a crowded, low-novelty direction. It ranks sixth because the scientific gap is causal attribution rather than an unoccupied method idea. An oracle-observation study is nevertheless a prerequisite: if it shows little headroom, several occlusion and uncertainty hypotheses weaken immediately.

### Why biomechanics and linguistics rank lower

- Biomechanics may improve plausibility and safety metrics, but DexAvatar's own body-prior ablation gives direct evidence that naive additional constraints can over-regularize, and generic joint/palm/contact constraints have extensive prior art.
- Linguistic correctness is scientifically important, but current evidence does not show that injecting semantics will lower TR-V2V. Its strongest role at this phase is error stratification and reviewer validation, not a guaranteed geometric mechanism.

---

## 6. The top five gaps in detail

### 6.1 Occlusion gap

**FACT:** direct sign papers discuss occlusion but report aggregate geometric errors or selected qualitative examples. DexAvatar has no explicit visibility state; long-duration hand occlusion is not tested.

**INFERENCE:** occlusion is not simply “hard imagery.” It changes the inverse problem by removing evidence and increasing the number of plausible depth, identity, and articulation solutions. Hand–hand and hand–body cases should not be pooled because they have different geometry and contact semantics.

**Research question:** how much error is caused by occlusion type, fraction, and duration after controlling for crop size, speed, blur, and detector confidence?

**Falsifier:** on matched frames and controlled masks, neither occlusion type nor duration has a practically meaningful residual effect on TR-V2V, depth order, identity, or recovery.

**Required evidence before method design:** visibility/overlap labels, real and controlled occlusion tests, oracle-observation ceiling, error-versus-gap curves, and post-occlusion recovery metrics.

### 6.2 Uncertainty gap

**FACT:** DexAvatar optimizes one pose and uses detector confidence as a weight. SignBPoser/SignHPoser Gaussian latent priors express a training manifold, not calibrated image-conditioned uncertainty. Adjacent methods such as ProHMR, POCO, MaskHand, and ScoreHMR establish that ambiguity can be represented or scored, but none validates calibrated sign-SMPL-X uncertainty.

**INFERENCE:** the scientifically relevant missing quantity is not merely a second pose sample; it is whether uncertainty rises on the frames/joints where 3D error, depth ambiguity, and occlusion are actually high.

**Research question:** can frame-, joint-, or hypothesis-level uncertainty predict sign reconstruction error and distinguish ambiguous from merely noisy observations?

**Falsifier:** uncertainty scores are uncalibrated, fail to rank errors better than detector confidence, or multiple plausible hypotheses collapse to the same wrong mode.

**Required evidence before method design:** reliability diagrams, risk–coverage curves, error/uncertainty correlation by visibility, and multi-view confirmation of alternative modes.

### 6.3 Evaluation gap

**FACT:** later papers reuse nearly identical SGNify tables while naming different alignments. The exact frame manifest/evaluator is not verified across methods. TR-V2V ignores temporal ordering and contact and can be affected by imperfect fitted GT. SGNify itself says per-frame V2V is inadequate for some perceptually important signing errors.

**INFERENCE:** a claimed sub-millimetre improvement is not persuasive until identical predictions are evaluated with the same frames, vertices, centering, and uncertainty treatment.

**Research question:** do method rankings survive a common evaluator, paired confidence intervals, per-sign tails, GT-quality audit, and temporal/contact/occlusion metrics?

**Falsifier:** rankings are stable across all documented alignment/evaluator variants and TR-V2V strongly predicts temporal/contact and proficient-signer judgments.

**Required evidence before method design:** public manifest and vertex indices; raw/root/TR/PA decomposition; paired bootstrap; per-sign distribution; visibility/contact/blur strata; temporal fidelity; blinded signer study.

### 6.4 Temporal gap

**FACT:** DexAvatar has a previous-body-pose penalty and no final hand temporal term. Smoothing can lower jerk without improving motion accuracy. Sign meaning depends on temporal evolution.

**INFERENCE:** the gap is fidelity, not smoothness alone. A good trajectory must suppress observation noise while retaining genuine fast articulation, phase, repetitions, holds, and contact events.

**Research question:** where is the Pareto frontier between per-frame geometry, jitter, GT velocity/acceleration, lag, and semantic event timing?

**Falsifier:** current local smoothing is already Pareto-optimal over speed/occlusion strata and produces no lag or attenuation of true hand dynamics.

**Required evidence before method design:** GT velocity/acceleration, speed bins, spectral/lag analysis, contact-onset timing, long-gap recovery, and hand-specific temporal metrics.

### 6.5 Interaction/contact gap

**FACT:** SGNify contains generic self-contact/interpenetration terms; DexAvatar's released objective exposes repulsion without intended contact. Neither reports sign contact accuracy. Generic TUCH/ARCTIC work establishes feasibility but not sign-specific correctness.

**INFERENCE:** “no penetration” and “correct contact” are different objectives. A separated hand can be collision-free and wrong; a true touch can be penalized by repulsion; correct 2D overlap can have reversed depth order.

**Research question:** how often do current methods recover the correct surface pair, depth order, onset, duration, and separation, and how much does this matter to local geometry and intelligibility?

**Falsifier:** contact-state and ordering errors are rare, uncorrelated with hand/upper-body error and signer judgments, and current collision handling lies on the optimal contact–penetration trade-off.

**Required evidence before method design:** surface-pair contact labels, signed distances/normals, contact precision/recall, penetration depth/volume, onset/offset error, and matched non-contact controls.

---

## 7. Reviewer-oriented validity checks

A strong failure-analysis paper or the diagnostic section of a future method paper should survive the following objections.

| Reviewer objection | Required response |
|---|---|
| “The gain comes only from a newer hand estimator.” | Same fitter with old/new/oracle 2D and initialization; report isolated and interaction effects. |
| “Your metric differs from DexAvatar/SOKE.” | Evaluate released predictions with one script, exact frame manifest, vertex sets, and centering; report raw/TR/PA separately. |
| “Occlusion is confounded with motion speed and crop size.” | Matched or factorial analysis controlling speed, pixel area, blur, and detector confidence. |
| “Lower jerk is just over-smoothing.” | Report GT velocity/acceleration, lag, spectral attenuation, and event timing alongside jerk. |
| “Contact accuracy merely tracks TR-V2V.” | Report contact metrics and contact-localized TR-V2V; show partial correlation after geometry, speed, and visibility controls. |
| “The ground truth is itself wrong.” | Independent marker/multi-view audit, repeated-fit uncertainty, uncertain-frame sensitivity, and stable rankings. |
| “One signer cannot establish generalization.” | Multi-signer held-out evaluation or explicitly restrict claims to within-signer isolated DGS. |
| “Plausibility is subjective.” | Predefined joint/palm/penetration/contact metrics plus blinded expert evaluation. |
| “Semantics were added after looking at errors.” | Pre-register phonological/contact/visibility strata and evaluation hypotheses. |
| “Your ablation changes multiple components.” | Factorial interventions with all other observations, priors, and optimization settings fixed. |

---

## 8. Evidence-strength summary by failure

| Evidence tier | Failure modes | Interpretation |
|---|---|---|
| Strong direct dependency or ablation | F1, F3, F9, F10, F13, F16, F17, F19 | The component/limitation is verified in paper or code; its exact causal error share still needs intervention. |
| Strong absence plus adjacent evidence | F2, F5–F8, F11–F12 | Direct methods lack the measurement/mechanism and adjacent work demonstrates relevance; sign-specific magnitude is not known. |
| Benchmark-coverage limitation | F14–F15 | The generalization claim is untestable on the one-signer primary benchmark. |
| Local reference-quality evidence, prevalence unknown | F18 | Some defects are documented, but a benchmark-wide inaccuracy claim would be unsupported. |
| Residual geometric error, attribution unresolved | F4 | Hand error remains, but wrist, fingers, reference quality, and alignment are currently entangled. |

---

## 9. Primary-source evidence ledger

Only primary papers, official project pages, and official repositories are used below.

### Direct sign reconstruction and benchmarks

- [DexAvatar paper](https://arxiv.org/abs/2512.21054) · [official repository](https://github.com/kaustesseract/DexAvatar)
- [SGNify paper](https://arxiv.org/abs/2304.10482) · [CVPR paper page](https://openaccess.thecvf.com/content/CVPR2023/html/Forte_Reconstructing_Signing_Avatars_From_Video_Using_Linguistic_Priors_CVPR_2023_paper.html) · [official repository](https://github.com/MPForte/SGNify)
- [Neural Sign Actors](https://arxiv.org/abs/2312.02702)
- [SOKE / Signs as Tokens](https://arxiv.org/abs/2411.17799) · [official repository](https://github.com/2000ZRL/SOKE)
- [Tamaththul3D](https://arxiv.org/abs/2605.05367)
- [SignAvatars](https://arxiv.org/abs/2310.20436) · [official repository](https://github.com/ZhengdiYu/SignAvatars)

### Observation, hand/body integration, occlusion, and interaction

- [HaMeR](https://arxiv.org/abs/2312.05251) · [official repository](https://github.com/geopavlakos/hamer)
- [WiLoR](https://arxiv.org/abs/2409.12259) · [official repository](https://github.com/rolpotamias/WiLoR)
- [Hand4Whole++](https://arxiv.org/abs/2603.14726) · [official repository](https://github.com/mks0601/Hand4Whole-plus-plus_RELEASE)
- [MaskHand](https://arxiv.org/abs/2412.13393) · [official project](https://m-usamasaleem.github.io/publication/MaskHand/MaskHand.html)
- [IntagHand](https://arxiv.org/abs/2203.09364) · [official repository](https://github.com/Dw1010/IntagHand)
- [Dyn-HaMR](https://arxiv.org/abs/2412.12861) · [official repository](https://github.com/ZhengdiYu/Dyn-HaMR)
- [Pose-Guided Temporal Enhancement for low-resolution hands](https://cv.nirc.top/2025/temp-lowres-hand/)

### Ambiguity, temporal motion, contact, and annotation quality

- [ProHMR](https://arxiv.org/abs/2108.11944) · [official repository](https://github.com/nkolot/ProHMR)
- [POCO](https://arxiv.org/abs/2308.12965) · [official repository](https://github.com/saidwivedi/POCO)
- [ScoreHMR](https://arxiv.org/abs/2403.09623) · [official repository](https://github.com/statho/ScoreHMR)
- [GLAMR](https://arxiv.org/abs/2112.01524) · [official repository](https://github.com/NVlabs/GLAMR)
- [DanceHMR](https://arxiv.org/abs/2605.18102)
- [On Self-Contact and Human Pose / TUCH](https://arxiv.org/abs/2104.03176) · [official repository](https://github.com/muelea/selfcontact)
- [KNOWN-Hand](https://arxiv.org/abs/2407.12307) · [official repository](https://github.com/zhangy76/KNOWN-Hand)
- [AMASS / MoSh++](https://arxiv.org/abs/1904.03278)

---

## Final Phase-3 conclusion

**FACT:** reconstruction error remains substantial, but the current literature cannot causally assign it to a module because the necessary oracle and factorial ablations are missing.

**INFERENCE:** the highest-value research space is the intersection of sign-specific occlusion, calibrated ambiguity, temporal fidelity, and interaction-aware evaluation. This conclusion is about where evidence is missing, not a claim that a particular architecture will outperform SOTA.

**DECISION:** do not select or propose the final method yet. First run the observation/initialization and wrist/finger oracle studies, construct visibility/contact/temporal strata, and reproduce all candidate predictions under one TR-V2V evaluator. Those results can falsify several top gaps before architecture search begins.
