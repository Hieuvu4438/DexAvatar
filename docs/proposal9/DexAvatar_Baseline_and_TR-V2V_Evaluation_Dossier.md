# DexAvatar Baseline and TR-V2V Evaluation Dossier

**Technical audit date:** 2026-08-25 (Asia/Bangkok)  
**Audit scope:** attached DexAvatar paper + all supplementary pages; official DexAvatar repository; attached evaluation protocol.  
**Authority order used:** evaluation script > repository source > paper/supplementary.  
**No reconstruction experiment was run:** numerical results below are either author-reported or arithmetic recomputed from published rounded values.

## Audit identity and citation convention

- **P** — attached `Kundu et al. - 2025 - DexAvatar 3D Sign Language Reconstruction with Hand and Body Pose Priors(1).pdf`; 21 PDF pages; SHA-256 `49d44f0d03c8d5a98e23594c2f43d8c4f9e7c07eeb701d99bac341e929077ed3`. Page references in this dossier are **PDF page numbers**.
- **R** — official repository [`kaustesseract/DexAvatar`](https://github.com/kaustesseract/DexAvatar), branch `main`, commit [`a0dfd427f60f5811aadb35c8657b3856d47f56b5`](https://github.com/kaustesseract/DexAvatar/tree/a0dfd427f60f5811aadb35c8657b3856d47f56b5), accessed 2026-08-25. All repository line references are at this commit.
- **E** — attached `evaluate_new_fitting(2).py`; 589 lines; SHA-256 `2722b5cd30d4baba23599a455cab483b143e6595d292f02de9643af4eebd5300`.
- The request names `evaluate_new_fitting(1).py`, but that file was not attached. **`evaluate_new_fitting(1).py`: NOT INSPECTED.** `[UNRESOLVED]` Whether it is byte-identical or behaviorally equivalent to the attached `(2).py` cannot be established. Every evaluation conclusion below therefore refers explicitly to **E = `(2).py`**.

---

## 1. Executive verified summary

`[VERIFIED]` DexAvatar takes a directory of monocular sign-language video frames and produces per-frame SMPL-X OBJ meshes plus parameter pickle files. The implemented driver processes sign folders sequentially and the fitter processes selected frames serially. The only trainable fitting variables in the default path are a 33-D SignBPoser latent and one or two independent 23-D SignHPoser latents; global orientation, per-sign mean shape, facial expression, jaw/eye pose, translation, and the 3×3 camera intrinsic matrix are fixed from preprocessing. The active body and hand poses are decoder outputs in axis-angle form. (P p.3 §3, p.4 Fig.2; R `run_dexavatar.py` L13–30; `smplifyx/fit_single_frame.py::fit_single_frame` L218–243, L476–503.)

`[VERIFIED]` Default preprocessing uses Sapiens 133-keypoint output, SMPLer-X body/camera/shape estimates, and HaMeR hand detections, rotations and keypoints. HaMeR rotations overwrite the SMPLer-X hand-pose fields before fitting, so the hand-pose supervision target is HaMeR-derived. The default HaMeR 3D-joint loss weight is zero. (R `data_parser.py::read_item` L381–440, L660–680; `fit_single_frame.py` L245–258; `fit_smplx_vposer_x.yaml` L50–77.)

`[VERIFIED]` The code implements 2D reprojection, body/hand latent penalties, body/hand initialization supervision, collision, body temporal consistency, body biomechanical bounds, an inherited SMPLify angle prior, and fixed-parameter shape/face penalties. It does **not** implement a hand-biomechanical fitting term, although Eq. (12) includes one. The paper’s quadratic biomechanical penalty is implemented for the body as a mean linear hinge violation. (P p.5 Eq.11–12, p.6 Eq.13–15; R `fitting.py::SMPLifyLoss.forward` L499–525, L527–664.)

`[VERIFIED]` E computes each TR-V2V region after independently subtracting the prediction and GT centroid of that region in each paired frame. It performs no scale, rotation, Procrustes, pelvis or wrist alignment. Frames are paired only by position in separately sorted file lists, not by filename or frame ID; all vertex-frame distances are concatenated and averaged, then multiplied by 1,000. (E `transl_point_error` L159–169; `main` L342–395, L432–461.)

`[VERIFIED]` `--central` is parsed but unused. `sign_seg` is always applied; each interval is multiplied by two and treated as inclusive. For `class_sign == "0"`, left-hand vertices are removed from every region except the left-hand region, and the left-hand metric is skipped entirely. (E `load_gt_obj_paths` L231–265; `main` L301–307, L380–395; module CLI L479–512.)

`[VERIFIED]` The paper reports DexAvatar at 30.13/13.53/13.08 mm for UBody(-F)/LHand/RHand. From the rounded table, improvement over Neural Sign Actors is 35.09%/16.33%/14.12%, whereas the paper reports 35.11%/16.32%/14.11. These values were not reproduced. (P p.6 Table 1 and §5.1.)

---

## 2. Formal problem specification

### 2.1 Task, data roles, and representations

| Category | Exact role | Evidence | Confidence |
|---|---|---|---|
| Task definition | Recover per-frame expressive whole-body SMPL-X geometry from a monocular sign-language frame sequence; body and hands are refined through learned pose manifolds. | P p.1 Abstract/Fig.1; p.3 §3; p.4 Fig.2. | `[VERIFIED]` |
| Runtime RGB input | A root directory whose children are sign folders; each sign folder is expected to contain `images/*.png` or `*.jpg`. | R `README.md` L16–35, L92–98; `run_dexavatar.py` L13–27; `data_parser.py::__init__` L137–168. | `[VERIFIED]` |
| SignBPoser training data | A subset of SignAvatars 3D body data reconstructed from How2Sign; these are explicitly described as pseudo-ground truth and filtered by upper-limb biomechanics/signer-space rules. | P p.2 contribution 1; p.3 §3.2.1; p.5 Fig.3. | `[VERIFIED]` |
| SignHPoser training data | Physical motion capture from 9 Vicon cameras and Manus gloves, 8 signers (6 Auslan, 2 ASL), 93 fingerspelled words; retargeted/baked onto SMPL-X and hand poses biomechanically rectified. | P p.4 §3.2.2; pp.9–11 §S3. | `[VERIFIED]` |
| Runtime pseudo-ground-truth | Sapiens body/face 2D keypoints; HaMeR hand 2D keypoints, hand rotations and 3D keypoints; SMPLer-X body pose, global orientation, shape, face, translation and intrinsics. These are detector/regressor outputs, not benchmark GT. | P p.3 §3, p.4 Fig.2, p.5 §3.4; R `inference.py` L153–198; `hamer/demo.py` L124–153; `data_parser.py` L381–440, L660–680. | `[VERIFIED]` |
| Test/evaluation RGB | Central portions of 57 German SGNify signs according to the paper and repository sign/segment files. | P p.6 §4 Datasets; R `data/signs.txt` L1–57; `data/segment.json` L1–229. | `[VERIFIED]` |
| Motion-capture GT | Per-frame SMPL-X OBJ meshes from SGNify, loaded from the evaluation `--gt_folder`. They are not used by fitting. | R `README.md` L16; E `load_gt_obj_paths` L231–265 and `main` L355–370. | `[VERIFIED]` |
| Paper frame count | Paper states 2,872 central frames. The repository segment endpoints sum to exactly 2,872 only under half-open counting, while both fitter and E use inclusive endpoints. Actual metric count depends on the unavailable GT/prediction file manifests. | P p.6 §4; R `data_parser.py::__init__` L147–168; E `load_gt_obj_paths` L237–249; independent arithmetic over R `segment.json`. | `[VERIFIED]` for stated counts/code; `[UNRESOLVED]` for actual evaluated count. |

### 2.2 Input and output

Let a sign clip be an ordered image sequence

\[
I_s = \{I_{s,t}\}_{t=a_s}^{b_s}, \qquad I_{s,t}\in[0,1]^{H_t\times W_t\times3}.
\]

`[VERIFIED]` The intended output per frame is an SMPL-X body with 10,475 vertices and 54 articulated joints, parameterized by pose \(\theta\), shape \(\beta\), and expression \(\psi\). (P p.9 §S1, Eq.16–17.)

`[VERIFIED]` The repository writes both:

1. `smplifyx/meshes/<frame>.obj`: the optimized mesh, then rotated 180° about the X axis before export. (R `fit_single_frame.py` L611–660.)
2. `smplifyx/results/<frame>.pkl`: named SMPL-X parameters, decoded optimized body pose, and `K`. (R `fit_single_frame.py` L590–609.)

`[VERIFIED]` These two outputs are **not parameter-equivalent**: the pickle overwrites `body_pose` with the decoded optimized SignBPoser pose, but does not overwrite left/right `hand_pose` with decoded optimized SignHPoser poses and does not store hand latents; the OBJ is generated with the decoded hand poses. The OBJ additionally receives a 180° X rotation absent from the saved parameter set. Therefore the saved pickle alone is insufficient to reconstruct the exported OBJ exactly. (R `fit_single_frame.py` L593–609 versus L627–660.)

### 2.3 Optimization variables and fixed quantities

| SMPL-X/camera component | Source/initial value | Actual fitting treatment | Representation and shape | Evidence | Confidence |
|---|---|---|---|---|---|
| Global orientation | SMPLer-X | Fixed; included in `reset_params`, absent from optimizer parameter list | Axis-angle, 3 | R `inference.py` L167–177; `fit_single_frame.py` L377–443, L476–503. | `[VERIFIED]` |
| Body pose | SMPLer-X supplies target; latent starts at zero | **Optimized indirectly** as \(D_B(z_B)\); stored body-model pose is overridden during every forward | \(z_B\in\mathbb R^{33}\); decoder output axis-angle \(1\times63=21\times3\) | R `fit_single_frame.py` L218–229, L443–479; `fitting.py` L244–265, L527–533. | `[VERIFIED]` |
| Left/right hand pose | HaMeR rotations overwrite SMPLer-X hand values and become targets; each active latent starts at zero | **Optimized indirectly** as independent \(D_H(z_L),D_H(z_R)\); one latent omitted for one-handed clips | Each latent \(1\times23\); decoder output axis-angle \(1\times45=15\times3\) | R `data_parser.py` L415–440; `fit_single_frame.py` L231–243, L448–492; `fitting.py` L258–277. | `[VERIFIED]` |
| Shape \(\beta\) | Mean of SMPLer-X 10-D shapes across available sign frames | Fixed during latent optimization | 10-D coefficients | R `M3_mean_shape_smplerx.py` L13–23; `data_parser.py` L205, L660; `fit_single_frame.py` L476–503. | `[VERIFIED]` |
| Expression \(\psi\) | SMPLer-X | Fixed; its loss is constant w.r.t. fitted latents | 10-D | R `inference.py` L175–176; `fit_single_frame.py` L386–425, L476–503; `fitting.py` L621–632. | `[VERIFIED]` |
| Jaw pose | SMPLer-X | Fixed; prior term is optimization-inert for the active variables | Axis-angle, 3 | Same locations as expression. | `[VERIFIED]` |
| Eye poses | Set to zero by SMPLer-X wrapper | Fixed | Two axis-angle vectors, 3 each | R `inference.py` L173–179; `fit_single_frame.py` L386–425, L476–503. | `[VERIFIED]` |
| Translation | SMPLer-X `cam_trans` | Fixed; absent from optimizer parameter list | 3-D | R `inference.py` L175–190; `fit_single_frame.py` L386–425, L476–503. | `[VERIFIED]` |
| Camera intrinsics | SMPLer-X focal/principal point | Fixed 3×3 perspective matrix; no optimized extrinsics | \(K\in\mathbb R^{3\times3}\) | R `data_parser.py` L660–667; `main.py` L248–253, L305–328; `assets/mapping_func.py::project_torch` L10–23. | `[VERIFIED]` |
| Camera initialization object | Generic `guess_init` result | Constructed with a zero body latent, but `camera_loss` is never used by the fitting loop | Estimated translation 1×3 | R `fit_single_frame.py` L329–365 and L462–570. | `[VERIFIED]` |
| Facial geometry | Expression/jaw/eyes fixed; facial 2D joints receive nonzero weights | Not directly optimized; body-latent changes can still move head/face rigidly through kinematics | SMPL-X face vertices/joints | R YAML L86–111; `fit_single_frame.py` L476–548. | `[VERIFIED]` |

`[VERIFIED]` Despite paper language that SMPLer-X/HaMeR estimates initialize pose, active pose latents are explicitly zeroed. The first mesh evaluated inside optimization therefore uses each learned decoder at zero latent, while the off-the-shelf pose acts as a loss target; non-pose parameters remain initialized from SMPLer-X. (P p.5 §3.4; R `fit_single_frame.py` L443–460; `fitting.py` L244–277.)

### 2.4 Temporal granularity and assumptions

- `[VERIFIED]` Fitting is per frame but **not independent**: optimized body axis-angle from frame \(t-1\) is passed to frame \(t\). The first frame uses the SMPLer-X body pose as the temporal target. No hand temporal term is implemented. (R `main.py` L226–243, L305–330; `fit_single_frame.py` L260–262, L728–730; `fitting.py` L499.)
- `[VERIFIED]` The root driver and fitting loop are serial in sign and frame order. (R `run_dexavatar.py` L13–30; `main.py` L243–330.)
- `[VERIFIED]` Runtime assumes one principal person: SMPLer-X chooses one bounding box when `multi_person` is false, and HaMeR iterates detected people/hands without an explicit signer identity tracker. (R `SMPLer-X/main/inference.py` L107–139; `hamer/demo.py` L81–126.)
- `[VERIFIED]` Neutral SMPL-X is forced by writing `gender.txt`, and the neutral model is created with flat-hand mean and without hand PCA, so the fitting path uses full 45-D hand axis-angle pose. (R `scripts/M3.5_hamer_extract.sh` L1–4; `smplifyx/main.py` L144–150.)
- `[VERIFIED]` Frame availability assumes a HaMeR record with at least one detected hand and an SMPLer-X pickle; otherwise the frame is removed before fitting. Two-handed parsing then indexes two HaMeR detections without a second explicit count check. (R `data_parser.py` L180–199, L397–421.)
- `[VERIFIED]` The paper describes both one- and two-handed signs. The repository’s default `signs.txt` has 15 class `0` and 42 class `~0` signs. Class `0` triggers a clip-level wrist-motion heuristic to select the active side; the ambiguous case maps to `one_hand_is_right=False`, hence the left-hand branch. (R `data/signs.txt` L1–57; `data_parser.py` L201–203, L285–346, L443–652.)
- `[VERIFIED]` Image paths are initially sorted numerically but are re-sorted lexicographically after missing-output filtering. Temporal order is therefore correct only if filenames are sufficiently zero-padded or lexical and numeric orders coincide. (R `data_parser.py` L140–143, L180–199.)
- `[VERIFIED]` The heuristic also returns per-frame `w_left`/`w_right` arrays, but no downstream code consumes them; only the discrete active-side label affects fitting. Its `active == 'both'` weight branch is unreachable from the preceding decision logic. (R `data_parser.py` L285–346; no other uses of `w_left`/`w_right`.)
- `[UNRESOLVED]` The sources do not establish that every SGNify signer is fully visible, that every class-`0` sign is right-handed, or that detection order remains stable with multiple people.

---

## 3. End-to-end pipeline

| Stage | Input | Output | Representation / shape | Optimized or fixed | Evidence | Confidence |
|---|---|---|---|---|---|---|
| 0. Sign orchestration | Root directory of sign folders; output root | One sequential shell invocation per sorted child name | Filesystem paths | No model optimization | R `run_dexavatar.py` L13–30. | `[VERIFIED]` |
| 1. Sapiens | `<sign>/images/*` RGB | `<output>/sapiens.pkl` expected by parser | Code expects body/face keypoints `[1,133,2]` plus confidence `[1,133]`; image-pixel coordinates. README expects RTMDet person detection and Sapiens-1B COCO-WholeBody pose assets | Detector output, later supervision | R `README.md` L37–54; `S1_sapiens_extract.sh` L3–6; `data_parser.py` L172–178, L381–391. Sapiens executable is absent from clone. | `[VERIFIED]` interface/expected assets; `[UNRESOLVED]` producer internals. |
| 2. SMPLer-X | RGB frames; Faster R-CNN R50-FPN person box; `smpler_x_h32` checkpoint | `smplerx/smplx/<frame>.pkl`, optional mesh/render/meta | global 3, body 63, L/R hands 45 each, jaw 3, eyes 3+3, shape 10, expression 10, translation 3, focal 2, principal point 2; 6-D rotations are converted to axis-angle before export | Initialization/fixed values and body target; not fitted here | R `script_smplerx.py` L16–25; `inference.py` L45–74, L107–198; `SMPLer_X.py` L176–215, L381–388. | `[VERIFIED]` |
| 3. Mean shape | All SMPLer-X pickles for a sign | `mean_shape_smplx.npy` | Mean of per-frame 10-D `betas` | Fixed in fitting | R `M3_mean_shape_smplerx.py` L13–23. | `[VERIFIED]` |
| 4. HaMeR | RGB; ViTDet person boxes; ViTPose hand keypoints; HaMeR default checkpoint | `hamer/hamer.pkl` and visual/mesh artifacts | Per detection: MANO hand rotation matrices `[15,3,3]`, 2D `[21,2]`, 3D `[21,3]`, center `[2]`, size scalar, handedness flag, camera translation `[3]` | Detector/regressor output; pose target and 2D supervision; 3D supervision disabled by default | R `hamer/demo.py` L21–68, L81–153, L183–216. | `[VERIFIED]` |
| 5. Central-frame/data fusion | RGB + segment interval + Sapiens + SMPLer-X + HaMeR + mean shape | One fused dictionary per retained frame | `keypoints [1,133,3]`; `K [3,3]`; SMPL-X dict; HaMeR 3D; active-side label | Supervision/initialization only | R `data_parser.py` L140–205, L381–440, L660–702. | `[VERIFIED]` |
| 6. Hand conversion/fusion | HaMeR crop-normalized 2D and rotation matrices | Pixel-coordinate hand keypoints; 45-D axis-angle hand target | 2D unnormalized by box size/center; each 3×3 rotation converted using `cv2.Rodrigues`; left AA Y/Z signs flipped | Supervision target; not optimized directly | R `data_parser.py` L250–282, L397–440. | `[VERIFIED]` |
| 7. One-hand decision | Class `0`; Sapiens wrist tracks | `left_hand` or `right_hand` label | Mean finite inter-frame 2D wrist speed; confidence >0.3; tie ratio 1.2 | Fixed clip-level routing decision | R `data_parser.py` L285–346. | `[VERIFIED]` |
| 8. Latent fitting | Fused targets, fixed SMPL-X/camera values, prior decoders | Optimized \(z_B\), \(z_L\), \(z_R\) as applicable | Body latent 33; each hand latent 23; decoded AA body 63 and hand 45 | Only these latents are optimizer parameters; 3 LBFGS-LS stages | R YAML L40–119; `fit_single_frame.py` L218–243, L462–570. | `[VERIFIED]` |
| 9. Temporal carry | Current optimized body pose | `joints_temp` for next frame | Axis-angle `[1,21,3]` | Fixed target in next frame | R `main.py` L226–243, L324–330; `fit_single_frame.py` L728–730. | `[VERIFIED]` |
| 10. Export | Optimized decoder outputs + fixed other SMPL-X parameters | Per-frame pickle, OBJ, overlay PNG | SMPL-X mesh 10,475 vertices / 20,908 faces in repository OBJ template; exported OBJ rotated 180° X | Final artifact | P p.9 §S1; R `fit_single_frame.py` L590–660; `assets/smplx_uv_new.obj` counted directly. | `[VERIFIED]` |

### 3.1 Coordinate-system and representation audit

- `[VERIFIED]` Sapiens targets are image-pixel keypoints. HaMeR hand 2D outputs are mirrored according to handedness, scaled by crop box size and shifted by crop center into image pixels before replacing Sapiens hand joints. (R `data_parser.py` L250–282, L397–413.)
- `[VERIFIED]` SMPL-X vertices/joints are projected by \(\pi_K(X)= (KX)_{xy}/(KX)_z\); no extrinsic matrix is applied in `project_torch`. Translation is already carried by the SMPL-X model parameter initialized from SMPLer-X. (R `assets/mapping_func.py` L10–23; `data_parser.py` L660–667.)
- `[VERIFIED]` Body/hand optimization and temporal comparison use axis-angle. Only the body biomechanics computation converts six joint rotations to intrinsic XYZ Euler angles in radians. (R `fitting.py` L499, L514–517; `utils.py` L221–265.)
- `[VERIFIED]` HaMeR provides MANO rotation matrices; the parser converts them to axis-angle. For left hands it flips two axis-angle components. HaMeR 3D left-hand X coordinates are also mirrored before adding camera translation. (R `data_parser.py` L415–433, L560–583.)
- `[VERIFIED]` The exported OBJ undergoes a 180° X-axis transform; E applies no additional transform except when the user-supplied method name is exactly `pixie`. (R `fit_single_frame.py` L650–660; E `read_verts_and_faces` L134–149.)
- `[UNRESOLVED]` No source provides a complete formal mapping among Sapiens, HaMeR, SMPLer-X camera coordinates, exported OBJ coordinates, and SGNify GT coordinates. The observable conversions above are the complete implemented evidence.

### 3.2 Error propagation to TR-V2V

| Stage failure | Direct downstream effect | TR-V2V consequence | Confidence/evidence |
|---|---|---|---|
| Missing Sapiens output | Parser fails on missing `sapiens.pkl`; unreliable joints alter reprojection target | Articulation/rotation error remains visible after subset-centroid alignment; pure global translation is removed | `[VERIFIED]` code path: R `data_parser.py::__init__` L172–178; R `SMPLifyLoss.forward` L520–525; E `transl_point_error` L159–169. |
| Missing SMPLer-X frame | Frame is removed | Positional evaluator pairing can shift every later prediction against a different GT frame | `[VERIFIED]` R `data_parser.py::__init__` L192–199; E `main` L342–361. |
| Wrong SMPLer-X orientation/shape/camera | These values are fixed; body pose is additionally supervised toward its estimate | Orientation/shape errors propagate to UBody(-F) and possibly hand meshes; subset translation is removed but not rotation/scale | `[VERIFIED]` R `fit_single_frame.py::fit_single_frame` L376–503; E `transl_point_error` L159–169. |
| Missing/wrong HaMeR hand | Frame can be removed; one-hand parser may reuse previous observation; wrong rotations become direct targets | L/R-hand error and upper-body error; a missing middle frame can also cause positional mispairing | `[VERIFIED]` R `data_parser.py::__init__` L180–199 and `read_item` L443–652; E `main` L342–361. |
| Incorrect one-hand side | Wrong arm/hand keypoints and latent are activated | E always excludes the left hand for class `0`, independently of fitted side; the reported regions can therefore assess a different side than the fitter treated as active | `[VERIFIED]` R `data_parser.py::__init__` L201–203 and `_compute_motion_scores_for_clip` L285–346; E `main` L380–395. |
| Temporal predecessor error | Current body latent is pulled toward erroneous previous AA pose | Propagates to later UBody(-F); hands have no temporal carry | `[VERIFIED]` R `fitting.py` L499, L654–657; `main.py` L305–330. |
| Export mismatch | Pickle lacks optimized hand pose and mesh-space X rotation | Re-evaluation from the saved pickle can differ from evaluation of exported OBJ | `[VERIFIED]` R `fit_single_frame.py` L590–660. |

---

## 4. Objective-to-code mapping

### 4.1 Implemented fitting objective

Define the elementwise Geman–McClure robustifier used by the repository:

\[
g_\rho(r)=\rho^2\frac{r^2}{r^2+\rho^2},\qquad \rho=100.
\]

(R `smplifyx/utils.py::GMoF` L60–71; YAML L31–33.)

`[VERIFIED]` For the default `use_signbposer=True`, `use_hposer3d=True` path, the differentiable loss accumulated by `SMPLifyLoss.forward` is:

\[
\begin{aligned}
L_{\text{code}} ={}&L_{2D}+w_{3D}L_{H,z}^{3D}+L_{B,\mathrm{latent}}+L_{B,\mathrm{target}}\\
&+L_{H,\mathrm{latent}}+L_{H,\mathrm{target}}+L_{\mathrm{shape}}+L_{\mathrm{angle}}\\
&+L_{\mathrm{expr}}+L_{\mathrm{jaw}}+L_{\mathrm{pen}}+L_{\mathrm{temp}}+L_{B,\mathrm{bio}}.
\end{aligned}
\]

There is no \(L_{H,\mathrm{bio}}\) term. With the default YAML, \(w_{3D}=0\) in every stage. Because the optimizer list contains only pose latents, shape/expression/jaw and any inactive-hand generic prior terms are constant with respect to the optimized variables. (R `fit_single_frame.py` L476–503; `fitting.py` L499–664; YAML L50–119.)

The exact implemented 2D term is

\[
L_{2D}=\left(\frac{1000}{H}\right)^2
\sum_{j,d}\left(\gamma_j c_j\right)^2
g_{100}\!\left(u_{j,d}-\pi_K(X_j)_d\right),
\]

where the code uses a sum, not the paper’s \(1/|J|\) mean. Lower-body weights at indices `11:23` are zeroed; body, face, and hand multipliers change by optimization stage. (R `fit_single_frame.py` L371–375, L506–548; `fitting.py` L503–525.)

### 4.2 Loss/constraint audit

| Loss / constraint | Paper equation | Actual code location and tensor | Target source / robustifier | Default weight and stage | Enabled by default? | Consistency status |
|---|---|---|---|---|---|---|
| 2D joint reprojection | Eq.2; part of Eq.12: \(\lvert J\rvert^{-1}\sum\gamma_i\omega_i\psi(\pi(D_i)-K_i)\) | R `fitting.py::SMPLifyLoss.forward` L434–455, L503–525; projected SMPL-X joints vs `gt_joints` | Sapiens body/face plus HaMeR-replaced hands; GMoF \(\rho=100\); confidence and region weights are squared | Image factor `(1000/H)^2`; body `[0.5,1,1.5]`, hands `[0.5,1.5,2.5]`, face `[1,1,2]`; all 3 stages | Yes | `PARTIALLY CONSISTENT`: same signal/robustifier, but code sums rather than divides by \(\lvert J\rvert\), and applies squared weights. |
| Lower-body exclusion | Text below Eq.12: \(\omega_i=0\) | R `fitting.py` L503–507; `fit_single_frame.py` L516–544 | Joint mask | Zero at indices `11:23`, every stage | Yes | `CONSISTENT` for the 2D term; lower-body rotations remain indirectly controlled by the body latent and angle prior. |
| One-hand non-dominant 2D exclusion | Text below Eq.12: disable non-dominant arm/hand | R `fit_single_frame.py` L516–544 | Class file + wrist-motion active-side decision | Zeroes three arm joints and 21 hand joints on inactive side | Yes for class `0` | `PARTIALLY CONSISTENT`: reprojection weights are zero, but the single body latent still decodes the full 21-joint body; no independent arm parameter is frozen. |
| Body latent Gaussian prior | Eq.13 plus Eq.3 | R `fitting.py` L527–533; `pose_embedding` | Zero-mean unit Gaussian represented as \(\lVert z_B\rVert_2^2\) | `4.78²` in all 3 stages | Yes | `CONSISTENT` in latent form. |
| SMPLer-X body-pose supervision | Eq.13 uses robust \(\psi(\theta_b-\hat\theta_b)\) | R `fitting.py` L531–533; decoded AA split at first 11 joints | `psmplx_bodyGT` from SMPLer-X; **L1 sum**, no GMoF | Core `1200`, non-core `1200`, all 3 stages | Yes | `CONTRADICTORY`: source matches, distance does not; code comment claiming later weights are zero is contradicted by YAML. |
| Hand latent Gaussian prior | Eq.14 and Eq.3 | R `fitting.py` L543–590; left/right latent tensors | Zero-mean unit Gaussian; independent \(\lVert z_L\rVert^2,\lVert z_R\rVert^2\) | `[0,4.78²,4.78²]`; active hands only | Yes after stage 0 | `CONSISTENT` for active latent regularization. |
| HaMeR hand-pose supervision | Eq.14 uses robust \(\psi(\theta_h-\hat\theta_h)\) | R `data_parser.py` L415–440; `fitting.py` L547–590 | HaMeR rotation matrices converted to 45-D AA; code adds **both** L1 sum and GMoF sum | Each target component weighted `1200` in every stage | Yes | `CONTRADICTORY`: target source matches, but the code double-counts L1 + robust loss rather than the stated single robust term. |
| HaMeR 3D-hand supervision | Paper broadly lists 3D hand parameters as pseudo-GT (p.3 §3; p.5 §3.4); no separate Eq.12 term | R `fitting.py` L443–501, L524–525 | Only Z coordinates of selected mapped joints; subtract first selected depth, then standardize; GMoF | `data_3d_weights=[0,0,0]` | **No** | `NOT IMPLEMENTED` in the default objective (code exists but weight is zero). |
| Interpenetration | Eq.4; \(L_{pen}\) in Eq.12 | R `fit_single_frame.py` L263–295; `fitting.py` L634–652 | BVH collision pairs + conic distance-field penetration | `[0.5,1.0,1.5]`; cone height `0.0001`; max collisions 128 | Yes; CUDA required | `PARTIALLY CONSISTENT`: same family of penalty; exact library distance/normalization is not shown to equal Eq.4 term-for-term. |
| Explicit contact matching | Paper contribution language refers to “contact-aware terms” (p.2), but Eq.12 specifies penetration only | No hand–hand/hand–body contact target or attraction term; collision is repulsive | None | None | No | `NOT IMPLEMENTED` as a distinct contact-preservation objective. |
| Body temporal consistency | Eq.15: robust difference from prior frame body pose | R `fitting.py` L441, L499, L654–657; carry in `main.py` L305–330 | GMoF of current vs prior 21×3 body AA; first target is SMPLer-X pose | Hard-coded `2000`, every stage | Yes when `use_hposer3d=True` | `PARTIALLY CONSISTENT`: pose-difference intent matches; code directly penalizes AA parameters and couples term availability to the hand-prior flag. |
| Body biomechanics in fitting | Eq.11 and Eq.12: squared violation outside bounds, six joints | R `body_constants.py` L4–17; `utils.py` L221–265; `fitting.py` L512–517, L654–657 | Six body joints (`body_pose[:,15:]`) converted AA → intrinsic XYZ Euler; bounds in radians | `100` all stages | Yes | `CONTRADICTORY`: code uses the sum of two **mean linear hinge** violations, not the squared norm in Eq.11. |
| Hand biomechanics in fitting | \(L_{hbiomech}\) in Eq.12; Eq.11 over 15 hand joints | No term in R `fitting.py::SMPLifyLoss.forward` L430–664; no YAML weight | None | None | No | `NOT IMPLEMENTED` |
| SMPLify angle/bending prior | Not identified in Eq.12 | R `prior.py::SMPLifyAnglePrior` L53–89; `fitting.py` L613–617 | Exponential penalty on four selected elbow/knee angle components | `3.17 × 4.78 = 15.1526` all stages | Yes | `NOT DOCUMENTED` (code-only relative to DexAvatar objective). |
| Shape L2 prior | Not in Eq.12 | R `fitting.py` L611–612; generic `L2Prior` L92–97 | Fixed per-sign mean `betas` | `5²=25` all stages | Present but gradient-constant | `NOT DOCUMENTED`; no effect on optimized latents. |
| Expression L2 prior | Not in Eq.12 | R `fitting.py` L621–626 | Fixed SMPLer-X expression | `[0²,5²,5²]` | Present but gradient-constant | `NOT DOCUMENTED`; no effect on optimized latents. |
| Jaw L2 prior | Not in Eq.12 | R `fitting.py` L628–632; YAML L94–99 | Fixed SMPLer-X jaw AA | YAML contains three repeated vectors `47.8,478,478` | Present but gradient-constant | `NOT DOCUMENTED`; no effect on optimized latents. |
| Generic inactive-hand L2 prior | Not separately described | R `fitting.py` L565–608; `prior.py` L92–97 | Stored inactive SMPL-X hand parameter; omitted active decoder latent | Hand prior stage weights | Present for inactive hand, but zero/fixed | `NOT DOCUMENTED`; optimization-inert under default one-hand reset. |
| Camera initialization loss | Not in DexAvatar Eq.12 | R `fit_single_frame.py` L329–365; `fitting.py` L668–719 | `guess_init` depth and 2D initialization joints | Constructed only | No; never passed to optimizer loop | `NOT IMPLEMENTED` in actual fitting loop. |

`[VERIFIED]` If the disabled HaMeR 3D term were assigned nonzero weight, its index comments would not match the actual post-mapping joint order. After the SMPL-X/COCO intersection, indices `12:42` contain 15 non-tip joints from **each** hand, while `53:63` contains five fingertips from **each** hand. Consequently the one-hand `right_hand` branch (`53:63`) selects both hands’ tips, and the `left_hand` branch (`12:42`) selects both hands’ non-tip joints. This has no effect in the default YAML because `data_3d_weight=0`. (R `fitting.py` L452–501; `assets/mapping_func.py::get_mapping` L186–256; `assets/joint_mapping.py` L1–209; YAML L54–57.)

### 4.3 Prior-training objective versus fitting objective

| Training loss | Paper formula / source | SignBPoser weight | SignHPoser weight | Public code verification | Confidence |
|---|---|---:|---:|---|---|
| KL | Eq.6, \(KL(q(Z\mid R)\parallel\mathcal N(0,I))\) | 0.001 | 0.0001 | Training code absent | `[VERIFIED]` paper-only (P p.4 Eq.5–6). |
| Axis-angle reconstruction | Eq.7, \(\lVert\alpha-\hat\alpha\rVert_2^2\) | 0.999 | 0.999 | Training code absent | `[VERIFIED]` paper-only (P pp.4–5). |
| Mesh reconstruction | Eq.8, \(\lVert M-\hat M\rVert_2^2\) | 0.999 | 0.999 | Training code absent | `[VERIFIED]` paper-only (P p.5). |
| Rotation orthogonality | Eq.9, \(\lVert\hat R\hat R^\top-I\rVert_2^2\) | 0.01 | 0.01 | Training code absent | `[VERIFIED]` paper-only (P p.5). |
| Parameter regularization | Eq.10, \(\lVert\phi\rVert_2^2\) | 0.0001 | 0.0001 | Training code absent | `[VERIFIED]` paper-only (P p.5). |
| Biomechanics | Eq.11; used only in `+bio` variants according to Tables S1–S2 | 1.5 selected | 1.5 selected | Training code absent | `[VERIFIED]` paper-only (P p.4 Eq.5; p.5 Eq.11; p.13 Tables S1–S2). |

`[VERIFIED]` These four concepts are distinct in the evidence:

- **training-data filtering/rectification** changes which/corrected poses train the prior (BPf/HPf); it is not a fitting loss. (P pp.3–5 §3.2, Figs.3–4.)
- **training biomechanics** is Eq.11 inside Eq.5 for the BPf+bio/HPf+bio prior variants. (P p.4 Eq.5; p.13 Tables S1–S2.)
- **latent Gaussian regularization during fitting** penalizes the optimized latent norm. (P p.6 Eq.13–14; R `fitting.py` L527–590.)
- **prior supervision toward SMPLer-X/HaMeR** penalizes decoded pose against detector/regressor pose targets; it is separate from latent regularization. (P p.6 Eq.13–14; R `fitting.py` L531–590.)

---

## 5. SignBPoser / SignHPoser audit

| Property | SignBPoser | SignHPoser | Evidence / confidence |
|---|---|---|---|
| Scientific role | Sign-language body pose manifold | Sign-language hand pose manifold | `[VERIFIED]` P p.2 contributions; p.4 §3.3. |
| Training source | Subset of SignAvatars 3D reconstructions from How2Sign | New physical capture: 9 Vicon cameras + Manus gloves, 8 signers, 93 fingerspelled words | `[VERIFIED]` P pp.2–4. |
| Ground-truth character | Pseudo-ground-truth reconstructed body data | Mocap-derived, then retargeted to SMPL-X and rectified; not direct native SMPL-X measurements | `[VERIFIED]` P p.3 §3.2.1; p.4 §3.2.2; pp.9–11 §S3. |
| Preprocessing | Reject frames outside biomechanical ROM/signer-space bounds for shoulders, elbow/forearm and wrists | Correct 15 hand joints in bending/splaying/twisting after aligning MANO axes to biomechanical convention | `[VERIFIED]` P pp.3–5, Figs.3–4; p.9 §S2. |
| Joints represented | Code target/output: 21 SMPL-X body joints, 63 AA values; biomechanics focuses on six upper-limb joints | Code target/output: 15 MANO/SMPL-X hand joints per hand, 45 AA values | `[VERIFIED]` R `fit_single_frame.py` L255–262; `fitting.py` L514–517, L527–590. |
| Network input representation | Paper says per-joint rotation matrix \(R\in\mathbb R^{3\times3}\) | Same paper formulation | `[VERIFIED]` as paper statement only: P p.4 Eq.6. Code training path is unavailable. |
| Reconstruction/output representations | Paper trains AA reconstruction plus rotation-matrix orthogonality; fitting calls decoder `output_type='aa'` | Same; fitting outputs 15×3 AA per hand | `[VERIFIED]` P pp.4–5 Eq.7–9; R `fitting.py` L248–277. |
| Latent dimension | 33 selected and hard-coded in fitting | 23 selected for HPf/HPf+bio and hard-coded; HPu sweep prefers 24 | `[VERIFIED]` P p.13 Tables S1–S2; R `fit_single_frame.py` L223–239. |
| Paper latent notation | Eq.6 says \(Z\in\mathbb R^{33}\) without distinguishing hand; Table S2 varies hand latent 22/23/24 | Table S2 and code establish 23 for selected filtered hand prior | `[VERIFIED]` internal documentation inconsistency: P p.4 Eq.6 vs p.13 Table S2. |
| Architecture | Encoder-decoder VAE; paper reports three linear layers and 512 embedding/neuron size | Same | `[VERIFIED]` paper-only: P p.6 Implementation Details; p.13 neuron column. |
| Optimizer / LR for training | Adam, \(10^{-3}\) | Adam, \(10^{-3}\) | `[VERIFIED]` paper-only: P p.6 Implementation Details. |
| Training loss | Eq.5–11; BPu, BPf and BPf+bio isolate raw/filtering/+biomech | Eq.5–11; HPu, HPf and HPf+bio isolate raw/rectification/+biomech | `[VERIFIED]` P pp.4–5; p.13 Tables S1–S2. |
| Biomechanics stages | Filtering in preprocessing; optionally loss in `+bio` training; body biomech also exists in fitting | Rectification in preprocessing; optionally loss in `+bio` training; **no hand-biomech fitting term in code** | `[VERIFIED]` P pp.3–5; R `fitting.py` L430–664. |
| Fitting use | Body pose is **parameterized on** the decoder manifold; latent optimized directly, with latent norm and SMPLer-X target | Each active hand is **parameterized on** the decoder manifold; latent optimized directly, with latent norm and HaMeR target | `[VERIFIED]` R `fitting.py` L244–277, L527–590. This is stronger than merely evaluating a regularizer on a free pose. |
| Left/right sharing | N/A body | One `hposer3d` model/checkpoint is loaded; left and right have distinct 23-D latent tensors but share decoder weights | `[VERIFIED]` R `fit_single_frame.py` L231–243; `fitting.py` L258–277. |
| Training code public in inspected commit | No | No | `[VERIFIED]` no trainer/model definition under tracked repository; runtime loaders dynamically import model files expected inside downloaded directories. R `test_bposer.py` L21–43; `test_hposer.py` L21–43. |
| Checkpoint public status | README advertises a Google Drive download, but directory is absent from Git and external contents could not be inspected | Same | `[UNRESOLVED]`; R `README.md` L84–90. Advertised availability is not a verified artifact. |
| Exact training dataset public | Exact selected/filtered training tensors and split are absent | Captured dataset and exact retargeted/rectified tensors are absent | `[VERIFIED]` absent from inspected repo; `[UNRESOLVED]` whether hosted elsewhere. |
| Exact retraining possible from supplied material | No: exact examples/splits/filter implementation/model file/trainer are missing | No: raw/retargeted data, rectifier, split, model file and trainer are missing | `[VERIFIED]` repository-content audit; P does not provide these artifacts. |

### 5.1 Loader behavior and reproducibility implication

`[VERIFIED]` The repository loaders choose the most recently modified `snapshots/*.pt`, load the first `.ini`, dynamically import `signbposer.py` or `signhposer*.py` from the external experiment directory, instantiate using `num_neurons`, `latentD`, and `data_shape`, and load the state dictionary. Exact architecture and checkpoint identity therefore depend on files absent from the commit and on filesystem modification time rather than an immutable filename. (R `test_bposer.py` L4–43; `test_hposer.py` L4–43.)

`[UNRESOLVED]` The following paper-reported details cannot be verified against training code: rotation-conversion convention at VAE input/output, per-layer activation/normalization, train/dev/test split construction, batch size/epochs, mesh layer settings, loss reduction, preprocessing thresholds, and random seeds.

---

## 6. Exact TR-V2V specification from E

### 6.1 Authority and evaluated objects

`[VERIFIED]` E, not the paper wording, determines the protocol audited here. It loads ASCII OBJ vertices/faces, optionally rotates predictions only when `method == 'pixie'`, and asserts exact equality of the predicted and GT face-index arrays. (E `load_obj` L99–131; `read_verts_and_faces` L134–149; `main` L355–370.)

`[VERIFIED]` E assumes SMPL-X topology through `np.arange(0,10475)` and external SMPL-X/MANO masks. The repository’s export template also has 10,475 vertices. (E module `__main__` L534–567; R `assets/smplx_uv_new.obj`; P p.9 §S1.)

### 6.2 Sign and frame selection

Let the sign file define an insertion-ordered mapping \(s\mapsto c_s\), subsequently sorted lexicographically by sign name. Let `sign_seg[s]=[a_s,b_s]`. E constructs:

\[
G_s=\operatorname{numericSort}\left\{\text{GT OBJ }n:\;2a_s\le n\le2b_s,\ n\text{ exists}\right\},
\]

and

\[
P_s=\operatorname{sortFirstDigit}\{\text{prediction OBJ in }s/\texttt{smplifyx/meshes}\}.
\]

It then pairs \((P_s[i],G_s[i])\) for \(i=0,\ldots,|G_s|-1\). (E `load_gt_obj_paths` L231–265; `load_mocap_obj_paths` L268–280; `main` L317–361.)

- `[VERIFIED]` `--central` is required and assigned to `central`, but never read thereafter. Passing `true` or `false` has identical behavior. (E module CLI L479–512; no other reference.)
- `[VERIFIED]` `--sign_seg` is always loaded and applied. (E `main` L295–307; module CLI L499–502, L588.)
- `[VERIFIED]` Both endpoints are inclusive because Python uses `range(start_idx, end_idx + 1)`. (E `load_gt_obj_paths` L237–249.)
- `[UNRESOLVED]` No inspected source explains why each segment index is multiplied by two. `[INFERENCE]` It may map video-frame indices to a GT naming cadence, but this is not author-confirmed and is not encoded as a validated mapping.
- `[VERIFIED]` Pairing is not by filename or semantic frame ID. A missing GT file is silently omitted; a missing prediction in the middle shifts all subsequent positional pairs. If predictions are shorter than GT selection, indexing raises an error; prediction extras are ignored. (E `load_gt_obj_paths` L243–249; `main` L342–361.)
- `[VERIFIED]` The paper states 2,872 frames. The 57 repository intervals yield 2,872 under \(\sum_s(b_s-a_s)\), 2,929 under \(\sum_s(b_s-a_s+1)\), and 5,801 for a dense inclusive integer range after doubling. Because E filters by existing GT filenames, its actual total cannot be computed without the GT directory manifest. (P p.6 §4; R `segment.json`; E `load_gt_obj_paths` L237–249.)

### 6.3 Vertex sets

| Region key in E | Set definition | Exact membership/count known? | One-handed class-`0` rule | Evidence |
|---|---|---|---|---|
| `all` | \(\{0,\ldots,10474\}\) | Yes: 10,475 | Left-hand IDs removed | E module `__main__` L554–558; `main` L380–395. |
| `left hand` | `mano_data['left_hand']` | **No**: asset not supplied | Entire metric skipped | E module `__main__` L536–548; `main` L391–395. |
| `right hand` | `mano_data['right_hand']` | **No**: asset not supplied | Left set difference applied but normally leaves right set unchanged | E module `__main__` L536–558; `main` L380–395. |
| `above pelvis upper body` | external `upper_body.npy` | **No**: asset not supplied | Left-hand IDs removed | E module `__main__` L550–567; `main` L380–395. |
| `above pelvis minus head` | external `upper_body_minus_head.npy` | **No**: asset not supplied | Left-hand IDs removed | Same. |
| `above pelvis minus face` = UBody(-F) | external `upper_body_minus_face.npy` | **No**: asset not supplied | Left-hand IDs removed | Same. |

`[UNRESOLVED]` `MANO_SMPLX_vertex_ids.pkl`, `upper_body.npy`, `upper_body_minus_head.npy`, `upper_body_minus_face.npy`, and the evaluation `SMPLX_NEUTRAL.npz` were not attached and are not in the DexAvatar commit. Exact hand vertex counts, exact pelvis/head/face boundaries, overlap/disjointness, and mask hashes cannot be verified.

### 6.4 Alignment

For prediction vertices \(p_{siv}\), GT vertices \(g_{siv}\), and a region-specific vertex set \(A_s^r\), define independent subset centroids

\[
\mu^P_{si,r}=\frac{1}{|A_s^r|}\sum_{v\in A_s^r}p_{siv},\qquad
\mu^G_{si,r}=\frac{1}{|A_s^r|}\sum_{v\in A_s^r}g_{siv}.
\]

The per-vertex error is

\[
d_{siv}^{r}=\left\|(p_{siv}-\mu^P_{si,r})-(g_{siv}-\mu^G_{si,r})\right\|_2.
\]

`[VERIFIED]` This is a **per-frame, per-subset, centroid translation alignment**. It is recomputed independently for UBody(-F), left hand, right hand, and every other subset. There is no rotation, scale, Procrustes, pelvis, or wrist-root alignment. (E `transl_point_error` L159–169; `main` L380–395.)

`[VERIFIED]` `align_by_pelvis` and `point_error_center` are defined but unused. The separately logged functions named `V2V left wrist`/`right wrist` call `point_error_common_center`; because the same GT wrist `center` is added to both already centroid-centered point sets, it cancels algebraically and gives the same distances as `transl_point_error` for the same hand set. It is **not** wrist-root alignment. (E `point_error_common_center`, `point_error_center`, `align_by_pelvis` L172–203; `main` L372–411.)

### 6.5 Final metrics and one-hand rules

Let \(S\) be signs actually entered by E. Let \(F_s\) contain positional pair indices whose predicted mesh has no NaN. Define base masks \(U,L,R\). With class label \(c_s\):

\[
U_s=\begin{cases}U\setminus L,&c_s=\texttt{"0"},\\U,&\text{otherwise},\end{cases}
\quad
R_s=\begin{cases}R\setminus L,&c_s=\texttt{"0"},\\R,&\text{otherwise},\end{cases}
\]

and the left-hand sign set \(S_L=\{s\in S:c_s\ne\texttt{"0"}\}\).

The three reported quantities are:

\[
\boxed{\operatorname{UBody(-F)}=1000\,
\frac{\sum_{s\in S}\sum_{i\in F_s}\sum_{v\in U_s}d^{U_s}_{siv}}
{\sum_{s\in S}\sum_{i\in F_s}|U_s|}}
\]

\[
\boxed{\operatorname{LHand}=1000\,
\frac{\sum_{s\in S_L}\sum_{i\in F_s}\sum_{v\in L}d^{L}_{siv}}
{\sum_{s\in S_L}\sum_{i\in F_s}|L|}}
\]

\[
\boxed{\operatorname{RHand}=1000\,
\frac{\sum_{s\in S}\sum_{i\in F_s}\sum_{v\in R_s}d^{R_s}_{siv}}
{\sum_{s\in S}\sum_{i\in F_s}|R_s|}}.
\]

`[VERIFIED]` E multiplies the final mean by 1,000 and labels the result millimetres. `[INFERENCE]` This conversion assumes that input OBJ coordinates are in metres; E does not validate units. The code concatenates all per-frame vertex arrays before one global mean: it is vertex-frame-weighted, not a mean of sign means or frame means. Signs with more retained frames receive greater weight. For UBody(-F), class-`0` frames also have fewer vertices and therefore a smaller contribution than two-hand frames. (E `main` L432–461.)

`[VERIFIED]` For every class-`0` sign, E removes the **left** hand from UBody(-F) and all other non-left subsets and suppresses LHand entirely. E never reads the fitter’s active-side decision. Thus the metric protocol encodes a fixed left-hand exclusion, not a generic “non-dominant-hand” exclusion. (E `main` L380–395; compare R `data_parser.py::__init__` L201–203.)

### 6.6 Implementation-independent pseudocode

```text
read sign_file -> class_by_sign
sort sign names lexicographically
read sign_seg JSON

load external vertex masks: full, left, right, upper,
                            upper-minus-head, upper-minus-face

for each sign s:
    a, b = sign_seg[s]
    gt = numeric-sort(all existing GT OBJ files whose integer stem is in [2a, 2b])
    pred = sort(all predicted OBJ files by first digit run in filename stem)

    for i in 0 .. len(gt)-1:
        P = vertices(pred[i])       # positional pairing only
        G = vertices(gt[i])
        assert faces(pred[i]) == faces(gt[i]) exactly
        if any NaN in P: continue

        U_s = upper_minus_face
        R_s = right
        if class_by_sign[s] == "0":
            U_s = set_difference(U_s, left_hand_ids)
            R_s = set_difference(R_s, left_hand_ids)

        append subset_centroid_translation_error(P[U_s], G[U_s]) to U errors
        append subset_centroid_translation_error(P[R_s], G[R_s]) to R errors

        if class_by_sign[s] != "0":
            append subset_centroid_translation_error(P[left], G[left]) to L errors

return 1000 * mean(concatenate(U errors)),
       1000 * mean(concatenate(L errors)),
       1000 * mean(concatenate(R errors))
```

### 6.7 NaNs, missing data, hard-coded state, and edge cases

| Behavior / edge case | Exact effect | Evidence | Confidence |
|---|---|---|---|
| Predicted NaN | Entire paired frame skipped for all regions | E `main` L364–366 | `[VERIFIED]` |
| GT NaN | Not checked; propagates to distance and ordinary `.mean()`, potentially making final metric NaN | E `main` L360–370, L455–461 | `[VERIFIED]` |
| Missing GT within range | Silently omitted before positional pairing | E `load_gt_obj_paths` L243–249 | `[VERIFIED]` |
| Missing prediction within sequence | Later prediction/GT pairs shift; if list becomes too short, `IndexError` | E `main` L342–361 | `[VERIFIED]` |
| Extra predictions | Ignored after \(\lvert G_s\rvert\) pairs | E `main` L342–361 | `[VERIFIED]` |
| Empty GT segment | Later `gt_objs[idx][0]` print crashes | E `main` L442 | `[VERIFIED]` |
| Topology mismatch | Exact face-array assertion fails; there is no explicit vertex-count assertion, but mask indexing assumes 10,475 | E `main` L369–370; module `__main__` L554–567 | `[VERIFIED]` |
| Method label `pixie` | Applies a 180° X rotation around the predicted mesh centroid; other labels do not | E `read_verts_and_faces` L142–149 | `[VERIFIED]` |
| `--method` default | Declared `nargs='+'` but default is a string; omitting it makes the final loop iterate characters | E module CLI L473–477, L575–588 | `[VERIFIED]` |
| Class globals | `class_sign` and `left_hand_ids` are read as globals inside `main`, not passed explicitly | E `main` L380–395; module `__main__` L515–548 | `[VERIFIED]` |
| Asset path | Evaluation masks and SMPL-X model are loaded from hard-coded `/home/haipd/...` | E module `__main__` L534–567 | `[VERIFIED]` |
| Unused/misleading variables | `central`, returned `segment_indices`, `mesh_ids`, `save_vis`, `METHOD_PATH_DICT`, `METHOD_GET_PATH_DICT`, `point_error_center`, `align_by_pelvis`, and commented similarity-transform import do not affect reported TR metrics | E module globals L22–96; helper definitions L183–203; `load_gt_obj_paths` L231–265; `main` L318–320; module CLI L479–532 | `[VERIFIED]` |
| Region centering | A rigid translation error of any magnitude is invisible; a global rotation/scale error remains | Direct consequence of E `transl_point_error` L159–169 | `[VERIFIED]` mathematically. |
| Different region masks | Each region chooses its own best translation, so UBody and hand metrics are not measured in one common aligned coordinate frame | E `main` L380–395 | `[VERIFIED]` |
| One-hand mask size | UBody(-F) aggregation weights class-`0` frames by fewer vertices after left-hand removal | E `main` L382–395, L455–461 | `[VERIFIED]` |

---

## 7. Quantitative results and ablations

**Result labels used:** `REPORTED` = transcribed from paper; `RECOMPUTED` = arithmetic performed on printed values; `REPRODUCED` would require executing the exact pipeline/data/protocol. No value in this dossier is `REPRODUCED`.

### 7.1 Main TR-V2V table

All values below are **author-reported** in millimetres; none is reproduced in this audit. Source: P p.6 Table 1.

| Method | UBody(-F) ↓ | LHand ↓ | RHand ↓ | Result status |
|---|---:|---:|---:|---|
| FrankMoCap | 78.07 | 20.47 | 19.62 | `REPORTED`, `[VERIFIED]` transcription |
| PIXIE | 60.11 | 25.02 | 22.42 | `REPORTED`, `[VERIFIED]` transcription |
| PyMAF-X | 68.61 | 21.46 | 19.19 | `REPORTED`, `[VERIFIED]` transcription |
| SMPLify-SL | 56.07 | 22.23 | 18.83 | `REPORTED`, `[VERIFIED]` transcription |
| SGNify | 55.63 | 19.22 | 17.50 | `REPORTED`, `[VERIFIED]` transcription |
| OSX | 47.32 | 18.34 | 18.12 | `REPORTED`, `[VERIFIED]` transcription |
| Neural Sign Actors | 46.42 | 16.17 | 15.23 | `REPORTED`, `[VERIFIED]` transcription |
| EVA* | **40.38** | **13.73** | **13.68** | `REPORTED`; numerically second-best in all three columns |
| DexAvatar | **30.13** | **13.53** | **13.08** | `REPORTED`; best in all three columns |

`[VERIFIED]` The numeric runner-up in Table 1 is EVA*, not Neural Sign Actors. Relative improvement over the rounded EVA* values is 25.38% UBody(-F), 1.46% LHand, and 4.39% RHand. These are `RECOMPUTED` arithmetic, not reproduced model results.

### 7.2 Audit of the 35.11% claim

The paper explicitly compares DexAvatar to **Neural Sign Actors**, using the relative error reduction

\[
100\frac{E_{\mathrm{NSA}}-E_{\mathrm{Dex}}}{E_{\mathrm{NSA}}}.
\]

| Region | Rounded values used | Arithmetic from rounded values | Paper text | Audit status |
|---|---|---:|---:|---|
| UBody(-F) | 46.42 → 30.13 | 35.0926% → **35.09%** | 35.11% | `RECOMPUTED`; `[VERIFIED]` 0.02 pp mismatch |
| LHand | 16.17 → 13.53 | 16.3265% → **16.33%** | 16.32% | `RECOMPUTED`; `[VERIFIED]` 0.01 pp mismatch |
| RHand | 15.23 → 13.08 | 14.1169% → **14.12%** | 14.11% | `RECOMPUTED`; `[VERIFIED]` 0.01 pp mismatch |

`[INFERENCE]` The small discrepancies are compatible with percentages computed from higher-precision, unpublished scores and a table rounded to two decimals. `[UNRESOLVED]` The unrounded per-method values and aggregation outputs are unavailable, so that explanation cannot be confirmed. (P p.6 Table 1 and §5.1.)

### 7.3 SignBPoser fitting ablation

Source: P p.6 Table 2 and p.7 §5.1. Values are author-reported TR-V2V-like vertex errors; the table does not restate units.

| Variant | FBody ↓ | UBody ↓ | UBody(-H) ↓ | UBody(-F) ↓ | Meaning stated by paper | Status |
|---|---:|---:|---:|---:|---|---|
| BPu | 43.18 | 29.95 | 44.72 | 34.06 | Prior trained on unfiltered pseudo-GT | `REPORTED` |
| BPf | 42.32 | 26.78 | 41.35 | 30.28 | Prior trained on biomechanically filtered pseudo-GT | `REPORTED` |
| BPf+bio | 42.38 | 26.93 | 41.88 | 30.44 | Filtered data + biomechanics in prior training | `REPORTED` |

Recomputed from the rounded table:

| Comparison | FBody | UBody | UBody(-H) | UBody(-F) | Paper text | Audit |
|---|---:|---:|---:|---:|---|---|
| BPu → BPf relative reduction | 1.99% | 10.58% | 7.54% | 11.10% | 2.0%, 10.6%, 7.5%, 11.1% | `[VERIFIED]` matches rounding |
| BPf → BPf+bio relative degradation | 0.14% | 0.56% | 1.28% | 0.53% | +0.14%, +0.56%, +1.28%, +0.53% | `[VERIFIED]` matches rounding |

`[VERIFIED]` The data support the narrow claims that BPf is lower than BPu on all four reported subsets and that BPf+bio is slightly worse than BPf on all four. They do not isolate whether the difference is due to retained sample count, changed pose distribution, filtering rules, or training randomness. (P pp.6–7.)

`[VERIFIED]` The paper separately reports—but does not tabulate—adding body biomechanics **during optimization** to BPf: relative reductions of 0.17% FBody, 0.37% UBody, 0.05% UBody(-H), and 0.33% UBody(-F). No absolute scores, variance, seeds, or configuration are provided, so only the percentages are `REPORTED`; they cannot be recomputed or reproduced here. (P p.7 §5.1.)

### 7.4 SignHPoser fitting ablation

Source: P p.7 Table 3 and §5.1.

| Variant | UBody(-F) ↓ | LHand ↓ | RHand ↓ | Meaning stated by paper | Status |
|---|---:|---:|---:|---|---|
| HPu | 31.34 | 14.19 | 13.92 | Prior trained on uncorrected hand data | `REPORTED` |
| HPf | 30.17 | 13.55 | 13.06 | Prior trained on corrected hand data | `REPORTED` |
| HPf+bio | 30.13 | 13.53 | 13.08 | Corrected data + biomechanics | `REPORTED` |

| Comparison | UBody(-F) | LHand | RHand | Paper text | Audit |
|---|---:|---:|---:|---|---|
| HPu → HPf relative reduction | 3.73% | 4.51% | 6.18% | 3.7%, 4.5%, 6.2% | `[VERIFIED]` matches rounding |
| HPf → HPf+bio | 0.1326% improvement | 0.1476% improvement | 0.1531% degradation | 0.13%, 0.15%, 0.2% degradation | `[VERIFIED]` direction matches; rounded RHand arithmetic gives 0.15%, not 0.2% |

`[VERIFIED]` The table supports that correction (HPf) improves all three reported metrics over HPu and that HPf+bio slightly improves UBody(-F)/LHand while slightly degrading RHand relative to HPf. `[UNRESOLVED]` The paper attributes the latter to a biomechanical regularizer “in the fitting process,” while its variant definition says HPf+bio is trained with hand biomechanics, and the inspected fitting code has no hand-biomechanical term. The experimental intervention represented by Table 3 is therefore not recoverable from the released code. (P p.7 §5.1; R `fitting.py` L430–664.)

### 7.5 Prior hyperparameter sweeps

All values are `REPORTED`; MPJPE/MPVPE units are not specified in the table captions. Source: P p.13 Tables S1–S2.

#### SignBPoser (Table S1)

| Variant | KL | Neurons | Latent | Biomech constant | DEV MPJPE | DEV MPVPE | TEST MPJPE | TEST MPVPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BPu | .001 | 512 | 33 | none | 5.87 | 3.73 | 5.69 | 3.62 |
| BPu | .001 | 512 | 32 | none | 5.99 | 4.17 | 5.98 | 4.21 |
| BPu | .001 | 512 | 31 | none | 7.56 | 5.05 | 7.28 | 5.00 |
| BPf | .001 | 512 | 33 | none | 7.21 | 4.33 | 7.04 | 4.14 |
| BPf | .001 | 512 | 32 | none | 7.45 | 4.68 | 7.43 | 4.32 |
| BPf | .001 | 512 | 31 | none | 7.37 | 4.41 | 7.17 | 4.24 |
| BPf+bio | .001 | 512 | 33 | .5 | 7.42 | 4.43 | 7.21 | 4.32 |
| BPf+bio | .001 | 512 | 33 | 1.5 | 7.30 | 4.39 | 7.10 | 4.25 |
| BPf+bio | .001 | 512 | 33 | 2.5 | 7.37 | 4.42 | 7.29 | 4.29 |

`[VERIFIED]` Within each reported SignBPoser variant, Table S1 supports selecting latent 33 for BPu/BPf and biomech constant 1.5 for BPf+bio. `[VERIFIED]` Across variants, however, BPu has lower prior TEST errors than BPf/BPf+bio, even though BPf yields better downstream DexAvatar Table 2 errors. Therefore prior reconstruction error and final TR-V2V do not order variants identically. (P pp.12–13.)

#### SignHPoser (Table S2)

| Variant | KL | Neurons | Latent | Biomech constant | DEV MPJPE | DEV MPVPE | TEST MPJPE | TEST MPVPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HPu | .0001 | 512 | 24 | none | .56 | .55 | .56 | .54 |
| HPu | .0001 | 512 | 23 | none | .57 | .58 | .57 | .55 |
| HPu | .0001 | 512 | 22 | none | .59 | .58 | .60 | .58 |
| HPf | .0001 | 512 | 24 | none | .40 | .38 | .40 | .38 |
| HPf | .0001 | 512 | 23 | none | .37 | .35 | .37 | .35 |
| HPf | .0001 | 512 | 22 | none | .38 | .40 | .38 | .40 |
| HPf+bio | .0001 | 512 | 23 | .5 | .40 | .41 | .40 | .41 |
| HPf+bio | .0001 | 512 | 23 | 1.5 | .39 | .38 | .39 | .38 |
| HPf+bio | .0001 | 512 | 23 | 2.5 | .43 | .45 | .43 | .45 |

`[VERIFIED]` Table S2 supports latent 24 for HPu, latent 23 for HPf, and weight 1.5 within HPf+bio. It also shows HPf latent-23 prior errors lower than HPf+bio weight-1.5 errors, whereas final Table 3 has mixed HPf+bio changes. (P pp.12–13.)

### 7.6 SignHPoser with VPoser body prior

Source: P p.13 Table S3. All values are `REPORTED`.

| Variant | UBody(-F) ↓ | LHand ↓ | RHand ↓ |
|---|---:|---:|---:|
| HPu | 37.25 | 13.56 | 14.53 |
| HPf | 36.79 | 13.39 | 14.06 |
| HPf+bio | 36.77 | 13.37 | 13.82 |

`[VERIFIED]` These values support the paper’s stated directions: HPf improves over HPu; HPf+bio further improves all three in this VPoser-body setting. They do not isolate detector, initialization, or temporal effects. (P p.12 §S5 and p.13 Table S3.)

### 7.7 Missing causal ablations

| Claimed/implemented component | Isolated ablation available in paper/supplement? | What remains unidentified | Confidence |
|---|---|---|---|
| SMPLer-X pose initialization/target | No | Initialization versus supervision versus fixed non-pose parameters | `[VERIFIED]` absence in P Tables 1–3/S1–S3 |
| Sapiens detector | No | Detector contribution independent of optimizer/prior | `[VERIFIED]` |
| HaMeR detector/pose target | No | 2D replacement versus pose target versus disabled 3D term | `[VERIFIED]` |
| Temporal term | No | Per-frame gain and sequence drift contribution | `[VERIFIED]` |
| One-hand class/active-side decision | No isolated row | Class masking versus active-side heuristic versus evaluation exclusion | `[VERIFIED]` |
| SignBPoser versus no SignBPoser | No no-prior row | Absolute contribution of the body manifold | `[VERIFIED]` |
| SignHPoser versus no hand prior | No no-prior row; VPoser comparison changes body prior | Absolute contribution of the hand manifold | `[VERIFIED]` |
| Body biomechanics during fitting | Percentages reported, table/config absent | Exact absolute scores and intervention | `[VERIFIED]` |
| Hand biomechanics during fitting | Claimed in text, absent in inspected code | Whether experimental code differed from release or the text refers to training | `[UNRESOLVED]` |

`[VERIFIED]` No statistical uncertainty, repeated-seed variance, confidence interval, or significance test accompanies the reported fitting results. (P pp.6–7, Tables 1–3.)

---

## 8. Paper–code–evaluation consistency matrix

| Topic | Paper says | Code does | Evaluation assumes | Status | Consequence |
|---|---|---|---|---|---|
| Task/input | Monocular sign-language videos → 3D whole-body mesh | Processes frame folders per sign through four stages | Reads exported per-frame OBJ meshes | `CONSISTENT` | Baseline task identity is stable across sources. |
| Output representation | SMPL-X pose/mesh | Writes OBJ plus parameter pickle | Uses only OBJ | `PARTIALLY CONSISTENT` | Evaluation does not validate saved parameter correctness. |
| Saved pose parameters | Framework recovers body and hand poses | Pickle stores optimized body pose but stored hand fields are targets, not decoded optimized hands; hand latents absent | Not checked | `CONTRADICTORY` | Pickle cannot exactly regenerate evaluated hand mesh. |
| Export coordinates | Not documented as a 180°-rotated artifact | Rotates OBJ 180° about X after model forward | Applies no extra rotation unless method label is `pixie` | `NOT DOCUMENTED` | Parameter-space and mesh-space outputs use different frames. |
| Pose initialization | SMPLer-X/HaMeR estimates initialize DexAvatar | Active latents are zeroed; decoder-at-zero supplies initial active pose; estimates act as targets | Not observed | `PARTIALLY CONSISTENT` | “Initialization” and “supervision” are not interchangeable in the implementation. |
| Optimization variables | Eq.13–14 expose body/hand latents but the paper does not enumerate which remaining SMPL-X/camera terms are fixed | Optimizer contains only body and active-hand latents | Evaluator sees mesh only | `NOT DOCUMENTED` | Global orientation, shape, face, translation and camera cannot be assumed optimized from paper wording. |
| Camera treatment | Camera estimates are pseudo-GT inputs/initialization | SMPLer-X `K` and translation are fixed; constructed camera-init loss is unused | Per-subset centering removes translation at scoring time | `PARTIALLY CONSISTENT` | Camera error can affect pose fitting, while final subset translation error is discarded. |
| Body prior | SignBPoser is a differentiable regularizer with SMPLer-X supervision | Body pose is constrained to decoder manifold and latent is the free variable | Evaluates resulting vertices | `PARTIALLY CONSISTENT` | Code uses a hard manifold parameterization, not a free pose plus prior-only penalty. |
| Hand prior | Independent L/R latents, HaMeR target | Independent latents share one decoder/checkpoint | Evaluates hand vertex subsets separately | `CONSISTENT` | Left/right have independent codes but not separate learned manifolds. |
| Body target distance | Robust \(\psi\) in Eq.13 | L1 sum | N/A | `CONTRADICTORY` | Actual objective landscape differs from the equation. |
| Hand target distance | One robust \(\psi\) term in Eq.14 | Adds L1 and GMoF terms | N/A | `CONTRADICTORY` | HaMeR target is double-counted relative to Eq.14. |
| 3D HaMeR supervision | 3D hand parameters listed as pseudo-GT inputs | Z-only standardized term exists but default weights are zero | N/A | `PARTIALLY CONSISTENT` | Default results cannot be attributed to active HaMeR 3D-joint loss. |
| 2D joint loss | Confidence/weight × robust reprojection, mean over joints | Robust sum with squared confidence/weights and image-height scaling | N/A | `PARTIALLY CONSISTENT` | Equation does not fully specify code reduction/scaling. |
| Lower body | Lower-body \(\omega_i=0\) | Zeroes lower-body 2D weights | UBody-only masks | `CONSISTENT` | Direct 2D lower-body evidence is excluded; body latent/angle prior still couple pose. |
| Interpenetration | Eq.4 conic signed-distance concept | BVH + distance-field penetration loss | Vertex error does not measure penetration directly | `PARTIALLY CONSISTENT` | Plausibility and TR-V2V can disagree. |
| Temporal term | Robust previous-frame body-pose consistency | GMoF AA difference ×2000; no hand temporal term; enabled only inside hand-prior total branch | Frames evaluated independently after reconstruction | `PARTIALLY CONSISTENT` | Temporal behavior is body-only and implementation-coupled to `use_hposer3d`. |
| Body biomechanics | Squared out-of-range violation | Mean linear hinge over six Euler-converted joints | Not measured directly | `CONTRADICTORY` | Reported Eq.11 is not the released fitting penalty. |
| Hand biomechanics in fitting | Explicit \(L_{hbiomech}\) in Eq.12 | No such term | Not measured directly | `NOT IMPLEMENTED` | Released default cannot realize the stated hand-biomechanical fitting contribution. |
| One-/two-hand classifier | Uses classifier from SGNify; disables non-dominant arm and hand | Uses static class file for one/two and a Sapiens wrist-speed heuristic for side; zeros 2D weights | For class `0`, always removes left hand and skips LHand | `CONTRADICTORY` | Fitter and evaluator need not agree on the hand considered inactive. |
| Central frames | 57 signs, 2,872 central frames | Fitter uses inclusive `[a,b]` | E ignores `--central`, always uses inclusive `[2a,2b]` intersected with existing GT | `CONTRADICTORY` | Paper count is not enforced; actual evaluated count is data-manifest dependent. |
| Frame correspondence | “Central frames” protocol | Outputs filenames inherited from images | Positional pairing after separate sorts; no name assertion | `NOT DOCUMENTED` | Missing frames can silently compare different times. |
| UBody(-F) definition | Above pelvis, excluding face | No evaluation masks in repository | Loads hard-coded external `.npy` mask | `UNRESOLVED` | Exact vertex membership/count cannot be audited. |
| TR alignment | Paper names TR-V2V but does not state algorithm in main text | N/A | Independently subtracts each subset centroid per frame | `NOT DOCUMENTED` | Metric is translation-only aligned separately per region, not root-relative or PA. |
| Aggregation | Mean V2V wording | N/A | Global mean over concatenated vertex-frame samples | `NOT DOCUMENTED` | Signs are weighted by retained frame count; UBody also by class-dependent vertex count. |
| Topology | SMPL-X has 10,475 vertices | Export template has 10,475 vertices | Full mask is 0:10475 and faces must exactly match | `CONSISTENT` | Same-topology correspondences are required. |
| 35.11% claim | Improvement over Neural Sign Actors | No result generation script/output supplied | E alone cannot reproduce without assets/data | `PARTIALLY CONSISTENT` | Rounded table yields 35.09%; exact claim needs unrounded scores. |

Evidence for this matrix: P pp.3–7 Eq.2–15, Tables 1–3; R paths/lines detailed in §§2–5; E paths/lines detailed in §6.

---

## 9. Reproducibility audit

The status vocabulary in this section is restricted to the requested five labels.

| Component | Status | Evidence and precise limitation |
|---|---|---|
| Attached paper + supplementary | `AVAILABLE` | 21-page PDF inspected in full; hash recorded above. |
| Official DexAvatar Git source | `AVAILABLE` | Branch `main`, commit `a0dfd427...`, clean checkout; 2,076 tracked entries. |
| Runtime orchestration/fitting source | `AVAILABLE` | Drivers, default YAML, parser, optimizer/loss and export paths are tracked and inspected. |
| Exact source for Sapiens invocation | `MISSING` | `sapiens` is a Gitlink (`08dce797...`) with no `.gitmodules` mapping and an empty checkout; called `pose_keypoints133.sh` is absent. R `S1_sapiens_extract.sh` L3–6. |
| Sapiens checkpoints | `UNVERIFIED` | README supplies an external Drive link and expected paths; contents were not accessible for inspection. R `README.md` L37–54. |
| SMPLer-X wrapper/source | `AVAILABLE` | Wrapper and vendored inference code present. |
| SMPLer-X and detector checkpoints | `UNVERIFIED` | Expected under absent `checkpoints/`; README external download not inspected. R `README.md` L71–83. |
| HaMeR wrapper/source | `AVAILABLE` | Demo and vendored project present; runtime downloader invoked. |
| HaMeR/ViTDet pretrained weights | `UNVERIFIED` | Downloaded at runtime from external sources; exact cached bytes/hashes not pinned in audit. R `hamer/demo.py` L36–55. |
| SignBPoser checkpoint/model/config | `UNVERIFIED` | README advertises Drive folder, but `smplifyx/signbposer` is absent from Git and external contents were not inspectable. |
| SignHPoser checkpoint/model/config | `UNVERIFIED` | Same; `smplifyx/signhposer` absent. |
| SignBPoser training code | `MISSING` | Only runtime dynamic loader is tracked; no trainer or tracked model definition. |
| SignHPoser training code | `MISSING` | Same. |
| Exact filtered SignBPoser dataset/splits | `MISSING` | Paper names upstream source but release contains no exact training tensor, retained-frame manifest, split, or filtering program. |
| SignHPoser raw/retargeted dataset/splits | `MISSING` | No raw capture, FBX, baked SMPL-X animations, rectified tensors, participant split, or processing code in release. |
| Default sign class/segment metadata | `AVAILABLE` | `data/signs.txt` and `data/segment.json` present; 57 matching keys. |
| SGNify RGB frames | `REQUIRES LICENSE` | README directs user to SGNify download; absent from Git. R `README.md` L16–35. |
| SGNify SMPL-X GT meshes | `REQUIRES LICENSE` | Same; README explicitly requests GT download for evaluation. |
| SMPL-X neutral model files | `REQUIRES LICENSE` | `human_model_files` absent; fitting README/license restricts SMPL-X assets to licensed non-commercial scientific use. R `dexavatar_fitting/README.md` L21–24; `dexavatar_fitting/LICENSE` L1–27. |
| Collision part segmentation | `REQUIRES LICENSE` | `smplx_parts_segm.pkl` is referenced by script but absent; fitting README directs SMPL-X download. R `script.py` L20–26; fitting README L94–98. |
| Attached evaluation script E | `AVAILABLE` | Full 589-line source and hash recorded. |
| Evaluation MANO/SMPL-X masks | `MISSING` | All five hard-coded assets required at E module `__main__` L534–567 are absent from attachment/repository. |
| Evaluation frame manifests | `MISSING` | GT and predicted directory listings needed to know actual pair count/order were not supplied. |
| Published per-frame/per-sign errors | `MISSING` | Paper gives aggregate tables only; evaluation saving code is commented out. E `main` L463–464. |
| Environment specification | `PARTIALLY AVAILABLE` | README and scripts give conda/pip commands, but dependencies span three environments, external code/checkpoints are unpinned, Sapiens Gitlink is incomplete, and NumPy is installed as 1.23.5 then finally 1.26.3. R `README.md` L9–90; `scripts/env_install.sh` L1–14; `requirements.txt` L1–26. |
| Exact evaluation command/config used for paper tables | `MISSING` | No invocation, mask bundle, file manifest, method labels, outputs, or immutable environment is supplied. |
| Main reported metrics reproduction | `UNVERIFIED` | No pipeline/evaluation run was possible with exact data, checkpoints and masks; all table values remain `REPORTED`. |

`[VERIFIED]` The root project is MIT-licensed, but the included SMPL-X/SMPLify-X fitting subtree carries a non-commercial scientific research license and requires separately licensed assets. These license layers must not be conflated. (R root `LICENSE` L1–20; `dexavatar_fitting/LICENSE` L1–27.)

---

## 10. Evidence-backed bottleneck register

The final column contains only a minimal diagnostic test; it does not prescribe a new model or method.

| ID | Bottleneck / failure mode | Evidence | Pipeline stage | Affected metric | Does current TR-V2V expose it? | Confidence | Minimal diagnostic |
|---|---|---|---|---|---|---|---|
| B01 | Motion blur in fast signing | P p.1 Abstract; p.2 §1; pp.13,16 §S7.1/Fig.S8 | Sapiens/HaMeR detection; hand target | LHand, RHand, UBody(-F) | Only through resulting vertex mismatch; blur itself is not stratified | `[VERIFIED]` | Partition frames by measured blur score and report the existing three metrics per bin. |
| B02 | Hand–hand self-occlusion | P p.2 §1; p.14 §S7.2; p.17 Fig.S9 | HaMeR handed detections and hand pose | Both hands; upper body | Yes for geometry, but not overlap-order correctness separately | `[VERIFIED]` | Annotate overlap order/contact for the cited examples and inspect paired-frame errors. |
| B03 | Hand–body occlusion/contact ambiguity | P p.1 §1; p.2 §1 | 2D/hand inference and collision | Hand and UBody(-F) | Partially; subset centering cannot identify contact semantics | `[VERIFIED]` | Stratify current errors by visible hand fraction/contact condition. |
| B04 | Gaussian noise / missing keypoints | P p.14 §S7.3 reports SGNify no-mesh cases; R skips missing HaMeR/SMPLer frames | Preprocessing and frame retention | All; also frame correspondence | A skipped prediction can shift positional pairing rather than merely increase error | `[VERIFIED]` | Log retained filenames at every stage and compare one-to-one with GT names. |
| B05 | Unreliable 2D keypoints | Paper motivates robust loss; code applies confidences/GMoF and overwrites hand joints | Sapiens/HaMeR → 2D objective | All articulated regions | Yes indirectly; no detector-quality metric is reported | `[VERIFIED]` | Correlate per-frame existing TR errors with 2D confidence/reprojection residual. |
| B06 | Depth ambiguity | P p.1 §1 and p.3 before §3; code’s HaMeR 3D depth term is weight zero | Monocular initialization/fitting | Hands and upper body | Vertex GT exposes final depth error after centering, but input ambiguity is not identified | `[VERIFIED]` | Report per-axis vertex residuals before Euclidean reduction for the same paired frames. |
| B07 | SMPLer-X initialization/target error | Non-pose values fixed; body decoder supervised at weight 1200 | SMPLer-X and body fitting | Primarily UBody(-F), also hands via kinematics | Rotation/shape errors remain; pure subset translation is removed | `[VERIFIED]` | Compare SMPLer-X-only mesh and final mesh under the identical E pairing/masks. |
| B08 | HaMeR target error | HaMeR replaces hand 2D and hand AA; L1+GMoF target weight 1200 | HaMeR/data fusion/hand fitting | LHand/RHand | Yes as mesh error, but source contribution is confounded | `[VERIFIED]` | Log HaMeR pose-target residual and final hand TR error per frame. |
| B09 | Left/right or detection-order ambiguity | HaMeR supplies flags, but two-hand rotation code accesses `[0]` as left and `[1]` as right; active-side branches have fallbacks | Data parser | LHand/RHand and UBody | Yes if wrong geometry; evaluation cannot identify label swap | `[VERIFIED]` | Compare stored handedness flags with the array positions used for each retained frame. |
| B10 | Incorrect one-hand classification/side | Static `0` class + wrist-speed side heuristic; ambiguous defaults left | One-hand decision/fitting | Active hand and UBody(-F) | Protocol always excludes left and skips LHand, so it may hide or misassign the failure | `[VERIFIED]` | Produce a 57-sign table of class, selected active side, and evaluated masks. |
| B11 | Temporal drift or stale predecessor | Body-only prior-frame penalty; resume skips output frames without updating `joints_temp` | Sequential fitting | UBody(-F) over later frames | Framewise TR error sees drift but published aggregate hides time order | `[VERIFIED]` | Plot existing per-frame UBody(-F) error against frame index and mark skipped/resumed frames. |
| B12 | Over-regularization | BPf+bio worsens all Table 2 metrics; HPf+bio worsens RHand | Prior training/fitting | Body subsets; RHand | Yes, because reported errors rise | `[VERIFIED]` | Compare the already defined variant outputs on the identical frame/mask manifest. |
| B13 | Anatomically implausible SGNify hand GT | P p.12 §S6 and p.15 Fig.S7: collapsed fingers/irregular spacing | Evaluation reference | LHand/RHand, possibly UBody | TR-V2V can penalize a more plausible pose | `[VERIFIED]` | Flag cited GT frames with joint-limit/interpenetration diagnostics and report their metric contribution. |
| B14 | Plausibility–vertex-error mismatch | P p.12 §S6 explicitly states plausibility may not reduce TR-V2V | Evaluation interpretation | Hands | No direct plausibility measurement | `[VERIFIED]` | Report existing TR error alongside a separate diagnostic count of GT/predicted bound violations, without combining scores. |
| B15 | Positional frame mispairing | E pairs sorted lists by index and silently filters GT | Evaluation | All three | Can arbitrarily change metric without a 3D-quality change | `[VERIFIED]` | Emit `(prediction filename, GT filename)` pairs and assert intended IDs before aggregation. |
| B16 | Class-dependent metric weighting | Class `0` loses left-hand vertices in UBody and skips LHand | Evaluation | UBody(-F), LHand | Metric value changes with class composition even at equal per-vertex error | `[VERIFIED]` | Report denominators: retained frames and vertex-frame sample count per sign/region. |
| B17 | Translation blindness and independent subset alignment | E subtracts separate centroids | Evaluation | All three | Pure translation failure is invisible; relative hand-to-body placement is weakened by independent hand centering | `[VERIFIED]` | Record removed centroid translation vectors and compare them with the current aligned errors. |
| B18 | Saved-parameter/mesh mismatch | Optimized hand latents absent from pickle; OBJ rotated post hoc | Export/reuse | Recomputed metrics from parameters | E on OBJ does not expose parameter inconsistency | `[VERIFIED]` | Regenerate mesh from each saved pickle and compare vertices to its exported OBJ. |
| B19 | Missing or low-confidence first active-hand observation | One-hand fallback uses previous arrays initialized `None` | Data parser | Active hand, potential crash | May prevent a frame/output rather than yield a measured error | `[VERIFIED]` | Check the first retained frame of every class-`0` sign for matching active-side HaMeR detection. |
| B20 | Anatomical penalty differs from paper | Body uses linear mean hinge; hand term absent | Fitting objective | Body and hands | TR-V2V only sees downstream geometry | `[VERIFIED]` | Log each implemented loss component and bound violation on the released default run. |
| B21 | Lexical re-sort after numeric frame ordering | Parser changes from numeric sort to plain `sorted(temp)` after filtering | Sequence ordering / temporal term | Primarily UBody(-F), indirectly hands | Aggregate metric does not reveal processing order | `[VERIFIED]` | Compare the actual parser order with numeric frame IDs for every sign. |

---

## 11. Unresolved questions

| ID | Unresolved question | Why current evidence is insufficient | Required evidence to resolve it |
|---|---|---|---|
| U01 | Is the requested `evaluate_new_fitting(1).py` identical to E `(2).py`? | Only `(2).py` was attached. | The exact `(1).py` bytes or a cryptographic hash comparison. |
| U02 | What exact GT/prediction frames produced Table 1? | E filters available GT files and pairs positionally; no manifests are supplied. | Immutable per-sign GT and prediction filename manifests plus the exact command. |
| U03 | Why are segment endpoints multiplied by two? | No paper, comment, metadata, or assertion explains the mapping. | Author confirmation or dataset documentation mapping video IDs to GT OBJ IDs. |
| U04 | Were 2,872 or 2,929 central video frames actually evaluated? | Paper count is half-open; fitter/evaluator ranges are inclusive; actual E count depends on existing files. | Exact file manifests and evaluator logs including per-sign retained counts. |
| U05 | What vertices precisely define UBody, UBody(-H), UBody(-F), LHand and RHand? | Required `.npy`/`.pkl` masks are absent. | The five exact mask assets with hashes and vertex counts. |
| U06 | Is class `0` guaranteed to mean a right-hand-dominant sign? | E always removes left hand, whereas fitting chooses either side. | Dataset class semantics and per-sign dominant-hand annotations. |
| U07 | Which optimized parameter set generated each published OBJ? | Pickle omits decoded optimized hand pose/latents and OBJ is rotated afterward. | Saved final latents/decoded poses and an export verification record. |
| U08 | Which exact SignBPoser/SignHPoser checkpoint and architecture were used? | Loader chooses files by modification time from absent external directories. | Immutable checkpoint/config/model files and hashes. |
| U09 | Can the priors be retrained exactly? | Training code, exact datasets/splits, preprocessing implementations and seeds are absent. | Full trainer, model definitions, processed data manifests, split IDs, hyperparameters and seeds. |
| U10 | What are the exact prior input/output rotation conventions? | Paper gives matrix/AA losses but not conversion ordering; training code absent. | Model source and preprocessing code defining joint order and conversions. |
| U11 | Was hand biomechanics used during published fitting? | Eq.12/text says yes; released `SMPLifyLoss` has no term. | Experimental commit/config or author confirmation that published runs used different code. |
| U12 | Was body biomechanics active in every Table 2 variant? | Released default enables it, while paper describes a separately excluded optimization result. | Exact config for every Table 2 row and the unreported body-biomechanics run. |
| U13 | Are Table 3 changes training-time, fitting-time, or both? | Variant definition and narrative differ; released fitting lacks hand biomechanics. | Row-specific checkpoint provenance and loss logs/configuration. |
| U14 | What unrounded scores yield 35.11%/16.32%/14.11%? | Only two-decimal table values are published. | Full-precision aggregate outputs. |
| U15 | What units are used in Tables S1–S2 MPJPE/MPVPE? | Captions do not state units. | Evaluation code/config or author clarification. |
| U16 | How are prior DEV/TEST splits constructed and are signers disjoint? | No split definition is provided. | Dataset split manifests and participant IDs. |
| U17 | What exact Sapiens code revision and checkpoint bytes were used? | Gitlink lacks `.gitmodules` mapping; external checkpoint unverified. | Resolved submodule URL/commit and checkpoint hashes. |
| U18 | What happens on ambiguous/multiple-person HaMeR detection in the paper runs? | Code has positional/fallback behavior but no logs. | Per-frame detection outputs, handedness flags and retained-frame logs. |
| U19 | What coordinate transform relates SGNify GT to exported DexAvatar OBJ? | Only the final 180° X transform is visible; GT coordinate documentation/assets absent. | GT coordinate specification and export/evaluation transform documentation. |
| U20 | Are paper baselines evaluated with exactly the same masks, frame pairs, one-hand rule and aggregation? | E contains one method-specific `pixie` rotation but no published run records for all baselines. | Commands, per-method file manifests, method labels and full logs. |
| U21 | Does an actual run ever activate the implemented HaMeR 3D Z loss? | Default YAML sets all weights to zero; no alternative published config is supplied. | Exact experimental configs and loss-component logs. |
| U22 | What are the actual numbers of valid vertex-frame samples per reported region? | NaN skipping, missing files and class-dependent masks alter denominators. | Evaluator output containing per-sign/per-region denominators. |

---

## 12. Source manifest

### 12.1 Repository identity

- **Remote:** `https://github.com/kaustesseract/DexAvatar.git`
- **Branch inspected:** `main`
- **Commit:** `a0dfd427f60f5811aadb35c8657b3856d47f56b5`
- **Commit timestamp/subject:** `2026-05-03T16:41:49+08:00 — Update README.md`
- **Access date:** 2026-08-25
- **Working-tree state at audit:** clean
- **Tracked entries:** 2,076
- `[VERIFIED]` `sapiens` is recorded as Gitlink mode `160000` at object `08dce797f7b40f5b41388f518cac85535c3f5d13`, but no `.gitmodules` mapping exists in the inspected commit and its checkout is empty.

### 12.2 Paper pages read

| PDF pages | Content inspected | Status |
|---|---|---|
| 1–8 | Main paper: abstract/introduction, related work, method, preprocessing, priors, optimization Eq.1–15, experiments, Tables 1–3, conclusion | Read in full; text extracted and pages visually rendered/checked |
| 9–18 | Entire supplementary: SMPL-X background; body ROM/signer space; mocap retargeting; prior sweeps Tables S1–S2; VPoser ablation Table S3; SGNify GT limitations; qualitative blur/occlusion/noise analyses | Read in full; text extracted and pages visually rendered/checked |
| 19–21 | References | Read in full for completeness; not used to launch a broad literature review |

### 12.3 Repository files read in full

The following files were inspected completely at R commit `a0dfd427...`:

- `README.md`
- `LICENSE`
- `run_dexavatar.py`
- `Full_running_command.sh`
- `M3_mean_shape_smplerx.py`
- `requirements.txt`
- `scripts/config.sh`
- `scripts/config_sapiens.sh`
- `scripts/config_smplerx.sh`
- `scripts/S1_sapiens_extract.sh`
- `scripts/S1_smplerx_extract.sh`
- `scripts/M3.5_hamer_extract.sh`
- `scripts/M4_smplifyx_pose.sh`
- `scripts/env_install.sh`
- `scripts/bug_fix_dexavatar.sh`
- `data/signs.txt`
- `data/segment.json`
- `dexavatar_fitting/LICENSE`
- `dexavatar_fitting/README.md`
- `dexavatar_fitting/script.py`
- `dexavatar_fitting/cfg_files/fit_smplx_vposer_x.yaml`
- `dexavatar_fitting/cfg_files/fit_smplx.yaml`
- `dexavatar_fitting/smplifyx/main.py`
- `dexavatar_fitting/smplifyx/cmd_parser.py`
- `dexavatar_fitting/smplifyx/fit_single_frame.py`
- `dexavatar_fitting/smplifyx/fitting.py`
- `dexavatar_fitting/smplifyx/data_parser.py`
- `dexavatar_fitting/smplifyx/prior.py`
- `dexavatar_fitting/smplifyx/body_constants.py`
- `dexavatar_fitting/smplifyx/camera.py`
- `dexavatar_fitting/smplifyx/test_bposer.py`
- `dexavatar_fitting/smplifyx/test_hposer.py`
- `dexavatar_fitting/smplifyx/utils.py`
- `dexavatar_fitting/smplifyx/optimizers/optim_factory.py`
- `dexavatar_fitting/smplifyx/optimizers/lbfgs_ls.py`
- `dexavatar_fitting/assets/mapping_func.py`
- `hamer/demo.py`
- `SMPLer-X/main/script_smplerx.py`
- `SMPLer-X/main/inference.py`
- `SMPLer-X/main/config.py`
- `SMPLer-X/main/config/config_smpler_x_h32.py`
- associated top-level/SMPLer-X/HaMeR requirement files referenced by the installation path.

`[VERIFIED]` `dexavatar_fitting/assets/smplx_uv_new.obj` was not read semantically line-by-line; its topology was programmatically counted as 10,475 `v` records and 20,908 `f` records.

### 12.4 Repository files/directories read only in part

| Path | Portions inspected | Reason |
|---|---|---|
| `dexavatar_fitting/rewrite_body_model.py` | Constructor/parameter reset and relevant SMPL-X forward path, approximately L40–320 and L891–1302 | Establish zero-reset behavior, parameter fallback, full-pose assembly and mesh generation |
| `dexavatar_fitting/assets/joint_mapping.py` | Relevant COCO-WholeBody and SMPL-X name lists/mapping sections | Establish joint intersection and review 3D-hand index comments |
| `SMPLer-X/` vendored source beyond files above | Entry/config/model-output interfaces only | Full third-party framework internals are outside the baseline control-flow question |
| `SMPLer-X/main/SMPLer_X.py` | L168–220 and L370–390 | Verify 6-D-to-axis-angle conversion and exported parameter tensors |
| `hamer/` vendored source beyond `demo.py` | Runtime interface and output dictionary usage only | Full third-party training/model internals were not audited |
| `torch-mesh-isect/` | Called interfaces/constructor signatures only | Exact CUDA kernel internals were not needed to establish whether collision loss is enabled |
| `neural_renderer/` | OBJ loader call site only | Used only in export path |

### 12.5 Evaluation script coverage

- **E was read in full: L1–589.**
- Key audited ranges:
  - L58–96: hard-coded legacy paths/method mappings.
  - L99–149: OBJ loading and method-specific rotation.
  - L152–203: raw, centroid-translation, common-center and pelvis error functions.
  - L231–265: GT sign-segment selection, ×2 endpoints, inclusive interval, missing-file filtering.
  - L268–280: predicted mesh discovery and sort rule.
  - L283–461: pairing, topology assertion, NaN behavior, one-hand masking, alignment and aggregation.
  - L467–589: CLI arguments, unused `central`, sign/class parsing, hard-coded assets and vertex subsets.

### 12.6 Files and assets not inspected

The following are explicitly **NOT INSPECTED**; no content was inferred as fact:

- `evaluate_new_fitting(1).py` — not supplied.
- Sapiens source called at `sapiens/lite/scripts/demo/torchscript/pose_keypoints133.sh` — absent because the Gitlink checkout is empty/incompletely mapped.
- `dexavatar_fitting/smplifyx/signbposer/` and `signhposer/` external model/config/checkpoint contents — absent from Git; advertised Drive directories could not be opened in this environment.
- `checkpoints/` for SMPLer-X/mmdet — absent.
- `SMPLer-X/common/utils/human_model_files/` — absent and license-gated.
- SGNify `data/images_sgnify/` RGB and `smplxgt` mesh files — absent/license-gated.
- Evaluation assets under E’s hard-coded `data_base_dir`: `MANO_SMPLX_vertex_ids.pkl`, `SMPLX_NEUTRAL.npz`, `upper_body.npy`, `upper_body_minus_head.npy`, `upper_body_minus_face.npy` — absent.
- Exact training datasets, processed prior tensors, train/dev/test manifests, prior training logs, and random seeds — not released in the inspected commit.
- Exact published prediction directories, GT filename manifest, evaluator stdout/logs, per-frame/per-sign errors, and full-precision table outputs — not supplied.

### 12.7 Ancillary dependency check

`[VERIFIED]` The official SGNify repository was checked only as an ancillary dependency for missing evaluation assets: branch `master`, commit `bae2a71d8388df73af56117731f7f454e36e5b2e`. Its manifest was searched; it did not supply the masks or exact DexAvatar evaluation bundle required by E. No SGNify method claims were added from that check.

---

## Final quality-control record

- `[VERIFIED]` No new reconstruction method, module, literature survey, SOTA combination, or implementation plan is proposed.
- `[VERIFIED]` Paper intent, released-code behavior, and evaluator behavior are kept separate in every substantive comparison.
- `[VERIFIED]` TR-V2V alignment, pairing, one-hand exclusion, aggregation, NaN behavior and unit conversion were derived from E, not copied from the paper.
- `[VERIFIED]` All numeric model results are labeled `REPORTED`; only arithmetic derived from printed values is labeled `RECOMPUTED`; nothing is labeled reproduced.
- `[VERIFIED]` Inferences and unresolved facts are explicitly marked, including the ×2 frame rationale, exact masks, exact evaluated count, external prior artifacts and biomechanics experiment provenance.
- `[VERIFIED]` Contradictions and missing components are retained rather than reconciled speculatively.

STEP 1 COMPLETE — READY FOR EXTERNAL REVIEW BEFORE DEFINING THE NEXT RESEARCH STEP.
