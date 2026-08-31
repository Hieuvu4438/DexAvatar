<!-- Faithful text extraction from the uploaded PDF. Layout is preserved in fenced text blocks. -->

<!-- PAGE 1 -->

```text
                                                Tamaththul3D: High-Fidelity 3D Saudi Sign
                                                 Language Avatars from Monocular Video

                                         Eyad Alghamdi1 , Sattam Altuuaim2 , Obay Ghulam1 , Abdulrahman Qutah1 ,
                                                                  and Yousef Basoodan1

                                                              University of Jeddah, Jeddah, Saudi Arabia
                                                              1
                                            2
                                                King Abdullah University of Science and Technology, Thuwal, Saudi Arabia
arXiv:2605.05367v2 [cs.CV] 4 Jun 2026




                                                       6              5             4             3             2              1




                                        Fig. 1: Tamaththul3D reconstructs high-fidelity 3D sign language avatars from
                                        monocular video. Top: a Saudi signer performing a Saudi Sign Language sentence.
                                        Bottom: the SMPL-X avatars reconstructed by Tamaththul3D, recovering accurate
                                        hands and body posture despite loose traditional attire.

                                                 Abstract. Existing 3D sign language avatar reconstruction methods are
                                                 developed and evaluated exclusively on Western sign languages, and no
                                                 3D parametric annotations exist for any Arabic Sign Language dataset,
                                                 a gap that blocks the development of avatar-based accessibility applica-
                                                 tions for the Arab Deaf community. We release the first SMPL-X para-
                                                 metric annotations for the Ishara-500 Saudi Sign Language dataset, en-
                                                 abling quantitative evaluation and downstream sign language generation
                                                 for Arabic Sign Language. We introduce Tamaththul3D, a reconstruc-
                                                 tion pipeline that aligns hand and body estimates through geometric
                                                 inverse kinematics on the forearm chain followed by 2D-supervised shoul-
                                                 der refinement. The closed-form integration is decoupled from the specific
                                                 choice of body and hand estimators: any SMPL-X-compatible body esti-
                                                 mator and any MANO-compatible hand estimator can be substituted, as
                                                 we demonstrate by swapping each module independently. Tamaththul3D
                                                 achieves up to 32% lower hand error than prior methods, runs 32× faster
                                                 than the strongest baseline, and generalizes across five typologically dis-
                                                 tinct sign languages without dataset-specific adaptation.


                                        1       Introduction
                                        Arabic Sign Language (ArSL) and its regional variants, like Saudi Sign Lan-
                                        guage (SSL), serve as the primary communication systems for Deaf communities
```

<!-- PAGE 2 -->

```text
2      Alghamdi et al.

throughout the Arab world, where sign languages remain the day-to-day medium
of communication. Recent years have seen major progress in 3D human pose es-
timation and avatar reconstruction [1–4], yet no specialized system exists for
creating detailed 3D signing avatars from ArSL videos with accurate parametric
annotations.
    The Role of 3D Parametric Annotations. High-quality 3D parametric
annotations are foundational infrastructure for sign language avatar research.
They enable quantitative evaluation of reconstruction methods, provide ground-
truth supervision for learning-based approaches, and serve as the motion rep-
resentation that downstream sign language generation and translation systems
build on directly [5]. For Western sign languages, datasets such as How2Sign [6]
have been enriched with SMPL-X annotations [5], enabling a generation of
avatar-based applications. For ArSL, no such annotations exist. Existing ArSL
datasets [7,8] provide only 2D video, and Ishara-500 [9], the largest SSL dataset,
likewise lacks any 3D parametric representation. This absence blocks the devel-
opment of avatar-based accessibility applications for the Arab Deaf community
entirely.
    Limitations of Existing Approaches. Existing sign language avatar re-
construction methods [5,10] have several shortcomings. SGNify [10] uses linguis-
tic priors but produces visually incorrect hand gestures. Neural Sign Actors [5]
focuses on generating sign language from text and developed a curation process
for SMPL-X datasets using OSX [3] and MediaPipe [11]. DexAvatar [12] uses
sign-language-aware priors for optimization-based reconstruction but requires
21.60 seconds per frame, making large-scale annotation impractical. Critically,
all existing methods are developed and evaluated exclusively on Western sign
languages (ASL and German Sign Language), with no consideration for ArSL’s
unique characteristics.
    Unique Challenges of ArSL. Sign language reconstruction presents dis-
tinct challenges compared to general human pose estimation [13]. First, hand
articulation complexity: sign languages convey meaning through precise finger
configurations, palm orientations, and rapid hand motions. General-purpose pose
estimation methods [1, 3] often fail to capture these intricate details that ren-
der signs incomprehensible. Second, cultural specificity: ArSL exhibits distinct
characteristics in hand shapes, movement patterns, and signing space that differ
from ASL or BSL, and signers frequently wear culturally specific attire (thobes,
abayas, and hijabs) that obscures body shape cues in models trained on West-
ern data. Third, multi-modal integration: effective signing requires simultaneous
capture of hand pose, body orientation, and facial expressions, all of which con-
tribute to semantic meaning.
    Contributions. We introduce Tamaththul3D (from Arabic تَمَثُّل, mean-
ing “representation” or “likeness”), a reconstruction pipeline for Arabic Sign Lan-
guage avatars that generalizes across five typologically distinct sign languages.
Our primary technical contribution is a geometric forearm alignment method
that solves for the elbow rotation aligning the kinematic chain with WiLoR’s
global wrist orientation, combined with 2D-supervised shoulder optimization,
```

<!-- PAGE 3 -->

```text
                                                            Tamaththul3D        3

achieving state-of-the-art hand accuracy at 32× the speed of the strongest base-
line.
    Our contributions are:

 – First 3D parametric annotations for Arabic Sign Language. We
   release SMPL-X annotations for the Ishara-500 Saudi Sign Language dataset,
   the first 3D parametric annotations for any Arabic Sign Language dataset.
 – Geometric hand-body integration. We solve for the elbow rotation
   that aligns the SMPL-X kinematic chain with WiLoR’s global wrist ori-
   entation via swing-twist decomposition, then refine only the shoulder using
   confidence-weighted MediaPipe keypoints, resolving shoulder-forearm incon-
   sistencies without disrupting hand accuracy.
 – Generalization study. The closed-form geometric design generalizes across
   five typologically distinct sign languages without dataset-specific adaptation,
   achieving up to 32% lower hand error than prior methods at 32× the speed
   of the strongest baseline.


2     Related Work

2.1   Whole-Body 3D Pose Estimation

Recent advances enable detailed 3D reconstruction from monocular images using
parametric body models. The foundational SMPL model [14] represents the hu-
man body through shape and pose parameters learned from diverse 3D meshes,
providing a unified parametric representation. SMPL-X [1] extends this to in-
corporate expressive hands and face, enabling whole-body pose estimation with
a unified parameter space.
    Regression-based methods directly predict body parameters from images.
Kanazawa et al. [15] introduced HMR, which uses adversarial training to re-
cover 3D meshes without paired 3D supervision; SPIN [16] closed the loop by
incorporating SMPLify optimization within the training loop. FrankMocap [17]
employs separate modules for body, hands, and face but achieves limited integra-
tion. PIXIE [18] uses collaborative regression for expressive bodies, and PyMAF-
X [19] refines whole-body fits via mesh-aligned feedback. Video-based methods
such as VIBE [20] exploit temporal context but operate on the body only, omit-
ting expressive hand and face detail. Hybrid regression-IK approaches such as
HybrIK [21] combine learned pose estimation with analytic inverse kinematics,
a design tradition our forearm alignment builds on. OSX [3] introduces a one-
stage transformer with component-aware attention, and SMPLer-X [2] scales up
with a ViT-Huge backbone, demonstrating state-of-the-art performance across
multiple benchmarks. 2D pose detectors such as MediaPipe [11] complement
these methods by providing real-time confidence-weighted keypoint supervision
that resolves depth ambiguities in shoulder and elbow joints, a role our pipeline
explicitly exploits.
```

<!-- PAGE 4 -->

```text
4      Alghamdi et al.

2.2   Hand Pose Estimation

Accurate 3D hand pose estimation is challenging due to self-occlusions, com-
plex articulations, and depth ambiguity [4,13]. Model-based methods use MANO
parametric hand models [22] to ensure anatomically plausible predictions. Hand4Whole [4]
focuses on hand pose within whole-body estimation but treats hands indepen-
dently from body context. Transformer- and graph-based mesh reconstruction
(METRO [23], MeshGraphormer [24]) recover vertex positions directly without
an intermediate parametric stage. HaMeR [25] employs a transformer-based ar-
chitecture with ViT-Huge backbone, achieving strong results across occlusions,
hand-object interactions, and diverse viewpoints. WiLoR [26] introduces end-to-
end hand localization and reconstruction through transformer-based refinement,
achieving superior accuracy via automatic hand detection. Our pipeline leverages
WiLoR’s precise hand predictions while addressing the non-trivial challenge of
integrating MANO-format outputs with SMPL-X body parameters. Progress in
this domain has been driven by large-scale benchmarks including FreiHAND [27],
InterHand2.6M [28], and HO-3D [29].


2.3   Sign Language Avatar Reconstruction

Sign language reconstruction methods have advanced substantially on Western
sign languages. SGNify [10] introduces linguistic priors for isolated signs us-
ing sign-language-aware optimization constraints, though at high computational
cost. Neural Sign Actors [5] provides the first high-quality 3D SMPL-X annota-
tions for the How2Sign dataset [6], establishing critical infrastructure for ASL
avatar-based applications, an analogous contribution to what we provide for
Arabic Sign Language. DexAvatar [12] proposes sign-language-aware priors for
optimization-based reconstruction, achieving 30.13mm body and 13mm hand
errors, but requires 21.60 seconds per frame and is evaluated exclusively on
Western sign languages.
    The downstream value of accurate parametric annotations extends beyond
reconstruction. Sign-language production systems [5,30] and self-supervised sign-
recognition models [31] rely directly on annotation quality , noisy or geometri-
cally inconsistent SMPL-X parameters in training data propagate errors into
generated or recognised signing motions. This makes the absence of 3D ArSL
annotations not merely an evaluation gap, but a barrier to the entire research
ecosystem. As a concrete demonstration of this downstream value, we use the re-
leased Tamaththul3D annotations to train a gloss-conditioned 3D sign-language
production model for ArSL; its motion tokenizer and qualitative results are de-
tailed in the supplementary material.
    Arabic Sign Language Gap. Limited work addresses ArSL reconstruc-
tion. ArabSign [7] provides a continuous ArSL dataset with 9,335 samples, while
KArSL [8] offers 502 isolated ArSL signs with 75,300 total samples. Large-scale
recognition benchmarks including WLASL [32] and PHOENIX-2014 [33] have
driven progress in Western sign languages, but no prior work provides 3D para-
metric annotations or specialized reconstruction methods for ArSL avatar gen-
```

<!-- PAGE 5 -->

```text
                       M ANO
    W iLoR




                                                                                                SM PL-X
                       SM PL-X
    SM PLer -X
                                                                                                          Tamaththul3D              5




    M ediaPipe
                                 pose r oot

                                 pose hands
                                                                                                                    3D Reconstruction
    Reference                    shape



                                 pose hands                             Geom etr ic
                                 pose body                                 IK
                                                                                      Tw ist Alignm ent
                                 expr essions
                                 shape
                                 cam er a
                                                                        Alignm ent & Optim ization



                                                Detected 2D Keypoints




Fig. 2: Tamaththul3D pipeline. SMPLer-X, WiLoR, and MediaPipe feed a geometric
forearm-alignment stage and a 2D-supervised shoulder optimization, producing the
final SMPL-X parameters.

eration. All existing 3D reconstruction methods focus on ASL, German SL, or
British SL, with no consideration for ArSL’s unique characteristics.


3                Method
3.1              Overview
Tamaththul3D reconstructs expressive 3D avatars from monocular RGB video
by decoupling hand and body estimation and fusing them through geometric in-
verse kinematics. The pipeline operates in four stages, illustrated in Fig. 2. First,
three complementary modules extract features from each frame: SMPLer-X [2]
estimates whole-body SMPL-X parameters, WiLoR [26] reconstructs detailed
hand poses in MANO format, and MediaPipe [11] extracts confidence-weighted
2D keypoints for supervision. Second, the geometric IK solver aligns the fore-
arm kinematic chain with WiLoR’s accurate global wrist orientation. Third,
the aligned hand and body parameters are fused through coordinate conver-
sion, left-hand mirroring, and 2D-supervised shoulder optimization. Finally, the
corrected SMPL-X parameters render the output avatar. Formally, the pipeline
takes monocular RGB frames I ∈ RH×W ×3 and outputs SMPL-X parameters
Θ = {β, θbody , θhand , ψ, ϕ} representing shape, body pose, hand pose, expression,
and global orientation respectively.

3.2              Initial Pose Estimation
Three off-the-shelf modules supply the inputs to our integration stage. SMPLer-
                                                                         init
X [2] regresses initial whole-body SMPL-X parameters Θinit = {βinit , θbody   , ψinit , ϕinit }.
WiLoR [26] reconstructs each detected hand Hi (i ∈ {left, right}) in MANO for-
                                            i
mat [22], producing finger joint rotations θhand ∈ R15×3 , hand shape βhand
                                                                        i
                                                                              ∈ R10 ,
                         i                                     3
global wrist rotation Rwrist ∈ SO(3), and translation ti ∈ R . MediaPipe [11] ex-
tracts 2D keypoints {kj , cj } with confidence cj ∈ [0, 1] at the shoulders, elbows,
and wrists; high-confidence keypoints supervise the optimization in Sec. 3.4 while
low-confidence ones are down-weighted.
```

<!-- PAGE 6 -->

```text
6            Alghamdi et al.

3.3       Hand-Body Integration

This is the central contribution of the pipeline (Stage 2 of Fig. 2): integrating
MANO-format hand poses from WiLoR with SMPL-X body parameters from
SMPLer-X. Direct substitution fails due to: (1) different coordinate systems
(MANO uses hand-centric coordinates while SMPL-X uses body-centric; (2) dif-
ferent parametrizations (MANO encodes poses relative to its own mean hand
pose, while SMPL-X uses a body-centric rest pose space; and (3) wrist rotation
ambiguity.
    Coordinate Conversion. To convert WiLoR’s MANO poses to SMPL-X
format, we convert rotation matrices to axis-angle representation and subtract
                         MANO
the MANO mean pose θ̄hand      :

                                                                      \theta _{\text {hand}}^{\text {SMPL-X}} = \theta _{\text {hand}}^{\text {MANO}} - \bar {\theta }_{\text {hand}}^{\text {MANO}}                                (1)

This removes MANO’s pose bias, yielding hand poses in SMPL-X’s rest pose
space.
    Left Hand Mirroring. WiLoR processes left hands as mirrored right hands
for model efficiency. We apply a YZ-plane reflection to recover proper MANO
left hand format:
                             R_{\text {left}} = \mathbf {M} \cdot R_{\text {WiLoR}} \cdot \mathbf {M}^{\top }  (2)
applied to both wrist rotation and all 15 finger joint rotations.
    Geometric Forearm Alignment. WiLoR provides highly accurate global
wrist rotations in world space, while SMPL-X requires local joint rotations in
its kinematic tree. We solve for the elbow rotation geometrically to ensure the
entire forearm chain matches WiLoR’s wrist placement. Building the forward
kinematics chain:

                                                   R_{\text {world}}^j = \begin {cases} R_{\text {local}}^0 & \text {if } j = 0 \\ R_{\text {world}}^{p(j)} \cdot R_{\text {local}}^j & \text {otherwise} \end {cases}              (3)

                                                                             wrist
where p(j) is the parent of joint j. Given the target global wrist rotation Rtarget
from WiLoR, we solve for the elbow rotation that achieves exact alignment:

                                R_{\text {elbow}}^{\text {local,new}} = (R_{\text {shoulder}}^{\text {world}})^{\top } \cdot R_{\text {target}}^{\text {wrist}} \cdot (R_{\text {wrist}}^{\text {local,cur}})^{\top }               (4)

    Twist Extraction for Forearm Rotation. Forearm rotation (twist along
the arm axis) requires special treatment. We apply swing-twist decomposition [34]
to extract the twist component:

                                      \mathbf {a}_{\text {twist}} = \mathbf {f} \cdot (\mathbf {a}_{\text {rel}} \cdot \mathbf {f}), \quad \mathbf {a}_{\text {swing}} = \mathbf {a}_{\text {rel}} - \mathbf {a}_{\text {twist}}    (5)

where f is the forearm axis and arel is the relative rotation between target and
current wrist configurations. The twist is then applied to the geometric elbow
solution:
                          R_{\text {elbow}}^{\text {final}} = \exp (\mathbf {a}_{\text {twist}}) \cdot R_{\text {elbow}}^{\text {local,new}}  (6)
```

<!-- PAGE 7 -->

```text
                                                                                                                                                                                                                                                                Tamaththul3D                                                                    7

3.4                   2D-Supervised Upper Body Optimization

After geometric forearm alignment (Stage 3 of Fig. 2), the elbow and wrist
joints are precisely positioned to match WiLoR’s hand predictions. However,
the shoulder may require adjustment since the geometric solution modifies the
forearm without considering the full arm chain. We optimize only the shoulder
θshoulder while keeping the geometrically-aligned elbow and wrist fixed, using a
pose consistency loss:

   \mathcal {L} = \lambda _{\text {reg}} \left \lVert \theta _{\text {shoulder}} - \theta _{\text {shoulder}}^{\text {init}} \right \rVert ^2 + \lambda _{\text {2D}} \sum _{j \in J} w_j\, c_j \cdot \left \lVert \pi (\mathbf {p}_j) - \mathbf {k}_j \right \rVert _1 + \lambda _{\text {pose}} \left \lVert \mathbf {z} \right \rVert ^2    (7)


where π(pj ) projects 3D joint positions to 2D and kj are MediaPipe keypoints
with per-joint weights wj and confidence cj . The final term is a learned body-
pose prior ∥z∥2 on the VPoser [1] latent encoding of the full body pose, pushing
the optimized shoulder toward statistically plausible configurations; the supple-
mentary material discusses this prior in depth. Loss weights are optimized using
Adam with learning rate 1 × 10−2 for 50 iterations per frame.


3.5                   Temporal Smoothing

For video sequences, we suppress per-frame jitter through post-hoc multi-order
derivative minimization [35]:

                                                                        \mathcal {L}_{\text {temp}} = \lambda _{\text {data}} \mathcal {L}_{\text {data}} + \lambda _1 \mathcal {L}_{\text {d1}} + \lambda _2 \mathcal {L}_{\text {d2}} + \lambda _3 \mathcal {L}_{\text {d3}},                                                                (8)

where the four terms enforce fidelity to the per-frame estimates and penalize
velocity, acceleration, and jerk respectively. Derivative penalties are weighted
more strongly for hand joints than body joints, preserving fine finger articulation
while suppressing torso and arm noise. Side-by-side comparisons of smoothed vs.
unsmoothed output across multiple datasets are demonstrated qualitatively in
supplementary videos accompanying the annotation release.


3.6                   Modularity

The geometric integration in Sec. 3.3 is closed-form and makes no assumption
about how its inputs were produced: it requires only any estimator that outputs
SMPL-X body parameters and any estimator that outputs MANO-format hand
parameters with a global wrist rotation. SMPLer-X and WiLoR are selected
as the current best-performing choices, but neither the IK solver nor the opti-
mization stage depends on their internals, so improved body or hand estimators
can be substituted without re-deriving the alignment. This decoupling distin-
guishes our approach from end-to-end methods whose architectural assumptions
are baked into the model weights, and is validated empirically in Sec. 4.4 by
swapping the body and hand estimators independently.
```

<!-- PAGE 8 -->

```text
8      Alghamdi et al.

4     Experiments

4.1   Datasets

Ishara-500 The Ishara-500 subset of the Ishara dataset [9] is a large-scale con-
tinuous Saudi Sign Language dataset comprising 30,000 video samples captured
in unconstrained environments using smartphone cameras. Ishara-500 contains
videos from 18 diverse signers performing over 500 unique SSL sentences, with
natural variations in camera angles, distances, lighting conditions, and back-
grounds.


Ishara-500 Annotations This work produces the first high-quality SMPL-X
parameter annotations for Ishara-500, establishing it as the first 3D Arabic Sign
Language dataset with parametric avatar representations. Annotations will be
publicly released to enable future research in Arabic Sign Language avatar re-
construction and related applications. The dataset was collected and released by
Alyami et al. [9] under their institutional review board (IRB) approval; this work
performs no new human-subject data collection and uses Ishara-500 solely for
annotation and evaluation purposes, so no additional IRB review was required.


Cultural Clothing and Model Bias A significant challenge when process-
ing Ishara-500 concerns traditional Saudi attire: male signers frequently wear
thobes (ankle-length white robes) and female signers wear hijabs and abayas.
These culturally significant garments are absent from the training data of foun-
dation models like SMPLer-X, which were trained predominantly on Western
datasets with form-fitting clothing. The loose, flowing fabric obscures body shape
cues and joint locations, presenting a critical case of model bias when applying
state-of-the-art methods to underrepresented populations. MediaPipe-guided 2D
optimization partially addresses this by providing clothing-invariant keypoint
guidance for shoulder and elbow refinement.


SGNify Benchmark For quantitative evaluation and comparison with prior
methods, the SGNify mocap dataset [10] is used, containing ground-truth SMPL-
X annotations for sign language sequences captured with high-quality motion
capture, enabling direct numerical comparison with existing reconstruction meth-
ods.


4.2   Evaluation Metrics

Procrustes-Aligned Mean Per Vertex Position Error (PA-MPVPE) [1, 15] is re-
ported in millimeters, computed separately for body, left hand, and right hand
regions. PA-MPVPE applies Procrustes alignment [36] before computing dis-
tances, making the metric invariant to global rotation and translation while
focusing on shape and pose accuracy.
```

<!-- PAGE 9 -->

```text
                                                          Tamaththul3D        9

4.3   Implementation Details
Components. Official pretrained weights for SMPLer-X [2] and WiLoR [26] are
used without any fine-tuning, demonstrating strong out-of-the-box generaliza-
tion. MediaPipe [11] uses default Pose and Hand models with no modifications.
    Preprocessing. Input frames are processed at their original resolution. Per-
son bounding boxes are detected via the integrated detector in SMPLer-X. Hand
regions are automatically localized by WiLoR’s built-in detection module, re-
moving the need for manual cropping or region proposals.
    SMPL-X and MANO Models. The neutral-gender SMPL-X model [1]
is used with 10 shape coefficients and 10 expression coefficients, without PCA
compression, to preserve the full articulation range. MANO models [22] use 45
PCA components with the flat hand mean disabled to ensure compatibility with
SMPL-X’s hand pose space.
    Camera Model. Camera intrinsics (focal length and principal point) are ex-
tracted directly from SMPLer-X predictions. A weak-perspective camera model
is used for all 2D-to-3D correspondences during optimization.
    Runtime. All experiments were conducted on an NVIDIA RTX 5070 Ti
with 16GB GPU memory and 32GB CPU memory. The full pipeline processes a
150-frame SSL video at 30 fps in roughly 100 seconds (0.67 s/frame), compared
to 54 minutes (21.60 s/frame) for DexAvatar on identical hardware.

4.4   Results
Main Results Tamaththul3D is benchmarked against seven existing methods
on the SGNify dataset, spanning general whole-body estimators, sign-language-
specific reconstruction methods, and the strongest prior baseline DexAvatar.
Tab. 1 reports PA-MPVPE for body, left hand, and right hand; per-frame run-
time against the strongest baseline is reported separately in Tab. 4.
Table 1: Quantitative comparison on the SGNify dataset. PA-MPVPE (mm) measures
geometric accuracy; lower is better.

             Method            Body ↓     L. Hand ↓    R. Hand ↓
             FrankMocap [17]     78.07       20.47        19.62
             PIXIE [18]          60.11       25.02        22.42
             SMPLify-X [1]       56.07       22.23        18.83
             SGNify [10]         55.63       19.22        17.50
             OSX [3]             60.79       19.10        18.79
             NSA [5]             46.42       16.17        15.23
             DexAvatar [12]      30.13       13.53        13.08
             Tamaththul3D       29.28       10.65         8.90

   Tamaththul3D achieves state-of-the-art hand accuracy across all baselines
while running 32× faster than the strongest baseline DexAvatar (Tab. 4). General
methods (FrankMocap, PIXIE, SMPLify-X) produce poor hand accuracy (18–
25mm) as they are not designed for fine-grained hand articulation. Sign-specific
```

<!-- PAGE 10 -->

```text
10     Alghamdi et al.

methods (SGNify, NSA) improve hands (15–19mm) but remain insufficient for
semantic clarity. The NSA [5] fitting pipeline was confirmed unavailable for ex-
ternal release at the time of submission following direct correspondence with the
authors.


        Table 2: Ablation study on SGNify dataset (PA-MPVPE in mm).

      Configuration                    Body ↓    L. Hand ↓            R. Hand ↓
      SMPLer-X                          28.46                 18.17     17.47
      W/ 2D Supervision                 28.35                 18.17     17.47
      W/ WiLoR (Coord. Conv.)           28.46                 10.71     9.03
      W/ WiLoR (Geometric Align.)       29.53                 10.68     8.95
      Full pipeline                     29.28                 10.65     8.90

Ablation Study Each stage of the pipeline



                                                  Reference
contributes incrementally to the final result.
SMPLer-X alone produces robust body pose
estimates (28.46mm) but poor hand accuracy
(18.17mm left, 17.47mm right), insufficient       Naive

for sign language semantics. Adding 2D su-
pervision alone yields only marginal improve-
ment on the body (28.35mm) and none on
                                                  Aligned




the hands, confirming that 2D keypoints with-
out explicit hand refinement have limited ef-
fect. Coordinate conversion replaces SMPL- Fig. 3: Kinematic artifacts
X hand poses with WiLoR predictions, pro- without geometric forearm align-
ducing a large improvement in hand accuracy ment. Top: reference; middle:
(10.71mm left, 9.03mm right, a 48% reduction naive WiLoR substitution; bot-
in right-hand error) while preserving body tom: with geometric alignment.
pose (28.46mm). Geometric alignment further
refines results by solving for elbow rotation aligned with WiLoR’s global wrist
(8.95mm right). Geometric alignment yields only marginal numerical improve-
ment over coordinate substitution, but its primary value is qualitative: it guar-
antees kinematic consistency between body and hand, preventing “broken wrist”
artifacts that occur with naive substitution (Fig. 3). The full pipeline achieves
the best overall trade-off, with competitive body pose (29.28mm, a 0.82mm
degradation from baseline, < 3%) and the most accurate hands (10.65mm left,
8.90mm right).

Modularity Analysis To empirically validate the modularity claim of Sec. 3.6,
we replace the body and hand estimators independently and re-evaluate on SG-
Nify. We consider three body backbones (OSX [3], SMPLest-X [37], and SMPLer-
X [2]) and three hand estimators (HaMeR [25], Hamba [38], and WiLoR [26]),
yielding nine combinations plus body-only baselines (Tab. 3).
```

<!-- PAGE 11 -->

```text
                                                                 Tamaththul3D         11

Table 3: Modularity ablation on the SGNify dataset [10]. PA-MPVPE (mm) reported
per region; lower is better. Bold indicates the best result per column overall; underline
indicates the best result per column among configurations that include a hand module.

       Body                Hands          Body ↓      L. Hand ↓      R. Hand ↓
                           None             60.79        19.10          18.79
                           HaMeR [25]       61.06        12.36           9.38
       OSX [3]
                           Hamba [38]       62.20        12.47           9.58
                           WiLoR [26]       61.69        12.03           8.92
                           None             35.68        16.64          18.37
                           HaMeR [25]       37.14        11.84           9.42
       SMPLest-X [37]
                           Hamba [38]       37.11        12.13           9.69
                           WiLoR [26]       36.55        11.47           8.94
                           None            28.46         18.17          17.47
                           HaMeR [25]      30.07         12.30           9.36
       SMPLer-X [2]
                           Hamba [38]      30.12         11.54           9.65
                           WiLoR [26]      29.28         10.65          8.90

    Findings. Table 3 supports three claims about the modularity of the de-
sign. (1) Hand accuracy is invariant to body-backbone choice. Body PA-MPVPE
varies by over 30 mm across the three body estimators (OSX 60.79, SMPLest-X
35.68, SMPLer-X 28.46), yet right-hand error varies by at most 0.11 mm across
body backbones for any fixed hand module (e.g. WiLoR yields 8.90/8.92/8.94
across the three bodies). The geometric IK stage therefore propagates hand accu-
racy independently of the body backbone. (2) Hand-module ranking is consistent
across body backbones. For every body backbone, right-hand error obeys WiLoR
< HaMeR < Hamba: the integration composes predictably with the hand esti-
mator and scales with its accuracy rather than being tuned to any specific model.
(3) The 2D-supervised shoulder refinement preserves body fidelity. Adding any
hand module degrades body PA-MPVPE by at most ∼2 mm, confirming that the
optimization does not disrupt the body pose produced by the chosen backbone.
Together these results show that the geometric IK integration is decoupled from
the specific choice of body and hand estimators, and that SMPLer-X + WiLoR is
the strongest combination among current estimators rather than a configuration
the pipeline is tied to.


Temporal Smoothness As quantified in Tab. 4, temporal smoothing produces
substantially more stable reconstructions than the per-frame baseline. Following
WiLoR [26], we report Jitter (the mean magnitude of the third derivative (jerk)
of joint position, which isolates high-frequency motion noise from intentional
articulation), and RTE, the mean frame-to-frame wrist displacement, which
captures spatial stability. Compared to DexAvatar’s per-frame output, Tamath-
thul3D reduces hand jitter by 83.2%, body jitter by 83.9%, and RTE by 62.3%.
The evaluation spans a 560-frame multi-signer sequence with finite-difference
windows masked at signer boundaries so scene cuts do not inflate the metrics.
```

<!-- PAGE 12 -->

```text
 12                             Alghamdi et al.

                        20000                                                    DexAvatar                                                                                   DexAvatar
                                                                                 Tamaththul3D                                6000                                            Tamaththul3D




                                                                                                   Vertex deviation (mm/f)
                                                                                                                             5000
Jitter (mm/f3)          15000
                                                                                                                             4000
                        10000                                                                                                3000
                                                                                                                             2000
                         5000
                                                                                                                             1000
                            0                                                                                                   0
                                0              100   200     300           400      500                                             0   100        200      300   400              500
                                                           Frame                                                                                          Frame
 Fig. 4: Per-frame upper-body + hand jit- Fig. 5: Per-frame mesh vertex deviation
 ter (mm/f3 ). Dashed lines mark signer (mm/f), capturing total avatar surface mo-
 boundaries.                              tion frame-to-frame.

 Table 4: Runtime and temporal stability on 560 frames spanning five signers (identical
 hardware). Jitter and RTE follow WiLoR [26]. Lower is better.

               Method                                      Time (s/f ) ↓                  Jitter Hands ↓                                      Jitter Body ↓                 RTE ↓
               DexAvatar [12]                                  21.60                              1783.64                                            1791.15                 572.52
               Tamaththul3D                                        0.67                           299.14                                             289.02                 215.53

     Figures 4 and 5 visualise this stability frame-by-frame. DexAvatar (orange)
 exhibits sharp, high-amplitude spikes throughout every signer segment: these
 are the per-frame optimisation residuals that manifest as visible avatar jitter.
 while Tamaththul3D (blue) tracks a substantially lower baseline, with isolated
 low-amplitude excursions corresponding to genuine sign articulation. The two
 metrics agree: joint-space jerk and full-mesh vertex deviation both show the
 same DexAvatar–Tamaththul3D contraction across all five signer segments, con-
 firming that the smoothing acts uniformly across body, hands, and the resulting
 mesh surface. This stability is essential for the rendered avatar to be intelligi-
 ble to a viewer: a jittery output is perceived as noise regardless of per-frame
 reconstruction accuracy.
                                    How2Sign                       KArSL                        CSL‐Daily                                 PHOENIX‐2014T                 ISL‐CSLR
         Reference
         Baseline
         SGNify
         DexAvatar
         Tamaththul3D




 Fig. 6: Qualitative generalization across five sign language datasets: How2Sign [6],
 KArSL [8], CSL-Daily [39], PHOENIX-2014T [33], and ISL-CSLTR [40]. Rows: refer-
 ence, SMPLer-X, SGNify [10], DexAvatar [12], Tamaththul3D.
```

<!-- PAGE 13 -->

```text
                                                                                                                                     Tamaththul3D                          13

Generalization Across Sign Languages Tamaththul3D is a fully closed-
form pipeline with no learned parameters of its own, meaning its geometric
IK solver and coordinate conversion apply universally without dataset-specific
adaptation. The pipeline is applied to five sign language datasets spanning four
languages and three continents. As shown in Fig. 6, Tamaththul3D consistently
outperforms all baselines across all five datasets. SGNify, relying on ASL/GSL
linguistic priors, produces severe kinematic artifacts on out-of-distribution sign-
ing styles. DexAvatar exhibits artifacts particularly on KArSL and ISL-CSLTR,
where learned optimization priors generalize poorly outside Western sign lan-
guage distributions. The geometry-driven approach is unaffected by this distri-
bution shift, confirming that closed-form geometric alignment generalizes where
learned priors do not.


               (Dont sit on the table) ‫ﻻ ﺗﺠﻠﺲ ﻋﲆ اﻟﻄﺎوﻟﺔ‬   (The teacher is calling my sister) ‫اﻟﻤﻌﻠﻤﺔ ﺗﻨﺎدي اﺧﺘﻲ‬   (In the summer, I go ﬁshing) ‫ﻓﻲ اﻟﺼﻴﻒ اذﻫﺐ ﻟﺼﻴﺪ اﻟﺴﻤﻚ‬
Reference
SGNify
DexAvatar
Tamaththul3D




Fig. 7: Qualitative evaluation on Ishara-500. Three signers in culturally distinct attire
(abaya/niqab, thobe/guthra, form-fitting); four frames per sentence with translation
above. Rows: reference, SGNify [10], DexAvatar [12], Tamaththul3D (ours).

Evaluation on Ishara-500 Fig. 7 evaluates Tamaththul3D on Ishara-500
across three signers wearing culturally distinct attire. Since no 3D ground-truth
annotations exist for any Arabic Sign Language dataset, evaluation is qualita-
tive, a limitation of the field’s current annotation infrastructure that the released
Ishara-500 annotations are intended to help address. Compared to DexAvatar,
Tamaththul3D produces more accurate hand articulation and anatomically plau-
sible body posture across all three cultural clothing conditions.


5                           Limitations
Tamaththul3D inherits constraints from each constituent component. The most
persistent challenge is traditional Saudi clothing: SMPLer-X was trained pre-
dominantly on form-fitting Western attire, so thobes and abayas cause the model
to lose the silhouette cues it relies on for shoulder and torso localization. Me-
diaPipe 2D supervision partially compensates by providing clothing-invariant
keypoint guidance, but underlying shape estimation remains unreliable in these
cases.
```

<!-- PAGE 14 -->

```text
14     Alghamdi et al.

    WiLoR’s hand localization fails predictably under two conditions: severe
inter-hand occlusion and extreme lateral viewing angles, which together affect
roughly 5% of frames; fallback to SMPLer-X’s native hand estimates in these
cases produces noticeably weaker results.
    Finally, quantitative evaluation is constrained by the absence of 3D ground-
truth annotations for any Arabic Sign Language dataset. The SGNify benchmark
covers Western sign languages; SSL results are therefore evaluated perceptually
rather than numerically, a limitation of the field’s annotation infrastructure that
the released Ishara-500 annotations are intended to help address.


6    Conclusion

We presented Tamaththul3D, a reconstruction pipeline for 3D sign language
avatars that integrates SMPLer-X [2], WiLoR [26], and MediaPipe [11] through
closed-form geometric forearm alignment and 2D-supervised shoulder refine-
ment. The integration is decoupled from any specific body or hand estimator:
substituting alternative backbones preserves hand accuracy to within 0.1 mm,
making the design directly reusable as the underlying estimators improve. The
pipeline achieves state-of-the-art hand accuracy while maintaining competitive
body pose, and generalizes without dataset-specific adaptation. Applied to Ishara-
500 [9], Tamaththul3D produces the first SMPL-X annotations for any Arabic
Sign Language dataset, providing the foundation for downstream accessibility
technologies in education, telecommunication, and cultural preservation for the
Arab Deaf community.
    Ethical Statement. This work performs no new human-subject data collec-
tion. Ishara-500 [9] and the SGNify benchmark [10] are used under their respec-
tive institutional review board approvals, solely for annotation and evaluation
purposes; no additional IRB review was required.


References

 1. Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A.
    Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3d
    hands, face, and body from a single image, 2019. 2, 3, 7, 8, 9
 2. Zhongang Cai, Wanqi Yin, Ailing Zeng, Chen Wei, Qingping Sun, Yanjun Wang,
    Hui En Pang, Haiyi Mei, Mingyuan Zhang, Lei Zhang, Chen Change Loy, Lei Yang,
    and Ziwei Liu. Smpler-x: Scaling up expressive human pose and shape estimation,
    2024. 2, 3, 5, 9, 10, 11, 14
 3. Jing Lin, Ailing Zeng, Haoqian Wang, Lei Zhang, and Yu Li. One-stage 3d whole-
    body mesh recovery with component aware transformer, 2023. 2, 3, 9, 10, 11
 4. Gyeongsik Moon, Hongsuk Choi, and Kyoung Mu Lee. Accurate 3d hand pose
    estimation for whole-body 3d human mesh estimation, 2022. 2, 4
 5. Vasileios Baltatzis, Rolandos Alexandros Potamias, Evangelos Ververas, Guanx-
    iong Sun, Jiankang Deng, and Stefanos Zafeiriou. Neural sign actors: A diffusion
    model for 3d sign language production from text, 2024. 2, 4, 9, 10
```

<!-- PAGE 15 -->

```text
                                                               Tamaththul3D        15

 6. Amanda Duarte, Shruti Palaskar, Lucas Ventura, Deepti Ghadiyaram, Kenneth
    DeHaan, Florian Metze, Jordi Torres, and Xavier Giro i Nieto. How2sign: A large-
    scale multimodal dataset for continuous american sign language, 2021. 2, 4, 12
 7. Hamzah Luqman. Arabsign: A multi-modality dataset and benchmark for continu-
    ous arabic sign language recognition. In 2023 IEEE 17th International Conference
    on Automatic Face and Gesture Recognition, FG 2023, 2023 IEEE 17th Interna-
    tional Conference on Automatic Face and Gesture Recognition, FG 2023, United
    States, 2023. Institute of Electrical and Electronics Engineers Inc. 2, 4
 8. {Ala Addin I.} Sidig, Hamzah Luqman, Sabri Mahmoud, and Mohamed Mohandes.
    Karsl: Arabic sign language database. ACM Transactions on Asian and Low-
    Resource Language Information Processing, 20(1), April 2021. Publisher Copyright:
    © 2021 ACM. 2, 4, 12
 9. Sarah Alyami, Hamzah Luqman, Sadam Al-Azani, Maad Alowaifeer, Yazeed Al-
    harbi, and Yaser Alonaizan. Isharah: A large-scale multi-scene dataset for contin-
    uous sign language recognition, 2025. 2, 8, 14
10. Maria-Paola Forte, Peter Kulits, Chun-Hao Paul Huang, Vasileios Choutas, Dim-
    itrios Tzionas, Katherine J. Kuchenbecker, and Michael J. Black. Reconstructing
    signing avatars from video using linguistic priors. In IEEE/CVF Conf. on Com-
    puter Vision and Pattern Recognition (CVPR), pages 12791–12801, June 2023. 2,
    4, 8, 9, 11, 12, 13, 14
11. Camillo Lugaresi, Jiuqiang Tang, Hadon Nash, Chris McClanahan, Esha Uboweja,
    Michael Hays, Fan Zhang, Chuo-Ling Chang, Ming G Yong, Juhyun Lee, et al.
    MediaPipe: A framework for building perception pipelines. arXiv preprint
    arXiv:1906.08172, 2019. 2, 3, 5, 9, 14
12. Kaustubh Kundu, Hrishav Bakul Barua, Lucy Robertson-Bell, Zhixi Cai, and Kalin
    Stefanov. Dexavatar: 3d sign language reconstruction with hand and body pose
    priors, 2025. 2, 4, 9, 12, 13
13. Ce Zheng, Wenhan Wu, Chen Chen, Taojiannan Yang, Sijie Zhu, Ju Shen, Nasser
    Kehtarnavaz, and Mubarak Shah. Deep learning-based human pose estimation: A
    survey, 2023. 2, 4
14. Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and
    Michael J. Black. SMPL: A skinned multi-person linear model. ACM Trans.
    Graphics (Proc. SIGGRAPH Asia), 34(6):248:1–248:16, October 2015. 3
15. Angjoo Kanazawa, Michael J. Black, David W. Jacobs, and Jitendra Malik. End-
    to-end recovery of human shape and pose. In Computer Vision and Pattern Recog-
    nition (CVPR), 2018. 3, 8
16. Nikos Kolotouros, Georgios Pavlakos, Michael J. Black, and Kostas Daniilidis.
    Learning to reconstruct 3d human pose and shape via model-fitting in the loop,
    2019. 3
17. Yu Rong, Takaaki Shiratori, and Hanbyul Joo. Frankmocap: Fast monocular 3d
    hand and body motion capture by regression and integration, 2020. 3, 9
18. Yao Feng, Vasileios Choutas, Timo Bolkart, Dimitrios Tzionas, and Michael J.
    Black. Collaborative regression of expressive bodies using moderation, 2021. 3, 9
19. Hongwen Zhang, Yating Tian, Yuxiang Zhang, Mengcheng Li, Liang An, Zhenan
    Sun, and Yebin Liu. PyMAF-X: Towards well-aligned full-body model regression
    from monocular images. IEEE TPAMI, 2023. 3
20. Muhammed Kocabas, Nikos Athanasiou, and Michael J. Black. VIBE: Video in-
    ference for human body pose and shape estimation. In CVPR, 2020. 3
21. Jiefeng Li, Chao Xu, Zhicun Chen, Siyuan Bian, Lixin Yang, and Cewu Lu. Hy-
    brIK: A hybrid analytical-neural inverse kinematics solution for 3D human pose
    and shape estimation. In CVPR, 2021. 3
```

<!-- PAGE 16 -->

```text
16      Alghamdi et al.

22. Javier Romero, Dimitrios Tzionas, and Michael J. Black. Embodied hands: mod-
    eling and capturing hands and bodies together. ACM Transactions on Graphics,
    36(6):1–17, November 2017. 4, 5, 9
23. Kevin Lin, Lijuan Wang, and Zicheng Liu. End-to-end human pose and mesh
    reconstruction with transformers. In CVPR, 2021. 4
24. Kevin Lin, Lijuan Wang, and Zicheng Liu. Mesh Graphormer. In ICCV, 2021. 4
25. Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David
    Fouhey, and Jitendra Malik. Reconstructing hands in 3D with transformers. In
    CVPR, 2024. 4, 10, 11
26. Rolandos Alexandros Potamias, Jinglei Zhang, Jiankang Deng, and Stefanos
    Zafeiriou. Wilor: End-to-end 3d hand localization and reconstruction in-the-wild,
    2025. 4, 5, 9, 10, 11, 12, 14
27. Christian Zimmermann, Duygu Ceylan, Jimei Yang, Bryan Russel, Max Argus,
    and Thomas Brox. Freihand: A dataset for markerless capture of hand pose and
    shape from single rgb images. In IEEE International Conference on Computer
    Vision (ICCV), 2019. 4
28. Gyeongsik Moon, Shoou-I Yu, He Wen, Takaaki Shiratori, and Kyoung Mu Lee.
    Interhand2.6m: A dataset and baseline for 3d interacting hand pose estimation
    from a single rgb image. In European Conference on Computer Vision (ECCV),
    2020. 4
29. Shreyas Hampali, Mahdi Rad, Markus Oberweger, and Vincent Lepetit. Honno-
    tate: A method for 3d annotation of hand and object poses, 2020. 4
30. Ben Saunders, Necati Cihan Camgoz, and Richard Bowden. Progressive trans-
    formers for end-to-end sign language production. In ECCV, 2020. 4
31. Hezhen Hu, Weichao Zhao, Wengang Zhou, and Houqiang Li. SignBERT+: Hand-
    model-aware self-supervised pre-training for sign language understanding. IEEE
    TPAMI, 2023. 4
32. Dongxu Li, Cristian Rodriguez, Xin Yu, and Hongdong Li. Word-level deep sign
    language recognition from video: A new large-scale dataset and methods compari-
    son. In The IEEE Winter Conference on Applications of Computer Vision, pages
    1459–1469, 2020. 4
33. Oscar Koller, Jens Forster, and Hermann Ney. Continuous sign language recog-
    nition: Towards large vocabulary statistical recognition systems handling multiple
    signers. Computer Vision and Image Understanding, 141:108–125, 2015. 4, 12
34. Przemysław Dobrowolski. Swing-twist decomposition in clifford algebra, 2015. 6
35. Jingjing Qi, Zhenjiang Miao, Zhifei Wang, and Shujun Zhang. Several methods of
    smoothing motion capture data. Proceedings of SPIE - The International Society
    for Optical Engineering, 8009, 04 2011. 7
36. John C. Gower. Generalized procrustes analysis. Psychometrika, 40(1):33–51, 1975.
    8
37. Wanqi Yin, Zhongang Cai, Ruisi Wang, Ailing Zeng, Chen Wei, Qingping Sun,
    Haiyi Mei, Yanjun Wang, Hui En Pang, Mingyuan Zhang, Lei Zhang, Chen Change
    Loy, Atsushi Yamashita, Lei Yang, and Ziwei Liu. Smplest-x: Ultimate scaling
    for expressive human pose and shape estimation. IEEE Transactions on Pattern
    Analysis and Machine Intelligence, 48(2):1778–1794, 2026. 10, 11
38. Haoye Dong, Aviral Chharia, Wenbo Gou, Francisco Vicente Carrasco, and Fer-
    nando D De la Torre. Hamba: Single-view 3d hand reconstruction with graph-
    guided bi-scanning mamba. Advances in Neural Information Processing Systems,
    37:2127–2160, 2024. 10, 11
```

<!-- PAGE 17 -->

```text
                                                              Tamaththul3D        17

39. Hao Zhou, Wengang Zhou, Weizhen Qi, Junfu Pu, and Houqiang Li. Improving
    sign language translation with monolingual data by sign back-translation. In Pro-
    ceedings of the IEEE/CVF conference on computer vision and pattern recognition,
    pages 1316–1325, 2021. 12
40. R Elakkiya and B Natarajan. ISL-CSLTR: Indian Sign Language Dataset for
    Continuous Sign Language Translation and Recognition, 2021. 12
```
