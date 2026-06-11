# Reconstructing Signing Avatars From Video Using Linguistic Priors

**Maria-Paola Forte**$^1$ · **Peter Kulits**$^2$ · **Chun-Hao Huang**$^2$ · **Vasileios Choutas**$^2$ · **Dimitrios Tzionas**$^2$ · **Katherine J. Kuchenbecker**$^1$ · **Michael J. Black**$^2$  
$^1$*Max Planck Institute for Intelligent Systems, Stuttgart and Tübingen, Germany*  
$^2$*Max Planck Institute for Intelligent Systems, Tübingen, Germany*  
`{forte, kjk}@is.mpg.de`, `{kulits, chuang2, vchoutas, dtzionas, black}@tue.mpg.de`  
**Project Page:** [sgnify.is.tue.mpg.de](https://sgnify.is.tue.mpg.de)

---

## Abstract
Sign language (SL) is the primary method of communication for the 70 million Deaf people around the world. Video dictionaries of isolated signs are a core SL learning tool. Replacing these with 3D avatars can aid learning and enable AR/VR applications, improving access to technology and online media. However, little work has attempted to estimate expressive 3D avatars from SL video; occlusion, noise, and motion blur make this task difficult. We address this by introducing novel linguistic priors that are universally applicable to SL and provide constraints on 3D hand pose that help resolve ambiguities within isolated signs. Our method, **SGNify**, captures fine-grained hand pose, facial expression, and body movement fully automatically from in-the-wild monocular SL videos. 

We evaluate SGNify quantitatively by using a commercial motion-capture system to compute 3D avatars synchronized with monocular video. SGNify outperforms state-of-the-art 3D body-pose- and shape-estimation methods on SL videos. A perceptual study shows that SGNify’s 3D reconstructions are significantly more comprehensible and natural than those of previous methods and are on par with the source videos. Code and data are available at `sgnify.is.tue.mpg.de`.

---

## 1. Introduction
It is estimated that over 466 million people have disabling hearing loss [11] and more than 70 million people use sign language (SL) as their primary means of communication [53]. Increasing use of digital communication motivates research on capturing, understanding, modeling, and synthesizing expressive 3D SL avatars. Existing datasets and dictionaries used in SL recognition (SLR), translation (SLT), and production (SLP) are primarily limited to 2D video because the technology required to capture 3D movement is prohibitively expensive, requires expertise to operate, and may limit the movements of the signer. 

Dictionaries of isolated signs are a core SL learning tool, and many SLs have online 2D video dictionaries. The Deaf community is actively seeking 3D dictionaries of isolated signs to aid learning [40]. The current approach to creating such 3D signing dictionaries is fully manual, requiring an artist or a HamNoSys [19] expert, and the resulting avatars often move unnaturally [1]. We aim to automatically reconstruct expressive 3D signing avatars from monocular SL video, which we term **Sign Language Capture (SLC)**. We focus on SLC of isolated signs.

3D reconstruction of human pose and shape has received significant attention, but accurate 3D hand-pose estimation remains challenging from in-the-wild video. Challenges include the high number of degrees of freedom present in hands [3], frequent occurrence of self-contact and self-occlusions [37, 47], low resolution, and motion blur caused by fast motions [51] that cause hand pose to be unrecognizable in many frames (see Fig. 1). To address these issues, we exploit the linguistic nature of SL itself to develop novel priors that help disambiguate hand poses in SL videos, leading to accurate 3D reconstructions. This is a novel use of linguistic "side information" to improve 3D reconstruction.

Based on hand movements and poses, Battison [2] defines five linguistic classes that contain all SL signs. We build on that work to define eight classes and formalize these as mathematical priors on 3D hand shape. We combine Battison’s first two classes and place all one-handed signs in class 0, while two-handed signs are arranged in classes 1, 2, or 3, depending on how the non-dominant hand participates in the articulation of the sign. We then divide each of these four classes into two subclasses depending on whether the pose of the active hand(s) changes during the articulation of the sign. We introduce two class-dependent SL linguistic constraints that capture **1) symmetry** and **2) hand-pose invariance**. 

Under Battison’s SL symmetry condition [2], when both hands actively move, the articulation of the fingers must be identical; the same is true for one class of two-handed signs in which only the dominant hand moves. We formalize this concept as a regularization term that encourages the pose of the two hands to be similar for such signs. Coupling the hand poses in this way effectively increases the image evidence for a pose, which improves estimates for challenging videos. Our invariance constraint uses the observation that hand pose is either static or transitions smoothly from one pose to another during the articulation of the sign; other significant changes to hand pose are not common in SL. Specifically, we extract a characteristic "reference pose sequence" (RPS) to describe each local hand pose during the sign articulation, and we penalize differences between the RPS and the estimated hand pose in each frame. These two priors of symmetry and hand-pose invariance are universally applicable to all sign languages.

The hands alone, however, are not sufficient to accurately reproduce SL. Information is conveyed holistically in SL through hand gestures, facial expressions, and upper-body movements in 3D space. To combine these, we use a 3D whole-body model, SMPL-X [42], that jointly models this information.

Our novel hand-pose constraints are formulated to be incorporated into the loss function for training a neural network regressor or into the objective function of optimization-based methods. In general, optimization-based methods are more computationally intensive but produce more accurate results when limited training data is available, so we take this approach here and build on the SMPLify-X method [42]. To appropriately incorporate our terms into the objective function, we need to know the class of the sign. We train a simple model that extracts features from the raw video and determines the class to which the depicted sign belongs. While SMPLify-X is a good foundation for the hands and body, we find that it does not capture expressive facial motions well. Consequently, we use a more expressive face regressor, SPECTRE [17], to capture the face parameters. We call our method **SGNify**.

To quantitatively evaluate SGNify, we capture a native German (DGS) signer with a frontal RGB camera synchronized with a 54-camera Vicon motion capture system and recover ground-truth meshes from the Vicon markers [33]. We run SGNify on the RGB video and compute 3D vertex-to-vertex (V2V) error between our resulting avatars and the ground-truth meshes. We find that SGNify reconstructs SMPL-X meshes more accurately than the competition.

We conduct a perceptual evaluation in which we present proficient signers with a video of either an estimated SMPL-X avatar or the real-person source video and task them with identifying the sign being performed. Participants also rate their ease in recognizing the sign and the naturalness of the articulation. Our results show that SGNify reconstructs 3D signs that are as recognizable as the original videos and consistently more recognizable, easier to understand, and more natural than the existing state of the art. We also evaluate SGNify in a multi-view setting and on continuous signing videos. Despite not being designed for the latter, SGNify captures the meaning in continuous SL.

---

## 2. Related Work

### Expressive 3D Humans From RGB Images
Until recently, human-pose estimation has focused on the estimation of 2D [10] or 3D [52] joints of the hands and body, as well as those of facial features [4] from single images. In addition to methods that estimate a sparse set of landmarks, there are multiple methods that estimate the parameters of morphable models for the hand [20, 30, 36, 59], face [9, 14, 16, 18], and body [6, 23, 25–27, 32, 39, 58]. The advent of expressive 3D body models like SMPL-X [42], Adam [24], and GHUM [55] has enabled research on estimating the full 3D body surface [7, 15, 42, 45, 50, 54, 56]. Such body models are ideal for representing the expressiveness of SL but have rarely been applied to this domain [28].

### Human Pose for Sign Language
To enable detailed 3D pose estimation from images, How2Sign [12] provides 3D skeleton reconstructions for three hours of data captured in a Panoptic Studio [22]. However, the skeletal representation lacks the richness of a full 3D body model and omits surface details that are important for communication [38]. Kratimenos *et al.* [28] use SMPLify-X to estimate 3D pose and shape on the GSLSI sign-language dataset [49]. They compare SL recognition accuracy using features from raw RGB images, OpenPose [5] 2D skeletons, and SMPL-X bodies and observe the best automated recognition results with SMPL-X, illustrating the benefit of using a 3D model. They also highlight the importance of capturing the face and body; in an ablation study, they show that neglecting the face and body harms recognition accuracy [28]. However, their SMPL-X reconstructions use existing off-the-shelf methods and lack visual realism. 

SMPLify-X [42] and other recent 3D pose-reconstruction methods [15, 45], as well as keypoint detectors, struggle when applied to SL video due to challenging self-occlusion, hand-hand and hand-body interactions [38], motion blur [51], and cropping inherent to SL. SignPose [29] is a 3D-pose-lifting method for SL; it uses manually created synthetic SL animations to infer a textured avatar from single RGB images. SignPose requires all OpenPose keypoints above the pelvis to be detected, which is unrealistic in noisy SL videos. We address these challenges by incorporating sign-language knowledge in the form of linguistic constraints. Since the early 2000s, the integration of linguistic information has been known to be beneficial to both SLR [8] and SLP [34], but this strategy has not previously been applied to SLC.

---

## 3. Method
We introduce SGNify, an offline method for reconstructing 3D body shape and pose of SL from monocular RGB video. SGNify centers around a key insight: SL signs follow universal linguistic rules that can be formulated as class-specific priors and used to improve hand-pose estimation.

### 3.1. SMPLify-SL: Baseline for Sign-Language Video
Our baseline method builds on SMPLify-X [42], which estimates SMPL-X parameters from RGB images. SMPL-X is a 3D body model, representing whole-body pose and shape, including finger articulations and facial expressions. SMPL-X is a function, $M(	heta, eta, \psi)$, parameterized by body pose $	heta$ (including hand pose $	heta_h$), body shape $eta$, and facial expressions $\psi$, that outputs a 3D body mesh.

To create a strong baseline, we extend SMPLify-X to video by adapting it in the following ways:
1. We cope with the upper-body framing typical of SL videos by changing the heuristic used for camera initialization and the estimation of the out-of-view lower-body joints.
2. Since human motion is locally smooth in time, we initialize $	heta_t \in \mathbb{R}^{|	heta|}$ with $	heta_{t-1}$ and include a zero-velocity loss on the hands and body to encourage smooth reconstructions.
3. We estimate shape parameters ($eta$) over multiple frames by taking the median of the parameter estimates and not optimizing them during the per-frame reconstruction.
4. To better capture the frequent hand-hand and hand-body interactions (mainly with the face and the chest), we employ the more robust self-contact loss of Müller *et al.* [39] instead of the original SMPLify-X interpenetration term.
5. For each frame, we pre-compute the facial expressions ($\psi$) and jaw poses with SPECTRE [17]. These parameters are substituted into SMPL-X at the end of the optimization. SPECTRE can be swapped for any method whose expression parameters are consistent with those of SMPL-X, *e.g.*, EMOCA [9]. 

We denote the baseline **SMPLify-SL**.

### 3.2. Linguistic Constraints
State-of-the-art optimization- and regression-based human pose estimation methods struggle on SL video, particularly with the estimation of hand pose. We address this challenge by formulating linguistic constraints as additional losses on hand pose and integrating them into the SMPLify-SL objective function. First, we adapt the five sign-classification and morpheme-structure conditions introduced for American Sign Language (ASL) by Battison [2] to divide signs into four primary classes:
* **Class 0:** One-handed signs in which only the dominant hand articulates the sign.
* **Class 1:** Two-handed signs in which both hands are active. They share the same poses and perform the same movement in a synchronous or alternating pattern. This class includes all signs that follow Battison’s symmetry condition [2].
* **Class 2:** Two-handed signs in which the dominant hand is active, the non-dominant hand is passive (its position and pose do not change during the articulation of the sign), and the two hands have the same initial pose.
* **Class 3:** Two-handed signs in which the dominant hand is active, the non-dominant hand is passive, and the two hands have different hand poses. All signs in this class follow Battison’s dominance condition [2].

We further divide each class into two subclasses: **subclass a** contains signs in which the hand pose of the active hand(s) does not change throughout the articulation of the sign (static), and **subclass b** contains all signs in which the hand pose changes (transitioning). Note that the division into these classes is not limited to ASL; Eccarius *et al.* [13] show that the phonological and prosodic properties of ASL can be successfully transferred to other sign-language lexicons.

We then convert these linguistic classes into two 3D pose constraints: **hand-pose symmetry** and **hand-pose invariance**. Signs in the same class share the same constraints (see Tab. 1).

#### 3.2.1 Hand-Pose Symmetry
We encourage the left and right hand poses to match for the relevant classes (classes 1a, 1b, and 2a in Tab. 1):
$$L_s = \lambda_s \|	heta_t^r - r(	heta_t^l)\|_2^2$$
where $	heta_t^r$ is the finger articulation of the right hand, and $r(	heta_t^l)$ is a reflection function to represent the articulation of the fingers of the left hand as if it were a right hand. This loss penalizes differences in finger poses between the hands.

#### 3.2.2 Hand-Pose Invariance
Each sign has a characteristic reference hand pose sequence (RPS). The RPS defines the hand pose that we expect at each time $t$ during the articulation of the sign. The hand-pose-invariance constraint penalizes differences between the reference hand pose $	heta_{	ext{ref},t}^h \in 	ext{RPS}^h$ and the estimated hand pose $	heta_t^h$:
$$L_i^h = \lambda_i \|	heta_t^h - 	heta_{	ext{ref},t}^h\|_2^2$$
where $h$ represents either the left or the right hand.

Throughout each sign, the hand pose either stays static or transitions between two poses. When static, only one hand pose, $	heta_{	ext{ref}}^h$, is representative of the RPS. Signs where the hand pose is transitioning are characterized by two reference hand poses, $	heta_{	ext{ref},i}^h$ and $	heta_{	ext{ref},f}^h$, corresponding respectively to the initial and final poses. We interpolate $	heta_{	ext{ref},i}^h$ and $	heta_{	ext{ref},f}^h$ with spherical linear interpolation [46] to obtain intermediate poses. We presently do not consider signs with repeated hand-pose transitions, *e.g.*, STORY in ASL, which occur in a small percentage of signs (~3%).

| Class | Hand-Pose Symmetry | Hand-Pose Invariance (Dominant / Non-dominant) |
| :---: | :---: | :---: |
| **0a** | $	imes$ | static / $	imes$ |
| **0b** | $	imes$ | transitioning / $	imes$ |
| **1a** | $\checkmark$ | static / static |
| **1b** | $\checkmark$ | transitioning / transitioning |
| **2a** | $\checkmark$ | static / static |
| **2b** | $	imes$ | transitioning / static |
| **3a** | $	imes$ | static / static |
| **3b** | $	imes$ | transitioning / static |

**Table 1.** Linguistic constraints defining the eight sign classes.

### 3.3. Automatization
To work fully automatically, SGNify must:
1. Estimate the poses needed to enforce the hand-pose-invariance constraint.
2. Classify which sign group is present in a video sequence.

To estimate the reference hand poses ($	heta_{	ext{ref}}^h$, $	heta_{	ext{ref},i}^h$, and $	heta_{	ext{ref},f}^h$), our method selects candidate frames in the core part of the sign using hand-keypoint detection confidences, and it uses SMPLify-X (adapted to SL cropping) to reconstruct a preliminary 3D hand pose for each candidate frame. With static hand poses, $	heta_{	ext{ref}}^h$ is obtained by taking the average hand poses of these candidates. With transitioning hand poses, the core part of a sign is divided into two intervals, and $	heta_{	ext{ref},i}^h$ and $	heta_{	ext{ref},f}^h$ correspond to the average hand poses of the candidate frames in the first and second intervals, respectively.

The constraints applied to each sign depend on its sign group; we have six sign groups because classes 1a & 2a share the same constraints, as do 2b & 3b (see Tab. 1). Since there is insufficient paired data to train a CNN classifier, we use an intuitive and interpretable decision tree trained on extracted 2D and 3D pose features. Our features are invariant to the handedness of the signer and include:
1. The minimum of the maximum height differences of each wrist across the sequence: $\min(\{w_r\}_{\max} - \{w_r\}_{\min}, \{w_l\}_{\max} - \{w_l\}_{\min})$, where $w_l$ and $w_r$ are the heights of the left and right wrists, respectively.
2. The cosine distance between the initial poses of each hand: $	ext{CosDist}(	heta_{	ext{ref},i}^r, 	heta_{	ext{ref},i}^l)$.
3. The maximum of the two cosine distances between each initial and final hand pose: $\max(	ext{CosDist}(	heta_{	ext{ref},i}^r, 	heta_{	ext{ref},f}^r), 	ext{CosDist}(	heta_{	ext{ref},i}^l, 	heta_{	ext{ref},f}^l))$.

We train our sign-group classifier on over 3,000 videos from the Corpus-based Dictionary of Polish Sign Language (CDPSL) [31], which are annotated with HamNoSys [19]. We construct a grammar to convert HamNoSys annotations into our group labels. This dataset is not used in our quantitative analysis or perceptual study.

### 3.4. SGNify Extensions
First, we follow Huang *et al.* [21] to extend SGNify to work on multi-view videos. Second, we propose a baseline method for continuous SLC (CSLC). CSLC introduces additional challenges, such as the segmentation of sentences into signs; this is an active field of research. When a sentence is given as input, we use Renz *et al.* [44] to segment the input video and then process each segment with SGNify. The first frame of each segment is initialized from the last frame of the previous one.

---

## 4. Dataset
To quantitatively evaluate SGNify as a viable method for SLC, we collected motion-capture data with ground-truth SMPL-X bodies articulating signs. Our dataset represents the first publicly available expressive full-4D capture of isolated SL signs. The experimental procedure was reviewed by the ethics council of the University of Tübingen without objections or remarks (709/2021B02).

In consultation with a Deaf DGS teacher and a DGS interpreter, we defined a German SL corpus consisting of 57 isolated signs. The selected signs cover a wide range of challenges for SLC, such as self-contact and self-occlusion. Table 2 summarizes the number of signs collected for each of the eight classes. Signs of subclass b are less common, and this is reflected in our corpus.

| Class | 0a | 0b | 1a | 1b | 2a | 2b | 3a | 3b | Total |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **# Signs** | 12 | 3 | 14 | 3 | 11 | 2 | 10 | 2 | **57** |

**Table 2.** Number of signs captured for each class.

We captured a native right-handed DGS signer with a Vicon mocap system at 120 fps, synchronized with a frontal $4112 	imes 3008$ RGB camera at 60 fps, framing an upper-body view as typically found in SL video. The hands start and end at rest at the signer’s sides, and each sign lasts between 1.7 and 3.5 seconds after trimming. In total, our dataset comprises 16,608 mocap frames and 8,304 RGB frames. 

To obtain ground-truth SMPL-X meshes, we scanned the participant in a 4D body scanner in several poses. The SMPL-X mesh was registered to these scans and averaged to obtain a personalized body-shape mesh. MoSh++ was then used to fit this mesh to the mocap markers [33]. Marker-based mocap is useful for evaluating ground truth but is not practical for SLC at scale: it is expensive, requires expertise, the reflective markers can influence contact-heavy motions, and processing the resulting data is time-consuming. If our monocular method can approach the performance of mocap, it will be widely applicable.

---

## 5. Experiments

### 5.1. Quantitative Evaluation
To emulate in-the-wild data, which might have very low resolution, low framerate, and an occluded lower body, we pre-processed our high-quality video data to a resolution of $514 	imes 300$ at 30 fps, and we cropped the input images above the pelvis. We used the synchronized meshes captured from the observed Vicon markers [33] as ground truth for evaluation. Since all tested methods estimate SMPL-X meshes with the same topology, we compute the mean per-vertex error (TR-V2V) by considering the vertices above the pelvis. The prefix "TR" means that we translationally align the mesh reconstructed for each frame with the ground truth before computing errors. We compute the quantitative results on only the central frames (in total, 2,872 RGB frames).

| Method | Upper Body | Left Hand | Right Hand |
| :--- | :---: | :---: | :---: |
| **FrankMocap** [45] | 78.07 | 20.47 | 19.62 |
| **PIXIE** [15] | 60.11 | 25.02 | 22.42 |
| **PyMAF-X** [57] | 68.61 | 21.46 | 19.19 |
| **SMPLify-SL** (Baseline) | 56.07 | 22.23 | 18.83 |
| **SGNify** (Ours) | **55.63** | **19.22** | **17.50** |

**Table 3.** Evaluation on our ground-truth mocap dataset: mean TR-V2V error (mm) for five methods and three body regions.

| Method | Symmetry | Invariance | Left Hand | Right Hand | Both Hands |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SMPLify-SL** | $	imes$ | $	imes$ | 20.30 | 18.78 | 19.54 |
| **SGNify** | $\checkmark$ | $	imes$ | 18.44 | 18.39 | 18.41 |
| **SGNify** | $	imes$ | $\checkmark$ | 19.76 | **17.16** | 18.46 |
| **SGNify** | $\checkmark$ | $\checkmark$ | **17.72** | 17.29 | **17.50** |

**Table 4.** Evaluating how the linguistic constraints of symmetry and invariance affect mean TR-V2V error (mm) in symmetric signs.

| Method | Invariance | Left Hand | Right Hand | Both Hands |
| :--- | :---: | :---: | :---: | :---: |
| **SMPLify-SL** | $	imes$ | 26.09 | 18.89 | 20.50 |
| **SGNify** | $\checkmark$ | **22.22** | **17.70** | **18.60** |

**Table 5.** Evaluating how the linguistic constraint of hand-pose invariance affects mean TR-V2V error (mm) in asymmetric signs.

### 5.2. Perceptual Study
We conduct an online perceptual study to:
1. Compare SGNify with the best-performing state-of-the-art method for SL (PyMAF-X).
2. Evaluate the improvement derived from the linguistic constraints (SGNify vs. SMPLify-SL).

Our study involves 20 adult participants who all stated that they have an advanced level of proficiency (expert level) in ASL (75% are Deaf). We used SGNify, SMPLify-SL, and PyMAF-X to reconstruct avatars from 50 videos taken from *The American Sign Language Handshape Dictionary* [48].

* **Recognition Rate:** Participants recognize signs in real-person videos with an average accuracy of 90.9% and signs reconstructed by SGNify with **86.2%** accuracy. Signs reconstructed with SMPLify-SL and PyMAF-X are recognized less accurately, at 74.8% and 62.0%, respectively. Sign recognition rates with real video and SGNify are not significantly different from one another.
* **Perceived Easiness & Naturalness:** SGNify and SMPLify-SL are significantly easier to recognize than PyMAF-X. SGNify achieves significantly higher naturalness ratings compared to SMPLify-SL and PyMAF-X.

A second follow-up perceptual study (13 participants) evaluating avatar appearance (solid purple vs. wearing a t-shirt vs. fully textured human character) revealed that adding clothing and texturing do not significantly benefit actual or perceived sign recognition or the perceived naturalness of the reconstruction.

---

## 6. Discussion
Our results show that SGNify performs quantitatively better than the state of the art, in particular due to the inclusion of our novel linguistic constraints. However, we believe that a per-frame metric is not ideal for SL. To recognize a sign, the temporal evolution is crucial, and this is not captured by V2V. For example, a few slightly inaccurate frames can confuse signers during a perceptual study even if the overall V2V error remains small. In the end, what matters is whether the meaning is clear to a human. 

The perceptual study indicated that SGNify significantly outperforms the state of the art and, most importantly, produces the first 3D avatars to achieve a sign-recognition accuracy that is not statistically different from the source videos. Our study also highlights the next challenge for SLC: the need for improvements in the face including facial expressions, tongue and eye movements, mouth morphemes, and eyebrows.

---

## 7. Conclusions
We present SGNify, which estimates 3D avatars of isolated SL signs from monocular RGB video. Quantitative and qualitative experiments show that SGNify outperforms the state of the art in estimating challenging SL hand poses by leveraging constraints derived from linguistics. SGNify represents a step towards the capture of realistic 3D avatars from SL videos in the wild. Future work should explore the use of our constraints in training regression methods, real-time processing, and continuous signing.

---

## References
1. Ahmed H. Aliwy and A. Alethary Ahmed. Development of Arabic sign language dictionary using 3D avatar technologies. *Indonesian Journal of Electrical Engineering and Computer Science*, 21(1):609–616, 2021.
2. Robbin Battison. *Lexical Borrowing in American Sign Language*. Education Resources Information Center (ERIC), 1978.
3. Sara Bilal, Rini Akmeliawati, Momoh Jimoh El Salami, and Amir A. Shafie. Vision-based hand posture detection and recognition for sign language a study. In *International Conference on Mechatronics (ICOM)*, pages 1–6, 2011.
4. Adrian Bulat and Georgios Tzimiropoulos. How far are we from solving the 2D & 3D face alignment problem? (and a dataset of 230,000 3D facial landmarks). In *International Conference on Computer Vision (ICCV)*, pages 1021–1030, 2017.
5. Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, and Yaser Sheikh. OpenPose: Realtime multi-person 2D pose estimation using part affinity fields. *Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 43(1):172–186, 2021.
6. Hongsuk Choi, Gyeongsik Moon, and Kyoung Mu Lee. Beyond static features for temporally consistent 3D human pose and shape from a video. In *Computer Vision and Pattern Recognition (CVPR)*, pages 1964–1973, 2021.
7. Vasileios Choutas, Georgios Pavlakos, Timo Bolkart, Dimitrios Tzionas, and Michael J. Black. Monocular expressive body regression through body-driven attention. In *European Conference on Computer Vision (ECCV)*, pages 20–40, 2020.
8. Helen Cooper, Brian Holt, and Richard Bowden. Sign language recognition. In *Visual analysis of humans*, pages 539–562. Springer, 2011.
9. Radek Daněček, Michael J. Black, and Timo Bolkart. EMOCA: Emotion driven monocular face capture and animation. In *Computer Vision and Pattern Recognition (CVPR)*, pages 20311–20322, 2022.
10. Qi Dang, Jianqin Yin, Bin Wang, and Wenqing Zheng. Deep learning based 2D human pose estimation: A survey. *Tsinghua Science and Technology*, 24(6):663–676, 2019.
11. Adrian C. Davis and Howard J. Hoffman. Hearing loss: Rising prevalence and impact. *Bulletin of the World Health Organization*, 97(10):646, 2019.
12. Amanda Duarte, Shruti Palaskar, Lucas Ventura, Deepti Ghadiyaram, Kenneth DeHaan, Florian Metze, Jordi Torres, and Xavier Giro-i Nieto. How2Sign: A large-scale multimodal dataset for continuous american sign language. In *Computer Vision and Pattern Recognition (CVPR)*, pages 2735–2744, 2021.
13. Petra Eccarius and Diane Brentari. Symmetry and dominance: A cross-linguistic study of signs and classifier constructions. *Lingua*, 117(7):1169–1201, 2007.
14. Bernhard Egger, William A. P. Smith, Ayush Tewari, Stefanie Wuhrer, Michael Zollhoefer, Thabo Beeler, Florian Bernard, Timo Bolkart, Adam Kortylewski, Sami Romdhani, Christian Theobalt, Volker Blanz, and Thomas Vetter. 3D morphable face models - past, present and future. *ACM Transactions on Graphics*, 39(5), 2020.
15. Yao Feng, Vasileios Choutas, Timo Bolkart, Dimitrios Tzionas, and Michael Black. Collaborative regression of expressive bodies using moderation. In *International Conference on 3D Vision (3DV)*, pages 792–804, 2021.
16. Yao Feng, Haiwen Feng, Michael J. Black, and Timo Bolkart. Learning an animatable detailed 3D face model from in-the-wild images. *ACM Transactions on Graphics (SIGGRAPH)*, 40(4):1–13, 2021.
17. Panagiotis P. Filntisis, George Retsinas, Foivos Paraperas-Papantoniou, Athanasios Katsamanis, Anastasios Roussos, and Petros Maragos. Visual speech-aware perceptual 3D facial expression reconstruction from videos. *arXiv preprint arXiv:2207.11094*, 2022.
18. Jianzhu Guo, Xiangyu Zhu, Yang Yang, Fan Yang, Zhen Lei, and Stan Z. Li. Towards fast, accurate and stable 3D dense face alignment. In *European Conference on Computer Vision (ECCV)*, pages 152–168, 2020.
19. Thomas Hanke. HamNoSys - representing sign language data in language resources and language processing contexts. In *International Conference on Language Resources and Evaluation (LREC)*, volume 4, pages 1–6, 2004.
20. Yana Hasson, Gül Varol, Dimitrios Tzionas, Igor Kalevatykh, Michael J. Black, Ivan Laptev, and Cordelia Schmid. Learning joint reconstruction of hands and manipulated objects. In *Computer Vision and Pattern Recognition (CVPR)*, pages 11807–11816, 2019.
21. Chun-Hao Huang, Hongwei Yi, Markus Höschle, Matvey Safroshkin, Tsvetelina Alexiadis, Senya Polikovsky, Daniel Scharstein, and Michael J. Black. Capturing and inferring dense full-body human-scene contact. In *Computer Vision and Pattern Recognition (CVPR)*, pages 13274–13285, 2022.
22. Hanbyul Joo, Hao Liu, Lei Tan, Lin Gui, Bart Nabbe, Iain Matthews, Takeo Kanade, Shohei Nobuhara, and Yaser Sheikh. Panoptic studio: A massively multi-view system for social motion capture. In *International Conference on Computer Vision (ICCV)*, pages 3334–3342, 2015.
23. Hanbyul Joo, Natalia Neverova, and Andrea Vedaldi. Exemplar fine-tuning for 3D human pose fitting towards in-the-wild 3D human pose estimation. In *International Conference on 3D Vision (3DV)*, pages 42–52, 2021.
24. Hanbyul Joo, Tomas Simon, and Yaser Sheikh. Total capture: A 3D deformation model for tracking faces, hands, and bodies. In *Computer Vision and Pattern Recognition (CVPR)*, pages 8320–8329, 2018.
25. Angjoo Kanazawa, Michael J. Black, David W. Jacobs, and Jitendra Malik. End-to-end recovery of human shape and pose. In *Computer Vision and Pattern Recognition (CVPR)*, pages 7122–7131, 2018.
26. Muhammed Kocabas, Chun-Hao P. Huang, Otmar Hilliges, and Michael J. Black. PARE: Part attention regressor for 3D human body estimation. In *International Conference on Computer Vision (ICCV)*, pages 11127–11137, 2021.
27. Nikos Kolotouros, Georgios Pavlakos, Michael J. Black, and Kostas Daniilidis. Learning to reconstruct 3D human pose and shape via model-fitting in the loop. In *International Conference on Computer Vision (ICCV)*, pages 2252–2261, 2019.
28. Agelos Kratimenos, Georgios Pavlakos, and Petros Maragos. Independent sign language recognition with 3D body, hands, and face reconstruction. In *International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pages 4270–4274, 2021.
29. Shyam Krishna, Vignesh P. Vijay, and Babu J. Dinesh. SignPose: Sign language animation through 3D pose lifting. In *International Conference on Computer Vision (ICCV)*, pages 2640–2649, 2021.
30. Dominik Kulon, Riza Alp Guler, Iasonas Kokkinos, Michael M. Bronstein, and Stefanos Zafeiriou. Weakly-supervised mesh-convolutional hand reconstruction in the wild. In *Computer Vision and Pattern Recognition (CVPR)*, June 2020.
31. Joanna Łacheta, Małgorzata Czajkowska-Kisil, Jadwiga Linde-Usiekniewicz, and Paweł Rutkowski. *Korpusowy słownik polskiego języka migowego / Corpus-based Dictionary of Polish Sign Language*, 2016.
32. Jiefeng Li, Chao Xu, Zhicun Chen, Siyuan Bian, Lixin Yang, and Cewu Lu. HybrIK: A hybrid analytical-neural inverse kinematics solution for 3D human pose and shape estimation. In *Computer Vision and Pattern Recognition (CVPR)*, pages 3383–3393, 2021.
33. Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, and Michael J. Black. AMASS: Archive of motion capture as surface shapes. In *International Conference on Computer Vision (ICCV)*, pages 5441–5450, 2019.
34. Ian Marshall and Éva Sáfár. A prototype text to British Sign Language (BSL) translation system. In *Meeting of the Association for Computational Linguistics (ACL)*, pages 113–116, 2003.
35. Meshcapade GmbH, Tübingen, Germany. `https://meshcapade.com`, 2022.
36. Gyeongsik Moon and Kyoung Mu Lee. I2L-MeshNet: Image-to-lixel prediction network for accurate 3D human pose and mesh estimation from a single RGB image. In *European Conference on Computer Vision (ECCV)*, pages 752–768, 2020.
37. Gyeongsik Moon, Shoou-I Yu, He Wen, Takaaki Shiratori, and Kyoung Mu Lee. InterHand2.6M: A dataset and baseline for 3D interacting hand pose estimation from a single RGB image. In *European Conference on Computer Vision (ECCV)*, pages 548–564, 2020.
38. Amit Moryossef, Ioannis Tsochantaridis, Joe Dinn, Necati Cihan Camgoz, Richard Bowden, Tao Jiang, Annette Rios, Mathias Müller, and Sarah Ebling. Evaluating the immediate applicability of pose estimation for sign language recognition. In *Computer Vision and Pattern Recognition (CVPR)*, pages 3434–3440, 2021.
39. Lea Müller, Ahmed A. A. Osman, Siyu Tang, Chun-Hao P. Huang, and Michael J. Black. On self-contact and human pose. In *Computer Vision and Pattern Recognition (CVPR)*, pages 9990–9999, 2021.
40. Lucie Naert, Caroline Larboulette, and Sylvie Gibet. A survey on the animation of signing avatars: From sign representation to utterance synthesis. *Computers & Graphics (CG)*, 92:76–98, 2020.
41. Jorge Nocedal and Stephen J. Wright. *Numerical Optimization*. Springer, New York, NY, USA, second edition, 2006.
42. Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed A. A. Osman, Dimitrios Tzionas, and Michael J. Black. Expressive body capture: 3D hands, face, and body from a single image. In *Computer Vision and Pattern Recognition (CVPR)*, pages 10975–10985, 2019.
43. Real SASL: Real South African Sign Language. `https://www.realsasl.com/`.
44. Katrin Renz, Nicolaj C. Stache, Neil Fox, Gül Varol, and Samuel Albanie. Sign segmentation with changepoint-modulated pseudo-labelling. In *Computer Vision and Pattern Recognition Workshops (CVPRw)*, 2021.
45. Yu Rong, Takaaki Shiratori, and Hanbyul Joo. FrankMocap: A monocular 3D whole-body pose estimation system via regression and integration. In *Computer Vision and Pattern Recognition Workshops (ICCVW)*, pages 1749–1759, 2021.
46. Ken Shoemake. Animating rotation with quaternion curves. In *International Conference on Computer Graphics and Interactive Techniques (SIGGRAPH)*, pages 245–254, 1985.
47. Breannan Smith, Chenglei Wu, He Wen, Patrick Peluse, Yaser Sheikh, Jessica K. Hodgins, and Takaaki Shiratori. Constraining dense hand surface tracking with elasticity. *ACM Transactions on Graphics*, 39(6):1–14, 2020.
48. Richard A. Tennant, Marianne Gluszak, and Marianne Gluszak Brown. *The American Sign Language Handshape Dictionary*. Gallaudet University Press, 2010.
49. Stavros Theodorakis, Vassilis Pitsikalis, and Petros Maragos. Dynamic-static unsupervised sequentiality, statistical subunits and lexicon for sign language recognition. *Image and Vision Computing (IVS)*, 32(8):533–549, 2014.
50. Shashank Tripathi, Lea Müller, Chun-Hao P. Huang, Taheri Omid, Michael J. Black, and Dimitrios Tzionas. 3D human pose estimation via intuitive physics. In *Computer Vision and Pattern Recognition (CVPR)*, 2023.
51. Manuel Vázquez-Enríquez, Jose L. Alba-Castro, Laura Docío-Fernández, and Eduardo Rodríguez-Banga. Isolated sign language recognition with multi-scale spatial-temporal graph convolutional networks. In *Computer Vision and Pattern Recognition (CVPR)*, pages 3462–3471, 2021.
52. Jinbao Wang, Shujie Tan, Xiantong Zhen, Shuo Xu, Feng Zheng, Zhenyu He, and Ling Shao. Deep 3D human pose estimation: A review. *Computer Vision and Image Understanding*, 210:103225, 2021.
53. World Federation of the Deaf. Who we are. `http://wfdeaf.org/who-we-are/`.
54. Donglai Xiang, Hanbyul Joo, and Yaser Sheikh. Monocular total capture: Posing face, body, and hands in the wild. In *Computer Vision and Pattern Recognition (CVPR)*, pages 10957–10966, 2019.
55. Hongyi Xu, Eduard Gabriel Bazavan, Andrei Zanfir, William T. Freeman, Rahul Sukthankar, and Cristian Sminchisescu. GHUM & GHUML: Generative 3D human shape and articulated pose models. In *Computer Vision and Pattern Recognition (CVPR)*, pages 6183–6192, 2020.
56. Andrei Zanfir, Eduard Gabriel Bazavan, Mihai Zanfir, William T. Freeman, Rahul Sukthankar, and Cristian Sminchisescu. Neural descent for visual 3D human pose and shape. In *Computer Vision and Pattern Recognition (CVPR)*, pages 14484–14493, 2021.
57. Hongwen Zhang, Yating Tian, Yuxiang Zhang, Mengcheng Li, Liang An, Zhenan Sun, and Yebin Liu. PyMAF-X: Towards well-aligned full-body model regression from monocular images. *arXiv preprint arXiv:2207.06400*, 2022.
58. Hongwen Zhang, Yating Tian, Xinchi Zhou, Wanli Ouyang, Yebin Liu, Limin Wang, and Zhenan Sun. PyMAF: 3D human pose and shape regression with pyramidal mesh alignment feedback loop. In *International Conference on Computer Vision (ICCV)*, pages 11426–11436, 2021.
59. Christian Zimmermann, Duygu Ceylan, Jimei Yang, Bryan Russell, Max Argus, and Thomas Brox. FreiHAND: A dataset for markerless capture of hand pose and shape from single RGB images. In *International Conference on Computer Vision (ICCV)*, pages 813–822, 2019.

---

## Appendices

### A. Examples of Sign Classes
Table A.1 provides representative images of our eight sign classes to supplement Tab. 1. The videos of these signs appear in the supplemental video.

### B. SGNify Objective
The full objective function of SGNify is:
$$E(	heta, \psi, eta) = \lambda_{	heta_b} E_{	heta_b} + \lambda_{m_h} E_{m_h} + E_J + \lambda_lpha E_lpha + E_O + \lambda_P E_P + \lambda_A E_A + L_s + \sum_{h \in \{r, l\}} L_i^h + \lambda_t L_t + \lambda_{st} L_{st}$$

where $	heta$ is the full set of optimizable pose parameters, and $	heta_b$ and $m_h$ are the pose vectors for the body and the two hands. The body pose is modeled by a VAE (called Vposer) that transforms the body pose $	heta_b$ into a latent vector $Z$. We enforce an L2 prior in this space, *i.e.*, $E_{	heta_b}(	heta_b) = \|Z\|^2$. For the hands, SMPL-X uses a low-dimensional PCA pose space such that $	heta_h = \sum_{n=1}^{|m_h|} m_{h_n} \mathcal{M}_n$ where $\mathcal{M}$ are principal components capturing the finger pose variations and $m_{h_n}$ are the corresponding PCA coefficients. Thus, $E_{m_h}(m_h)$ is an L2 prior on the coefficients $m_h$. 

$E_J$ represents the joint re-projection loss, and $E_lpha(	heta_b)$ is a prior penalizing extreme bending only for elbows and knees. For more details on these terms, please refer to the original paper of SMPLify-X [42]. $E_O$ is a bone-orientation term, which factors out the residual of the parent joint from the residual of the child joint [21]. $E_P$ and $E_A$ are used to prevent self-interpenetration [39]. 

We added $L_s$ and $L_i^h$ to enforce our linguistic constraints as described in Sec. 3.2. We also added a temporal loss $L_t$ on the body and hand-pose vectors and a standing loss $L_{st}$ to penalize deviations from a standing pose when none of the feet keypoints are detected. We optimize our objective function using the trust-region Newton conjugate gradient method [41]. We do not optimize for the shape $eta$ and the facial expressions $\psi$.

### C. Intervals for Selecting the Candidate Frames for the Reference Hand Poses
When articulating an isolated sign, signers start and end in a rest pose. SGNify identifies the beginning and end of the sequence based on when the hands begin to move. We assume the core part of a sign to happen between $0.5 	imes T/8 < t < 7 	imes T/8$, where $T$ is the number of frames. To identify the two key poses representing the initial and final hand poses $	heta_{	ext{ref},i}^h$ and $	heta_{	ext{ref},f}^h$, we consider two different intervals (Interval 1 and Interval 2) within this core window.

### D. HamNoSys Parsing
We construct an Extended Backus-Naur form (EBNF) grammar to parse HamNoSys [19] annotations:
* **Class 0a:** There is one `handshape_block` nonterminal and no `SYMMETRY` terminal is present.
* **Class 0b:** There are two `handshape_block` nonterminals, they are not equal, a `HAMREPLACE` terminal is present, and no `SYMMETRY` or `REPEAT` terminals are present.
* **Class 1a:** There is one `handshape_block` nonterminal and a `SYMMETRY` terminal is present.
* **Class 1b:** There are two `handshape_block` nonterminals, they are not equal, a `HAMREPLACE` terminal is present, and a `SYMMETRY` terminal is present.
* **Class 2a:** There are two `handshape_block` nonterminals, they are equal, they fall within a `par` nonterminal, and no `SYMMETRY` terminal is present.
* **Class 2b:** There are three `handshape_block` nonterminals, the first two are equal, a `HAMREPLACE` terminal is present, and no `SYMMETRY` or `REPEAT` terminals are present.
* **Class 3a:** There are two `handshape_block` nonterminals, they are not equal, they fall within a `par` nonterminal, and no `SYMMETRY` terminal is present.
* **Class 3b:** There are three `handshape_block` nonterminals, the first is not equal to the second, a `HAMREPLACE` terminal is present, and no `SYMMETRY` or `REPEAT` terminals are present.

### E. SGNify Extensions

#### E.1. Multi-view
SGNify can easily be extended if multi-view video is available. We used 12 synchronized RGB cameras at 90 fps to capture signers performing our German Sign Language (DGS) corpus. We follow Huang *et al.* [21] to combine the keypoint predictions of different cameras.

#### E.2. Continuous Sign Language Capture (CSLC)
SGNify can also be used for CSLC. We conduct an exploratory quantitative study with twelve sentences (ten main sentences and two variations) collected and analyzed as in Sec. 5.1.

| Method | Upper Body | Left Hand | Right Hand |
| :--- | :---: | :---: | :---: |
| **FrankMocap** [45] | 74.93 | 23.70 | 19.57 |
| **PIXIE** [15] | 59.09 | 24.79 | 20.19 |
| **PyMAF-X** [57] | 68.30 | 22.51 | 18.49 |
| **SMPLify-SL** | 55.71 | 21.14 | 18.60 |
| **SGNify** | **54.72** | **20.28** | **17.44** |

**Table E.1.** Mean TR-V2V error (mm) on fluid sentences.
