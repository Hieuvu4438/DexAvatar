# Prompt vẽ figure phương pháp SignEFT-X

## Master prompt (khuyến nghị dùng bằng tiếng Anh)

```text
Act as a senior scientific illustrator and computer-vision paper figure designer. Create a publication-ready METHOD OVERVIEW figure for a top-tier CV/ML conference (CVPR/ICCV/ECCV/NeurIPS visual standard). The figure explains “Signer-Consistent Palm-Canonical Refinement for Monocular 3D Sign Language Reconstruction.” It must communicate one central idea within five seconds:

“Preserve the whole-body model’s global signing state; import only the hand expert’s relative finger geometry.”

OUTPUT AND STYLE
- Full-width, two-column landscape figure, approximately 175 mm wide, aspect ratio around 2.25:1, white background, clean flat vector graphics suitable for SVG/PDF recreation.
- Crisp thin strokes, restrained typography, generous whitespace, no decorative background, no gradients, no shadows, no photorealism, no 3D glossy effects.
- Use a professional scientific-diagram aesthetic: precise geometry, consistent arrows, aligned modules, balanced margins, and readable at final paper size.
- Use a colorblind-safe semantic palette consistently:
  * protected whole-body/global state: graphite #34495E;
  * shared signer-consistent SMPL-X identity: purple #6C4BC3;
  * frozen WiLoR expert observation: orange #E76F00;
  * optimized finger articulation/final refinement: teal #008A7C;
  * palm-canonical coordinate frame: blue #2D8CFF;
  * inactive/fallback paths: medium gray #8A8F98.
- All normal text should be dark charcoal. Colors encode meaning, not decoration.
- Use solid black or dark-gray arrows for computation/data flow, dashed gray arrows for immutable conditioning or fallback, padlock icons for frozen variables, and a red prohibition icon only for explicitly forbidden changes.
- Keep all labels short. Use one sans-serif typeface similar to Arial/Helvetica. At final 175 mm width, no text may appear smaller than 8.5 pt. Panel letters and stage titles must remain readable after down-scaling.

COMPOSITION AND VISUAL HIERARCHY
Organize the figure as a left-to-right pipeline with four numbered stages, but make Stages 2 and 3 the largest visual “hero” region because they contain the main technical contribution. Do not split the canvas into five equally dense boxes. Use subtle grouping boundaries, not heavy panel frames.

At the very top, place a short visual thesis spanning the central pipeline:
“Global signing state is preserved”  +  “Relative handshape is refined”
Show “preserved” with a padlock/graphite accent and “refined” with a teal accent.

STAGE 1 — SIGNER-CONSISTENT INITIALIZATION (left, about 22% width)
- Show a compact monocular RGB signing clip as three overlapping frames of the same signer.
- From the clip, create two frozen branches:
  1) a whole-body initializer producing an SMPL-X signer and the global signing state;
  2) a frozen WiLoR hand expert producing a 21-joint hand skeleton.
- In the whole-body branch, show several translucent per-frame SMPL-X bodies with slightly inconsistent shapes, followed by “Pose-diverse sampling, K = 200” and “Robust shared shape β*”. Collapse them into one purple signer identity reused across all frames.
- Add a small concise label under the result: “Neutral SMPL-X · 10,475 vertices · one β*”.
- The whole-body state should visibly anchor body pose, arm trajectory, wrist position/orientation, face, translation, and camera. Do not list all variables in a large paragraph; summarize with the label “Global signing state” and a padlock icon.

STAGE 2 — PALM-CANONICAL FACTORIZATION (center-left, about 33% width; largest and most visually explicit)
- Use two parallel horizontal lanes that converge into one shared canonical space.
- Top lane: “Canonical SMPL-X hand J(δ)” in purple/graphite.
- Bottom lane: “Expert hand Jᴱ” in orange.
- In each lane, depict the same three transformations as small, visually matched steps:
  1) root-center at wrist J₀ — translation removed;
  2) align a proper palm frame Q = [x y z], det(Q) = +1 — palm rotation removed;
  3) normalize by palm scale s = ||J₉ − J₀||₂ — absolute hand scale removed.
- Draw the x, y, z palm axes directly on both 21-joint skeletons. Use blue for all axes and canonical-space cues.
- Both lanes must end inside one light-blue “Shared palm-canonical space” box, where the purple and orange hand skeletons have the same wrist origin, scale, and palm orientation but retain visibly different finger configurations.
- Place one compact equation beneath this stage, typeset accurately:
  C(J) = ((J − J₀) Q) / ||J₉ − J₀||₂
- Under the equation, add four compact semantic tokens with icons:
  crossed out “translation”; crossed out “scale”; crossed out “palm rotation”; green check “relative finger geometry”.
- Make it visually unmistakable that canonicalization is performed independently on both sources before comparison. Never show direct copying of WiLoR rotations into SMPL-X.

STAGE 3 — BOUNDED FINGER-ONLY RETARGETING (center-right, about 27% width; second hero region)
- At the top, show the purple canonical SMPL-X hand and orange expert target side by side in the shared palm-canonical frame, connected by a bidirectional geometry-matching cue labeled “SmoothL1 canonical fit”.
- Below, show a compact optimizer module titled “Finger-only optimization”. Inside it, visually highlight exactly 15 local finger joints in teal while the wrist remains graphite with a padlock.
- Around one finger-joint update, draw a small spherical/geodesic trust-region glyph labeled “ρ = 12°”. Avoid an Euler-angle box or unconstrained arrow.
- Include the compact manifold update equation, typeset accurately:
  Rₖ(δₖ) = Exp(clipρ(δₖ)) Rₖ⁰
- Include only these implementation labels: “40 Adam steps”, “15 local finger joints”, and “λδ = 0.2”.
- Directly beneath the optimizer, place a dashed protected-state strip with padlocks and the concise text:
  “Frozen: β*, body, arms, wrists, face, translation, camera”
- Use a red prohibition cue at the wrist and global body to show that the optimizer cannot modify palm orientation or signing-space location.

STAGE 4 — UNIFIED OUTPUT (right, about 18% width)
- Show one clean neutral-topology SMPL-X signer in purple/graphite with the refined fingers highlighted in teal.
- Add two circular zoom-in insets:
  1) a detailed teal/purple hand labeled “Handshape refined”;
  2) a palm coordinate triad and wrist lock labeled “Palm orientation + location preserved”.
- Add a small output badge: “One valid SMPL-X forward pass”. Make clear that this is not a pasted MANO hand and not a hybrid mesh.
- At the bottom, add a subtle dashed fallback branch:
  “No expert proposal → retain canonical hand exactly”.

SCIENTIFIC ACCURACY CONSTRAINTS
- The method is test-time optimization; do not draw a training loop, dataset supervision, neural-network fine-tuning, Transformer, temporal network, ground-truth mesh, evaluator mask, semantic sign label, or loss to ground truth.
- WiLoR is frozen and supplies only 3D hand geometry. It does not replace the SMPL-X wrist, arm, body, shape, camera, or translation.
- Only the 15 local finger rotations for an available hand side may change during the final refinement.
- Palm orientation and signing-space location must visibly remain inherited from the whole-body reconstruction.
- The final output must remain a single neutral SMPL-X mesh with shared signer shape β* and consistent 10,475-vertex topology.
- Do not include quantitative benchmark results in this method figure.

GRAPHIC QUALITY CHECK BEFORE FINALIZING
- Check the left-to-right flow and ensure no arrow suggests direct rotation transfer from WiLoR to SMPL-X.
- Check that purple means signer-consistent SMPL-X, orange means expert-only observation, blue means canonical coordinates, teal means optimized finger articulation, and graphite means protected state everywhere.
- Check that all equations and symbols are spelled exactly and all padlocks are attached to frozen quantities.
- Check that the central palm-canonical factorization is visually dominant and understandable without reading the caption.
- Check that labels remain legible at two-column publication width and that the complete figure contains no redundant legend or repeated prose.
```

## Negative prompt (nếu công cụ hỗ trợ)

```text
Avoid: photorealistic people, stock-photo look, glossy 3D rendering, gradients, shadows, dark background, decorative icons, comic style, neon colors, rainbow palette, red-green-only encoding, dense prose, tiny text, overlapping labels, five equally narrow panels, excessive equations, result charts, training loops, Transformer blocks, ground-truth supervision, direct WiLoR rotation copying, unlocked wrist, modified body pose, pasted MANO mesh, inconsistent hand joint counts, malformed hands, wrong anatomy, improper or mirrored coordinate frame, ambiguous arrows, and any claim that global signing state is optimized during finger refinement.
```

## Text overlay bắt buộc

Nếu mô hình sinh ảnh làm sai chữ hoặc công thức, hãy yêu cầu nó chỉ tạo bố cục và thay các nhãn sau bằng text vector thủ công trong Figma/Illustrator/Inkscape:

1. `Signer-consistent initialization`
2. `Pose-diverse sampling, K = 200`
3. `Robust shared shape β*`
4. `Neutral SMPL-X · 10,475 vertices · one β*`
5. `Palm-canonical factorization`
6. `Canonical SMPL-X hand J(δ)`
7. `Expert hand Jᴱ`
8. `Shared palm-canonical space`
9. `C(J) = ((J − J₀) Q) / ||J₉ − J₀||₂`
10. `Finger-only optimization`
11. `SmoothL1 canonical fit`
12. `15 local finger joints`
13. `ρ = 12°`
14. `Rₖ(δₖ) = Exp(clipρ(δₖ)) Rₖ⁰`
15. `Frozen: β*, body, arms, wrists, face, translation, camera`
16. `Handshape refined`
17. `Palm orientation + location preserved`
18. `One valid SMPL-X forward pass`
19. `No expert proposal → retain canonical hand exactly`

## Caption đề xuất

**Overview of SignEFT-X.** A frozen whole-body initializer provides the global signing state, while pose-diverse frames determine one signer-consistent SMPL-X shape, \(\boldsymbol\beta^*\), shared by the full sequence. The canonical SMPL-X hand and frozen WiLoR proposal are independently root-centered, aligned to a proper palm frame, and scale-normalized, exposing only their relative finger geometry. SignEFT-X then optimizes the 15 local finger rotations within a \(12^\circ\) geodesic trust region while freezing morphology, body and arm pose, wrists, face, translation, and camera. The result is a single neutral-topology SMPL-X avatar with refined handshape and preserved palm orientation and signing-space location; if no expert proposal is available, the canonical hand is retained exactly.

