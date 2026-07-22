# DPoser-X Pipeline: Fitting Quality Analysis & Improvement Proposals

**Date:** 2025-06-18  
**Issue:** Rendered mesh is offset from the person (0 overlapping pixels between mesh and skin). Fitting quality is very poor.

---

## 1. Root Cause Analysis

### Problem: Mesh rendered at wrong position — 0% overlap with person

```
Image: 1341×804
Person skin bbox:  x=[350, 848], y=[283, 659]
Mesh bbox:         x=[506, 986], y=[563, 777]
Mesh-person overlap: 0 pixels
Mesh center offset:  dx=+170, dy=+301  (far down-right from person)
```

### Root Cause Chain

```
NLF init provides global_orient + transl
        │
        ▼
┌─────────────────────────────────────────────┐
│ BOTH global_orient AND transl are FROZEN    │
│ (lines 653-659 in fit_single_frame.py)      │
│ "deliberately KEPT OUT of final_params"      │
└─────────────────────────────────────────────┘
        │
        ▼
Camera projection is locked to NLF init values
        │
        ▼
If NLF init has ANY error in global_orient or transl,
the mesh projection is permanently misaligned
        │
        ▼
MESH DOES NOT OVERLAP WITH PERSON IN IMAGE
```

### What IS Being Optimized vs. What is Frozen

| Parameter | Dimension | Optimized? | Notes |
|---|---|---|---|
| `global_orient` | 3 | ❌ Frozen | NLF init = 2.71 rad (155° rotation) |
| `transl` | 3 | ❌ Frozen | NLF init ≈ [0, 0.75, 17.7] |
| `body_pose` | 63 | ❌ Frozen during L-BFGS | Only DPoser-X denoised post-fit |
| `lhand_embedding3d` | 23 | ✅ Optimized | SignHPoser latent |
| `rhand_embedding3d` | 23 | ✅ Optimized | SignHPoser latent |
| `betas` | 10 | ✅ Optimized | Body shape (from body_model params) |
| `expression` | 10 | ✅ Optimized | Face expression |

### Why the Camera is Frozen

The code comment at `fit_single_frame.py:653-659` explains:

> "transl and global_orient are deliberately KEPT OUT of final_params. They are frozen at their NLF init values. The NLF adapter now produces calibrated transl (~14-20m) that matches the VAE prior's training distribution."

This was a reasonable decision when the body_pose was ALSO frozen (to prevent L-BFGS from walking to spurious local minima). However, it means the mesh projection quality depends entirely on the NLF init accuracy, which varies across frames and signers.

---

## 2. Image Evidence

The rendered image (`low_149.png`) shows:
- A person performing a sign language gesture (skin visible, facing camera)
- A blue mesh wireframe rendered significantly **below and to the right** of the person
- **Zero overlapping pixels** between mesh and person

This is the classic symptom of incorrect camera extrinsics (global_orient + transl) combined with frozen body pose.

---

## 3. Improvement Proposals

### Priority 1: Optimize Camera Parameters (transl + global_orient)

**Current:** `transl` and `global_orient` are never optimized.  
**Fix:** Add a camera-only optimization stage BEFORE the main body/hand fitting.

```python
# Stage 0: Optimize camera (global_orient + transl) with 2D reprojection loss
# Keep body_pose, hand_pose, betas frozen during this stage
# Use Adam with small learning rate (~1e-3)
# Run for ~50-100 iterations
```

The `camera_loss` function already exists at `fit_single_frame.py:472` but is never called. Wire it into an optimization loop before the main fitting.

**Expected improvement:** Mesh alignment with person. The camera projection will adapt to the actual 2D keypoints.

**Risk:** Low. Camera-only optimization is standard in SMPLify-X. The reprojection loss constrains the camera well when body pose is reasonable (NLF init is reasonable).

### Priority 2: Optimize Body Pose with Light Regularization

**Current:** Body pose is completely frozen during L-BFGS. Only DPoser-X post-fit denoising is applied (10 steps, ~0.36 rad mean change).  
**Fix:** Add body pose to `final_params` with a light L2 regularizer toward NLF init:

```python
body_reg_weight = 50.0  # light regularization
pprior_loss = body_reg_weight * torch.sum((body_pose_direct - body_init) ** 2)
```

This allows the body pose to adapt to 2D keypoints while staying close to the NLF init (which is already high quality). The DPoser-X denoising is then applied as a final refinement step.

**Expected improvement:** Better body keypoint alignment. Hands and arms will fit the 2D keypoints better.

**Risk:** Medium. Without a strong pose prior, the body can take implausible poses. But the L2 anchor to NLF init prevents extreme drift.

### Priority 3: Multi-Orientation Trial

**Current:** Only ONE orientation is tried (from NLF init, 2.71 rad). The original SMPLify-X tries 4 orientations (0°, 90°, 180°, 270°) and picks the best.  
**Fix:** Add multiple orientation trials:

```python
orientations = [
    init_smplx_param['global_orient'][None],
    init_smplx_param['global_orient'][None] + np.array([0, np.pi/2, 0]),
    init_smplx_param['global_orient'][None] + np.array([0, np.pi, 0]),
    init_smplx_param['global_orient'][None] + np.array([0, 3*np.pi/2, 0]),
]
```

For sign language (upper body focused), the orientation around the Y-axis has the biggest impact. Trying ±90° variants can catch cases where NLF mis-predicts the facing direction.

**Expected improvement:** Better orientation in cases where NLF init is significantly wrong.

**Risk:** Low. This adds ~4× computation but each orientation trial is fast (~1 second for camera-only optimization).

### Priority 4: Stronger DPoser-X Denoising

**Current:** 10 denoising steps, ~0.36 rad mean change.  
**Fix:** Increase to 50-100 steps, or run multiple rounds of denoising:

```python
for _ in range(3):  # iterative refinement
    bp_tensor = dposerx_refine_prior.decode_to_pose(bp_tensor, num_steps=50)
```

More denoising steps allow the diffusion model to make larger, more meaningful changes to the body pose while maintaining plausibility.

**Expected improvement:** Body pose becomes more natural (closer to AMASS distribution).

**Risk:** Low. Pure inference, no gradient issues. Slightly slower (~1 sec → ~5 sec per frame).

### Priority 5: Per-Stage Body Pose Optimization with Decreasing Regularization

**Current:** Body pose is frozen across all 4 L-BFGS stages.  
**Fix:** Start with strong regularization toward NLF init, then decrease:

| Stage | body_reg_weight | Description |
|---|---|---|
| 1 | 500 | Strong anchor to NLF init |
| 2 | 200 | Moderate freedom |
| 3 | 50 | More freedom to fit keypoints |
| 4 | 10 | Mostly free, DPoser-X denoised post-fit |

This coarse-to-fine schedule is standard in SMPLify-X and allows the body pose to gradually adapt to 2D evidence.

**Expected improvement:** Progressive refinement of body pose, avoiding local minima.

**Risk:** Medium. Need to ensure weights don't cause NaN (use body_pose clipping as safety net).

---

## 4. Implementation Priority

| Priority | Change | Effort | Impact |
|---|---|---|---|
| **P0** | Add camera optimization stage | ~20 lines | Fixes mesh-person misalignment |
| **P1** | Optimize body_pose with L2 anchor | ~10 lines | Improves body keypoint fit |
| **P2** | Multi-orientation trial | ~15 lines | Handles bad NLF orientations |
| **P3** | Stronger DPoser-X denoising | ~3 lines | More natural body poses |
| **P4** | Per-stage body regularization schedule | ~10 lines | Coarse-to-fine refinement |

---

## 5. Recommended Quick Fix (P0 only — highest impact for least effort)

In `fit_single_frame.py`, add a camera-only optimization stage before the main fitting loop:

```python
# ---- NEW: Camera-only optimization ----
# Optimize global_orient + transl to align mesh projection with 2D keypoints.
body_model.reset_params(**new_params)
cam_params = []
for name in ['global_orient', 'transl']:
    param = getattr(body_model, name)
    if param is not None:
        param.requires_grad_(True)
        cam_params.append(param)

cam_optimizer = torch.optim.Adam(cam_params, lr=1e-2)
for cam_iter in range(50):
    cam_optimizer.zero_grad()
    output = body_model(return_verts=False)
    proj = camera(output.joints)
    loss = ((proj[:, src2inter] - gt_joints[:, dst2inter]) ** 2).sum()
    loss.backward()
    cam_optimizer.step()
    if cam_iter > 0 and abs(prev_loss - loss.item()) < 1e-6:
        break
    prev_loss = loss.item()

# Freeze camera params again
for p in cam_params:
    p.requires_grad_(False)
```

This alone should fix the mesh-person misalignment.
