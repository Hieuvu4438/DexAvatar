# Thay thế VAE Prior (SignBPoser & SignHPoser) cho 3D Sign Language Reconstruction

## Ngày: 2026-06-05
## Mục tiêu: Tìm phương pháp thay thế VAE prior để cải thiện TR-V2V metric và tạo contribution mới cho paper

---

## 1. Phân tích VAE Prior hiện tại

### 1.1 SignBPoser — Body Pose Prior

**File:** `dexavatar_fitting/smplifyx/signbposer/`

**Cách hoạt động:**
- VAE encoder: body_pose (63-dim) → latent (33-dim)
- VAE decoder: latent (33-dim) → body_pose (63-dim)
- Training data: sign language body poses
- Optimization: optimize latent code, decoder tạo ra body pose

**Vai trò trong pipeline:**
```python
# fit_single_frame.py
pose_embedding = torch.zeros([1, 33], requires_grad=True)  # latent code
body_pose = signbposer.decode(pose_embedding, output_type='aa')  # decode → SMPL-X body pose
# Loss = data_loss + λ * ||pose_embedding||²  (L2 regularization on latent)
```

**Bottleneck:**
- 33-dim latent space → không thể express tất cả body poses
- Decoder bị giới hạn bởi training distribution
- L2 trên latent ≠ L2 trên pose space (nonlinear mapping)

### 1.2 SignHPoser — Hand Pose Prior

**File:** `dexavatar_fitting/smplifyx/signhposer/`

**Cách hoạt động:**
- VAE encoder: hand_pose (45-dim) → latent (23-dim)
- VAE decoder: latent (23-dim) → hand_pose (45-dim)
- Training data: sign language hand poses
- Optimization: optimize latent code cho mỗi tay

**Vai trò trong pipeline:**
```python
# fit_single_frame.py
lhand_embedding3d = torch.zeros([1, 23], requires_grad=True)
rhand_embedding3d = torch.zeros([1, 23], requires_grad=True)
lhand_pose = hposer3d.decode(lhand_embedding3d, output_type='aa')
rhand_pose = hposer3d.decode(rhand_embedding3d, output_type='aa')
# Loss = data_loss + λ₁ * ||embedding||² + λ₂ * ||decoded - init||²
```

**Bottleneck:**
- 23-dim latent cho mỗi tay → 46-dim tổng cho hands
- Hand có 15 joints × 3 DoF = 45 dims → latent compression quá mạnh
- Sign language hand shapes rất đa dạng → VAE mode-seeking bỏ qua nhiều modes

---

## 2. Phương pháp thay thế SignBPoser (Body Prior)

### 2.1 GMM Prior (Gaussian Mixture Model)

**Tại sao sử dụng:**
- Đơn giản, không cần train neural network
- Closed-form log-likelihood → gradient ổn định
- Capture multimodal distribution tốt hơn single Gaussian
- Đã được chứng minh hiệu quả trong SMPLify gốc

**Cách hoạt động:**
```
Training:
  1. Thu thập body poses từ sign language data (SMPLer-X output)
  2. Fit GMM với K components (K=8-16)
  3. Lưu means, covariances, weights

Optimization:
  - Log-likelihood loss: L_gmm = -log p(body_pose | GMM)
  - Gradient: ∇L_gmm = -∇log p(body_pose) → guide optimization
  - Kết hợp với data loss: L = L_data + λ * L_gmm
```

**Implementation sketch:**
```python
class GMMPrior:
    def __init__(self, means, covariances, weights):
        self.means = means        # (K, 63)
        self.covariances = covariances  # (K, 63, 63)
        self.weights = weights    # (K,)
    
    def log_likelihood(self, pose):
        # pose: (1, 63)
        diffs = pose - self.means  # (K, 63)
        mahalanobis = torch.sum(diffs @ inv_cov * diffs, dim=-1)
        log_probs = -0.5 * mahalanobis + torch.log(self.weights)
        return torch.logsumexp(log_probs, dim=0)
    
    def loss(self, pose):
        return -self.log_likelihood(pose)
```

**Ưu điểm:**
- Nhanh, ổn định, không cần GPU
- Gradient chính xác (không qua nonlinear decoder)
- Có thể interpret được ( mỗi component = 1 kiểu pose)

**Nhược điểm:**
- Linear trong pose space → không capture được nonlinear manifold
- Cần đủ data để fit GMM chính xác

**Expected improvement:** 3-5% so với VAE

---

### 2.2 Normalizing Flow Prior

**Tại sao sử dụng:**
- Expressiveness hơn GMM (nonlinear mapping)
- Exact log-likelihood computation (không approximate như VAE)
- Invertible mapping → gradient không bị vanish
- Đã được chứng minh trong các motion prior papers

**Cách hoạt động:**
```
Training:
  1. Learn invertible transformation: f: pose → latent
  2. Latent space: standard Gaussian
  3. Loss: negative log-likelihood via change of variables

Optimization:
  - L_flow = -log p_flow(body_pose)
  - Gradient: ∇L_flow qua invertible transformation
  - Stable gradient vì transformation invertible
```

**Implementation sketch:**
```python
class FlowPrior(nn.Module):
    def __init__(self, n_flows=8, dim=63):
        self.flows = nn.ModuleList([CouplingLayer(dim) for _ in range(n_flows)])
    
    def log_prob(self, x):
        log_det = 0
        z = x
        for flow in self.flows:
            z, ld = flow(z)
            log_det += ld
        # Standard Gaussian log prob
        log_pz = -0.5 * (z**2 + np.log(2*np.pi)).sum(-1)
        return log_pz + log_det
    
    def loss(self, pose):
        return -self.log_prob(pose)
```

**Ưu điểm:**
- Nonlinear → capture complex distributions
- Exact likelihood → optimization chính xác hơn
- Invertible → gradient stable

**Nhược điểm:**
- Cần train flow model
- Chậm hơn GMM
- Architecture design phức tạp hơn

**Expected improvement:** 5-8% so với VAE

---

### 2.3 Diffusion Prior (Score-Based)

**Tại sao sử dụng:**
- State-of-the-art trong generative modeling
- Capture distribution phức tạp nhất trong các phương pháp
- Có thể conditioning trên context (sign class, temporal position)
- Gradient guidance linh hoạt

**Cách hoạt động:**
```
Training:
  1. Forward process: add noise to body poses theo schedule
  2. Learn score function: ∇_x log p(x) tại mỗi noise level
  3. Training: denoise body poses

Optimization (Test time):
  1. Start từ SMPLer-X init
  2. Score function guide: x_new = x + η * score(x, t) + data_gradient
  3. Iterative refinement với decreasing noise level
```

**Implementation sketch:**
```python
class DiffusionPrior:
    def __init__(self, score_model, n_steps=1000):
        self.model = score_model  # U-Net or Transformer
        self.n_steps = n_steps
        self.betas = cosine_schedule(n_steps)
    
    def score(self, x, t):
        # Returns ∇_x log p(x|t)
        return self.model(x, t)
    
    def loss(self, pose, t=None):
        if t is None:
            t = torch.randint(0, self.n_steps, (1,))
        noise = torch.randn_like(pose)
        noisy_pose = sqrt_alpha[t] * pose + sqrt_one_minus_alpha[t] * noise
        predicted_noise = self.model(noisy_pose, t)
        return F.mse_loss(predicted_noise, noise)
    
    def guidance(self, pose, data_grad, t, guidance_scale=1.0):
        score = self.score(pose, t)
        return score + guidance_scale * data_grad
```

**Ưu điểm:**
- Expressiveness cao nhất
- Conditioning linh hoạt (sign class, hand state, temporal)
- Gradient guidance → kết hợp tốt với data term

**Nhược điểm:**
- Cần train diffusion model (compute-intensive)
- Iterative sampling → chậm hơn single-forward methods
- Memory-intensive

**Expected improvement:** 8-12% so với VAE

---

### 2.4 Transformer Sequence Prior

**Tại sao sử dụng:**
- Captures temporal dependencies (sign language có grammar)
- Attention mechanism → focus vào relevant context
- Parallel processing → nhanh hơn autoregressive
- Đã chứng minh hiệu quả trong motion generation

**Cách hoạt động:**
```
Training:
  1. Input: sequence of body poses [T, 63]
  2. Transformer encoder: capture temporal patterns
  3. Loss: reconstruction + temporal consistency

Optimization:
  1. Feed all frames vào transformer
  2. Output: refined poses with temporal awareness
  3. Loss = data_loss + temporal_consistency_loss + transformer_prior_loss
```

**Implementation sketch:**
```python
class TemporalBodyPrior(nn.Module):
    def __init__(self, d_model=256, nhead=8, nlayers=6):
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), nlayers)
        self.pose_proj = nn.Linear(63, d_model)
        self.pose_unproj = nn.Linear(d_model, 63)
    
    def forward(self, poses):
        # poses: (T, 63)
        x = self.pose_proj(poses)
        x = self.encoder(x)
        refined = self.pose_unproj(x)
        return refined
    
    def loss(self, pred_poses, gt_poses):
        recon = F.mse_loss(pred_poses, gt_poses)
        # Temporal smoothness
        velocity = pred_poses[1:] - pred_poses[:-1]
        smooth = torch.mean(velocity**2)
        return recon + 0.1 * smooth
```

**Ưu điểm:**
- Temporal awareness → giảm jitter
- Captures sign language grammar patterns
- Parallel → nhanh hơn RNN

**Nhược điểm:**
- Cần sequence data để train
- Fixed window size → không handle sequence dài
- Memory: O(T²) với sequence length T

**Expected improvement:** 5-8% so với VAE

---

### 2.5 Direct Optimization + Adaptive Regularization

**Tại sao sử dụng:**
- Đơn giản nhất, không cần train model mới
- Loại bỏ hoàn toàn bottleneck của VAE decoder
- Gradient trực tiếp trên pose space → optimization stable
- Kết hợp tốt với SMPLer-X initialization

**Cách hoạt động:**
```
Thay vì optimize VAE latent:
  pose_embedding (33-dim) → decoder → body_pose (63-dim)

Trực tiếp optimize:
  body_pose (63-dim) với regularization từ multiple sources

Regularization:
  1. L2 vs SMPLer-X init: ||body_pose - smplerx_pose||²
  2. Biomechanical constraints: joint angle limits
  3. Temporal smoothness: ||body_pose_t - body_pose_{t-1}||²
  4. Learned statistics: ||body_pose - mean_pose||² cho outlier detection
```

**Implementation sketch:**
```python
# Direct optimization với adaptive regularization
body_pose = smplerx_init.clone().requires_grad_(True)

# Adaptive weight: tăng regularization khi far from init
distance = torch.norm(body_pose - smplerx_init)
adaptive_weight = base_weight * (1 + torch.exp(distance - threshold))

# Loss
data_loss = compute_data_loss(body_pose)
reg_loss = adaptive_weight * torch.sum((body_pose - smplerx_init)**2)
biomech_loss = compute_biomechanics(body_pose)
smooth_loss = compute_temporal_smooth(body_pose, prev_pose)

total_loss = data_loss + reg_loss + biomech_loss + smooth_loss
```

**Ưu điểm:**
- Không cần train model → implement nhanh
- Gradient trực tiếp → stable optimization
- Flexible regularization → adapt theo data
- SMPLer-X init tốt → regularization nhẹ là đủ

**Nhược điểm:**
- Không có learned prior → có thể overfit noisy data
- Regularization weights cần tune thủ công
- Không capture multimodal distribution

**Expected improvement:** 5-10% so với VAE

---

## 3. Phương pháp thay thế SignHPoser (Hand Prior)

### 3.1 MANO-Based Direct Optimization

**Tại sao sử dụng:**
- MANO là standard hand model → đã được validate extensively
- Direct optimization trên MANO params → không qua VAE bottleneck
- SMPL-X đã tích hợp MANO → compatible sẵn
- WiLoR/HaMeR output là MANO params → direct init

**Cách hoạt động:**
```
Thay vì:
  hposer3d_embedding (23-dim) → decoder → hand_pose (45-dim)

Trực tiếp optimize:
  hand_pose (45-dim) = MANO hand pose params
  
  MANO hand pose structure:
    - Global wrist rotation (3-dim)
    - 5 fingers × 3 joints × 3 DoF = 45-dim
    - Total: 48-dim (wrist) hoặc 45-dim (chỉ fingers)
```

**Implementation sketch:**
```python
# Direct MANO optimization
lhand_pose = wilor_init['left_hand_pose'].clone().requires_grad_(True)
rhand_pose = wilor_init['right_hand_pose'].clone().requires_grad_(True)

# MANO-specific regularization
def mano_prior_loss(hand_pose):
    # Penalize extreme finger bends
    finger_angles = hand_pose.reshape(15, 3)
    bend_penalty = torch.sum(torch.relu(torch.abs(finger_angles) - 1.5))
    
    # Natural hand shape: fingers shouldn't cross
    # (implement collision detection between fingers)
    cross_penalty = compute_finger_collision(finger_angles)
    
    return bend_penalty + cross_penalty

# Inverse Kinematics constraint
def ik_constraint(hand_pose, wrist_position):
    # Ensure hand pose is achievable given wrist position
    # Use forward kinematics to check reachability
    hand_joints = mano_forward_kinematics(hand_pose, wrist_position)
    return consistency_loss(hand_joints, expected_positions)
```

**Ưu điểm:**
- Không qua decoder → gradient chính xác
- MANO params interpretable → debug dễ
- Compatible với tất cả MANO-based models

**Nhược điểm:**
- MANO có 45-dim → high-dimensional optimization
- Có thể converge to implausible poses nếu không có prior
- Need careful regularization

**Expected improvement:** 5-8% so với VAE

---

### 3.2 Hand Shape Classifier + Prior

**Tại sao sử dụng:**
- Sign language có finite set of handshapes (hơn 50 handshapes trong ASL)
- Classification → discrete prior → regularization mạnh hơn
- Có thể use existing sign language handshape datasets
- Novel contribution: handshape-aware fitting

**Cách hoạt động:**
```
Training:
  1. Train classifier: image → handshape class
  2. Train per-class handshape prior: class → distribution of hand poses
  3. Dataset: ASL handshape dataset, hoặc annotate sign language data

Optimization:
  1. Classify handshape từ image
  2. Load corresponding handshape prior
  3. Optimize hand pose với class-specific regularization
```

**Implementation sketch:**
```python
class HandshapePrior:
    def __init__(self, handshape_models):
        # handshape_models: dict {class_name: {mean, cov}}
        self.models = handshape_models
    
    def get_prior(self, handshape_class):
        return self.models[handshape_class]
    
    def loss(self, hand_pose, handshape_class):
        prior = self.get_prior(handshape_class)
        # Mahalanobis distance
        diff = hand_pose - prior['mean']
        inv_cov = torch.inverse(prior['cov'])
        loss = diff @ inv_cov @ diff.T
        return loss

class HandshapeClassifier(nn.Module):
    def __init__(self, n_classes=50):
        self.backbone = ResNet18()
        self.head = nn.Linear(512, n_classes)
    
    def forward(self, image):
        features = self.backbone(image)
        return self.head(features)  # logits
```

**Ưu điểm:**
- Strong prior → constrain optimization effectively
- Class-specific → more accurate than generic hand prior
- Novel: chưa ai làm handshape-aware fitting

**Nhược điểm:**
- Cần handshape dataset (có thể cần annotate)
- Classifier accuracy ảnh hưởng trực tiếp
- Limited to known handshapes

**Expected improvement:** 8-12% so với VAE

---

### 3.3 Contact-Aware Hand Prior

**Tại sao sử dụng:**
- Sign language hands thường touch nhau hoặc touch body
- Contact constraints → strong regularization
- Physical plausibility → realistic results
- Novel: contact-aware sign language reconstruction

**Cách hoạt động:**
```
Training:
  1. Learn contact patterns từ sign language data
  2. When hands close → encourage finger interlock
  3. When hand near body → encourage surface contact

Optimization:
  1. Detect contact zones (hand-hand, hand-body)
  2. Apply contact-specific constraints
  3. Balance contact loss với data loss
```

**Implementation sketch:**
```python
class ContactAwarePrior:
    def __init__(self, contact_threshold=0.05):
        self.threshold = contact_threshold
    
    def hand_hand_contact(self, lhand, rhand):
        # Detect when hands are close
        distances = torch.cdist(lhand, rhand)
        in_contact = distances < self.threshold
        
        # When in contact, encourage specific patterns
        contact_loss = 0
        if in_contact.any():
            # Fingers should interlock, not overlap
            contact_loss = self.finger_interlock_loss(lhand, rhand)
            
            # Palms should face each other
            contact_loss += self.palm_orientation_loss(lhand, rhand)
        
        return contact_loss
    
    def hand_body_contact(self, hand, body_mesh):
        # When hand touches body, should be flat on surface
        distances = point_to_mesh_distance(hand, body_mesh)
        touching = distances < self.threshold
        
        if touching.any():
            # Hand should be parallel to body surface
            surface_normal = get_surface_normal(body_mesh, hand)
            hand_normal = get_palm_normal(hand)
            return 1 - F.cosine_similarity(surface_normal, hand_normal)
        return 0
```

**Ưu điểm:**
- Physical plausibility → realistic results
- Strong constraints → reduce ambiguity
- Novel contribution

**Nhược điểm:**
- Complex implementation (collision detection, surface normals)
- May over-constrain if thresholds wrong
- Need to handle occlusion

**Expected improvement:** 5-10% so với VAE

---

### 3.4 Finger Articulation Graph Prior

**Tại sao sử dụng:**
- Fingers có dependencies (cùng tendon → correlated motion)
- Graph structure capture anatomical constraints
- Learned graph → more accurate than hand-crafted rules
- Novel: graph-based hand prior for sign language

**Cách hoạt động:**
```
Training:
  1. Build graph: nodes = finger joints, edges = anatomical connections
  2. Learn edge weights từ sign language data
  3. Graph neural network → predict plausible finger configurations

Optimization:
  1. Feed current hand pose vào GNN
  2. GNN output: refined hand pose
  3. Loss = data_loss + ||GNN(input) - target||²
```

**Implementation sketch:**
```python
class FingerGraphPrior(nn.Module):
    def __init__(self, n_joints=20, d_model=64):
        # Build hand graph
        self.graph = self.build_hand_graph()
        
        # Graph neural network
        self.gnn = GATConv(d_model, d_model, heads=4)
        self.proj_in = nn.Linear(3, d_model)
        self.proj_out = nn.Linear(d_model, 3)
    
    def build_hand_graph(self):
        # Nodes: 0=wrist, 1-4=thumb, 5-8=index, etc.
        edges = []
        # Connect joints within each finger
        for finger_start in [1, 5, 9, 13, 17]:
            for i in range(finger_start, finger_start+3):
                edges.append([i, i+1])
                edges.append([i+1, i])
        # Connect palm joints
        edges.extend([[1,5], [5,9], [9,13], [13,17]])
        return torch.tensor(edges).T
    
    def forward(self, hand_joints):
        # hand_joints: (20, 3)
        x = self.proj_in(hand_joints)
        x = self.gnn(x, self.graph)
        refined = self.proj_out(x)
        return refined
    
    def loss(self, pred_joints, gt_joints):
        refined = self.forward(pred_joints)
        return F.mse_loss(refined, gt_joints)
```

**Ưu điểm:**
- Capture anatomical constraints
- Learnable → adapt to sign language patterns
- Graph structure → interpretable

**Nhược điểm:**
- Need graph construction expertise
- GNN training data requirements
- May not generalize well

**Expected improvement:** 5-8% so với VAE

---

### 3.5 Contrastive Hand Pose Prior

**Tại sao sử dụng:**
- Sign language handshapes: same class = similar pose, different class = different pose
- Contrastive learning → learn discriminative representation
- Novel: contrastive prior for sign language hands

**Cách hoạt động:**
```
Training:
  1. Positive pairs: same handshape, different frames
  2. Negative pairs: different handshapes
  3. Learn embedding space: same class close, different class far

Optimization:
  1. Encode current hand pose
  2. Pull toward positive cluster
  3. Push away from negative clusters
```

**Implementation sketch:**
```python
class ContrastiveHandPrior(nn.Module):
    def __init__(self, d_embed=64, temperature=0.07):
        self.encoder = nn.Sequential(
            nn.Linear(45, 128), nn.ReLU(),
            nn.Linear(128, d_embed))
        self.temperature = temperature
    
    def forward(self, hand_pose):
        return F.normalize(self.encoder(hand_pose))
    
    def contrastive_loss(self, anchor, positives, negatives):
        # anchor: (1, d_embed)
        # positives: (N, d_embed) same handshape
        # negatives: (M, d_embed) different handshape
        
        pos_sim = F.cosine_similarity(anchor, positives) / self.temperature
        neg_sim = F.cosine_similarity(anchor, negatives) / self.temperature
        
        logits = torch.cat([pos_sim, neg_sim])
        labels = torch.zeros(len(pos_sim), dtype=torch.long)
        
        return F.cross_entropy(logits.unsqueeze(0), labels)
    
    def prior_loss(self, hand_pose, handshape_class):
        embedding = self.forward(hand_pose)
        # Pull toward class centroid
        centroid = self.class_centroids[handshape_class]
        return F.mse_loss(embedding, centroid)
```

**Ưu điểm:**
- Discriminative → better separation between handshapes
- Embedding space → can visualize and interpret
- Novel contribution

**Nhược điểm:**
- Need labeled handshape data
- Contrastive training can be unstable
- Class centroids need to be maintained

**Expected improvement:** 6-10% so với VAE

---

## 4. Combined Strategy: Hybrid Prior Architecture

### 4.1 Multi-Prior Ensemble

**Tại sao sử dụng:**
- Mỗi prior có strengths khác nhau
- Ensemble → robust hơn single prior
- Adaptive weighting → context-dependent

**Cách hoạt động:**
```
Total prior loss = w₁ * GMM_loss + w₂ * Flow_loss + w₃ * Contact_loss + ...

Adaptive weights:
  - Khi confident về pose → giảm prior weight
  - Khi uncertain → tăng prior weight
  - Khi near contact → tăng contact prior weight
```

### 4.2 Hierarchical Prior

**Tại sao sử dụng:**
- Body và hand có different characteristics
- Separate priors → more specialized
- Hierarchical structure → capture dependencies

**Cách hoạt động:**
```
Stage 1: Body prior → refine body pose
Stage 2: Hand prior (conditioned on body) → refine hand pose
Stage 3: Contact prior (conditioned on body + hand) → enforce consistency
```

---

## 5. Khuyến nghị Implementation

### Phase 1 (1-2 tháng): Direct Optimization
- Thay VAE bằng direct optimization
- L2 regularization vs SMPLer-X init
- Biomechanical constraints
- **Expected:** 5-10% improvement

### Phase 2 (2-4 tháng): GMM + Contact Prior
- Train GMM trên sign language data
- Implement contact-aware constraints
- **Expected:** 8-12% improvement

### Phase 3 (4-8 tháng): Diffusion/Flow Prior
- Train diffusion/flow model
- Score-based guidance
- **Expected:** 10-15% improvement

### Paper Strategy:
- **Workshop paper:** Phase 1 + 2 (WACV/3DV workshop)
- **Conference paper:** Phase 3 (CVPR/ECCV)
- **Journal paper:** Full system + ablation studies (TPAMI)

---

## 6. Tài liệu tham khảo

| Paper | Venue | Relevance |
|-------|-------|-----------|
| SMPLify (original) | ECCV 2016 | GMM body prior |
| SMPLify-X | ICCV 2019 | VAE body+hand prior |
| PHD | ICCV 2025 | Diffusion body prior |
| ScorePose | 2024 | Score-based pose prior |
| MotionGPT | 2024 | Transformer motion prior |
| HandDGP | ECCV 2024 | Hand-specific priors |
| ContactOpt | 2021 | Contact-aware optimization |
| TUCH | ICCV 2021 | Uncertainty-aware fitting |
