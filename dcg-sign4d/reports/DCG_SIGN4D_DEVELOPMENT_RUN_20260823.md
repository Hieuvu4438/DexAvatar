# Báo cáo triển khai và chạy DCG-Sign4D — 2026-08-23

## Kết luận

Pipeline DCG-Sign4D đã được triển khai độc lập trong `dcg-sign4d/` và đã chạy xuyên qua các
thành phần thật trên SGNify: dữ liệu 57 clip, SMPL-X, DPoser-X chính thức, selfcontact chính
thức, hai checkpoint được huấn luyện, reconstruction một clip và evaluator dùng asset của tác
giả. Trạng thái kỹ thuật là **PASS cho development integration**.

Kết quả hiện tại **không phải kết quả khoa học cuối cùng của DCG-Sign4D**. Các đầu vào còn thiếu
để đạt protocol chính thức là nhãn calibration độc lập, gold contact được gán nhãn đôi, patch map
geodesic được duyệt, signer ID/signer-disjoint split, ranker fit từ candidate validation thật và
training/inference budget đã freeze.

## Tài nguyên thực tế đã dùng

| Tài nguyên | Cách dùng và bằng chứng | Trạng thái |
|---|---|---|
| SMPL-X neutral | Forward 57 clip/1.493 frame; SHA-256 `37602144...992` | PASS |
| SMPLer-X checkpoint | Ghi provenance trong initialization; SHA-256 `3d405111...33b` | PASS reuse |
| DPoser-X wholebody | Official frozen backbone; audit 13 frame cho output `[1,13,337]` hữu hạn | PASS |
| selfcontact essentials | Registry SHA-256 `14886f1e...a09`; signed winding-number chạy trên frame thật | PASS với `test_segments=false` |
| Sapiens PKL | 198.569 keypoint được chuyển vào observation contract | PASS development; thiếu checkpoint provenance |
| SGNify outputs | 57 clip/1.493 frame, split development 16/12/5/24 | PASS development; không signer-disjoint |
| MANO, detector, VQ-VAE | Đã kiểm kê nhưng không cần cho runtime chính đã freeze | Không đưa vào runtime |

`selfcontact_test_segments=true` không được dùng: constructor `BodySegment` của upstream chưa hoàn
tất sau hơn 10 phút do vòng lặp Python theo vertex/face. Với `false`, vẫn dùng signed generalized
winding-number chính thức; một frame có 60 edge, 24 edge penetration, depth cực đại 0,18618 m và
area 0,00381355 m², toàn bộ hữu hạn. Audit nằm tại
`artifacts/audits/selfcontact_sgnify_tisch_frame0_real_v1.json`.

## Ma trận tuân thủ proposal

| Phần proposal | Hiện thực/bằng chứng | Kết luận |
|---|---|---|
| 0–3: method và flow | Init → geometry → contact → semi-Markov → diffusion/guidance → alternating → K/ranker | PASS code |
| 4: layout/dependency | Package riêng; 4 repo chính thức pin commit trong `third_party/manifest.yaml` | PASS engineering |
| 5: contracts/toạ độ | Manifest, camera, trajectory 337-D, observation, graph, patch-map schema; strict hashes | PASS |
| 6: observation/calibration | Cache đầy đủ và cue-mask; synthetic scalar calibrator | PARTIAL, development only |
| 7: DexAvatar initialization | 57/57 clip; state, beta, camera, SMPL-X forward; replay error 0 | PASS development |
| 8: contact geometry | Patch distance, normal, velocity, signed depth/area qua official selfcontact | PASS integration |
| 9: labels | State machine và audit G1 có đủ; label hiện tại là proximity pseudo-label rất thưa | BLOCKED scientific |
| 10: dynamic proposal | Temporal edge model, edge identity, cue stats, balanced loss/Brier/dropout | PASS code; provisional training |
| 11: semi-Markov | Transition/duration constraint và exact decoding theo giây | PASS |
| 12: holistic diffusion | Một denoiser 337-D, shape/contact conditioning, official DPoser-X, windowing SO(3) | PASS development |
| 13: guided sampler | Keypoint/silhouette/track/depth/contact gradients, clipping và trust region | PASS code/tests |
| 14: alternating inference | Fixed rounds, re-estimate graph, same-seed retry, artifact từng round | PASS |
| 15: multi-hypothesis/ranking | Seed độc lập, GT-free scoring và artifact contract | PARTIAL; ranker bootstrap synthetic |
| 16–18: training/config/CLI | Stage 2/3 trainers, immutable checkpoints, readiness, reconstruct/evaluate CLIs | PASS development |
| 19: prediction contract | Tensor/hash/seed/config/source/ranking/completion validation | PASS |
| 20: evaluation | Strict coverage/topology, root/wrist/body/temporal/contact/UQ utilities | PASS engineering |
| 21–22: baselines/gates | Baseline B0 evidence và G0/G1 machinery | PARTIAL; B1–B7/G2–G5 chưa hợp lệ |
| 23–25: tests/compute/governance | 111 tests, ruff, timing/memory/provenance/license audit | PASS engineering; clean-env chưa chạy |
| 26–30: DoD/claim/backlog | Development reconstruction chạy; scientific DoD chưa đạt và claim bị chặn | PARTIAL đúng thiết kế |

Khoảng trống cụ thể ở M4: optional 2D cues được đưa vào dưới dạng thống kê global và runtime
đã có gradient cho từng cue, nhưng SGNify cache hiện không có mask/track/depth và chưa có mapping
joint-to-patch được author freeze. Vì vậy không nên mô tả đây là bản đầy đủ của patch-specific 2D
feature trong protocol khoa học.

## Huấn luyện development thật

| Stage | Dữ liệu | Thiết lập | Kết quả |
|---|---|---|---|
| Contact proposal | 16 train window / 5 validation window, 60 edge | 100 GPU step | best validation loss `0.0010603`, step 30, 45,44 s |
| Holistic diffusion | Cùng split, official DPoser-X frozen | 100 GPU step | validation `1.90761` → `1.57607`, 56,39 s |

Contact labels có mất cân bằng cực mạnh: toàn bộ corpus provisional có 89.512 `off`, 9 `onset`,
55 `hold`, 4 `release`, 371 `uncertain`. Balanced window sampling đã hoạt động, nhưng loss thấp
không chứng minh proposal học được contact thật.

## Reconstruction và evaluator thật

Run `artifacts/runtime_smoke/muell_dcg_real_components_cpu_v2` dùng 15 frame, một hypothesis và
một diffusion step trên CPU. Assembly mất 3,65 s, inference mất 239,67 s, seed `1523047237`,
không retry. GPU không đủ bộ nhớ tại thời điểm thử vì các process ngoài scope chỉ để lại khoảng
1,6–2,8 GB, trong khi selfcontact cần khoảng 3,51 GB; không process nào của người dùng bị dừng.

Evaluator v3 dùng cùng manifest, cùng 15 frame và cùng author GT cho cả ba output:

| Output | Root hand PVE ↓ | Wrist hand PVE ↓ | Legacy hand PVE ↓ | Body MPJPE ↓ | Hand velocity ↓ |
|---|---:|---:|---:|---:|---:|
| DCG development smoke | 93,2422 | 21,3754 | 15,6398 | 57,1136 | 274,1976 |
| DexAvatar reference | 92,9340 | 22,0581 | 16,3934 | 57,1258 | 268,3707 |
| Signal4D v5 reference | 89,2168 | 21,2451 | 15,5706 | 54,8619 | 320,8183 |

Đơn vị PVE/MPJPE là mm, velocity là mm/s. So với DexAvatar, smoke DCG kém `+0,3082` mm ở primary
root-aligned hand PVE, tốt hơn `-0,6827` mm ở wrist-aligned PVE, gần như ngang body MPJPE
(`-0,0122` mm) và kém `+5,8269` mm/s ở velocity. So với Signal4D v5, primary kém `+4,0254` mm
nhưng velocity tốt hơn `-46,6207` mm/s.

Đây chỉ là một clip `Muell`, 15 frame và một tay phải theo sign metadata. Không có bootstrap,
không có ý nghĩa thống kê, không được dùng để nói DCG tốt hơn hoặc kém hơn baseline tổng thể.
Machine-readable comparison: `reports/dcg/muell_development_comparison_v1.json`.

## Gate và claim boundary

| Gate | Trạng thái hiện tại |
|---|---|
| G0 evaluator | PASS engineering; BLOCKED scientific freeze/signer policy |
| G1 contact labels | BLOCKED vì không có real double annotation/gold audit |
| B0 | Có baseline evidence 57 clip, chưa phải registered freeze |
| B1–B7 | Development checkpoint có, matched scientific runs chưa có |
| A-INF0/A-INF1/A-K | Chưa chạy theo budget/freeze chính thức |
| G2–G5 | Chưa testable |
| Scientific DCG claim | BLOCKED |

Các baseline full-set trước đó vẫn giữ nguyên: DexAvatar root-hand PVE `67,6893` mm và Signal4D
v5 `65,9737` mm trên 57 clip/1.493 frame. Chênh lệch này là clip-bootstrap sensitivity, không
phải signer-cluster CI và không chứa output DCG.

## Kiểm chứng cuối

- `pytest -q`: 111 passed.
- `ruff check src tests`: passed.
- Development readiness: READY cho 5 validation clip, nhưng report ghi rõ bootstrap ranker.
- Production readiness: BLOCKED và không tạo prediction artifact.
- Real runtime smoke: completed, finite, zero retry.
- Author evaluator: coverage 1,0 cho cả ba output trên cùng 15 frame.

Muốn nâng trạng thái lên scientific run cần: bổ sung signer ID và split signer-disjoint; tạo nhãn
calibration độc lập; double-annotate gold contact; duyệt patch map geodesic; huấn luyện theo budget
đã freeze; tạo candidates validation thật để fit ranker; chạy đủ B0–B7, A-INF0/A-INF1/A-K và
signer-cluster bootstrap.
