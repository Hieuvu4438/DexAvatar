# G2 cheap-baseline decision

Development split: 21 clips / 578 frames, manifest SHA-256 `fe43a102dfe36340d50015a910bf45fd4141d8df0cb9fbb83da006f10334c5f9`.

| Method | Upper TR-V2V | Left TR-V2V | Right TR-V2V | Velocity error |
|---|---:|---:|---:|---:|
| Raw M0 hybrid | 25.4663 | 37.9528 | 19.8187 | 6.2882 |
| Full-tree 2D/temporal smoother | 26.5217 | 36.9836 | 19.5867 | 6.1803 |

All geometric values are equal-weight clip-macro millimeters. Coverage is 100% for both methods.

Paired clip-bootstrap candidate-minus-baseline deltas (10,000 replicates):

- Upper body: +1.0555 mm, 95% CI [+0.5315, +1.5890] — definite regression.
- Left hand: -0.9692 mm, 95% CI [-1.5049, -0.4357] — improvement.
- Right hand: -0.2321 mm, 95% CI [-0.7089, +0.2196] — inconclusive.
- Velocity error: -0.1080, 95% CI [-0.1254, -0.0907].

Decision: reject full-tree M0 smoothing because it violates the body non-regression requirement. Keep raw M0 as the cheap baseline and restrict later refinement to the hand/forearm chains with demonstrated headroom.
