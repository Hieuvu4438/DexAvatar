# Third-party components

SignEFT-X source does not vendor model repositories or checkpoints. A complete
run expects locally installed/licensed copies of:

| Component | Role | Distributed here |
|---|---|---|
| SMPL-X / `smplx` | Parametric body decoder | Python package only; body-model assets are external |
| WiLoR | Frozen monocular hand observations and primary initialization | No |
| HaMeR | Deterministic coverage fallback for missing primary frames | No |
| Ultralytics detector | Hand bounding boxes used by the WiLoR adapter | No |

Users must obtain each repository, checkpoint, and body-model asset under its
original terms. The project MIT license covers only code authored or adapted
for this repository; it does not relicense third-party assets.
