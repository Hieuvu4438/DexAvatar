# Licensed asset placement

SIGNAL-4D does not redistribute SGNify images/ground truth, SMPL-X model files,
SMPLer-X checkpoints, WiLoR outputs, or legacy DexAvatar outputs. Obtain each
asset from its owner under its applicable license and place it at the paths
expected by the frozen manifests/configuration.

Required local inputs for the recorded experiment:

- `data/frames/` and `data/smplx_gt/`: SGNify endpoint assets.
- `SMPLer-X/common/utils/human_model_files/smplx/SMPLX_NEUTRAL.npz`: licensed
  neutral SMPL-X model; recorded SHA-256
  `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`.
- `outputs/output_baseline/`: read-only SMPLer-X body and WiLoR hand estimates.
- `outputs/method_biomech/`: optional read-only legacy fitted hypothesis used as
  the strongest same-protocol control. SIGNAL-4D never edits this directory.

The prospective extended-post experiment also reads the original DexAvatar
Ensemble and HaMeR fitting code/configuration without modifying them. New fits
are redirected to `signal4d/artifacts/legacy_a1_*`; the legacy source tree and
all pre-existing output roots remain read-only. If both fitted sources lack a
declared frame, the preregistered terminal baseline source is the corresponding
raw SMPLer-X A0 file under `outputs/output_baseline/`.

The frozen 769-frame prospective comparator resolved availability per frame as
607 balanced Ensemble A1, 145 original-HaMeR A0, and 17 terminal raw SMPLer-X
A0 frames. These counts are recorded before GT evaluation in
`artifacts/legacy_a1_hamer_extended_post_v1/fallback_finalize.json`.

Run `signal4d preprocess` only after these assets are present. Every consumed
file hash is written to canonical cache metadata, and every cache artifact hash
is written to `run.json`. Dataset/model licenses remain authoritative; this
repository's license does not grant rights to those external assets.
