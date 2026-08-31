# License boundary

DCG-Sign4D does not redistribute proprietary SMPL-X/MANO/FLAME models or SGNify
data. Their access terms remain with the asset owner.

Official source checkouts are stored under `third_party/` and retain their own
license files. `third_party/manifest.yaml` records the exact source, commit and
license SHA-256 used for development. In particular, TUCH and selfcontact are
research code whose license must be reviewed for the intended use; public GitHub
availability is not treated as unrestricted permission.

Dataset manifests must carry `dataset_version` and `license_id`; placeholders are
rejected by the production validator.

