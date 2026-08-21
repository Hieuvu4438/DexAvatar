# Strict DexAvatar OBJ export and author evaluation

Date: 2026-08-20

## Outcome

The A1 and SIGNAL-4D predictions were materialized as DexAvatar-compatible
SMPL-X OBJ trees and evaluated after being read back through the author's own
`read_verts_and_faces` function. The central-subset A1 and final SIGNAL-4D
results also match a direct invocation of the unmodified author `main()` at all
reported four-decimal values.

The earlier safetensors-transport table is superseded by this report. Directly
centering float32 camera-space tensors caused about 0.008 mm of cancellation
error at the approximately 17 m coordinate magnitude. The strict route is:

`safetensors -> DexAvatar OBJ (8 decimal places) -> author OBJ reader -> author metrics`

## Material passport

| Property | Strict value |
|---|---|
| Layout | `<method>/<sign>/smplifyx/meshes/low_<frame>.obj` |
| Header | `# https://github.com/mikedh/trimesh` |
| Vertex record | `v %.8f %.8f %.8f` |
| Face record | `f %d %d %d`, one-indexed |
| SMPL-X topology | 10,475 vertices, 20,908 triangular faces |
| Coordinate convention | OpenCV: x right, y down, z forward |
| Round-trip bound | max absolute coordinate error 0.000005000001 mm |
| Integrity | per-OBJ SHA-256 plus source-artifact SHA-256 |
| Coverage | exact expected filenames, 100%; missing/extra files are fatal |
| Incomplete exports | none |

The evaluated delivery set contains 4,172 OBJ files and occupies approximately
3.0 GB (excluding the 14-file reproducibility duplicate).
The prospective comparator contains 56 clips/769 frames per method. The
author-central set contains 24 clips/655 frames per method.

A fresh repeat export of the 14-frame smoke slice is byte-for-byte identical
to the first export (`diff -qr` returned no differences), including its export
manifest. This validates deterministic serialization independently of metric
reproducibility.

## Prospective extended-post comparison

These are the author's vertex-micro means in millimetres; lower is better.
The exact author metric functions, region assets, MANO IDs, class-0 left-hand
rule, and OBJ parser are used over the frozen prospective manifest.

| Method | TR all | TR upper body | TR left hand | TR right hand |
|---|---:|---:|---:|---:|
| A1 | 43.0164 | 29.2932 | 14.4918 | **12.8414** |
| SIGNAL-4D v5 | **43.0071** | **29.2763** | **14.0353** | 12.8425 |
| SIGNAL-4D minus A1 | **-0.0093** | **-0.0169** | **-0.4565** | +0.0012 |

SIGNAL-4D improves left-hand TR-V2V by 0.4565 mm (3.15%) under the author's
vertex-micro aggregation. The 0.0012 mm right-hand increase is negligible but
is reported rather than hidden. This table differs from the preregistered
equal-weight clip-macro table because the author code pools vertex errors over
all frames; it is not a contradiction or a replacement for that estimand.

## Author-central comparison

This is the exact central-frame policy encoded by the attached author evaluator.

| Method | TR all | TR upper body | TR left hand | TR right hand |
|---|---:|---:|---:|---:|
| A1 | 42.1632 | 26.5917 | 12.5624 | **12.0760** |
| M0 | **37.5740** | **25.5687** | 20.3612 | 20.5028 |
| M1 | 42.3166 | 26.8269 | 13.0772 | 12.3445 |
| SIGNAL-4D v5 | 42.1661 | 26.6117 | **11.9978** | 12.0777 |
| SIGNAL-4D v5 minus A1 | +0.0028 | +0.0200 | **-0.5646** | +0.0017 |

M0's low all-body/upper-body error is accompanied by severe hand degradation,
so it is not the final method. SIGNAL-4D v5 improves the targeted left hand by
0.5646 mm while retaining the remaining A1 geometry almost exactly.

## Direct original-main validation

The author source was not edited. Its SHA-256 is
`400bfbd736fc59fcc1867af7650188b61772136982f64b623df31494e6116877`.
For both A1 and SIGNAL-4D v5, every reported central metric from direct
`main()` invocation equals the structured strict result after the author's
four-decimal formatting. Status: **PASS**.

The prospective population is not passed to the author's fixed central-frame
selection. It instead uses the frozen prospective frame manifest while keeping
the author's OBJ parser, metric functions, regions, class rule, and
vertex-micro aggregation unchanged. Therefore “exact original main” applies to
the central table; “exact author metric kernel on a preregistered extended
population” applies to the prospective table.

## Artifacts

- `prospective_extended_post/A1`: strict A1 OBJ tree.
- `prospective_extended_post/SIGNAL4D_v5`: strict final fitting OBJ tree.
- `central_test/{A1,M0,M1,M1_v5_multigate_revealed}`: strict central OBJ trees.
- `central_test/comparison.{json,csv,md}`: machine-readable central results.
- `prospective_extended_post/comparison.{json,csv,md}`: machine-readable
  prospective results.
- `original_main_validation.json`: direct-original-main equality record.
- `signal4d/logs/author_original_main_strict_central_{A1,M1_v5}.log`: raw logs.

Every export directory includes `export_manifest.json`, containing dimensions,
coordinate convention, source hashes, output hashes, and per-file round-trip
error. The strict evaluator verifies these hashes before computing any metric.

## Claim boundary

This establishes a best result against the same-protocol A1 comparator on the
frozen prospective SIGNAL-4D endpoint, especially for left-hand fitting. It is
not evidence of an external published-leaderboard or unseen-signer SOTA result.
