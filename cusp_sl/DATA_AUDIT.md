# CUSP-SL local data-lineage audit

Audit date: 2026-08-22. Presence on disk is not sufficient for admission to a
Methods experiment. A source must preserve sequence identity, timestamps,
complete modeled rotations, observation provenance, and a source-disjoint split.

| Local source | Inspected evidence | Admission decision |
|---|---|---|
| How2Sign temporal cache | `cache/phase2/t2_how2sign_2d_temporal_reprojection_v2/`; 10,822 training clips, a source-disjoint validation manifest, immutable source-video IDs/timestamps, 51 SMPL-X rotations and quality-controlled 2D-track temporal pseudo-targets | Admitted as pseudo-target training/development data. It is not called mocap truth. Source groups are split before window sampling, and SGNify is excluded. |
| Legacy How2Sign body-prior array | `data/signbposer_data/raw/how2sign/body_poses.npy`, shape `(990,63)`, SHA-256 `da4758a455e5a53aa6ae397d5f0c0de36b0e5df5a1a133477675fcb123e0c2fb` | Not admitted to CUSP Q/G. It contains body rotations only and does not preserve the complete temporal/frontend error contract required by the proposed method. |
| PHOENIX-2014-T RGB release | `data/signbposer_data/raw/phoenix/PHOENIX-2014-T-release-v3/`; the local README identifies nine signers and official train/dev/test data | Potential future RGB source only after a new source-keyed, split-preserving frontend extraction. Merely existing locally does not make it paired 3D supervision. |
| Legacy PHOENIX SMPLer-X array | `data/signbposer_data/raw/phoenix/extracted/body_poses.npy`, shape `(822,63)`, SHA-256 `4a29a4f2e12bcbe09ec35c7a712dd57fe95912e8922b062472ef7d1cba316e1f`; `extraction.log` records 42 processed videos from the release `test/` directory, 20 sampled frames per video | Rejected for CUSP training and model selection. It is test-derived, body-only, has no MANO hands/global pose/camera/timestamps, and the aggregate NPY lacks a row-to-source manifest. Admission would create both representation mismatch and leakage risk. |
| SGNify | `outputs/cusp_sl/protocol_inputs_1493_v1/manifest.json`; 1,493 targetless input frames across 57 signs | Evaluation only. It is never read by Q/G training, gate fitting, energy fitting, or candidate selection. Prior exploratory exposure is disclosed. |
| SignAvatars / independent clean 3D | No licensed, source-associated complete artifact admitted by the current CUSP run | Not silently substituted. These remain conditional future sources requiring license, source association, representation mapping and held-out retargeting validation. |

The current run therefore uses How2Sign pseudo-targets for training/development
and SGNify only for the frozen evaluation protocol. PHOENIX is deliberately not
used to inflate sample count. A future PHOENIX extension must start from the
official train split, retain `(signer, video, frame, timestamp)` keys, rerun both
frozen frontends, preserve all 51 local rotations and camera/crop metadata, and
freeze source-disjoint train/validation/test manifests before fitting anything.
