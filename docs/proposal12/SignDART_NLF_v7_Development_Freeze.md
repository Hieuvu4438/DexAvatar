# SignDART-NLF v7: Development Freeze for Inference Experiments

## Material Passport

- **Artifact type:** development freeze after an explicitly retrospective Pareto analysis
- **Created:** 2026-09-02 (Asia/Ho_Chi_Minh)
- **Status:** final Engineering12 candidate-bank specification; not an independent preregistration
- **Development evidence:** `reports/design/risk_tradeoff_engineering12.json`
- **Next independent gate:** GT-free selection on frames/signs not used to fit the selector

## Frozen method

V7 uses the finite collar--shoulder--elbow--wrist depth branch tree from v5 and the distal deformation risk from v6. A side branch is valid only when its centered MANO-region deformation relative to H1 is at most **0.5 mm**. Exact H1 remains the abstention candidate. No SMPL-X parameter outside the collar/shoulder/elbow/wrist chains is changed; global wrist frames and finger local rotations remain those of H1.

The 0.5 mm operating point was selected on Engineering12 from the declared risk--accuracy Pareto curve. It is therefore a tuned development parameter, not confirmatory evidence. It was chosen as the knee: compared with 0.25 mm it retains substantially more UBody-H ceiling, whereas moving to 0.75 mm provides little additional ceiling and violates the pre-existing right-hand margin.

## Frozen development gates

The oracle thresholds are unchanged: UBody-H gain >=0.75 mm, UBody gain >=0.30 mm, All gain >=0.15 mm, and left/right hand regression <=0.02 mm. These values will be materialized by the standard G1/G2 scripts for provenance, but their v7 pass/fail is an internal consistency check because the Pareto diagnostic already exposed the same development labels.

## Selector boundary

From this freeze onward, a valid inference method must:

1. use no 3D ground truth at inference;
2. score only candidates that pass the 0.5 mm distal-risk filter;
3. retain exact H1 as an explicit abstention outcome;
4. freeze feature definitions, training signs, regularization, and decision threshold before evaluating untouched45;
5. report oracle ceiling separately from selector performance.

The intended evidence consists of NLF 2.5D joint predictions and uncertainties, candidate projection/depth features, and H1 state information. Direct wholesale replacement by an NLF mesh or pose is outside scope.

