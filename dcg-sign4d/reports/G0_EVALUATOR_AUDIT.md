# G0 evaluator audit

Status: **engineering implementation passes; scientific G0 is not yet closed for DCG-Sign4D**.

The attached author evaluator computes each hand's `TR` value after centering
that hand region on its own mean. Its `V2V * wrist` call also centers each hand
on its own mean and adds the same target wrist to both point sets. Algebraically,
the added wrist cancels, so this is the same region-translation-aligned metric;
it does not preserve hand placement relative to the body and is not a true
wrist-aligned articulation metric.

DCG-Sign4D therefore reports three distinct endpoints:

1. primary root-aligned hand PVE: prediction and target are independently
   centered at their own pelvis/root, preserving hand-to-body placement;
2. wrist-aligned hand PVE: each hand is centered at its own regressed wrist,
   measuring local articulation;
3. legacy region-TR hand PVE: retained only for direct comparison with prior
   attached-author numbers.

`tests/unit/test_hand_metrics.py` proves global translation invariance and that
a rigid hand-placement error is visible to the primary root-aligned endpoint
while being removed by wrist/region alignment. Strict SGNify evaluation also
requires exact frame coverage and the frozen 10,475-vertex/20,908-face topology.

G0 cannot be marked scientifically complete until the final evaluation config,
SGNify split, author-approved coordinate convention, and signer/bootstrap policy
are frozen. Local SGNify assets do not expose usable signer IDs.

