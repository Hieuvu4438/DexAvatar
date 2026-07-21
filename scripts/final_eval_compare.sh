#!/bin/bash
# Final evaluation: compare all method variants on the Exp2 sign subset.
# Produces a per-frame CSV then an active-hand TR-V2V comparison table restricted
# to the signs present in exp2_base_nlf (the clean-isolation set).
source /home/haipd/miniconda3/etc/profile.d/conda.sh
conda activate dexavatar
cd /home/haipd/DexAvatar

OUT=outputs/exp2_base_nlf
python evaluation/eval_mpvpe_regions.py \
    --methods exp1_paper_nlf exp2_base_nlf output_baseline method_nlf_wilor \
    --method_names paper-nlf base-nlf baseline base-direct-nlf \
    --central_frames \
    --output_csv $OUT/eval_all.csv > $OUT/eval_all.log 2>&1

echo "=== Comparison restricted to Exp2 signs (active-hand TR-V2V, central) ==="
python evaluation/compare_methods.py --csv $OUT/eval_all.csv --ref_method base-nlf
