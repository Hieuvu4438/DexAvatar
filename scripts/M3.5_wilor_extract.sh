#!/bin/bash
set -euo pipefail

python M3_mean_shape_smplerx.py --input_path ${ROOT_PATH} --output_path ${OUTPUT_PATH}
echo "neutral" > ${OUTPUT_PATH}/gender.txt

cd WiLoR
unset CXX
unset CC
/home/haipd/miniconda3/envs/wilor/bin/python export_hamer_pkl.py \
  --img_folder ${ROOT_PATH} \
  --out_folder ${OUTPUT_PATH} \
  --fast

test -f "${OUTPUT_PATH}/hamer/hamer.pkl"
