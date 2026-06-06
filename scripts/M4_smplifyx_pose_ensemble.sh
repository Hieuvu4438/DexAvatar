
cd "${FITTING_EXPERIMENT}"
export PYTHONPATH=$PYTHONPATH:$(pwd)/smplifyx
export PYTHONPATH=$PYTHONPATH:$(pwd)

python script.py --path ${ROOT_PATH} --out_path ${OUTPUT_PATH} --gpu_id 0 --split_num 1 --config cfg_files/fit_smplx_vposer_x_ensemble.yaml
