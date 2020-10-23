#!/bin/bash

PYTHON_EXE_NAME="/home/ryan.lagerquist/anaconda3/bin/python3.6"
CODE_DIR_NAME="/home/ryan.lagerquist/ml4rt/ml4rt"

EXAMPLE_DIR_NAME="/home/ryan.lagerquist/ml4rt_project/examples"
TOP_MODEL_DIR_NAME="/home/ryan.lagerquist/ml4rt_project/u_net/dual_weighted_mse_loss/unnorm_target"
EXTENSIONLESS_MODEL_FILE_NAME="model_epoch=115_val-loss=0.428658"

SUBSET_NAMES=("best-prmse" "worst-prmse" "best-bias" "worst-low-bias" "worst-high-bias")

model_dir_name="${TOP_MODEL_DIR_NAME}/${EXTENSIONLESS_MODEL_FILE_NAME}"
model_file_name="${model_dir_name}.h5"

for((i=0; i<${#SUBSET_NAMES[@]}; i++)); do
    "${PYTHON_EXE_NAME}" -u "${CODE_DIR_NAME}/scripts/make_saliency_maps_all_targets.py" \
    --input_model_file_name="${model_file_name}" \
    --input_example_dir_name="${EXAMPLE_DIR_NAME}" \
    --input_example_id_file_name="${model_dir_name}/validation/predictions_${SUBSET_NAMES[$i]}.nc" \
    --ideal_activation=42 \
    --vector_output_layer_name="leaky_re_lu_20" \
    --output_saliency_file_name="${model_dir_name}/validation/saliency_maps/saliency_${SUBSET_NAMES[$i]}.nc"
done
