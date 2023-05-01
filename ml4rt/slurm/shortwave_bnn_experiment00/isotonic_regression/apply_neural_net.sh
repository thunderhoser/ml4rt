#!/bin/bash

# set -x

model_dir_name=$1
model_file_name="${model_dir_name}/model.h5"
model_dir_name="${model_dir_name}/model"

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_standalone/ml4rt"
EXAMPLE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/shortwave_examples_600days/orig_heights/normalized_not_flux/isotonic_all_perturbed_for_uq"

FIRST_EVAL_TIME_STRING="2019-12-24-000000"
LAST_EVAL_TIME_STRING="2020-12-31-235959"

prediction_file_name="${model_dir_name}/isotonic_regression/training/predictions.nc"

python3 -u "${CODE_DIR_NAME}/apply_neural_net.py" \
--input_model_file_name="${model_file_name}" \
--input_example_dir_name="${EXAMPLE_DIR_NAME}" \
--first_time_string="${FIRST_EVAL_TIME_STRING}" \
--last_time_string="${LAST_EVAL_TIME_STRING}" \
--num_bnn_iterations=50 \
--max_ensemble_size=50 \
--output_file_name="${prediction_file_name}"
