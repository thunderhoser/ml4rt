#!/bin/bash

# set -x

model_dir_name=$1
prediction_file_index=$2
prediction_file_index="$((prediction_file_index-1))"

model_file_name="${model_dir_name}/model.h5"
model_dir_name="${model_dir_name}/model"

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_standalone/ml4rt"
EXAMPLE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/shortwave_examples_600days/orig_heights/normalized_not_flux/testing_all_perturbed_for_uq"

FIRST_EVAL_TIME_STRINGS=("2020-01-18-000000" "2020-02-28-000000" "2020-04-09-000000" "2020-05-22-000000" "2020-07-02-000000" "2020-08-12-000000" "2020-09-22-000000" "2020-11-05-000000" "2020-12-16-000000")
LAST_EVAL_TIME_STRINGS=("2020-01-31-235959" "2020-03-12-235959" "2020-04-22-235959" "2020-06-04-235959" "2020-07-15-235959" "2020-08-25-235959" "2020-10-07-235959" "2020-11-18-235959" "2020-12-23-235959")
PATHLESS_OUTPUT_FILE_NAMES=("predictions_part01.nc" "predictions_part02.nc" "predictions_part03.nc" "predictions_part04.nc" "predictions_part05.nc" "predictions_part06.nc" "predictions_part07.nc" "predictions_part08.nc" "predictions_part09.nc")

first_eval_time_string=${FIRST_EVAL_TIME_STRINGS[$prediction_file_index]}
last_eval_time_string=${LAST_EVAL_TIME_STRINGS[$prediction_file_index]}
prediction_file_name="${model_dir_name}/testing_perturbed_for_uq/${PATHLESS_OUTPUT_FILE_NAMES[$prediction_file_index]}"

python3 -u "${CODE_DIR_NAME}/apply_neural_net.py" \
--input_model_file_name="${model_file_name}" \
--input_example_dir_name="${EXAMPLE_DIR_NAME}" \
--first_time_string="${first_eval_time_string}" \
--last_time_string="${last_eval_time_string}" \
--num_bnn_iterations=50 \
--max_ensemble_size=50 \
--output_file_name="${prediction_file_name}"
