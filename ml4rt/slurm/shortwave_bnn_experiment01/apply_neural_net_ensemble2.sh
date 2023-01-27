#!/bin/bash

# set -x

model_dir_name=$1
model_file_name="${model_dir_name}/model.h5"
model_dir_name="${model_dir_name}/model"

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_standalone/ml4rt"
EXAMPLE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/shortwave_examples_600days/orig_heights/normalized_with_perturbed_training_data"

FIRST_EVAL_TIME_STRINGS=("2020-01-02-000000" "2020-02-12-000000" "2020-03-24-000000" "2020-05-05-000000" "2020-06-16-000000" "2020-07-27-000000" "2020-09-06-000000" "2020-10-19-000000" "2020-11-30-000000")
LAST_EVAL_TIME_STRINGS=("2020-01-15-235959" "2020-02-25-235959" "2020-04-06-235959" "2020-05-18-235959" "2020-06-29-235959" "2020-08-09-235959" "2020-09-19-235959" "2020-11-02-235959" "2020-12-13-235959")
PATHLESS_OUTPUT_FILE_NAMES=("predictions_part01.nc" "predictions_part02.nc" "predictions_part03.nc" "predictions_part04.nc" "predictions_part05.nc" "predictions_part06.nc" "predictions_part07.nc" "predictions_part08.nc" "predictions_part09.nc")

for i in ${!FIRST_EVAL_TIME_STRINGS[@]}; do
    first_eval_time_string=${FIRST_EVAL_TIME_STRINGS[$i]}
    last_eval_time_string=${LAST_EVAL_TIME_STRINGS[$i]}
    prediction_file_name="${model_dir_name}/validation_perturbed_for_uq/${PATHLESS_OUTPUT_FILE_NAMES[$i]}"
    
    python3 -u "${CODE_DIR_NAME}/apply_neural_net.py" \
    --input_model_file_name="${model_file_name}" \
    --input_example_dir_name="${EXAMPLE_DIR_NAME}" \
    --first_time_string="${first_eval_time_string}" \
    --last_time_string="${last_eval_time_string}" \
    --num_bnn_iterations=2 \
    --max_ensemble_size=200 \
    --output_file_name="${prediction_file_name}"
done
