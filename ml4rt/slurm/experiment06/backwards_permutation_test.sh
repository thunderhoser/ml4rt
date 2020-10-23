#!/bin/bash

PYTHON_EXE_NAME="/home/ryan.lagerquist/anaconda3/bin/python3.6"
CODE_DIR_NAME="/home/ryan.lagerquist/ml4rt/ml4rt"
TOP_MODEL_DIR_NAME="/home/ryan.lagerquist/ml4rt_project/u_net/dual_weighted_mse_loss/unnorm_target"
EXTENSIONLESS_MODEL_FILE_NAME="model_epoch=115_val-loss=0.428658"

EXAMPLE_FILE_NAME="/home/ryan.lagerquist/ml4rt_project/examples/radiative_transfer_examples_2019.nc"

model_dir_name="${TOP_MODEL_DIR_NAME}/${EXTENSIONLESS_MODEL_FILE_NAME}"
model_file_name="${model_dir_name}.h5"

"${PYTHON_EXE_NAME}" -u "${CODE_DIR_NAME}/scripts/run_permutation.py" \
--input_model_file_name="${model_file_name}" \
--input_example_file_name="${EXAMPLE_FILE_NAME}" \
--num_examples=100000 \
--do_backwards_test=1 \
--shuffle_profiles_together=1 \
--cost_function_name="mse" \
--num_bootstrap_reps=1000 \
--output_file_name="${model_dir_name}/validation/backwards_permutation_test_mse.nc"

"${PYTHON_EXE_NAME}" -u "${CODE_DIR_NAME}/scripts/plot_permutation_results.py" \
--input_file_name="${model_dir_name}/validation/backwards_permutation_test_mse.nc" \
--confidence_level=0.95 \
--output_dir_name="${model_dir_name}/validation/backwards_permutation_test_mse"

"${PYTHON_EXE_NAME}" -u "${CODE_DIR_NAME}/scripts/run_permutation.py" \
--input_model_file_name="${model_file_name}" \
--input_example_file_name="${EXAMPLE_FILE_NAME}" \
--num_examples=100000 \
--do_backwards_test=1 \
--shuffle_profiles_together=1 \
--cost_function_name="dual_weighted_mse" \
--num_bootstrap_reps=1000 \
--output_file_name="${model_dir_name}/validation/backwards_permutation_test_dwmse.nc"

"${PYTHON_EXE_NAME}" -u "${CODE_DIR_NAME}/scripts/plot_permutation_results.py" \
--input_file_name="${model_dir_name}/validation/backwards_permutation_test_dwmse.nc" \
--confidence_level=0.95 \
--output_dir_name="${model_dir_name}/validation/backwards_permutation_test_dwmse"
