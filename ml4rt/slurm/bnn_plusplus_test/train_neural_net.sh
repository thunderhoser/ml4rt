#!/bin/sh

CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_standalone/ml4rt"
TEMPLATE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_models/bnn_plusplus_test/template"
TOP_OUTPUT_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_models/bnn_plusplus_test"

TRAINING_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/shortwave_examples_600days/orig_heights/normalized_with_perturbed_training_data/training_all_perturbed_for_uq"
VALIDATION_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/shortwave_examples_600days/orig_heights/normalized_with_perturbed_training_data/validation_all_perturbed_for_uq"
NORMALIZATION_FILE_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/shortwave_examples_600days/orig_heights/training_all_perturbed_for_uq/learning_examples_for_norm_20180901-20191221.nc"

template_file_name="${TEMPLATE_DIR_NAME}/model.h5"
output_dir_name="${TOP_OUTPUT_DIR_NAME}"
echo $output_dir_name

python3 -u "${CODE_DIR_NAME}/train_neural_net.py" \
--net_type_string="u_net" \
--input_training_dir_name="${TRAINING_DIR_NAME}" \
--input_validation_dir_name="${VALIDATION_DIR_NAME}" \
--input_normalization_file_name="${NORMALIZATION_FILE_NAME}" \
--input_model_file_name="${template_file_name}" \
--output_model_dir_name="${output_dir_name}" \
--use_generator_for_training=0 \
--use_generator_for_validn=0 \
--predictor_names "zenith_angle_radians" "albedo" "aerosol_single_scattering_albedo" "aerosol_asymmetry_param" "pressure_pascals" "temperature_kelvins" "specific_humidity_kg_kg01" "relative_humidity_unitless" "liquid_water_content_kg_m03" "ice_water_content_kg_m03" "liquid_water_path_kg_m02" "ice_water_path_kg_m02" "vapour_path_kg_m02" "upward_liquid_water_path_kg_m02" "upward_ice_water_path_kg_m02" "upward_vapour_path_kg_m02" "liquid_effective_radius_metres" "ice_effective_radius_metres" "o3_mixing_ratio_kg_kg01" "co2_concentration_ppmv" "ch4_concentration_ppmv" "n2o_concentration_ppmv" "aerosol_extinction_metres01" "height_m_agl" "height_thickness_metres" "pressure_thickness_pascals" \
--target_names "shortwave_heating_rate_k_day01" "shortwave_surface_down_flux_w_m02" "shortwave_toa_up_flux_w_m02" \
--heights_m_agl 21 44 68 93 120 149 179 212 246 282 321 361 405 450 499 550 604 661 722 785 853 924 999 1078 1161 1249 1342 1439 1542 1649 1762 1881 2005 2136 2272 2415 2564 2720 2882 3051 3228 3411 3601 3798 4002 4214 4433 4659 4892 5132 5379 5633 5894 6162 6436 6716 7003 7296 7594 7899 8208 8523 8842 9166 9494 9827 10164 10505 10849 11198 11550 11906 12266 12630 12997 13368 13744 14123 14506 14895 15287 15686 16090 16501 16920 17350 17791 18246 18717 19205 19715 20249 20809 21400 22022 22681 23379 24119 24903 25736 26619 27558 28556 29616 30743 31940 33211 34566 36012 37560 39218 40990 42882 44899 47042 49299 51644 54067 56552 59089 61677 64314 67001 69747 72521 75256 77803 \
--multiply_preds_by_layer_thickness=0 \
--first_training_time_string="2018-09-01-000000" \
--last_training_time_string="2019-12-21-235959" \
--first_validn_time_string="2019-12-24-000000" \
--last_validn_time_string="2020-12-31-235959" \
--uniformize=1 \
--vector_target_norm_type_string="" \
--scalar_target_norm_type_string="" \
--num_examples_per_batch=724 \
--num_epochs=1000 \
--num_training_batches_per_epoch=1000 \
--num_validn_batches_per_epoch=1000 \
--plateau_lr_multiplier=0.6 \
--num_deep_supervision_layers=0