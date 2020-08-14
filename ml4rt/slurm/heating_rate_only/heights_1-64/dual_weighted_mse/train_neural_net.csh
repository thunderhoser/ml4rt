#!/bin/tcsh

module load cuda/10.1
source /scratch2/BMC/gsd-hpcs/Jebb.Q.Stewart/conda3.7/etc/profile.d/conda.csh
conda activate base

set CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_standalone/ml4rt"
set INPUT_MODEL_FILE_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_models/templates/u_net_64x1.h5"
set OUTPUT_MODEL_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_models/heating_rate_only/heights_1-64/dual_weighted_mse"

python3 -u "${CODE_DIR_NAME}/train_neural_net.py" \
--net_type_string="u_net" \
--input_model_file_name="${INPUT_MODEL_FILE_NAME}" \
--output_model_dir_name="${OUTPUT_MODEL_DIR_NAME}" \
--use_generator_for_training=0 \
--use_generator_for_validn=0 \
--target_names "shortwave_heating_rate_k_day01" \
--heights_m_agl 10 20 40 60 80 100 120 140 160 180 200 225 250 275 300 350 400 450 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000 2200 2400 2600 2800 3000 3200 3400 3600 3800 4000 4200 4400 4600 4800 5000 5500 6000 6500 7000 8000 9000 10000 11000 12000 13000 14000 15000 18000 20000 22000 \
--omit_heating_rate=0 \
--vector_target_norm_type_string='' \
--scalar_target_norm_type_string=''
