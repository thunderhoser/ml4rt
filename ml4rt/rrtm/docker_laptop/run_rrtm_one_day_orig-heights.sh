#!/usr/bin/bash

# Master script for running the goddamn RRTM.
# Argument 1 is the string ID for the Docker container.
# Argument 2 is the date (format "yyyyJJJ", where "JJJ" is the ordinal date from 001...366).
# Argument 3 is the sudo password.

container_id_string=$1
date_string=$2
sudo_password=$3
year_string=${date_string:0:4}

SEPARATOR_STRING="\r\n--------------------------------------------------\r\n"

GITHUB_CODE_DIR_NAME="/home/ralager/ml4rt/ml4rt/rrtm"
TOP_DATA_DIR_NAME_ACTUAL="/home/ralager/condo/swatwork/ralager/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/orig_heights"
TOP_DATA_DIR_NAME_DOCKER="/home/user/data"

cp -v "${GITHUB_CODE_DIR_NAME}/make_rrtm_sw_calc.pro" "${TOP_DATA_DIR_NAME_ACTUAL}/"
cp -v "${GITHUB_CODE_DIR_NAME}/runit_ML_dataset_builder_orig-heights.pro" "${TOP_DATA_DIR_NAME_ACTUAL}/"

actual_gfs_file_name="${TOP_DATA_DIR_NAME_ACTUAL}/profiler_sitesF.ncep_rap.${date_string}.0000.nc"
echo "GFS file for date ${date_string}: ${actual_gfs_file_name}"

data_dir_name_actual="${TOP_DATA_DIR_NAME_ACTUAL}/${date_string}"
data_dir_name_docker="${TOP_DATA_DIR_NAME_DOCKER}/${date_string}"

rm -v "${data_dir_name_actual}/output_file.${year_string}.cdf"
mkdir -p $data_dir_name_actual
cp $actual_gfs_file_name "${data_dir_name_actual}/"

gdl_script_file_name="${data_dir_name_actual}/run_rrtm_${date_string}.gdl"
echo "Writing GDL batch script to: '${gdl_script_file_name}'..."

echo ".compile /home/user/data/make_rrtm_sw_calc" > $gdl_script_file_name
echo ".run /home/user/data/runit_ML_dataset_builder_orig-heights.pro" >> $gdl_script_file_name
echo "runit,${year_string}" >> $gdl_script_file_name
echo "exit" >> $gdl_script_file_name

echo ${sudo_password} | sudo -S -k docker exec ${container_id_string} /bin/sh -c "cd ${data_dir_name_docker}; gdl -e '@run_rrtm_${date_string}.gdl'"

rm -v ${data_dir_name_actual}/core.*
rm -v ${data_dir_name_actual}/TAPE*
rm -v ${data_dir_name_actual}/*RRTM
