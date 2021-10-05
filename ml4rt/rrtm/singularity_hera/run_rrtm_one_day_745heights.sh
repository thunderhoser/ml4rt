#!/usr/bin/bash

# Master script for running the goddamn RRTM.
# Argument 1 is the date (format "yyyyJJJ", where "JJJ" is the ordinal date from 001...366).

SEPARATOR_STRING="\r\n--------------------------------------------------\r\n"

SINGULARITY_CONTAINER_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/rrtm_docker/rrtm_hyperres_take2.sif"
TOP_DATA_DIR_NAME_ACTUAL="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/new_heights"
TOP_DATA_DIR_NAME_DOCKER="/home/user/data"

date_string=$1
year_string=${date_string:0:4}

actual_gfs_file_name="${TOP_DATA_DIR_NAME_ACTUAL}/profiler_sitesF.ncep_rap.${date_string}.0000.nc"
echo "GFS file for date ${date_string}: ${actual_gfs_file_name}"

data_dir_name_actual="${TOP_DATA_DIR_NAME_ACTUAL}/${date_string}"
data_dir_name_docker="${TOP_DATA_DIR_NAME_DOCKER}/${date_string}"

rm -rfv $data_dir_name_actual
mkdir $data_dir_name_actual
cp $actual_gfs_file_name "${data_dir_name_actual}/"

gdl_script_file_name="${data_dir_name_actual}/run_rrtm_${date_string}.gdl"
echo "Writing GDL batch script to: '${gdl_script_file_name}'..."

echo ".compile /home/user/data/make_rrtm_sw_calc" > $gdl_script_file_name
echo ".run /home/user/data/runit_ML_dataset_builder_745heights.pro" >> $gdl_script_file_name
echo "runit,${year_string}" >> $gdl_script_file_name
echo "exit" >> $gdl_script_file_name

singularity exec -C -B "${TOP_DATA_DIR_NAME_ACTUAL}":"${TOP_DATA_DIR_NAME_DOCKER}" "${SINGULARITY_CONTAINER_NAME}" /bin/sh -c "cd ${data_dir_name_docker}; gdl -e '@run_rrtm_${date_string}.gdl'"
