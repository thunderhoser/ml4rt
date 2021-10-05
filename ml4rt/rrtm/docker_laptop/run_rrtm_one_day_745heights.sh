#!/usr/bin/bash

# Master script for running the goddamn RRTM.
# Argument 1 is the string ID for the Docker container.
# Argument 2 is the date (format "yyyyJJJ", where "JJJ" is the ordinal date from 001...366).
# Argument 3 is the sudo password.

SEPARATOR_STRING="\r\n--------------------------------------------------\r\n"

TOP_DATA_DIR_NAME_ACTUAL="/home/ralager/rrtm_docker"
TOP_DATA_DIR_NAME_DOCKER="/home/user/data"

container_id_string=$1
date_string=$2
sudo_password=$3
year_string=${date_string:0:4}

shopt -s nullglob
actual_gfs_file_name="${TOP_DATA_DIR_NAME_ACTUAL}/gfs_745heights/profiler_sitesF.ncep_rap.${date_string}.0000.nc"

echo "GFS file for date ${date_string}: ${actual_gfs_file_name}"

data_dir_name_actual="${TOP_DATA_DIR_NAME_ACTUAL}/gfs_745heights/${date_string}"
data_dir_name_docker="${TOP_DATA_DIR_NAME_DOCKER}/gfs_745heights/${date_string}"
echo ${sudo_password} | sudo -S -k rm -rfv $data_dir_name_actual

mkdir $data_dir_name_actual
cp $actual_gfs_file_name "${data_dir_name_actual}/"

gdl_script_file_name="${data_dir_name_actual}/run_rrtm_${date_string}.gdl"
echo "Writing GDL batch script to: '${gdl_script_file_name}'..."

echo ".compile /home/user/data/make_rrtm_sw_calc" > $gdl_script_file_name
echo ".run /home/user/data/runit_ML_dataset_builder_745heights.pro" >> $gdl_script_file_name
echo "runit,${year_string}" >> $gdl_script_file_name
echo "exit" >> $gdl_script_file_name

echo ${sudo_password} | sudo -S -k docker exec ${container_id_string} /bin/sh -c "cd ${data_dir_name_docker}; gdl -e '@run_rrtm_${date_string}.gdl'"
