#!/usr/bin/bash

# Master script for running the longwave RRTM.
# Argument 1 is the date (format "yyyyJJJ", where "JJJ" is the ordinal date from 001...366).

date_string=$1
year_string=${date_string:0:4}

SEPARATOR_STRING="\r\n--------------------------------------------------\r\n"

GITHUB_CODE_DIR_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_standalone/ml4rt/rrtm"
SINGULARITY_CONTAINER_NAME="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/rrtm_docker/rrtm_hyperres_take2.sif"
TOP_DATA_DIR_NAME_ACTUAL="/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/gfs_data/more_sites/processed/orig_heights"
TOP_DATA_DIR_NAME_DOCKER="/home/user/data"

cp -v "${GITHUB_CODE_DIR_NAME}/make_rrtm_lw_calc.pro" "${TOP_DATA_DIR_NAME_ACTUAL}/"
cp -v "${GITHUB_CODE_DIR_NAME}/run_rrtm_lw_orig_heights.pro" "${TOP_DATA_DIR_NAME_ACTUAL}/"

actual_gfs_file_name="${TOP_DATA_DIR_NAME_ACTUAL}/profiler_sitesF.ncep_rap.${date_string}.0000.nc"
echo "GFS file for date ${date_string}: ${actual_gfs_file_name}"

data_dir_name_actual="${TOP_DATA_DIR_NAME_ACTUAL}/${date_string}"
data_dir_name_docker="${TOP_DATA_DIR_NAME_DOCKER}/${date_string}"

rm -v "${data_dir_name_actual}/output_file_lw.${year_string}.cdf"
mkdir -p $data_dir_name_actual
cp $actual_gfs_file_name "${data_dir_name_actual}/"

gdl_script_file_name="${data_dir_name_actual}/run_rrtm_lw_${date_string}.gdl"
echo "Writing GDL batch script to: '${gdl_script_file_name}'..."

echo ".compile /home/user/data/make_rrtm_lw_calc" > $gdl_script_file_name
echo ".run /home/user/data/run_rrtm_lw_orig_heights.pro" >> $gdl_script_file_name
echo "runit,${year_string}" >> $gdl_script_file_name
echo "exit" >> $gdl_script_file_name

singularity exec -C -B "${TOP_DATA_DIR_NAME_ACTUAL}":"${TOP_DATA_DIR_NAME_DOCKER}" "${SINGULARITY_CONTAINER_NAME}" /bin/sh -c "cd ${data_dir_name_docker}; gdl -e '@run_rrtm_lw_${date_string}.gdl'"

rm -v ${data_dir_name_actual}/core.*
rm -v ${data_dir_name_actual}/TAPE*
rm -v ${data_dir_name_actual}/*RRTM
