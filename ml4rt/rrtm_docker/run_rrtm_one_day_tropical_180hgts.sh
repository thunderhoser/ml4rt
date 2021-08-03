#!/usr/bin/bash

# Master script for running the goddamn RRTM.
# Argument 1 is the string ID for the Docker container.
# Argument 2 is the date (format "yyyyJJJ", where "JJJ" is the ordinal date from 001...366).
# Argument 3 is the sudo password.

SEPARATOR_STRING="\r\n--------------------------------------------------\r\n"

TOP_DATA_DIR_NAME_ACTUAL="/home/ralager/rrtm_docker"
TOP_DATA_DIR_NAME_DOCKER="/home/user/data"
NEW_HEIGHTS_ONE_STRING_M_AGL="10 17 23 30 36 43 49 56 62 69 76 82 89 95 102 108 115 121 128 134 141 148 154 161 167 174 180 187 193 200 208 217 225 233 242 250 258 267 275 283 292 300 317 333 350 367 383 400 417 433 450 467 483 500 533 567 600 633 667 700 733 767 800 833 867 900 933 967 1000 1033 1067 1100 1133 1167 1200 1233 1267 1300 1333 1367 1400 1433 1467 1500 1533 1567 1600 1633 1667 1700 1733 1767 1800 1833 1867 1900 1933 1967 2000 2067 2133 2200 2267 2333 2400 2467 2533 2600 2667 2733 2800 2867 2933 3000 3067 3133 3200 3267 3333 3400 3467 3533 3600 3667 3733 3800 3867 3933 4000 4067 4133 4200 4267 4333 4400 4467 4533 4600 4667 4733 4800 4867 4933 5000 5167 5333 5500 5667 5833 6000 6167 6333 6500 6667 6833 7000 7333 7667 8000 8333 8667 9000 9333 9667 10000 10333 10667 11000 11333 11667 12000 12333 12667 13000 13333 13667 14000 14333 14667 15000"

container_id_string=$1
date_string=$2
sudo_password=$3
year_string=${date_string:0:4}

shopt -s nullglob
actual_rap_file_names=${TOP_DATA_DIR_NAME_ACTUAL}/profiler_sites*${date_string}*00.nc
actual_rap_file_names_one_string=$(IFS=" " ; echo ${actual_rap_file_names[*]})

echo "Found RAP files for date ${date_string}: ${actual_rap_file_names_one_string}"

data_dir_name_actual="${TOP_DATA_DIR_NAME_ACTUAL}/180heights/${date_string}"
data_dir_name_docker="${TOP_DATA_DIR_NAME_DOCKER}/180heights/${date_string}"
echo ${sudo_password} | sudo -S -k rm -rfv $data_dir_name_actual

/home/ralager/anaconda3/bin/python3.7 /home/ralager/ml4rt/ml4rt/scripts/interp_rap_profiles.py \
--input_rap_file_names ${actual_rap_file_names_one_string} \
--new_heights_m_agl ${NEW_HEIGHTS_ONE_STRING_M_AGL} \
--output_dir_name="${data_dir_name_actual}"

ret=$?
if [ $ret -ne 0 ]; then
     exit 1
fi

gdl_script_file_name="${data_dir_name_actual}/run_rrtm_${date_string}.gdl"
echo "Writing GDL batch script to: '${gdl_script_file_name}'..."

echo ".run /home/user/data/runit_ML_dataset_builder_tropical-sites.pro" > $gdl_script_file_name
echo "runit,${year_string}" >> $gdl_script_file_name
echo "exit" >> $gdl_script_file_name

echo ${sudo_password} | sudo -S -k docker exec ${container_id_string} /bin/sh -c "cd ${data_dir_name_docker}; gdl -e '@run_rrtm_${date_string}.gdl'"
