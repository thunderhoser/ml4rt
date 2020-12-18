"""Creates testing files.

USE ONCE AND DESTROY.
"""

import os
import sys

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import example_io
import example_utils

FIRST_TESTING_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '20170101', '%Y%m%d'
)
LAST_TESTING_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2017-12-24-235959', '%Y-%m-%d-%H%M%S'
)

TROPICAL_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/examples/'
    'tropical_sites'
)
ASSORTED2_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/examples/'
    'assorted2_sites'
)
OUTPUT_TROPICAL_FILE_NAME = (
    '{0:s}/learning_examples_20170101-20171224.nc'.format(TROPICAL_DIR_NAME)
)
OUTPUT_ASSORTED2_FILE_NAME = (
    '{0:s}/learning_examples_20170101-20171224.nc'.format(ASSORTED2_DIR_NAME)
)

tropical_2017_file_name = example_io.find_file(
    directory_name=TROPICAL_DIR_NAME, year=2017, raise_error_if_missing=True
)
assorted2_2017_file_name = example_io.find_file(
    directory_name=ASSORTED2_DIR_NAME, year=2017, raise_error_if_missing=True
)

print('Reading data from: "{0:s}"...'.format(tropical_2017_file_name))
tropical_2017_example_dict = example_io.read_file(tropical_2017_file_name)
print(len(tropical_2017_example_dict[example_utils.VALID_TIMES_KEY]))

tropical_2017_example_dict = example_utils.subset_by_time(
    example_dict=tropical_2017_example_dict,
    first_time_unix_sec=FIRST_TESTING_TIME_UNIX_SEC,
    last_time_unix_sec=LAST_TESTING_TIME_UNIX_SEC
)[0]

print(len(tropical_2017_example_dict[example_utils.VALID_TIMES_KEY]))

print('Writing data to: "{0:s}"...'.format(OUTPUT_TROPICAL_FILE_NAME))
example_io.write_file(
    example_dict=tropical_2017_example_dict,
    netcdf_file_name=OUTPUT_TROPICAL_FILE_NAME
)

print('Reading data from: "{0:s}"...'.format(assorted2_2017_file_name))
assorted2_2017_example_dict = example_io.read_file(assorted2_2017_file_name)
print(len(assorted2_2017_example_dict[example_utils.VALID_TIMES_KEY]))

assorted2_2017_example_dict = example_utils.subset_by_time(
    example_dict=assorted2_2017_example_dict,
    first_time_unix_sec=FIRST_TESTING_TIME_UNIX_SEC,
    last_time_unix_sec=LAST_TESTING_TIME_UNIX_SEC
)[0]

print(len(assorted2_2017_example_dict[example_utils.VALID_TIMES_KEY]))

print('Writing data to: "{0:s}"...'.format(OUTPUT_ASSORTED2_FILE_NAME))
example_io.write_file(
    example_dict=assorted2_2017_example_dict,
    netcdf_file_name=OUTPUT_ASSORTED2_FILE_NAME
)
