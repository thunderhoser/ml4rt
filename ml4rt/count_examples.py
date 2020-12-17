"""Counts examples in each set."""

import os
import sys

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import example_io
import example_utils

FIRST_TRAINING_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '20190101', '%Y%m%d'
)
LAST_TRAINING_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '20201231', '%Y%m%d'
)
FIRST_ISOTONIC_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '20180101', '%Y%m%d'
)
LAST_ISOTONIC_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2018-12-24-235959', '%Y-%m-%d-%H%M%S'
)
FIRST_TESTING_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '20170101', '%Y%m%d'
)
LAST_TESTING_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2017-12-24-235959', '%Y-%m-%d-%H%M%S'
)

ASSORTED1_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/examples/'
    'assorted1_sites'
)
ASSORTED2_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/examples/'
    'assorted2_sites'
)

example_file_names = example_io.find_many_files(
    directory_name=ASSORTED1_DIR_NAME,
    first_time_unix_sec=FIRST_TRAINING_TIME_UNIX_SEC,
    last_time_unix_sec=LAST_TRAINING_TIME_UNIX_SEC,
    raise_error_if_any_missing=True
)

num_examples = 0

for this_file_name in example_file_names:
    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_example_dict = example_io.read_file(this_file_name)

    this_example_dict = example_utils.subset_by_time(
        example_dict=this_example_dict,
        first_time_unix_sec=FIRST_TRAINING_TIME_UNIX_SEC,
        last_time_unix_sec=LAST_TRAINING_TIME_UNIX_SEC
    )[0]

    num_examples += len(this_example_dict[example_utils.VALID_TIMES_KEY])

print('Number of U-net-training examples = {0:d}'.format(num_examples))

example_file_names = example_io.find_many_files(
    directory_name=ASSORTED1_DIR_NAME,
    first_time_unix_sec=FIRST_ISOTONIC_TIME_UNIX_SEC,
    last_time_unix_sec=LAST_ISOTONIC_TIME_UNIX_SEC,
    raise_error_if_any_missing=True
)

num_examples = 0

for this_file_name in example_file_names:
    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_example_dict = example_io.read_file(this_file_name)

    this_example_dict = example_utils.subset_by_time(
        example_dict=this_example_dict,
        first_time_unix_sec=FIRST_ISOTONIC_TIME_UNIX_SEC,
        last_time_unix_sec=LAST_ISOTONIC_TIME_UNIX_SEC
    )[0]

    num_examples += len(this_example_dict[example_utils.VALID_TIMES_KEY])

print('Number of IR-training examples = {0:d}'.format(num_examples))

example_file_names = example_io.find_many_files(
    directory_name=ASSORTED1_DIR_NAME,
    first_time_unix_sec=FIRST_TESTING_TIME_UNIX_SEC,
    last_time_unix_sec=LAST_TESTING_TIME_UNIX_SEC,
    raise_error_if_any_missing=True
)

num_examples = 0

for this_file_name in example_file_names:
    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_example_dict = example_io.read_file(this_file_name)

    this_example_dict = example_utils.subset_by_time(
        example_dict=this_example_dict,
        first_time_unix_sec=FIRST_TESTING_TIME_UNIX_SEC,
        last_time_unix_sec=LAST_TESTING_TIME_UNIX_SEC
    )[0]

    num_examples += len(this_example_dict[example_utils.VALID_TIMES_KEY])

print('Number of validation examples = {0:d}'.format(num_examples))

example_file_names = example_io.find_many_files(
    directory_name=ASSORTED2_DIR_NAME,
    first_time_unix_sec=FIRST_TESTING_TIME_UNIX_SEC,
    last_time_unix_sec=LAST_TESTING_TIME_UNIX_SEC,
    raise_error_if_any_missing=True
)

num_examples = 0

for this_file_name in example_file_names:
    print('Reading data from: "{0:s}"...'.format(this_file_name))
    this_example_dict = example_io.read_file(this_file_name)

    this_example_dict = example_utils.subset_by_time(
        example_dict=this_example_dict,
        first_time_unix_sec=FIRST_TESTING_TIME_UNIX_SEC,
        last_time_unix_sec=LAST_TESTING_TIME_UNIX_SEC
    )[0]

    num_examples += len(this_example_dict[example_utils.VALID_TIMES_KEY])

print('Number of testing examples = {0:d}'.format(num_examples))
