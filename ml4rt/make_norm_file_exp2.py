"""Makes normalization file for Paper Experiment 1."""

import os
import sys
import numpy

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
    '2020-10-31-235959', '%Y-%m-%d-%H%M%S'
)

EXAMPLE_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/examples/'
    'assorted1_sites'
)
OUTPUT_FILE_NAME = (
    '{0:s}/learning_examples_20190101-20201031.nc'
).format(EXAMPLE_DIR_NAME)

example_file_names = example_io.find_many_files(
    directory_name=EXAMPLE_DIR_NAME,
    first_time_unix_sec=FIRST_TRAINING_TIME_UNIX_SEC,
    last_time_unix_sec=LAST_TRAINING_TIME_UNIX_SEC,
    raise_error_if_any_missing=True
)

example_dicts = [
    example_io.read_file(netcdf_file_name=f) for f in example_file_names
]
example_dict = example_utils.concat_examples(example_dicts)
del example_dicts

num_examples = len(example_dict[example_utils.EXAMPLE_IDS_KEY])
print(num_examples)

example_dict = example_utils.subset_by_time(
    example_dict=example_dict, first_time_unix_sec=FIRST_TRAINING_TIME_UNIX_SEC,
    last_time_unix_sec=LAST_TRAINING_TIME_UNIX_SEC
)[0]

num_examples = len(example_dict[example_utils.EXAMPLE_IDS_KEY])
print(num_examples)

desired_indices = numpy.linspace(
    0, num_examples - 1, num=num_examples, dtype=int
)
desired_indices = numpy.random.choice(
    desired_indices, size=100000, replace=False
)

example_dict = example_utils.subset_by_index(
    example_dict=example_dict, desired_indices=desired_indices
)

num_examples = len(example_dict[example_utils.EXAMPLE_IDS_KEY])
print(num_examples)

example_io.write_file(
    example_dict=example_dict, netcdf_file_name=OUTPUT_FILE_NAME
)
