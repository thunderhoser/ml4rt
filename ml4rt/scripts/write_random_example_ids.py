"""Writes random example IDs to file."""

import argparse
import numpy
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from ml4rt.io import example_io
from ml4rt.utils import example_utils

TIME_FORMAT = '%Y-%m-%d-%H%M%S'

EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_FILE_ARG_NAME = 'output_id_file_name'

EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with example files.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)

TIME_HELP_STRING = (
    'Will draw random example IDs from the time period `{0:s}`...`{1:s}`.  '
    'Both times must be in format "yyyy-mm-dd-HHMMSS".'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

NUM_EXAMPLES_HELP_STRING = 'Number of random example IDs to draw.'
OUTPUT_FILE_HELP_STRING = (
    'Path to output (NetCDF) file.  Random example IDs drawn will be written '
    'here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(example_dir_name, first_time_string, last_time_string, num_examples,
         output_file_name):
    """Writes random example IDs to file.

    This is effectively the main method.

    :param example_dir_name: See documentation at top of file.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_examples: Same.
    :param output_file_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, TIME_FORMAT
    )

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )

    example_id_strings = []

    for this_file_name in example_file_names:
        print('Reading IDs from: "{0:s}"...'.format(this_file_name))
        example_id_strings += example_io.read_file(this_file_name)[
            example_utils.EXAMPLE_IDS_KEY
        ]

    if num_examples < len(example_id_strings):
        example_id_strings = numpy.random.choice(
            numpy.array(example_id_strings), size=num_examples, replace=False
        ).tolist()

    print('Writing IDs to: "{0:s}"...'.format(output_file_name))

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    dataset_object = netCDF4.Dataset(output_file_name, 'w', format='NETCDF4')

    num_examples = len(example_id_strings)
    num_example_id_chars = numpy.max(numpy.array([
        len(id) for id in example_id_strings
    ]))

    dataset_object.createDimension(
        example_io.EXAMPLE_DIMENSION_KEY, num_examples
    )
    dataset_object.createDimension(
        example_io.EXAMPLE_ID_CHAR_DIM_KEY, num_example_id_chars
    )

    # Add example IDs.
    this_string_format = 'S{0:d}'.format(num_example_id_chars)
    example_ids_char_array = netCDF4.stringtochar(numpy.array(
        example_id_strings, dtype=this_string_format
    ))

    dataset_object.createVariable(
        example_utils.EXAMPLE_IDS_KEY, datatype='S1',
        dimensions=(
            example_io.EXAMPLE_DIMENSION_KEY,
            example_io.EXAMPLE_ID_CHAR_DIM_KEY
        )
    )
    dataset_object.variables[example_utils.EXAMPLE_IDS_KEY][:] = numpy.array(
        example_ids_char_array
    )

    dataset_object.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
