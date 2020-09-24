"""Reprocesses example files with new example IDs."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import example_io
import example_utils

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_YEAR_ARG_NAME = 'first_year'
LAST_YEAR_ARG_NAME = 'last_year'
OUTPUT_DIR_ARG_NAME = 'output_example_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of directory with original example files.  Files therein will be '
    'found by `example_io.find_file` and read by `example_io.read_file`.'
)
YEAR_HELP_STRING = (
    'Will update example files for all years from `{0:s}` to `{1:s}`.'
).format(FIRST_YEAR_ARG_NAME, LAST_YEAR_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  New example files will be written here by '
    '`example_io.write_file`, with file names determined by '
    '`example_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_example_dir_name, first_year, last_year,
         output_example_dir_name):
    """Reprocesses example files with new example IDs.

    This is effectively the main method.

    :param input_example_dir_name: See documentation at top of file.
    :param first_year: Same.
    :param last_year: Same.
    :param output_example_dir_name: Same.
    """

    error_checking.assert_is_geq(last_year, first_year)
    years = numpy.linspace(
        first_year, last_year, num=last_year - first_year + 1, dtype=int
    )

    input_file_names = [
        example_io.find_file(
            directory_name=input_example_dir_name,
            year=y, raise_error_if_missing=True
        )
        for y in years
    ]

    output_file_names = [
        example_io.find_file(
            directory_name=output_example_dir_name,
            year=y, raise_error_if_missing=False
        )
        for y in years
    ]

    this_input_file_name = '{0:s}/learning_examples_20170101-2018224.nc'.format(
        input_example_dir_name
    )

    if os.path.isfile(this_input_file_name):
        input_file_names.append(this_input_file_name)

        this_output_file_name = (
            '{0:s}/learning_examples_20170101-2018224.nc'
        ).format(output_example_dir_name)

        output_file_names.append(this_output_file_name)

    num_files = len(input_file_names)

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        this_example_dict = example_io.read_file(input_file_names[i])

        this_example_dict[example_utils.EXAMPLE_IDS_KEY] = (
            example_utils.create_example_ids(this_example_dict)
        )
        print(this_example_dict[example_utils.EXAMPLE_IDS_KEY][-1])

        print('Writing data with new example IDs to: "{0:s}"...\n'.format(
            output_file_names[i]
        ))
        example_io.write_file(
            example_dict=this_example_dict,
            netcdf_file_name=output_file_names[i]
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_example_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_year=getattr(INPUT_ARG_OBJECT, FIRST_YEAR_ARG_NAME),
        last_year=getattr(INPUT_ARG_OBJECT, LAST_YEAR_ARG_NAME),
        output_example_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
