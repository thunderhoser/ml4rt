"""Investigates the distribution of longwave heating rates."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import example_io
import example_utils

FIRST_TIME_UNIX_SEC = time_conversion.string_to_unix_sec('20000101', '%Y%m%d')
LAST_TIME_UNIX_SEC = time_conversion.string_to_unix_sec('20300101', '%Y%m%d')

PERCENTILE_LEVELS = numpy.concatenate((
    numpy.linspace(0, 1, num=101, dtype=float),
    numpy.linspace(99, 100, num=101, dtype=float)
))

EXAMPLE_DIRS_ARG_NAME = 'input_example_dir_names'
EXAMPLE_DIRS_HELP_STRING = (
    'List of directory names where example files will be found.  The files will'
    ' be found by `example_io.find_file` and read by `example_io.read_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIRS_ARG_NAME, type=str, nargs='+', required=True,
    help=EXAMPLE_DIRS_HELP_STRING
)


def _run(example_dir_names):
    """Investigates the distribution of longwave heating rates.

    This is effectively the main method.

    :param example_dir_names: See documentation at top of file.
    :raises: ValueError: if no example files are found.
    """

    example_file_names = []

    for this_dir_name in example_dir_names:
        example_file_names += example_io.find_many_files(
            directory_name=this_dir_name,
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC,
            raise_error_if_any_missing=False, raise_error_if_all_missing=False
        )

    if len(example_file_names) == 0:
        error_string = (
            'Cannot find any example files in the following directories:\n{0:s}'
        ).format(str(example_dir_names))

        raise ValueError(error_string)

    example_file_names.sort()
    heating_rate_matrix_k_day01 = numpy.array([])

    for this_file_name in example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(
            netcdf_file_name=this_file_name, exclude_summit_greenland=False,
            max_shortwave_heating_k_day01=numpy.inf
        )
        this_heating_rate_matrix_k_day01 = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.LONGWAVE_HEATING_RATE_NAME
        )

        if heating_rate_matrix_k_day01.size == 0:
            heating_rate_matrix_k_day01 = this_heating_rate_matrix_k_day01 + 0.
        else:
            heating_rate_matrix_k_day01 = numpy.concatenate(
                (heating_rate_matrix_k_day01, this_heating_rate_matrix_k_day01),
                axis=0
            )

    print('\n')

    heating_rate_percentiles_k_day01 = numpy.percentile(
        a=numpy.ravel(heating_rate_matrix_k_day01), q=PERCENTILE_LEVELS
    )

    for i in range(len(heating_rate_percentiles_k_day01)):
        print((
            '{0:.2f}th-percentile longwave heating rate = {1:.4f} K day^-1'
        ).format(
            PERCENTILE_LEVELS[i], heating_rate_percentiles_k_day01[i]
        ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_dir_names=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIRS_ARG_NAME)
    )
