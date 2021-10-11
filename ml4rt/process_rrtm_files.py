"""Converts daily RRTM files to example files."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import rrtm_io
import example_io
import example_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

ORIG_DATE_FORMAT = time_conversion.SPC_DATE_FORMAT
RRTM_DATE_FORMAT = '%Y%J'

INPUT_DIR_ARG_NAME = 'input_rrtm_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
DUMMY_HEIGHTS_ARG_NAME = 'dummy_heights_m_agl'
OUTPUT_FILE_ARG_NAME = 'output_example_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level directory with daily RRTM files.  Files therein will be '
    'read by `rrtm_io.read_file`.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  This script will convert RRTM files for the '
    'period `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

DUMMY_HEIGHTS_HELP_STRING = (
    'List of dummy heights (metres above ground level).  These will be used '
    'only if RRTM files contain data on different height grids.  If the RRTM '
    'files contain data all on the same height grid, leave this argument alone.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `example_io.write_file`).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DUMMY_HEIGHTS_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=DUMMY_HEIGHTS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _remove_duplicate_examples(example_dict):
    """Removes duplicate examples from dictionary.

    :param example_dict: Dictionary in format created by `example_io.read_file`.
    :return: example_dict: Same but maybe with fewer examples.
    """

    example_id_strings = example_dict[example_utils.EXAMPLE_IDS_KEY]
    unique_indices = numpy.unique(
        numpy.array(example_id_strings), return_index=True
    )[1]

    print('{0:d} of {1:d} examples are unique!'.format(
        len(unique_indices), len(example_id_strings)
    ))

    return example_utils.subset_by_index(
        example_dict=example_dict, desired_indices=unique_indices
    )


def _run(top_rrtm_dir_name, first_date_string, last_date_string,
         dummy_heights_m_agl, output_file_name):
    """Converts daily RRTM files to example files.

    This is effectively the main method.

    :param top_rrtm_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param dummy_heights_m_agl: Same.
    :param output_file_name: Same.
    :raises: ValueError: if no RRTM files can be found.
    """

    if len(dummy_heights_m_agl) == 1 and dummy_heights_m_agl[0] < 0:
        dummy_heights_m_agl = None

    date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )
    dates_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(d, ORIG_DATE_FORMAT)
        for d in date_strings
    ], dtype=int)
    date_strings = [
        time_conversion.unix_sec_to_string(d, RRTM_DATE_FORMAT)
        for d in dates_unix_sec
    ]

    rrtm_file_names = []

    for this_date_string in date_strings:
        this_file_name = '{0:s}/{1:s}/output_file.{2:s}.cdf'.format(
            top_rrtm_dir_name, this_date_string, this_date_string[:4]
        )

        if not os.path.isfile(this_file_name):
            continue

        rrtm_file_names.append(this_file_name)

    if len(rrtm_file_names) == 0:
        error_string = (
            'Cannot find any RRTM files for days {0:s} to {1:s} in directory: '
            '"{2:s}"'
        ).format(date_strings[0], date_strings[1], top_rrtm_dir_name)

        raise ValueError(error_string)

    example_dicts = []

    for this_file_name in rrtm_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = rrtm_io.read_file(
            netcdf_file_name=this_file_name, allow_bad_values=True,
            dummy_heights_m_agl=dummy_heights_m_agl
        )

        example_dicts.append(this_example_dict)

    example_dict = example_utils.concat_examples(example_dicts)
    del example_dicts

    print(SEPARATOR_STRING)
    example_dict = _remove_duplicate_examples(example_dict)

    print('Writing {0:d} examples to file: "{1:s}"...'.format(
        len(example_dict[example_utils.VALID_TIMES_KEY]),
        output_file_name
    ))
    example_io.write_file(
        example_dict=example_dict, netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        top_rrtm_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        dummy_heights_m_agl=getattr(INPUT_ARG_OBJECT, DUMMY_HEIGHTS_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
