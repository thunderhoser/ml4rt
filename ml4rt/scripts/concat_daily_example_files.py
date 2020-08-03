"""Concatenates daily example files into yearly example files."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DAYS_TO_SECONDS = 86400
LATLNG_TOLERANCE_DEG = 0.001

INPUT_DAILY_DIR_ARG_NAME = 'input_daily_example_dir_name'
INPUT_YEARLY_DIR_ARG_NAME = 'input_yearly_example_dir_name'
FIRST_YEAR_ARG_NAME = 'first_year'
LAST_YEAR_ARG_NAME = 'last_year'
OUTPUT_YEARLY_DIR_ARG_NAME = 'output_yearly_example_dir_name'

INPUT_DAILY_DIR_HELP_STRING = (
    'Name of top-level directory with daily example files (containing profiles '
    'with RAP predictors and RRTM outputs).  Files therein will be read by '
    '`example_io.read_file`.'
)
INPUT_YEARLY_DIR_HELP_STRING = (
    'Name of directory with yearly example files.  Files therein will be found '
    'by `example_io.find_file` and read by `example_io.read_file`.  These files'
    ' will be concatenated with daily files to create yearly output files.  If '
    'there are no yearly input files, leave this argument alone.'
)
YEAR_HELP_STRING = (
    'Will create yearly output files for all years from `{0:s}` to `{1:s}`.'
).format(FIRST_YEAR_ARG_NAME, LAST_YEAR_ARG_NAME)

OUTPUT_YEARLY_DIR_HELP_STRING = (
    'Name of output directory.  Yearly files will be written here by '
    '`example_io.write_file`, with file names determined by '
    '`example_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DAILY_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DAILY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_YEARLY_DIR_ARG_NAME, type=str, required=False, default='',
    help=INPUT_YEARLY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_YEARLY_DIR_ARG_NAME, type=str, required=False, default='',
    help=OUTPUT_YEARLY_DIR_HELP_STRING
)


def _find_daily_files(daily_example_dir_name, year):
    """Finds daily example files for the given year.

    :param daily_example_dir_name: See documentation at top of file.
    :param year: Year (integer).
    :return: example_file_names: 1-D list of file paths.
    :raises: ValueError: if no daily files can be found for the given year.
    """

    first_time_unix_sec, last_time_unix_sec = (
        time_conversion.first_and_last_times_in_year(year)
    )
    num_seconds_in_year = last_time_unix_sec - first_time_unix_sec + 1
    num_days_in_year = int(numpy.round(
        float(num_seconds_in_year) / DAYS_TO_SECONDS
    ))

    day_indices = numpy.linspace(
        1, num_days_in_year, num=num_days_in_year, dtype=int
    )
    example_file_names = []

    for i in day_indices:
        this_file_name = '{0:s}/{1:04d}{2:03d}/output_file.{1:04d}.cdf'.format(
            daily_example_dir_name, year, i
        )

        if not os.path.isfile(this_file_name):
            continue

        example_file_names.append(this_file_name)

    if len(example_file_names) > 0:
        return example_file_names

    error_string = (
        'Cannot find daily examples for year {0:d} in directory "{1:s}".'
    ).format(year, daily_example_dir_name)

    raise ValueError(error_string)


def _concat_files_one_year(
        input_daily_example_dir_name, input_yearly_example_dir_name, year):
    """Concatenates example files for one year.

    :param input_daily_example_dir_name: See documentation at top of file.
    :param input_yearly_example_dir_name: Same.
    :param year: Year (integer).
    :return: example_dict: Dictionary with all examples for the given year.  For
        a list of keys, see `example_io.read_file`.
    """

    input_file_names = _find_daily_files(
        daily_example_dir_name=input_daily_example_dir_name, year=year
    )

    if input_yearly_example_dir_name is not None:
        this_file_name = example_io.find_file(
            example_dir_name=input_yearly_example_dir_name, year=year,
            raise_error_if_missing=False
        )

        if os.path.isfile(this_file_name):
            input_file_names.append(this_file_name)

    num_files = len(input_file_names)
    example_dicts = [dict()] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        example_dicts[i] = example_io.read_file(
            example_file_name=input_file_names[i], allow_bad_values=True
        )

    return example_io.concat_examples(example_dicts)


def _remove_duplicate_examples(example_dict):
    """Removes duplicate examples from dictionary.

    :param example_dict: Dictionary in format created by `example_io.read_file`.
    :return: example_dict: Same but maybe with fewer examples.
    """

    example_id_strings = example_dict[example_io.EXAMPLE_IDS_KEY]
    unique_indices = numpy.unique(
        numpy.array(example_id_strings), return_index=True
    )[1]

    print('{0:d} of {1:d} examples are unique!'.format(
        len(unique_indices), len(example_id_strings)
    ))

    return example_io.subset_by_index(
        example_dict=example_dict, desired_indices=unique_indices
    )


def _run(input_daily_example_dir_name, input_yearly_example_dir_name,
         first_year, last_year, output_yearly_example_dir_name):
    """Concatenates daily example files into yearly example files.

    This is effectively the main method.

    :param input_daily_example_dir_name: See documentation at top of file.
    :param input_yearly_example_dir_name: Same.
    :param first_year: Same.
    :param last_year: Same.
    :param output_yearly_example_dir_name: Same.
    """

    if input_yearly_example_dir_name == '':
        input_yearly_example_dir_name = None

    error_checking.assert_is_geq(last_year, first_year)
    years = numpy.linspace(
        first_year, last_year, num=last_year - first_year + 1, dtype=int
    )

    for this_year in years:
        this_concat_example_dict = _concat_files_one_year(
            input_daily_example_dir_name=input_daily_example_dir_name,
            input_yearly_example_dir_name=input_yearly_example_dir_name,
            year=this_year
        )
        print('\n')

        this_concat_example_dict = (
            _remove_duplicate_examples(this_concat_example_dict)
        )
        this_concat_file_name = example_io.find_file(
            example_dir_name=output_yearly_example_dir_name, year=this_year,
            raise_error_if_missing=False
        )

        print('Writing {0:d} examples to file: "{1:s}"...'.format(
            len(this_concat_example_dict[example_io.VALID_TIMES_KEY]),
            this_concat_file_name
        ))
        example_io.write_file(
            example_dict=this_concat_example_dict,
            netcdf_file_name=this_concat_file_name
        )

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_daily_example_dir_name=getattr(
            INPUT_ARG_OBJECT, INPUT_DAILY_DIR_ARG_NAME
        ),
        input_yearly_example_dir_name=getattr(
            INPUT_ARG_OBJECT, INPUT_YEARLY_DIR_ARG_NAME
        ),
        first_year=getattr(INPUT_ARG_OBJECT, FIRST_YEAR_ARG_NAME),
        last_year=getattr(INPUT_ARG_OBJECT, LAST_YEAR_ARG_NAME),
        output_yearly_example_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_YEARLY_DIR_ARG_NAME
        )
    )
