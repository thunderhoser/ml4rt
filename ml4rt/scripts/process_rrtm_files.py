"""Converts daily RRTM files to yearly example files."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import rrtm_io
from ml4rt.io import example_io
from ml4rt.utils import example_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DAYS_TO_SECONDS = 86400

RRTM_DIRECTORY_ARG_NAME = 'input_rrtm_directory_name'
INPUT_EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_YEAR_ARG_NAME = 'first_year'
LAST_YEAR_ARG_NAME = 'last_year'
OUTPUT_EXAMPLE_DIR_ARG_NAME = 'output_example_dir_name'

RRTM_DIRECTORY_HELP_STRING = (
    'Name of top-level directory with daily RRTM files.  Files therein will be '
    'read by `rrtm_io.read_file`.'
)

# TODO(thunderhoser): Fix documentation for this input arg.  It exists only
# because of the first couple files that Dave Turner sent me.
INPUT_EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with input example files.  Files therein will be found '
    'by `rrtm_io.find_file` and read by `rrtm_io.read_file`.  These files'
    ' will be concatenated with RRTM files to create new example files.  If you'
    ' do not have existing example files, leave this argument alone.'
)
YEAR_HELP_STRING = (
    'Will create example files for all years from `{0:s}` to `{1:s}`.'
).format(FIRST_YEAR_ARG_NAME, LAST_YEAR_ARG_NAME)

OUTPUT_EXAMPLE_DIR_HELP_STRING = (
    'Name of output directory.  Example files will be written here by '
    '`example_io.write_file`, with file names determined by '
    '`example_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + RRTM_DIRECTORY_ARG_NAME, type=str, required=True,
    help=RRTM_DIRECTORY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_EXAMPLE_DIR_ARG_NAME, type=str, required=False, default='',
    help=INPUT_EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_EXAMPLE_DIR_ARG_NAME, type=str, required=False, default='',
    help=OUTPUT_EXAMPLE_DIR_HELP_STRING
)


def _find_rrtm_files(rrtm_directory_name, year):
    """Finds RRTM files for the given year.

    :param rrtm_directory_name: See documentation at top of file.
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
            rrtm_directory_name, year, i
        )

        if not os.path.isfile(this_file_name):
            continue

        example_file_names.append(this_file_name)

    if len(example_file_names) > 0:
        return example_file_names

    error_string = (
        'Cannot find daily examples for year {0:d} in directory "{1:s}".'
    ).format(year, rrtm_directory_name)

    raise ValueError(error_string)


def _process_files_one_year(
        rrtm_directory_name, input_example_dir_name, year):
    """Processes RRTM files for one year.

    :param rrtm_directory_name: See documentation at top of file.
    :param input_example_dir_name: Same.
    :param year: Year (integer).
    :return: example_dict: Dictionary with all examples for the given year.  For
        a list of keys, see `example_io.read_file`.
    """

    rrtm_file_names = _find_rrtm_files(
        rrtm_directory_name=rrtm_directory_name, year=year
    )

    num_files = len(rrtm_file_names)
    example_dicts = [dict()] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(rrtm_file_names[i]))
        example_dicts[i] = rrtm_io.read_file(
            netcdf_file_name=rrtm_file_names[i], allow_bad_values=True
        )

    if input_example_dir_name is None:
        example_file_name = None
    else:
        example_file_name = rrtm_io.find_file(
            directory_name=input_example_dir_name, year=year,
            raise_error_if_missing=False
        )

        if not os.path.isfile(example_file_name):
            example_file_name = None

    if example_file_name is not None:
        print('Reading data from: "{0:s}"...'.format(example_file_name))
        example_dicts.append(rrtm_io.read_file(example_file_name))

    return example_utils.concat_examples(example_dicts)


def _remove_examples_in_wrong_year(example_dict, desired_year):
    """Removes examples in the wrong year.

    :param example_dict: Dictionary in format created by `example_io.read_file`.
    :param desired_year: Year that belongs in this dictionary (integer).
    :return: example_dict: Same as input but maybe with fewer examples.
    """

    first_time_unix_sec, last_time_unix_sec = (
        time_conversion.first_and_last_times_in_year(desired_year)
    )

    num_examples_orig = len(example_dict[example_utils.VALID_TIMES_KEY])
    example_dict = example_utils.subset_by_time(
        example_dict=example_dict, first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec
    )[0]
    num_examples = len(example_dict[example_utils.VALID_TIMES_KEY])

    print('Removed {0:d} of {1:d} examples for being in the wrong year.'.format(
        num_examples_orig - num_examples, num_examples_orig
    ))
    return example_dict


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


def _run(rrtm_directory_name, input_example_dir_name,
         first_year, last_year, output_example_dir_name):
    """Converts daily RRTM files to yearly example files.

    This is effectively the main method.

    :param rrtm_directory_name: See documentation at top of file.
    :param input_example_dir_name: Same.
    :param first_year: Same.
    :param last_year: Same.
    :param output_example_dir_name: Same.
    """

    if input_example_dir_name == '':
        input_example_dir_name = None

    error_checking.assert_is_geq(last_year, first_year)
    years = numpy.linspace(
        first_year, last_year, num=last_year - first_year + 1, dtype=int
    )

    for this_year in years:
        this_example_dict = _process_files_one_year(
            rrtm_directory_name=rrtm_directory_name,
            input_example_dir_name=input_example_dir_name,
            year=this_year
        )
        print('\n')

        this_example_dict = _remove_examples_in_wrong_year(
            example_dict=this_example_dict, desired_year=this_year
        )
        this_example_dict = _remove_duplicate_examples(this_example_dict)
        this_output_file_name = example_io.find_file(
            directory_name=output_example_dir_name, year=this_year,
            raise_error_if_missing=False
        )

        print('Writing {0:d} examples to file: "{1:s}"...'.format(
            len(this_example_dict[example_utils.VALID_TIMES_KEY]),
            this_output_file_name
        ))
        example_io.write_file(
            example_dict=this_example_dict,
            netcdf_file_name=this_output_file_name
        )

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        rrtm_directory_name=getattr(INPUT_ARG_OBJECT, RRTM_DIRECTORY_ARG_NAME),
        input_example_dir_name=getattr(
            INPUT_ARG_OBJECT, INPUT_EXAMPLE_DIR_ARG_NAME
        ),
        first_year=getattr(INPUT_ARG_OBJECT, FIRST_YEAR_ARG_NAME),
        last_year=getattr(INPUT_ARG_OBJECT, LAST_YEAR_ARG_NAME),
        output_example_dir_name=getattr(
            INPUT_ARG_OBJECT, OUTPUT_EXAMPLE_DIR_ARG_NAME
        )
    )
