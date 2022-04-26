"""Removes duplicate examples from directory."""

import argparse
import numpy
from ml4rt.io import example_io
from ml4rt.utils import example_utils

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
YEARS_ARG_NAME = 'years'
OUTPUT_DIR_ARG_NAME = 'output_example_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of directory with input examples, which may not be all unique.  Files'
    ' therein will be found by `example_io.find_file` and read by '
    '`example_io.read_file`.'
)
YEARS_HELP_STRING = 'Will handle these years only.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory, to which only unique examples will be written.  '
    'Files will be written by `example_io.write_file`, to exact locations in '
    'this directory determined by `example_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEARS_ARG_NAME, type=int, nargs='+', required=True,
    help=YEARS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_example_dir_name, years, output_example_dir_name):
    """Removes duplicate examples from directory.

    This is effectively the main method.

    :param input_example_dir_name: See documentation at top of file.
    :param years: Same.
    :param output_example_dir_name: Same.
    """

    input_example_file_names = []

    for this_year in years:
        input_example_file_names += example_io.find_files_one_year(
            directory_name=input_example_dir_name, year=this_year,
            raise_error_if_missing=False
        )

    assert len(input_example_file_names) > 0
    example_id_strings = []

    for this_file_name in input_example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(
            netcdf_file_name=this_file_name, exclude_summit_greenland=False,
            max_shortwave_heating_k_day01=numpy.inf
        )
        example_id_strings += this_example_dict[example_utils.EXAMPLE_IDS_KEY]

    unique_example_id_strings, unique_counts = numpy.unique(
        numpy.array(example_id_strings), return_counts=True
    )

    if numpy.all(unique_counts == 1):
        print((
            'All {0:d} examples are unique, so no need to remove duplicates!'
        ).format(
            len(example_id_strings)
        ))

        return

    bad_id_strings = unique_example_id_strings[unique_counts > 1]

    for input_file_name in input_example_file_names:
        print('Reading data from: "{0:s}"...'.format(input_file_name))
        example_dict = example_io.read_file(
            netcdf_file_name=input_file_name,
            exclude_summit_greenland=False,
            max_shortwave_heating_k_day01=numpy.inf
        )

        all_id_strings = numpy.array(
            example_dict[example_utils.EXAMPLE_IDS_KEY]
        )
        good_indices = numpy.where(
            numpy.invert(numpy.in1d(all_id_strings, bad_id_strings))
        )[0]

        num_examples_orig = len(all_id_strings)
        num_examples = len(good_indices)
        example_dict = example_utils.subset_by_index(
            example_dict=example_dict, desired_indices=good_indices
        )

        output_file_name = example_io.find_file(
            directory_name=output_example_dir_name,
            year=example_io.file_name_to_year(input_file_name),
            year_part_number=
            example_io.file_name_to_year_part(input_file_name),
            raise_error_if_missing=False
        )

        print((
            'Writing {0:d} unique examples (out of {1:d} total) to: "{2:s}"...'
        ).format(
            num_examples, num_examples_orig, output_file_name
        ))

        example_io.write_file(
            example_dict=example_dict, netcdf_file_name=output_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_example_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        years=numpy.array(getattr(INPUT_ARG_OBJECT, YEARS_ARG_NAME), dtype=int),
        output_example_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
