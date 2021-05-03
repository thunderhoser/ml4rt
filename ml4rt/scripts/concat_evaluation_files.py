"""Concatenates evaluation files with different bootstrap replicates."""

import argparse
import numpy
import xarray
from ml4rt.utils import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILES_ARG_NAME = 'input_file_names'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files, each containing a small number of bootstrap '
    'replicates.  These will be read by `evaluation.read_file`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file, which will contain all bootstrap replicates.  Will be'
    ' written by `evaluation.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_file_names, output_file_name):
    """Concatenates evaluation files with different bootstrap replicates.

    This is effectively the main method.

    :param input_file_names: See documentation at top of file.
    :param output_file_name: Same.
    """

    result_tables_xarray = []
    num_bootstrap_reps_read = 0

    for this_file_name in input_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_result_table_xarray = evaluation.read_file(this_file_name)

        num_bootstrap_reps_new = len(
            this_result_table_xarray.coords[evaluation.BOOTSTRAP_REP_DIM].values
        )
        these_indices = numpy.linspace(
            num_bootstrap_reps_read,
            num_bootstrap_reps_read + num_bootstrap_reps_new - 1,
            num=num_bootstrap_reps_new, dtype=int
        )
        this_result_table_xarray.coords[evaluation.BOOTSTRAP_REP_DIM].values = (
            these_indices
        )

        result_tables_xarray.append(this_result_table_xarray)
        num_bootstrap_reps_read += num_bootstrap_reps_new

    print(SEPARATOR_STRING)

    result_table_xarray = xarray.concat(
        objs=result_tables_xarray, dim=evaluation.BOOTSTRAP_REP_DIM
    )
    print(result_table_xarray)

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    evaluation.write_file(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
