"""Merges discard-test results based on different example sets.

i.e., discard-test results created from different prediction files
"""

import argparse
import numpy
from ml4rt.utils import discard_test_utils as dt_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILES_ARG_NAME = 'input_file_names'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files.  Each will be read by '
    '`discard_test_utils.read_results`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output (merged) file.  Will be written here by '
    '`discard_test_utils.write_results`.'
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
    """Merges discard-test results based on different example sets.

    This is effectively the main method.

    :param input_file_names: See documentation at top of file.
    :param output_file_name: Same.
    """

    num_input_files = len(input_file_names)
    input_tables_xarray = [None] * num_input_files

    for i in range(num_input_files):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        input_tables_xarray[i] = dt_utils.read_results(input_file_names[i])

    output_table_xarray = dt_utils.merge_results_over_examples(
        input_tables_xarray
    )
    del input_tables_xarray
    print(SEPARATOR_STRING)

    discard_fractions = (
        1. - output_table_xarray[dt_utils.EXAMPLE_FRACTION_KEY].values
    )
    post_discard_errors = (
        output_table_xarray[dt_utils.POST_DISCARD_ERROR_KEY].values
    )

    for i in range(len(discard_fractions)):
        print('Error with discard fraction of {0:.4f} = {1:.4g}'.format(
            discard_fractions[i], post_discard_errors[i]
        ))

    print('\nMonotonicity fraction = {0:.4f}'.format(
        output_table_xarray.attrs[dt_utils.MONO_FRACTION_KEY]
    ))
    print('Mean discard improvement = {0:.4f}'.format(
        output_table_xarray.attrs[dt_utils.MEAN_DI_KEY]
    ))
    print(SEPARATOR_STRING)

    t = output_table_xarray
    scalar_target_names = t.coords[dt_utils.SCALAR_FIELD_DIM].values

    for k in range(len(scalar_target_names)):
        print('Variable = {0:s} ... MF = {1:f} ... DI = {2:f}'.format(
            scalar_target_names[k],
            t[dt_utils.SCALAR_MONO_FRACTION_KEY].values[k],
            t[dt_utils.SCALAR_MEAN_DI_KEY].values[k]
        ))

    print(SEPARATOR_STRING)
    vector_target_names = t.coords[dt_utils.VECTOR_FIELD_DIM].values
    heights_m_agl = t.coords[dt_utils.HEIGHT_DIM].values

    for k in range(len(vector_target_names)):
        print('Variable = {0:s} ... MF = {1:f} ... DI = {2:f}'.format(
            vector_target_names[k],
            t[dt_utils.VECTOR_FLAT_MONO_FRACTION_KEY].values[k],
            t[dt_utils.VECTOR_FLAT_MEAN_DI_KEY].values[k]
        ))

        for j in range(len(heights_m_agl)):
            print((
                'Variable = {0:s} at {1:d} m AGL ... MF = {2:f} ... DI = {3:f}'
            ).format(
                vector_target_names[k], int(numpy.round(heights_m_agl[j])),
                t[dt_utils.VECTOR_MONO_FRACTION_KEY].values[k, j],
                t[dt_utils.VECTOR_MEAN_DI_KEY].values[k, j]
            ))

        print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            t.coords[dt_utils.AUX_TARGET_FIELD_DIM].values
        )
        aux_predicted_field_names = (
            t.coords[dt_utils.AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_target_field_names = []
        aux_predicted_field_names = []

    for k in range(len(aux_target_field_names)):
        print((
            'Target variable = {0:s} ... predicted variable = "{1:s}" ... '
            'MF = {2:f} ... DI = {3:f}'
        ).format(
            aux_target_field_names[k], aux_predicted_field_names[k],
            t[dt_utils.AUX_MONO_FRACTION_KEY].values[k],
            t[dt_utils.AUX_MEAN_DI_KEY].values[k]
        ))

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    dt_utils.write_results(
        result_table_xarray=output_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
