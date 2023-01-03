"""Merges spread-skill results based on different example sets.

i.e., spread-skill results created from different prediction files
"""

import argparse
import numpy
from ml4rt.utils import spread_skill_utils as ss_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILES_ARG_NAME = 'input_file_names'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files.  Each will be read by '
    '`spread_skill_utils.read_results`.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output (merged) file.  Will be written here by '
    '`spread_skill_utils.write_results`.'
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
    """Merges spread-skill results based on different example sets.

    This is effectively the main method.

    :param input_file_names: See documentation at top of file.
    :param output_file_name: Same.
    """

    num_input_files = len(input_file_names)
    input_tables_xarray = [None] * num_input_files

    for i in range(num_input_files):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        input_tables_xarray[i] = ss_utils.read_results(input_file_names[i])

    output_table_xarray = ss_utils.merge_results_over_examples(
        input_tables_xarray
    )
    del input_tables_xarray
    print(SEPARATOR_STRING)

    t = output_table_xarray
    scalar_target_names = t.coords[ss_utils.SCALAR_FIELD_DIM].values

    for k in range(len(scalar_target_names)):
        print((
            'Variable = "{0:s}" ... SSREL = {1:f} ... SSRAT = {2:f}'
        ).format(
            scalar_target_names[k],
            t[ss_utils.SCALAR_SSREL_KEY].values[k],
            t[ss_utils.SCALAR_SSRAT_KEY].values[k]
        ))

    print(SEPARATOR_STRING)

    vector_target_names = t.coords[ss_utils.VECTOR_FIELD_DIM].values
    heights_m_agl = t.coords[ss_utils.HEIGHT_DIM].values

    for k in range(len(vector_target_names)):
        print((
            'Variable = "{0:s}" ... SSREL = {1:f} ... SSRAT = {2:f}'
        ).format(
            vector_target_names[k],
            t[ss_utils.VECTOR_FLAT_SSREL_KEY].values[k],
            t[ss_utils.VECTOR_FLAT_SSRAT_KEY].values[k]
        ))

        for j in range(len(heights_m_agl)):
            print((
                'Variable = "{0:s}" at {1:d} m AGL ... SSREL = {2:f} ... '
                'SSRAT = {3:f}'
            ).format(
                vector_target_names[k], int(numpy.round(heights_m_agl[j])),
                t[ss_utils.VECTOR_SSREL_KEY].values[k, j],
                t[ss_utils.VECTOR_SSRAT_KEY].values[k, j]
            ))

        print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            t.coords[ss_utils.AUX_TARGET_FIELD_DIM].values
        )
        aux_predicted_field_names = (
            t.coords[ss_utils.AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_target_field_names = []
        aux_predicted_field_names = []

    for k in range(len(aux_target_field_names)):
        print((
            'Target variable = "{0:s}" ... predicted variable = "{1:s}" ... '
            'SSREL = {2:f} ... SSRAT = {3:f}'
        ).format(
            aux_target_field_names[k], aux_predicted_field_names[k],
            t[ss_utils.AUX_SSREL_KEY].values[k],
            t[ss_utils.AUX_SSRAT_KEY].values[k]
        ))

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    ss_utils.write_results(
        result_table_xarray=output_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
