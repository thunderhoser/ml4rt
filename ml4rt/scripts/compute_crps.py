"""Computes CRPS (continuous ranked prob score) for each target variable."""

import argparse
import numpy
from ml4rt.utils import crps_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
NUM_LEVELS_ARG_NAME = 'num_integration_levels'
ENSEMBLE_SIZE_FOR_CLIMO_ARG_NAME = 'ensemble_size_for_climo'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions and target values.  Will be '
    'read by `prediction_io.read_file`.'
)
NUM_LEVELS_HELP_STRING = (
    'Number of levels used to approximate integral over predictive '
    'distribution (y_pred).'
)
ENSEMBLE_SIZE_FOR_CLIMO_HELP_STRING = (
    'Ensemble size used to compute CRPS for climatological model.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output (NetCDF) file.  Results will be written here by '
    '`crps_utils.write_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_LEVELS_ARG_NAME, type=int, required=True,
    help=NUM_LEVELS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ENSEMBLE_SIZE_FOR_CLIMO_ARG_NAME, type=int, required=True,
    help=ENSEMBLE_SIZE_FOR_CLIMO_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_file_name, num_integration_levels, ensemble_size_for_climo,
         output_file_name):
    """Computes CRPS (continuous ranked prob score) for each target variable.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param num_integration_levels: Same.
    :param ensemble_size_for_climo: Same.
    :param output_file_name: Same.
    """

    result_table_xarray = crps_utils.get_crps_related_scores_all_vars(
        prediction_file_name=prediction_file_name,
        num_integration_levels=num_integration_levels,
        ensemble_size_for_climo=ensemble_size_for_climo
    )
    print(SEPARATOR_STRING)

    t = result_table_xarray
    scalar_target_names = t.coords[crps_utils.SCALAR_FIELD_DIM].values

    for k in range(len(scalar_target_names)):
        print((
            'Variable = {0:s} ... CRPS = {1:f} ... CRPSS = {2:f} ... '
            'DWCRPS = {3:f}'
        ).format(
            scalar_target_names[k],
            t[crps_utils.SCALAR_CRPS_KEY].values[k],
            t[crps_utils.SCALAR_CRPSS_KEY].values[k],
            t[crps_utils.SCALAR_DWCRPS_KEY].values[k]
        ))

    print(SEPARATOR_STRING)
    vector_target_names = t.coords[crps_utils.VECTOR_FIELD_DIM].values
    heights_m_agl = t.coords[crps_utils.HEIGHT_DIM].values

    for k in range(len(vector_target_names)):
        for j in range(len(heights_m_agl)):
            print((
                'Variable = {0:s} at {1:d} m AGL ... CRPS = {2:f} ... '
                'CRPSS = {3:f} ... DWCRPS = {4:f}'
            ).format(
                vector_target_names[k], int(numpy.round(heights_m_agl[j])),
                t[crps_utils.VECTOR_CRPS_KEY].values[k, j],
                t[crps_utils.VECTOR_CRPSS_KEY].values[k, j],
                t[crps_utils.VECTOR_DWCRPS_KEY].values[k, j]
            ))

        print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            t.coords[crps_utils.AUX_TARGET_FIELD_DIM].values
        )
        aux_predicted_field_names = (
            t.coords[crps_utils.AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_target_field_names = []
        aux_predicted_field_names = []

    for k in range(len(aux_target_field_names)):
        print((
            'Target variable = {0:s} ... predicted variable = {1:s} ... '
            'CRPS = {2:f} ... CRPSS = {3:f} ... DWCRPS = {4:f}'
        ).format(
            aux_target_field_names[k], aux_predicted_field_names[k],
            t[crps_utils.AUX_CRPS_KEY].values[k],
            t[crps_utils.AUX_CRPSS_KEY].values[k],
            t[crps_utils.AUX_DWCRPS_KEY].values[k]
        ))

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    crps_utils.write_results(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_integration_levels=getattr(INPUT_ARG_OBJECT, NUM_LEVELS_ARG_NAME),
        ensemble_size_for_climo=getattr(
            INPUT_ARG_OBJECT, ENSEMBLE_SIZE_FOR_CLIMO_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
