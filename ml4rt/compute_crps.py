"""Computes CRPS (continuous ranked prob score) for each target variable."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import crps_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
METRES_TO_MICRONS = 1e6

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

    rtx = result_table_xarray
    scalar_target_names = rtx.coords[crps_utils.SCALAR_FIELD_DIM].values
    wavelengths_microns = (
        METRES_TO_MICRONS * rtx.coords[crps_utils.WAVELENGTH_DIM].values
    )

    for t in range(len(scalar_target_names)):
        for w in range(len(wavelengths_microns)):
            print((
                'Variable = {0:s} at {1:.2f} microns ... '
                'CRPS = {2:f} ... CRPSS = {3:f} ... DWCRPS = {4:f}'
            ).format(
                scalar_target_names[t],
                wavelengths_microns[w],
                rtx[crps_utils.SCALAR_CRPS_KEY].values[t, w],
                rtx[crps_utils.SCALAR_CRPSS_KEY].values[t, w],
                rtx[crps_utils.SCALAR_DWCRPS_KEY].values[t, w]
            ))

        print(SEPARATOR_STRING)

    vector_target_names = rtx.coords[crps_utils.VECTOR_FIELD_DIM].values
    heights_m_agl = rtx.coords[crps_utils.HEIGHT_DIM].values

    for t in range(len(vector_target_names)):
        for w in range(len(wavelengths_microns)):
            for h in range(len(heights_m_agl)):
                print((
                    'Variable = {0:s} at {1:.2f} microns and {2:d} m AGL ... '
                    'CRPS = {3:f} ... CRPSS = {4:f} ... DWCRPS = {5:f}'
                ).format(
                    vector_target_names[t],
                    wavelengths_microns[w],
                    int(numpy.round(heights_m_agl[h])),
                    rtx[crps_utils.VECTOR_CRPS_KEY].values[t, h, w],
                    rtx[crps_utils.VECTOR_CRPSS_KEY].values[t, h, w],
                    rtx[crps_utils.VECTOR_DWCRPS_KEY].values[t, h, w]
                ))

            print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            rtx.coords[crps_utils.AUX_TARGET_FIELD_DIM].values
        )
        aux_predicted_field_names = (
            rtx.coords[crps_utils.AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_target_field_names = []
        aux_predicted_field_names = []

    for t in range(len(aux_target_field_names)):
        for w in range(len(wavelengths_microns)):
            print((
                'Target variable = {0:s} at {1:.2f} microns ... '
                'predicted variable = {2:s} at {1:.2f} microns ... '
                'CRPS = {3:f} ... CRPSS = {4:f} ... DWCRPS = {5:f}'
            ).format(
                aux_target_field_names[t],
                wavelengths_microns[w],
                aux_predicted_field_names[t],
                rtx[crps_utils.AUX_CRPS_KEY].values[t],
                rtx[crps_utils.AUX_CRPSS_KEY].values[t],
                rtx[crps_utils.AUX_DWCRPS_KEY].values[t]
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
