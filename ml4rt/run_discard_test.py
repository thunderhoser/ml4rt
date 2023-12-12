"""Runs discard test to determine quality of uncertainty estimates."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import prediction_io
import uq_evaluation
import discard_test_utils as dt_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
METRES_TO_MICRONS = 1e6

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
DISCARD_FRACTIONS_ARG_NAME = 'discard_fractions'
PRED_FILES_FOR_UNC_THRES_ARG_NAME = 'prediction_file_names_for_unc_thres'
DWMSE_SCALING_ARG_NAME = 'scaling_factor_for_dwmse'
FLUX_MSE_SCALING_ARG_NAME = 'scaling_factor_for_flux_mse'
USE_HR_FOR_UNCERTAINTY_ARG_NAME = 'use_hr_for_uncertainty'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions and target values.  Will be '
    'read by `prediction_io.read_file`.'
)
DISCARD_FRACTIONS_HELP_STRING = (
    'List of discard fractions, ranging from (0, 1).  This script will '
    'automatically use 0 as the lowest discard fraction.'
)
PRED_FILES_FOR_UNC_THRES_HELP_STRING = (
    'List of paths to prediction files.  These files (read by `prediction_io.'
    'read_file`) will be used to convert discard fractions to uncertainty '
    'thresholds.'
)
DWMSE_SCALING_HELP_STRING = (
    'This script will use error function returned by '
    '`uq_evaluation.make_error_function_dwmse_plus_flux_mse`, with this '
    'scaling factor, which multiplies the dual-weighted MSE for heating rate.'
)
FLUX_MSE_SCALING_HELP_STRING = (
    'This script will use error function returned by '
    '`uq_evaluation.make_error_function_dwmse_plus_flux_mse`, with this '
    'scaling factor, which multiplies the MSE for flux.'
)
USE_HR_FOR_UNCERTAINTY_HELP_STRING = (
    'Boolean flag.  If 1 (0), will use standard deviation of heating-rate '
    '(flux) predictions to quantify uncertainty.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output (NetCDF) file.  Results will be written here by '
    '`dt_utils.write_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DISCARD_FRACTIONS_ARG_NAME, type=float, nargs='+', required=True,
    help=DISCARD_FRACTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PRED_FILES_FOR_UNC_THRES_ARG_NAME, type=str, nargs='+',
    required=True, help=PRED_FILES_FOR_UNC_THRES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DWMSE_SCALING_ARG_NAME, type=float, required=True,
    help=DWMSE_SCALING_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FLUX_MSE_SCALING_ARG_NAME, type=float, required=True,
    help=FLUX_MSE_SCALING_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_HR_FOR_UNCERTAINTY_ARG_NAME, type=int, required=True,
    help=USE_HR_FOR_UNCERTAINTY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_file_name, discard_fractions,
         prediction_file_names_for_unc_thres, scaling_factor_for_dwmse,
         scaling_factor_for_flux_mse, use_hr_for_uncertainty, output_file_name):
    """Runs discard test to determine quality of uncertainty estimates.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param discard_fractions: Same.
    :param prediction_file_names_for_unc_thres: Same.
    :param scaling_factor_for_dwmse: Same.
    :param scaling_factor_for_flux_mse: Same.
    :param use_hr_for_uncertainty: Same.
    :param output_file_name: Same.
    """

    # Check discard fractions.
    error_checking.assert_is_greater_numpy_array(discard_fractions, 0.)
    error_checking.assert_is_less_than_numpy_array(discard_fractions, 1.)

    discard_fractions = numpy.concatenate((
        numpy.array([0.]),
        discard_fractions
    ))
    discard_fractions = numpy.sort(discard_fractions)

    num_fractions = len(discard_fractions)
    assert num_fractions >= 2

    # Convert discard fractions to uncertainty thresholds.
    if use_hr_for_uncertainty:
        uncertainty_function = uq_evaluation.make_heating_rate_stdev_function()
    else:
        uncertainty_function = uq_evaluation.make_flux_stdev_function()

    uncertainty_values = numpy.array([])

    for this_file_name in prediction_file_names_for_unc_thres:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_prediction_dict = prediction_io.read_file(this_file_name)
        uncertainty_values = numpy.concatenate((
            uncertainty_values,
            uncertainty_function(this_prediction_dict)
        ))

    print(SEPARATOR_STRING)

    percentile_levels = 100 * (1. - discard_fractions)
    uncertainty_thresholds = numpy.percentile(
        uncertainty_values, percentile_levels
    )

    for i in range(len(discard_fractions)):
        print((
            'Uncertainty threshold for discard fraction of {0:.4f} = {1:.4g}'
        ).format(
            discard_fractions[i], uncertainty_thresholds[i]
        ))

    print(SEPARATOR_STRING)

    # Do actual stuff.
    error_function = uq_evaluation.make_error_function_dwmse_plus_flux_mse(
        scaling_factor_for_dwmse=scaling_factor_for_dwmse,
        scaling_factor_for_flux_mse=scaling_factor_for_flux_mse
    )

    result_table_xarray = dt_utils.run_discard_test(
        prediction_file_name=prediction_file_name,
        uncertainty_thresholds=uncertainty_thresholds,
        error_function=error_function,
        uncertainty_function=uncertainty_function, is_error_pos_oriented=False,
        error_function_for_hr_1height=
        uq_evaluation.make_error_function_dwmse_1height(),
        error_function_for_flux_1var=
        uq_evaluation.make_error_function_flux_mse_1var()
    )
    print(SEPARATOR_STRING)

    rtx = result_table_xarray
    discard_fractions = 1. - rtx[dt_utils.EXAMPLE_FRACTION_KEY].values
    post_discard_errors = rtx[dt_utils.POST_DISCARD_ERROR_KEY].values

    for i in range(len(discard_fractions)):
        print('Error with discard fraction of {0:.4f} = {1:.4g}'.format(
            discard_fractions[i], post_discard_errors[i]
        ))

    print('\nMonotonicity fraction = {0:.4f}'.format(
        rtx.attrs[dt_utils.MONO_FRACTION_KEY]
    ))
    print('Mean discard improvement = {0:.4f}'.format(
        rtx.attrs[dt_utils.MEAN_DI_KEY]
    ))
    print(SEPARATOR_STRING)

    scalar_target_names = rtx.coords[dt_utils.SCALAR_FIELD_DIM].values
    wavelengths_microns = (
        METRES_TO_MICRONS * rtx.coords[dt_utils.WAVELENGTH_DIM].values
    )

    for t in range(len(scalar_target_names)):
        for w in range(len(wavelengths_microns)):
            print((
                'Variable = {0:s} at {1:.2f} microns ... '
                'MF = {2:f} ... DI = {3:f}'
            ).format(
                scalar_target_names[t],
                wavelengths_microns[w],
                rtx[dt_utils.SCALAR_MONO_FRACTION_KEY].values[t, w],
                rtx[dt_utils.SCALAR_MEAN_DI_KEY].values[t, w]
            ))

        print(SEPARATOR_STRING)

    vector_target_names = rtx.coords[dt_utils.VECTOR_FIELD_DIM].values
    heights_m_agl = rtx.coords[dt_utils.HEIGHT_DIM].values

    for t in range(len(vector_target_names)):
        for w in range(len(wavelengths_microns)):
            print((
                'Variable = {0:s} at {1:.2f} microns ... '
                'MF = {2:f} ... DI = {3:f}'
            ).format(
                vector_target_names[t],
                wavelengths_microns[w],
                rtx[dt_utils.VECTOR_FLAT_MONO_FRACTION_KEY].values[t, w],
                rtx[dt_utils.VECTOR_FLAT_MEAN_DI_KEY].values[t, w]
            ))

        print(SEPARATOR_STRING)

    for t in range(len(vector_target_names)):
        for w in range(len(wavelengths_microns)):
            for h in range(len(heights_m_agl)):
                print((
                    'Variable = {0:s} at {1:.2f} microns and {2:d} m AGL ... '
                    'MF = {3:f} ... DI = {4:f}'
                ).format(
                    vector_target_names[t],
                    wavelengths_microns[w],
                    int(numpy.round(heights_m_agl[h])),
                    rtx[dt_utils.VECTOR_MONO_FRACTION_KEY].values[t, h, w],
                    rtx[dt_utils.VECTOR_MEAN_DI_KEY].values[t, h, w]
                ))

            print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            rtx.coords[dt_utils.AUX_TARGET_FIELD_DIM].values
        )
        aux_predicted_field_names = (
            rtx.coords[dt_utils.AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_target_field_names = []
        aux_predicted_field_names = []

    for t in range(len(aux_target_field_names)):
        for w in range(len(wavelengths_microns)):
            print((
                'Target variable = {0:s} at {1:.2f} microns ... '
                'predicted variable = {2:s} at {1:.2f} microns ... '
                'MF = {3:f} ... DI = {4:f}'
            ).format(
                aux_target_field_names[t],
                wavelengths_microns[w],
                aux_predicted_field_names[t],
                rtx[dt_utils.AUX_MONO_FRACTION_KEY].values[t, w],
                rtx[dt_utils.AUX_MEAN_DI_KEY].values[t, w]
            ))

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    dt_utils.write_results(
        result_table_xarray=rtx,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        discard_fractions=numpy.array(
            getattr(INPUT_ARG_OBJECT, DISCARD_FRACTIONS_ARG_NAME), dtype=float
        ),
        prediction_file_names_for_unc_thres=getattr(
            INPUT_ARG_OBJECT, PRED_FILES_FOR_UNC_THRES_ARG_NAME
        ),
        scaling_factor_for_dwmse=getattr(
            INPUT_ARG_OBJECT, DWMSE_SCALING_ARG_NAME
        ),
        scaling_factor_for_flux_mse=getattr(
            INPUT_ARG_OBJECT, FLUX_MSE_SCALING_ARG_NAME
        ),
        use_hr_for_uncertainty=bool(
            getattr(INPUT_ARG_OBJECT, USE_HR_FOR_UNCERTAINTY_ARG_NAME)
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
