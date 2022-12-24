"""Runs discard test to determine quality of uncertainty estimates."""

import argparse
import numpy
from ml4rt.utils import uq_evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
DISCARD_FRACTIONS_ARG_NAME = 'discard_fractions'
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
    '`uq_evaluation.write_discard_results`.'
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


def _run(prediction_file_name, discard_fractions, scaling_factor_for_dwmse,
         scaling_factor_for_flux_mse, use_hr_for_uncertainty, output_file_name):
    """Runs discard test to determine quality of uncertainty estimates.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param discard_fractions: Same.
    :param scaling_factor_for_dwmse: Same.
    :param scaling_factor_for_flux_mse: Same.
    :param use_hr_for_uncertainty: Same.
    :param output_file_name: Same.
    """

    error_function = uq_evaluation.make_error_function_dwmse_plus_flux_mse(
        scaling_factor_for_dwmse=scaling_factor_for_dwmse,
        scaling_factor_for_flux_mse=scaling_factor_for_flux_mse
    )

    if use_hr_for_uncertainty:
        uncertainty_function = uq_evaluation.make_heating_rate_stdev_function()
    else:
        uncertainty_function = uq_evaluation.make_flux_stdev_function()

    result_table_xarray = uq_evaluation.run_discard_test_all_vars(
        prediction_file_name=prediction_file_name,
        discard_fractions=discard_fractions,
        error_function=error_function,
        uncertainty_function=uncertainty_function, is_error_pos_oriented=False,
        error_function_for_hr_1height=
        uq_evaluation.make_error_function_dwmse_1height(),
        error_function_for_flux_1var=
        uq_evaluation.make_error_function_flux_mse_1var()
    )
    print(SEPARATOR_STRING)

    discard_fractions = (
        result_table_xarray.coords[uq_evaluation.DISCARD_FRACTION_DIM].values
    )
    post_discard_errors = (
        result_table_xarray[uq_evaluation.POST_DISCARD_ERROR_KEY].values
    )

    for i in range(len(discard_fractions)):
        print('Error with discard fraction of {0:.4f} = {1:.4g}'.format(
            discard_fractions[i], post_discard_errors[i]
        ))

    print('\nMonotonicity fraction = {0:.4f}'.format(
        result_table_xarray.attrs[uq_evaluation.MONOTONICITY_FRACTION_KEY]
    ))
    print('Mean discard improvement = {0:.4f}'.format(
        result_table_xarray.attrs[uq_evaluation.MEAN_DISCARD_IMPROVEMENT_KEY]
    ))
    print(SEPARATOR_STRING)

    t = result_table_xarray
    scalar_target_names = t.coords[uq_evaluation.SCALAR_FIELD_DIM].values

    for k in range(len(scalar_target_names)):
        print('Variable = {0:s} ... MF = {1:f} ... DI = {2:f}'.format(
            scalar_target_names[k],
            t[uq_evaluation.SCALAR_MONOTONICITY_FRACTION_KEY].values[k],
            t[uq_evaluation.SCALAR_MEAN_DISCARD_IMPROVEMENT_KEY].values[k]
        ))

    print(SEPARATOR_STRING)
    vector_target_names = t.coords[uq_evaluation.VECTOR_FIELD_DIM].values
    heights_m_agl = t.coords[uq_evaluation.HEIGHT_DIM].values

    for k in range(len(vector_target_names)):
        for j in range(len(heights_m_agl)):
            print((
                'Variable = {0:s} at {1:d} m AGL ... MF = {2:f} ... DI = {3:f}'
            ).format(
                vector_target_names[k], int(numpy.round(heights_m_agl[j])),
                t[uq_evaluation.VECTOR_MONOTONICITY_FRACTION_KEY].values[k, j],
                t[uq_evaluation.VECTOR_MEAN_DISCARD_IMPROVEMENT_KEY].values[
                    k, j
                ]
            ))

        print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            t.coords[uq_evaluation.AUX_TARGET_FIELD_DIM].values
        )
        aux_predicted_field_names = (
            t.coords[uq_evaluation.AUX_PREDICTED_FIELD_DIM].values
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
            t[uq_evaluation.AUX_MONOTONICITY_FRACTION_KEY].values[k],
            t[uq_evaluation.AUX_MEAN_DISCARD_IMPROVEMENT_KEY].values[k]
        ))

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    uq_evaluation.write_discard_results(
        discard_test_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        discard_fractions=numpy.array(
            getattr(INPUT_ARG_OBJECT, DISCARD_FRACTIONS_ARG_NAME), dtype=float
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
