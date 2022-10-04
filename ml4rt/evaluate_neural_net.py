"""Evaluates trained neural net."""

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
import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
NUM_BOOTSTRAP_REPS_ARG_NAME = 'num_bootstrap_reps'
NUM_HEATING_RATE_BINS_ARG_NAME = 'num_heating_rate_bins'
MIN_HEATING_RATE_ARG_NAME = 'min_heating_rate_k_day01'
MAX_HEATING_RATE_ARG_NAME = 'max_heating_rate_k_day01'
MIN_HEATING_RATE_PRCTILE_ARG_NAME = 'min_heating_rate_percentile'
MAX_HEATING_RATE_PRCTILE_ARG_NAME = 'max_heating_rate_percentile'
NUM_FLUX_BINS_ARG_NAME = 'num_flux_bins'
MIN_RAW_FLUX_ARG_NAME = 'min_raw_flux_w_m02'
MAX_RAW_FLUX_ARG_NAME = 'max_raw_flux_w_m02'
MIN_NET_FLUX_ARG_NAME = 'min_net_flux_w_m02'
MAX_NET_FLUX_ARG_NAME = 'max_net_flux_w_m02'
MIN_FLUX_PRCTILE_ARG_NAME = 'min_flux_percentile'
MAX_FLUX_PRCTILE_ARG_NAME = 'max_flux_percentile'
MIN_HEATING_TO_EVAL_ARG_NAME = 'min_actual_hr_to_eval_k_day01'
MAX_HEATING_TO_EVAL_ARG_NAME = 'max_actual_hr_to_eval_k_day01'
APPLY_MINMAX_PER_HEIGHT_ARG_NAME = 'apply_minmax_at_each_height'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predicted and actual target values.  Will '
    'be read by `prediction_io.read_file`.'
)
NUM_BOOTSTRAP_REPS_HELP_STRING = 'Number of bootstrap replicates.'

NUM_HEATING_RATE_BINS_HELP_STRING = (
    'Number of heating-rate bins for reliability curves.'
)
MIN_HEATING_RATE_HELP_STRING = (
    'Minimum heating rate (Kelvins per day) for reliability curves.  If you '
    'instead want minimum heating rate to be a percentile over the data -- '
    'chosen independently at each height -- leave this argument alone.'
)
MAX_HEATING_RATE_HELP_STRING = 'See documentation for `{0:s}`.'.format(
    MIN_HEATING_RATE_ARG_NAME
)
MIN_HEATING_RATE_PRCTILE_HELP_STRING = (
    'Determines minimum heating rate for reliability curves.  This percentile '
    '(ranging from 0...100) will be taken independently at each height.'
)
MAX_HEATING_RATE_PRCTILE_HELP_STRING = 'See documentation for `{0:s}`.'.format(
    MIN_HEATING_RATE_PRCTILE_ARG_NAME
)

NUM_FLUX_BINS_HELP_STRING = 'Number of flux bins for reliability curves.'
MIN_RAW_FLUX_HELP_STRING = (
    'Minimum raw flux (surface downwelling or TOA upwelling) for reliability '
    'curves.  If you instead want minimum flux to be a percentile over the '
    'data -- chosen independently for each flux variable -- leave this '
    'argument alone.'
)
MAX_RAW_FLUX_HELP_STRING = 'See documentation for `{0:s}`.'.format(
    MIN_RAW_FLUX_ARG_NAME
)
MIN_NET_FLUX_HELP_STRING = (
    'Minimum net flux for reliability curves.  If you instead want minimum net '
    'flux to be a percentile over the data, leave this argument alone.'
)
MAX_NET_FLUX_HELP_STRING = 'See documentation for `{0:s}`.'.format(
    MIN_NET_FLUX_ARG_NAME
)
MIN_FLUX_PRCTILE_HELP_STRING = (
    'Determines minimum flux for reliability curves.  This percentile (ranging '
    'from 0...100) will be taken independently for each flux variable.'
)
MAX_FLUX_PRCTILE_HELP_STRING = 'See documentation for `{0:s}`.'.format(
    MIN_FLUX_PRCTILE_ARG_NAME
)
MIN_HEATING_TO_EVAL_HELP_STRING = (
    'Will evaluate only profiles containing at least one actual heating rate '
    'in [min value, max value].'
)
MAX_HEATING_TO_EVAL_HELP_STRING = 'See documentation for `{0:s}`.'.format(
    MIN_HEATING_TO_EVAL_ARG_NAME
)
APPLY_MINMAX_PER_HEIGHT_HELP_STRING = (
    'Boolean flag.  If 1, will apply min and max actual heating rate '
    'independently to each height in each profile.  If 0, will just apply to '
    'each profile.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Evaluation scores will be written here by '
    '`evaluation.write_file`, to a file name determined by '
    '`evaluation.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BOOTSTRAP_REPS_ARG_NAME, type=int, required=True,
    help=NUM_BOOTSTRAP_REPS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HEATING_RATE_BINS_ARG_NAME, type=int, required=False, default=20,
    help=NUM_HEATING_RATE_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_HEATING_RATE_ARG_NAME, type=float, required=False, default=1,
    help=MIN_HEATING_RATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_HEATING_RATE_ARG_NAME, type=float, required=False, default=-1,
    help=MAX_HEATING_RATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_HEATING_RATE_PRCTILE_ARG_NAME, type=float, required=False,
    default=0.5, help=MIN_HEATING_RATE_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_HEATING_RATE_PRCTILE_ARG_NAME, type=float, required=False,
    default=99.5, help=MAX_HEATING_RATE_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_FLUX_BINS_ARG_NAME, type=int, required=False, default=20,
    help=NUM_FLUX_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_RAW_FLUX_ARG_NAME, type=float, required=False, default=1,
    help=MIN_RAW_FLUX_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_RAW_FLUX_ARG_NAME, type=float, required=False, default=-1,
    help=MAX_RAW_FLUX_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_NET_FLUX_ARG_NAME, type=float, required=False, default=1,
    help=MIN_NET_FLUX_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_NET_FLUX_ARG_NAME, type=float, required=False, default=-1,
    help=MAX_NET_FLUX_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_FLUX_PRCTILE_ARG_NAME, type=float, required=False,
    default=0.5, help=MIN_FLUX_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_FLUX_PRCTILE_ARG_NAME, type=float, required=False,
    default=99.5, help=MAX_FLUX_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_HEATING_TO_EVAL_ARG_NAME, type=float, required=False,
    default=-1e10, help=MIN_HEATING_TO_EVAL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_HEATING_TO_EVAL_ARG_NAME, type=float, required=False,
    default=1e10, help=MAX_HEATING_TO_EVAL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + APPLY_MINMAX_PER_HEIGHT_ARG_NAME, type=int, required=False,
    default=0, help=APPLY_MINMAX_PER_HEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(prediction_file_name, num_bootstrap_reps, num_heating_rate_bins,
         min_heating_rate_k_day01, max_heating_rate_k_day01,
         min_heating_rate_percentile, max_heating_rate_percentile,
         num_flux_bins, min_raw_flux_w_m02, max_raw_flux_w_m02,
         min_net_flux_w_m02, max_net_flux_w_m02,
         min_flux_percentile, max_flux_percentile,
         min_actual_hr_to_eval_k_day01, max_actual_hr_to_eval_k_day01,
         apply_minmax_at_each_height, output_dir_name):
    """Evaluates trained neural net.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param num_bootstrap_reps: Same.
    :param num_heating_rate_bins: Same.
    :param min_heating_rate_k_day01: Same.
    :param max_heating_rate_k_day01: Same.
    :param min_heating_rate_percentile: Same.
    :param max_heating_rate_percentile: Same.
    :param num_flux_bins: Same.
    :param min_raw_flux_w_m02: Same.
    :param max_raw_flux_w_m02: Same.
    :param min_net_flux_w_m02: Same.
    :param max_net_flux_w_m02: Same.
    :param min_flux_percentile: Same.
    :param max_flux_percentile: Same.
    :param min_actual_hr_to_eval_k_day01: Same.
    :param max_actual_hr_to_eval_k_day01: Same.
    :param apply_minmax_at_each_height: Same.
    :param output_dir_name: Same.
    """

    if min_actual_hr_to_eval_k_day01 < -1e9:
        min_actual_hr_to_eval_k_day01 = -numpy.inf
    if max_actual_hr_to_eval_k_day01 > 1e9:
        max_actual_hr_to_eval_k_day01 = numpy.inf

    if min_heating_rate_k_day01 >= max_heating_rate_k_day01:
        min_heating_rate_k_day01 = None
        max_heating_rate_k_day01 = None

        error_checking.assert_is_leq(min_heating_rate_percentile, 10.)
        error_checking.assert_is_geq(max_heating_rate_percentile, 90.)
    else:
        min_heating_rate_percentile = None
        max_heating_rate_percentile = None

    if (
            min_raw_flux_w_m02 >= max_raw_flux_w_m02 or
            min_net_flux_w_m02 >= max_net_flux_w_m02
    ):
        min_raw_flux_w_m02 = None
        max_raw_flux_w_m02 = None
        min_net_flux_w_m02 = None
        max_net_flux_w_m02 = None

        error_checking.assert_is_leq(min_flux_percentile, 10.)
        error_checking.assert_is_geq(max_flux_percentile, 90.)
    else:
        error_checking.assert_is_geq(min_raw_flux_w_m02, 0.)

        min_flux_percentile = None
        max_flux_percentile = None

    file_metadata_dict = prediction_io.file_name_to_metadata(
        prediction_file_name
    )
    output_file_name = evaluation.find_file(
        directory_name=output_dir_name,
        zenith_angle_bin=file_metadata_dict[prediction_io.ZENITH_ANGLE_BIN_KEY],
        albedo_bin=file_metadata_dict[prediction_io.ALBEDO_BIN_KEY],
        month=file_metadata_dict[prediction_io.MONTH_KEY],
        shortwave_sfc_down_flux_bin=
        file_metadata_dict[prediction_io.SHORTWAVE_SFC_DOWN_FLUX_BIN_KEY],
        aerosol_optical_depth_bin=
        file_metadata_dict[prediction_io.AEROSOL_OPTICAL_DEPTH_BIN_KEY],
        surface_temp_bin=file_metadata_dict[prediction_io.SURFACE_TEMP_BIN_KEY],
        longwave_sfc_down_flux_bin=
        file_metadata_dict[prediction_io.LONGWAVE_SFC_DOWN_FLUX_BIN_KEY],
        longwave_toa_up_flux_bin=
        file_metadata_dict[prediction_io.LONGWAVE_TOA_UP_FLUX_BIN_KEY],
        grid_row=file_metadata_dict[prediction_io.GRID_ROW_KEY],
        grid_column=file_metadata_dict[prediction_io.GRID_COLUMN_KEY],
        raise_error_if_missing=False
    )

    result_table_xarray = evaluation.get_scores_all_variables(
        prediction_file_name=prediction_file_name,
        num_bootstrap_reps=num_bootstrap_reps,
        num_heating_rate_bins=num_heating_rate_bins,
        min_heating_rate_k_day01=min_heating_rate_k_day01,
        max_heating_rate_k_day01=max_heating_rate_k_day01,
        min_heating_rate_percentile=min_heating_rate_percentile,
        max_heating_rate_percentile=max_heating_rate_percentile,
        num_flux_bins=num_flux_bins,
        min_raw_flux_w_m02=min_raw_flux_w_m02,
        max_raw_flux_w_m02=max_raw_flux_w_m02,
        min_net_flux_w_m02=min_net_flux_w_m02,
        max_net_flux_w_m02=max_net_flux_w_m02,
        min_flux_percentile=min_flux_percentile,
        max_flux_percentile=max_flux_percentile,
        min_actual_hr_to_eval_k_day01=min_actual_hr_to_eval_k_day01,
        max_actual_hr_to_eval_k_day01=max_actual_hr_to_eval_k_day01,
        apply_minmax_at_each_height=apply_minmax_at_each_height
    )
    print(SEPARATOR_STRING)

    t = result_table_xarray
    scalar_target_names = t.coords[evaluation.SCALAR_FIELD_DIM].values

    for k in range(len(scalar_target_names)):
        print((
            'Variable = "{0:s}" ... stdev of target and predicted values = '
            '{1:f}, {2:f} ... MSE and skill score = {3:f}, {4:f} ... '
            'MAE and skill score = {5:f}, {6:f} ... bias = {7:f} ... '
            'correlation = {8:f} ... KGE = {9:f}'
        ).format(
            scalar_target_names[k],
            numpy.nanmean(t[evaluation.SCALAR_TARGET_STDEV_KEY].values[k, :]),
            numpy.nanmean(
                t[evaluation.SCALAR_PREDICTION_STDEV_KEY].values[k, :]
            ),
            numpy.nanmean(t[evaluation.SCALAR_MSE_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.SCALAR_MSE_SKILL_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.SCALAR_MAE_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.SCALAR_MAE_SKILL_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.SCALAR_BIAS_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.SCALAR_CORRELATION_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.SCALAR_KGE_KEY].values[k, :])
        ))

    print(SEPARATOR_STRING)

    vector_target_names = t.coords[evaluation.VECTOR_FIELD_DIM].values
    heights_m_agl = t.coords[evaluation.HEIGHT_DIM].values

    for k in range(len(vector_target_names)):
        print('Variable = "{0:s}" ... PRMSE = {1:f}'.format(
            vector_target_names[k],
            numpy.nanmean(t[evaluation.VECTOR_PRMSE_KEY].values[k, :])
        ))

    print(SEPARATOR_STRING)

    for k in range(len(vector_target_names)):
        for j in range(len(heights_m_agl)):
            print((
                'Variable = "{0:s}" at {1:d} m AGL ... '
                'stdev of target and predicted values = {2:f}, {3:f} ... '
                'MSE and skill score = {4:f}, {5:f} ... '
                'MAE and skill score = {6:f}, {7:f} ... bias = {8:f} ... '
                'correlation = {9:f} ... KGE = {10:f}'
            ).format(
                vector_target_names[k], int(numpy.round(heights_m_agl[j])),
                numpy.nanmean(
                    t[evaluation.VECTOR_TARGET_STDEV_KEY].values[j, k, :]
                ),
                numpy.nanmean(
                    t[evaluation.VECTOR_PREDICTION_STDEV_KEY].values[j, k, :]
                ),
                numpy.nanmean(t[evaluation.VECTOR_MSE_KEY].values[j, k, :]),
                numpy.nanmean(
                    t[evaluation.VECTOR_MSE_SKILL_KEY].values[j, k, :]
                ),
                numpy.nanmean(t[evaluation.VECTOR_MAE_KEY].values[j, k, :]),
                numpy.nanmean(
                    t[evaluation.VECTOR_MAE_SKILL_KEY].values[j, k, :]
                ),
                numpy.nanmean(t[evaluation.VECTOR_BIAS_KEY].values[j, k, :]),
                numpy.nanmean(
                    t[evaluation.VECTOR_CORRELATION_KEY].values[j, k, :]
                ),
                numpy.nanmean(t[evaluation.VECTOR_KGE_KEY].values[j, k, :])
            ))

        print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            t.coords[evaluation.AUX_TARGET_FIELD_DIM].values
        )
        aux_predicted_field_names = (
            t.coords[evaluation.AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_target_field_names = []
        aux_predicted_field_names = []

    for k in range(len(aux_target_field_names)):
        print((
            'Target variable = "{0:s}" ... predicted variable = "{1:s}" ... '
            'stdev of target and predicted values = {2:f}, {3:f} ... '
            'MSE and skill score = {4:f}, {5:f} ... '
            'MAE and skill score = {6:f}, {7:f} ... bias = {8:f} ... '
            'correlation = {9:f} ... KGE = {10:f}'
        ).format(
            aux_target_field_names[k], aux_predicted_field_names[k],
            numpy.nanmean(t[evaluation.AUX_TARGET_STDEV_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_PREDICTION_STDEV_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_MSE_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_MSE_SKILL_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_MAE_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_MAE_SKILL_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_BIAS_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_CORRELATION_KEY].values[k, :]),
            numpy.nanmean(t[evaluation.AUX_KGE_KEY].values[k, :])
        ))

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    evaluation.write_file(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_bootstrap_reps=getattr(
            INPUT_ARG_OBJECT, NUM_BOOTSTRAP_REPS_ARG_NAME
        ),
        num_heating_rate_bins=getattr(
            INPUT_ARG_OBJECT, NUM_HEATING_RATE_BINS_ARG_NAME
        ),
        min_heating_rate_k_day01=getattr(
            INPUT_ARG_OBJECT, MIN_HEATING_RATE_ARG_NAME
        ),
        max_heating_rate_k_day01=getattr(
            INPUT_ARG_OBJECT, MAX_HEATING_RATE_ARG_NAME
        ),
        min_heating_rate_percentile=getattr(
            INPUT_ARG_OBJECT, MIN_HEATING_RATE_PRCTILE_ARG_NAME
        ),
        max_heating_rate_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_HEATING_RATE_PRCTILE_ARG_NAME
        ),
        num_flux_bins=getattr(INPUT_ARG_OBJECT, NUM_FLUX_BINS_ARG_NAME),
        min_raw_flux_w_m02=getattr(INPUT_ARG_OBJECT, MIN_RAW_FLUX_ARG_NAME),
        max_raw_flux_w_m02=getattr(INPUT_ARG_OBJECT, MAX_RAW_FLUX_ARG_NAME),
        min_net_flux_w_m02=getattr(INPUT_ARG_OBJECT, MIN_NET_FLUX_ARG_NAME),
        max_net_flux_w_m02=getattr(INPUT_ARG_OBJECT, MAX_NET_FLUX_ARG_NAME),
        min_flux_percentile=getattr(
            INPUT_ARG_OBJECT, MIN_FLUX_PRCTILE_ARG_NAME
        ),
        max_flux_percentile=getattr(
            INPUT_ARG_OBJECT, MAX_FLUX_PRCTILE_ARG_NAME
        ),
        min_actual_hr_to_eval_k_day01=getattr(
            INPUT_ARG_OBJECT, MIN_HEATING_TO_EVAL_ARG_NAME
        ),
        max_actual_hr_to_eval_k_day01=getattr(
            INPUT_ARG_OBJECT, MAX_HEATING_TO_EVAL_ARG_NAME
        ),
        apply_minmax_at_each_height=bool(getattr(
            INPUT_ARG_OBJECT, APPLY_MINMAX_PER_HEIGHT_ARG_NAME
        )),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
