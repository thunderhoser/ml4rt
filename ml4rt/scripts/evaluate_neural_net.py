"""Evaluates trained neural net."""

import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import prediction_io
from ml4rt.utils import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

METRES_TO_MICRONS = 1e6

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
NUM_BOOTSTRAP_REPS_ARG_NAME = 'num_bootstrap_reps'
NUM_HEATING_RATE_BINS_ARG_NAME = 'num_heating_rate_bins'
HEATING_RATE_LIMITS_ARG_NAME = 'heating_rate_limits_k_day01'
HEATING_RATE_LIMITS_PRCTILE_ARG_NAME = 'heating_rate_limits_percentile'
NUM_RAW_FLUX_BINS_ARG_NAME = 'num_raw_flux_bins'
RAW_FLUX_LIMITS_ARG_NAME = 'raw_flux_limits_w_m02'
RAW_FLUX_LIMITS_PRCTILE_ARG_NAME = 'raw_flux_limits_percentile'
NUM_NET_FLUX_BINS_ARG_NAME = 'num_net_flux_bins'
NET_FLUX_LIMITS_ARG_NAME = 'net_flux_limits_w_m02'
NET_FLUX_LIMITS_PRCTILE_ARG_NAME = 'net_flux_limits_percentile'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predicted and actual target values.  Will '
    'be read by `prediction_io.read_file`.'
)
NUM_BOOTSTRAP_REPS_HELP_STRING = 'Number of bootstrap replicates.'
NUM_HEATING_RATE_BINS_HELP_STRING = (
    'Number of heating-rate bins for reliability curves.'
)
HEATING_RATE_LIMITS_HELP_STRING = (
    'Min and max heating rates (list of two values, in Kelvins per day) for '
    'reliability curves.  If you want to specify min/max by percentiles '
    'instead -- chosen independently at each height -- leave this argument '
    'alone.'
)
HEATING_RATE_LIMITS_PRCTILE_HELP_STRING = (
    'Min and max heating-rate percentiles -- taken independently at each '
    'height -- for reliability curves.  If you want to specify min/max by '
    'physical values instead, leave this argument alone.'
)
NUM_RAW_FLUX_BINS_HELP_STRING = (
    'Number of bins for reliability curves on raw flux variables (surface '
    'downwelling and TOA upwelling).'
)
RAW_FLUX_LIMITS_HELP_STRING = (
    'Min and max fluxes (list of two values, in Watts per m^2) for reliability '
    'curves on raw flux variables (surface downwelling and TOA upwelling).  If '
    'you want to specify min/max by percentiles instead -- chosen '
    'independently for each variable -- leave this argument alone.'
)
RAW_FLUX_LIMITS_PRCTILE_HELP_STRING = (
    'Min and max flux percentiles -- taken independently for each raw flux '
    'variable (surface downwelling and TOA upwelling) -- for reliability '
    'curves.  If you want to specify min/max by physical values instead, leave '
    'this argument alone.'
)
NUM_NET_FLUX_BINS_HELP_STRING = 'Number of net-flux bins for reliability curve.'
NET_FLUX_LIMITS_HELP_STRING = (
    'Min and max net flux (list of two values, in Watts per m^2) for '
    'reliability curve.  If you want to specify min/max by percentiles '
    'instead, leave this argument alone.'
)
NET_FLUX_LIMITS_PRCTILE_HELP_STRING = (
    'Min and max net-flux percentiles for reliability curve.  If you want to '
    'specify min/max by physical values instead, leave this argument alone.'
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
    '--' + HEATING_RATE_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=HEATING_RATE_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HEATING_RATE_LIMITS_PRCTILE_ARG_NAME, type=float, nargs=2,
    required=False, default=[0.5, 99.5],
    help=HEATING_RATE_LIMITS_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_RAW_FLUX_BINS_ARG_NAME, type=int, required=False, default=20,
    help=NUM_RAW_FLUX_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RAW_FLUX_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=RAW_FLUX_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + RAW_FLUX_LIMITS_PRCTILE_ARG_NAME, type=float, nargs=2,
    required=False, default=[0.5, 99.5],
    help=RAW_FLUX_LIMITS_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_NET_FLUX_BINS_ARG_NAME, type=int, required=False, default=20,
    help=NUM_NET_FLUX_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NET_FLUX_LIMITS_ARG_NAME, type=float, nargs=2, required=False,
    default=[1, -1], help=NET_FLUX_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NET_FLUX_LIMITS_PRCTILE_ARG_NAME, type=float, nargs=2,
    required=False, default=[0.5, 99.5],
    help=NET_FLUX_LIMITS_PRCTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(prediction_file_name, num_bootstrap_reps, num_heating_rate_bins,
         heating_rate_limits_k_day01, heating_rate_limits_percentile,
         num_raw_flux_bins, raw_flux_limits_w_m02, raw_flux_limits_percentile,
         num_net_flux_bins, net_flux_limits_w_m02, net_flux_limits_percentile,
         output_dir_name):
    """Evaluates trained neural net.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param num_bootstrap_reps: Same.
    :param num_heating_rate_bins: Same.
    :param heating_rate_limits_k_day01: Same.
    :param heating_rate_limits_percentile: Same.
    :param num_raw_flux_bins: Same.
    :param raw_flux_limits_w_m02: Same.
    :param raw_flux_limits_percentile: Same.
    :param num_net_flux_bins: Same.
    :param net_flux_limits_w_m02: Same.
    :param net_flux_limits_percentile: Same.
    :param output_dir_name: Same.
    """

    min_heating_rate_k_day01 = heating_rate_limits_k_day01[0]
    max_heating_rate_k_day01 = heating_rate_limits_k_day01[1]
    min_heating_rate_percentile = heating_rate_limits_percentile[0]
    max_heating_rate_percentile = heating_rate_limits_percentile[1]

    if min_heating_rate_k_day01 >= max_heating_rate_k_day01:
        min_heating_rate_k_day01 = None
        max_heating_rate_k_day01 = None

        error_checking.assert_is_leq(min_heating_rate_percentile, 10.)
        error_checking.assert_is_geq(max_heating_rate_percentile, 90.)
    else:
        min_heating_rate_percentile = None
        max_heating_rate_percentile = None

    min_raw_flux_w_m02 = raw_flux_limits_w_m02[0]
    max_raw_flux_w_m02 = raw_flux_limits_w_m02[1]
    min_raw_flux_percentile = raw_flux_limits_percentile[0]
    max_raw_flux_percentile = raw_flux_limits_percentile[1]

    if min_raw_flux_w_m02 >= max_raw_flux_w_m02:
        min_raw_flux_w_m02 = None
        max_raw_flux_w_m02 = None

        error_checking.assert_is_leq(min_raw_flux_percentile, 10.)
        error_checking.assert_is_geq(max_raw_flux_percentile, 90.)
    else:
        min_raw_flux_percentile = None
        max_raw_flux_percentile = None

    min_net_flux_w_m02 = net_flux_limits_w_m02[0]
    max_net_flux_w_m02 = net_flux_limits_w_m02[1]
    min_net_flux_percentile = net_flux_limits_percentile[0]
    max_net_flux_percentile = net_flux_limits_percentile[1]

    if min_net_flux_w_m02 >= max_net_flux_w_m02:
        min_net_flux_w_m02 = None
        max_net_flux_w_m02 = None

        error_checking.assert_is_leq(min_net_flux_percentile, 10.)
        error_checking.assert_is_geq(max_net_flux_percentile, 90.)
    else:
        min_net_flux_percentile = None
        max_net_flux_percentile = None

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
        num_raw_flux_bins=num_raw_flux_bins,
        min_raw_flux_w_m02=min_raw_flux_w_m02,
        max_raw_flux_w_m02=max_raw_flux_w_m02,
        min_raw_flux_percentile=min_raw_flux_percentile,
        max_raw_flux_percentile=max_raw_flux_percentile,
        num_net_flux_bins=num_net_flux_bins,
        min_net_flux_w_m02=min_net_flux_w_m02,
        max_net_flux_w_m02=max_net_flux_w_m02,
        min_net_flux_percentile=min_net_flux_percentile,
        max_net_flux_percentile=max_net_flux_percentile
    )
    print(SEPARATOR_STRING)

    rtx = result_table_xarray
    scalar_target_names = rtx.coords[evaluation.SCALAR_FIELD_DIM].values
    wavelengths_microns = (
        METRES_TO_MICRONS * rtx.coords[evaluation.WAVELENGTH_DIM].values
    )

    for t in range(len(scalar_target_names)):
        for w in range(len(wavelengths_microns)):
            print((
                'Variable = "{0:s}" at {1:.2f} microns ... '
                'stdev of target and predicted values = {2:f}, {3:f} ... '
                'MSE and skill score = {4:f}, {5:f} ... '
                'MAE and skill score = {6:f}, {7:f} ... '
                'bias = {8:f} ... correlation = {9:f} ... KGE = {10:f}'
            ).format(
                scalar_target_names[t],
                wavelengths_microns[w],
                numpy.nanmean(rtx[evaluation.SCALAR_TARGET_STDEV_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.SCALAR_PREDICTION_STDEV_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.SCALAR_MSE_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.SCALAR_MSE_SKILL_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.SCALAR_MAE_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.SCALAR_MAE_SKILL_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.SCALAR_BIAS_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.SCALAR_CORRELATION_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.SCALAR_KGE_KEY].values[w, t, :])
            ))

        print(SEPARATOR_STRING)

    vector_target_names = rtx.coords[evaluation.VECTOR_FIELD_DIM].values
    heights_m_agl = rtx.coords[evaluation.HEIGHT_DIM].values

    for t in range(len(vector_target_names)):
        for w in range(len(wavelengths_microns)):
            print((
                'Variable = "{0:s}" at {1:.2f} microns ... PRMSE = {2:f}'
            ).format(
                vector_target_names[t],
                wavelengths_microns[w],
                numpy.nanmean(rtx[evaluation.VECTOR_PRMSE_KEY].values[w, t, :])
            ))

    print(SEPARATOR_STRING)

    for t in range(len(vector_target_names)):
        for w in range(len(wavelengths_microns)):
            for h in range(len(heights_m_agl)):
                print((
                    'Variable = "{0:s}" at {1:.2f} microns and {2:d} m AGL ... '
                    'stdev of target and predicted values = {3:f}, {4:f} ... '
                    'MSE and skill score = {5:f}, {6:f} ... '
                    'MAE and skill score = {7:f}, {8:f} ... bias = {9:f} ... '
                    'correlation = {10:f} ... KGE = {11:f}'
                ).format(
                    vector_target_names[t],
                    wavelengths_microns[w],
                    int(numpy.round(heights_m_agl[h])),
                    numpy.nanmean(rtx[evaluation.VECTOR_TARGET_STDEV_KEY].values[h, w, t, :]),
                    numpy.nanmean(rtx[evaluation.VECTOR_PREDICTION_STDEV_KEY].values[h, w, t, :]),
                    numpy.nanmean(rtx[evaluation.VECTOR_MSE_KEY].values[h, w, t, :]),
                    numpy.nanmean(rtx[evaluation.VECTOR_MSE_SKILL_KEY].values[h, w, t, :]),
                    numpy.nanmean(rtx[evaluation.VECTOR_MAE_KEY].values[h, w, t, :]),
                    numpy.nanmean(rtx[evaluation.VECTOR_MAE_SKILL_KEY].values[h, w, t, :]),
                    numpy.nanmean(rtx[evaluation.VECTOR_BIAS_KEY].values[h, w, t, :]),
                    numpy.nanmean(rtx[evaluation.VECTOR_CORRELATION_KEY].values[h, w, t, :]),
                    numpy.nanmean(rtx[evaluation.VECTOR_KGE_KEY].values[h, w, t, :])
                ))

            print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            rtx.coords[evaluation.AUX_TARGET_FIELD_DIM].values
        )
        aux_predicted_field_names = (
            rtx.coords[evaluation.AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_target_field_names = []
        aux_predicted_field_names = []

    for t in range(len(aux_target_field_names)):
        for w in range(len(wavelengths_microns)):
            print((
                'Target variable = "{0:s}" at {1:.2f} microns ... '
                'predicted variable = "{2:s}" at {1:.2f} microns ... '
                'stdev of target and predicted values = {3:f}, {4:f} ... '
                'MSE and skill score = {5:f}, {6:f} ... '
                'MAE and skill score = {7:f}, {8:f} ... '
                'bias = {9:f} ... correlation = {10:f} ... KGE = {11:f}'
            ).format(
                aux_target_field_names[t],
                wavelengths_microns[w],
                aux_predicted_field_names[t],
                numpy.nanmean(rtx[evaluation.AUX_TARGET_STDEV_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.AUX_PREDICTION_STDEV_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.AUX_MSE_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.AUX_MSE_SKILL_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.AUX_MAE_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.AUX_MAE_SKILL_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.AUX_BIAS_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.AUX_CORRELATION_KEY].values[w, t, :]),
                numpy.nanmean(rtx[evaluation.AUX_KGE_KEY].values[w, t, :])
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
        heating_rate_limits_k_day01=numpy.array(
            getattr(INPUT_ARG_OBJECT, HEATING_RATE_LIMITS_ARG_NAME),
            dtype=float
        ),
        heating_rate_limits_percentile=numpy.array(
            getattr(INPUT_ARG_OBJECT, HEATING_RATE_LIMITS_PRCTILE_ARG_NAME),
            dtype=float
        ),
        num_raw_flux_bins=getattr(INPUT_ARG_OBJECT, NUM_RAW_FLUX_BINS_ARG_NAME),
        raw_flux_limits_w_m02=numpy.array(
            getattr(INPUT_ARG_OBJECT, RAW_FLUX_LIMITS_ARG_NAME),
            dtype=float
        ),
        raw_flux_limits_percentile=numpy.array(
            getattr(INPUT_ARG_OBJECT, RAW_FLUX_LIMITS_PRCTILE_ARG_NAME),
            dtype=float
        ),
        num_net_flux_bins=getattr(INPUT_ARG_OBJECT, NUM_NET_FLUX_BINS_ARG_NAME),
        net_flux_limits_w_m02=numpy.array(
            getattr(INPUT_ARG_OBJECT, NET_FLUX_LIMITS_ARG_NAME),
            dtype=float
        ),
        net_flux_limits_percentile=numpy.array(
            getattr(INPUT_ARG_OBJECT, NET_FLUX_LIMITS_PRCTILE_ARG_NAME),
            dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
