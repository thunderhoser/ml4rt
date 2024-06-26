"""Evaluates trained neural net."""

import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4rt.utils import spread_skill_utils as ss_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
METRES_TO_MICRONS = 1e6

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
NUM_HEATING_RATE_BINS_ARG_NAME = 'num_heating_rate_bins'
HEATING_RATE_LIMITS_ARG_NAME = 'heating_rate_limits_k_day01'
HEATING_RATE_LIMITS_PRCTILE_ARG_NAME = 'heating_rate_limits_percentile'
NUM_RAW_FLUX_BINS_ARG_NAME = 'num_raw_flux_bins'
RAW_FLUX_LIMITS_ARG_NAME = 'raw_flux_limits_w_m02'
RAW_FLUX_LIMITS_PRCTILE_ARG_NAME = 'raw_flux_limits_percentile'
NUM_NET_FLUX_BINS_ARG_NAME = 'num_net_flux_bins'
NET_FLUX_LIMITS_ARG_NAME = 'net_flux_limits_w_m02'
NET_FLUX_LIMITS_PRCTILE_ARG_NAME = 'net_flux_limits_percentile'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predicted and actual target values.  Will '
    'be read by `prediction_io.read_file`.'
)
NUM_HEATING_RATE_BINS_HELP_STRING = (
    'Number of bins for predictive stdev of heating rate.'
)
HEATING_RATE_LIMITS_HELP_STRING = (
    'Min and max bin values for predictive stdev of heating rate.  If you want '
    'to specify min/max by percentiles instead -- chosen independently at each '
    'height -- leave this argument alone.'
)
HEATING_RATE_LIMITS_PRCTILE_HELP_STRING = (
    'Min and max percentiles -- taken independently at each height -- used to '
    'create bin values for predictive stdev of heating rate.  If you want to '
    'specify min/max by physical values instead, leave this argument alone.'
)
NUM_RAW_FLUX_BINS_HELP_STRING = (
    'Number of bins for predictive stdev of raw flux variable (surface '
    'downwelling or TOA upwelling).'
)
RAW_FLUX_LIMITS_HELP_STRING = (
    'Min and max bin values for predictive stdev of raw flux variable.  If you '
    'want to specify min/max by percentiles instead -- chosen independently '
    'for each raw flux variable (surface downwelling and TOA upwelling) -- '
    'leave this argument alone.'
)
RAW_FLUX_LIMITS_PRCTILE_HELP_STRING = (
    'Min and max percentiles -- taken independently for each raw flux variable '
    '(surface downwelling and TOA upwelling) -- used to create bin values for '
    'predictive stdev of flux.  If you want to specify min/max by physical '
    'values instead, leave this argument alone.'
)
NUM_NET_FLUX_BINS_HELP_STRING = (
    'Number of bins for predictive stdev of net flux.'
)
NET_FLUX_LIMITS_HELP_STRING = (
    'Min and max bin values for predictive stdev of net flux.  If you want to '
    'specify min/max by percentiles instead, leave this argument alone.'
)
NET_FLUX_LIMITS_PRCTILE_HELP_STRING = (
    'Min and max percentiles, used to create bin values for predictive stdev '
    'of net flux.  If you want to specify min/max by physical values instead, '
    'leave this argument alone.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output (NetCDF) file.  Results will be written here by '
    '`spread_skill_utils.write_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HEATING_RATE_BINS_ARG_NAME, type=int, required=False,
    default=100, help=NUM_HEATING_RATE_BINS_HELP_STRING
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
    '--' + NUM_RAW_FLUX_BINS_ARG_NAME, type=int, required=False, default=100,
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
    '--' + NUM_NET_FLUX_BINS_ARG_NAME, type=int, required=False, default=100,
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
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_file_name, num_heating_rate_bins,
         heating_rate_limits_k_day01, heating_rate_limits_percentile,
         num_raw_flux_bins, raw_flux_limits_w_m02, raw_flux_limits_percentile,
         num_net_flux_bins, net_flux_limits_w_m02, net_flux_limits_percentile,
         output_file_name):
    """Evaluates trained neural net.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param num_heating_rate_bins: Same.
    :param heating_rate_limits_k_day01: Same.
    :param heating_rate_limits_percentile: Same.
    :param num_raw_flux_bins: Same.
    :param raw_flux_limits_w_m02: Same.
    :param raw_flux_limits_percentile: Same.
    :param num_net_flux_bins: Same.
    :param net_flux_limits_w_m02: Same.
    :param net_flux_limits_percentile: Same.
    :param output_file_name: Same.
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

    result_table_xarray = ss_utils.get_results_all_vars(
        prediction_file_name=prediction_file_name,
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
    scalar_target_names = rtx.coords[ss_utils.SCALAR_FIELD_DIM].values
    wavelengths_microns = (
        METRES_TO_MICRONS * rtx.coords[ss_utils.WAVELENGTH_DIM].values
    )

    for t in range(len(scalar_target_names)):
        for w in range(len(wavelengths_microns)):
            print((
                'Variable = "{0:s}" at {1:.2f} microns ... '
                'SSREL = {2:f} ... SSRAT = {3:f}'
            ).format(
                scalar_target_names[t],
                wavelengths_microns[w],
                rtx[ss_utils.SCALAR_SSREL_KEY].values[t, w],
                rtx[ss_utils.SCALAR_SSRAT_KEY].values[t, w]
            ))

        print(SEPARATOR_STRING)

    vector_target_names = rtx.coords[ss_utils.VECTOR_FIELD_DIM].values
    heights_m_agl = rtx.coords[ss_utils.HEIGHT_DIM].values

    for t in range(len(vector_target_names)):
        for w in range(len(wavelengths_microns)):
            print((
                'Variable = "{0:s}" at {1:.2f} microns ... '
                'SSREL = {2:f} ... SSRAT = {3:f}'
            ).format(
                vector_target_names[t],
                wavelengths_microns[w],
                rtx[ss_utils.VECTOR_FLAT_SSREL_KEY].values[t, w],
                rtx[ss_utils.VECTOR_FLAT_SSRAT_KEY].values[t, w]
            ))

        print(SEPARATOR_STRING)

    for t in range(len(vector_target_names)):
        for w in range(len(wavelengths_microns)):
            for h in range(len(heights_m_agl)):
                print((
                    'Variable = "{0:s}" at {1:.2f} microns and {2:d} m AGL ... '
                    'SSREL = {3:f} ... SSRAT = {4:f}'
                ).format(
                    vector_target_names[t],
                    wavelengths_microns[w],
                    int(numpy.round(heights_m_agl[h])),
                    rtx[ss_utils.VECTOR_SSREL_KEY].values[t, h, w],
                    rtx[ss_utils.VECTOR_SSRAT_KEY].values[t, h, w]
                ))

            print(SEPARATOR_STRING)

    try:
        aux_target_field_names = (
            rtx.coords[ss_utils.AUX_TARGET_FIELD_DIM].values
        )
        aux_predicted_field_names = (
            rtx.coords[ss_utils.AUX_PREDICTED_FIELD_DIM].values
        )
    except:
        aux_target_field_names = []
        aux_predicted_field_names = []

    for t in range(len(aux_target_field_names)):
        for w in range(len(wavelengths_microns)):
            print((
                'Target variable = "{0:s}" at {1:.2f} microns ... '
                'predicted variable = "{2:s}" at {1:.2f} microns ... '
                'SSREL = {3:f} ... SSRAT = {4:f}'
            ).format(
                aux_target_field_names[t],
                wavelengths_microns[w],
                aux_predicted_field_names[t],
                rtx[ss_utils.AUX_SSREL_KEY].values[t, w],
                rtx[ss_utils.AUX_SSRAT_KEY].values[t, w]
            ))

    print(SEPARATOR_STRING)

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    ss_utils.write_results(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
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
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
