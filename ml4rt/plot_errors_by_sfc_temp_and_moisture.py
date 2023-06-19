"""Plots error metrics vs. surface temperature and surface humidity."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import gg_plotting_utils
import imagemagick_utils
import example_io
import prediction_io
import example_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
KG_TO_GRAMS = 1000.

MIN_SURFACE_TEMP_KELVINS = 190.
MAX_SURFACE_TEMP_KELVINS = 330.
MAX_SURFACE_SPEC_HUMIDITY_KG_KG01 = 0.024

LONGWAVE_ALL_FLUX_NAME = 'all_longwave_flux_w_m02'
LONGWAVE_NET_FLUX_NAME = 'net_longwave_flux_w_m02'

STATISTIC_NAMES_BASIC = [
    'longwave_mae', 'longwave_near_sfc_mae',
    'longwave_bias', 'longwave_near_sfc_bias',
    'longwave_all_flux_mae', 'longwave_net_flux_mae',
    'longwave_net_flux_bias', 'num_examples'
]
STATISTIC_NAMES_DETAILED = [
    'longwave_mae', 'longwave_near_sfc_mae',
    'longwave_bias', 'longwave_near_sfc_bias',
    'longwave_down_flux_mae', 'longwave_down_flux_bias',
    'longwave_up_flux_mae', 'longwave_up_flux_bias',
    'longwave_net_flux_mae', 'longwave_net_flux_bias',
    'num_examples'
]
STATISTIC_NAME_TO_FANCY = {
    'longwave_mae': r'HR MAE (K day$^{-1}$)',
    'longwave_near_sfc_mae': r'Near-surface HR MAE (K day$^{-1}$)',
    'longwave_bias': r'HR bias (K day$^{-1}$)',
    'longwave_near_sfc_bias': r'Near-surface HR bias (K day$^{-1}$)',
    'longwave_down_flux_mae': r'Downwelling-flux MAE (W m$^{-2}$)',
    'longwave_down_flux_bias': r'Downwelling-flux bias (W m$^{-2}$)',
    'longwave_up_flux_mae': r'Upwelling-flux MAE (W m$^{-2}$)',
    'longwave_up_flux_bias': r'Upwelling-flux bias (W m$^{-2}$)',
    'longwave_net_flux_mae': r'Net-flux MAE (W m$^{-2}$)',
    'longwave_net_flux_bias': r'Net-flux bias (W m$^{-2}$)',
    'longwave_all_flux_mae': r'All-flux MAE (W m$^{-2}$)',
    'num_examples': 'Number of examples'
}
STATISTIC_NAME_TO_FANCY_FRACTIONAL = {
    'longwave_mae': 'Relative HR MAE (%)',
    'longwave_near_sfc_mae': 'Relative near-surface HR MAE (%)',
    'longwave_bias': 'Relative HR bias (%)',
    'longwave_near_sfc_bias': 'Relative near-surface HR bias (%)',
    'longwave_down_flux_mae': 'Relative downwelling-flux MAE (%)',
    'longwave_down_flux_bias': 'Relative downwelling-flux bias (%)',
    'longwave_up_flux_mae': 'Relative upwelling-flux MAE (%)',
    'longwave_up_flux_bias': 'Relative upwelling-flux bias (%)',
    'longwave_net_flux_mae': 'Relative net-flux MAE (%)',
    'longwave_net_flux_bias': 'Relative net-flux bias (%)',
    'longwave_all_flux_mae': 'Relative all-flux MAE (%)',
    'num_examples': 'Number of examples'
}
STATISTIC_NAME_TO_TARGET_NAME = {
    'longwave_mae': example_utils.LONGWAVE_HEATING_RATE_NAME,
    'longwave_near_sfc_mae': example_utils.LONGWAVE_HEATING_RATE_NAME,
    'longwave_bias': example_utils.LONGWAVE_HEATING_RATE_NAME,
    'longwave_near_sfc_bias': example_utils.LONGWAVE_HEATING_RATE_NAME,
    'longwave_down_flux_mae': example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME,
    'longwave_down_flux_bias': example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME,
    'longwave_up_flux_mae': example_utils.LONGWAVE_TOA_UP_FLUX_NAME,
    'longwave_up_flux_bias': example_utils.LONGWAVE_TOA_UP_FLUX_NAME,
    'longwave_net_flux_mae': LONGWAVE_NET_FLUX_NAME,
    'longwave_net_flux_bias': LONGWAVE_NET_FLUX_NAME,
    'longwave_all_flux_mae': LONGWAVE_ALL_FLUX_NAME,
    'num_examples': ''
}
STATISTIC_NAME_TO_TARGET_HEIGHT_INDEX = {
    'longwave_mae': -1,
    'longwave_near_sfc_mae': 0,
    'longwave_bias': -1,
    'longwave_near_sfc_bias': 0,
    'longwave_down_flux_mae': -1,
    'longwave_down_flux_bias': -1,
    'longwave_up_flux_mae': -1,
    'longwave_up_flux_bias': -1,
    'longwave_net_flux_mae': -1,
    'longwave_net_flux_bias': -1,
    'longwave_all_flux_mae': -1,
    'num_examples': -1
}

NUM_PANEL_COLUMNS = 2
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

MAIN_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis', lut=20)
BIAS_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='seismic', lut=20)
NUM_EXAMPLES_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='cividis', lut=20)

MAIN_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))
BIAS_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))
NUM_EXAMPLES_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
NUM_TEMPERATURE_BINS_ARG_NAME = 'num_temperature_bins'
MIN_TEMP_LAPSE_RATE_ARG_NAME = 'min_temp_lapse_rate_k_km01'
MAX_TEMP_LAPSE_RATE_ARG_NAME = 'max_temp_lapse_rate_k_km01'
NUM_HUMIDITY_BINS_ARG_NAME = 'num_humidity_bins'
MIN_HUMIDITY_LAPSE_RATE_ARG_NAME = 'min_humidity_lapse_rate_km01'
MAX_HUMIDITY_LAPSE_RATE_ARG_NAME = 'max_humidity_lapse_rate_km01'
PLOT_FRACTIONAL_ERRORS_ARG_NAME = 'plot_fractional_errors'
PLOT_DETAILS_ARG_NAME = 'plot_details'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions and observations.  Will be read'
    ' by `prediction_io.read_file`.'
)
NUM_TEMPERATURE_BINS_HELP_STRING = 'Number of bins for surface temperature.'
MIN_TEMP_LAPSE_RATE_HELP_STRING = (
    'Minimum temperature lapse rate between the lowest two grid levels.  If '
    'you want to plot errors as a function of raw surface temperature, leave '
    'this alone.'
)
MAX_TEMP_LAPSE_RATE_HELP_STRING = (
    'Max temperature lapse rate between the lowest two grid levels.  If you '
    'want to plot errors as a function of raw surface temperature, leave this '
    'alone.'
)
NUM_HUMIDITY_BINS_HELP_STRING = 'Number of bins for surface humidity.'
MIN_HUMIDITY_LAPSE_RATE_HELP_STRING = (
    'Minimum specific-humidity lapse rate (kg kg^-1 km^-1) between the lowest '
    'two grid levels.  If you want to plot errors as a function of raw near-'
    'surface specific humidity, leave this alone.'
)
MAX_HUMIDITY_LAPSE_RATE_HELP_STRING = (
    'Max specific-humidity lapse rate (kg kg^-1 km^-1) between the lowest two '
    'grid levels.  If you want to plot errors as a function of raw near-'
    'surface specific humidity, leave this alone.'
)
PLOT_FRACTIONAL_ERRORS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot fractional (raw) errors for '
    'each metric -- "fractional" meaning as a fraction of the mean.'
)
PLOT_DETAILS_HELP_STRING = (
    'Boolean flag.  If 1, will plot all the details, including two metrics '
    '(MAE and bias) for every flux variable.  If 0, will plot only three flux-'
    'based metrics: all-flux MAE, net-flux MAE, and net-flux bias.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with example files.  Temperature and humidity values '
    'will be read from here.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_TEMPERATURE_BINS_ARG_NAME, type=int, required=False, default=14,
    help=NUM_TEMPERATURE_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_TEMP_LAPSE_RATE_ARG_NAME, type=float, required=False, default=1.,
    help=MIN_TEMP_LAPSE_RATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_TEMP_LAPSE_RATE_ARG_NAME, type=float, required=False,
    default=-1., help=MAX_TEMP_LAPSE_RATE_ARG_NAME
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_HUMIDITY_BINS_ARG_NAME, type=int, required=False, default=12,
    help=NUM_HUMIDITY_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_HUMIDITY_LAPSE_RATE_ARG_NAME, type=float, required=False,
    default=1., help=MIN_HUMIDITY_LAPSE_RATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_HUMIDITY_LAPSE_RATE_ARG_NAME, type=float, required=False,
    default=1., help=MAX_HUMIDITY_LAPSE_RATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_FRACTIONAL_ERRORS_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_FRACTIONAL_ERRORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_DETAILS_ARG_NAME, type=int, required=False, default=0,
    help=PLOT_DETAILS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_temperature_values(prediction_dict, example_dir_name, get_lapse_rates):
    """Returns surface temperature for each profile.

    E = number of examples

    :param prediction_dict: Dictionary returned by `prediction_io.read_file`.
    :param example_dir_name: See documentation at top of file.
    :param get_lapse_rates: Boolean flag.  If True (False), will return
        near-surface temperature lapse rates (actual surface temperatures).
    :return: output_values: If `get_lapse_rates == False`, this is a length-E
        numpy array of surface temperatures in Kelvins.  If
        `get_lapse_rates == True`, this is a length-E numpy array of
        near-surface lapse rates in Kelvins per km.
    """

    valid_times_unix_sec = example_utils.parse_example_ids(
        prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    )[example_utils.VALID_TIMES_KEY]

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=numpy.min(valid_times_unix_sec),
        last_time_unix_sec=numpy.max(valid_times_unix_sec),
        raise_error_if_any_missing=False
    )

    example_id_strings = []
    temperature_values = numpy.array([])

    for this_file_name in example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(
            netcdf_file_name=this_file_name, exclude_summit_greenland=False,
            max_shortwave_heating_k_day01=numpy.inf,
            min_longwave_heating_k_day01=-1 * numpy.inf,
            max_longwave_heating_k_day01=numpy.inf
        )

        example_id_strings += this_example_dict[example_utils.EXAMPLE_IDS_KEY]

        if not get_lapse_rates:
            these_values = example_utils.get_field_from_dict(
                example_dict=this_example_dict,
                field_name=example_utils.SURFACE_TEMPERATURE_NAME
            )
            temperature_values = numpy.concatenate(
                (temperature_values, these_values), axis=0
            )

            continue

        this_temp_matrix_kelvins = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.TEMPERATURE_NAME
        )[:, :2]

        these_temp_diffs_kelvins = (
            -1 * numpy.diff(this_temp_matrix_kelvins, axis=1)[:, 0]
        )

        if (
                example_utils.HEIGHT_NAME in
                this_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
        ):
            this_height_matrix_m_agl = example_utils.get_field_from_dict(
                example_dict=this_example_dict,
                field_name=example_utils.HEIGHT_NAME
            )[:, :2]

            these_height_diffs_metres = numpy.diff(
                this_height_matrix_m_agl, axis=1
            )[:, 0]
        else:
            these_height_diffs_metres = numpy.full(
                len(these_temp_diffs_kelvins),
                numpy.diff(this_example_dict[example_utils.HEIGHTS_KEY][:2])
            )

        these_values = (
            1000 * these_temp_diffs_kelvins / these_height_diffs_metres
        )
        temperature_values = numpy.concatenate(
            (temperature_values, these_values), axis=0
        )

    desired_indices = example_utils.find_examples(
        all_id_strings=example_id_strings,
        desired_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        allow_missing=False
    )

    return temperature_values[desired_indices]


def _get_humidity_values(prediction_dict, example_dir_name, get_lapse_rates):
    """Returns surface humidity for each profile.

    E = number of examples

    :param prediction_dict: Dictionary returned by `prediction_io.read_file`.
    :param example_dir_name: See documentation at top of file.
    :param get_lapse_rates: Boolean flag.  If True (False), will return lapse
        rates (raw values) of near-surface humidity.
    :return: output_values: If `get_lapse_rates == False`, this is a length-E
        numpy array of near-surface specific humidities in kg kg^-1.  If
        `get_lapse_rates == True`, this is a length-E numpy array of
        near-surface lapse rates in kg kg^-1 km^-1.
    """

    valid_times_unix_sec = example_utils.parse_example_ids(
        prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    )[example_utils.VALID_TIMES_KEY]

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=numpy.min(valid_times_unix_sec),
        last_time_unix_sec=numpy.max(valid_times_unix_sec),
        raise_error_if_any_missing=False
    )

    example_id_strings = []
    humidity_values = numpy.array([])

    for this_file_name in example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(
            netcdf_file_name=this_file_name, exclude_summit_greenland=False,
            max_shortwave_heating_k_day01=numpy.inf,
            min_longwave_heating_k_day01=-1 * numpy.inf,
            max_longwave_heating_k_day01=numpy.inf
        )

        example_id_strings += this_example_dict[example_utils.EXAMPLE_IDS_KEY]

        if not get_lapse_rates:
            these_values = example_utils.get_field_from_dict(
                example_dict=this_example_dict,
                field_name=example_utils.SPECIFIC_HUMIDITY_NAME
            )[:, 0]

            humidity_values = numpy.concatenate(
                (humidity_values, these_values), axis=0
            )

            continue

        this_humidity_matrix_kg_kg01 = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.SPECIFIC_HUMIDITY_NAME
        )[:, :2]

        these_humidity_diffs_kg_kg01 = (
            -1 * numpy.diff(this_humidity_matrix_kg_kg01, axis=1)[:, 0]
        )

        if (
                example_utils.HEIGHT_NAME in
                this_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
        ):
            this_height_matrix_m_agl = example_utils.get_field_from_dict(
                example_dict=this_example_dict,
                field_name=example_utils.HEIGHT_NAME
            )[:, :2]

            these_height_diffs_metres = numpy.diff(
                this_height_matrix_m_agl, axis=1
            )[:, 0]
        else:
            these_height_diffs_metres = numpy.full(
                len(these_humidity_diffs_kg_kg01),
                numpy.diff(this_example_dict[example_utils.HEIGHTS_KEY][:2])
            )

        these_values = (
            1000 * these_humidity_diffs_kg_kg01 / these_height_diffs_metres
        )
        humidity_values = numpy.concatenate(
            (humidity_values, these_values), axis=0
        )

    desired_indices = example_utils.find_examples(
        all_id_strings=example_id_strings,
        desired_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        allow_missing=False
    )

    return humidity_values[desired_indices]


def _plot_score_2d(
        score_matrix, colour_map_object, colour_norm_object, x_tick_labels,
        y_tick_labels):
    """Plots one score on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param colour_norm_object: Normalizer for colour scheme (instance of
        `matplotlib.pyplot.Normalize` or similar).
    :param x_tick_labels: length-N list of tick labels.
    :param y_tick_labels: length-M list of tick labels.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.imshow(
        score_matrix, cmap=colour_map_object, norm=colour_norm_object,
        origin='lower'
    )

    x_tick_values = numpy.linspace(
        0, score_matrix.shape[1] - 1, num=score_matrix.shape[1], dtype=float
    )
    y_tick_values = numpy.linspace(
        0, score_matrix.shape[0] - 1, num=score_matrix.shape[0], dtype=float
    )
    pyplot.xticks(x_tick_values, x_tick_labels, rotation=90)
    pyplot.yticks(y_tick_values, y_tick_labels)

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=score_matrix[numpy.invert(numpy.isnan(score_matrix))],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=True, extend_max=True,
        fraction_of_axis_length=0.8
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _run(prediction_file_name, num_temperature_bins,
         min_temp_lapse_rate_k_km01, max_temp_lapse_rate_k_km01,
         num_humidity_bins,
         min_humidity_lapse_rate_km01, max_humidity_lapse_rate_km01,
         plot_fractional_errors, plot_details,
         example_dir_name, output_dir_name):
    """Plots error metrics vs. surface temperature and surface humidity.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param num_temperature_bins: Same.
    :param min_temp_lapse_rate_k_km01: Same.
    :param max_temp_lapse_rate_k_km01: Same.
    :param num_humidity_bins: Same.
    :param min_humidity_lapse_rate_km01: Same.
    :param max_humidity_lapse_rate_km01: Same.
    :param plot_fractional_errors: Same.
    :param plot_details: Same.
    :param example_dir_name: Same.
    :param output_dir_name: Same.
    """

    if min_temp_lapse_rate_k_km01 > max_temp_lapse_rate_k_km01:
        plot_temp_lapse_rates = False
    else:
        plot_temp_lapse_rates = True

    if min_humidity_lapse_rate_km01 > max_humidity_lapse_rate_km01:
        plot_humidity_lapse_rates = False
    else:
        plot_humidity_lapse_rates = True

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)
    prediction_dict = prediction_io.get_ensemble_mean(prediction_dict)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    if plot_temp_lapse_rates:

        # Bin examples by near-surface temperature lapse rate.
        edge_temperature_values = numpy.linspace(
            min_temp_lapse_rate_k_km01, max_temp_lapse_rate_k_km01,
            num=num_temperature_bins + 1, dtype=float
        )
        edge_temperature_values[0] = -numpy.inf
        edge_temperature_values[-1] = numpy.inf

        temperature_tick_labels = [
            '[{0:.2g}, {1:.2g})'.format(a, b) for a, b in zip(
                edge_temperature_values[:-1],
                edge_temperature_values[1:]
            )
        ]
        temperature_tick_labels[-1] = '>= {0:.2g}'.format(
            edge_temperature_values[-2]
        )
        temperature_tick_labels[0] = '< {0:.2g}'.format(
            edge_temperature_values[1]
        )
    else:

        # Bin examples by surface temperature.
        edge_temperature_values = numpy.linspace(
            MIN_SURFACE_TEMP_KELVINS, MAX_SURFACE_TEMP_KELVINS,
            num=num_temperature_bins + 1, dtype=float
        )
        edge_temperature_values[0] = -numpy.inf
        edge_temperature_values[-1] = numpy.inf

        temperature_tick_labels = [
            '[{0:.1f}, {1:.1f})'.format(a, b) for a, b in zip(
                edge_temperature_values[:-1],
                edge_temperature_values[1:]
            )
        ]
        temperature_tick_labels[-1] = '>= {0:.1f}'.format(
            edge_temperature_values[-2]
        )
        temperature_tick_labels[0] = '< {0:.1f}'.format(
            edge_temperature_values[1]
        )

    actual_temperature_values = _get_temperature_values(
        prediction_dict=prediction_dict, example_dir_name=example_dir_name,
        get_lapse_rates=plot_temp_lapse_rates
    )
    temperature_bin_indices = numpy.digitize(
        x=actual_temperature_values, bins=edge_temperature_values,
        right=False
    ) - 1

    if plot_humidity_lapse_rates:

        # Bin examples by near-surface humidity lapse rate.
        edge_humidity_values = numpy.linspace(
            min_humidity_lapse_rate_km01, max_humidity_lapse_rate_km01,
            num=num_humidity_bins + 1, dtype=float
        )
        edge_humidity_values[0] = -numpy.inf
        edge_humidity_values[-1] = numpy.inf

        humidity_tick_labels = [
            '[{0:.2g}, {1:.2g})'.format(a, b) for a, b in zip(
                KG_TO_GRAMS * edge_humidity_values[:-1],
                KG_TO_GRAMS * edge_humidity_values[1:]
            )
        ]
        humidity_tick_labels[-1] = '>= {0:.2g}'.format(
            KG_TO_GRAMS * edge_humidity_values[-2]
        )
        humidity_tick_labels[0] = '< {0:.2g}'.format(
            KG_TO_GRAMS * edge_humidity_values[1]
        )
    else:

        # Bin examples by near-surface humidity.
        edge_humidity_values = numpy.linspace(
            0, MAX_SURFACE_SPEC_HUMIDITY_KG_KG01,
            num=num_humidity_bins + 1, dtype=float
        )
        edge_humidity_values[0] = -numpy.inf
        edge_humidity_values[-1] = numpy.inf

        humidity_tick_labels = [
            '[{0:.1f}, {1:.1f})'.format(a, b) for a, b in zip(
                KG_TO_GRAMS * edge_humidity_values[:-1],
                KG_TO_GRAMS * edge_humidity_values[1:]
            )
        ]
        humidity_tick_labels[-1] = '>= {0:.1f}'.format(
            KG_TO_GRAMS * edge_humidity_values[-2]
        )
        humidity_tick_labels[0] = '< {0:.1f}'.format(
            KG_TO_GRAMS * edge_humidity_values[1]
        )

    actual_humidity_values = _get_humidity_values(
        prediction_dict=prediction_dict, example_dir_name=example_dir_name,
        get_lapse_rates=plot_humidity_lapse_rates
    )
    humidity_bin_indices = numpy.digitize(
        x=actual_humidity_values, bins=edge_humidity_values,
        right=False
    ) - 1

    available_target_names = [
        training_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY] +
        training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
    ]

    while isinstance(available_target_names[0], list):
        available_target_names = available_target_names[0]

    if (
            example_utils.LONGWAVE_TOA_UP_FLUX_NAME in available_target_names
            and example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME in
            available_target_names
    ):
        available_target_names += [
            LONGWAVE_ALL_FLUX_NAME, LONGWAVE_NET_FLUX_NAME
        ]

    if plot_details:
        statistic_names = STATISTIC_NAMES_DETAILED
    else:
        statistic_names = STATISTIC_NAMES_BASIC

    target_name_by_statistic = [
        STATISTIC_NAME_TO_TARGET_NAME[s] for s in statistic_names
    ]
    target_height_index_by_statistic = numpy.array([
        STATISTIC_NAME_TO_TARGET_HEIGHT_INDEX[s] for s in statistic_names
    ], dtype=int)

    found_data_flags = numpy.array(
        [t in available_target_names for t in target_name_by_statistic],
        dtype=bool
    )
    found_data_flags = numpy.array([
        True if target_name_by_statistic[i] == '' else found_data_flags[i]
        for i in range(len(found_data_flags))
    ], dtype=bool)

    plot_statistic_indices = numpy.where(found_data_flags)[0]
    num_statistics = len(plot_statistic_indices)

    pd = prediction_dict
    letter_label = None
    panel_file_names = [''] * num_statistics

    for m in range(num_statistics):
        print(SEPARATOR_STRING)

        k = plot_statistic_indices[m]
        metric_matrix = numpy.full(
            (num_temperature_bins, num_humidity_bins), numpy.nan
        )

        if (
                target_name_by_statistic[k] ==
                example_utils.LONGWAVE_HEATING_RATE_NAME
        ):
            channel_index = training_option_dict[
                neural_net.VECTOR_TARGET_NAMES_KEY
            ].index(target_name_by_statistic[k])

            actual_values = (
                pd[prediction_io.VECTOR_TARGETS_KEY][..., channel_index]
            )
            predicted_values = (
                pd[prediction_io.VECTOR_PREDICTIONS_KEY][..., channel_index]
            )

            if target_height_index_by_statistic[k] > -1:
                actual_values = (
                    actual_values[:, target_height_index_by_statistic[k]]
                )
                predicted_values = (
                    predicted_values[:, target_height_index_by_statistic[k]]
                )
        elif target_name_by_statistic[k] in [
                LONGWAVE_NET_FLUX_NAME, LONGWAVE_ALL_FLUX_NAME
        ]:
            down_flux_index = training_option_dict[
                neural_net.SCALAR_TARGET_NAMES_KEY
            ].index(example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME)

            up_flux_index = training_option_dict[
                neural_net.SCALAR_TARGET_NAMES_KEY
            ].index(example_utils.LONGWAVE_TOA_UP_FLUX_NAME)

            actual_net_flux_values = (
                pd[prediction_io.SCALAR_TARGETS_KEY][:, down_flux_index] -
                pd[prediction_io.SCALAR_TARGETS_KEY][:, up_flux_index]
            )
            predicted_net_flux_values = (
                pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, down_flux_index] -
                pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, up_flux_index]
            )

            if target_name_by_statistic[k] == LONGWAVE_NET_FLUX_NAME:
                actual_values = actual_net_flux_values
                predicted_values = predicted_net_flux_values
            else:
                actual_values = numpy.concatenate((
                    pd[prediction_io.SCALAR_TARGETS_KEY][:, down_flux_index],
                    pd[prediction_io.SCALAR_TARGETS_KEY][:, up_flux_index],
                    actual_net_flux_values
                ))
                predicted_values = numpy.concatenate((
                    pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, down_flux_index],
                    pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, up_flux_index],
                    predicted_net_flux_values
                ))
        else:
            channel_index = training_option_dict[
                neural_net.SCALAR_TARGET_NAMES_KEY
            ].index(target_name_by_statistic[k])

            actual_values = (
                pd[prediction_io.SCALAR_TARGETS_KEY][..., channel_index]
            )
            predicted_values = (
                pd[prediction_io.SCALAR_PREDICTIONS_KEY][..., channel_index]
            )

        for i in range(num_temperature_bins):
            for j in range(num_humidity_bins):
                these_indices = numpy.where(numpy.logical_and(
                    temperature_bin_indices == i, humidity_bin_indices == j
                ))[0]

                if statistic_names[k] == 'num_examples':
                    metric_matrix[i, j] = len(these_indices)
                    continue

                if 'mae' in statistic_names[k]:
                    these_errors = numpy.absolute(
                        actual_values[these_indices] -
                        predicted_values[these_indices]
                    )
                else:
                    these_errors = (
                        predicted_values[these_indices] -
                        actual_values[these_indices]
                    )

                if plot_fractional_errors:
                    metric_matrix[i, j] = (
                        100 * numpy.mean(these_errors) /
                        numpy.mean(numpy.absolute(actual_values[these_indices]))
                    )
                else:
                    metric_matrix[i, j] = numpy.mean(these_errors)

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        if 'bias' in statistic_names[k]:
            max_colour_value = numpy.nanpercentile(
                numpy.absolute(metric_matrix), 99.5
            )
            min_colour_value = -1 * max_colour_value
            colour_map_object = BIAS_COLOUR_MAP_OBJECT
        elif statistic_names[k] == 'num_examples':
            metric_matrix = numpy.log10(metric_matrix)
            metric_matrix[numpy.isinf(metric_matrix)] = numpy.nan

            min_colour_value = numpy.nanpercentile(metric_matrix, 0.5)
            max_colour_value = numpy.nanpercentile(metric_matrix, 99.5)
            colour_map_object = NUM_EXAMPLES_COLOUR_MAP_OBJECT
        else:
            min_colour_value = numpy.nanpercentile(metric_matrix, 0.5)
            max_colour_value = numpy.nanpercentile(metric_matrix, 99.5)
            colour_map_object = MAIN_COLOUR_MAP_OBJECT

        colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )

        figure_object, axes_object = _plot_score_2d(
            score_matrix=metric_matrix, colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            x_tick_labels=humidity_tick_labels,
            y_tick_labels=temperature_tick_labels
        )

        if plot_temp_lapse_rates:
            axes_object.set_ylabel(
                r'Temperature lapse rate (K km$^{-1}$)'
            )
        else:
            axes_object.set_ylabel('Surface temperature (K)')

        if plot_humidity_lapse_rates:
            axes_object.set_xlabel(
                r'Humidity lapse rate (g kg$^{-1}$ km$^{-1}$)'
            )
        else:
            axes_object.set_xlabel(
                r'Near-surface specific humidity (g kg$^{-1}$)'
            )

        axes_object.set_title(
            STATISTIC_NAME_TO_FANCY_FRACTIONAL[statistic_names[k]]
            if plot_fractional_errors
            else STATISTIC_NAME_TO_FANCY[statistic_names[k]]
        )
        gg_plotting_utils.label_axes(
            axes_object=axes_object, label_string='({0:s})'.format(letter_label)
        )

        panel_file_names[m] = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, statistic_names[k]
        )

        print('Saving figure to: "{0:s}"...'.format(panel_file_names[m]))
        figure_object.savefig(
            panel_file_names[m], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    num_panel_rows = int(numpy.ceil(
        float(num_statistics) / NUM_PANEL_COLUMNS
    ))

    concat_file_name = '{0:s}/errors_by_sfc_temp_and_humidity.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=NUM_PANEL_COLUMNS
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_file_name,
        output_file_name=concat_file_name,
        output_size_pixels=int(1e7)
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_temperature_bins=getattr(
            INPUT_ARG_OBJECT, NUM_TEMPERATURE_BINS_ARG_NAME
        ),
        min_temp_lapse_rate_k_km01=getattr(
            INPUT_ARG_OBJECT, MIN_TEMP_LAPSE_RATE_ARG_NAME
        ),
        max_temp_lapse_rate_k_km01=getattr(
            INPUT_ARG_OBJECT, MAX_TEMP_LAPSE_RATE_ARG_NAME
        ),
        num_humidity_bins=getattr(
            INPUT_ARG_OBJECT, NUM_HUMIDITY_BINS_ARG_NAME
        ),
        min_humidity_lapse_rate_km01=getattr(
            INPUT_ARG_OBJECT, MIN_HUMIDITY_LAPSE_RATE_ARG_NAME
        ),
        max_humidity_lapse_rate_km01=getattr(
            INPUT_ARG_OBJECT, MAX_HUMIDITY_LAPSE_RATE_ARG_NAME
        ),
        plot_fractional_errors=bool(
            getattr(INPUT_ARG_OBJECT, PLOT_FRACTIONAL_ERRORS_ARG_NAME)
        ),
        plot_details=bool(getattr(INPUT_ARG_OBJECT, PLOT_DETAILS_ARG_NAME)),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
