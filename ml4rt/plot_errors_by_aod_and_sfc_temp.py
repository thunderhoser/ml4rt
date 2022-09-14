"""Plots error metrics vs. aerosol optical depth and surface temperature."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.integrate import simps

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import gg_plotting_utils
import imagemagick_utils
import example_io
import prediction_io
import evaluation
import example_utils
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MAX_AEROSOL_OPTICAL_DEPTH = 1.8
MIN_SURFACE_TEMP_KELVINS = 190.
MAX_SURFACE_TEMP_KELVINS = 330.

LONGWAVE_ALL_FLUX_NAME = 'all_longwave_flux_w_m02'
LONGWAVE_NET_FLUX_NAME = 'net_longwave_flux_w_m02'

STATISTIC_NAMES = [
    'longwave_rmse', 'longwave_near_sfc_rmse',
    'longwave_bias', 'longwave_near_sfc_bias',
    'longwave_all_flux_rmse', 'longwave_net_flux_rmse',
    'longwave_net_flux_bias'
]
STATISTIC_NAMES_FANCY = [
    r'RMSE$_{hr}$ (K day$^{-1}$)',
    r'Near-surface RMSE$_{hr}$ (K day$^{-1}$)',
    r'Bias$_{hr}$ (K day$^{-1}$)',
    r'Near-surface bias$_{hr}$ (K day$^{-1}$)',
    r'RMSE$_{flux}$ (W m$^{-2}$)',
    r'RMSE for net flux only (W m$^{-2}$)',
    r'Bias for net flux only (W m$^{-2}$)'
]
TARGET_NAME_BY_STATISTIC = [
    example_utils.LONGWAVE_HEATING_RATE_NAME,
    example_utils.LONGWAVE_HEATING_RATE_NAME,
    example_utils.LONGWAVE_HEATING_RATE_NAME,
    example_utils.LONGWAVE_HEATING_RATE_NAME,
    LONGWAVE_ALL_FLUX_NAME,
    LONGWAVE_NET_FLUX_NAME,
    LONGWAVE_NET_FLUX_NAME
]
TARGET_HEIGHT_INDEX_BY_STATISTIC = numpy.array(
    [-1, 0, -1, 0, -1, -1, -1], dtype=int
)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis')
COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
NUM_SURFACE_TEMP_BINS_ARG_NAME = 'num_surface_temp_bins'
MIN_TEMP_GRADIENT_ARG_NAME = 'min_temp_gradient_k_km01'
MAX_TEMP_GRADIENT_ARG_NAME = 'max_temp_gradient_k_km01'
NUM_AOD_BINS_ARG_NAME = 'num_aod_bins'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions and observations.  Will be read'
    ' by `prediction_io.read_file`.'
)
NUM_SURFACE_TEMP_BINS_HELP_STRING = 'Number of bins for surface temperature.'
MIN_TEMP_GRADIENT_HELP_STRING = (
    'Minimum temperature gradient between surface and lowest grid level.  If '
    'you want to plot errors as a function of surface temperature itself and '
    'not gradient, leave this alone.'
)
MAX_TEMP_GRADIENT_HELP_STRING = (
    'Max temperature gradient between surface and lowest grid level.  If '
    'you want to plot errors as a function of surface temperature itself and '
    'not gradient, leave this alone.'
)
NUM_AOD_BINS_HELP_STRING = 'Number of bins for aerosol optical depth (AOD).'
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with example files.  AOD values will be read from here.'
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
    '--' + NUM_SURFACE_TEMP_BINS_ARG_NAME, type=int, required=False, default=14,
    help=NUM_SURFACE_TEMP_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_TEMP_GRADIENT_ARG_NAME, type=float, required=False, default=1.,
    help=MIN_TEMP_GRADIENT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_TEMP_GRADIENT_ARG_NAME, type=float, required=False, default=-1.,
    help=MAX_TEMP_GRADIENT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_AOD_BINS_ARG_NAME, type=int, required=False, default=9,
    help=NUM_AOD_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _get_aerosol_optical_depths(prediction_dict, example_dir_name):
    """Computes aerosol optical depth (AOD) for each profile.

    :param prediction_dict: Dictionary returned by `prediction_io.read_file`.
    :param example_dir_name: See documentation at top of file.
    :return: aerosol_optical_depths: 1-D numpy array of AOD values, one per
        example.
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
    aerosol_extinction_matrix_metres01 = numpy.array([])
    height_matrix_m_agl = numpy.array([])

    for this_file_name in example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(
            netcdf_file_name=this_file_name, exclude_summit_greenland=False,
            max_shortwave_heating_k_day01=numpy.inf,
            min_longwave_heating_k_day01=-1 * numpy.inf,
            max_longwave_heating_k_day01=numpy.inf
        )

        example_id_strings += this_example_dict[example_utils.EXAMPLE_IDS_KEY]
        this_extinction_matrix_metres01 = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.AEROSOL_EXTINCTION_NAME
        )

        if aerosol_extinction_matrix_metres01.size == 0:
            aerosol_extinction_matrix_metres01 = (
                this_extinction_matrix_metres01 + 0.
            )
        else:
            aerosol_extinction_matrix_metres01 = numpy.concatenate((
                aerosol_extinction_matrix_metres01,
                this_extinction_matrix_metres01
            ), axis=0)

        if (
                example_utils.HEIGHT_NAME in
                this_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
        ):
            this_height_matrix_m_agl = example_utils.get_field_from_dict(
                example_dict=this_example_dict,
                field_name=example_utils.HEIGHT_NAME
            )

            if height_matrix_m_agl.size == 0:
                height_matrix_m_agl = this_height_matrix_m_agl + 0.
            else:
                height_matrix_m_agl = numpy.concatenate(
                    (height_matrix_m_agl, this_height_matrix_m_agl), axis=0
                )
        else:
            if height_matrix_m_agl.size == 0:
                height_matrix_m_agl = (
                    this_example_dict[example_utils.HEIGHTS_KEY] + 0.
                )

    desired_indices = example_utils.find_examples(
        all_id_strings=example_id_strings,
        desired_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        allow_missing=False
    )
    del example_id_strings

    aerosol_extinction_matrix_metres01 = (
        aerosol_extinction_matrix_metres01[desired_indices, :]
    )

    if len(height_matrix_m_agl.shape) == 2:
        height_matrix_m_agl = height_matrix_m_agl[desired_indices, :]
        num_examples = aerosol_extinction_matrix_metres01.shape[0]
        aerosol_optical_depths = numpy.full(num_examples, numpy.nan)
        print('\n')

        for i in range(num_examples):
            if numpy.mod(i, 1000) == 0:
                print((
                    'Have computed aerosol optical depth for {0:d} of {1:d} '
                    'profiles...'
                ).format(
                    i, num_examples
                ))

            aerosol_optical_depths[i] = simps(
                y=aerosol_extinction_matrix_metres01[i, :],
                x=height_matrix_m_agl[i, :],
                even='avg'
            )

        print((
            'Have computed aerosol optical depth for all {0:d} profiles!\n'
        ).format(
            num_examples
        ))
    else:
        aerosol_optical_depths = simps(
            y=aerosol_extinction_matrix_metres01, x=height_matrix_m_agl,
            axis=-1, even='avg'
        )

    return aerosol_optical_depths


def _get_surface_temperatures(prediction_dict, example_dir_name, get_gradients):
    """Returns surface temperature for each profile.

    E = number of examples

    :param prediction_dict: Dictionary returned by `prediction_io.read_file`.
    :param example_dir_name: See documentation at top of file.
    :param get_gradients: Boolean flag.  If True (False), will return
        near-surface temperature gradients (actual surface temperatures).
    :return: output_values: If `get_gradients == False`, this is a length-E
        numpy array of surface temperatures in Kelvins.  If
        `get_gradients == True`, this is a length-E numpy array of
        surface-to-lowest-grid-level lapse rates in Kelvins per km.
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
    surface_temps_kelvins = numpy.array([])
    lowest_level_temps_kelvins = numpy.array([])
    lowest_level_heights_m_agl = numpy.array([])

    for this_file_name in example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(
            netcdf_file_name=this_file_name, exclude_summit_greenland=False,
            max_shortwave_heating_k_day01=numpy.inf,
            min_longwave_heating_k_day01=-1 * numpy.inf,
            max_longwave_heating_k_day01=numpy.inf
        )

        example_id_strings += this_example_dict[example_utils.EXAMPLE_IDS_KEY]
        these_surface_temps_kelvins = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.SURFACE_TEMPERATURE_NAME
        )
        surface_temps_kelvins = numpy.concatenate(
            (surface_temps_kelvins, these_surface_temps_kelvins), axis=0
        )

        if get_gradients:
            these_ll_temps_kelvins = example_utils.get_field_from_dict(
                example_dict=this_example_dict,
                field_name=example_utils.TEMPERATURE_NAME
            )[:, 0]

            lowest_level_temps_kelvins = numpy.concatenate(
                (lowest_level_temps_kelvins, these_ll_temps_kelvins), axis=0
            )

            if (
                    example_utils.HEIGHT_NAME in
                    this_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
            ):
                these_ll_heights_m_agl = example_utils.get_field_from_dict(
                    example_dict=this_example_dict,
                    field_name=example_utils.HEIGHT_NAME
                )[:, 0]
            else:
                these_ll_heights_m_agl = numpy.full(
                    len(these_ll_temps_kelvins),
                    this_example_dict[example_utils.HEIGHTS_KEY][0]
                )

            lowest_level_heights_m_agl = numpy.concatenate(
                (lowest_level_heights_m_agl, these_ll_heights_m_agl), axis=0
            )

    desired_indices = example_utils.find_examples(
        all_id_strings=example_id_strings,
        desired_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        allow_missing=False
    )

    if not get_gradients:
        return surface_temps_kelvins[desired_indices]

    surface_temps_kelvins = surface_temps_kelvins[desired_indices]
    lowest_level_temps_kelvins = lowest_level_temps_kelvins[desired_indices]
    lowest_level_heights_m_agl = lowest_level_heights_m_agl[desired_indices]

    return (
        1000 * (lowest_level_temps_kelvins - surface_temps_kelvins) /
        lowest_level_heights_m_agl
    )


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


def _run(prediction_file_name, num_surface_temp_bins, min_temp_gradient_k_km01,
         max_temp_gradient_k_km01, num_aod_bins, example_dir_name,
         output_dir_name):
    """Plots error metrics vs. aerosol optical depth and surface temperature.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param num_surface_temp_bins: Same.
    :param min_temp_gradient_k_km01: Same.
    :param max_temp_gradient_k_km01: Same.
    :param num_aod_bins: Same.
    :param example_dir_name: Same.
    :param output_dir_name: Same.
    """

    if min_temp_gradient_k_km01 > max_temp_gradient_k_km01:
        plot_temp_gradients = False
    else:
        plot_temp_gradients = True

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    if plot_temp_gradients:

        # Bin examples by near-surface temperature gradient.
        edge_surface_temps_kelvins = numpy.linspace(
            min_temp_gradient_k_km01, max_temp_gradient_k_km01,
            num=num_surface_temp_bins + 1, dtype=float
        )
        edge_surface_temps_kelvins[0] = -numpy.inf
        edge_surface_temps_kelvins[-1] = numpy.inf

        actual_surface_temps_kelvins = _get_surface_temperatures(
            prediction_dict=prediction_dict, example_dir_name=example_dir_name,
            get_gradients=True
        )
        surface_temp_bin_indices = numpy.digitize(
            x=actual_surface_temps_kelvins, bins=edge_surface_temps_kelvins,
            right=False
        ) - 1

        surface_temp_tick_labels = [
            '[{0:.2g}, {1:.2g})'.format(a, b) for a, b in zip(
                edge_surface_temps_kelvins[:-1],
                edge_surface_temps_kelvins[1:]
            )
        ]
        surface_temp_tick_labels[-1] = '>= {0:.2g}'.format(
            edge_surface_temps_kelvins[-2]
        )
        surface_temp_tick_labels[0] = '< {0:.2g}'.format(
            edge_surface_temps_kelvins[1]
        )
    else:

        # Bin examples by surface temperature.
        edge_surface_temps_kelvins = numpy.linspace(
            MIN_SURFACE_TEMP_KELVINS, MAX_SURFACE_TEMP_KELVINS,
            num=num_surface_temp_bins + 1, dtype=float
        )
        edge_surface_temps_kelvins[0] = -numpy.inf
        edge_surface_temps_kelvins[-1] = numpy.inf

        actual_surface_temps_kelvins = _get_surface_temperatures(
            prediction_dict=prediction_dict, example_dir_name=example_dir_name,
            get_gradients=False
        )
        surface_temp_bin_indices = numpy.digitize(
            x=actual_surface_temps_kelvins, bins=edge_surface_temps_kelvins,
            right=False
        ) - 1

        surface_temp_tick_labels = [
            '[{0:.1f}, {1:.1f})'.format(a, b) for a, b in zip(
                edge_surface_temps_kelvins[:-1],
                edge_surface_temps_kelvins[1:]
            )
        ]
        surface_temp_tick_labels[-1] = '>= {0:.1f}'.format(
            edge_surface_temps_kelvins[-2]
        )
        surface_temp_tick_labels[0] = '< {0:.1f}'.format(
            edge_surface_temps_kelvins[1]
        )

    # Bin examples by AOD.
    edge_aerosol_optical_depths = numpy.linspace(
        0, MAX_AEROSOL_OPTICAL_DEPTH, num=num_aod_bins + 1, dtype=float
    )
    edge_aerosol_optical_depths[-1] = numpy.inf

    actual_aerosol_optical_depths = _get_aerosol_optical_depths(
        prediction_dict=prediction_dict, example_dir_name=example_dir_name
    )
    aod_bin_indices = numpy.digitize(
        x=actual_aerosol_optical_depths, bins=edge_aerosol_optical_depths,
        right=False
    ) - 1

    aod_tick_labels = [
        '[{0:.1f}, {1:.1f})'.format(a, b) for a, b in zip(
            edge_aerosol_optical_depths[:-1],
            edge_aerosol_optical_depths[1:]
        )
    ]
    aod_tick_labels[-1] = '>= {0:.1f}'.format(edge_aerosol_optical_depths[-2])

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

    found_data_flags = numpy.array(
        [t in available_target_names for t in TARGET_NAME_BY_STATISTIC],
        dtype=bool
    )
    plot_statistic_indices = numpy.where(found_data_flags)[0]
    num_statistics = len(plot_statistic_indices)

    pd = prediction_dict
    letter_label = None
    panel_file_names = [''] * num_statistics

    for m in range(num_statistics):
        print(SEPARATOR_STRING)

        k = plot_statistic_indices[m]
        metric_matrix = numpy.full(
            (num_surface_temp_bins, num_aod_bins), numpy.nan
        )

        if (
                TARGET_NAME_BY_STATISTIC[k] ==
                example_utils.LONGWAVE_HEATING_RATE_NAME
        ):
            channel_index = training_option_dict[
                neural_net.VECTOR_TARGET_NAMES_KEY
            ].index(TARGET_NAME_BY_STATISTIC[k])

            actual_values = (
                pd[prediction_io.VECTOR_TARGETS_KEY][..., channel_index]
            )
            predicted_values = (
                pd[prediction_io.VECTOR_PREDICTIONS_KEY][..., channel_index]
            )

            if TARGET_HEIGHT_INDEX_BY_STATISTIC[k] > -1:
                actual_values = (
                    actual_values[:, TARGET_HEIGHT_INDEX_BY_STATISTIC[k]]
                )
                predicted_values = (
                    predicted_values[:, TARGET_HEIGHT_INDEX_BY_STATISTIC[k]]
                )
        else:
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

            if TARGET_NAME_BY_STATISTIC[k] == LONGWAVE_NET_FLUX_NAME:
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

        for i in range(num_surface_temp_bins):
            for j in range(num_aod_bins):
                these_indices = numpy.where(numpy.logical_and(
                    surface_temp_bin_indices == i, aod_bin_indices == j
                ))[0]

                if 'rmse' in STATISTIC_NAMES[k]:
                    metric_matrix[i, j] = evaluation._get_mse_one_scalar(
                        target_values=actual_values[these_indices],
                        predicted_values=predicted_values[these_indices]
                    )[0]

                    metric_matrix[i, j] = numpy.sqrt(metric_matrix[i, j])
                elif 'dwmse' in STATISTIC_NAMES[k]:
                    these_weights = numpy.maximum(
                        numpy.absolute(actual_values[these_indices]),
                        numpy.absolute(predicted_values[these_indices])
                    )
                    these_squared_errors = (
                        actual_values[these_indices] -
                        predicted_values[these_indices]
                    ) ** 2

                    metric_matrix[i, j] = numpy.mean(
                        these_weights * these_squared_errors
                    )
                else:
                    metric_matrix[i, j] = evaluation._get_bias_one_scalar(
                        target_values=actual_values[these_indices],
                        predicted_values=predicted_values[these_indices]
                    )

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        min_colour_value = numpy.nanpercentile(metric_matrix, 0.)
        max_colour_value = numpy.nanpercentile(metric_matrix, 99.)
        colour_norm_object = pyplot.Normalize(
            vmin=min_colour_value, vmax=max_colour_value
        )

        figure_object, axes_object = _plot_score_2d(
            score_matrix=metric_matrix, colour_map_object=COLOUR_MAP_OBJECT,
            colour_norm_object=colour_norm_object,
            x_tick_labels=aod_tick_labels,
            y_tick_labels=surface_temp_tick_labels
        )

        axes_object.set_xlabel('Aerosol optical depth (unitless)')
        if plot_temp_gradients:
            axes_object.set_ylabel(
                r'Near-surface temperature gradient (K km$^{-1}$)'
            )
        else:
            axes_object.set_ylabel('Surface temperature (K)')

        axes_object.set_title(STATISTIC_NAMES_FANCY[k])
        gg_plotting_utils.label_axes(
            axes_object=axes_object, label_string='({0:s})'.format(letter_label)
        )

        panel_file_names[m] = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, STATISTIC_NAMES[k]
        )

        print('Saving figure to: "{0:s}"...'.format(panel_file_names[m]))
        figure_object.savefig(
            panel_file_names[m], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_statistics)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_statistics) / num_panel_rows
    ))

    concat_file_name = '{0:s}/errors_by_aod_and_sfc_temp.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
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
        num_surface_temp_bins=getattr(
            INPUT_ARG_OBJECT, NUM_SURFACE_TEMP_BINS_ARG_NAME
        ),
        min_temp_gradient_k_km01=getattr(
            INPUT_ARG_OBJECT, MIN_TEMP_GRADIENT_ARG_NAME
        ),
        max_temp_gradient_k_km01=getattr(
            INPUT_ARG_OBJECT, MAX_TEMP_GRADIENT_ARG_NAME
        ),
        num_aod_bins=getattr(INPUT_ARG_OBJECT, NUM_AOD_BINS_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
