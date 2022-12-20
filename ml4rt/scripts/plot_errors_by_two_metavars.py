"""Plots errors as a function of two metadata variables."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.integrate import simps
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4rt.io import example_io
from ml4rt.io import prediction_io
from ml4rt.utils import evaluation
from ml4rt.utils import example_utils
from ml4rt.machine_learning import neural_net
from ml4rt.scripts import split_predictions_by_time as split_predictions

TOLERANCE = 1e-6
RADIANS_TO_DEGREES = 180. / numpy.pi

MAX_ZENITH_ANGLE_RAD = split_predictions.MAX_ZENITH_ANGLE_RAD
MAX_SHORTWAVE_SFC_DOWN_FLUX_W_M02 = (
    split_predictions.MAX_SHORTWAVE_SFC_DOWN_FLUX_W_M02
)
MAX_AEROSOL_OPTICAL_DEPTH = split_predictions.MAX_AEROSOL_OPTICAL_DEPTH
MIN_SURFACE_TEMP_KELVINS = split_predictions.MIN_SURFACE_TEMP_KELVINS
MAX_SURFACE_TEMP_KELVINS = split_predictions.MAX_SURFACE_TEMP_KELVINS
MAX_LONGWAVE_SFC_DOWN_FLUX_W_M02 = (
    split_predictions.MAX_LONGWAVE_SFC_DOWN_FLUX_W_M02
)
MAX_LONGWAVE_TOA_UP_FLUX_W_M02 = (
    split_predictions.MAX_LONGWAVE_TOA_UP_FLUX_W_M02
)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

MAIN_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis')
BIAS_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='seismic')
NUM_EXAMPLES_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='plasma')

MAIN_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))
BIAS_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
PLOT_SHORTWAVE_ERRORS_ARG_NAME = 'plot_shortwave_errors'
HEATING_RATE_HEIGHT_ARG_NAME = 'heating_rate_height_m_agl'
NUM_ANGLE_BINS_ARG_NAME = 'num_zenith_angle_bins'
NUM_FLUX_BINS_ARG_NAME = 'num_shortwave_sfc_down_flux_bins'
NUM_AOD_BINS_ARG_NAME = 'num_aod_bins'
NUM_SURFACE_TEMP_BINS_ARG_NAME = 'num_surface_temp_bins'
NUM_LW_DOWN_FLUX_BINS_ARG_NAME = 'num_longwave_sfc_down_flux_bins'
NUM_LW_UP_FLUX_BINS_ARG_NAME = 'num_longwave_toa_up_flux_bins'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions and observations.  Will be read'
    ' by `prediction_io.read_file`.'
)
PLOT_SHORTWAVE_ERRORS_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot errors for shortwave (longwave) '
    'predictions.'
)
HEATING_RATE_HEIGHT_HELP_STRING = (
    'Will plot error metrics for heating rate at this height (metres above '
    'ground level).  If you want to plot error metrics for net flux instead, '
    'leave this argument alone.'
)
NUM_ANGLE_BINS_HELP_STRING = (
    'Number of bins for zenith angle.  If you do not want plots as a function '
    'of zenith angle, leave this argument alone.'
)
NUM_FLUX_BINS_HELP_STRING = (
    'Number of bins for shortwave surface downwelling flux.  If you do not '
    'want plots as a function of SW sfc down flux, leave this argument alone.'
)
NUM_AOD_BINS_HELP_STRING = (
    'Number of bins for aerosol optical depth (AOD).  If you do not want plots '
    'as a function of AOD, leave this argument alone.'
)
NUM_SURFACE_TEMP_BINS_HELP_STRING = (
    'Number of bins for surface temperature.  If you do not want plots as a '
    'function of surface temperature, make this <= 0.'
)
NUM_LW_DOWN_FLUX_BINS_HELP_STRING = (
    'Number of bins for longwave surface downwelling flux.  If you do not want '
    'plots as a function of LW sfc down flux, make this <= 0.'
)
NUM_LW_UP_FLUX_BINS_HELP_STRING = (
    'Number of bins for longwave TOA upwelling flux.  If you do not want plots '
    'as a function of LW TOA up flux, make this <= 0.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with example files.  Needed if you want plots as a '
    'function of aerosol optical depth or surface temperature.'
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
    '--' + PLOT_SHORTWAVE_ERRORS_ARG_NAME, type=int, required=True,
    help=PLOT_SHORTWAVE_ERRORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HEATING_RATE_HEIGHT_ARG_NAME, type=int, required=False, default=-1,
    help=HEATING_RATE_HEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ANGLE_BINS_ARG_NAME, type=int, required=False, default=9,
    help=NUM_ANGLE_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_FLUX_BINS_ARG_NAME, type=int, required=False, default=12,
    help=NUM_FLUX_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_AOD_BINS_ARG_NAME, type=int, required=False, default=9,
    help=NUM_AOD_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_SURFACE_TEMP_BINS_ARG_NAME, type=int, required=False, default=14,
    help=NUM_SURFACE_TEMP_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_LW_DOWN_FLUX_BINS_ARG_NAME, type=int, required=False, default=10,
    help=NUM_LW_DOWN_FLUX_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_LW_UP_FLUX_BINS_ARG_NAME, type=int, required=False, default=10,
    help=NUM_LW_UP_FLUX_BINS_HELP_STRING
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


def _get_surface_temperatures(prediction_dict, example_dir_name):
    """Returns surface temperature for each profile.

    :param prediction_dict: Dictionary returned by `prediction_io.read_file`.
    :param example_dir_name: See documentation at top of file.
    :return: surface_temps_kelvins: 1-D numpy array of surface temperatures, one
        per example.
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

    desired_indices = example_utils.find_examples(
        all_id_strings=example_id_strings,
        desired_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        allow_missing=False
    )

    return surface_temps_kelvins[desired_indices]


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


def _plot_all_scores_one_var(
        num_first_bins, num_second_bins,
        example_to_first_bin_indices, example_to_second_bin_indices,
        target_values, predicted_values, climo_value,
        x_axis_label, y_axis_label, x_tick_labels, y_tick_labels,
        heating_rate_height_m_agl, plot_shortwave_errors, output_dir_name):
    """Plots all scores for one variable.

    E = number of examples
    F = number of bins for first metadata variable
    S = number of bins for second metadata variable

    :param num_first_bins: F in the above definitions.
    :param num_second_bins: S in the above definitions.
    :param example_to_first_bin_indices: length-E numpy array of integers from
        0...(F - 1), indicating bin membership for first metadata variable.
    :param example_to_second_bin_indices: length-E numpy array of integers from
        0...(S - 1), indicating bin membership for second metadata variable.
    :param target_values: length-E numpy array of target values.
    :param predicted_values: length-E numpy array of predicted value.
    :param climo_value: Climatological value (average in training data).
    :param x_axis_label: String used to label entire x-axis.
    :param y_axis_label: String used to label entire y-axis.
    :param x_tick_labels: length-S list of strings.
    :param y_tick_labels: length-F list of strings.
    :param heating_rate_height_m_agl: Heating-rate height (metres above ground
        level).  If the target variable here is not heating rate, make this
        None.
    :param plot_shortwave_errors: Boolean flag.  If True (False), plotting
        errors for shortwave (longwave) predictions.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if plot_shortwave_errors:
        band_string = 'SW'
    else:
        band_string = 'LW'

    if heating_rate_height_m_agl is None:
        target_var_name = band_string + ' net flux'
        target_var_unit_string = r'W m$^{-2}$'
        target_var_squared_unit_string = r'W$^{2}$ m$^{-4}$'
    else:
        target_var_name = '{0:d}-metre {1:s} heating rate'.format(
            heating_rate_height_m_agl, band_string
        )
        target_var_unit_string = r'K day$^{-1}$'
        target_var_squared_unit_string = r'K$^{2}$ day$^{-2}$'

    dimensions = (num_first_bins, num_second_bins)
    bias_matrix = numpy.full(dimensions, numpy.nan)
    correlation_matrix = numpy.full(dimensions, numpy.nan)
    mae_matrix = numpy.full(dimensions, numpy.nan)
    mae_skill_score_matrix = numpy.full(dimensions, numpy.nan)
    mse_matrix = numpy.full(dimensions, numpy.nan)
    mse_skill_score_matrix = numpy.full(dimensions, numpy.nan)
    kge_matrix = numpy.full(dimensions, numpy.nan)
    num_examples_matrix = numpy.full(dimensions, 0, dtype=int)

    for i in range(num_first_bins):
        for j in range(num_second_bins):
            these_indices = numpy.where(numpy.logical_and(
                example_to_first_bin_indices == i,
                example_to_second_bin_indices == j
            ))[0]

            if len(these_indices) == 0:
                continue

            bias_matrix[i, j] = evaluation._get_bias_one_scalar(
                target_values=target_values[these_indices],
                predicted_values=predicted_values[these_indices]
            )
            correlation_matrix[i, j] = evaluation._get_correlation_one_scalar(
                target_values=target_values[these_indices],
                predicted_values=predicted_values[these_indices]
            )
            mae_matrix[i, j] = evaluation._get_mae_one_scalar(
                target_values=target_values[these_indices],
                predicted_values=predicted_values[these_indices]
            )
            mse_matrix[i, j] = evaluation._get_mse_one_scalar(
                target_values=target_values[these_indices],
                predicted_values=predicted_values[these_indices]
            )[0]
            kge_matrix[i, j] = evaluation._get_kge_one_scalar(
                target_values=target_values[these_indices],
                predicted_values=predicted_values[these_indices]
            )
            mae_skill_score_matrix[i, j] = evaluation._get_mae_ss_one_scalar(
                target_values=target_values[these_indices],
                predicted_values=predicted_values[these_indices],
                mean_training_target_value=climo_value
            )
            mse_skill_score_matrix[i, j] = evaluation._get_mse_ss_one_scalar(
                target_values=target_values[these_indices],
                predicted_values=predicted_values[these_indices],
                mean_training_target_value=climo_value
            )
            num_examples_matrix[i, j] = len(these_indices)

    max_colour_value = numpy.nanpercentile(numpy.absolute(bias_matrix), 99)
    colour_norm_object = pyplot.Normalize(
        vmin=-1 * max_colour_value, vmax=max_colour_value
    )
    figure_object, axes_object = _plot_score_2d(
        score_matrix=bias_matrix, colour_map_object=BIAS_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Bias for {0:s} ({1:s})'.format(
        target_var_name, target_var_unit_string
    ))

    output_file_name = '{0:s}/bias.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    max_colour_value = numpy.nanpercentile(correlation_matrix, 99)
    min_colour_value = max([
        numpy.nanpercentile(correlation_matrix, 1), 0
    ])
    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )
    figure_object, axes_object = _plot_score_2d(
        score_matrix=correlation_matrix,
        colour_map_object=MAIN_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Correlation for {0:s}'.format(
        target_var_name
    ))

    output_file_name = '{0:s}/correlation.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    colour_norm_object = pyplot.Normalize(
        vmin=numpy.nanpercentile(mae_matrix, 1),
        vmax=numpy.nanpercentile(mae_matrix, 99)
    )
    figure_object, axes_object = _plot_score_2d(
        score_matrix=mae_matrix, colour_map_object=MAIN_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Mean absolute error for {0:s} ({1:s})'.format(
        target_var_name, target_var_unit_string
    ))

    output_file_name = '{0:s}/mae.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    colour_norm_object = pyplot.Normalize(
        vmin=numpy.nanpercentile(mse_matrix, 1),
        vmax=numpy.nanpercentile(mse_matrix, 99)
    )
    figure_object, axes_object = _plot_score_2d(
        score_matrix=mse_matrix, colour_map_object=MAIN_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Mean squared error for {0:s} ({1:s})'.format(
        target_var_name, target_var_squared_unit_string
    ))

    output_file_name = '{0:s}/mse.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    max_colour_value = numpy.nanpercentile(kge_matrix, 99)
    min_colour_value = max([
        numpy.nanpercentile(kge_matrix, 1), -1
    ])
    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )
    figure_object, axes_object = _plot_score_2d(
        score_matrix=kge_matrix, colour_map_object=MAIN_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Kling-Gupta efficiency for {0:s}'.format(
        target_var_name
    ))

    output_file_name = '{0:s}/kge.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    max_colour_value = numpy.nanpercentile(mae_skill_score_matrix, 99)
    min_colour_value = max([
        numpy.nanpercentile(mae_skill_score_matrix, 1), -1
    ])
    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )
    figure_object, axes_object = _plot_score_2d(
        score_matrix=mae_skill_score_matrix,
        colour_map_object=MAIN_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('MAE skill score for {0:s}'.format(
        target_var_name
    ))

    output_file_name = '{0:s}/maess.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    max_colour_value = numpy.nanpercentile(mse_skill_score_matrix, 99)
    min_colour_value = max([
        numpy.nanpercentile(mse_skill_score_matrix, 1), -1
    ])
    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )
    figure_object, axes_object = _plot_score_2d(
        score_matrix=mse_skill_score_matrix,
        colour_map_object=MAIN_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('MSE skill score for {0:s}'.format(
        target_var_name
    ))

    output_file_name = '{0:s}/msess.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    colour_norm_object = pyplot.Normalize(
        vmin=numpy.nanpercentile(num_examples_matrix, 1),
        vmax=numpy.nanpercentile(num_examples_matrix, 99)
    )
    figure_object, axes_object = _plot_score_2d(
        score_matrix=num_examples_matrix,
        colour_map_object=NUM_EXAMPLES_COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )
    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title('Number of examples')

    output_file_name = '{0:s}/num_examples.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(prediction_file_name, plot_shortwave_errors, heating_rate_height_m_agl,
         num_zenith_angle_bins, num_shortwave_sfc_down_flux_bins, num_aod_bins,
         num_surface_temp_bins, num_longwave_sfc_down_flux_bins,
         num_longwave_toa_up_flux_bins, example_dir_name, top_output_dir_name):
    """Plots errors as a function of two metadata variables.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param plot_shortwave_errors: Same.
    :param heating_rate_height_m_agl: Same.
    :param num_zenith_angle_bins: Same.
    :param num_shortwave_sfc_down_flux_bins: Same.
    :param num_aod_bins: Same.
    :param num_surface_temp_bins: Same.
    :param num_longwave_sfc_down_flux_bins: Same.
    :param num_longwave_toa_up_flux_bins: Same.
    :param example_dir_name: Same.
    :param top_output_dir_name: Same.
    """

    if heating_rate_height_m_agl < 0:
        heating_rate_height_m_agl = None

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
    normalization_file_name = (
        training_option_dict[neural_net.NORMALIZATION_FILE_KEY]
    )

    print((
        'Reading training examples (for climatology) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)

    # Extract predicted and observed values of target variable.
    if heating_rate_height_m_agl is None:
        if plot_shortwave_errors:
            j = training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY].index(
                example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
            )
            k = training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY].index(
                example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
            )

            training_down_fluxes_w_m02 = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
            )
            training_up_fluxes_w_m02 = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
            )
        else:
            j = training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY].index(
                example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
            )
            k = training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY].index(
                example_utils.LONGWAVE_TOA_UP_FLUX_NAME
            )

            training_down_fluxes_w_m02 = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
            )
            training_up_fluxes_w_m02 = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=example_utils.LONGWAVE_TOA_UP_FLUX_NAME
            )

        target_values = (
            prediction_dict[prediction_io.SCALAR_TARGETS_KEY][:, j] -
            prediction_dict[prediction_io.SCALAR_TARGETS_KEY][:, k]
        )
        predicted_values = (
            prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][:, j] -
            prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][:, k]
        )
        climo_value = numpy.mean(
            training_down_fluxes_w_m02 - training_up_fluxes_w_m02
        )
    else:
        if plot_shortwave_errors:
            k = training_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY].index(
                example_utils.SHORTWAVE_HEATING_RATE_NAME
            )
            training_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME,
                height_m_agl=heating_rate_height_m_agl
            )
        else:
            k = training_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY].index(
                example_utils.LONGWAVE_HEATING_RATE_NAME
            )
            training_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=example_utils.LONGWAVE_HEATING_RATE_NAME,
                height_m_agl=heating_rate_height_m_agl
            )

        height_diffs_metres = numpy.absolute(
            heating_rate_height_m_agl -
            prediction_dict[prediction_io.HEIGHTS_KEY]
        )
        j = numpy.where(height_diffs_metres <= TOLERANCE)[0][0]

        target_values = (
            prediction_dict[prediction_io.VECTOR_TARGETS_KEY][:, j, k]
        )
        predicted_values = (
            prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][:, j, k]
        )
        climo_value = numpy.mean(training_values)

    # Bin examples by shortwave surface downwelling flux.
    if num_shortwave_sfc_down_flux_bins > 0:
        edge_shortwave_down_fluxes_w_m02 = numpy.linspace(
            0, MAX_SHORTWAVE_SFC_DOWN_FLUX_W_M02,
            num=num_shortwave_sfc_down_flux_bins + 1, dtype=float
        )
        edge_shortwave_down_fluxes_w_m02[-1] = numpy.inf

        k = training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY].index(
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
        )
        shortwave_down_flux_bin_indices = numpy.digitize(
            x=prediction_dict[prediction_io.SCALAR_TARGETS_KEY][:, k],
            bins=edge_shortwave_down_fluxes_w_m02, right=False
        ) - 1

        these_strings = [
            r'$\infty$' if numpy.isinf(t) and t > 0
            else r'-$\infty$' if numpy.isinf(t) and t < 0
            else '{0:d}'.format(int(numpy.round(t)))
            for t in edge_shortwave_down_fluxes_w_m02
        ]

        shortwave_down_flux_tick_labels = [
            '[{0:s}, {1:s}]'.format(a, b)
            for a, b in zip(these_strings[:-1], these_strings[1:])
        ]
    else:
        shortwave_down_flux_bin_indices = None
        shortwave_down_flux_tick_labels = None

    # Bin examples by solar zenith angle.
    if num_zenith_angle_bins > 0:
        edge_zenith_angles_rad = numpy.linspace(
            0, MAX_ZENITH_ANGLE_RAD, num=num_zenith_angle_bins + 1, dtype=float
        )
        edge_zenith_angles_rad[-1] = numpy.inf

        actual_zenith_angles_rad = example_utils.parse_example_ids(
            prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
        )[example_utils.ZENITH_ANGLES_KEY]

        zenith_angle_bin_indices = numpy.digitize(
            x=actual_zenith_angles_rad, bins=edge_zenith_angles_rad, right=False
        ) - 1

        edge_zenith_angles_rad[-1] = MAX_ZENITH_ANGLE_RAD
        edge_zenith_angles_deg = edge_zenith_angles_rad * RADIANS_TO_DEGREES
        zenith_angle_tick_labels = [
            '[{0:d}, {1:d}]'.format(
                int(numpy.round(a)), int(numpy.round(b))
            ) for a, b in
            zip(edge_zenith_angles_deg[:-1], edge_zenith_angles_deg[1:])
        ]
    else:
        zenith_angle_bin_indices = None
        zenith_angle_tick_labels = None

    # Bin examples by AOD.
    if num_aod_bins > 0:
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
            '[{0:.1f}, {1:.1f}]'.format(a, b) for a, b in zip(
                edge_aerosol_optical_depths[:-1],
                edge_aerosol_optical_depths[1:]
            )
        ]
    else:
        aod_bin_indices = None
        aod_tick_labels = None

    # Bin examples by surface temperature.
    if num_surface_temp_bins > 0:
        edge_surface_temps_kelvins = numpy.linspace(
            MIN_SURFACE_TEMP_KELVINS, MAX_SURFACE_TEMP_KELVINS,
            num=num_surface_temp_bins + 1, dtype=float
        )
        edge_surface_temps_kelvins[0] = -numpy.inf
        edge_surface_temps_kelvins[-1] = numpy.inf

        actual_surface_temps_kelvins = _get_surface_temperatures(
            prediction_dict=prediction_dict, example_dir_name=example_dir_name
        )
        surface_temp_bin_indices = numpy.digitize(
            x=actual_surface_temps_kelvins, bins=edge_surface_temps_kelvins,
            right=False
        ) - 1

        these_strings = [
            r'$\infty$' if numpy.isinf(t) and t > 0
            else r'-$\infty$' if numpy.isinf(t) and t < 0
            else '{0:d}'.format(int(numpy.round(t)))
            for t in edge_surface_temps_kelvins
        ]

        surface_temp_tick_labels = [
            '[{0:s}, {1:s}]'.format(a, b)
            for a, b in zip(these_strings[:-1], these_strings[1:])
        ]
    else:
        surface_temp_bin_indices = None
        surface_temp_tick_labels = None

    # Bin examples by longwave surface downwelling flux.
    if num_longwave_sfc_down_flux_bins > 0:
        edge_longwave_down_fluxes_w_m02 = numpy.linspace(
            0, MAX_LONGWAVE_SFC_DOWN_FLUX_W_M02,
            num=num_longwave_sfc_down_flux_bins + 1, dtype=float
        )
        edge_longwave_down_fluxes_w_m02[-1] = numpy.inf

        k = training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY].index(
            example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
        )
        longwave_down_flux_bin_indices = numpy.digitize(
            x=prediction_dict[prediction_io.SCALAR_TARGETS_KEY][:, k],
            bins=edge_longwave_down_fluxes_w_m02, right=False
        ) - 1

        these_strings = [
            r'$\infty$' if numpy.isinf(t) and t > 0
            else r'-$\infty$' if numpy.isinf(t) and t < 0
            else '{0:d}'.format(int(numpy.round(t)))
            for t in edge_longwave_down_fluxes_w_m02
        ]

        longwave_down_flux_tick_labels = [
            '[{0:s}, {1:s}]'.format(a, b)
            for a, b in zip(these_strings[:-1], these_strings[1:])
        ]
    else:
        longwave_down_flux_bin_indices = None
        longwave_down_flux_tick_labels = None

    # Bin examples by longwave TOA upwelling flux.
    if num_longwave_toa_up_flux_bins > 0:
        edge_longwave_up_fluxes_w_m02 = numpy.linspace(
            0, MAX_LONGWAVE_TOA_UP_FLUX_W_M02,
            num=num_longwave_toa_up_flux_bins + 1, dtype=float
        )
        edge_longwave_up_fluxes_w_m02[-1] = numpy.inf

        k = training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY].index(
            example_utils.LONGWAVE_TOA_UP_FLUX_NAME
        )
        longwave_up_flux_bin_indices = numpy.digitize(
            x=prediction_dict[prediction_io.SCALAR_TARGETS_KEY][:, k],
            bins=edge_longwave_up_fluxes_w_m02, right=False
        ) - 1

        these_strings = [
            r'$\infty$' if numpy.isinf(t) and t > 0
            else r'-$\infty$' if numpy.isinf(t) and t < 0
            else '{0:d}'.format(int(numpy.round(t)))
            for t in edge_longwave_up_fluxes_w_m02
        ]

        longwave_up_flux_tick_labels = [
            '[{0:s}, {1:s}]'.format(a, b)
            for a, b in zip(these_strings[:-1], these_strings[1:])
        ]
    else:
        longwave_up_flux_bin_indices = None
        longwave_up_flux_tick_labels = None

    num_bins_by_metavar = [
        num_shortwave_sfc_down_flux_bins, num_zenith_angle_bins,
        num_aod_bins, num_surface_temp_bins,
        num_longwave_sfc_down_flux_bins, num_longwave_toa_up_flux_bins
    ]
    bin_indices_by_metavar = [
        shortwave_down_flux_bin_indices, zenith_angle_bin_indices,
        aod_bin_indices, surface_temp_bin_indices,
        longwave_down_flux_bin_indices, longwave_up_flux_bin_indices
    ]
    axis_title_by_metavar = [
        r'Shortwave surface downwelling flux (W m$^{-2}$)',
        'Solar zenith angle (deg)',
        'Aerosol optical depth (unitless)', 'Surface temperature (K)',
        r'Longwave surface downwelling flux (W m$^{-2}$)',
        r'Longwave TOA upwelling flux (W m$^{-2}$)'
    ]
    tick_labels_by_metavar = [
        shortwave_down_flux_tick_labels, zenith_angle_tick_labels,
        aod_tick_labels, surface_temp_tick_labels,
        longwave_down_flux_tick_labels, longwave_up_flux_tick_labels
    ]
    file_name_abbrev_by_metavar = [
        'sw-sfc-down-flux', 'sza', 'aod', 'sfc-temp', 'lw-sfc-down-flux',
        'lw-toa-up-flux'
    ]

    for i in range(len(num_bins_by_metavar)):
        for j in range(i + 1, len(num_bins_by_metavar)):
            if num_bins_by_metavar[i] == 0 or num_bins_by_metavar[j] == 0:
                continue

            output_dir_name = '{0:s}/versus_{1:s}_and_{2:s}'.format(
                top_output_dir_name, file_name_abbrev_by_metavar[i],
                file_name_abbrev_by_metavar[j]
            )

            _plot_all_scores_one_var(
                num_first_bins=num_bins_by_metavar[i],
                num_second_bins=num_bins_by_metavar[j],
                example_to_first_bin_indices=bin_indices_by_metavar[i],
                example_to_second_bin_indices=bin_indices_by_metavar[j],
                target_values=target_values, predicted_values=predicted_values,
                climo_value=climo_value,
                x_axis_label=axis_title_by_metavar[j],
                y_axis_label=axis_title_by_metavar[i],
                x_tick_labels=tick_labels_by_metavar[j],
                y_tick_labels=tick_labels_by_metavar[i],
                heating_rate_height_m_agl=heating_rate_height_m_agl,
                plot_shortwave_errors=plot_shortwave_errors,
                output_dir_name=output_dir_name
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        plot_shortwave_errors=bool(getattr(
            INPUT_ARG_OBJECT, PLOT_SHORTWAVE_ERRORS_ARG_NAME
        )),
        heating_rate_height_m_agl=getattr(
            INPUT_ARG_OBJECT, HEATING_RATE_HEIGHT_ARG_NAME
        ),
        num_zenith_angle_bins=getattr(
            INPUT_ARG_OBJECT, NUM_ANGLE_BINS_ARG_NAME
        ),
        num_shortwave_sfc_down_flux_bins=getattr(
            INPUT_ARG_OBJECT, NUM_FLUX_BINS_ARG_NAME
        ),
        num_aod_bins=getattr(INPUT_ARG_OBJECT, NUM_AOD_BINS_ARG_NAME),
        num_surface_temp_bins=getattr(
            INPUT_ARG_OBJECT, NUM_SURFACE_TEMP_BINS_ARG_NAME
        ),
        num_longwave_sfc_down_flux_bins=getattr(
            INPUT_ARG_OBJECT, NUM_LW_DOWN_FLUX_BINS_ARG_NAME
        ),
        num_longwave_toa_up_flux_bins=getattr(
            INPUT_ARG_OBJECT, NUM_LW_UP_FLUX_BINS_ARG_NAME
        ),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
