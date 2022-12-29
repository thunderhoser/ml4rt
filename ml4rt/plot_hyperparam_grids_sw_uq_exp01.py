"""Plots each metric in hyperparam space for first shortwave UQ experiment."""

import os
import sys
import glob
import argparse
import numpy
from scipy.stats import rankdata
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gg_plotting_utils
import evaluation
import uq_evaluation
import example_utils
import prediction_io
import file_system_utils
import imagemagick_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

# UQ_METHOD_STRINGS = ['mc-dropout', 'crps', 'mc-crps']
# UQ_METHOD_STRINGS_FANCY = [
#     'Monte Carlo dropout', 'CRPS-LF', 'Combined approach'
# ]
# FIRST_LAYER_CHANNEL_COUNTS = numpy.array([32, 64, 96], dtype=int)
# DENSE_LAYER_DROPOUT_RATES = numpy.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])


UQ_METHOD_STRINGS = ['mc-dropout', 'crps']
UQ_METHOD_STRINGS_FANCY = [
    'Monte Carlo dropout', 'CRPS-LF'
]
FIRST_LAYER_CHANNEL_COUNTS = numpy.array([32, 96], dtype=int)
DENSE_LAYER_DROPOUT_RATES = numpy.array([0.01, 0.5])

COLUMN_AVG_HR_DWMSE_NAME = 'column_avg_hr_dwmse'
NEAR_SFC_HR_DWMSE_NAME = 'near_sfc_hr_dwmse'
COLUMN_AVG_HR_BIAS_NAME = 'column_avg_hr_bias'
NEAR_SFC_HR_BIAS_NAME = 'near_sfc_hr_bias'
ALL_FLUX_RMSE_NAME = 'all_flux_rmse'
NET_FLUX_RMSE_NAME = 'net_flux_rmse'
NET_FLUX_BIAS_NAME = 'net_flux_bias'

COLUMN_AVG_HR_SSREL_NAME = 'column_avg_hr_ssrel'
NEAR_SFC_HR_SSREL_NAME = 'near_sfc_hr_ssrel'
ALL_FLUX_SSREL_NAME = 'all_flux_ssrel'
NET_FLUX_SSREL_NAME = 'net_flux_ssrel'
COLUMN_AVG_HR_SSRAT_NAME = 'column_avg_hr_ssrat'
NEAR_SFC_HR_SSRAT_NAME = 'near_sfc_hr_ssrat'
ALL_FLUX_SSRAT_NAME = 'all_flux_ssrat'
NET_FLUX_SSRAT_NAME = 'net_flux_ssrat'

COLUMN_AVG_HR_MONO_FRACTION_NAME = 'column_avg_hr_mono_fraction'
NEAR_SFC_HR_MONO_FRACTION_NAME = 'near_sfc_hr_mono_fraction'
ALL_FLUX_MONO_FRACTION_NAME = 'all_flux_mono_fraction'
NET_FLUX_MONO_FRACTION_NAME = 'net_flux_mono_fraction'
COLUMN_AVG_HR_PITD_NAME = 'column_avg_hr_pitd'
NEAR_SFC_HR_PITD_NAME = 'near_sfc_hr_pitd'
ALL_FLUX_PITD_NAME = 'all_flux_pitd'
NET_FLUX_PITD_NAME = 'net_flux_pitd'

METRIC_NAMES = [
    COLUMN_AVG_HR_DWMSE_NAME, NEAR_SFC_HR_DWMSE_NAME,
    COLUMN_AVG_HR_BIAS_NAME, NEAR_SFC_HR_BIAS_NAME,
    ALL_FLUX_RMSE_NAME, NET_FLUX_RMSE_NAME, NET_FLUX_BIAS_NAME,
    COLUMN_AVG_HR_DWMSE_NAME + '_mlc', NEAR_SFC_HR_DWMSE_NAME + '_mlc',
    COLUMN_AVG_HR_BIAS_NAME + '_mlc', NEAR_SFC_HR_BIAS_NAME + '_mlc',
    ALL_FLUX_RMSE_NAME + '_mlc', NET_FLUX_RMSE_NAME + '_mlc',
    NET_FLUX_BIAS_NAME + '_mlc',
    COLUMN_AVG_HR_SSREL_NAME, NEAR_SFC_HR_SSREL_NAME,
    ALL_FLUX_SSREL_NAME, NET_FLUX_SSREL_NAME,
    COLUMN_AVG_HR_SSRAT_NAME, NEAR_SFC_HR_SSRAT_NAME,
    ALL_FLUX_SSRAT_NAME, NET_FLUX_SSRAT_NAME,
    COLUMN_AVG_HR_MONO_FRACTION_NAME, NEAR_SFC_HR_MONO_FRACTION_NAME,
    ALL_FLUX_MONO_FRACTION_NAME, NET_FLUX_MONO_FRACTION_NAME,
    COLUMN_AVG_HR_PITD_NAME, NEAR_SFC_HR_PITD_NAME,
    ALL_FLUX_PITD_NAME, NET_FLUX_PITD_NAME,
    COLUMN_AVG_HR_SSREL_NAME + '_mlc', NEAR_SFC_HR_SSREL_NAME + '_mlc',
    ALL_FLUX_SSREL_NAME + '_mlc', NET_FLUX_SSREL_NAME + '_mlc',
    COLUMN_AVG_HR_SSRAT_NAME + '_mlc', NEAR_SFC_HR_SSRAT_NAME + '_mlc',
    ALL_FLUX_SSRAT_NAME + '_mlc', NET_FLUX_SSRAT_NAME + '_mlc',
    COLUMN_AVG_HR_MONO_FRACTION_NAME + '_mlc',
    NEAR_SFC_HR_MONO_FRACTION_NAME + '_mlc',
    ALL_FLUX_MONO_FRACTION_NAME + '_mlc', NET_FLUX_MONO_FRACTION_NAME + '_mlc',
    COLUMN_AVG_HR_PITD_NAME + '_mlc', NEAR_SFC_HR_PITD_NAME + '_mlc',
    ALL_FLUX_PITD_NAME + '_mlc', NET_FLUX_PITD_NAME + '_mlc'
]

METRIC_NAMES_FANCY = [
    'column-averaged HR DWMSE', 'near-surface HR DWMSE',
    'column-averaged HR bias', 'near-surface HR bias',
    'all-flux RMSE', 'net-flux RMSE', 'net-flux bias',
    'column-averaged HR DWMSE for MLC', 'near-surface HR DWMSE for MLC',
    'column-averaged HR bias for MLC', 'near-surface HR bias for MLC',
    'all-flux RMSE for MLC', 'net-flux RMSE for MLC', 'net-flux bias for MLC',
    'column-averaged HR SSREL', 'near-surface HR SSREL',
    'all-flux SSREL', 'net-flux SSREL',
    'column-averaged HR SSRAT', 'near-surface HR SSRAT',
    'all-flux SSRAT', 'net-flux SSRAT',
    'column-averaged HR MF', 'near-surface HR MF',
    'all-flux MF', 'net-flux MF',
    'column-averaged HR PITD', 'near-surface HR PITD',
    'all-flux PITD', 'net-flux PITD',
    'column-averaged HR SSREL for MLC', 'near-surface HR SSREL for MLC',
    'all-flux SSREL for MLC', 'net-flux SSREL for MLC',
    'column-averaged HR SSRAT for MLC', 'near-surface HR SSRAT for MLC',
    'all-flux SSRAT for MLC', 'net-flux SSRAT for MLC',
    'column-averaged HR MF for MLC', 'near-surface HR MF for MLC',
    'all-flux MF for MLC', 'net-flux MF for MLC',
    'column-averaged HR PITD for MLC', 'near-surface HR PITD for MLC',
    'all-flux PITD for MLC', 'net-flux PITD for MLC'
]

METRIC_UNITS = [
    r'K$^{3}$ day$^{-3}$', r'K$^{3}$ day$^{-3}$',
    r'K day$^{-1}$', r'K day$^{-1}$',
    r'W m$^{-2}$', r'W m$^{-2}$', r'W m$^{-2}$',
    r'K$^{3}$ day$^{-3}$', r'K$^{3}$ day$^{-3}$',
    r'K day$^{-1}$', r'K day$^{-1}$',
    r'W m$^{-2}$', r'W m$^{-2}$', r'W m$^{-2}$',
    r'K day$^{-1}$', r'K day$^{-1}$',
    r'W m$^{-2}$', r'W m$^{-2}$',
    'unitless', 'unitless',
    'unitless', 'unitless',
    'unitless', 'unitless',
    'unitless', 'unitless',
    'unitless', 'unitless',
    'unitless', 'unitless',
    r'K day$^{-1}$', r'K day$^{-1}$',
    r'W m$^{-2}$', r'W m$^{-2}$',
    'unitless', 'unitless',
    'unitless', 'unitless',
    'unitless', 'unitless',
    'unitless', 'unitless',
    'unitless', 'unitless',
    'unitless', 'unitless'
]

# METRIC_NAMES = METRIC_NAMES[:14]
# METRIC_NAMES_FANCY = METRIC_NAMES_FANCY[:14]
# METRIC_UNITS = METRIC_UNITS[:14]

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.175
WHITE_COLOUR = numpy.full(3, 1.)
BLACK_COLOUR = numpy.full(3, 0.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.175
SELECTED_MARKER_INDICES = numpy.array([0, 0, 0], dtype=int)

MAIN_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis', lut=20)
BIAS_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='seismic', lut=20)
MONO_FRACTION_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='cividis', lut=20)
SSRAT_COLOUR_MAP_NAME = 'seismic'

MAIN_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))
BIAS_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))
MONO_FRACTION_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))

FONT_SIZE = 26
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

EXPERIMENT_DIR_ARG_NAME = 'experiment_dir_name'
EVAL_DATASET_ARG_NAME = 'eval_dataset_name'
ISOTONIC_FLAG_ARG_NAME = 'isotonic_flag'

EXPERIMENT_DIR_HELP_STRING = 'Name of top-level directory with models.'
EVAL_DATASET_HELP_STRING = (
    'Name of dataset used for evaluation.  This name will be expected as the '
    'name of a subdirectory.  Some examples are "validation," "testing," and '
    '"testing_perturbed_for_uq".'
)
ISOTONIC_FLAG_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot results with(out) isotonic regression.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EVAL_DATASET_ARG_NAME, type=str, required=True,
    help=EVAL_DATASET_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ISOTONIC_FLAG_ARG_NAME, type=int, required=True,
    help=ISOTONIC_FLAG_HELP_STRING
)


def _finite_percentile(input_array, percentile_level):
    """Takes percentile of input array, considering only finite values.

    :param input_array: numpy array.
    :param percentile_level: Percentile level, ranging from 0...100.
    :return: output_percentile: Percentile value.
    """

    return numpy.percentile(
        input_array[numpy.isfinite(input_array)], percentile_level
    )


def _get_ssrat_colour_scheme(max_colour_value):
    """Returns colour scheme for spread-skill ratio (SSRAT).

    :param colour_map_name: Name of colour scheme (must be accepted by
        `matplotlib.pyplot.get_cmap`).
    :return: colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :return: colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
    """

    orig_colour_map_object = pyplot.get_cmap(SSRAT_COLOUR_MAP_NAME)

    negative_values = numpy.linspace(0, 1, num=1001, dtype=float)
    positive_values = numpy.linspace(1, max_colour_value, num=1001, dtype=float)
    bias_values = numpy.concatenate((negative_values, positive_values))

    normalized_values = numpy.linspace(0, 1, num=len(bias_values), dtype=float)
    rgb_matrix = orig_colour_map_object(normalized_values)[:, :-1]

    colour_map_object = matplotlib.colors.ListedColormap(rgb_matrix)
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        bias_values, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def _plot_scores_2d(
        score_matrix, colour_map_object, colour_norm_object, x_tick_labels,
        y_tick_labels):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm`).
    :param colour_norm_object: Colour-normalizer (maps from data space to
        colour-bar space, which goes from 0...1).  This is an instance of
        `matplotlib.colors.Normalize`.
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
    pyplot.xticks(x_tick_values, x_tick_labels)
    pyplot.yticks(y_tick_values, y_tick_labels)

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=score_matrix[numpy.invert(numpy.isnan(score_matrix))],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        font_size=FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _read_metrics_one_model(model_dir_name, eval_dataset_name, isotonic_flag):
    """Reads metrics for one model.

    :param model_dir_name: Name of directory with trained model and evaluation
        data.
    :param eval_dataset_name: See documentation at top of file.
    :param isotonic_flag: Same.
    :return: metric_dict: Dictionary, where each key is a string from the list
        `METRIC_NAMES` and each value is a scalar.
    """

    metric_dict = {}
    for this_metric_name in METRIC_NAMES:
        metric_dict[this_metric_name] = numpy.nan

    model_file_pattern = '{0:s}/model*.h5'.format(model_dir_name)
    model_file_names = glob.glob(model_file_pattern)

    if len(model_file_names) == 0:
        return metric_dict

    model_file_names.sort()
    model_file_name = model_file_names[-1]

    eval_file_name_overall = '{0:s}/{1:s}{2:s}/evaluation.nc'.format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else '',
        eval_dataset_name
    )

    eval_file_name_mlc = (
        '{0:s}/{1:s}{2:s}/by_cloud_regime/multi_layer_cloud/evaluation.nc'
    ).format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else '',
        eval_dataset_name
    )

    evaluation_file_names = [eval_file_name_overall, eval_file_name_mlc]
    metric_name_suffixes = ['', '_mlc']

    for i in range(len(evaluation_file_names)):
        if not os.path.isfile(evaluation_file_names[i]):
            continue

        print('Reading data from: "{0:s}"...'.format(evaluation_file_names[i]))
        evaluation_table_xarray = evaluation.read_file(evaluation_file_names[i])
        prediction_file_name = (
            evaluation_table_xarray.attrs[evaluation.PREDICTION_FILE_KEY]
        )

        scalar_target_names = evaluation_table_xarray.coords[
            evaluation.SCALAR_FIELD_DIM
        ].values.tolist()

        aux_target_names = evaluation_table_xarray.coords[
            evaluation.AUX_TARGET_FIELD_DIM
        ].values.tolist()

        j = scalar_target_names.index(
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
        )
        down_flux_mse_w_m02 = numpy.nanmean(
            evaluation_table_xarray[evaluation.SCALAR_MSE_KEY].values[j, :]
        )

        j = scalar_target_names.index(example_utils.SHORTWAVE_TOA_UP_FLUX_NAME)
        up_flux_mse_w_m02 = numpy.nanmean(
            evaluation_table_xarray[evaluation.SCALAR_MSE_KEY].values[j, :]
        )

        j = aux_target_names.index(evaluation.SHORTWAVE_NET_FLUX_NAME)
        net_flux_mse_w_m02 = numpy.nanmean(
            evaluation_table_xarray[evaluation.AUX_MSE_KEY].values[j, :]
        )
        metric_dict[NET_FLUX_BIAS_NAME + metric_name_suffixes[i]] = (
            numpy.nanmean(
                evaluation_table_xarray[evaluation.AUX_BIAS_KEY].values[j, :]
            )
        )

        metric_dict[ALL_FLUX_RMSE_NAME + metric_name_suffixes[i]] = numpy.sqrt(
            (down_flux_mse_w_m02 + up_flux_mse_w_m02 + net_flux_mse_w_m02) / 3
        )
        metric_dict[NET_FLUX_RMSE_NAME + metric_name_suffixes[i]] = numpy.sqrt(
            net_flux_mse_w_m02
        )

        print('Reading data from: "{0:s}"...'.format(prediction_file_name))
        prediction_dict = prediction_io.read_file(prediction_file_name)
        vector_target_matrix = prediction_dict[prediction_io.VECTOR_TARGETS_KEY]
        mean_vector_prediction_matrix = numpy.mean(
            prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY], axis=-1
        )

        weight_matrix = numpy.maximum(
            numpy.absolute(vector_target_matrix),
            numpy.absolute(mean_vector_prediction_matrix)
        )
        this_error_matrix = (
            weight_matrix *
            (mean_vector_prediction_matrix - vector_target_matrix) ** 2
        )

        metric_dict[COLUMN_AVG_HR_DWMSE_NAME + metric_name_suffixes[i]] = (
            numpy.mean(this_error_matrix)
        )
        metric_dict[NEAR_SFC_HR_DWMSE_NAME + metric_name_suffixes[i]] = (
            numpy.mean(this_error_matrix[:, 0, :])
        )
        metric_dict[COLUMN_AVG_HR_BIAS_NAME + metric_name_suffixes[i]] = (
            numpy.mean(mean_vector_prediction_matrix - vector_target_matrix)
        )
        metric_dict[NEAR_SFC_HR_BIAS_NAME + metric_name_suffixes[i]] = (
            numpy.mean(
                mean_vector_prediction_matrix[:, 0, :] -
                vector_target_matrix[:, 0, :]
            )
        )

    eval_file_name_overall = '{0:s}/{1:s}{2:s}/pit_histograms.nc'.format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else '',
        eval_dataset_name
    )

    eval_file_name_mlc = (
        '{0:s}/{1:s}{2:s}/by_cloud_regime/multi_layer_cloud/pit_histograms.nc'
    ).format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else '',
        eval_dataset_name
    )

    evaluation_file_names = [eval_file_name_overall, eval_file_name_mlc]
    metric_name_suffixes = ['', '_mlc']

    for i in range(len(evaluation_file_names)):
        if not os.path.isfile(evaluation_file_names[i]):
            continue

        print('Reading data from: "{0:s}"...'.format(evaluation_file_names[i]))
        pit_table_xarray = uq_evaluation.read_pit_histograms(
            evaluation_file_names[i]
        )

        scalar_target_names = pit_table_xarray.coords[
            uq_evaluation.SCALAR_FIELD_DIM
        ].values.tolist()

        aux_target_names = pit_table_xarray.coords[
            uq_evaluation.AUX_TARGET_FIELD_DIM
        ].values.tolist()

        j = scalar_target_names.index(
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
        )
        down_flux_pitd = (
            pit_table_xarray[uq_evaluation.SCALAR_PITD_KEY].values[j]
        )

        j = scalar_target_names.index(example_utils.SHORTWAVE_TOA_UP_FLUX_NAME)
        up_flux_pitd = (
            pit_table_xarray[uq_evaluation.SCALAR_PITD_KEY].values[j]
        )

        j = aux_target_names.index(uq_evaluation.SHORTWAVE_NET_FLUX_NAME)
        net_flux_pitd = (
            pit_table_xarray[uq_evaluation.AUX_PITD_KEY].values[j]
        )
        metric_dict[NET_FLUX_PITD_NAME + metric_name_suffixes[i]] = (
            net_flux_pitd + 0.
        )
        metric_dict[ALL_FLUX_PITD_NAME + metric_name_suffixes[i]] = (
            down_flux_pitd + up_flux_pitd + net_flux_pitd
        ) / 3

        vector_target_names = pit_table_xarray.coords[
            uq_evaluation.VECTOR_FIELD_DIM
        ].values.tolist()

        j = vector_target_names.index(example_utils.SHORTWAVE_HEATING_RATE_NAME)

        metric_dict[COLUMN_AVG_HR_PITD_NAME + metric_name_suffixes[i]] = (
            numpy.nanmean(
                pit_table_xarray[uq_evaluation.VECTOR_PITD_KEY].values[j, :]
            )
        )

        metric_dict[NEAR_SFC_HR_PITD_NAME + metric_name_suffixes[i]] = (
            pit_table_xarray[uq_evaluation.VECTOR_PITD_KEY].values[j, 0]
        )

    eval_file_name_overall = '{0:s}/{1:s}{2:s}/spread_vs_skill.nc'.format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else '',
        eval_dataset_name
    )

    eval_file_name_mlc = (
        '{0:s}/{1:s}{2:s}/by_cloud_regime/multi_layer_cloud/spread_vs_skill.nc'
    ).format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else '',
        eval_dataset_name
    )

    evaluation_file_names = [eval_file_name_overall, eval_file_name_mlc]
    metric_name_suffixes = ['', '_mlc']

    for i in range(len(evaluation_file_names)):
        if not os.path.isfile(evaluation_file_names[i]):
            continue

        print('Reading data from: "{0:s}"...'.format(evaluation_file_names[i]))
        ss_table_xarray = uq_evaluation.read_spread_vs_skill(
            evaluation_file_names[i]
        )

        scalar_target_names = ss_table_xarray.coords[
            uq_evaluation.SCALAR_FIELD_DIM
        ].values.tolist()

        aux_target_names = ss_table_xarray.coords[
            uq_evaluation.AUX_TARGET_FIELD_DIM
        ].values.tolist()

        j = scalar_target_names.index(
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
        )
        down_flux_ssrel = (
            ss_table_xarray[uq_evaluation.SCALAR_SSREL_KEY].values[j]
        )
        down_flux_ssrat = (
            ss_table_xarray[uq_evaluation.SCALAR_SSRAT_KEY].values[j]
        )

        j = scalar_target_names.index(example_utils.SHORTWAVE_TOA_UP_FLUX_NAME)
        up_flux_ssrel = (
            ss_table_xarray[uq_evaluation.SCALAR_SSREL_KEY].values[j]
        )
        up_flux_ssrat = (
            ss_table_xarray[uq_evaluation.SCALAR_SSRAT_KEY].values[j]
        )

        j = aux_target_names.index(uq_evaluation.SHORTWAVE_NET_FLUX_NAME)
        net_flux_ssrel = (
            ss_table_xarray[uq_evaluation.AUX_SSREL_KEY].values[j]
        )
        net_flux_ssrat = (
            ss_table_xarray[uq_evaluation.AUX_SSRAT_KEY].values[j]
        )

        metric_dict[NET_FLUX_SSREL_NAME + metric_name_suffixes[i]] = (
            net_flux_ssrel + 0.
        )
        metric_dict[NET_FLUX_SSRAT_NAME + metric_name_suffixes[i]] = (
            net_flux_ssrat + 0.
        )
        metric_dict[ALL_FLUX_SSREL_NAME + metric_name_suffixes[i]] = (
            down_flux_ssrel + up_flux_ssrel + net_flux_ssrel
        ) / 3
        metric_dict[ALL_FLUX_SSRAT_NAME + metric_name_suffixes[i]] = (
            down_flux_ssrat + up_flux_ssrat + net_flux_ssrat
        ) / 3

        vector_target_names = ss_table_xarray.coords[
            uq_evaluation.VECTOR_FIELD_DIM
        ].values.tolist()

        j = vector_target_names.index(example_utils.SHORTWAVE_HEATING_RATE_NAME)

        metric_dict[COLUMN_AVG_HR_SSREL_NAME + metric_name_suffixes[i]] = (
            numpy.nanmean(
                ss_table_xarray[uq_evaluation.VECTOR_SSREL_KEY].values[j, :]
            )
        )

        metric_dict[COLUMN_AVG_HR_SSRAT_NAME + metric_name_suffixes[i]] = (
            numpy.nanmean(
                ss_table_xarray[uq_evaluation.VECTOR_SSRAT_KEY].values[j, :]
            )
        )

        metric_dict[NEAR_SFC_HR_SSREL_NAME + metric_name_suffixes[i]] = (
            ss_table_xarray[uq_evaluation.VECTOR_SSREL_KEY].values[j, 0]
        )

        metric_dict[NEAR_SFC_HR_SSRAT_NAME + metric_name_suffixes[i]] = (
            ss_table_xarray[uq_evaluation.VECTOR_SSRAT_KEY].values[j, 0]
        )

    eval_file_name_overall = (
        '{0:s}/{1:s}{2:s}/discard_test_for_heating_rates.nc'
    ).format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else '',
        eval_dataset_name
    )

    eval_file_name_mlc = (
        '{0:s}/{1:s}{2:s}/by_cloud_regime/multi_layer_cloud/'
        'discard_test_for_heating_rates.nc'
    ).format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else '',
        eval_dataset_name
    )

    evaluation_file_names = [eval_file_name_overall, eval_file_name_mlc]
    metric_name_suffixes = ['', '_mlc']

    for i in range(len(evaluation_file_names)):
        if not os.path.isfile(evaluation_file_names[i]):
            continue

        print('Reading data from: "{0:s}"...'.format(evaluation_file_names[i]))
        discard_test_table_xarray = uq_evaluation.read_discard_results(
            evaluation_file_names[i]
        )
        t = discard_test_table_xarray

        scalar_target_names = (
            t.coords[uq_evaluation.SCALAR_FIELD_DIM].values.tolist()
        )
        aux_target_names = (
            t.coords[uq_evaluation.AUX_TARGET_FIELD_DIM].values.tolist()
        )

        j = scalar_target_names.index(
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
        )
        down_flux_mono_fraction = (
            t[uq_evaluation.SCALAR_MONOTONICITY_FRACTION_KEY].values[j]
        )

        j = scalar_target_names.index(
            example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
        )
        up_flux_mono_fraction = (
            t[uq_evaluation.SCALAR_MONOTONICITY_FRACTION_KEY].values[j]
        )

        j = aux_target_names.index(uq_evaluation.SHORTWAVE_NET_FLUX_NAME)
        net_flux_mono_fraction = (
            t[uq_evaluation.AUX_MONOTONICITY_FRACTION_KEY].values[j]
        )

        metric_dict[NET_FLUX_MONO_FRACTION_NAME + metric_name_suffixes[i]] = (
            net_flux_mono_fraction + 0.
        )
        metric_dict[ALL_FLUX_MONO_FRACTION_NAME + metric_name_suffixes[i]] = (
            down_flux_mono_fraction + up_flux_mono_fraction +
            net_flux_mono_fraction
        ) / 3

        vector_target_names = (
            t.coords[uq_evaluation.VECTOR_FIELD_DIM].values.tolist()
        )
        j = vector_target_names.index(example_utils.SHORTWAVE_HEATING_RATE_NAME)

        metric_dict[
            COLUMN_AVG_HR_MONO_FRACTION_NAME + metric_name_suffixes[i]
        ] = numpy.nanmean(
            t[uq_evaluation.VECTOR_MONOTONICITY_FRACTION_KEY].values[j, :]
        )

        metric_dict[
            NEAR_SFC_HR_MONO_FRACTION_NAME + metric_name_suffixes[i]
        ] = t[uq_evaluation.VECTOR_MONOTONICITY_FRACTION_KEY].values[j, 0]

    return metric_dict


def _print_ranking_all_metrics(metric_matrix, main_metric_name):
    """Prints ranking for all metrics.

    U = number of UQ methods
    D = number of dropout rates
    S = number of spectral complexities
    M = number of metrics

    :param metric_matrix: U-by-D-by-S-by-M numpy array of metric values.
    :param main_metric_name: Name of main metric.
    """

    main_metric_index = METRIC_NAMES.index(main_metric_name)
    values_1d = numpy.ravel(metric_matrix[..., main_metric_index])

    if 'bias' in main_metric_name:
        values_1d[numpy.isnan(values_1d)] = numpy.inf
        sort_indices_1d = numpy.argsort(numpy.absolute(values_1d))
    elif 'mono_fraction' in main_metric_name:
        values_1d[numpy.isnan(values_1d)] = -numpy.inf
        sort_indices_1d = numpy.argsort(-values_1d)
    elif 'ssrat' in main_metric_name:
        values_1d[numpy.isnan(values_1d)] = numpy.inf
        sort_indices_1d = numpy.argsort(numpy.absolute(1. - values_1d))
    else:
        values_1d[numpy.isnan(values_1d)] = numpy.inf
        sort_indices_1d = numpy.argsort(values_1d)

    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, metric_matrix.shape[:-1]
    )

    metric_rank_matrix = numpy.full(metric_matrix.shape, numpy.nan)

    for m in range(len(METRIC_NAMES)):
        these_values = numpy.ravel(metric_matrix[..., m])

        if 'bias' in main_metric_name:
            these_values = numpy.absolute(these_values)
            these_values[numpy.isnan(these_values)] = numpy.inf
        elif 'mono_fraction' in main_metric_name:
            these_values = -1 * these_values
            these_values[numpy.isnan(these_values)] = -numpy.inf
        elif 'ssrat' in main_metric_name:
            these_values = numpy.absolute(1. - these_values)
            these_values[numpy.isnan(these_values)] = numpy.inf
        else:
            these_values[numpy.isnan(these_values)] = numpy.inf

        metric_rank_matrix[..., m] = numpy.reshape(
            rankdata(these_values, method='average'),
            metric_rank_matrix.shape[:-1]
        )

    names = METRIC_NAMES
    mrm = metric_rank_matrix

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:s} ... dropout rate = {1:.2f} ... spectral complexity = {2:d}\n'
            'DWMSE ranks (all, sfc, MLC, MLC sfc) = '
            '{3:.1f}, {4:.1f}, {5:.1f}, {6:.1f} ... '
            'HR-bias ranks (all, sfc, MLC, MLC sfc) = '
            '{7:.1f}, {8:.1f}, {9:.1f}, {10:.1f}\n'
            'flux-RMSE ranks (all, net, MLC, MLC net) = '
            '{11:.1f}, {12:.1f}, {13:.1f}, {14:.1f} ... '
            'net-flux-bias ranks (all, MLC) = '
            '{15:.1f}, {16:.1f}\n'
            'HR-SSREL ranks (all, sfc, MLC, MLC sfc) = '
            '{17:.1f}, {18:.1f}, {19:.1f}, {20:.1f} ... '
            'flux-SSREL ranks (all, net, MLC, MLC net) = '
            '{21:.1f}, {22:.1f}, {23:.1f}, {24:.1f}\n'
            'HR-SSRAT ranks (all, sfc, MLC, MLC sfc) = '
            '{25:.1f}, {26:.1f}, {27:.1f}, {28:.1f} ... '
            'flux-SSRAT ranks (all, net, MLC, MLC net) = '
            '{29:.1f}, {30:.1f}, {31:.1f}, {32:.1f}\n'
            'HR-MF ranks (all, sfc, MLC, MLC sfc) = '
            '{33:.1f}, {34:.1f}, {35:.1f}, {36:.1f} ... '
            'flux-MF ranks (all, net, MLC, MLC net) = '
            '{37:.1f}, {38:.1f}, {39:.1f}, {40:.1f}\n'
            'HR-PITD ranks (all, sfc, MLC, MLC sfc) = '
            '{41:.1f}, {42:.1f}, {43:.1f}, {44:.1f} ... '
            'flux-PITD ranks (all, net, MLC, MLC net) = '
            '{45:.1f}, {46:.1f}, {47:.1f}, {48:.1f}\n\n'
        ).format(
            UQ_METHOD_STRINGS[i],
            DENSE_LAYER_DROPOUT_RATES[j],
            FIRST_LAYER_CHANNEL_COUNTS[k],
            mrm[i, j, k, names.index(COLUMN_AVG_HR_DWMSE_NAME)],
            mrm[i, j, k, names.index(NEAR_SFC_HR_DWMSE_NAME)],
            mrm[i, j, k, names.index(COLUMN_AVG_HR_DWMSE_NAME + '_mlc')],
            mrm[i, j, k, names.index(NEAR_SFC_HR_DWMSE_NAME + '_mlc')],
            mrm[i, j, k, names.index(COLUMN_AVG_HR_BIAS_NAME)],
            mrm[i, j, k, names.index(NEAR_SFC_HR_BIAS_NAME)],
            mrm[i, j, k, names.index(COLUMN_AVG_HR_BIAS_NAME + '_mlc')],
            mrm[i, j, k, names.index(NEAR_SFC_HR_BIAS_NAME + '_mlc')],
            mrm[i, j, k, names.index(ALL_FLUX_RMSE_NAME)],
            mrm[i, j, k, names.index(NET_FLUX_RMSE_NAME)],
            mrm[i, j, k, names.index(ALL_FLUX_RMSE_NAME + '_mlc')],
            mrm[i, j, k, names.index(NET_FLUX_RMSE_NAME + '_mlc')],
            mrm[i, j, k, names.index(NET_FLUX_BIAS_NAME)],
            mrm[i, j, k, names.index(NET_FLUX_BIAS_NAME + '_mlc')],
            mrm[i, j, k, names.index(COLUMN_AVG_HR_SSREL_NAME)],
            mrm[i, j, k, names.index(NEAR_SFC_HR_SSREL_NAME)],
            mrm[i, j, k, names.index(COLUMN_AVG_HR_SSREL_NAME + '_mlc')],
            mrm[i, j, k, names.index(NEAR_SFC_HR_SSREL_NAME + '_mlc')],
            mrm[i, j, k, names.index(ALL_FLUX_SSREL_NAME)],
            mrm[i, j, k, names.index(NET_FLUX_SSREL_NAME)],
            mrm[i, j, k, names.index(ALL_FLUX_SSREL_NAME + '_mlc')],
            mrm[i, j, k, names.index(NET_FLUX_SSREL_NAME + '_mlc')],
            mrm[i, j, k, names.index(COLUMN_AVG_HR_SSRAT_NAME)],
            mrm[i, j, k, names.index(NEAR_SFC_HR_SSRAT_NAME)],
            mrm[i, j, k, names.index(COLUMN_AVG_HR_SSRAT_NAME + '_mlc')],
            mrm[i, j, k, names.index(NEAR_SFC_HR_SSRAT_NAME + '_mlc')],
            mrm[i, j, k, names.index(ALL_FLUX_SSRAT_NAME)],
            mrm[i, j, k, names.index(NET_FLUX_SSRAT_NAME)],
            mrm[i, j, k, names.index(ALL_FLUX_SSRAT_NAME + '_mlc')],
            mrm[i, j, k, names.index(NET_FLUX_SSRAT_NAME + '_mlc')],
            mrm[i, j, k, names.index(COLUMN_AVG_HR_MONO_FRACTION_NAME)],
            mrm[i, j, k, names.index(NEAR_SFC_HR_MONO_FRACTION_NAME)],
            mrm[i, j, k, names.index(COLUMN_AVG_HR_MONO_FRACTION_NAME + '_mlc')],
            mrm[i, j, k, names.index(NEAR_SFC_HR_MONO_FRACTION_NAME + '_mlc')],
            mrm[i, j, k, names.index(ALL_FLUX_MONO_FRACTION_NAME)],
            mrm[i, j, k, names.index(NET_FLUX_MONO_FRACTION_NAME)],
            mrm[i, j, k, names.index(ALL_FLUX_MONO_FRACTION_NAME + '_mlc')],
            mrm[i, j, k, names.index(NET_FLUX_MONO_FRACTION_NAME + '_mlc')],
            mrm[i, j, k, names.index(COLUMN_AVG_HR_PITD_NAME)],
            mrm[i, j, k, names.index(NEAR_SFC_HR_PITD_NAME)],
            mrm[i, j, k, names.index(COLUMN_AVG_HR_PITD_NAME + '_mlc')],
            mrm[i, j, k, names.index(NEAR_SFC_HR_PITD_NAME + '_mlc')],
            mrm[i, j, k, names.index(ALL_FLUX_PITD_NAME)],
            mrm[i, j, k, names.index(NET_FLUX_PITD_NAME)],
            mrm[i, j, k, names.index(ALL_FLUX_PITD_NAME + '_mlc')],
            mrm[i, j, k, names.index(NET_FLUX_PITD_NAME + '_mlc')]
        ))


def _print_ranking_one_metric(metric_matrix, metric_index):
    """Prints ranking for one metric.

    U = number of UQ methods
    D = number of dropout rates
    S = number of spectral complexities
    M = number of metrics

    :param metric_matrix: U-by-D-by-S-by-M numpy array of metric values.
    :param metric_index: Will print ranking for [i]th metric, where
        i = `metric_index`.
    """

    values_1d = numpy.ravel(metric_matrix[..., metric_index])

    if 'bias' in METRIC_NAMES[metric_index]:
        values_1d[numpy.isnan(values_1d)] = numpy.inf
        sort_indices_1d = numpy.argsort(numpy.absolute(values_1d))
    elif 'mono_fraction' in METRIC_NAMES[metric_index]:
        values_1d[numpy.isnan(values_1d)] = -numpy.inf
        sort_indices_1d = numpy.argsort(-values_1d)
    elif 'ssrat' in METRIC_NAMES[metric_index]:
        values_1d[numpy.isnan(values_1d)] = numpy.inf
        sort_indices_1d = numpy.argsort(numpy.absolute(1. - values_1d))
    else:
        values_1d[numpy.isnan(values_1d)] = numpy.inf
        sort_indices_1d = numpy.argsort(values_1d)

    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, metric_matrix.shape[:-1]
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:d}th-best {1:s} = '
            '{2:.3g} {3:s} ... '
            '{4:s} ... '
            'dropout rate = {5:.2f} ... '
            'spectral complexity = {6:d}'
        ).format(
            m + 1, METRIC_NAMES_FANCY[metric_index],
            metric_matrix[i, j, k, metric_index], METRIC_UNITS[metric_index],
            UQ_METHOD_STRINGS[i],
            DENSE_LAYER_DROPOUT_RATES[j],
            FIRST_LAYER_CHANNEL_COUNTS[k]
        ))


def _run(experiment_dir_name, eval_dataset_name, isotonic_flag):
    """Plots each metric in hyperparam space for first shortwave UQ experiment.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param eval_dataset_name: Same.
    :param isotonic_flag: Same.
    """

    num_uq_methods = len(UQ_METHOD_STRINGS)
    num_dropout_rates = len(DENSE_LAYER_DROPOUT_RATES)
    num_channel_counts = len(FIRST_LAYER_CHANNEL_COUNTS)
    num_metrics = len(METRIC_NAMES)

    y_tick_labels = [
        '{0:.2f}'.format(d) for d in DENSE_LAYER_DROPOUT_RATES
    ]
    x_tick_labels = [
        '{0:d}'.format(c) for c in FIRST_LAYER_CHANNEL_COUNTS
    ]

    y_axis_label = 'Dropout rate for dense layers'
    x_axis_label = 'Spectral complexity'

    metric_matrix = numpy.full(
        (num_uq_methods, num_dropout_rates, num_channel_counts, num_metrics),
        numpy.nan
    )

    for i in range(num_uq_methods):
        for j in range(num_dropout_rates):
            for k in range(num_channel_counts):
                this_model_dir_name = (
                    '{0:s}/{1:s}_num-first-layer-channels={2:03d}_'
                    'dense-layer-dropout-rate={3:.2f}'
                ).format(
                    experiment_dir_name,
                    UQ_METHOD_STRINGS[i].replace('_', '-'),
                    FIRST_LAYER_CHANNEL_COUNTS[k],
                    DENSE_LAYER_DROPOUT_RATES[j]
                )

                this_metric_dict = _read_metrics_one_model(
                    model_dir_name=this_model_dir_name,
                    eval_dataset_name=eval_dataset_name,
                    isotonic_flag=isotonic_flag
                )

                for m in range(num_metrics):
                    metric_matrix[i, j, k, m] = this_metric_dict[
                        METRIC_NAMES[m]
                    ]

    print(SEPARATOR_STRING)

    for m in range(num_metrics):
        _print_ranking_one_metric(metric_matrix=metric_matrix, metric_index=m)
        print(SEPARATOR_STRING)

    _print_ranking_all_metrics(
        metric_matrix=metric_matrix, main_metric_name=COLUMN_AVG_HR_DWMSE_NAME
    )
    print(SEPARATOR_STRING)

    _print_ranking_all_metrics(
        metric_matrix=metric_matrix, main_metric_name=COLUMN_AVG_HR_SSREL_NAME
    )
    print(SEPARATOR_STRING)

    output_dir_name = '{0:s}/hyperparam_grids/shortwave{1:s}'.format(
        experiment_dir_name, '/isotonic_regression' if isotonic_flag else ''
    )
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    for m in range(num_metrics):
        if numpy.all(numpy.isnan(metric_matrix[..., m])):
            continue

        panel_file_names = [''] * num_uq_methods

        for i in range(num_uq_methods):
            if 'bias' in METRIC_NAMES[m]:
                max_colour_value = _finite_percentile(
                    numpy.absolute(metric_matrix[..., m]), 95
                )
                min_colour_value = -1 * max_colour_value

                colour_norm_object = matplotlib.colors.Normalize(
                    vmin=min_colour_value, vmax=max_colour_value, clip=False
                )
                colour_map_object = BIAS_COLOUR_MAP_OBJECT

                best_linear_index = numpy.nanargmin(
                    numpy.absolute(numpy.ravel(metric_matrix[..., m]))
                )
                marker_colour = BLACK_COLOUR

            elif 'mono_fraction' in METRIC_NAMES[m]:
                max_colour_value = _finite_percentile(
                    numpy.absolute(metric_matrix[..., m]), 100
                )
                min_colour_value = _finite_percentile(
                    numpy.absolute(metric_matrix[..., m]), 5
                )

                colour_norm_object = matplotlib.colors.Normalize(
                    vmin=min_colour_value, vmax=max_colour_value, clip=False
                )
                colour_map_object = MONO_FRACTION_COLOUR_MAP_OBJECT

                best_linear_index = numpy.nanargmax(
                    numpy.ravel(metric_matrix[..., m])
                )
                marker_colour = BLACK_COLOUR

            elif 'ssrat' in METRIC_NAMES[m]:
                this_offset = _finite_percentile(
                    numpy.absolute(metric_matrix[..., m] - 1.), 97.5
                )
                colour_map_object, colour_norm_object = (
                    _get_ssrat_colour_scheme(max_colour_value=1. + this_offset)
                )

                best_linear_index = numpy.nanargmin(
                    numpy.absolute(numpy.ravel(metric_matrix[..., m]) - 1.)
                )
                marker_colour = BLACK_COLOUR

            else:
                max_colour_value = _finite_percentile(
                    numpy.absolute(metric_matrix[..., m]), 95
                )
                min_colour_value = _finite_percentile(
                    numpy.absolute(metric_matrix[..., m]), 0
                )

                colour_norm_object = matplotlib.colors.Normalize(
                    vmin=min_colour_value, vmax=max_colour_value, clip=False
                )
                colour_map_object = MAIN_COLOUR_MAP_OBJECT

                best_linear_index = numpy.nanargmin(
                    numpy.ravel(metric_matrix[..., m])
                )
                marker_colour = WHITE_COLOUR

            figure_object, axes_object = _plot_scores_2d(
                score_matrix=metric_matrix[i, ..., m],
                colour_map_object=colour_map_object,
                colour_norm_object=colour_norm_object,
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            best_indices = numpy.unravel_index(
                best_linear_index, metric_matrix[..., m].shape
            )

            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / metric_matrix.shape[2]
            )

            if best_indices[0] == i:
                axes_object.plot(
                    best_indices[2], best_indices[1],
                    linestyle='None', marker=BEST_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=marker_colour,
                    markeredgecolor=marker_colour
                )

            if SELECTED_MARKER_INDICES[0] == i:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[2], SELECTED_MARKER_INDICES[1],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=marker_colour,
                    markeredgecolor=marker_colour
                )

            axes_object.set_xlabel(x_axis_label)
            axes_object.set_ylabel(y_axis_label)
            axes_object.set_title(UQ_METHOD_STRINGS_FANCY[i])

            panel_file_names[i] = '{0:s}/{1:s}_{2:s}.jpg'.format(
                output_dir_name,
                METRIC_NAMES[m].replace('_', '-'),
                UQ_METHOD_STRINGS[i].replace('_', '-')
            )

            print('Saving figure to: "{0:s}"...'.format(panel_file_names[i]))
            figure_object.savefig(
                panel_file_names[i], dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

        num_panel_rows = int(numpy.floor(
            numpy.sqrt(num_uq_methods)
        ))
        num_panel_columns = int(numpy.ceil(
            float(num_uq_methods) / num_panel_rows
        ))
        concat_figure_file_name = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, METRIC_NAMES[m]
        )

        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))
        imagemagick_utils.concatenate_images(
            input_file_names=panel_file_names,
            output_file_name=concat_figure_file_name,
            num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=int(1e7)
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        eval_dataset_name=getattr(INPUT_ARG_OBJECT, EVAL_DATASET_ARG_NAME),
        isotonic_flag=bool(getattr(INPUT_ARG_OBJECT, ISOTONIC_FLAG_ARG_NAME))
    )
