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

UQ_METHOD_STRINGS = ['mc-dropout', 'crps', 'mc-crps']
UQ_METHOD_STRINGS_FANCY = [
    'Monte Carlo dropout', 'CRPS-LF', 'Combined approach'
]
FIRST_LAYER_CHANNEL_COUNTS = numpy.array([32, 64, 96], dtype=int)
DENSE_LAYER_DROPOUT_RATES = numpy.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])

METRIC_NAMES_ABBREV = [
    'column_avg_hr_dwmse', 'near_sfc_hr_dwmse',
    'column_avg_hr_bias', 'near_sfc_hr_bias',
    'all_flux_rmse', 'net_flux_rmse', 'net_flux_bias',
    'column_avg_hr_dwmse_mlc', 'near_sfc_hr_dwmse_mlc',
    'column_avg_hr_bias_mlc', 'near_sfc_hr_bias_mlc',
    'all_flux_rmse_mlc', 'net_flux_rmse_mlc', 'net_flux_bias_mlc',
    'column_avg_hr_ssrel', 'near_sfc_hr_ssrel',
    'all_flux_ssrel', 'net_flux_ssrel',
    'column_avg_hr_ssrat', 'near_sfc_hr_ssrat',
    'all_flux_ssrat', 'net_flux_ssrat',
    'column_avg_hr_mono_fraction', 'near_sfc_hr_mono_fraction',
    'all_flux_mono_fraction', 'net_flux_mono_fraction',
    'column_avg_hr_pitd', 'near_sfc_hr_pitd',
    'all_flux_pitd', 'net_flux_pitd',
    'column_avg_hr_ssrel_mlc', 'near_sfc_hr_ssrel_mlc',
    'all_flux_ssrel_mlc', 'net_flux_ssrel_mlc',
    'column_avg_hr_ssrat_mlc', 'near_sfc_hr_ssrat_mlc',
    'all_flux_ssrat_mlc', 'net_flux_ssrat_mlc',
    'column_avg_hr_mono_fraction_mlc', 'near_sfc_hr_mono_fraction_mlc',
    'all_flux_mono_fraction_mlc', 'net_flux_mono_fraction_mlc',
    'column_avg_hr_pitd_mlc', 'near_sfc_hr_pitd_mlc',
    'all_flux_pitd_mlc', 'net_flux_pitd_mlc'
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

METRIC_IS_BIAS_FLAGS = numpy.array(
    [0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1], dtype=bool
)
METRIC_IS_BIAS_FLAGS = numpy.concatenate((
    METRIC_IS_BIAS_FLAGS,
    numpy.full(32, False, dtype=bool)
))

METRIC_IS_POS_ORIENTED_FLAGS = numpy.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0
], dtype=bool)

METRIC_IS_SSRAT_FLAGS = numpy.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0
], dtype=bool)

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.175
WHITE_COLOUR = numpy.full(3, 1.)
BLACK_COLOUR = numpy.full(3, 0.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.175
SELECTED_MARKER_INDICES = numpy.array([0, 0, 0], dtype=int)

# TODO(thunderhoser): Plot SSRAT like freq bias.
# TODO(thunderhoser): Make eval set (perturbed testing or whatever) an input arg.

MAIN_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis', lut=20)
BIAS_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='seismic', lut=20)
MAIN_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))
BIAS_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))

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
ISOTONIC_FLAG_ARG_NAME = 'isotonic_flag'

EXPERIMENT_DIR_HELP_STRING = 'Name of top-level directory with models.'
ISOTONIC_FLAG_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot results with(out) isotonic regression.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
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


def _plot_scores_2d(
        score_matrix, min_colour_value, max_colour_value, x_tick_labels,
        y_tick_labels):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
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

    if min_colour_value is None:
        colour_map_object = BIAS_COLOUR_MAP_OBJECT
        min_colour_value = -1 * max_colour_value
    else:
        colour_map_object = MAIN_COLOUR_MAP_OBJECT

    axes_object.imshow(
        score_matrix, cmap=colour_map_object, origin='lower',
        vmin=min_colour_value, vmax=max_colour_value
    )

    x_tick_values = numpy.linspace(
        0, score_matrix.shape[1] - 1, num=score_matrix.shape[1], dtype=float
    )
    y_tick_values = numpy.linspace(
        0, score_matrix.shape[0] - 1, num=score_matrix.shape[0], dtype=float
    )

    pyplot.xticks(x_tick_values, x_tick_labels)
    pyplot.yticks(y_tick_values, y_tick_labels)

    colour_norm_object = matplotlib.colors.Normalize(
        vmin=min_colour_value, vmax=max_colour_value, clip=False
    )

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


def _read_metrics_one_model(model_dir_name, isotonic_flag):
    """Reads metrics for one model.

    :param model_dir_name: Name of directory with trained model and evaluation
        data.
    :param isotonic_flag: Boolean flag.  If True (False), will return metrics for
        model with(out) isotonic regression.
    :return: metric_dict: Dictionary, where each key is a string from the list
        `METRIC_NAMES_ABBREV` and each value is a scalar.
    """

    metric_dict = {}
    for this_metric_name in METRIC_NAMES_ABBREV:
        metric_dict[this_metric_name] = numpy.nan

    model_file_pattern = '{0:s}/model*.h5'.format(model_dir_name)
    model_file_names = glob.glob(model_file_pattern)

    if len(model_file_names) == 0:
        return metric_dict

    model_file_names.sort()
    model_file_name = model_file_names[-1]

    eval_file_name_overall = (
        '{0:s}/{1:s}testing_perturbed_for_uq/evaluation.nc'
    ).format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else ''
    )

    eval_file_name_mlc = (
        '{0:s}/{1:s}testing_perturbed_for_uq/by_cloud_regime/multi_layer_cloud/'
        'evaluation.nc'
    ).format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else ''
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
        metric_dict['net_flux_bias' + metric_name_suffixes[i]] = numpy.nanmean(
            evaluation_table_xarray[evaluation.AUX_BIAS_KEY].values[j, :]
        )

        metric_dict['all_flux_rmse' + metric_name_suffixes[i]] = numpy.sqrt(
            (down_flux_mse_w_m02 + up_flux_mse_w_m02 + net_flux_mse_w_m02) / 3
        )
        metric_dict['net_flux_rmse' + metric_name_suffixes[i]] = numpy.sqrt(
            net_flux_mse_w_m02
        )

        print('Reading data from: "{0:s}"...'.format(prediction_file_name))
        prediction_dict = prediction_io.read_file(prediction_file_name)
        vector_target_matrix = prediction_dict[prediction_io.VECTOR_TARGETS_KEY]
        vector_prediction_matrix = (
            prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY]
        )

        weight_matrix = numpy.maximum(
            numpy.absolute(vector_target_matrix),
            numpy.absolute(vector_prediction_matrix)
        )
        this_error_matrix = (
            weight_matrix *
            (vector_prediction_matrix - vector_target_matrix) ** 2
        )

        metric_dict['column_avg_hr_dwmse' + metric_name_suffixes[i]] = (
            numpy.mean(this_error_matrix)
        )
        metric_dict['near_sfc_hr_dwmse' + metric_name_suffixes[i]] = numpy.mean(
            this_error_matrix[:, 0, :]
        )
        metric_dict['column_avg_hr_bias' + metric_name_suffixes[i]] = (
            numpy.mean(
                vector_prediction_matrix - vector_target_matrix
            )
        )
        metric_dict['near_sfc_hr_bias' + metric_name_suffixes[i]] = numpy.mean(
            vector_prediction_matrix[:, 0, :] - vector_target_matrix[:, 0, :]
        )

    eval_file_name_overall = (
        '{0:s}/{1:s}testing_perturbed_for_uq/pit_histograms.nc'
    ).format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else ''
    )

    eval_file_name_mlc = (
        '{0:s}/{1:s}testing_perturbed_for_uq/by_cloud_regime/multi_layer_cloud/'
        'pit_histograms.nc'
    ).format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else ''
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
        metric_dict['net_flux_pitd' + metric_name_suffixes[i]] = (
            net_flux_pitd + 0.
        )
        metric_dict['all_flux_pitd' + metric_name_suffixes[i]] = (
            down_flux_pitd + up_flux_pitd + net_flux_pitd
        ) / 3

        vector_target_names = pit_table_xarray.coords[
            uq_evaluation.VECTOR_FIELD_DIM
        ].values.tolist()

        j = vector_target_names.index(example_utils.SHORTWAVE_HEATING_RATE_NAME)

        metric_dict[
            'column_avg_hr_pitd' + metric_name_suffixes[i]
        ] = numpy.nanmean(
            pit_table_xarray[uq_evaluation.VECTOR_PITD_KEY].values[j, :]
        )

        metric_dict[
            'near_sfc_hr_pitd' + metric_name_suffixes[i]
        ] = pit_table_xarray[uq_evaluation.VECTOR_PITD_KEY].values[j, 0]

    # TODO(thunderhoser): Keep adding UQ metrics to script.  Left off here.
    # TODO(thunderhoser): Turn all metric names into constants (defined).

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

    main_metric_index = METRIC_NAMES_ABBREV.index(main_metric_name)

    if METRIC_IS_BIAS_FLAGS[main_metric_index]:
        these_values = numpy.ravel(numpy.absolute(
            metric_matrix[..., main_metric_index]
        ))
    else:
        these_values = numpy.ravel(metric_matrix[..., main_metric_index])

    these_values[numpy.isnan(these_values)] = numpy.inf
    sort_indices_1d = numpy.argsort(these_values)
    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, metric_matrix.shape[:-1]
    )

    metric_rank_matrix = numpy.full(metric_matrix.shape, numpy.nan)

    for m in range(len(METRIC_NAMES_ABBREV)):
        if METRIC_IS_BIAS_FLAGS[m]:
            these_values = numpy.ravel(numpy.absolute(metric_matrix[..., m]))
        else:
            these_values = numpy.ravel(metric_matrix[..., m])

        these_values[numpy.isnan(these_values)] = numpy.inf
        metric_rank_matrix[..., m] = numpy.reshape(
            rankdata(these_values, method='average'),
            metric_rank_matrix.shape[:-1]
        )

    names = METRIC_NAMES_ABBREV

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:s} ... dropout rate = {1:.2f} ... spectral cplxity = {2:d} ... '
            'DWMSE ranks (all, sfc, MLC, MLC sfc) = '
            '{3:.1f}, {4:.1f}, {5:.1f}, {6:.1f} ... '
            'HR-bias ranks (all, sfc, MLC, MLC sfc) = '
            '{7:.1f}, {8:.1f}, {9:.1f}, {10:.1f} ... '
            'flux-RMSE ranks (all, net, MLC, MLC net) = '
            '{11:.1f}, {12:.1f}, {13:.1f}, {14:.1f} ... '
            'net-flux-bias ranks (all, MLC) = '
            '{15:.1f}, {16:.1f}'
        ).format(
            UQ_METHOD_STRINGS[i],
            DENSE_LAYER_DROPOUT_RATES[j],
            FIRST_LAYER_CHANNEL_COUNTS[k],
            metric_rank_matrix[i, j, k, names.index('column_avg_hr_dwmse')],
            metric_rank_matrix[i, j, k, names.index('near_sfc_hr_dwmse')],
            metric_rank_matrix[i, j, k, names.index('column_avg_hr_dwmse_mlc')],
            metric_rank_matrix[i, j, k, names.index('near_sfc_hr_dwmse_mlc')],
            metric_rank_matrix[i, j, k, names.index('column_avg_hr_bias')],
            metric_rank_matrix[i, j, k, names.index('near_sfc_hr_bias')],
            metric_rank_matrix[i, j, k, names.index('column_avg_hr_bias_mlc')],
            metric_rank_matrix[i, j, k, names.index('near_sfc_hr_bias_mlc')],
            metric_rank_matrix[i, j, k, names.index('all_flux_rmse')],
            metric_rank_matrix[i, j, k, names.index('net_flux_rmse')],
            metric_rank_matrix[i, j, k, names.index('all_flux_rmse_mlc')],
            metric_rank_matrix[i, j, k, names.index('net_flux_rmse_mlc')],
            metric_rank_matrix[i, j, k, names.index('net_flux_bias')],
            metric_rank_matrix[i, j, k, names.index('net_flux_bias_mlc')]
        ))


def _print_ranking_one_metric(metric_matrix, metric_name):
    """Prints ranking for one metric.

    U = number of UQ methods
    D = number of dropout rates
    S = number of spectral complexities

    :param metric_matrix: U-by-D-by-S numpy array of metric values.
    :param metric_name: Name of metric.
    """

    values_1d = numpy.ravel(metric_matrix)
    values_1d[numpy.isnan(values_1d)] = numpy.inf
    sort_indices_1d = numpy.argsort(values_1d)
    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, metric_matrix.shape
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:d}th-lowest {1:s} = {2:.4g} ... {3:s} ... '
            'dropout rate = {4:.2f} ... '
            'spectral complexity = {5:d}'
        ).format(
            m + 1, metric_name, metric_matrix[i, j, k],
            UQ_METHOD_STRINGS[i],
            DENSE_LAYER_DROPOUT_RATES[j],
            FIRST_LAYER_CHANNEL_COUNTS[k]
        ))


def _run(experiment_dir_name, isotonic_flag):
    """Plots each metric in hyperparam space for first shortwave UQ experiment.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param isotonic_flag: Same.
    """

    num_uq_methods = len(UQ_METHOD_STRINGS)
    num_dropout_rates = len(DENSE_LAYER_DROPOUT_RATES)
    num_channel_counts = len(FIRST_LAYER_CHANNEL_COUNTS)

    # TODO(thunderhoser): HACK.
    # num_metrics = len(METRIC_NAMES_ABBREV)
    num_metrics = 14

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
                    isotonic_flag=isotonic_flag
                )

                for m in range(num_metrics):
                    metric_matrix[i, j, k, m] = this_metric_dict[
                        METRIC_NAMES_ABBREV[m]
                    ]

    print(SEPARATOR_STRING)

    for m in range(num_metrics):
        this_metric_name = '{0:s} ({1:s})'.format(
            METRIC_NAMES_FANCY[m], METRIC_UNITS[m]
        )

        _print_ranking_one_metric(
            metric_matrix=metric_matrix[..., m], metric_name=this_metric_name
        )
        print(SEPARATOR_STRING)

    _print_ranking_all_metrics(
        metric_matrix=metric_matrix, main_metric_name='column_avg_hr_dwmse'
    )
    print(SEPARATOR_STRING)

    output_dir_name = '{0:s}/hyperparam_grids/shortwave{1:s}'.format(
        experiment_dir_name, '/isotonic_regression' if isotonic_flag else ''
    )
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    for m in range(num_metrics):
        panel_file_names = [''] * num_uq_methods

        for i in range(num_uq_methods):
            figure_object, axes_object = _plot_scores_2d(
                score_matrix=metric_matrix[i, ..., m],
                min_colour_value=(
                    None if METRIC_IS_BIAS_FLAGS[m]
                    else _finite_percentile(metric_matrix[..., m], 0)
                ),
                max_colour_value=(
                    _finite_percentile(
                        numpy.absolute(metric_matrix[..., m]), 95
                    )
                    if METRIC_IS_BIAS_FLAGS[m]
                    else _finite_percentile(metric_matrix[..., m], 95)
                ),
                x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
            )

            this_index = (
                numpy.nanargmin(
                    numpy.ravel(numpy.absolute(metric_matrix[..., m]))
                )
                if METRIC_IS_BIAS_FLAGS[m]
                else numpy.nanargmin(numpy.ravel(metric_matrix[..., m]))
            )
            best_indices = numpy.unravel_index(
                this_index, metric_matrix[..., m].shape
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
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            if SELECTED_MARKER_INDICES[0] == i:
                axes_object.plot(
                    SELECTED_MARKER_INDICES[2], SELECTED_MARKER_INDICES[1],
                    linestyle='None', marker=SELECTED_MARKER_TYPE,
                    markersize=marker_size_px, markeredgewidth=0,
                    markerfacecolor=WHITE_COLOUR,
                    markeredgecolor=WHITE_COLOUR
                )

            axes_object.set_xlabel(x_axis_label)
            axes_object.set_ylabel(y_axis_label)
            axes_object.set_title(UQ_METHOD_STRINGS_FANCY[i])

            panel_file_names[i] = '{0:s}/{1:s}_{2:s}.jpg'.format(
                output_dir_name,
                METRIC_NAMES_ABBREV[m].replace('_', '-'),
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
            output_dir_name, METRIC_NAMES_ABBREV[m]
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
        isotonic_flag=bool(getattr(INPUT_ARG_OBJECT, ISOTONIC_FLAG_ARG_NAME))
    )
