"""Plots each metric vs. hyperparams for 2022 SW experiment, one NN type."""

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
import example_utils
import prediction_io
import file_system_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_DEPTH_WIDTH_STRINGS = [
    '3, 1', '3, 2', '3, 3', '3, 4',
    '4, 1', '4, 2', '4, 3', '4, 4',
    '5, 1', '5, 2', '5, 3', '5, 4'
]

FIRST_LAYER_CHANNEL_COUNTS = numpy.array([4, 8, 16, 32, 64, 128], dtype=int)

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.3
MARKER_COLOUR = numpy.full(3, 1.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.2
SELECTED_MARKER_INDICES = numpy.array([2, 0], dtype=int)

FONT_SIZE = 30
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


def _plot_scores_2d(
        score_matrix, min_colour_value, max_colour_value, x_tick_labels,
        y_tick_labels, colour_map_object=pyplot.get_cmap('plasma')
):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param x_tick_labels: length-N list of tick labels.
    :param y_tick_labels: length-M list of tick labels.
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

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
        orientation_string='horizontal', extend_min=False, extend_max=False,
        padding=0.09, fraction_of_axis_length=0.8, font_size=FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _read_scores_one_model(model_dir_name, multilayer_cloud_flag,
                           isotonic_flag):
    """Reads scores for one model.

    :param model_dir_name: Name of directory with trained model and evaluation
        data.
    :param multilayer_cloud_flag: Boolean flag.  If True (False), will return
        scores for profiles with multi-layer cloud (all profiles).
    :param isotonic_flag: Boolean flag.  If True (False), will return scores for
        model with(out) isotonic regression.
    :return: dwmse_k3_day03: Dual-weighted mean squared error (DWMSE) for
        heating rate.
    :return: near_sfc_dwmse_k3_day03: DWMSE for near-surface heating rate.
    :return: flux_rmse_w_m02: All-flux root mean squared error (RMSE).
    :return: net_flux_rmse_w_m02: Net-flux RMSE.
    """

    model_file_pattern = '{0:s}/model*.h5'.format(model_dir_name)
    model_file_names = glob.glob(model_file_pattern)

    if len(model_file_names) == 0:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan

    model_file_names.sort()
    model_file_name = model_file_names[-1]

    evaluation_file_name = '{0:s}/{1:s}validation/{2:s}evaluation.nc'.format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else '',
        'by_cloud_regime/multi_layer_cloud/' if multilayer_cloud_flag else ''
    )

    if not os.path.isfile(evaluation_file_name):
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan

    print('Reading data from: "{0:s}"...'.format(evaluation_file_name))
    evaluation_table_xarray = evaluation.read_file(evaluation_file_name)
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

    flux_rmse_w_m02 = numpy.sqrt(
        (down_flux_mse_w_m02 + up_flux_mse_w_m02 + net_flux_mse_w_m02) / 3
    )
    net_flux_rmse_w_m02 = numpy.sqrt(net_flux_mse_w_m02)

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
        weight_matrix * (vector_prediction_matrix - vector_target_matrix) ** 2
    )

    dwmse_k3_day03 = numpy.mean(this_error_matrix)
    near_sfc_dwmse_k3_day03 = numpy.mean(this_error_matrix[:, 0, :])

    return (
        dwmse_k3_day03, near_sfc_dwmse_k3_day03,
        flux_rmse_w_m02, net_flux_rmse_w_m02
    )


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.

    D = number of depth/width combos
    F = number of channel counts in first conv layer

    :param score_matrix: D-by-F numpy array of scores.
    :param score_name: Name of score.
    """

    scores_1d = numpy.ravel(score_matrix)
    scores_1d[numpy.isnan(scores_1d)] = numpy.inf
    sort_indices_1d = numpy.argsort(scores_1d)
    i_sort_indices, j_sort_indices = numpy.unravel_index(
        sort_indices_1d, score_matrix.shape
    )

    for k in range(len(i_sort_indices)):
        i = i_sort_indices[k]
        j = j_sort_indices[k]

        print((
            '{0:d}th-lowest {1:s} = {2:.4g} ... NN depth = {3:s} ... '
            'NN width = {4:s} ... num channels in first conv layer = {5:d}'
        ).format(
            k + 1, score_name, score_matrix[i, j],
            MODEL_DEPTH_WIDTH_STRINGS[i].split(',')[0].strip(),
            MODEL_DEPTH_WIDTH_STRINGS[i].split(',')[1].strip(),
            FIRST_LAYER_CHANNEL_COUNTS[j]
        ))


def _print_ranking_all_scores(
        dwmse_matrix_k3_day03, near_sfc_dwmse_matrix_k3_day03,
        flux_rmse_matrix_w_m02, net_flux_rmse_matrix_w_m02,
        dwmse_matrix_mlc_k3_day03, near_sfc_dwmse_matrix_mlc_k3_day03,
        flux_rmse_matrix_mlc_w_m02, net_flux_rmse_matrix_mlc_w_m02):
    """Prints ranking for all scores.

    D = number of depth/width combos
    F = number of channel counts in first conv layer

    :param dwmse_matrix_k3_day03: D-by-F numpy array of dual-weighted mean
        squared error (DWMSE) for all profiles.
    :param near_sfc_dwmse_matrix_k3_day03: D-by-F numpy array of near-surface
        DWMSE for all profiles.
    :param flux_rmse_matrix_w_m02: D-by-F numpy array of all-flux RMSE for all
        profiles.
    :param net_flux_rmse_matrix_w_m02: D-by-F numpy array of net-flux RMSE for
        all profiles.
    :param dwmse_matrix_mlc_k3_day03: D-by-F numpy array of DWMSE for
        multi-layer cloud.
    :param near_sfc_dwmse_matrix_mlc_k3_day03: D-by-F numpy array of
        near-surface DWMSE for multi-layer cloud.
    :param flux_rmse_matrix_mlc_w_m02: D-by-F numpy array of all-flux RMSE for
        multi-layer cloud.
    :param net_flux_rmse_matrix_mlc_w_m02: D-by-F numpy array of net-flux RMSE
        for multi-layer cloud.
    """

    these_scores = numpy.ravel(dwmse_matrix_k3_day03)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    sort_indices_1d = numpy.argsort(these_scores)
    i_sort_indices, j_sort_indices = numpy.unravel_index(
        sort_indices_1d, dwmse_matrix_k3_day03.shape
    )

    these_scores = numpy.ravel(near_sfc_dwmse_matrix_k3_day03)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    near_sfc_dwmse_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        near_sfc_dwmse_matrix_k3_day03.shape
    )

    these_scores = numpy.ravel(flux_rmse_matrix_w_m02)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    flux_rmse_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        flux_rmse_matrix_w_m02.shape
    )

    these_scores = numpy.ravel(net_flux_rmse_matrix_w_m02)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    net_flux_rmse_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        net_flux_rmse_matrix_w_m02.shape
    )

    these_scores = numpy.ravel(dwmse_matrix_mlc_k3_day03)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    dwmse_mlc_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        dwmse_matrix_mlc_k3_day03.shape
    )

    these_scores = numpy.ravel(near_sfc_dwmse_matrix_mlc_k3_day03)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    near_sfc_dwmse_mlc_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        near_sfc_dwmse_matrix_mlc_k3_day03.shape
    )

    these_scores = numpy.ravel(flux_rmse_matrix_mlc_w_m02)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    flux_rmse_mlc_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        flux_rmse_matrix_mlc_w_m02.shape
    )

    these_scores = numpy.ravel(net_flux_rmse_matrix_mlc_w_m02)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    net_flux_rmse_mlc_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        net_flux_rmse_matrix_mlc_w_m02.shape
    )

    for k in range(len(i_sort_indices)):
        i = i_sort_indices[k]
        j = j_sort_indices[k]

        print((
            '{0:d}th-lowest DWMSE = {1:.4g} K^3 day^-3 ... '
            'NN depth = {2:s} ... '
            'NN width = {3:s} ... '
            'num channels in first conv layer = {4:d} ... '
            'near-surface DWMSE rank = {5:.1f} ... '
            'all-flux RMSE rank = {6:.1f} ... net-flux RMSE rank = {7:.1f} ... '
            'DWMSE rank for MLC = {8:.1f} ... '
            'near-surface DWMSE rank for MLC = {9:.1f} ... '
            'all-flux RMSE rank for MLC = {10:.1f} ... '
            'net-flux RMSE rank for MLC = {11:.1f}'
        ).format(
            k + 1, dwmse_matrix_k3_day03[i, j],
            MODEL_DEPTH_WIDTH_STRINGS[i].split(',')[0].strip(),
            MODEL_DEPTH_WIDTH_STRINGS[i].split(',')[1].strip(),
            FIRST_LAYER_CHANNEL_COUNTS[j],
            near_sfc_dwmse_rank_matrix[i, j],
            flux_rmse_rank_matrix[i, j], net_flux_rmse_rank_matrix[i, j],
            dwmse_mlc_rank_matrix[i, j],
            near_sfc_dwmse_mlc_rank_matrix[i, j],
            flux_rmse_mlc_rank_matrix[i, j],
            net_flux_rmse_mlc_rank_matrix[i, j]
        ))


def _run(experiment_dir_name, isotonic_flag):
    """Plots each metric vs. hyperparams for 2022 SW experiment, one NN type.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param isotonic_flag: Same.
    """

    num_depth_width_combos = len(MODEL_DEPTH_WIDTH_STRINGS)
    num_channel_counts = len(FIRST_LAYER_CHANNEL_COUNTS)

    y_tick_labels = [
        '{0:s}'.format(s) for s in MODEL_DEPTH_WIDTH_STRINGS
    ]
    x_tick_labels = [
        '{0:d}'.format(c) for c in FIRST_LAYER_CHANNEL_COUNTS
    ]

    y_axis_label = 'NN depth, width'
    x_axis_label = 'Spectral complexity'

    dimensions = (num_depth_width_combos, num_channel_counts)
    dwmse_matrix_k3_day03 = numpy.full(dimensions, numpy.nan)
    near_sfc_dwmse_matrix_k3_day03 = numpy.full(dimensions, numpy.nan)
    flux_rmse_matrix_w_m02 = numpy.full(dimensions, numpy.nan)
    net_flux_rmse_matrix_w_m02 = numpy.full(dimensions, numpy.nan)
    dwmse_matrix_mlc_k3_day03 = numpy.full(dimensions, numpy.nan)
    near_sfc_dwmse_matrix_mlc_k3_day03 = numpy.full(dimensions, numpy.nan)
    flux_rmse_matrix_mlc_w_m02 = numpy.full(dimensions, numpy.nan)
    net_flux_rmse_matrix_mlc_w_m02 = numpy.full(dimensions, numpy.nan)

    for i in range(num_depth_width_combos):
        for j in range(num_channel_counts):
            this_model_dir_name = (
                '{0:s}/depth={1:s}_num-conv-layers-per-block={2:s}_'
                'num-first-layer-channels={3:03d}'
            ).format(
                experiment_dir_name,
                MODEL_DEPTH_WIDTH_STRINGS[i].split(',')[0].strip(),
                MODEL_DEPTH_WIDTH_STRINGS[i].split(',')[1].strip(),
                FIRST_LAYER_CHANNEL_COUNTS[j]
            )

            (
                dwmse_matrix_k3_day03[i, j],
                near_sfc_dwmse_matrix_k3_day03[i, j],
                flux_rmse_matrix_w_m02[i, j],
                net_flux_rmse_matrix_w_m02[i, j]
            ) = _read_scores_one_model(
                model_dir_name=this_model_dir_name,
                multilayer_cloud_flag=False,
                isotonic_flag=isotonic_flag
            )

            (
                dwmse_matrix_mlc_k3_day03[i, j],
                near_sfc_dwmse_matrix_mlc_k3_day03[i, j],
                flux_rmse_matrix_mlc_w_m02[i, j],
                net_flux_rmse_matrix_mlc_w_m02[i, j]
            ) = _read_scores_one_model(
                model_dir_name=this_model_dir_name,
                multilayer_cloud_flag=True,
                isotonic_flag=isotonic_flag
            )

    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=dwmse_matrix_k3_day03, score_name='DWMSE (K^3 day^-3)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=near_sfc_dwmse_matrix_k3_day03,
        score_name='near-surface DWMSE (K^3 day^-3)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=flux_rmse_matrix_w_m02,
        score_name='all-flux RMSE (W m^-2)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=net_flux_rmse_matrix_w_m02,
        score_name='net-flux RMSE (W m^-2)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=dwmse_matrix_mlc_k3_day03,
        score_name='DWMSE for MLC (K^3 day^-3)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=near_sfc_dwmse_matrix_mlc_k3_day03,
        score_name='near-surface DWMSE for MLC (K^3 day^-3)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=flux_rmse_matrix_mlc_w_m02,
        score_name='all-flux RMSE for MLC (W m^-2)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=net_flux_rmse_matrix_mlc_w_m02,
        score_name='net-flux RMSE for MLC (W m^-2)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_all_scores(
        dwmse_matrix_k3_day03=dwmse_matrix_k3_day03,
        near_sfc_dwmse_matrix_k3_day03=near_sfc_dwmse_matrix_k3_day03,
        flux_rmse_matrix_w_m02=flux_rmse_matrix_w_m02,
        net_flux_rmse_matrix_w_m02=net_flux_rmse_matrix_w_m02,
        dwmse_matrix_mlc_k3_day03=dwmse_matrix_mlc_k3_day03,
        near_sfc_dwmse_matrix_mlc_k3_day03=near_sfc_dwmse_matrix_mlc_k3_day03,
        flux_rmse_matrix_mlc_w_m02=flux_rmse_matrix_mlc_w_m02,
        net_flux_rmse_matrix_mlc_w_m02=net_flux_rmse_matrix_mlc_w_m02
    )
    print(SEPARATOR_STRING)

    # Plot DWMSE for all profiles.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=dwmse_matrix_k3_day03,
        min_colour_value=numpy.nanpercentile(dwmse_matrix_k3_day03, 0),
        max_colour_value=numpy.nanpercentile(dwmse_matrix_k3_day03, 95),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )

    this_index = numpy.argmin(numpy.ravel(dwmse_matrix_k3_day03))
    best_indices = numpy.unravel_index(this_index, dwmse_matrix_k3_day03.shape)

    figure_width_px = (
        figure_object.get_size_inches()[0] * figure_object.dpi
    )
    marker_size_px = figure_width_px * (
        BEST_MARKER_SIZE_GRID_CELLS / dwmse_matrix_k3_day03.shape[1]
    )
    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )

    # TODO(thunderhoser): Need NN type in title.
    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    title_string = r'DWMSE$_{hr}$ for all profiles (K$^3$ day$^{-3}$)'
    axes_object.set_title(title_string)

    figure_file_name = '{0:s}/{1:s}dwmse.jpg'.format(
        experiment_dir_name,
        'isotonic_regression/' if isotonic_flag else ''
    )
    file_system_utils.mkdir_recursive_if_necessary(file_name=figure_file_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot near-surface DWMSE for all profiles.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=near_sfc_dwmse_matrix_k3_day03,
        min_colour_value=numpy.nanpercentile(near_sfc_dwmse_matrix_k3_day03, 0),
        max_colour_value=
        numpy.nanpercentile(near_sfc_dwmse_matrix_k3_day03, 95),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )

    this_index = numpy.argmin(numpy.ravel(near_sfc_dwmse_matrix_k3_day03))
    best_indices = numpy.unravel_index(
        this_index, near_sfc_dwmse_matrix_k3_day03.shape
    )

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    title_string = (
        r'Near-surface DWMSE$_{hr}$ for all profiles (K$^3$ day$^{-3}$)'
    )
    axes_object.set_title(title_string)

    figure_file_name = '{0:s}/{1:s}near_surface_dwmse.jpg'.format(
        experiment_dir_name,
        'isotonic_regression/' if isotonic_flag else ''
    )

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot all-flux RMSE for all profiles.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=flux_rmse_matrix_w_m02,
        min_colour_value=numpy.nanpercentile(flux_rmse_matrix_w_m02, 0),
        max_colour_value=numpy.nanpercentile(flux_rmse_matrix_w_m02, 95),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )

    this_index = numpy.argmin(numpy.ravel(flux_rmse_matrix_w_m02))
    best_indices = numpy.unravel_index(
        this_index, flux_rmse_matrix_w_m02.shape
    )

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    title_string = r'RMSE$_{flux}$ for all profiles (W m$^{-2}$)'
    axes_object.set_title(title_string)

    figure_file_name = '{0:s}/{1:s}flux_rmse.jpg'.format(
        experiment_dir_name,
        'isotonic_regression/' if isotonic_flag else ''
    )

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot net-flux RMSE for all profiles.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=net_flux_rmse_matrix_w_m02,
        min_colour_value=numpy.nanpercentile(net_flux_rmse_matrix_w_m02, 0),
        max_colour_value=numpy.nanpercentile(net_flux_rmse_matrix_w_m02, 95),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )

    this_index = numpy.argmin(numpy.ravel(net_flux_rmse_matrix_w_m02))
    best_indices = numpy.unravel_index(
        this_index, net_flux_rmse_matrix_w_m02.shape
    )

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    title_string = r'Net-flux RMSE for all profiles (W m$^{-2}$)'
    axes_object.set_title(title_string)

    figure_file_name = '{0:s}/{1:s}net_flux_rmse.jpg'.format(
        experiment_dir_name,
        'isotonic_regression/' if isotonic_flag else ''
    )

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot DWMSE for profiles with multi-layer cloud.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=dwmse_matrix_mlc_k3_day03,
        min_colour_value=numpy.nanpercentile(dwmse_matrix_mlc_k3_day03, 0),
        max_colour_value=numpy.nanpercentile(dwmse_matrix_mlc_k3_day03, 95),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )

    this_index = numpy.argmin(numpy.ravel(dwmse_matrix_mlc_k3_day03))
    best_indices = numpy.unravel_index(
        this_index, dwmse_matrix_mlc_k3_day03.shape
    )

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    title_string = r'DWMSE$_{hr}$ for multi-layer cloud (K$^3$ day$^{-3}$)'
    axes_object.set_title(title_string)

    figure_file_name = '{0:s}/{1:s}dwmse_mlc.jpg'.format(
        experiment_dir_name,
        'isotonic_regression/' if isotonic_flag else ''
    )

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot near-surface DWMSE for profiles with multi-layer cloud.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=near_sfc_dwmse_matrix_mlc_k3_day03,
        min_colour_value=
        numpy.nanpercentile(near_sfc_dwmse_matrix_mlc_k3_day03, 0),
        max_colour_value=
        numpy.nanpercentile(near_sfc_dwmse_matrix_mlc_k3_day03, 95),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )

    this_index = numpy.argmin(numpy.ravel(near_sfc_dwmse_matrix_mlc_k3_day03))
    best_indices = numpy.unravel_index(
        this_index, near_sfc_dwmse_matrix_mlc_k3_day03.shape
    )

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    title_string = (
        r'Near-surface DWMSE$_{hr}$ for multi-layer cloud (K$^3$ day$^{-3}$)'
    )
    axes_object.set_title(title_string)

    figure_file_name = '{0:s}/{1:s}near_surface_dwmse_mlc.jpg'.format(
        experiment_dir_name,
        'isotonic_regression/' if isotonic_flag else ''
    )

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot all-flux RMSE for profiles with multi-layer cloud.
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=flux_rmse_matrix_mlc_w_m02,
        min_colour_value=numpy.nanpercentile(flux_rmse_matrix_mlc_w_m02, 0),
        max_colour_value=numpy.nanpercentile(flux_rmse_matrix_mlc_w_m02, 95),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )

    this_index = numpy.argmin(numpy.ravel(flux_rmse_matrix_mlc_w_m02))
    best_indices = numpy.unravel_index(
        this_index, flux_rmse_matrix_mlc_w_m02.shape
    )

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    title_string = r'RMSE$_{flux}$ for multi-layer cloud (W m$^{-2}$)'
    axes_object.set_title(title_string)

    figure_file_name = '{0:s}/{1:s}flux_rmse_mlc.jpg'.format(
        experiment_dir_name,
        'isotonic_regression/' if isotonic_flag else ''
    )

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot net-flux RMSE for profiles with multi-layer cloud..
    figure_object, axes_object = _plot_scores_2d(
        score_matrix=net_flux_rmse_matrix_mlc_w_m02,
        min_colour_value=numpy.nanpercentile(net_flux_rmse_matrix_mlc_w_m02, 0),
        max_colour_value=
        numpy.nanpercentile(net_flux_rmse_matrix_mlc_w_m02, 95),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )

    this_index = numpy.argmin(numpy.ravel(net_flux_rmse_matrix_mlc_w_m02))
    best_indices = numpy.unravel_index(
        this_index, net_flux_rmse_matrix_mlc_w_m02.shape
    )

    axes_object.plot(
        best_indices[1], best_indices[0],
        linestyle='None', marker=BEST_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )
    axes_object.plot(
        SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
        linestyle='None', marker=SELECTED_MARKER_TYPE,
        markersize=marker_size_px, markeredgewidth=0,
        markerfacecolor=MARKER_COLOUR,
        markeredgecolor=MARKER_COLOUR
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    title_string = (
        r'Net-flux RMSE for profiles with multi-layer cloud (W m$^{-2}$)'
    )
    axes_object.set_title(title_string)

    figure_file_name = '{0:s}/{1:s}net_flux_rmse_mlc.jpg'.format(
        experiment_dir_name,
        'isotonic_regression/' if isotonic_flag else ''
    )

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        isotonic_flag=bool(getattr(INPUT_ARG_OBJECT, ISOTONIC_FLAG_ARG_NAME))
    )