"""Plots each evaluation score vs. hyperparams for 2022 shortwave experiment."""

import os
import sys
import glob
import pickle
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

MODEL_DEPTHS = numpy.array([3, 4, 5], dtype=int)
CONV_LAYER_COUNTS = numpy.array([1, 2, 3, 4], dtype=int)
FIRST_LAYER_CHANNEL_COUNTS = numpy.array([4, 8, 16, 32, 64, 128], dtype=int)

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.3
MARKER_COLOUR = numpy.full(3, 1.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.2
SELECTED_MARKER_INDICES = numpy.array([2, 0, 0], dtype=int)

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
    '--' + ISOTONIC_FLAG_ARG_NAME, type=int, required=False, default=0,
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


def _read_scores_one_model(model_dir_name, isotonic_flag):
    """Reads scores for one model.

    :param model_dir_name: Name of directory with trained model and evaluation
        data.
    :param isotonic_flag: See documentation at top of file.
    :return: prmse_k_day01: Profile root mean squared error (RMSE):
    :return: dwmse_k3_day03: Dual-weighted mean squared error.
    :return: bottom_level_rmse_k_day01: RMSE for heating rate at bottom level.
    :return: down_flux_rmse_w_m02: RMSE for surface downwelling flux.
    :return: up_flux_rmse_w_m02: RMSE for TOA upwelling flux.
    :return: net_flux_rmse_w_m02: RMSE for net flux.
    """

    model_file_pattern = '{0:s}/model*.h5'.format(model_dir_name)
    model_file_names = glob.glob(model_file_pattern)

    if len(model_file_names) == 0:
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan

    model_file_names.sort()
    model_file_name = model_file_names[-1]

    evaluation_file_name = '{0:s}/{1:s}validation/evaluation.nc'.format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else ''
    )

    if not os.path.isfile(evaluation_file_name):
        return numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan

    print('Reading data from: "{0:s}"...'.format(evaluation_file_name))
    evaluation_table_xarray = evaluation.read_file(evaluation_file_name)
    prediction_file_name = (
        evaluation_table_xarray.attrs[evaluation.PREDICTION_FILE_KEY]
    )

    prmse_k_day01 = numpy.nanmean(
        evaluation_table_xarray[evaluation.VECTOR_PRMSE_KEY].values[0, :]
    )
    bottom_level_rmse_k_day01 = numpy.sqrt(numpy.nanmean(
        evaluation_table_xarray[evaluation.VECTOR_MSE_KEY].values[0, 0, :]
    ))

    scalar_target_names = evaluation_table_xarray.coords[
        evaluation.SCALAR_FIELD_DIM
    ].values.tolist()

    aux_target_names = evaluation_table_xarray.coords[
        evaluation.AUX_TARGET_FIELD_DIM
    ].values.tolist()

    j = scalar_target_names.index(
        example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
    )
    down_flux_rmse_w_m02 = numpy.nanmean(numpy.sqrt(
        evaluation_table_xarray[evaluation.SCALAR_MSE_KEY].values[j, :]
    ))

    j = scalar_target_names.index(example_utils.SHORTWAVE_TOA_UP_FLUX_NAME)
    up_flux_rmse_w_m02 = numpy.nanmean(numpy.sqrt(
        evaluation_table_xarray[evaluation.SCALAR_MSE_KEY].values[j, :]
    ))

    j = aux_target_names.index(evaluation.SHORTWAVE_NET_FLUX_NAME)
    net_flux_rmse_w_m02 = numpy.nanmean(numpy.sqrt(
        evaluation_table_xarray[evaluation.AUX_MSE_KEY].values[j, :]
    ))

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
    dwmse_k3_day03 = numpy.mean(
        weight_matrix * (vector_prediction_matrix - vector_target_matrix) ** 2
    )
    return (
        prmse_k_day01, dwmse_k3_day03, bottom_level_rmse_k_day01,
        down_flux_rmse_w_m02, up_flux_rmse_w_m02, net_flux_rmse_w_m02
    )


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.

    D = number of model depths
    C = number of conv-layer counts
    F = number of channel counts in first conv layer

    :param score_matrix: D-by-C-by-F numpy array of scores.
    :param score_name: Name of score.
    """

    scores_1d = numpy.ravel(score_matrix)
    scores_1d[numpy.isnan(scores_1d)] = numpy.inf
    sort_indices_1d = numpy.argsort(scores_1d)
    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, score_matrix.shape
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:d}th-lowest {1:s} = {2:.4g} ... model depth = {3:d} ... '
            'num conv layers per block = {4:d} ... '
            'num channels in first conv layer = {5:d}'
        ).format(
            m + 1, score_name, score_matrix[i, j, k], MODEL_DEPTHS[i],
            CONV_LAYER_COUNTS[j], FIRST_LAYER_CHANNEL_COUNTS[k]
        ))


def _print_ranking_all_scores(
        prmse_matrix_k_day01, dwmse_matrix_k3_day03,
        bottom_level_rmse_matrix_k_day01, down_flux_rmse_matrix_w_m02,
        up_flux_rmse_matrix_w_m02, net_flux_rmse_matrix_w_m02):
    """Prints ranking for one score.
    
    D = number of model depths
    C = number of conv-layer counts
    F = number of channel counts in first conv layer

    :param prmse_matrix_k_day01: D-by-C-by-F numpy array of profile RMSE.
    :param dwmse_matrix_k3_day03: D-by-C-by-F numpy array of dual-weighted MSE.
    :param bottom_level_rmse_matrix_k_day01: D-by-C-by-F numpy array of
        bottom-level RMSE.
    :param down_flux_rmse_matrix_w_m02: D-by-C-by-F numpy array of RMSE for
        surface downwelling flux.
    :param up_flux_rmse_matrix_w_m02: D-by-C-by-F numpy array of RMSE for TOA
        upwelling flux.
    :param net_flux_rmse_matrix_w_m02: D-by-C-by-F numpy array of RMSE for net
        flux.
    """

    these_scores = numpy.ravel(prmse_matrix_k_day01)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    sort_indices_1d = numpy.argsort(these_scores)
    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, prmse_matrix_k_day01.shape
    )

    these_scores = numpy.ravel(dwmse_matrix_k3_day03)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    dwmse_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'), dwmse_matrix_k3_day03.shape
    )

    these_scores = numpy.ravel(bottom_level_rmse_matrix_k_day01)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    bottom_rmse_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        bottom_level_rmse_matrix_k_day01.shape
    )

    these_scores = numpy.ravel(down_flux_rmse_matrix_w_m02)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    down_flux_rmse_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        down_flux_rmse_matrix_w_m02.shape
    )

    these_scores = numpy.ravel(up_flux_rmse_matrix_w_m02)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    up_flux_rmse_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        up_flux_rmse_matrix_w_m02.shape
    )

    these_scores = numpy.ravel(net_flux_rmse_matrix_w_m02)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    net_flux_rmse_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        net_flux_rmse_matrix_w_m02.shape
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:d}th-lowest PRMSE = {1:.4g} K day^-1 ... '
            'model depth = {2:d} ... num conv layers per block = {3:d} ... '
            'num channels in first conv layer = {4:d} ... '
            'DWMSE rank = {5:.1f} ... bottom-level-RMSE rank = {6:.1f} ... '
            'down-flux-RMSE rank = {7:.1f} ... up-flux-RMSE rank = {8:.1f} ... '
            'net-flux-RMSE rank = {9:.1f}'
        ).format(
            m + 1, prmse_matrix_k_day01[i, j, k],
            MODEL_DEPTHS[i], CONV_LAYER_COUNTS[j],
            FIRST_LAYER_CHANNEL_COUNTS[k],
            dwmse_rank_matrix[i, j, k],
            bottom_rmse_rank_matrix[i, j, k],
            down_flux_rmse_rank_matrix[i, j, k],
            up_flux_rmse_rank_matrix[i, j, k],
            net_flux_rmse_rank_matrix[i, j, k]
        ))


def _run(experiment_dir_name, isotonic_flag):
    """Plots each evaluation score vs. hyperparams for 2022 SW experiment.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param isotonic_flag: Same.
    """

    num_model_depths = len(MODEL_DEPTHS)
    num_conv_layer_counts = len(CONV_LAYER_COUNTS)
    num_channel_counts = len(FIRST_LAYER_CHANNEL_COUNTS)

    y_tick_labels = [
        '{0:d}'.format(c) for c in CONV_LAYER_COUNTS
    ]
    x_tick_labels = [
        '{0:d}'.format(c) for c in FIRST_LAYER_CHANNEL_COUNTS
    ]

    y_axis_label = 'Number of conv layers per block'
    x_axis_label = 'Number of channels in first conv layer'

    score_file_name = '{0:s}/{1:s}scores_on_hyperparam_grid.p'.format(
        experiment_dir_name,
        'isotonic_regression/' if isotonic_flag else ''
    )
    file_system_utils.mkdir_recursive_if_necessary(file_name=score_file_name)

    if os.path.isfile(score_file_name):
        print('Reading scores from: "{0:s}"...'.format(score_file_name))
        pickle_file_handle = open(score_file_name, 'rb')
        prmse_matrix_k_day01 = pickle.load(pickle_file_handle)
        dwmse_matrix_k3_day03 = pickle.load(pickle_file_handle)
        bottom_level_rmse_matrix_k_day01 = pickle.load(pickle_file_handle)
        down_flux_rmse_matrix_w_m02 = pickle.load(pickle_file_handle)
        up_flux_rmse_matrix_w_m02 = pickle.load(pickle_file_handle)
        net_flux_rmse_matrix_w_m02 = pickle.load(pickle_file_handle)
        pickle_file_handle.close()
    else:
        dimensions = (
            num_model_depths, num_conv_layer_counts, num_channel_counts
        )
        prmse_matrix_k_day01 = numpy.full(dimensions, numpy.nan)
        dwmse_matrix_k3_day03 = numpy.full(dimensions, numpy.nan)
        bottom_level_rmse_matrix_k_day01 = numpy.full(dimensions, numpy.nan)
        down_flux_rmse_matrix_w_m02 = numpy.full(dimensions, numpy.nan)
        up_flux_rmse_matrix_w_m02 = numpy.full(dimensions, numpy.nan)
        net_flux_rmse_matrix_w_m02 = numpy.full(dimensions, numpy.nan)

        for i in range(num_model_depths):
            for j in range(num_conv_layer_counts):
                for k in range(num_channel_counts):
                    this_model_dir_name = (
                        '{0:s}/depth={1:d}_num-conv-layers-per-block={2:d}_'
                        'num-first-layer-channels={3:03d}'
                    ).format(
                        experiment_dir_name, MODEL_DEPTHS[i],
                        CONV_LAYER_COUNTS[j], FIRST_LAYER_CHANNEL_COUNTS[k]
                    )

                    (
                        prmse_matrix_k_day01[i, j, k],
                        dwmse_matrix_k3_day03[i, j, k],
                        bottom_level_rmse_matrix_k_day01[i, j, k],
                        down_flux_rmse_matrix_w_m02[i, j, k],
                        up_flux_rmse_matrix_w_m02[i, j, k],
                        net_flux_rmse_matrix_w_m02[i, j, k]
                    ) = _read_scores_one_model(
                        model_dir_name=this_model_dir_name,
                        isotonic_flag=isotonic_flag
                    )

        print(SEPARATOR_STRING)

    print('Writing scores to: "{0:s}"...'.format(score_file_name))
    pickle_file_handle = open(score_file_name, 'wb')
    pickle.dump(prmse_matrix_k_day01, pickle_file_handle)
    pickle.dump(dwmse_matrix_k3_day03, pickle_file_handle)
    pickle.dump(bottom_level_rmse_matrix_k_day01, pickle_file_handle)
    pickle.dump(down_flux_rmse_matrix_w_m02, pickle_file_handle)
    pickle.dump(up_flux_rmse_matrix_w_m02, pickle_file_handle)
    pickle.dump(net_flux_rmse_matrix_w_m02, pickle_file_handle)
    pickle_file_handle.close()

    _print_ranking_one_score(
        score_matrix=prmse_matrix_k_day01, score_name='PRMSE (K day^-1)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=dwmse_matrix_k3_day03, score_name='DWMSE (K^3 day^-3)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=bottom_level_rmse_matrix_k_day01,
        score_name='bottom-level RMSE (K day^-1)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=down_flux_rmse_matrix_w_m02,
        score_name='RMSE for down flux (W m^-2)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=up_flux_rmse_matrix_w_m02,
        score_name='RMSE for up flux (W m^-2)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=net_flux_rmse_matrix_w_m02,
        score_name='RMSE for net flux (W m^-2)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_all_scores(
        prmse_matrix_k_day01=prmse_matrix_k_day01,
        dwmse_matrix_k3_day03=dwmse_matrix_k3_day03,
        bottom_level_rmse_matrix_k_day01=bottom_level_rmse_matrix_k_day01,
        down_flux_rmse_matrix_w_m02=down_flux_rmse_matrix_w_m02,
        up_flux_rmse_matrix_w_m02=up_flux_rmse_matrix_w_m02,
        net_flux_rmse_matrix_w_m02=net_flux_rmse_matrix_w_m02
    )
    print(SEPARATOR_STRING)

    this_index = numpy.argmin(numpy.ravel(prmse_matrix_k_day01))
    min_prmse_indices = numpy.unravel_index(
        this_index, prmse_matrix_k_day01.shape
    )

    this_index = numpy.argmin(numpy.ravel(dwmse_matrix_k3_day03))
    min_dwmse_indices = numpy.unravel_index(
        this_index, dwmse_matrix_k3_day03.shape
    )

    this_index = numpy.argmin(numpy.ravel(bottom_level_rmse_matrix_k_day01))
    min_bottom_rmse_indices = numpy.unravel_index(
        this_index, bottom_level_rmse_matrix_k_day01.shape
    )

    this_index = numpy.argmin(numpy.ravel(down_flux_rmse_matrix_w_m02))
    min_down_flux_rmse_indices = numpy.unravel_index(
        this_index, down_flux_rmse_matrix_w_m02.shape
    )

    this_index = numpy.argmin(numpy.ravel(up_flux_rmse_matrix_w_m02))
    min_up_flux_rmse_indices = numpy.unravel_index(
        this_index, up_flux_rmse_matrix_w_m02.shape
    )

    this_index = numpy.argmin(numpy.ravel(net_flux_rmse_matrix_w_m02))
    min_net_flux_rmse_indices = numpy.unravel_index(
        this_index, net_flux_rmse_matrix_w_m02.shape
    )

    for i in range(num_model_depths):
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=prmse_matrix_k_day01[i, ...],
            min_colour_value=numpy.nanpercentile(prmse_matrix_k_day01, 0),
            max_colour_value=numpy.nanpercentile(prmse_matrix_k_day01, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        if min_prmse_indices[0] == i:
            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / prmse_matrix_k_day01.shape[2]
            )

            axes_object.plot(
                min_prmse_indices[2], min_prmse_indices[1],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        if SELECTED_MARKER_INDICES[0] == i:
            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                SELECTED_MARKER_SIZE_GRID_CELLS / prmse_matrix_k_day01.shape[2]
            )

            axes_object.plot(
                SELECTED_MARKER_INDICES[2], SELECTED_MARKER_INDICES[1],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Model depth = {0:d}'.format(
            MODEL_DEPTHS[i]
        ))

        figure_file_name = (
            '{0:s}/{1:s}depth={2:d}_prmse_grid.jpg'
        ).format(
            experiment_dir_name,
            'isotonic_regression/' if isotonic_flag else '',
            MODEL_DEPTHS[i]
        )

        file_system_utils.mkdir_recursive_if_necessary(
            file_name=figure_file_name
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        figure_object, axes_object = _plot_scores_2d(
            score_matrix=dwmse_matrix_k3_day03[i, ...],
            min_colour_value=numpy.nanpercentile(dwmse_matrix_k3_day03, 0),
            max_colour_value=numpy.nanpercentile(dwmse_matrix_k3_day03, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        if min_dwmse_indices[0] == i:
            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / prmse_matrix_k_day01.shape[2]
            )
            axes_object.plot(
                min_dwmse_indices[2], min_dwmse_indices[1],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        if SELECTED_MARKER_INDICES[0] == i:
            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                SELECTED_MARKER_SIZE_GRID_CELLS / prmse_matrix_k_day01.shape[2]
            )

            axes_object.plot(
                SELECTED_MARKER_INDICES[2], SELECTED_MARKER_INDICES[1],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Model depth = {0:d}'.format(
            MODEL_DEPTHS[i]
        ))

        figure_file_name = (
            '{0:s}/{1:s}depth={2:d}_dwmse_grid.jpg'
        ).format(
            experiment_dir_name,
            'isotonic_regression/' if isotonic_flag else '',
            MODEL_DEPTHS[i]
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        figure_object, axes_object = _plot_scores_2d(
            score_matrix=bottom_level_rmse_matrix_k_day01[i, ...],
            min_colour_value=
            numpy.nanpercentile(bottom_level_rmse_matrix_k_day01, 0),
            max_colour_value=
            numpy.nanpercentile(bottom_level_rmse_matrix_k_day01, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        if min_bottom_rmse_indices[0] == i:
            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / prmse_matrix_k_day01.shape[2]
            )
            axes_object.plot(
                min_bottom_rmse_indices[2], min_bottom_rmse_indices[1],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        if SELECTED_MARKER_INDICES[0] == i:
            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                SELECTED_MARKER_SIZE_GRID_CELLS / prmse_matrix_k_day01.shape[2]
            )

            axes_object.plot(
                SELECTED_MARKER_INDICES[2], SELECTED_MARKER_INDICES[1],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Model depth = {0:d}'.format(
            MODEL_DEPTHS[i]
        ))

        figure_file_name = (
            '{0:s}/{1:s}depth={2:d}_bottom_level_rmse_grid.jpg'
        ).format(
            experiment_dir_name,
            'isotonic_regression/' if isotonic_flag else '',
            MODEL_DEPTHS[i]
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        figure_object, axes_object = _plot_scores_2d(
            score_matrix=down_flux_rmse_matrix_w_m02[i, ...],
            min_colour_value=
            numpy.nanpercentile(down_flux_rmse_matrix_w_m02, 0),
            max_colour_value=
            numpy.nanpercentile(down_flux_rmse_matrix_w_m02, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        if min_down_flux_rmse_indices[0] == i:
            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / prmse_matrix_k_day01.shape[2]
            )
            axes_object.plot(
                min_down_flux_rmse_indices[2], min_down_flux_rmse_indices[1],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        if SELECTED_MARKER_INDICES[0] == i:
            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                SELECTED_MARKER_SIZE_GRID_CELLS / prmse_matrix_k_day01.shape[2]
            )

            axes_object.plot(
                SELECTED_MARKER_INDICES[2], SELECTED_MARKER_INDICES[1],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Model depth = {0:d}'.format(
            MODEL_DEPTHS[i]
        ))

        figure_file_name = (
            '{0:s}/{1:s}depth={2:d}_down_flux_rmse_grid.jpg'
        ).format(
            experiment_dir_name,
            'isotonic_regression/' if isotonic_flag else '',
            MODEL_DEPTHS[i]
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        figure_object, axes_object = _plot_scores_2d(
            score_matrix=up_flux_rmse_matrix_w_m02[i, ...],
            min_colour_value=numpy.nanpercentile(up_flux_rmse_matrix_w_m02, 0),
            max_colour_value=numpy.nanpercentile(up_flux_rmse_matrix_w_m02, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        if min_up_flux_rmse_indices[0] == i:
            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / prmse_matrix_k_day01.shape[2]
            )
            axes_object.plot(
                min_up_flux_rmse_indices[2], min_up_flux_rmse_indices[1],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        if SELECTED_MARKER_INDICES[0] == i:
            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                SELECTED_MARKER_SIZE_GRID_CELLS / prmse_matrix_k_day01.shape[2]
            )

            axes_object.plot(
                SELECTED_MARKER_INDICES[2], SELECTED_MARKER_INDICES[1],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Model depth = {0:d}'.format(
            MODEL_DEPTHS[i]
        ))

        figure_file_name = (
            '{0:s}/{1:s}depth={2:d}_up_flux_rmse_grid.jpg'
        ).format(
            experiment_dir_name,
            'isotonic_regression/' if isotonic_flag else '',
            MODEL_DEPTHS[i]
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        figure_object, axes_object = _plot_scores_2d(
            score_matrix=net_flux_rmse_matrix_w_m02[i, ...],
            min_colour_value=numpy.nanpercentile(net_flux_rmse_matrix_w_m02, 0),
            max_colour_value=
            numpy.nanpercentile(net_flux_rmse_matrix_w_m02, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        if min_net_flux_rmse_indices[0] == i:
            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                BEST_MARKER_SIZE_GRID_CELLS / prmse_matrix_k_day01.shape[2]
            )
            axes_object.plot(
                min_net_flux_rmse_indices[2], min_net_flux_rmse_indices[1],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        if SELECTED_MARKER_INDICES[0] == i:
            figure_width_px = (
                figure_object.get_size_inches()[0] * figure_object.dpi
            )
            marker_size_px = figure_width_px * (
                SELECTED_MARKER_SIZE_GRID_CELLS / prmse_matrix_k_day01.shape[2]
            )

            axes_object.plot(
                SELECTED_MARKER_INDICES[2], SELECTED_MARKER_INDICES[1],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=MARKER_COLOUR,
                markeredgecolor=MARKER_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Model depth = {0:d}'.format(
            MODEL_DEPTHS[i]
        ))

        figure_file_name = (
            '{0:s}/{1:s}depth={2:d}_net_flux_rmse_grid.jpg'
        ).format(
            experiment_dir_name,
            'isotonic_regression/' if isotonic_flag else '',
            MODEL_DEPTHS[i]
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
