"""Plots hyperparameter grids for Spectral Experiments 5LW (longwave).

One hyperparameter grid = one evaluation metric vs. all hyperparams.
"""

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
from tensorflow.keras import backend as K

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import gg_plotting_utils
import evaluation
import example_utils
import prediction_io
import neural_net
import file_system_utils
import imagemagick_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MIN_DUAL_WEIGHTS_AXIS1 = numpy.array([0.10, 0.25, 0.50, 0.75, 1.00])
BROADBAND_WEIGHTS_AXIS2 = numpy.array([0.01, 0.05, 0.10, 0.20])
NORMALIZATION_TYPE_STRINGS_AXIS3 = ['old', 'new']

BEST_MARKER_TYPE = '*'
BEST_MARKER_SIZE_GRID_CELLS = 0.175
WHITE_COLOUR = numpy.full(3, 1.)
BLACK_COLOUR = numpy.full(3, 0.)

SELECTED_MARKER_TYPE = 'o'
SELECTED_MARKER_SIZE_GRID_CELLS = 0.175
SELECTED_MARKER_INDICES = numpy.array([0, 0, 0], dtype=int)

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
WAVELENGTH_ARG_NAME = 'wavelength_metres'
ISOTONIC_FLAG_ARG_NAME = 'isotonic_flag'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXPERIMENT_DIR_HELP_STRING = 'Name of top-level directory with models.'
WAVELENGTH_HELP_STRING = (
    'Will plot results only for radiative transfer at this wavelength.'
)
ISOTONIC_FLAG_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot results with(out) isotonic regression.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WAVELENGTH_ARG_NAME, type=float, required=True,
    help=WAVELENGTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ISOTONIC_FLAG_ARG_NAME, type=int, required=True,
    help=ISOTONIC_FLAG_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
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


def _read_scores_one_model(
        model_dir_name, wavelength_metres, multilayer_cloud_flag,
        isotonic_flag):
    """Reads scores for one model.

    :param model_dir_name: Name of directory with trained model and evaluation
        data.
    :param wavelength_metres: Will read scores only for this wavelength.
    :param multilayer_cloud_flag: Boolean flag.  If True (False), will return
        scores for profiles with multi-layer cloud (all profiles).
    :param isotonic_flag: Boolean flag.  If True (False), will return scores for
        model with(out) isotonic regression.
    :return: dwmse_k3_day03: Dual-weighted mean squared error (DWMSE) for
        heating rate.
    :return: near_sfc_dwmse_k3_day03: DWMSE for near-surface heating rate.
    :return: bias_k_day01: Bias for heating rate.
    :return: near_sfc_bias_k_day01: Bias for near-surface heating rate.
    :return: flux_rmse_w_m02: All-flux root mean squared error (RMSE).
    :return: net_flux_rmse_w_m02: Net-flux RMSE.
    :return: net_flux_bias_w_m02: Net-flux bias.
    """

    evaluation_file_name = (
        '{0:s}/{1:s}validation/{2:s}percentiles_instead_of_abs_values/'
        'evaluation.nc'
    ).format(
        model_dir_name,
        'isotonic_regression/' if isotonic_flag else '',
        'by_cloud_regime/multi_layer_cloud/' if multilayer_cloud_flag else ''
    )

    if not os.path.isfile(evaluation_file_name):
        return (
            numpy.nan, numpy.nan,
            numpy.nan, numpy.nan,
            numpy.nan, numpy.nan, numpy.nan
        )

    print('Reading data from: "{0:s}"...'.format(evaluation_file_name))
    evaluation_table_xarray = evaluation.read_file(evaluation_file_name)
    prediction_file_name = (
        evaluation_table_xarray.attrs[evaluation.PREDICTION_FILE_KEY]
    )
    
    w = example_utils.match_wavelengths(
        wavelengths_metres=
        evaluation_table_xarray.coords[evaluation.WAVELENGTH_DIM].values,
        desired_wavelength_metres=wavelength_metres
    )

    scalar_target_names = evaluation_table_xarray.coords[
        evaluation.SCALAR_FIELD_DIM
    ].values.tolist()

    aux_target_names = evaluation_table_xarray.coords[
        evaluation.AUX_TARGET_FIELD_DIM
    ].values.tolist()

    j = scalar_target_names.index(
        example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
    )
    down_flux_mse_w_m02 = numpy.nanmean(
        evaluation_table_xarray[evaluation.SCALAR_MSE_KEY].values[w, j, :]
    )

    j = scalar_target_names.index(example_utils.LONGWAVE_TOA_UP_FLUX_NAME)
    up_flux_mse_w_m02 = numpy.nanmean(
        evaluation_table_xarray[evaluation.SCALAR_MSE_KEY].values[w, j, :]
    )

    j = aux_target_names.index(evaluation.LONGWAVE_NET_FLUX_NAME)
    net_flux_mse_w_m02 = numpy.nanmean(
        evaluation_table_xarray[evaluation.AUX_MSE_KEY].values[w, j, :]
    )
    net_flux_bias_w_m02 = numpy.nanmean(
        evaluation_table_xarray[evaluation.AUX_BIAS_KEY].values[w, j, :]
    )

    flux_rmse_w_m02 = numpy.sqrt(
        (down_flux_mse_w_m02 + up_flux_mse_w_m02 + net_flux_mse_w_m02) / 3
    )
    net_flux_rmse_w_m02 = numpy.sqrt(net_flux_mse_w_m02)

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    w = example_utils.match_wavelengths(
        wavelengths_metres=
        prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY],
        desired_wavelength_metres=wavelength_metres
    )
    
    vector_target_matrix = (
        prediction_dict[prediction_io.VECTOR_TARGETS_KEY][:, :, w, ...]
    )
    vector_prediction_matrix = numpy.mean(
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY][:, :, w, ...],
        axis=-1
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

    bias_k_day01 = numpy.mean(
        vector_prediction_matrix - vector_target_matrix
    )
    near_sfc_bias_k_day01 = numpy.mean(
        vector_prediction_matrix[:, 0, :] - vector_target_matrix[:, 0, :]
    )

    return (
        dwmse_k3_day03, near_sfc_dwmse_k3_day03,
        bias_k_day01, near_sfc_bias_k_day01,
        flux_rmse_w_m02, net_flux_rmse_w_m02, net_flux_bias_w_m02
    )


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.
    
    D = number of minimum dual weights
    B = number of broadband weights
    N = number of normalization types

    :param score_matrix: D-by-B-by-N numpy array of scores.
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
            '{0:d}th-lowest {1:s} = {2:.4g} ... '
            'min dual weight = {3:.2f} ... '
            'broadband weight = {4:.2f} ... '
            'normalization type = {5:s}'
        ).format(
            m + 1, score_name, score_matrix[i, j, k],
            MIN_DUAL_WEIGHTS_AXIS1[i],
            BROADBAND_WEIGHTS_AXIS2[j],
            NORMALIZATION_TYPE_STRINGS_AXIS3[k].upper()
        ))


def _print_ranking_all_scores(
        dwmse_matrix_k3_day03, near_sfc_dwmse_matrix_k3_day03,
        bias_matrix_k_day01, near_sfc_bias_matrix_k_day01,
        flux_rmse_matrix_w_m02, net_flux_rmse_matrix_w_m02,
        net_flux_bias_matrix_w_m02,
        dwmse_matrix_mlc_k3_day03, near_sfc_dwmse_matrix_mlc_k3_day03,
        bias_matrix_mlc_k_day01, near_sfc_bias_matrix_mlc_k_day01,
        flux_rmse_matrix_mlc_w_m02, net_flux_rmse_matrix_mlc_w_m02,
        net_flux_bias_matrix_mlc_w_m02):
    """Prints ranking for all scores.

    D = number of minimum dual weights
    B = number of broadband weights
    N = number of normalization types

    :param dwmse_matrix_k3_day03: D-by-B-by-N numpy array of dual-weighted mean
        squared error (DWMSE) for all profiles.
    :param near_sfc_dwmse_matrix_k3_day03: D-by-B-by-N numpy array of
        near-surface DWMSE for all profiles.
    :param bias_matrix_k_day01: D-by-B-by-N numpy array of bias for all profiles.
    :param near_sfc_bias_matrix_k_day01: D-by-B-by-N numpy array of
        near-surface bias for all profiles.
    :param flux_rmse_matrix_w_m02: D-by-B-by-N numpy array of all-flux RMSE for
        all profiles.
    :param net_flux_rmse_matrix_w_m02: D-by-B-by-N numpy array of net-flux RMSE
        for all profiles.
    :param net_flux_bias_matrix_w_m02: D-by-B-by-N numpy array of
        net-flux bias for all profiles.
    :param dwmse_matrix_mlc_k3_day03: D-by-B-by-N numpy array of DWMSE for
        multi-layer cloud.
    :param near_sfc_dwmse_matrix_mlc_k3_day03: D-by-B-by-N numpy array of
        near-surface DWMSE for multi-layer cloud.
    :param bias_matrix_mlc_k_day01: D-by-B-by-N numpy array of bias for
        multi-layer cloud.
    :param near_sfc_bias_matrix_mlc_k_day01: D-by-B-by-N numpy array of
        near-surface bias for multi-layer cloud.
    :param flux_rmse_matrix_mlc_w_m02: D-by-B-by-N numpy array of all-flux RMSE
        for multi-layer cloud.
    :param net_flux_rmse_matrix_mlc_w_m02: D-by-B-by-N numpy array of net-flux
        RMSE for multi-layer cloud.
    :param net_flux_bias_matrix_mlc_w_m02: D-by-B-by-N numpy array of
        net-flux bias for multi-layer cloud.
    """

    # these_scores = numpy.ravel(dwmse_matrix_k3_day03)
    # these_scores[numpy.isnan(these_scores)] = numpy.inf
    # sort_indices_1d = numpy.argsort(these_scores)
    # i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
    #     sort_indices_1d, dwmse_matrix_k3_day03.shape
    # )

    these_scores = numpy.ravel(dwmse_matrix_k3_day03)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    dwmse_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        dwmse_matrix_k3_day03.shape
    )

    these_scores = numpy.ravel(near_sfc_dwmse_matrix_k3_day03)
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    near_sfc_dwmse_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        near_sfc_dwmse_matrix_k3_day03.shape
    )

    these_scores = numpy.ravel(numpy.absolute(bias_matrix_k_day01))
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    bias_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        bias_matrix_k_day01.shape
    )

    these_scores = numpy.ravel(numpy.absolute(near_sfc_bias_matrix_k_day01))
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    near_sfc_bias_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        near_sfc_bias_matrix_k_day01.shape
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

    these_scores = numpy.ravel(numpy.absolute(net_flux_bias_matrix_w_m02))
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    net_flux_bias_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        net_flux_bias_matrix_w_m02.shape
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

    these_scores = numpy.ravel(numpy.absolute(bias_matrix_mlc_k_day01))
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    bias_mlc_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        bias_matrix_mlc_k_day01.shape
    )

    these_scores = numpy.ravel(numpy.absolute(near_sfc_bias_matrix_mlc_k_day01))
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    near_sfc_bias_mlc_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        near_sfc_bias_matrix_mlc_k_day01.shape
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

    these_scores = numpy.ravel(numpy.absolute(net_flux_bias_matrix_mlc_w_m02))
    these_scores[numpy.isnan(these_scores)] = numpy.inf
    net_flux_bias_mlc_rank_matrix = numpy.reshape(
        rankdata(these_scores, method='average'),
        net_flux_bias_matrix_mlc_w_m02.shape
    )

    overall_rank_matrix = numpy.mean(
        numpy.stack([
            dwmse_rank_matrix, near_sfc_dwmse_rank_matrix,
            dwmse_mlc_rank_matrix, near_sfc_dwmse_mlc_rank_matrix,
            bias_rank_matrix, near_sfc_bias_rank_matrix,
            bias_mlc_rank_matrix, near_sfc_bias_mlc_rank_matrix,
            flux_rmse_rank_matrix, net_flux_rmse_rank_matrix,
            flux_rmse_mlc_rank_matrix, net_flux_rmse_mlc_rank_matrix,
            net_flux_bias_rank_matrix, net_flux_bias_mlc_rank_matrix
        ], axis=-1),
        axis=-1
    )

    sort_indices_1d = numpy.argsort(numpy.ravel(overall_rank_matrix))
    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        sort_indices_1d, overall_rank_matrix.shape
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            'Min dual weight = {0:.2f} ... '
            'broadband weight = {1:.2f} ... '
            'normalization type = {2:s} ... '
            'DWMSE ranks (all, sfc, MLC, MLC sfc) = '
            '{3:.1f}, {4:.1f}, {5:.1f}, {6:.1f} ... '
            'HR-bias ranks (all, sfc, MLC, MLC sfc) = '
            '{7:.1f}, {8:.1f}, {9:.1f}, {10:.1f} ... '
            'flux-RMSE ranks (all, net, MLC, MLC net) = '
            '{11:.1f}, {12:.1f}, {13:.1f}, {14:.1f} ... '
            'net-flux-bias ranks (all, MLC) = '
            '{15:.1f}, {16:.1f}'
        ).format(
            MIN_DUAL_WEIGHTS_AXIS1[i],
            BROADBAND_WEIGHTS_AXIS2[j],
            NORMALIZATION_TYPE_STRINGS_AXIS3[k].upper(),
            dwmse_rank_matrix[i, j, k], near_sfc_dwmse_rank_matrix[i, j, k],
            dwmse_mlc_rank_matrix[i, j, k], near_sfc_dwmse_mlc_rank_matrix[i, j, k],
            bias_rank_matrix[i, j, k], near_sfc_bias_rank_matrix[i, j, k],
            bias_mlc_rank_matrix[i, j, k], near_sfc_bias_mlc_rank_matrix[i, j, k],
            flux_rmse_rank_matrix[i, j, k], net_flux_rmse_rank_matrix[i, j, k],
            flux_rmse_mlc_rank_matrix[i, j, k], net_flux_rmse_mlc_rank_matrix[i, j, k],
            net_flux_bias_rank_matrix[i, j, k], net_flux_bias_mlc_rank_matrix[i, j, k]
        ))


def _run(experiment_dir_name, wavelength_metres, isotonic_flag,
         output_dir_name):
    """Plots hyperparameter grids for Spectral Experiments 5LW (longwave).

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of this script.
    :param wavelength_metres: Same.
    :param isotonic_flag: Same.
    :param output_dir_name: Same.
    """
    
    num_min_dual_weights = len(MIN_DUAL_WEIGHTS_AXIS1)
    num_broadband_weights = len(BROADBAND_WEIGHTS_AXIS2)
    num_normalization_types = len(NORMALIZATION_TYPE_STRINGS_AXIS3)

    y_tick_labels = [
        '{0:.2f}'.format(d) for d in MIN_DUAL_WEIGHTS_AXIS1
    ]
    x_tick_labels = [
        '{0:.2f}'.format(b) for b in BROADBAND_WEIGHTS_AXIS2
    ]

    y_axis_label = 'Minimum dual weight'
    x_axis_label = 'Broadband weight'

    dimensions = (
        num_min_dual_weights, num_broadband_weights, num_normalization_types
    )

    dwmse_matrix_k3_day03 = numpy.full(dimensions, numpy.nan)
    near_sfc_dwmse_matrix_k3_day03 = numpy.full(dimensions, numpy.nan)
    bias_matrix_k_day01 = numpy.full(dimensions, numpy.nan)
    near_sfc_bias_matrix_k_day01 = numpy.full(dimensions, numpy.nan)
    flux_rmse_matrix_w_m02 = numpy.full(dimensions, numpy.nan)
    net_flux_rmse_matrix_w_m02 = numpy.full(dimensions, numpy.nan)
    net_flux_bias_matrix_w_m02 = numpy.full(dimensions, numpy.nan)

    dwmse_matrix_mlc_k3_day03 = numpy.full(dimensions, numpy.nan)
    near_sfc_dwmse_matrix_mlc_k3_day03 = numpy.full(dimensions, numpy.nan)
    bias_matrix_mlc_k_day01 = numpy.full(dimensions, numpy.nan)
    near_sfc_bias_matrix_mlc_k_day01 = numpy.full(dimensions, numpy.nan)
    flux_rmse_matrix_mlc_w_m02 = numpy.full(dimensions, numpy.nan)
    net_flux_rmse_matrix_mlc_w_m02 = numpy.full(dimensions, numpy.nan)
    net_flux_bias_matrix_mlc_w_m02 = numpy.full(dimensions, numpy.nan)

    for i in range(num_min_dual_weights):
        for j in range(num_broadband_weights):
            for k in range(num_normalization_types):
                this_model_dir_name = (
                    '{0:s}/spectral_experiment05lw_more_losses/'
                    'min-dual-weight={1:.2f}_broadband-weight={2:.2f}_'
                    'normalization-type={3:s}'
                ).format(
                    experiment_dir_name,
                    MIN_DUAL_WEIGHTS_AXIS1[i],
                    BROADBAND_WEIGHTS_AXIS2[j],
                    NORMALIZATION_TYPE_STRINGS_AXIS3[k]
                )

                (
                    dwmse_matrix_k3_day03[i, j, k],
                    near_sfc_dwmse_matrix_k3_day03[i, j, k],
                    bias_matrix_k_day01[i, j, k],
                    near_sfc_bias_matrix_k_day01[i, j, k],
                    flux_rmse_matrix_w_m02[i, j, k],
                    net_flux_rmse_matrix_w_m02[i, j, k],
                    net_flux_bias_matrix_w_m02[i, j, k]
                ) = _read_scores_one_model(
                    model_dir_name=this_model_dir_name,
                    wavelength_metres=wavelength_metres,
                    multilayer_cloud_flag=False,
                    isotonic_flag=isotonic_flag
                )

                (
                    dwmse_matrix_mlc_k3_day03[i, j, k],
                    near_sfc_dwmse_matrix_mlc_k3_day03[i, j, k],
                    bias_matrix_mlc_k_day01[i, j, k],
                    near_sfc_bias_matrix_mlc_k_day01[i, j, k],
                    flux_rmse_matrix_mlc_w_m02[i, j, k],
                    net_flux_rmse_matrix_mlc_w_m02[i, j, k],
                    net_flux_bias_matrix_mlc_w_m02[i, j, k]
                ) = _read_scores_one_model(
                    model_dir_name=this_model_dir_name,
                    wavelength_metres=wavelength_metres,
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
        score_matrix=numpy.absolute(bias_matrix_k_day01),
        score_name='absolute HR bias (K day^-1)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=numpy.absolute(near_sfc_bias_matrix_k_day01),
        score_name='near-surface absolute HR bias (K day^-1)'
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
        score_matrix=numpy.absolute(net_flux_bias_matrix_w_m02),
        score_name='absolute net-flux bias (W m^-2)'
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
        score_matrix=numpy.absolute(bias_matrix_mlc_k_day01),
        score_name='absolute HR bias for MLC (K day^-1)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=numpy.absolute(near_sfc_bias_matrix_mlc_k_day01),
        score_name='near-surface absolute HR bias for MLC (K day^-1)'
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

    _print_ranking_one_score(
        score_matrix=numpy.absolute(net_flux_bias_matrix_mlc_w_m02),
        score_name='absolute net-flux bias for MLC (W m^-2)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_all_scores(
        dwmse_matrix_k3_day03=dwmse_matrix_k3_day03,
        near_sfc_dwmse_matrix_k3_day03=near_sfc_dwmse_matrix_k3_day03,
        bias_matrix_k_day01=bias_matrix_k_day01,
        near_sfc_bias_matrix_k_day01=near_sfc_bias_matrix_k_day01,
        flux_rmse_matrix_w_m02=flux_rmse_matrix_w_m02,
        net_flux_rmse_matrix_w_m02=net_flux_rmse_matrix_w_m02,
        net_flux_bias_matrix_w_m02=net_flux_bias_matrix_w_m02,
        dwmse_matrix_mlc_k3_day03=dwmse_matrix_mlc_k3_day03,
        near_sfc_dwmse_matrix_mlc_k3_day03=near_sfc_dwmse_matrix_mlc_k3_day03,
        bias_matrix_mlc_k_day01=bias_matrix_mlc_k_day01,
        near_sfc_bias_matrix_mlc_k_day01=near_sfc_bias_matrix_mlc_k_day01,
        flux_rmse_matrix_mlc_w_m02=flux_rmse_matrix_mlc_w_m02,
        net_flux_rmse_matrix_mlc_w_m02=net_flux_rmse_matrix_mlc_w_m02,
        net_flux_bias_matrix_mlc_w_m02=net_flux_bias_matrix_mlc_w_m02
    )
    print(SEPARATOR_STRING)

    dwmse_panel_file_names = [''] * num_normalization_types
    near_sfc_dwmse_panel_file_names = [''] * num_normalization_types
    bias_panel_file_names = [''] * num_normalization_types
    near_sfc_bias_panel_file_names = [''] * num_normalization_types
    flux_rmse_panel_file_names = [''] * num_normalization_types
    net_flux_rmse_panel_file_names = [''] * num_normalization_types
    net_flux_bias_panel_file_names = [''] * num_normalization_types

    dwmse_mlc_panel_file_names = [''] * num_normalization_types
    near_sfc_dwmse_mlc_panel_file_names = [''] * num_normalization_types
    bias_mlc_panel_file_names = [''] * num_normalization_types
    near_sfc_bias_mlc_panel_file_names = [''] * num_normalization_types
    flux_rmse_mlc_panel_file_names = [''] * num_normalization_types
    net_flux_rmse_mlc_panel_file_names = [''] * num_normalization_types
    net_flux_bias_mlc_panel_file_names = [''] * num_normalization_types

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    for k in range(num_normalization_types):

        # Plot DWMSE for all profiles.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=dwmse_matrix_k3_day03[..., k],
            min_colour_value=_finite_percentile(dwmse_matrix_k3_day03, 0),
            max_colour_value=_finite_percentile(dwmse_matrix_k3_day03, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        this_index = numpy.nanargmin(numpy.ravel(dwmse_matrix_k3_day03))
        best_indices = numpy.unravel_index(
            this_index, dwmse_matrix_k3_day03.shape
        )

        figure_width_px = (
            figure_object.get_size_inches()[0] * figure_object.dpi
        )
        marker_size_px = figure_width_px * (
            BEST_MARKER_SIZE_GRID_CELLS / dwmse_matrix_k3_day03.shape[2]
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        if SELECTED_MARKER_INDICES[2] == k:
            axes_object.plot(
                SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Normalization type = {0:s}'.format(
            NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        ))

        dwmse_panel_file_names[k] = '{0:s}/dwmse_norm-type={1:s}.jpg'.format(
            output_dir_name, NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        )

        print('Saving figure to: "{0:s}"...'.format(dwmse_panel_file_names[k]))
        figure_object.savefig(
            dwmse_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot near-surface DWMSE for all profiles.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=near_sfc_dwmse_matrix_k3_day03[..., k],
            min_colour_value=_finite_percentile(near_sfc_dwmse_matrix_k3_day03, 0),
            max_colour_value=_finite_percentile(near_sfc_dwmse_matrix_k3_day03, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        this_index = numpy.nanargmin(numpy.ravel(near_sfc_dwmse_matrix_k3_day03))
        best_indices = numpy.unravel_index(
            this_index, near_sfc_dwmse_matrix_k3_day03.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        if SELECTED_MARKER_INDICES[2] == k:
            axes_object.plot(
                SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Normalization type = {0:s}'.format(
            NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        ))

        near_sfc_dwmse_panel_file_names[k] = (
            '{0:s}/near_surface_dwmse_{1:s}.jpg'
        ).format(
            output_dir_name, NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        )

        print('Saving figure to: "{0:s}"...'.format(
            near_sfc_dwmse_panel_file_names[k]
        ))
        figure_object.savefig(
            near_sfc_dwmse_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot HR bias for all profiles.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=bias_matrix_k_day01[..., k],
            min_colour_value=None,
            max_colour_value=_finite_percentile(
                numpy.absolute(bias_matrix_k_day01), 95
            ),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        this_index = numpy.nanargmin(numpy.ravel(
            numpy.absolute(bias_matrix_k_day01)
        ))
        best_indices = numpy.unravel_index(
            this_index, bias_matrix_k_day01.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=BLACK_COLOUR,
                markeredgecolor=BLACK_COLOUR
            )

        if SELECTED_MARKER_INDICES[2] == k:
            axes_object.plot(
                SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=BLACK_COLOUR,
                markeredgecolor=BLACK_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Normalization type = {0:s}'.format(
            NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        ))

        bias_panel_file_names[k] = '{0:s}/bias_{1:s}.jpg'.format(
            output_dir_name, NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        )

        print('Saving figure to: "{0:s}"...'.format(bias_panel_file_names[k]))
        figure_object.savefig(
            bias_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot near-surface bias for all profiles.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=near_sfc_bias_matrix_k_day01[..., k],
            min_colour_value=None,
            max_colour_value=_finite_percentile(
                numpy.absolute(near_sfc_bias_matrix_k_day01), 95
            ),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        this_index = numpy.nanargmin(numpy.ravel(
            numpy.absolute(near_sfc_bias_matrix_k_day01)
        ))
        best_indices = numpy.unravel_index(
            this_index, near_sfc_bias_matrix_k_day01.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=BLACK_COLOUR,
                markeredgecolor=BLACK_COLOUR
            )

        if SELECTED_MARKER_INDICES[2] == k:
            axes_object.plot(
                SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=BLACK_COLOUR,
                markeredgecolor=BLACK_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Normalization type = {0:s}'.format(
            NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        ))

        near_sfc_bias_panel_file_names[k] = (
            '{0:s}/near_surface_bias_{1:s}.jpg'
        ).format(
            output_dir_name, NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        )

        print('Saving figure to: "{0:s}"...'.format(
            near_sfc_bias_panel_file_names[k]
        ))
        figure_object.savefig(
            near_sfc_bias_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot all-flux RMSE for all profiles.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=flux_rmse_matrix_w_m02[..., k],
            min_colour_value=_finite_percentile(flux_rmse_matrix_w_m02, 0),
            max_colour_value=_finite_percentile(flux_rmse_matrix_w_m02, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        this_index = numpy.nanargmin(numpy.ravel(flux_rmse_matrix_w_m02))
        best_indices = numpy.unravel_index(
            this_index, flux_rmse_matrix_w_m02.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        if SELECTED_MARKER_INDICES[2] == k:
            axes_object.plot(
                SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Normalization type = {0:s}'.format(
            NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        ))

        flux_rmse_panel_file_names[k] = '{0:s}/flux_rmse_{1:s}.jpg'.format(
            output_dir_name, NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        )

        print('Saving figure to: "{0:s}"...'.format(
            flux_rmse_panel_file_names[k]
        ))
        figure_object.savefig(
            flux_rmse_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot net-flux RMSE for all profiles.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=net_flux_rmse_matrix_w_m02[..., k],
            min_colour_value=_finite_percentile(net_flux_rmse_matrix_w_m02, 0),
            max_colour_value=_finite_percentile(net_flux_rmse_matrix_w_m02, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        this_index = numpy.nanargmin(numpy.ravel(net_flux_rmse_matrix_w_m02))
        best_indices = numpy.unravel_index(
            this_index, net_flux_rmse_matrix_w_m02.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        if SELECTED_MARKER_INDICES[2] == k:
            axes_object.plot(
                SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Normalization type = {0:s}'.format(
            NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        ))

        net_flux_rmse_panel_file_names[k] = (
            '{0:s}/net_flux_rmse_{1:s}.jpg'
        ).format(
            output_dir_name, NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        )

        print('Saving figure to: "{0:s}"...'.format(
            net_flux_rmse_panel_file_names[k]
        ))
        figure_object.savefig(
            net_flux_rmse_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot net-flux bias for all profiles.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=net_flux_bias_matrix_w_m02[..., k],
            min_colour_value=None,
            max_colour_value=_finite_percentile(
                numpy.absolute(net_flux_bias_matrix_w_m02), 95
            ),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        this_index = numpy.nanargmin(numpy.ravel(
            numpy.absolute(net_flux_bias_matrix_w_m02)
        ))
        best_indices = numpy.unravel_index(
            this_index, net_flux_bias_matrix_w_m02.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=BLACK_COLOUR,
                markeredgecolor=BLACK_COLOUR
            )

        if SELECTED_MARKER_INDICES[2] == k:
            axes_object.plot(
                SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=BLACK_COLOUR,
                markeredgecolor=BLACK_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Normalization type = {0:s}'.format(
            NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        ))

        net_flux_bias_panel_file_names[k] = (
            '{0:s}/net_flux_bias_{1:s}.jpg'
        ).format(
            output_dir_name, NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        )

        print('Saving figure to: "{0:s}"...'.format(
            net_flux_bias_panel_file_names[k]
        ))
        figure_object.savefig(
            net_flux_bias_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot DWMSE for profiles with multi-layer cloud.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=dwmse_matrix_mlc_k3_day03[..., k],
            min_colour_value=_finite_percentile(dwmse_matrix_mlc_k3_day03, 0),
            max_colour_value=_finite_percentile(dwmse_matrix_mlc_k3_day03, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        this_index = numpy.nanargmin(numpy.ravel(dwmse_matrix_mlc_k3_day03))
        best_indices = numpy.unravel_index(
            this_index, dwmse_matrix_mlc_k3_day03.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        if SELECTED_MARKER_INDICES[2] == k:
            axes_object.plot(
                SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Normalization type = {0:s}'.format(
            NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        ))

        dwmse_mlc_panel_file_names[k] = '{0:s}/dwmse_mlc_{1:s}.jpg'.format(
            output_dir_name, NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        )

        print('Saving figure to: "{0:s}"...'.format(
            dwmse_mlc_panel_file_names[k]
        ))
        figure_object.savefig(
            dwmse_mlc_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot near-surface DWMSE for profiles with multi-layer cloud.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=near_sfc_dwmse_matrix_mlc_k3_day03[..., k],
            min_colour_value=_finite_percentile(near_sfc_dwmse_matrix_mlc_k3_day03, 0),
            max_colour_value=_finite_percentile(near_sfc_dwmse_matrix_mlc_k3_day03, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        this_index = numpy.nanargmin(numpy.ravel(near_sfc_dwmse_matrix_mlc_k3_day03))
        best_indices = numpy.unravel_index(
            this_index, near_sfc_dwmse_matrix_mlc_k3_day03.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        if SELECTED_MARKER_INDICES[2] == k:
            axes_object.plot(
                SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Normalization type = {0:s}'.format(
            NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        ))

        near_sfc_dwmse_mlc_panel_file_names[k] = (
            '{0:s}/near_surface_dwmse_mlc_{1:s}.jpg'
        ).format(
            output_dir_name, NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        )

        print('Saving figure to: "{0:s}"...'.format(
            near_sfc_dwmse_mlc_panel_file_names[k]
        ))
        figure_object.savefig(
            near_sfc_dwmse_mlc_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot HR bias for profiles with multi-layer cloud.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=bias_matrix_mlc_k_day01[..., k],
            min_colour_value=None,
            max_colour_value=_finite_percentile(
                numpy.absolute(bias_matrix_mlc_k_day01), 95
            ),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        this_index = numpy.nanargmin(numpy.ravel(
            numpy.absolute(bias_matrix_mlc_k_day01)
        ))
        best_indices = numpy.unravel_index(
            this_index, bias_matrix_mlc_k_day01.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=BLACK_COLOUR,
                markeredgecolor=BLACK_COLOUR
            )

        if SELECTED_MARKER_INDICES[2] == k:
            axes_object.plot(
                SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=BLACK_COLOUR,
                markeredgecolor=BLACK_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Normalization type = {0:s}'.format(
            NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        ))

        bias_mlc_panel_file_names[k] = '{0:s}/bias_mlc_{1:s}.jpg'.format(
            output_dir_name, NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        )

        print('Saving figure to: "{0:s}"...'.format(
            bias_mlc_panel_file_names[k]
        ))
        figure_object.savefig(
            bias_mlc_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot near-surface bias for all profiles.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=near_sfc_bias_matrix_mlc_k_day01[..., k],
            min_colour_value=None,
            max_colour_value=_finite_percentile(
                numpy.absolute(near_sfc_bias_matrix_mlc_k_day01), 95
            ),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        this_index = numpy.nanargmin(numpy.ravel(
            numpy.absolute(near_sfc_bias_matrix_mlc_k_day01)
        ))
        best_indices = numpy.unravel_index(
            this_index, near_sfc_bias_matrix_mlc_k_day01.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=BLACK_COLOUR,
                markeredgecolor=BLACK_COLOUR
            )

        if SELECTED_MARKER_INDICES[2] == k:
            axes_object.plot(
                SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=BLACK_COLOUR,
                markeredgecolor=BLACK_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Normalization type = {0:s}'.format(
            NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        ))

        near_sfc_bias_mlc_panel_file_names[k] = (
            '{0:s}/near_surface_bias_mlc_{1:s}.jpg'
        ).format(
            output_dir_name, NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        )

        print('Saving figure to: "{0:s}"...'.format(
            near_sfc_bias_mlc_panel_file_names[k]
        ))
        figure_object.savefig(
            near_sfc_bias_mlc_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot all-flux RMSE for profiles with multi-layer cloud.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=flux_rmse_matrix_mlc_w_m02[..., k],
            min_colour_value=_finite_percentile(flux_rmse_matrix_mlc_w_m02, 0),
            max_colour_value=_finite_percentile(flux_rmse_matrix_mlc_w_m02, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        this_index = numpy.nanargmin(numpy.ravel(flux_rmse_matrix_mlc_w_m02))
        best_indices = numpy.unravel_index(
            this_index, flux_rmse_matrix_mlc_w_m02.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        if SELECTED_MARKER_INDICES[2] == k:
            axes_object.plot(
                SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Normalization type = {0:s}'.format(
            NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        ))

        flux_rmse_mlc_panel_file_names[k] = (
            '{0:s}/flux_rmse_mlc_{1:s}.jpg'
        ).format(
            output_dir_name, NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        )

        print('Saving figure to: "{0:s}"...'.format(
            flux_rmse_mlc_panel_file_names[k]
        ))
        figure_object.savefig(
            flux_rmse_mlc_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot net-flux RMSE for profiles with multi-layer cloud.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=net_flux_rmse_matrix_mlc_w_m02[..., k],
            min_colour_value=_finite_percentile(net_flux_rmse_matrix_mlc_w_m02, 0),
            max_colour_value=_finite_percentile(net_flux_rmse_matrix_mlc_w_m02, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        this_index = numpy.nanargmin(numpy.ravel(net_flux_rmse_matrix_mlc_w_m02))
        best_indices = numpy.unravel_index(
            this_index, net_flux_rmse_matrix_mlc_w_m02.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        if SELECTED_MARKER_INDICES[2] == k:
            axes_object.plot(
                SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=WHITE_COLOUR,
                markeredgecolor=WHITE_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Normalization type = {0:s}'.format(
            NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        ))

        net_flux_rmse_mlc_panel_file_names[k] = (
            '{0:s}/net_flux_rmse_mlc_{1:s}.jpg'
        ).format(
            output_dir_name, NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        )

        print('Saving figure to: "{0:s}"...'.format(
            net_flux_rmse_mlc_panel_file_names[k]
        ))
        figure_object.savefig(
            net_flux_rmse_mlc_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        # Plot net-flux bias for profiles with multi-layer cloud.
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=net_flux_bias_matrix_mlc_w_m02[..., k],
            min_colour_value=None,
            max_colour_value=_finite_percentile(
                numpy.absolute(net_flux_bias_matrix_mlc_w_m02), 95
            ),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        this_index = numpy.nanargmin(numpy.ravel(
            numpy.absolute(net_flux_bias_matrix_mlc_w_m02)
        ))
        best_indices = numpy.unravel_index(
            this_index, net_flux_bias_matrix_mlc_w_m02.shape
        )

        if best_indices[2] == k:
            axes_object.plot(
                best_indices[1], best_indices[0],
                linestyle='None', marker=BEST_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=BLACK_COLOUR,
                markeredgecolor=BLACK_COLOUR
            )

        if SELECTED_MARKER_INDICES[2] == k:
            axes_object.plot(
                SELECTED_MARKER_INDICES[1], SELECTED_MARKER_INDICES[0],
                linestyle='None', marker=SELECTED_MARKER_TYPE,
                markersize=marker_size_px, markeredgewidth=0,
                markerfacecolor=BLACK_COLOUR,
                markeredgecolor=BLACK_COLOUR
            )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title('Normalization type = {0:s}'.format(
            NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        ))

        net_flux_bias_mlc_panel_file_names[k] = (
            '{0:s}/net_flux_bias_mlc_{1:s}.jpg'
        ).format(
            output_dir_name, NORMALIZATION_TYPE_STRINGS_AXIS3[k]
        )

        print('Saving figure to: "{0:s}"...'.format(
            net_flux_bias_mlc_panel_file_names[k]
        ))
        figure_object.savefig(
            net_flux_bias_mlc_panel_file_names[k], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_normalization_types)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_normalization_types) / num_panel_rows
    ))

    dwmse_concat_file_name = '{0:s}/dwmse.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(dwmse_concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=dwmse_panel_file_names,
        output_file_name=dwmse_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=dwmse_concat_file_name,
        output_file_name=dwmse_concat_file_name, output_size_pixels=int(1e7)
    )

    near_sfc_dwmse_concat_file_name = '{0:s}/near_surface_dwmse.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        near_sfc_dwmse_concat_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=near_sfc_dwmse_panel_file_names,
        output_file_name=near_sfc_dwmse_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=near_sfc_dwmse_concat_file_name,
        output_file_name=near_sfc_dwmse_concat_file_name,
        output_size_pixels=int(1e7)
    )

    bias_concat_file_name = '{0:s}/bias.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(bias_concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=bias_panel_file_names,
        output_file_name=bias_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=bias_concat_file_name,
        output_file_name=bias_concat_file_name, output_size_pixels=int(1e7)
    )

    near_sfc_bias_concat_file_name = '{0:s}/near_surface_bias.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        near_sfc_bias_concat_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=near_sfc_bias_panel_file_names,
        output_file_name=near_sfc_bias_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=near_sfc_bias_concat_file_name,
        output_file_name=near_sfc_bias_concat_file_name,
        output_size_pixels=int(1e7)
    )

    flux_rmse_concat_file_name = '{0:s}/flux_rmse.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(
        flux_rmse_concat_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=flux_rmse_panel_file_names,
        output_file_name=flux_rmse_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=flux_rmse_concat_file_name,
        output_file_name=flux_rmse_concat_file_name, output_size_pixels=int(1e7)
    )

    net_flux_rmse_concat_file_name = '{0:s}/net_flux_rmse.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        net_flux_rmse_concat_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=net_flux_rmse_panel_file_names,
        output_file_name=net_flux_rmse_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=net_flux_rmse_concat_file_name,
        output_file_name=net_flux_rmse_concat_file_name,
        output_size_pixels=int(1e7)
    )

    net_flux_bias_concat_file_name = '{0:s}/net_flux_bias.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        net_flux_bias_concat_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=net_flux_bias_panel_file_names,
        output_file_name=net_flux_bias_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=net_flux_bias_concat_file_name,
        output_file_name=net_flux_bias_concat_file_name,
        output_size_pixels=int(1e7)
    )

    dwmse_mlc_concat_file_name = '{0:s}/dwmse_mlc.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(
        dwmse_mlc_concat_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=dwmse_mlc_panel_file_names,
        output_file_name=dwmse_mlc_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=dwmse_mlc_concat_file_name,
        output_file_name=dwmse_mlc_concat_file_name, output_size_pixels=int(1e7)
    )

    near_sfc_dwmse_mlc_concat_file_name = (
        '{0:s}/near_surface_dwmse_mlc.jpg'
    ).format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        near_sfc_dwmse_mlc_concat_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=near_sfc_dwmse_mlc_panel_file_names,
        output_file_name=near_sfc_dwmse_mlc_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=near_sfc_dwmse_mlc_concat_file_name,
        output_file_name=near_sfc_dwmse_mlc_concat_file_name,
        output_size_pixels=int(1e7)
    )

    bias_mlc_concat_file_name = '{0:s}/bias_mlc.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(
        bias_mlc_concat_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=bias_mlc_panel_file_names,
        output_file_name=bias_mlc_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=bias_mlc_concat_file_name,
        output_file_name=bias_mlc_concat_file_name, output_size_pixels=int(1e7)
    )

    near_sfc_bias_mlc_concat_file_name = (
        '{0:s}/near_surface_bias_mlc.jpg'
    ).format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        near_sfc_bias_mlc_concat_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=near_sfc_bias_mlc_panel_file_names,
        output_file_name=near_sfc_bias_mlc_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=near_sfc_bias_mlc_concat_file_name,
        output_file_name=near_sfc_bias_mlc_concat_file_name,
        output_size_pixels=int(1e7)
    )

    flux_rmse_mlc_concat_file_name = '{0:s}/flux_rmse_mlc.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        flux_rmse_mlc_concat_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=flux_rmse_mlc_panel_file_names,
        output_file_name=flux_rmse_mlc_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=flux_rmse_mlc_concat_file_name,
        output_file_name=flux_rmse_mlc_concat_file_name,
        output_size_pixels=int(1e7)
    )

    net_flux_rmse_mlc_concat_file_name = '{0:s}/net_flux_rmse_mlc.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        net_flux_rmse_mlc_concat_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=net_flux_rmse_mlc_panel_file_names,
        output_file_name=net_flux_rmse_mlc_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=net_flux_rmse_mlc_concat_file_name,
        output_file_name=net_flux_rmse_mlc_concat_file_name,
        output_size_pixels=int(1e7)
    )

    net_flux_bias_mlc_concat_file_name = '{0:s}/net_flux_bias_mlc.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(
        net_flux_bias_mlc_concat_file_name
    ))
    imagemagick_utils.concatenate_images(
        input_file_names=net_flux_bias_mlc_panel_file_names,
        output_file_name=net_flux_bias_mlc_concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=net_flux_bias_mlc_concat_file_name,
        output_file_name=net_flux_bias_mlc_concat_file_name,
        output_size_pixels=int(1e7)
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME),
        wavelength_metres=getattr(INPUT_ARG_OBJECT, WAVELENGTH_ARG_NAME),
        isotonic_flag=bool(getattr(INPUT_ARG_OBJECT, ISOTONIC_FLAG_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )