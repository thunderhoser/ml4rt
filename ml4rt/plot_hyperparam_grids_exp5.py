"""Plots scores on hyperparameter grid for Experiment 5."""

import sys
import glob
import os.path
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

import evaluation
import plotting_utils
import example_utils
import prediction_io
import file_system_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DENSE_LAYER_COUNTS = numpy.array([2, 3, 4, 5], dtype=int)
DENSE_LAYER_DROPOUT_RATES = numpy.array([0.125, 0.25, 0.375, 0.5, 0.625, 0.75])
SCALAR_LOSS_FUNCTION_WEIGHTS = numpy.array([1, 2.5, 5, 10, 25, 50])

FONT_SIZE = 20
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

ORIG_LOCATION_SET_STRING = 'orig'
NEW_LOCATION_SET_STRING = 'new'
TROPICAL_LOCATION_SET_STRING = 'tropical'
VALID_LOCATION_SET_STRINGS = [
    ORIG_LOCATION_SET_STRING, NEW_LOCATION_SET_STRING,
    TROPICAL_LOCATION_SET_STRING
]

EXPERIMENT_DIR_ARG_NAME = 'experiment_dir_name'
ISOTONIC_FLAG_ARG_NAME = 'isotonic_flag'
LOCATION_SET_ARG_NAME = 'location_set_string'
EXCLUDE_SUMMIT_ARG_NAME = 'exclude_summit_greenland'

EXPERIMENT_DIR_HELP_STRING = 'Name of top-level directory with models.'
ISOTONIC_FLAG_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot results with(out) isotonic regression.'
)
LOCATION_SET_HELP_STRING = (
    'Location set.  Must be in the following list:\n{0:s}'
).format(str(VALID_LOCATION_SET_STRINGS))

EXCLUDE_SUMMIT_HELP_STRING = (
    'Boolean flag.  If 1, will not apply to examples from Summit.'
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
INPUT_ARG_PARSER.add_argument(
    '--' + LOCATION_SET_ARG_NAME, type=str, required=True,
    help=LOCATION_SET_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXCLUDE_SUMMIT_ARG_NAME, type=int, required=False, default=0,
    help=EXCLUDE_SUMMIT_HELP_STRING
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
    :param figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :param axes_object: Axes handle (instance of
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

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=score_matrix[numpy.invert(numpy.isnan(score_matrix))],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='horizontal', extend_min=False, extend_max=False,
        fraction_of_axis_length=0.8, font_size=FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _read_scores_one_model(model_dir_name, isotonic_flag, location_set_string,
                           exclude_summit_greenland):
    """Reads scores for one model.

    :param model_dir_name: Name of directory with trained model and evaluation
        data.
    :param isotonic_flag: See documentation at top of file.
    :param location_set_string: Same.
    :param exclude_summit_greenland: Same.
    :return: prmse_k_day01: Profile root mean squared error (RMSE):
    :return: dwmse_k3_day03: Dual-weighted mean squared error.
    :return: down_flux_rmse_w_m02: RMSE for surface downwelling flux.
    :return: up_flux_rmse_w_m02: RMSE for TOA upwelling flux.
    """

    model_file_pattern = '{0:s}/model*.h5'.format(model_dir_name)
    model_file_names = glob.glob(model_file_pattern)

    if len(model_file_names) == 0:
        return numpy.nan, numpy.nan

    model_file_names.sort()
    model_file_name = model_file_names[-1]

    if location_set_string == TROPICAL_LOCATION_SET_STRING:
        location_subdir_name = '_tropical_sites'
    elif exclude_summit_greenland:
        location_subdir_name = '_new_loc_sans_summit'
    elif location_set_string == NEW_LOCATION_SET_STRING:
        location_subdir_name = '_new_locations'
    else:
        location_subdir_name = ''

    evaluation_file_name = '{0:s}/{1:s}validation{2:s}/evaluation.nc'.format(
        model_file_name[:-3],
        'isotonic_regression/' if isotonic_flag else '',
        location_subdir_name
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

    prmse_k_day01 = (
        evaluation_table_xarray[evaluation.VECTOR_PRMSE_KEY].values[0]
    )

    j = scalar_target_names.index(
        example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
    )
    down_flux_rmse_w_m02 = numpy.sqrt(
        evaluation_table_xarray[evaluation.SCALAR_MSE_KEY].values[j]
    )

    j = scalar_target_names.index(example_utils.SHORTWAVE_TOA_UP_FLUX_NAME)
    up_flux_rmse_w_m02 = numpy.sqrt(
        evaluation_table_xarray[evaluation.SCALAR_MSE_KEY].values[j]
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
    dwmse_k3_day03 = numpy.mean(
        weight_matrix * (vector_prediction_matrix - vector_target_matrix) ** 2
    )
    return (
        prmse_k_day01, dwmse_k3_day03, down_flux_rmse_w_m02, up_flux_rmse_w_m02
    )


def _print_ranking_one_score(score_matrix, score_name):
    """Prints ranking for one score.

    C = number of dense-layer counts
    D = number of dense-layer dropout rates
    W = number of weights for scalar loss function

    :param score_matrix: C-by-D-by-W numpy array of scores.
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
            '{0:d}th-lowest {1:s} = {2:.4g} ... num dense layers = {3:d} ... '
            'dense-layer dropout rate = {4:.3f} ... '
            'weight for scalar loss function = {5:.1f}'
        ).format(
            m + 1, score_name, score_matrix[i, j, k], DENSE_LAYER_COUNTS[i],
            DENSE_LAYER_DROPOUT_RATES[j], SCALAR_LOSS_FUNCTION_WEIGHTS[k]
        ))


def _print_ranking_all_scores(
        prmse_matrix_k_day01, dwmse_matrix_k3_day03,
        down_flux_rmse_matrix_w_m02, up_flux_rmse_matrix_w_m02):
    """Prints ranking for one score.

    C = number of dense-layer counts
    D = number of dense-layer dropout rates
    W = number of weights for scalar loss function

    :param prmse_matrix_k_day01: C-by-D-by-W numpy array of profile RMSE.
    :param dwmse_matrix_k3_day03: C-by-D-by-W numpy array of dual-weighted MSE.
    :param down_flux_rmse_matrix_w_m02: C-by-D-by-W numpy array of RMSE for
        surface downwelling flux.
    :param up_flux_rmse_matrix_w_m02: C-by-D-by-W numpy array of RMSE for TOA
        upwelling flux.
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

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:d}th-lowest PRMSE = {1:.4g} K day^-1 ... '
            'num dense layers = {2:d} ... '
            'dense-layer dropout rate = {3:.3f} ... '
            'weight for scalar loss function = {4:.1f} ... '
            'ranks for DWMSE, down-flux RMSE, and up-flux RMSE = '
            '{5:.1f}, {6:.1f}, {7:.1f}'
        ).format(
            m + 1, prmse_matrix_k_day01[i, j, k],
            DENSE_LAYER_COUNTS[i], DENSE_LAYER_DROPOUT_RATES[j],
            SCALAR_LOSS_FUNCTION_WEIGHTS[k],
            dwmse_rank_matrix[i, j, k], down_flux_rmse_rank_matrix[i, j, k],
            up_flux_rmse_rank_matrix[i, j, k]
        ))


def _run(experiment_dir_name, isotonic_flag, location_set_string,
         exclude_summit_greenland):
    """Plots scores on hyperparameter grid for Experiment 5.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param isotonic_flag: Same.
    :param location_set_string: Same.
    :param exclude_summit_greenland: Same.
    :raises: ValueError: if
        `location_set_string not in VALID_LOCATION_SET_STRINGS`.
    """

    if location_set_string not in VALID_LOCATION_SET_STRINGS:
        error_string = (
            'Location set ("{0:s}") is not in the following list:\n{1:s}'
        ).format(location_set_string, str(VALID_LOCATION_SET_STRINGS))

        raise ValueError(error_string)

    exclude_summit_greenland = (
        exclude_summit_greenland
        and location_set_string == NEW_LOCATION_SET_STRING
    )

    num_dense_layer_counts = len(DENSE_LAYER_COUNTS)
    num_dropout_rates = len(DENSE_LAYER_DROPOUT_RATES)
    num_loss_function_weights = len(SCALAR_LOSS_FUNCTION_WEIGHTS)
    dimensions = (
        num_dense_layer_counts, num_dropout_rates, num_loss_function_weights
    )

    prmse_matrix_k_day01 = numpy.full(dimensions, numpy.nan)
    dwmse_matrix_k3_day03 = numpy.full(dimensions, numpy.nan)
    down_flux_rmse_matrix_w_m02 = numpy.full(dimensions, numpy.nan)
    up_flux_rmse_matrix_w_m02 = numpy.full(dimensions, numpy.nan)

    y_tick_labels = [
        '{0:.3f}'.format(d).replace('-1', '0')
        for d in DENSE_LAYER_DROPOUT_RATES
    ]
    x_tick_labels = [
        '{0:.1f}'.format(d).replace('-1', '0')
        for d in SCALAR_LOSS_FUNCTION_WEIGHTS
    ]
    y_axis_label = 'Dense-layer dropout rate'
    x_axis_label = 'Loss-function weight for fluxes'

    for i in range(num_dense_layer_counts):
        for j in range(num_dropout_rates):
            for k in range(num_loss_function_weights):
                this_model_dir_name = (
                    '{0:s}/num-dense-layers={1:d}_dense-dropout={2:.3f}_'
                    'scalar-lf-weight={3:05.1f}'
                ).format(
                    experiment_dir_name, DENSE_LAYER_COUNTS[i],
                    DENSE_LAYER_DROPOUT_RATES[j],
                    SCALAR_LOSS_FUNCTION_WEIGHTS[k]
                )

                (
                    prmse_matrix_k_day01[i, j, k],
                    dwmse_matrix_k3_day03[i, j, k],
                    down_flux_rmse_matrix_w_m02[i, j, k],
                    up_flux_rmse_matrix_w_m02[i, j, k]
                ) = _read_scores_one_model(
                    model_dir_name=this_model_dir_name,
                    isotonic_flag=isotonic_flag,
                    location_set_string=location_set_string,
                    exclude_summit_greenland=exclude_summit_greenland
                )

    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=prmse_matrix_k_day01, score_name='PRMSE (K day^-1)'
    )
    print(SEPARATOR_STRING)

    _print_ranking_one_score(
        score_matrix=dwmse_matrix_k3_day03, score_name='DWMSE (K^3 day^-3)'
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

    _print_ranking_all_scores(
        prmse_matrix_k_day01=prmse_matrix_k_day01,
        dwmse_matrix_k3_day03=dwmse_matrix_k3_day03,
        down_flux_rmse_matrix_w_m02=down_flux_rmse_matrix_w_m02,
        up_flux_rmse_matrix_w_m02=up_flux_rmse_matrix_w_m02
    )
    print(SEPARATOR_STRING)

    for i in range(num_dense_layer_counts):
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=prmse_matrix_k_day01[i, ...],
            min_colour_value=numpy.nanpercentile(prmse_matrix_k_day01, 0),
            max_colour_value=numpy.nanpercentile(prmse_matrix_k_day01, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        title_string = r'Profile RMSE (K day$^{-1}$)'
        title_string += ' with {0:d} dense layers'.format(DENSE_LAYER_COUNTS[i])
        axes_object.set_title(title_string)

        if location_set_string == TROPICAL_LOCATION_SET_STRING:
            location_subdir_name = 'tropical_sites/'
        elif exclude_summit_greenland:
            location_subdir_name = 'new_loc_sans_summit/'
        elif location_set_string == NEW_LOCATION_SET_STRING:
            location_subdir_name = 'new_locations/'
        else:
            location_subdir_name = ''

        figure_file_name = (
            '{0:s}/{1:s}{2:s}num-dense-layers={3:d}_prmse_grid.jpg'
        ).format(
            experiment_dir_name, location_subdir_name,
            'isotonic_regression/' if isotonic_flag else '',
            DENSE_LAYER_COUNTS[i]
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

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        title_string = r'Dual-weighted MSE (K$^{3}$ day$^{-3}$)'
        title_string += ' with {0:d} dense layers'.format(DENSE_LAYER_COUNTS[i])
        axes_object.set_title(title_string)

        figure_file_name = (
            '{0:s}/{1:s}{2:s}num-dense-layers={3:d}_dwmse_grid.jpg'
        ).format(
            experiment_dir_name, location_subdir_name,
            'isotonic_regression/' if isotonic_flag else '',
            DENSE_LAYER_COUNTS[i]
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

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        title_string = r'RMSE for surface down flux (W m$^{-2}$)'
        title_string += ' with {0:d} dense layers'.format(DENSE_LAYER_COUNTS[i])
        axes_object.set_title(title_string)

        figure_file_name = (
            '{0:s}/{1:s}{2:s}num-dense-layers={3:d}_down_flux_rmse_grid.jpg'
        ).format(
            experiment_dir_name, location_subdir_name,
            'isotonic_regression/' if isotonic_flag else '',
            DENSE_LAYER_COUNTS[i]
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

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        title_string = r'RMSE for TOA up flux (W m$^{-2}$)'
        title_string += ' with {0:d} dense layers'.format(DENSE_LAYER_COUNTS[i])
        axes_object.set_title(title_string)

        figure_file_name = (
            '{0:s}/{1:s}{2:s}num-dense-layers={3:d}_up_flux_rmse_grid.jpg'
        ).format(
            experiment_dir_name, location_subdir_name,
            'isotonic_regression/' if isotonic_flag else '',
            DENSE_LAYER_COUNTS[i]
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
        isotonic_flag=bool(getattr(INPUT_ARG_OBJECT, ISOTONIC_FLAG_ARG_NAME)),
        location_set_string=getattr(INPUT_ARG_OBJECT, LOCATION_SET_ARG_NAME),
        exclude_summit_greenland=bool(getattr(
            INPUT_ARG_OBJECT, EXCLUDE_SUMMIT_ARG_NAME
        )),
    )