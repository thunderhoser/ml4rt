"""Plots scores on hyperparameter grid for Experiment 3."""

import sys
import glob
import os.path
import argparse
import numpy
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

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

CONV_LAYER_DROPOUT_RATES = numpy.array([0.05, 0.1, 0.15, 0.2, 0.25])
UPCONV_LAYER_DROPOUT_RATES = numpy.array([
    -1, 0.05, 0.1, 0.15, 0.2, 0.25
])
SKIP_LAYER_DROPOUT_RATES = numpy.array([-1, 0.05, 0.1, 0.15, 0.2, 0.25])

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


def _read_scores_one_model(model_dir_name, isotonic_flag):
    """Reads scores (PRMSE and DWMSE) for one model.

    :param model_dir_name: Name of directory with trained model and evaluation
        data.
    :param isotonic_flag: See documentation at top of file.
    :return: prmse_k_day01: Profile root mean squared error.
    :return: dwmse_k3_day03: Dual-weighted mean squared error.
    """

    model_file_pattern = '{0:s}/model*.h5'.format(model_dir_name)
    model_file_names = glob.glob(model_file_pattern)

    if len(model_file_names) == 0:
        return numpy.nan, numpy.nan

    model_file_names.sort()
    model_file_name = model_file_names[-1]
    evaluation_file_name = '{0:s}/{1:s}validation/evaluation.nc'.format(
        model_file_name[:-3], 'isotonic_regression/' if isotonic_flag else ''
    )

    if not os.path.isfile(evaluation_file_name):
        return numpy.nan, numpy.nan

    pathless_model_file_name = os.path.split(model_file_name)[1]
    extensionless_model_file_name = (
        os.path.splitext(pathless_model_file_name)[0]
    )
    dwmse_k3_day03 = float(extensionless_model_file_name.split('=')[-1])

    print('Reading data from: "{0:s}"...'.format(evaluation_file_name))
    evaluation_table_xarray = evaluation.read_file(evaluation_file_name)
    prmse_k_day01 = (
        evaluation_table_xarray[evaluation.VECTOR_PRMSE_KEY].values[0]
    )

    return prmse_k_day01, dwmse_k3_day03


def _run(experiment_dir_name, isotonic_flag):
    """Plots scores on hyperparameter grid for Experiment 1.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    :param isotonic_flag: Same.
    """

    num_conv_dropout_rates = len(CONV_LAYER_DROPOUT_RATES)
    num_upconv_dropout_rates = len(UPCONV_LAYER_DROPOUT_RATES)
    num_skip_dropout_rates = len(SKIP_LAYER_DROPOUT_RATES)
    dimensions = (
        num_conv_dropout_rates, num_upconv_dropout_rates, num_skip_dropout_rates
    )

    prmse_matrix_k_day01 = numpy.full(dimensions, numpy.nan)
    dwmse_matrix_k3_day03 = numpy.full(dimensions, numpy.nan)

    y_tick_labels = ['{0:.2f}'.format(d) for d in UPCONV_LAYER_DROPOUT_RATES]
    x_tick_labels = ['{0:.2f}'.format(d) for d in SKIP_LAYER_DROPOUT_RATES]
    y_axis_label = 'Upconv-layer dropout rate'
    x_axis_label = 'Skip-layer dropout rate'

    for i in range(num_conv_dropout_rates):
        for j in range(num_upconv_dropout_rates):
            for k in range(num_skip_dropout_rates):
                this_model_dir_name = (
                    '{0:s}/conv-dropout={1:.2f}_upconv-dropout={2:.2f}_'
                    'skip-dropout={3:.2f}'
                ).format(
                    experiment_dir_name, CONV_LAYER_DROPOUT_RATES[i],
                    UPCONV_LAYER_DROPOUT_RATES[j], SKIP_LAYER_DROPOUT_RATES[k]
                )

                (
                    prmse_matrix_k_day01[i, j, k],
                    dwmse_matrix_k3_day03[i, j, k]
                ) = _read_scores_one_model(
                    model_dir_name=this_model_dir_name,
                    isotonic_flag=isotonic_flag
                )

    print(SEPARATOR_STRING)

    these_prmse_k_day01 = numpy.ravel(prmse_matrix_k_day01)
    these_prmse_k_day01[numpy.isnan(these_prmse_k_day01)] = numpy.inf
    linear_sort_indices = numpy.argsort(these_prmse_k_day01)
    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        linear_sort_indices, prmse_matrix_k_day01.shape
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:d}th-lowest PRMSE = {1:.2g} K day^-1 ... '
            'conv dropout rate = {2:.2f} ... upconv dropout rate = {3:.2f} ... '
            'skip-layer dropout rate = {4:.2f}'
        ).format(
            m + 1, prmse_matrix_k_day01[i, j, k], CONV_LAYER_DROPOUT_RATES[i],
            UPCONV_LAYER_DROPOUT_RATES[j], SKIP_LAYER_DROPOUT_RATES[k]
        ))

    print(SEPARATOR_STRING)

    these_dwmse_k3_day03 = numpy.ravel(dwmse_matrix_k3_day03)
    these_dwmse_k3_day03[numpy.isnan(these_dwmse_k3_day03)] = numpy.inf
    linear_sort_indices = numpy.argsort(these_dwmse_k3_day03)
    i_sort_indices, j_sort_indices, k_sort_indices = numpy.unravel_index(
        linear_sort_indices, dwmse_matrix_k3_day03.shape
    )

    for m in range(len(i_sort_indices)):
        i = i_sort_indices[m]
        j = j_sort_indices[m]
        k = k_sort_indices[m]

        print((
            '{0:d}th-lowest DWMSE = {1:.2g} K^3 day^-3 ... '
            'conv dropout rate = {2:.2f} ... upconv dropout rate = {3:.2f} ... '
            'skip-layer dropout rate = {4:.2f}'
        ).format(
            m + 1, dwmse_matrix_k3_day03[i, j, k], CONV_LAYER_DROPOUT_RATES[i],
            UPCONV_LAYER_DROPOUT_RATES[j], SKIP_LAYER_DROPOUT_RATES[k]
        ))

    print(SEPARATOR_STRING)

    for i in range(num_conv_dropout_rates):
        figure_object, axes_object = _plot_scores_2d(
            score_matrix=prmse_matrix_k_day01[i, ...],
            min_colour_value=numpy.nanpercentile(prmse_matrix_k_day01, 0),
            max_colour_value=numpy.nanpercentile(prmse_matrix_k_day01, 95),
            x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
        )

        axes_object.set_xlabel(x_axis_label)
        axes_object.set_ylabel(y_axis_label)
        axes_object.set_title(r'Profile RMSE (K day$^{-1}$)')
        figure_file_name = (
            '{0:s}/{1:s}conv-dropout={2:.2f}_prmse_grid.jpg'
        ).format(
            experiment_dir_name,
            'isotonic_regression/' if isotonic_flag else '',
            CONV_LAYER_DROPOUT_RATES[i]
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
        axes_object.set_title(r'Dual-weighted MSE (K$^{3}$ day$^{-3}$)')
        figure_file_name = (
            '{0:s}/{1:s}conv-dropout={2:.2f}_dwmse_grid.jpg'
        ).format(
            experiment_dir_name,
            'isotonic_regression/' if isotonic_flag else '',
            CONV_LAYER_DROPOUT_RATES[i]
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
