"""Plots scores on hyperparameter grid for Experiment 1."""

import sys
import glob
import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import evaluation
import plotting_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PLATEAU_LR_MULTIPLIERS = numpy.array([0.9, 0.8, 0.7, 0.6, 0.5])
BATCH_SIZES = numpy.array(
    [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], dtype=int
)

FONT_SIZE = 30
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

EXPERIMENT_DIR_ARG_NAME = 'experiment_dir_name'
EXPERIMENT_DIR_HELP_STRING = 'Name of top-level directory with models.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT_DIR_ARG_NAME, type=str, nargs='+', required=True,
    help=EXPERIMENT_DIR_HELP_STRING
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

    plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=score_matrix,
        colour_map_object=colour_map_object,
        min_value=min_colour_value, max_value=max_colour_value,
        orientation_string='vertical', extend_min=False, extend_max=False,
        fraction_of_axis_length=0.8, font_size=FONT_SIZE
    )

    return figure_object, axes_object


def _run(experiment_dir_name):
    """Plots scores on hyperparameter grid for Experiment 1.

    This is effectively the main method.

    :param experiment_dir_name: See documentation at top of file.
    """

    num_multipliers = len(PLATEAU_LR_MULTIPLIERS)
    num_batch_sizes = len(BATCH_SIZES)

    prmse_matrix_k_day01 = numpy.full(
        (num_multipliers, num_batch_sizes), numpy.nan
    )
    dwmse_matrix_k3_day03 = numpy.full(
        (num_multipliers, num_batch_sizes), numpy.nan
    )

    y_tick_labels = ['{0:.1f}'.format(m) for m in PLATEAU_LR_MULTIPLIERS]
    x_tick_labels = ['{0:d}'.format(s) for s in BATCH_SIZES]
    y_axis_label = 'Learning-rate multiplier'
    x_axis_label = 'Batch size'

    for i in range(num_multipliers):
        for j in range(num_batch_sizes):
            this_model_dir_name = (
                '{0:s}/plateau-lr-multiplier={1:.1f}_batch-size={2:04d}'
            ).format(
                experiment_dir_name, PLATEAU_LR_MULTIPLIERS[i], BATCH_SIZES[j]
            )

            this_model_file_pattern = '{0:s}/model*.h5'.format(
                this_model_dir_name
            )

            these_model_file_names = glob.glob(this_model_file_pattern)
            if len(these_model_file_names) == 0:
                continue

            these_model_file_names.sort()
            this_model_file_name = these_model_file_names[-1]
            this_eval_file_name = '{0:s}/validation/evaluation.nc'.format(
                this_model_file_name[:-3]
            )

            if not os.path.isfile(this_eval_file_name):
                continue

            this_pathless_model_file_name = (
                os.path.split(this_model_file_name)[1]
            )
            this_extensionless_model_file_name = (
                os.path.splitext(this_pathless_model_file_name)[0]
            )
            dwmse_matrix_k3_day03[i, j] = float(
                this_extensionless_model_file_name.split('=')[-1]
            )

            print('Reading data from: "{0:s}"...'.format(this_eval_file_name))
            this_eval_table_xarray = evaluation.read_file(this_eval_file_name)
            prmse_matrix_k_day01[i, j] = (
                this_eval_table_xarray[evaluation.VECTOR_PRMSE_KEY].values[0]
            )

    print(SEPARATOR_STRING)

    figure_object, axes_object = _plot_scores_2d(
        score_matrix=prmse_matrix_k_day01,
        min_colour_value=numpy.nanpercentile(prmse_matrix_k_day01, 1),
        max_colour_value=numpy.nanpercentile(prmse_matrix_k_day01, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title(r'Profile RMSE (K day$^{-1}$)')
    figure_file_name = '{0:s}/prmse_grid.jpg'.format(experiment_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    figure_object, axes_object = _plot_scores_2d(
        score_matrix=dwmse_matrix_k3_day03,
        min_colour_value=numpy.nanpercentile(dwmse_matrix_k3_day03, 1),
        max_colour_value=numpy.nanpercentile(dwmse_matrix_k3_day03, 99),
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title(r'Dual-weighted MSE (K$^{3}$ day$^{-3}$)')
    figure_file_name = '{0:s}/dwmse_grid.jpg'.format(experiment_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        experiment_dir_name=getattr(INPUT_ARG_OBJECT, EXPERIMENT_DIR_ARG_NAME)
    )
