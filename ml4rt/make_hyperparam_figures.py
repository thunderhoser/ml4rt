"""Creates figures showing model performance as a function of hyperparams."""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import imagemagick_utils

DENSE_LAYER_COUNTS = numpy.array([2, 3, 4, 5], dtype=int)

SCORE_NAMES = [
    'dwmse', 'prmse', 'down_flux_rmse', 'up_flux_rmse', 'net_flux_rmse'
]

NUM_PANEL_ROWS = 2
NUM_PANEL_COLUMNS = 2
PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(2e7)

INPUT_DIR_ARG_NAME = 'input_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of input directory, containing hyperparameter grids created by '
    'plot_hyperparam_grids_exp6.py or similar.  This script will panel those '
    'figures together.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Output images (paneled figure and temporary '
    'figures) will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, output_dir_name):
    """Creates figure showing model performance as a function of hyperparams.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_scores = len(SCORE_NAMES)
    num_dense_layer_counts = len(DENSE_LAYER_COUNTS)

    for j in range(num_scores):
        panel_file_names = [
            '{0:s}/num-dense-layers={1:d}_{2:s}_grid.jpg'.format(
                input_dir_name, d, SCORE_NAMES[j]
            ) for d in DENSE_LAYER_COUNTS
        ]
        resized_panel_file_names = [
            '{0:s}/num-dense-layers={1:d}_{2:s}_grid.jpg'.format(
                output_dir_name, d, SCORE_NAMES[j]
            ) for d in DENSE_LAYER_COUNTS
        ]

        for i in range(num_dense_layer_counts):
            imagemagick_utils.trim_whitespace(
                input_file_name=panel_file_names[i],
                output_file_name=resized_panel_file_names[i]
            )
            imagemagick_utils.resize_image(
                input_file_name=resized_panel_file_names[i],
                output_file_name=resized_panel_file_names[i],
                output_size_pixels=PANEL_SIZE_PX
            )

        concat_figure_file_name = '{0:s}/{1:s}_grid.jpg'.format(
            output_dir_name, SCORE_NAMES[j]
        )
        print('Concatenating panels to: "{0:s}"...'.format(
            concat_figure_file_name
        ))

        imagemagick_utils.concatenate_images(
            input_file_names=resized_panel_file_names,
            output_file_name=concat_figure_file_name,
            num_panel_rows=NUM_PANEL_ROWS, num_panel_columns=NUM_PANEL_COLUMNS
        )
        imagemagick_utils.resize_image(
            input_file_name=concat_figure_file_name,
            output_file_name=concat_figure_file_name,
            output_size_pixels=CONCAT_FIGURE_SIZE_PX
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
