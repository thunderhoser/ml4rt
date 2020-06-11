"""Plots mean example created by average_examples.py."""

import pickle
import argparse
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4rt.plotting import profile_plotting

FIGURE_RESOLUTION_DPI = 600
CONCAT_FIGURE_SIZE_PX = int(1e7)

INPUT_FILE_ARG_NAME = 'input_mean_example_file_name'
OUTPUT_DIR_ARG_NAME = 'output_figure_dir_name'

INPUT_FILE_HELP_STRING = 'Path to input (Pickle) file.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, output_dir_name):
    """Plots mean example created by average_examples.py.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    pickle_file_handle = open(input_file_name, 'rb')
    mean_example_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    # Plot predictors with liquid-water content (LWC).
    figure_object = profile_plotting.plot_predictors(
        example_dict=mean_example_dict, example_index=0,
        plot_ice=False, use_log_scale=True
    )[0]

    panel_file_names = ['foo'] * 3
    panel_file_names[0] = '{0:s}/predictors_with_lwc.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[0]))

    figure_object.savefig(
        panel_file_names[0], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot predictors with ice-water content (IWC).
    figure_object = profile_plotting.plot_predictors(
        example_dict=mean_example_dict, example_index=0,
        plot_ice=True, use_log_scale=True
    )[0]

    panel_file_names[1] = '{0:s}/predictors_with_iwc.jpg'.format(
        output_dir_name
    )
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[1]))

    figure_object.savefig(
        panel_file_names[1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot targets.
    figure_object = profile_plotting.plot_targets(
        example_dict=mean_example_dict, example_index=0, use_log_scale=True
    )[0]

    panel_file_names[2] = '{0:s}/targets.jpg'.format(output_dir_name)
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[2]))

    figure_object.savefig(
        panel_file_names[2], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Concatenate panels.
    concat_figure_file_name = '{0:s}/predictors_and_targets.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=2, num_panel_columns=2, border_width_pixels=50
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
