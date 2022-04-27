"""Plots predictors and targets for one example.

Specifically, this script plots an example with multi-layer liquid cloud and
non-zero ice content.
"""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4rt.io import example_io
from ml4rt.utils import example_utils
from ml4rt.plotting import profile_plotting
from ml4rt.scripts import plot_examples

MIN_CLOUD_LAYER_PATH_KG_M02 = 0.025
PREDICTOR_NAMES_BY_SET = plot_examples.PREDICTOR_NAMES_BY_SET
PREDICTOR_COLOURS_BY_SET = plot_examples.PREDICTOR_COLOURS_BY_SET

LETTER_LABEL_FONT_SIZE = 36
LETTER_LABEL_X_COORD = -0.05
LETTER_LABEL_Y_COORD = 1.05

LINE_WIDTH = plot_examples.LINE_WIDTH
FIGURE_RESOLUTION_DPI = plot_examples.FIGURE_RESOLUTION_DPI

EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXAMPLE_FILE_HELP_STRING = (
    'Path to file with desired example.  Will be read by '
    '`example_io.read_file`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _do_plotting(example_dict, example_index, output_dir_name):
    """Does the plotting.

    :param example_dict: See doc for `example_io.read_file`.
    :param example_index: Will plot predictors and targets for [k]th example in
        dictionary, where k = `example_index`.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    """

    example_id_string = (
        example_dict[example_utils.EXAMPLE_IDS_KEY][example_index]
    )
    num_predictor_sets = len(PREDICTOR_NAMES_BY_SET)

    letter_label = None
    panel_file_names = []

    for k in range(num_predictor_sets):
        these_flags = numpy.array([
            n in example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
            for n in PREDICTOR_NAMES_BY_SET[k]
        ], dtype=bool)

        these_indices = numpy.where(these_flags)[0]
        if len(these_indices) == 0:
            continue

        predictor_names = [PREDICTOR_NAMES_BY_SET[k][i] for i in these_indices]
        predictor_colours = [
            PREDICTOR_COLOURS_BY_SET[k][i] for i in these_indices
        ]

        handle_dict = profile_plotting.plot_predictors(
            example_dict=example_dict, example_index=example_index,
            predictor_names=predictor_names,
            predictor_colours=predictor_colours,
            predictor_line_widths=
            numpy.full(len(these_indices), LINE_WIDTH),
            predictor_line_styles=['solid'] * len(these_indices),
            use_log_scale=True
        )

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        plotting_utils.label_axes(
            axes_object=handle_dict[profile_plotting.AXES_OBJECTS_KEY][0],
            label_string='({0:s})'.format(letter_label),
            font_size=LETTER_LABEL_FONT_SIZE,
            x_coord_normalized=LETTER_LABEL_X_COORD,
            y_coord_normalized=LETTER_LABEL_Y_COORD
        )

        this_file_name = '{0:s}/{1:s}_predictor-set-{2:d}.jpg'.format(
            output_dir_name, example_id_string.replace('_', '-'), k
        )
        panel_file_names.append(this_file_name)
        print('Saving figure to: "{0:s}"...'.format(this_file_name))

        figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
        figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)

        imagemagick_utils.trim_whitespace(
            input_file_name=this_file_name, output_file_name=this_file_name
        )
        imagemagick_utils.resize_image(
            input_file_name=this_file_name, output_file_name=this_file_name,
            output_size_pixels=int(2.5e6)
        )

    handle_dict = profile_plotting.plot_targets(
        example_dict=example_dict, example_index=example_index,
        for_shortwave=True, use_log_scale=True, line_width=LINE_WIDTH,
        line_style='solid'
    )

    letter_label = chr(ord(letter_label) + 1)
    plotting_utils.label_axes(
        axes_object=handle_dict[profile_plotting.HEATING_RATE_HANDLE_KEY],
        label_string='({0:s})'.format(letter_label),
        font_size=LETTER_LABEL_FONT_SIZE,
        x_coord_normalized=LETTER_LABEL_X_COORD,
        y_coord_normalized=LETTER_LABEL_Y_COORD
    )

    this_file_name = '{0:s}/{1:s}_shortwave_targets.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )
    panel_file_names.append(this_file_name)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=this_file_name, output_file_name=this_file_name
    )
    imagemagick_utils.resize_image(
        input_file_name=this_file_name, output_file_name=this_file_name,
        output_size_pixels=int(2.5e6)
    )

    handle_dict = profile_plotting.plot_targets(
        example_dict=example_dict, example_index=example_index,
        for_shortwave=False, use_log_scale=True, line_width=LINE_WIDTH,
        line_style='solid'
    )

    letter_label = chr(ord(letter_label) + 1)
    plotting_utils.label_axes(
        axes_object=handle_dict[profile_plotting.HEATING_RATE_HANDLE_KEY],
        label_string='({0:s})'.format(letter_label),
        font_size=LETTER_LABEL_FONT_SIZE,
        x_coord_normalized=LETTER_LABEL_X_COORD,
        y_coord_normalized=LETTER_LABEL_Y_COORD
    )

    this_file_name = '{0:s}/{1:s}_longwave_targets.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )
    panel_file_names.append(this_file_name)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=this_file_name, output_file_name=this_file_name
    )
    imagemagick_utils.resize_image(
        input_file_name=this_file_name, output_file_name=this_file_name,
        output_size_pixels=int(2.5e6)
    )

    concat_file_name = '{0:s}/{1:s}_concat.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=2, num_panel_columns=3, border_width_pixels=25
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_file_name, output_file_name=concat_file_name
    )


def _run(example_file_name, output_dir_name):
    """Plots predictors and targets for one example.

    This is effectively the main method.

    :param example_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    print('Reading data from: "{0:s}"...'.format(example_file_name))
    example_dict = example_io.read_file(example_file_name)
    cloud_layer_counts = example_utils.find_cloud_layers(
        example_dict=example_dict, min_path_kg_m02=MIN_CLOUD_LAYER_PATH_KG_M02,
        for_ice=False
    )[1]

    desired_indices = numpy.where(cloud_layer_counts > 1)[0]
    example_dict = example_utils.subset_by_index(
        example_dict=example_dict, desired_indices=desired_indices
    )

    liquid_water_paths_kg_m02 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_utils.LIQUID_WATER_PATH_NAME, height_m_agl=10.
    )
    sort_indices = numpy.argsort(-1 * liquid_water_paths_kg_m02)
    desired_index = sort_indices[10]

    _do_plotting(
        example_dict=example_dict, example_index=desired_index,
        output_dir_name=output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
