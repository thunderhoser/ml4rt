"""Plots profiles (all target variables and 4 predictor variables)."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import imagemagick_utils
import example_utils
import misc as misc_utils
import profile_plotting
import make_saliency_maps

MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

BLACK_COLOUR = numpy.full(3, 0.)
ORANGE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
PURPLE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
GREEN_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255

SHORTWAVE_TARGET_NAMES = [
    example_utils.SHORTWAVE_DOWN_FLUX_NAME,
    example_utils.SHORTWAVE_UP_FLUX_NAME,
    example_utils.SHORTWAVE_HEATING_RATE_NAME
]

LONGWAVE_TARGET_NAMES = [
    example_utils.LONGWAVE_DOWN_FLUX_NAME,
    example_utils.LONGWAVE_UP_FLUX_NAME,
    example_utils.LONGWAVE_HEATING_RATE_NAME
]

SHORTWAVE_PREDICTOR_NAMES = [
    example_utils.LIQUID_WATER_CONTENT_NAME,
    example_utils.ICE_WATER_CONTENT_NAME,
    example_utils.SPECIFIC_HUMIDITY_NAME,
    example_utils.AEROSOL_EXTINCTION_NAME
]
SHORTWAVE_PREDICTOR_COLOURS = [
    ORANGE_COLOUR, PURPLE_COLOUR, GREEN_COLOUR, BLACK_COLOUR
]

LONGWAVE_PREDICTOR_NAMES = [
    example_utils.LIQUID_WATER_CONTENT_NAME,
    example_utils.ICE_WATER_CONTENT_NAME,
    example_utils.SPECIFIC_HUMIDITY_NAME,
    example_utils.TEMPERATURE_NAME
]
LONGWAVE_PREDICTOR_COLOURS = [
    ORANGE_COLOUR, PURPLE_COLOUR, GREEN_COLOUR, BLACK_COLOUR
]

LINE_WIDTH = 3
FIGURE_RESOLUTION_DPI = 300

EXAMPLE_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_FILE_ARG_NAME
NUM_EXAMPLES_ARG_NAME = make_saliency_maps.NUM_EXAMPLES_ARG_NAME
EXAMPLE_DIR_ARG_NAME = make_saliency_maps.EXAMPLE_DIR_ARG_NAME
EXAMPLE_ID_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_ID_FILE_ARG_NAME
PLOT_SHORTWAVE_ARG_NAME = 'plot_shortwave'
TARGET_WAVELENGTH_ARG_NAME = 'target_wavelength_metres'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXAMPLE_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_FILE_HELP_STRING
NUM_EXAMPLES_HELP_STRING = make_saliency_maps.NUM_EXAMPLES_HELP_STRING
EXAMPLE_DIR_HELP_STRING = make_saliency_maps.EXAMPLE_DIR_HELP_STRING
EXAMPLE_ID_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_ID_FILE_HELP_STRING
PLOT_SHORTWAVE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot shortwave (longwave) variables.'
)
TARGET_WAVELENGTH_HELP_STRING = 'Wavelength for target variables.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (figures will be saved here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_ID_FILE_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_ID_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SHORTWAVE_ARG_NAME, type=int, required=True,
    help=PLOT_SHORTWAVE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_WAVELENGTH_ARG_NAME, type=float, required=True,
    help=TARGET_WAVELENGTH_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_example(
        example_dict, example_index, plot_shortwave, target_wavelength_metres,
        output_dir_name):
    """Plots data for one example.

    :param example_dict: See doc for `example_io.read_file`.
    :param example_index: Will plot results for example with this array index.
    :param plot_shortwave: See documentation at top of file.
    :param target_wavelength_metres: Same.
    :param output_dir_name: Same.
    """

    example_id_string = (
        example_dict[example_utils.EXAMPLE_IDS_KEY][example_index]
    )

    if plot_shortwave:
        predictor_names = SHORTWAVE_PREDICTOR_NAMES
        predictor_colours = SHORTWAVE_PREDICTOR_COLOURS
    else:
        predictor_names = LONGWAVE_PREDICTOR_NAMES
        predictor_colours = LONGWAVE_PREDICTOR_COLOURS

    handle_dict = profile_plotting.plot_predictors(
        example_dict=example_dict, example_index=example_index,
        predictor_names=predictor_names, predictor_colours=predictor_colours,
        predictor_line_widths=numpy.full(len(predictor_names), LINE_WIDTH),
        predictor_line_styles=['solid'] * len(predictor_names),
        use_log_scale=True
    )
    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]

    this_file_name = '{0:s}/{1:s}_predictors.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )
    panel_file_names = [this_file_name]

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    handle_dict = profile_plotting.plot_targets(
        example_dict=example_dict,
        example_index=example_index,
        for_shortwave=plot_shortwave,
        wavelength_metres=target_wavelength_metres,
        use_log_scale=True, line_width=LINE_WIDTH, line_style='solid'
    )
    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]

    this_file_name = '{0:s}/{1:s}_targets.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )
    panel_file_names.append(this_file_name)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    num_panels = len(panel_file_names)
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_panels)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_panels) / num_panel_rows
    ))
    concat_file_name = '{0:s}/{1:s}.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )

    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=int(1e7)
    )

    for this_file_name in panel_file_names:
        os.remove(this_file_name)


def _run(example_file_name, num_examples, example_dir_name,
         example_id_file_name, plot_shortwave, target_wavelength_metres,
         output_dir_name):
    """Plots profiles (all target variables and 4 predictor variables).

    This is effectively the main method.

    :param example_file_name: See documentation at top of file.
    :param num_examples: Same.
    :param example_dir_name: Same.
    :param example_id_file_name: Same.
    :param plot_shortwave: Same.
    :param target_wavelength_metres: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    example_dict = misc_utils.get_raw_examples(
        example_file_name=example_file_name,
        num_examples=num_examples,
        example_dir_name=example_dir_name,
        example_id_file_name=example_id_file_name
    )[0]

    num_examples_total = len(example_dict[example_utils.VALID_TIMES_KEY])

    if 0 < num_examples < num_examples_total:
        desired_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )
        example_dict = example_utils.subset_by_index(
            example_dict=example_dict, desired_indices=desired_indices
        )
    else:
        num_examples = num_examples_total + 0

    for i in range(num_examples):
        _plot_one_example(
            example_dict=example_dict,
            example_index=i,
            plot_shortwave=plot_shortwave,
            target_wavelength_metres=target_wavelength_metres,
            output_dir_name=output_dir_name
        )

        print(MINOR_SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        example_id_file_name=getattr(
            INPUT_ARG_OBJECT, EXAMPLE_ID_FILE_ARG_NAME
        ),
        plot_shortwave=bool(getattr(INPUT_ARG_OBJECT, PLOT_SHORTWAVE_ARG_NAME)),
        target_wavelength_metres=getattr(
            INPUT_ARG_OBJECT, TARGET_WAVELENGTH_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
