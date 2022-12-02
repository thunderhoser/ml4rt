"""Plots profiles (vector predictor and target variables) for each example."""

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
import neural_net
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

FIRST_PREDICTOR_NAMES = [
    example_utils.SPECIFIC_HUMIDITY_NAME,
    example_utils.WATER_VAPOUR_PATH_NAME,
    example_utils.UPWARD_WATER_VAPOUR_PATH_NAME,
    example_utils.RELATIVE_HUMIDITY_NAME
]
FIRST_PREDICTOR_COLOURS = [
    ORANGE_COLOUR, PURPLE_COLOUR, GREEN_COLOUR, BLACK_COLOUR
]

SECOND_PREDICTOR_NAMES = [
    example_utils.LIQUID_WATER_CONTENT_NAME,
    example_utils.LIQUID_WATER_PATH_NAME,
    example_utils.UPWARD_LIQUID_WATER_PATH_NAME,
    example_utils.LIQUID_EFF_RADIUS_NAME
]
SECOND_PREDICTOR_COLOURS = [
    ORANGE_COLOUR, PURPLE_COLOUR, GREEN_COLOUR, BLACK_COLOUR
]

THIRD_PREDICTOR_NAMES = [
    example_utils.ICE_WATER_CONTENT_NAME,
    example_utils.ICE_WATER_PATH_NAME,
    example_utils.UPWARD_ICE_WATER_PATH_NAME,
    example_utils.ICE_EFF_RADIUS_NAME
]
THIRD_PREDICTOR_COLOURS = [
    ORANGE_COLOUR, PURPLE_COLOUR, GREEN_COLOUR, BLACK_COLOUR
]

FOURTH_PREDICTOR_NAMES = [
    example_utils.TEMPERATURE_NAME,
    example_utils.HEIGHT_THICKNESS_NAME,
    example_utils.PRESSURE_THICKNESS_NAME,
    example_utils.PRESSURE_NAME
]
FOURTH_PREDICTOR_COLOURS = [
    ORANGE_COLOUR, PURPLE_COLOUR, GREEN_COLOUR, BLACK_COLOUR
]

FIFTH_PREDICTOR_NAMES = [
    example_utils.CO2_CONCENTRATION_NAME,
    example_utils.N2O_CONCENTRATION_NAME,
    example_utils.CH4_CONCENTRATION_NAME,
    example_utils.O3_MIXING_RATIO_NAME
]
FIFTH_PREDICTOR_COLOURS = [
    ORANGE_COLOUR, PURPLE_COLOUR, GREEN_COLOUR, BLACK_COLOUR
]

SIXTH_PREDICTOR_NAMES_LONGWAVE = [
    example_utils.ZENITH_ANGLE_NAME,
    example_utils.SURFACE_TEMPERATURE_NAME,
    example_utils.SURFACE_EMISSIVITY_NAME
]
SIXTH_PREDICTOR_NAMES_SHORTWAVE = [
    example_utils.ZENITH_ANGLE_NAME, example_utils.ALBEDO_NAME,
    example_utils.AEROSOL_ALBEDO_NAME,
    example_utils.AEROSOL_ASYMMETRY_PARAM_NAME
]
SIXTH_PREDICTOR_COLOURS_LONGWAVE = [
    ORANGE_COLOUR, PURPLE_COLOUR, GREEN_COLOUR
]
SIXTH_PREDICTOR_COLOURS_SHORTWAVE = [
    ORANGE_COLOUR, PURPLE_COLOUR, GREEN_COLOUR, BLACK_COLOUR
]

LINE_WIDTH = 3
FIGURE_RESOLUTION_DPI = 300

EXAMPLE_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_FILE_ARG_NAME
NUM_EXAMPLES_ARG_NAME = make_saliency_maps.NUM_EXAMPLES_ARG_NAME
EXAMPLE_DIR_ARG_NAME = make_saliency_maps.EXAMPLE_DIR_ARG_NAME
EXAMPLE_ID_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_ID_FILE_ARG_NAME
USE_LOG_SCALE_ARG_NAME = 'use_log_scale'
MODEL_FILE_ARG_NAME = 'model_file_name'
PLOT_SHORTWAVE_ARG_NAME = 'plot_shortwave'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXAMPLE_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_FILE_HELP_STRING
NUM_EXAMPLES_HELP_STRING = make_saliency_maps.NUM_EXAMPLES_HELP_STRING
EXAMPLE_DIR_HELP_STRING = make_saliency_maps.EXAMPLE_DIR_HELP_STRING
EXAMPLE_ID_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_ID_FILE_HELP_STRING
USE_LOG_SCALE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will use logarithmic (linear) scale for height '
    'axis.'
)
MODEL_FILE_HELP_STRING = (
    '[optional] Path to model (readable by `neural_net.read_model`).  If '
    'specified, this script will plot only the variables/heights used by the '
    'model.'
)
PLOT_SHORTWAVE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot shortwave (longwave) variables.'
)
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
    '--' + USE_LOG_SCALE_ARG_NAME, type=int, required=False, default=1,
    help=USE_LOG_SCALE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=False, default='',
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SHORTWAVE_ARG_NAME, type=int, required=True,
    help=PLOT_SHORTWAVE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_example(
        example_dict, example_index, use_log_scale, plot_shortwave,
        output_dir_name):
    """Plots data for one example.

    :param example_dict: See doc for `example_io.read_file`.
    :param example_index: Will plot results for example with this array index.
    :param use_log_scale: See documentation at top of file.
    :param plot_shortwave: Same.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    """

    example_id_string = (
        example_dict[example_utils.EXAMPLE_IDS_KEY][example_index]
    )

    predictor_names_by_set = [
        FIRST_PREDICTOR_NAMES, SECOND_PREDICTOR_NAMES, THIRD_PREDICTOR_NAMES,
        FOURTH_PREDICTOR_NAMES, FIFTH_PREDICTOR_NAMES
    ]
    predictor_colours_by_set = [
        FIRST_PREDICTOR_COLOURS, SECOND_PREDICTOR_COLOURS,
        THIRD_PREDICTOR_COLOURS,
        FOURTH_PREDICTOR_COLOURS, FIFTH_PREDICTOR_COLOURS
    ]

    if plot_shortwave:
        predictor_names_by_set.append(SIXTH_PREDICTOR_NAMES_SHORTWAVE)
        predictor_colours_by_set.append(SIXTH_PREDICTOR_COLOURS_SHORTWAVE)
    else:
        predictor_names_by_set.append(SIXTH_PREDICTOR_NAMES_LONGWAVE)
        predictor_colours_by_set.append(SIXTH_PREDICTOR_COLOURS_LONGWAVE)

    num_predictor_sets = len(predictor_names_by_set)
    panel_file_names = []

    for k in range(num_predictor_sets):
        print(predictor_names_by_set)
        print('\n\n\n')

        handle_dict = profile_plotting.plot_predictors(
            example_dict=example_dict, example_index=example_index,
            predictor_names=predictor_names_by_set[k],
            predictor_colours=predictor_colours_by_set[k],
            predictor_line_widths=
            numpy.full(len(predictor_names_by_set[k]), LINE_WIDTH),
            predictor_line_styles=['solid'] * len(predictor_names_by_set[k]),
            use_log_scale=use_log_scale
        )
        figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]

        this_file_name = '{0:s}/{1:s}_predictor-set-{2:d}.jpg'.format(
            output_dir_name, example_id_string.replace('_', '-'), k
        )
        panel_file_names.append(this_file_name)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)

    handle_dict = profile_plotting.plot_targets(
        example_dict=example_dict, example_index=example_index,
        for_shortwave=plot_shortwave, use_log_scale=use_log_scale,
        line_width=LINE_WIDTH, line_style='solid'
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
         example_id_file_name, use_log_scale, model_file_name, plot_shortwave,
         output_dir_name):
    """Plots profiles (vector predictor and target variables) for each example.

    This is effectively the main method.

    :param example_file_name: See documentation at top of file.
    :param num_examples: Same.
    :param example_dir_name: Same.
    :param example_id_file_name: Same.
    :param use_log_scale: Same.
    :param model_file_name: Same.
    :param plot_shortwave: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    example_dict = misc_utils.get_raw_examples(
        example_file_name=example_file_name, num_examples=num_examples,
        example_dir_name=example_dir_name,
        example_id_file_name=example_id_file_name
    )

    num_examples_total = len(example_dict[example_utils.VALID_TIMES_KEY])

    if 0 < num_examples < num_examples_total:
        desired_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )
        example_dict = example_utils.subset_by_index(
            example_dict=example_dict, desired_indices=desired_indices
        )

    if model_file_name != '':
        model_metafile_name = neural_net.find_metafile(
            os.path.split(model_file_name)[0]
        )

        print('Reading model metadata from: "{0:s}"...\n'.format(
            model_metafile_name
        ))
        model_metadata_dict = neural_net.read_metafile(model_metafile_name)

        generator_option_dict = (
            model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
        )
        vector_predictor_names = (
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY]
        )
        all_field_names = vector_predictor_names + (
            SHORTWAVE_TARGET_NAMES if plot_shortwave
            else LONGWAVE_TARGET_NAMES
        )

        example_dict = example_utils.subset_by_field(
            example_dict=example_dict, field_names=all_field_names
        )
        example_dict = example_utils.subset_by_height(
            example_dict=example_dict,
            heights_m_agl=generator_option_dict[neural_net.HEIGHTS_KEY]
        )

    num_examples = len(example_dict[example_utils.VALID_TIMES_KEY])

    for i in range(num_examples):
        _plot_one_example(
            example_dict=example_dict, example_index=i,
            use_log_scale=use_log_scale, plot_shortwave=plot_shortwave,
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
        use_log_scale=bool(getattr(INPUT_ARG_OBJECT, USE_LOG_SCALE_ARG_NAME)),
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        plot_shortwave=bool(getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
