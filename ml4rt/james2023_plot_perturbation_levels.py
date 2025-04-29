"""For each data example, plots data at different levels of perturbation."""

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
import error_checking
import imagemagick_utils
import example_io
import example_utils
import profile_plotting

MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

CONVERT_EXE_NAME = 'convert'
TITLE_FONT_SIZE = 150
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

LWC_COLOUR = numpy.array([166, 206, 227], dtype=float) / 255
IWC_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
TEMPERATURE_COLOUR = numpy.full(3, 0.)
OZONE_COLOUR = numpy.array([51, 160, 44], dtype=float) / 255
HUMIDITY_COLOUR = numpy.array([251, 154, 153], dtype=float) / 255

PREDICTOR_NAMES = [
    example_utils.LIQUID_WATER_CONTENT_NAME,
    example_utils.ICE_WATER_CONTENT_NAME,
    example_utils.TEMPERATURE_NAME,
    example_utils.O3_MIXING_RATIO_NAME,
    example_utils.SPECIFIC_HUMIDITY_NAME
]

PREDICTOR_COLOURS = [
    LWC_COLOUR,
    IWC_COLOUR,
    TEMPERATURE_COLOUR,
    OZONE_COLOUR,
    HUMIDITY_COLOUR
]

LINE_WIDTH = 3
FIGURE_RESOLUTION_DPI = 300

INPUT_FILES_ARG_NAME = 'input_example_file_names'
FILE_DESCRIPTIONS_ARG_NAME = 'file_description_strings'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'Space-separated list of paths to example files (one per perturbation '
    'level).  Each file will be read by `example_io.read_file`.'
)
FILE_DESCRIPTIONS_HELP_STRING = (
    'Space-separated list of file descriptions, with the same length as '
    '`{0:s}`.  Within each space-separated item, underscores will be replaced '
    'by spaces before use in panel tites.'
).format(
    INPUT_FILES_ARG_NAME
)
NUM_EXAMPLES_HELP_STRING = (
    'Number of data examples to plot.  These will be randomly selected from '
    'examples in which all relevant predictors have been perturbed.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FILE_DESCRIPTIONS_ARG_NAME, type=str, nargs='+', required=True,
    help=FILE_DESCRIPTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _overlay_text(
        image_file_name, x_offset_from_left_px, y_offset_from_top_px,
        text_string):
    """Creates two figures showing overall evaluation of uncertainty quant (UQ).

    :param image_file_name: Path to image file.
    :param x_offset_from_left_px: Left-relative x-coordinate (pixels).
    :param y_offset_from_top_px: Top-relative y-coordinate (pixels).
    :param text_string: String to overlay.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    command_string = (
        '"{0:s}" "{1:s}" -pointsize {2:d} -font "{3:s}" '
        '-fill "rgb(0, 0, 0)" -annotate {4:+d}{5:+d} "{6:s}" "{1:s}"'
    ).format(
        CONVERT_EXE_NAME, image_file_name, TITLE_FONT_SIZE, TITLE_FONT_NAME,
        x_offset_from_left_px, y_offset_from_top_px, text_string
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _plot_one_example(example_dict_by_file, file_description_strings,
                      example_index, output_dir_name):
    """Plots one data example at different levels of perturbation.

    P = number of perturbation levels

    :param example_dict_by_file: length-P list of dictionaries returned by
        `example_io.read_file`.
    :param file_description_strings: length-P list of descriptions.
    :param example_index: Will plot the [k]th example, where k = example_index.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    """

    example_id_string = (
        example_dict_by_file[0][example_utils.EXAMPLE_IDS_KEY][example_index]
    )

    num_files = len(example_dict_by_file)
    panel_file_names = []

    for i in range(num_files):
        handle_dict = profile_plotting.plot_predictors(
            example_dict=example_dict_by_file[i],
            example_index=example_index,
            predictor_names=PREDICTOR_NAMES,
            predictor_colours=PREDICTOR_COLOURS,
            predictor_line_widths=numpy.full(len(PREDICTOR_NAMES), LINE_WIDTH),
            predictor_line_styles=['solid'] * len(PREDICTOR_NAMES),
            use_log_scale=True
        )
        figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
        axes_objects = handle_dict[profile_plotting.AXES_OBJECTS_KEY]

        axes_objects[0].set_title(file_description_strings[i])

        this_file_name = '{0:s}/{1:s}_level{2:d}.jpg'.format(
            output_dir_name, example_id_string.replace('_', '-'), i
        )
        panel_file_names.append(this_file_name)

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)

    panel_letter = None

    for this_file_name in panel_file_names:
        if panel_letter is None:
            panel_letter = 'a'
        else:
            panel_letter = chr(ord(panel_letter) + 1)

        imagemagick_utils.trim_whitespace(
            input_file_name=this_file_name,
            output_file_name=this_file_name,
            border_width_pixels=TITLE_FONT_SIZE + 75
        )
        _overlay_text(
            image_file_name=this_file_name,
            x_offset_from_left_px=TITLE_FONT_SIZE + 50,
            y_offset_from_top_px=TITLE_FONT_SIZE + 200,
            text_string='({0:s})'.format(panel_letter)
        )
        imagemagick_utils.trim_whitespace(
            input_file_name=this_file_name,
            output_file_name=this_file_name
        )

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


def _run(example_file_names, file_description_strings, num_examples,
         output_dir_name):
    """For each data example, plots data at different levels of perturbation.

    This is effectively the main method.

    :param example_file_names: See documentation at top of this script.
    :param file_description_strings: Same.
    :param num_examples: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    error_checking.assert_is_greater(num_examples, 0)

    num_files = len(example_file_names)
    error_checking.assert_is_numpy_array(
        numpy.array(file_description_strings),
        exact_dimensions=numpy.array([num_files], dtype=int)
    )

    file_description_strings = [
        f.replace('_', ' ') for f in file_description_strings
    ]
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read input data.
    example_dict_by_file = [dict()] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(example_file_names[i]))
        example_dict_by_file[i] = example_io.read_file(example_file_names[i])
        example_dict_by_file[i] = example_utils.subset_by_field(
            example_dict=example_dict_by_file[i],
            field_names=PREDICTOR_NAMES
        )

    # Find example IDs that appear in all files.
    example_id_strings = set()

    for i in range(num_files):
        these_id_strings = example_dict_by_file[i][
            example_utils.EXAMPLE_IDS_KEY
        ]
        print(these_id_strings[0])

        these_id_strings = [
            '_'.join(s.split('_')[:-1]) for s in these_id_strings
        ]
        print(these_id_strings[0])

        these_id_strings = [
            '{0:s}_temp-10m-kelvins=300.000000'.format(s)
            for s in these_id_strings
        ]

        print(these_id_strings[0])
        print('\n\n\n')

        if i == 0:
            example_id_strings = set(these_id_strings)
        else:
            example_id_strings = example_id_strings.intersection(these_id_strings)

    example_id_strings = list(example_id_strings)
    num_examples_total = len(example_id_strings)

    if num_examples < num_examples_total:
        desired_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )
        desired_indices = numpy.random.choice(
            desired_indices, size=num_examples, replace=False
        )
        example_id_strings = [example_id_strings[k] for k in desired_indices]

    for i in range(num_files):
        desired_indices = example_utils.find_examples(
            all_id_strings=
            example_dict_by_file[i][example_utils.EXAMPLE_IDS_KEY],
            desired_id_strings=example_id_strings
        )
        example_dict_by_file[i] = example_utils.subset_by_index(
            example_dict=example_dict_by_file[i],
            desired_indices=desired_indices
        )

    num_examples = len(example_dict_by_file[i][example_utils.VALID_TIMES_KEY])

    for i in range(num_examples):
        _plot_one_example(
            example_dict_by_file=example_dict_by_file,
            file_description_strings=file_description_strings,
            example_index=i,
            output_dir_name=output_dir_name
        )
        print(MINOR_SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        file_description_strings=getattr(
            INPUT_ARG_OBJECT, FILE_DESCRIPTIONS_ARG_NAME
        ),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
