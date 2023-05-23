"""Compares one field across datasets, using violin plots."""

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

import time_conversion
import temperature_conversions as temperature_conv
import file_system_utils
import error_checking
import imagemagick_utils
import example_io
import example_utils
import profile_plotting

# TODO(thunderhoser): Specify min and max to plot for each field

METRES_TO_KM = 0.001

DUMMY_FIRST_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '1900-01-01', '%Y-%m-%d'
)
DUMMY_LAST_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2100-01-01', '%Y-%m-%d'
)

VIOLIN_LINE_COLOUR = numpy.full(3, 0.)

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 100
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

DATASET_DIRS_ARG_NAME = 'input_dataset_dir_names'
DATASET_DESCRIPTIONS_ARG_NAME = 'dataset_description_strings'
DATASET_COLOURS_ARG_NAME = 'dataset_colours'
FIELD_ARG_NAME = 'field_name'
NUM_PANEL_ROWS_ARG_NAME = 'num_panel_rows'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

DATASET_DIRS_HELP_STRING = (
    'List of directory names, one for each dataset.  Within each directory, '
    'files will be found `example_io.find_file` and read by '
    '`example_io.read_file`.'
)
DATASET_DESCRIPTIONS_HELP_STRING = (
    'List of descriptive strings, one for each dataset.  This should be a '
    'space-separated list.  Within each item, underscores will be replaced by '
    'spaces.  Example: "perturbed_training clean_training" will be interpreted '
    'as a 2-element list: ["perturbed training", "clean training"].'
)
DATASET_COLOURS_HELP_STRING = (
    'List of colours, one for each dataset.  This should be a space-separated '
    'list, and each item should be an underscore-separated list of [R, G, B] '
    'values ranging from 0...255.  For example, if you want red and black: '
    '"255_0_0 0_0_0"'
)
FIELD_HELP_STRING = (
    'Name of field to compare.  This must be a vector (i.e., defined at '
    'every height), not a scalar.'
)
NUM_PANEL_ROWS_HELP_STRING = (
    'Number of rows in final concatenated figure (with one panel per dataset).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + DATASET_DIRS_ARG_NAME, type=str, nargs='+', required=True,
    help=DATASET_DIRS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DATASET_DESCRIPTIONS_ARG_NAME, type=str, nargs='+', required=True,
    help=DATASET_DESCRIPTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DATASET_COLOURS_ARG_NAME, type=str, nargs='+', required=True,
    help=DATASET_COLOURS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIELD_ARG_NAME, type=str, required=True, help=FIELD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=True,
    help=NUM_PANEL_ROWS_HELP_STRING
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


def _make_violin_plot_one_dataset(
        dataset_dir_name, dataset_colour, field_name):
    """Creates multi-height violin plot for one dataset.

    :param dataset_dir_name: Name of input directory.  Files therein will be
        found by `example_io.find_file` and read by `example_io.read_file`.
    :param dataset_colour: Colour for this dataset, as a numpy array with
        [R, G, B] values ranging from 0...1.
    :param field_name: Name of field.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    example_file_names = example_io.find_many_files(
        directory_name=dataset_dir_name,
        first_time_unix_sec=DUMMY_FIRST_TIME_UNIX_SEC,
        last_time_unix_sec=DUMMY_LAST_TIME_UNIX_SEC,
        raise_error_if_any_missing=False, raise_error_if_all_missing=True
    )

    num_files = len(example_file_names)
    example_dicts = [dict()] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(example_file_names[i]))
        example_dicts[i] = example_io.read_file(
            netcdf_file_name=example_file_names[i]
        )
        example_dicts[i] = example_utils.subset_by_field(
            example_dict=example_dicts[i], field_names=[field_name]
        )

    example_dict = example_utils.concat_examples(example_dicts)
    del example_dicts

    data_matrix = example_utils.get_field_from_dict(
        example_dict=example_dict, field_name=field_name
    )
    assert len(data_matrix.shape) == 2

    heights_m_agl = example_dict[example_utils.HEIGHTS_KEY]
    heights_km_agl = METRES_TO_KM * heights_m_agl
    num_heights = len(heights_m_agl)
    height_indices = 0.5 + numpy.linspace(
        0, num_heights - 1, num=num_heights, dtype=float
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    if field_name in [
            example_utils.TEMPERATURE_NAME,
            example_utils.SURFACE_TEMPERATURE_NAME,
            example_utils.DEWPOINT_NAME
    ]:
        data_matrix = temperature_conv.kelvins_to_celsius(
            data_matrix
        )
    else:
        data_matrix = (
            profile_plotting.PREDICTOR_NAME_TO_CONV_FACTOR[field_name] *
            data_matrix
        )

    violin_handles = axes_object.violinplot(
        numpy.transpose(data_matrix), positions=height_indices,
        vert=False, widths=1.,
        showmeans=True, showmedians=True, showextrema=True
    )

    for part_name in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
        try:
            this_handle = violin_handles[part_name]
        except:
            continue

        this_handle.set_edgecolor(VIOLIN_LINE_COLOUR)
        this_handle.set_linewidth(1)

    for this_handle in violin_handles['bodies']:
        this_handle.set_facecolor(dataset_colour)
        this_handle.set_edgecolor(dataset_colour)
        # this_handle.set_linewidth(0)

    y_tick_indices = axes_object.get_yticks()
    axes_object.set_yticklabels(
        y_tick_indices,
        ['{0:.2g}'.format(h) for h in heights_km_agl[y_tick_indices]]
    )

    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_xlabel(
        profile_plotting.PREDICTOR_NAME_TO_CONV_FACTOR[field_name]
    )

    return figure_object, axes_object


def _run(dataset_dir_names, dataset_description_strings, dataset_colours,
         field_name, num_panel_rows, output_dir_name):
    """Compares one field across datasets, using violin plots.

    This is effectively the main method.

    :param dataset_dir_names: See documentation at top of file.
    :param dataset_description_strings: Same.
    :param dataset_colours: Same.
    :param field_name: Same.
    :param num_panel_rows: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    num_datasets = len(dataset_dir_names)
    error_checking.assert_is_greater(num_datasets, 1)

    expected_dim = numpy.array([num_datasets], dtype=int)
    error_checking.assert_is_numpy_array(
        numpy.array(dataset_description_strings), exact_dimensions=expected_dim
    )
    dataset_description_strings = [
        s.replace('_', ' ') for s in dataset_description_strings
    ]

    error_checking.assert_is_numpy_array(
        numpy.array(dataset_colours), exact_dimensions=expected_dim
    )
    dataset_colours = [
        numpy.array([int(x) for x in c.split('_')], dtype=int)
        for c in dataset_colours
    ]
    dataset_colours = [c.astype(float) / 255 for c in dataset_colours]

    for this_colour in dataset_colours:
        error_checking.assert_is_geq_numpy_array(this_colour, 0.)
        error_checking.assert_is_leq_numpy_array(this_colour, 1.)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    panel_file_names = [''] * num_datasets
    letter_label = None

    for i in range(num_datasets):
        figure_object, axes_object = _make_violin_plot_one_dataset(
            dataset_dir_name=dataset_dir_names[i],
            dataset_colour=dataset_colours[i],
            field_name=field_name
        )
        axes_object.set_title(dataset_description_strings[i])

        panel_file_names[i] = '{0:s}/{1:s}.jpg'.format(
            output_dir_name,
            dataset_description_strings[i].replace(' ', '_')
        )

        print('Saving figure to: "{0:s}"...'.format(panel_file_names[i]))
        figure_object.savefig(
            panel_file_names[i], dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[i],
            output_file_name=panel_file_names[i],
            border_width_pixels=0
        )
        _overlay_text(
            image_file_name=panel_file_names[i],
            x_offset_from_left_px=TITLE_FONT_SIZE,
            y_offset_from_top_px=TITLE_FONT_SIZE,
            text_string='({0:s})'.format(letter_label)
        )

        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[i],
            output_file_name=panel_file_names[i],
            border_width_pixels=0
        )
        imagemagick_utils.resize_image(
            input_file_name=panel_file_names[i],
            output_file_name=panel_file_names[i],
            output_size_pixels=int(PANEL_SIZE_PX)
        )

    num_panel_columns = int(numpy.ceil(
        float(num_datasets) / num_panel_rows
    ))

    concat_figure_file_name = '{0:s}/all_datasets.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns,
        border_width_pixels=25
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        border_width_pixels=0
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        dataset_dir_names=getattr(INPUT_ARG_OBJECT, DATASET_DIRS_ARG_NAME),
        dataset_description_strings=getattr(
            INPUT_ARG_OBJECT, DATASET_DESCRIPTIONS_ARG_NAME
        ),
        dataset_colours=getattr(INPUT_ARG_OBJECT, DATASET_COLOURS_ARG_NAME),
        field_name=getattr(INPUT_ARG_OBJECT, FIELD_ARG_NAME),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
