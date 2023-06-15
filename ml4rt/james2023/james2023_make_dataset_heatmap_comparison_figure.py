"""Compares one field across datasets, using heat maps to show 2-D prob dist."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import temperature_conversions as temperature_conv
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import imagemagick_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4rt.io import example_io
from ml4rt.utils import example_utils
from ml4rt.plotting import profile_plotting

METRES_TO_KM = 0.001

DUMMY_FIRST_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '1900-01-01', '%Y-%m-%d'
)
DUMMY_LAST_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2100-01-01', '%Y-%m-%d'
)

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 250
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

COLOUR_MAP_OBJECT = pyplot.get_cmap('viridis')

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300
PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

DATASET_DIRS_ARG_NAME = 'input_dataset_dir_names'
DATASET_DESCRIPTIONS_ARG_NAME = 'dataset_description_strings'
FIELD_ARG_NAME = 'field_name'
MIN_VALUE_ARG_NAME = 'min_value_for_field'
MAX_VALUE_ARG_NAME = 'max_value_for_field'
NUM_BINS_ARG_NAME = 'num_bins_for_field'
NUM_PANEL_ROWS_ARG_NAME = 'num_panel_rows'
FIRST_PANEL_LETTER_ARG_NAME = 'first_panel_letter'
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
FIELD_HELP_STRING = (
    'Name of field to compare.  This must be a vector (i.e., defined at '
    'every height), not a scalar.'
)
MIN_VALUE_HELP_STRING = (
    'Minimum value for the given field.  This will be the bottom edge of the '
    'lowest bin.'
)
MAX_VALUE_HELP_STRING = (
    'Max value for the given field.  This will be the top edge of the highest '
    'bin.'
)
NUM_BINS_HELP_STRING = (
    'Number of bins into which the given field is discretized.'
)
NUM_PANEL_ROWS_HELP_STRING = (
    'Number of rows in final concatenated figure (with one panel per dataset).'
)
FIRST_PANEL_LETTER_HELP_STRING = 'Letter used to label first panel.'
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
    '--' + FIELD_ARG_NAME, type=str, required=True, help=FIELD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_VALUE_ARG_NAME, type=float, required=True,
    help=MIN_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_VALUE_ARG_NAME, type=float, required=True,
    help=MAX_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BINS_ARG_NAME, type=int, required=True,
    help=NUM_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PANEL_ROWS_ARG_NAME, type=int, required=True,
    help=NUM_PANEL_ROWS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_PANEL_LETTER_ARG_NAME, type=str, required=False, default='a',
    help=FIRST_PANEL_LETTER_HELP_STRING
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


def _plot_heat_map_one_dataset(
        dataset_dir_name, field_name, min_value_for_field, max_value_for_field,
        num_bins_for_field):
    """Plot 2-D heat map of probabilities for one dataset.

    :param dataset_dir_name: Name of input directory.  Files therein will be
        found by `example_io.find_file` and read by `example_io.read_file`.
    :param field_name: See documentation at top of file.
    :param min_value_for_field: Same.
    :param max_value_for_field: Same.
    :param num_bins_for_field: Same.
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

    bin_edges_for_field = numpy.linspace(
        min_value_for_field, max_value_for_field, num=num_bins_for_field + 1,
        dtype=float
    )
    bin_centers_for_field = (
        0.5 * (bin_edges_for_field[:-1] + bin_edges_for_field[1:])
    )

    heights_m_agl = example_dict[example_utils.HEIGHTS_KEY]
    num_heights = len(heights_m_agl)
    frequency_matrix = numpy.full((num_heights, num_bins_for_field), numpy.nan)

    for i in range(num_heights):
        for j in range(num_bins_for_field):
            if j == 0:
                these_flags = data_matrix[:, i] < bin_edges_for_field[j + 1]
            elif j == num_bins_for_field - 1:
                these_flags = data_matrix[:, i] >= bin_edges_for_field[j]
            else:
                these_flags = numpy.logical_and(
                    data_matrix[:, i] >= bin_edges_for_field[j],
                    data_matrix[:, i] < bin_edges_for_field[j + 1]
                )

            frequency_matrix[i, j] = numpy.mean(these_flags)

    frequency_matrix_log10 = numpy.log10(frequency_matrix)
    frequency_matrix_log10[numpy.isinf(frequency_matrix_log10)] = numpy.nan
    heights_km_agl = METRES_TO_KM * heights_m_agl

    if field_name in [
            example_utils.TEMPERATURE_NAME,
            example_utils.SURFACE_TEMPERATURE_NAME,
            example_utils.DEWPOINT_NAME
    ]:
        bin_centers_plotting_units = temperature_conv.kelvins_to_celsius(
            bin_centers_for_field
        )
    else:
        bin_centers_plotting_units = (
            profile_plotting.PREDICTOR_NAME_TO_CONV_FACTOR[field_name] *
            bin_centers_for_field
        )

    # colour_norm_object = pyplot.Normalize(vmin=-5., vmax=-1.)
    # frequency_matrix_log10[frequency_matrix_log10 < -5] = numpy.nan

    colour_norm_object = pyplot.Normalize(vmin=-6., vmax=-1.)
    frequency_matrix_log10[frequency_matrix_log10 < -6] = numpy.nan

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.imshow(
        frequency_matrix_log10, cmap=COLOUR_MAP_OBJECT, norm=colour_norm_object,
        origin='lower'
    )

    x_tick_values = numpy.linspace(
        0, num_bins_for_field - 1, num=11, dtype=float
    )
    x_tick_values = numpy.unique(
        numpy.round(x_tick_values).astype(int)
    )
    x_tick_labels = [
        '{0:.2f}'.format(c) for c in bin_centers_plotting_units[x_tick_values]
    ]
    pyplot.xticks(x_tick_values, x_tick_labels, rotation=90)

    y_tick_values = numpy.linspace(
        0, num_heights - 1, num=11, dtype=float
    )
    y_tick_values = numpy.unique(
        numpy.round(y_tick_values).astype(int)
    )
    y_tick_labels = [
        '{0:.2f}'.format(h) for h in heights_km_agl[y_tick_values]
    ]
    pyplot.yticks(y_tick_values, y_tick_labels)

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=frequency_matrix_log10[
            numpy.invert(numpy.isnan(frequency_matrix_log10))
        ],
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=True, extend_max=True,
        fraction_of_axis_length=1.
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    tick_strings = [r'10$^{' + s + r'}$' for s in tick_strings]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)
    colour_bar_object.set_label('Frequency')

    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_xlabel(
        profile_plotting.PREDICTOR_NAME_TO_VERBOSE[field_name]
    )

    return figure_object, axes_object


def _run(dataset_dir_names, dataset_description_strings, field_name,
         min_value_for_field, max_value_for_field, num_bins_for_field,
         num_panel_rows, first_panel_letter, output_dir_name):
    """Compares one field across datasets, using heat maps to show 2-D dist.

    This is effectively the main method.

    :param dataset_dir_names: See documentation at top of file.
    :param dataset_description_strings: Same.
    :param field_name: Same.
    :param min_value_for_field: Same.
    :param max_value_for_field: Same.
    :param num_bins_for_field: Same.
    :param num_panel_rows: Same.
    :param first_panel_letter: Same.
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

    error_checking.assert_is_greater(max_value_for_field, min_value_for_field)
    error_checking.assert_is_geq(num_bins_for_field, 10)
    error_checking.assert_equals(len(first_panel_letter), 1)

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Do actual stuff.
    panel_file_names = [''] * num_datasets
    letter_label = chr(ord(first_panel_letter) - 1)

    for i in range(num_datasets):
        figure_object, axes_object = _plot_heat_map_one_dataset(
            dataset_dir_name=dataset_dir_names[i],
            field_name=field_name,
            min_value_for_field=min_value_for_field,
            max_value_for_field=max_value_for_field,
            num_bins_for_field=num_bins_for_field
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
            border_width_pixels=159
        )
        _overlay_text(
            image_file_name=panel_file_names[i],
            x_offset_from_left_px=0,
            y_offset_from_top_px=TITLE_FONT_SIZE + 100,
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
        field_name=getattr(INPUT_ARG_OBJECT, FIELD_ARG_NAME),
        min_value_for_field=getattr(INPUT_ARG_OBJECT, MIN_VALUE_ARG_NAME),
        max_value_for_field=getattr(INPUT_ARG_OBJECT, MAX_VALUE_ARG_NAME),
        num_bins_for_field=getattr(INPUT_ARG_OBJECT, NUM_BINS_ARG_NAME),
        num_panel_rows=getattr(INPUT_ARG_OBJECT, NUM_PANEL_ROWS_ARG_NAME),
        first_panel_letter=getattr(
            INPUT_ARG_OBJECT, FIRST_PANEL_LETTER_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
