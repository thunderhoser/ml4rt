"""Makes animation to explain 1-D multivariate upsampling."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils

INPUT_TEMPERATURE_MATRIX = numpy.array([0, -4, -8])
INPUT_U_WIND_MATRIX = numpy.array([1, 10, 12])
INPUT_V_WIND_MATRIX = numpy.array([5, 5, 5])

INPUT_TEMPERATURE_MATRIX = numpy.expand_dims(INPUT_TEMPERATURE_MATRIX, axis=-1)
INPUT_U_WIND_MATRIX = numpy.expand_dims(INPUT_U_WIND_MATRIX, axis=-1)
INPUT_V_WIND_MATRIX = numpy.expand_dims(INPUT_V_WIND_MATRIX, axis=-1)
INPUT_FEATURE_MATRIX = numpy.stack(
    (INPUT_TEMPERATURE_MATRIX, INPUT_U_WIND_MATRIX, INPUT_V_WIND_MATRIX),
    axis=-1
).astype(float)

OUTPUT_TEMPERATURE_MATRIX = numpy.array([0, 0, -4, -4, -8, -8])
OUTPUT_U_WIND_MATRIX = numpy.array([1, 1, 10, 10, 12, 12])
OUTPUT_V_WIND_MATRIX = numpy.array([5, 5, 5, 5, 5, 5])

OUTPUT_TEMPERATURE_MATRIX = numpy.expand_dims(
    OUTPUT_TEMPERATURE_MATRIX, axis=-1
)
OUTPUT_U_WIND_MATRIX = numpy.expand_dims(OUTPUT_U_WIND_MATRIX, axis=-1)
OUTPUT_V_WIND_MATRIX = numpy.expand_dims(OUTPUT_V_WIND_MATRIX, axis=-1)
OUTPUT_FEATURE_MATRIX = numpy.stack(
    (OUTPUT_TEMPERATURE_MATRIX, OUTPUT_U_WIND_MATRIX, OUTPUT_V_WIND_MATRIX),
    axis=-1
).astype(float)

HORIZ_PANEL_SPACING = 0.2
VERTICAL_PANEL_SPACING = 0.
CAPTION_Y_COORD = 0.

DEFAULT_FONT_SIZE = 25
DEFAULT_LINE_WIDTH = 4

pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', linewidth=DEFAULT_LINE_WIDTH)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

COLOUR_LIST = [
    numpy.full(3, 1.),
    numpy.array([27, 158, 119], dtype=float) / 255
]

COLOUR_LIST[1] = matplotlib.colors.to_rgba(COLOUR_LIST[1], 0.5)
COLOUR_MAP_OBJECT = matplotlib.colors.ListedColormap(COLOUR_LIST)

DEFAULT_FONT_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
SPECIAL_FONT_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255

PANEL_LETTER_FONT_SIZE = 30
INTERPANEL_LINE_WIDTH = 2
INTERPANEL_LINE_COLOUR = numpy.full(3, 152. / 255)

NUM_PANEL_COLUMNS = 2
FIGURE_RESOLUTION_DPI = 200

FIGURE_CAPTION = (
    'Multivariate upsampling with one spatial dimension –\n'
    'in this case, using nearest-neighbour interpolation\n'
    '(NNI) and upsampling factor of 2.\n'
    '[a-c] Inputs.  Each panel is one feature map or\n'
    'one channel.\n'
    '[d-f] Outputs.  Each panel is one feature map or\n'
    'one channel.\n'
    'At each position of the upsampling window, the values\n'
    'highlighted in (d, e, f) are produced by NNI from that\n'
    'highlighted in (a, b, c), respectively.'
)

INCLUDE_CAPTION_ARG_NAME = 'include_caption'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INCLUDE_CAPTION_HELP_STRING = (
    'Boolean flag.  If 1, will include caption in each frame of animation.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory (individual frames and full GIF will be saved '
    'here).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INCLUDE_CAPTION_ARG_NAME, type=int, required=False, default=1,
    help=INCLUDE_CAPTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_feature_map(feature_matrix_2d, pooled_row, pooled_column, pooled,
                      axes_object):
    """Plots one feature map.

    M = number of rows in grid
    N = number of columns in grid

    :param feature_matrix_2d: M-by-N numpy array of feature values.
    :param pooled_row: Row index after pooling.
    :param pooled_column: Column index after pooling.
    :param pooled: Boolean flag.  If True (False), plotting (un)pooled feature
        matrix.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    num_rows = feature_matrix_2d.shape[0]
    num_columns = feature_matrix_2d.shape[1]

    if pooled:
        first_highlighted_row = pooled_row + 0
        last_highlighted_row = pooled_row + 0
        first_highlighted_column = pooled_column + 0
        last_highlighted_column = pooled_column + 0
    else:
        first_highlighted_row = pooled_row * 2
        last_highlighted_row = first_highlighted_row + 1
        first_highlighted_column = pooled_column * 2
        last_highlighted_column = first_highlighted_column + 1

    dummy_matrix = numpy.full(feature_matrix_2d.shape, numpy.nan)
    dummy_matrix[
        first_highlighted_row:(last_highlighted_row + 1),
        first_highlighted_column:(last_highlighted_column + 1)
    ] = 0

    axes_object.imshow(
        numpy.ma.masked_where(numpy.isnan(dummy_matrix), dummy_matrix),
        cmap=COLOUR_MAP_OBJECT, vmin=-1., vmax=0., origin='upper'
    )

    axes_object.set_xticks([], [])
    axes_object.set_yticks([], [])

    for i in range(num_rows):
        for j in range(num_columns):
            if i == pooled_row and j == pooled_column and pooled:
                this_colour = SPECIAL_FONT_COLOUR
            else:
                this_colour = DEFAULT_FONT_COLOUR

            this_label_string = '{0:d}'.format(
                int(numpy.round(feature_matrix_2d[i, j]))
            )
            axes_object.text(
                j, i, this_label_string, fontsize=DEFAULT_FONT_SIZE,
                color=this_colour,
                horizontalalignment='center', verticalalignment='center'
            )


def _plot_interpanel_lines(
        pooled_row, pooled_column, input_fm_axes_object, output_fm_axes_object):
    """Plots lines between input and output feature map.

    :param pooled_row: See doc for `_plot_feature_map`.
    :param pooled_column: Same.
    :param input_fm_axes_object: Axes on which input feature map is plotted
        (instance of `matplotlib.axes._subplots.AxesSubplot`).
    :param output_fm_axes_object: Axes on which output feature map is plotted
        (instance of `matplotlib.axes._subplots.AxesSubplot`).
    """

    first_output_row = pooled_row * 2 - 0.5
    last_output_row = first_output_row + 2
    first_output_column = pooled_column - 0.5
    last_output_column = first_output_column + 1

    this_connection_object = ConnectionPatch(
        xyA=(last_output_column, first_output_row),
        xyB=(pooled_column + 0.15, pooled_row),
        coordsA='data', coordsB='data',
        axesA=output_fm_axes_object, axesB=input_fm_axes_object,
        color=INTERPANEL_LINE_COLOUR, linewidth=INTERPANEL_LINE_WIDTH,
        linestyle='dashed'
    )

    output_fm_axes_object.add_artist(this_connection_object)

    this_connection_object = ConnectionPatch(
        xyA=(last_output_column, last_output_row),
        xyB=(pooled_column + 0.15, pooled_row),
        coordsA='data', coordsB='data',
        axesA=output_fm_axes_object, axesB=input_fm_axes_object,
        color=INTERPANEL_LINE_COLOUR, linewidth=INTERPANEL_LINE_WIDTH,
        linestyle='dashed'
    )

    output_fm_axes_object.add_artist(this_connection_object)


def _run(include_caption, output_dir_name):
    """Makes animation to explain pooling.

    This is effectively the main method.

    :param include_caption: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    num_input_rows = INPUT_FEATURE_MATRIX.shape[0]
    num_input_columns = INPUT_FEATURE_MATRIX.shape[1]
    num_channels = INPUT_FEATURE_MATRIX.shape[2]
    image_file_names = []

    for i in range(num_input_rows):
        for j in range(num_input_columns):
            this_figure_object, this_axes_object_matrix = (
                plotting_utils.create_paneled_figure(
                    num_rows=num_channels, num_columns=NUM_PANEL_COLUMNS,
                    horizontal_spacing=HORIZ_PANEL_SPACING,
                    vertical_spacing=VERTICAL_PANEL_SPACING,
                    shared_x_axis=False, shared_y_axis=False,
                    keep_aspect_ratio=True
                )
            )

            letter_label = None

            for k in range(num_channels):
                _plot_feature_map(
                    feature_matrix_2d=INPUT_FEATURE_MATRIX[..., k],
                    pooled_row=i, pooled_column=j, pooled=True,
                    axes_object=this_axes_object_matrix[k, 0]
                )

                if letter_label is None:
                    letter_label = 'a'
                else:
                    letter_label = chr(ord(letter_label) + 1)

                plotting_utils.label_axes(
                    axes_object=this_axes_object_matrix[k, 0],
                    label_string='({0:s})'.format(letter_label),
                    font_size=PANEL_LETTER_FONT_SIZE,
                    y_coord_normalized=0.85, x_coord_normalized=-0.02
                )

            for k in range(num_channels):
                _plot_feature_map(
                    feature_matrix_2d=OUTPUT_FEATURE_MATRIX[..., k],
                    pooled_row=i, pooled_column=j, pooled=False,
                    axes_object=this_axes_object_matrix[k, 1]
                )

                letter_label = chr(ord(letter_label) + 1)

                plotting_utils.label_axes(
                    axes_object=this_axes_object_matrix[k, 1],
                    label_string='({0:s})'.format(letter_label),
                    font_size=PANEL_LETTER_FONT_SIZE,
                    y_coord_normalized=0.85, x_coord_normalized=-0.02
                )

                _plot_interpanel_lines(
                    pooled_row=i, pooled_column=j,
                    input_fm_axes_object=this_axes_object_matrix[k, 0],
                    output_fm_axes_object=this_axes_object_matrix[k, 1]
                )

            if include_caption:
                this_figure_object.text(
                    0.5, CAPTION_Y_COORD, FIGURE_CAPTION,
                    fontsize=DEFAULT_FONT_SIZE, color='k',
                    horizontalalignment='center', verticalalignment='top'
                )

            image_file_names.append(
                '{0:s}/upsampling_animation_row{1:d}_column{2:d}.jpg'.format(
                    output_dir_name, i, j
                )
            )

            print('Saving figure to: "{0:s}"...'.format(image_file_names[-1]))
            this_figure_object.savefig(
                image_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(this_figure_object)

    animation_file_name = '{0:s}/upsampling_animation.gif'.format(
        output_dir_name
    )
    print('Creating animation: "{0:s}"...'.format(animation_file_name))

    imagemagick_utils.create_gif(
        input_file_names=image_file_names, output_file_name=animation_file_name,
        num_seconds_per_frame=0.5, resize_factor=0.5
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        include_caption=bool(getattr(
            INPUT_ARG_OBJECT, INCLUDE_CAPTION_ARG_NAME
        )),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
