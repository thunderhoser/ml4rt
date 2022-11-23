"""Plots number of weights vs. hyperparameters for longwave models."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils

NN_TYPE_STRINGS = [
    'plusplus', 'plusplus_deep', 'plusplusplus', 'plusplusplus_deep'
]
NN_TYPE_STRINGS_FANCY = [
    'U-net++ without DS', 'U-net++ with DS',
    'U-net3+ without DS', 'U-net3+ with DS'
]

MODEL_DEPTH_WIDTH_STRINGS = [
    '3, 1', '4, 1', '5, 1',
    '3, 2', '4, 2', '5, 2',
    '3, 3', '4, 3', '5, 3',
    '3, 4', '4, 4', '5, 4'
]
FIRST_LAYER_CHANNEL_COUNTS = numpy.array([4, 8, 16, 32, 64, 128], dtype=int)

SELECTED_MARKER_INDICES = numpy.array([2, 2, 4], dtype=int)
MARKER_TYPE = 'o'
MARKER_SIZE_GRID_CELLS = 0.175
MARKER_COLOUR = numpy.full(3, 1.)

MODEL_DEPTHS = numpy.array([3, 4, 5], dtype=int)
MODEL_WIDTHS = numpy.array([1, 2, 3, 4], dtype=int)

COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis', lut=20)
FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

FONT_SIZE = 26
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

NUM_WEIGHTS_ARRAY_PLUSPLUS = numpy.array([
    71276, 238451, 808919, 2768944, 9561655, 33239406, 76964, 260579, 896183,
    3115504, 10942903, 38754414, 82652, 282707, 983447, 3462064, 12324151,
    44269422, 88340, 304835, 1070711, 3808624, 13705399, 49784430, 88545,
    311391, 1114394, 4036284, 14782760, 54688770, 110973, 399735, 1465034,
    5433372, 20360168, 76976514, 133401, 488079, 1815674, 6830460, 25937576,
    99264258, 155829, 576423, 2166314, 8227548, 31514984, 121552002, 171743,
    651446, 2501636, 9675199, 37636606, 147105214, 260447, 1003382, 3903620,
    15271615, 59999230, 236509630, 349151, 1355318, 5305604, 20868031, 82361854,
    325914046, 437855, 1707254, 6707588, 26464447, 104724478, 415318462
], dtype=int)

NUM_WEIGHTS_ARRAY_PLUSPLUSPLUS = numpy.array([
    70440, 234859, 794055, 2708496, 9317879, 32260334, 77148, 261043, 897495,
    3119664, 10957367, 38807918, 83856, 287227, 1000935, 3530832, 12596855,
    45355502, 90564, 313411, 1104375, 3942000, 14236343, 51903086, 76049,
    261247, 913498, 3232060, 11564584, 41813506, 97829, 347143, 1254634,
    4591708, 16993384, 63509122, 119609, 433039, 1595770, 5951356, 22422184,
    85204738, 141389, 518935, 1936906, 7311004, 27850984, 106900354, 99523,
    362910, 1348180, 5062751, 19189566, 73322558, 174799, 661782, 2539204,
    9817919, 38192382, 149298110, 250075, 960654, 3730228, 14573087, 57195198,
    225273662, 325351, 1259526, 4921252, 19328255, 76198014, 301249214
], dtype=int)

NUM_WEIGHTS_ARRAY_PLUSPLUS_DEEP = numpy.array([
    71286, 238469, 808953, 2769010, 9561785, 33239664, 76974, 260597, 896217,
    3115570, 10943033, 38754672, 82662, 282725, 983481, 3462130, 12324281,
    44269680, 88350, 304853, 1070745, 3808690, 13705529, 49784688, 88560,
    311418, 1114445, 4036383, 14782955, 54689157, 110988, 399762, 1465085,
    5433471, 20360363, 76976901, 133416, 488106, 1815725, 6830559, 25937771,
    99264645, 155844, 576450, 2166365, 8227647, 31515179, 121552389, 171763,
    651482, 2501704, 9675331, 37636866, 147105730, 260467, 1003418, 3903688,
    15271747, 59999490, 236510146, 349171, 1355354, 5305672, 20868163, 82362114,
    325914562, 437875, 1707290, 6707656, 26464579, 104724738, 415318978
], dtype=int)

NUM_WEIGHTS_ARRAY_PLUSPLUSPLUS_DEEP = numpy.array([
    70507, 234990, 794314, 2709011, 9318906, 32262385, 77215, 261174, 897754,
    3120179, 10958394, 38809969, 83923, 287358, 1001194, 3531347, 12597882,
    45357553, 90631, 313542, 1104634, 3942515, 14237370, 51905137, 76177,
    261499, 913998, 3233056, 11566572, 41817478, 97957, 347395, 1255134,
    4592704, 16995372, 63513094, 119737, 433291, 1596270, 5952352, 22424172,
    85208710, 141517, 519187, 1937406, 7312000, 27852972, 106904326, 99752,
    363363, 1349081, 5064548, 19193155, 73329731, 175028, 662235, 2540105,
    9819716, 38195971, 149305283, 250304, 961107, 3731129, 14574884, 57198787,
    225280835, 325580, 1259979, 4922153, 19330052, 76201603, 301256387
], dtype=int)

OUTPUT_DIR_ARG_NAME = 'output_dir_name'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_scores_2d(
        score_matrix, min_colour_value, max_colour_value, x_tick_labels,
        y_tick_labels):
    """Plots scores on 2-D grid.

    M = number of rows in grid
    N = number of columns in grid

    :param score_matrix: M-by-N numpy array of scores.
    :param min_colour_value: Minimum value in colour scheme.
    :param max_colour_value: Max value in colour scheme.
    :param x_tick_labels: length-N list of tick labels.
    :param y_tick_labels: length-M list of tick labels.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    axes_object.imshow(
        score_matrix, cmap=COLOUR_MAP_OBJECT, origin='lower',
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

    colour_norm_object = matplotlib.colors.Normalize(
        vmin=min_colour_value, vmax=max_colour_value, clip=False
    )

    colour_bar_object = gg_plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object,
        data_matrix=score_matrix[numpy.invert(numpy.isnan(score_matrix))],
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        font_size=FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _run(output_dir_name):
    """Plots number of weights vs. hyperparameters for longwave models.

    This is effectively the main method.

    :param output_dir_name: See documentation at top of file.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Housekeeping.
    y_tick_labels = [
        '{0:s}'.format(s) for s in MODEL_DEPTH_WIDTH_STRINGS
    ]
    x_tick_labels = [
        '{0:d}'.format(c) for c in FIRST_LAYER_CHANNEL_COUNTS
    ]

    y_axis_label = 'NN depth, width'
    x_axis_label = 'Spectral complexity'

    # Plot grid for U-net++ without deep supervision.
    dimensions_3d = (
        len(MODEL_WIDTHS), len(MODEL_DEPTHS), len(FIRST_LAYER_CHANNEL_COUNTS)
    )
    num_weights_matrix = numpy.reshape(NUM_WEIGHTS_ARRAY_PLUSPLUS, dimensions_3d)
    num_weights_matrix = numpy.swapaxes(num_weights_matrix, 0, 1)

    num_depth_width_combos = len(MODEL_DEPTH_WIDTH_STRINGS)
    num_channel_counts = len(FIRST_LAYER_CHANNEL_COUNTS)
    dimensions_2d = (num_depth_width_combos, num_channel_counts)
    num_weights_matrix = numpy.reshape(num_weights_matrix, dimensions_2d)

    all_weights = numpy.concatenate((
        NUM_WEIGHTS_ARRAY_PLUSPLUS, NUM_WEIGHTS_ARRAY_PLUSPLUSPLUS,
        NUM_WEIGHTS_ARRAY_PLUSPLUS_DEEP, NUM_WEIGHTS_ARRAY_PLUSPLUSPLUS_DEEP
    ), axis=0)

    min_colour_value = numpy.min(numpy.log10(all_weights))
    max_colour_value = numpy.max(numpy.log10(all_weights))

    figure_object, axes_object = _plot_scores_2d(
        score_matrix=numpy.log10(num_weights_matrix),
        min_colour_value=min_colour_value,
        max_colour_value=max_colour_value,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )

    figure_width_px = (
        figure_object.get_size_inches()[0] * figure_object.dpi
    )
    marker_size_px = figure_width_px * (
        MARKER_SIZE_GRID_CELLS / num_weights_matrix.shape[1]
    )

    if SELECTED_MARKER_INDICES[0] == 0:
        axes_object.plot(
            SELECTED_MARKER_INDICES[2], SELECTED_MARKER_INDICES[1],
            linestyle='None', marker=MARKER_TYPE,
            markersize=marker_size_px, markeredgewidth=0,
            markerfacecolor=MARKER_COLOUR,
            markeredgecolor=MARKER_COLOUR
        )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title(NN_TYPE_STRINGS_FANCY[0])

    panel_file_names = [
        '{0:s}/num_weights_log10_{1:s}.jpg'.format(
            output_dir_name, NN_TYPE_STRINGS[0].replace('_', '-')
        )
    ]

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot grid for U-net++ with deep supervision.
    num_weights_matrix = numpy.reshape(
        NUM_WEIGHTS_ARRAY_PLUSPLUS_DEEP, dimensions_3d
    )
    num_weights_matrix = numpy.swapaxes(num_weights_matrix, 0, 1)
    num_weights_matrix = numpy.reshape(num_weights_matrix, dimensions_2d)

    figure_object, axes_object = _plot_scores_2d(
        score_matrix=numpy.log10(num_weights_matrix),
        min_colour_value=min_colour_value,
        max_colour_value=max_colour_value,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )

    if SELECTED_MARKER_INDICES[0] == 1:
        axes_object.plot(
            SELECTED_MARKER_INDICES[2], SELECTED_MARKER_INDICES[1],
            linestyle='None', marker=MARKER_TYPE,
            markersize=marker_size_px, markeredgewidth=0,
            markerfacecolor=MARKER_COLOUR,
            markeredgecolor=MARKER_COLOUR
        )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title(NN_TYPE_STRINGS_FANCY[1])

    panel_file_names.append(
        '{0:s}/num_weights_log10_{1:s}.jpg'.format(
            output_dir_name, NN_TYPE_STRINGS[1].replace('_', '-')
        )
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot grid for U-net3+ without deep supervision.
    num_weights_matrix = numpy.reshape(
        NUM_WEIGHTS_ARRAY_PLUSPLUSPLUS, dimensions_3d
    )
    num_weights_matrix = numpy.swapaxes(num_weights_matrix, 0, 1)
    num_weights_matrix = numpy.reshape(num_weights_matrix, dimensions_2d)

    figure_object, axes_object = _plot_scores_2d(
        score_matrix=numpy.log10(num_weights_matrix),
        min_colour_value=min_colour_value,
        max_colour_value=max_colour_value,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )

    if SELECTED_MARKER_INDICES[0] == 2:
        axes_object.plot(
            SELECTED_MARKER_INDICES[2], SELECTED_MARKER_INDICES[1],
            linestyle='None', marker=MARKER_TYPE,
            markersize=marker_size_px, markeredgewidth=0,
            markerfacecolor=MARKER_COLOUR,
            markeredgecolor=MARKER_COLOUR
        )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title(NN_TYPE_STRINGS_FANCY[2])

    panel_file_names.append(
        '{0:s}/num_weights_log10_{1:s}.jpg'.format(
            output_dir_name, NN_TYPE_STRINGS[2].replace('_', '-')
        )
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot grid for U-net3+ with deep supervision.
    num_weights_matrix = numpy.reshape(
        NUM_WEIGHTS_ARRAY_PLUSPLUSPLUS_DEEP, dimensions_3d
    )
    num_weights_matrix = numpy.swapaxes(num_weights_matrix, 0, 1)
    num_weights_matrix = numpy.reshape(num_weights_matrix, dimensions_2d)

    figure_object, axes_object = _plot_scores_2d(
        score_matrix=numpy.log10(num_weights_matrix),
        min_colour_value=min_colour_value,
        max_colour_value=max_colour_value,
        x_tick_labels=x_tick_labels, y_tick_labels=y_tick_labels
    )

    if SELECTED_MARKER_INDICES[0] == 3:
        axes_object.plot(
            SELECTED_MARKER_INDICES[2], SELECTED_MARKER_INDICES[1],
            linestyle='None', marker=MARKER_TYPE,
            markersize=marker_size_px, markeredgewidth=0,
            markerfacecolor=MARKER_COLOUR,
            markeredgecolor=MARKER_COLOUR
        )

    axes_object.set_xlabel(x_axis_label)
    axes_object.set_ylabel(y_axis_label)
    axes_object.set_title(NN_TYPE_STRINGS_FANCY[3])

    panel_file_names.append(
        '{0:s}/num_weights_log10_{1:s}.jpg'.format(
            output_dir_name, NN_TYPE_STRINGS[3].replace('_', '-')
        )
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Concatenate panels into one figure.
    concat_file_name = '{0:s}/num_weights_log10.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_file_name,
        num_panel_rows=2, num_panel_columns=2
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_file_name,
        output_file_name=concat_file_name, output_size_pixels=int(1e7)
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
