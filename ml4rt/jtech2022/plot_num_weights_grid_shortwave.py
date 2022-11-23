"""Plots number of weights vs. hyperparameters for shortwave models."""

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

SELECTED_MARKER_INDICES = numpy.array([0, 0, 5], dtype=int)
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
    71300, 238499, 809015, 2769136, 9562039, 33240174, 76988, 260627, 896279,
    3115696, 10943287, 38755182, 82676, 282755, 983543, 3462256, 12324535,
    44270190, 88364, 304883, 1070807, 3808816, 13705783, 49785198, 88569,
    311439, 1114490, 4036476, 14783144, 54689538, 110997, 399783, 1465130,
    5433564, 20360552, 76977282, 133425, 488127, 1815770, 6830652, 25937960,
    99265026, 155853, 576471, 2166410, 8227740, 31515368, 121552770, 171767,
    651494, 2501732, 9675391, 37636990, 147105982, 260471, 1003430, 3903716,
    15271807, 59999614, 236510398, 349175, 1355366, 5305700, 20868223,
    82362238, 325914814, 437879, 1707302, 6707684, 26464639, 104724862,
    415319230
], dtype=int)

NUM_WEIGHTS_ARRAY_PLUSPLUS_DEEP = numpy.array([
    71310, 238517, 809049, 2769202, 9562169, 33240432, 76998, 260645, 896313,
    3115762, 10943417, 38755440, 82686, 282773, 983577, 3462322, 12324665,
    44270448, 88374, 304901, 1070841, 3808882, 13705913, 49785456, 88584,
    311466, 1114541, 4036575, 14783339, 54689925, 111012, 399810, 1465181,
    5433663, 20360747, 76977669, 133440, 488154, 1815821, 6830751, 25938155,
    99265413, 155868, 576498, 2166461, 8227839, 31515563, 121553157, 171787,
    651530, 2501800, 9675523, 37637250, 147106498, 260491, 1003466, 3903784,
    15271939, 59999874, 236510914, 349195, 1355402, 5305768, 20868355,
    82362498, 325915330, 437899, 1707338, 6707752, 26464771, 104725122,
    415319746
], dtype=int)

NUM_WEIGHTS_ARRAY_PLUSPLUSPLUS = numpy.array([
    70464, 234907, 794151, 2708688, 9318263, 32261102, 77172, 261091, 897591,
    3119856, 10957751, 38808686, 83880, 287275, 1001031, 3531024, 12597239,
    45356270, 90588, 313459, 1104471, 3942192, 14236727, 51903854, 76073,
    261295, 913594, 3232252, 11564968, 41814274, 97853, 347191, 1254730,
    4591900, 16993768, 63509890, 119633, 433087, 1595866, 5951548, 22422568,
    85205506, 141413, 518983, 1937002, 7311196, 27851368, 106901122, 99547,
    362958, 1348276, 5062943, 19189950, 73323326, 174823, 661830, 2539300,
    9818111, 38192766, 149298878, 250099, 960702, 3730324, 14573279, 57195582,
    225274430, 325375, 1259574, 4921348, 19328447, 76198398, 301249982
], dtype=int)

NUM_WEIGHTS_ARRAY_PLUSPLUSPLUS_DEEP = numpy.array([
    70531, 235038, 794410, 2709203, 9319290, 32263153, 77239, 261222, 897850,
    3120371, 10958778, 38810737, 83947, 287406, 1001290, 3531539, 12598266,
    45358321, 90655, 313590, 1104730, 3942707, 14237754, 51905905, 76201,
    261547, 914094, 3233248, 11566956, 41818246, 97981, 347443, 1255230,
    4592896, 16995756, 63513862, 119761, 433339, 1596366, 5952544, 22424556,
    85209478, 141541, 519235, 1937502, 7312192, 27853356, 106905094, 99776,
    363411, 1349177, 5064740, 19193539, 73330499, 175052, 662283, 2540201,
    9819908, 38196355, 149306051, 250328, 961155, 3731225, 14575076, 57199171,
    225281603, 325604, 1260027, 4922249, 19330244, 76201987, 301257155
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
    """Plots number of weights vs. hyperparameters for shortwave models.

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
