"""Plots all sites with data."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from ml4rt.io import example_io
from ml4rt.utils import example_utils

FIRST_YEAR = 2017
LAST_YEAR = 2020
LATLNG_TOLERANCE_DEG = 0.001

MIN_PLOT_LATITUDE_DEG_N = 10.
MAX_PLOT_LATITUDE_DEG_N = 90.
MIN_PLOT_LONGITUDE_DEG_E = 0.
MAX_PLOT_LONGITUDE_DEG_E = 359.9999
NUM_PARALLELS = 9
NUM_MERIDIANS = 13

BORDER_WIDTH = 0.5
BORDER_COLOUR = numpy.full(3, 152. / 255)
GRID_LINE_WIDTH = 0.5
GRID_LINE_COLOUR = numpy.full(3, 0.)
FIGURE_RESOLUTION_DPI = 300

MARKER_TYPE = 'o'
MARKER_SIZE = 8

TROPICAL_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
MID_LATITUDE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
ARCTIC_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255

FONT_SIZE = 16
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

TROPICAL_DIR_ARG_NAME = 'input_tropical_dir_name'
NON_TROPICAL_DIR_ARG_NAME = 'input_non_tropical_dir_name'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

TROPICAL_DIR_HELP_STRING = (
    'Name of directory with examples for tropical sites.  Files therein will be'
    ' found by `example_io.find_file` and read by `example_io.read_file`.'
)
NON_TROPICAL_DIR_HELP_STRING = (
    'Same as `{0:s}` but for non-tropical sites.'.format(TROPICAL_DIR_ARG_NAME)
)
OUTPUT_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TROPICAL_DIR_ARG_NAME, type=str, required=True,
    help=TROPICAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NON_TROPICAL_DIR_ARG_NAME, type=str, required=True,
    help=NON_TROPICAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(tropical_example_dir_name, non_tropical_example_dir_name,
         output_file_name):
    """Plots all sites wtih data.

    This is effectively the main method.
    
    :param tropical_example_dir_name: See documentation at top of file.
    :param non_tropical_example_dir_name: Same.
    :param output_file_name: Same.
    """

    first_time_unix_sec = (
        time_conversion.first_and_last_times_in_year(FIRST_YEAR)[0]
    )
    last_time_unix_sec = (
        time_conversion.first_and_last_times_in_year(LAST_YEAR)[-1]
    )

    tropical_file_names = example_io.find_many_files(
        directory_name=tropical_example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False
    )

    non_tropical_file_names = example_io.find_many_files(
        directory_name=non_tropical_example_dir_name,
        first_time_unix_sec=first_time_unix_sec,
        last_time_unix_sec=last_time_unix_sec,
        raise_error_if_all_missing=True, raise_error_if_any_missing=False
    )

    latitudes_deg_n = numpy.array([])
    longitudes_deg_e = numpy.array([])

    for this_file_name in tropical_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(this_file_name)

        these_latitudes_deg_n = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.LATITUDE_NAME
        )
        these_longitudes_deg_e = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.LONGITUDE_NAME
        )

        latitudes_deg_n = numpy.concatenate((
            latitudes_deg_n, these_latitudes_deg_n
        ))
        longitudes_deg_e = numpy.concatenate((
            longitudes_deg_e, these_longitudes_deg_e
        ))

    for this_file_name in non_tropical_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(this_file_name)

        these_latitudes_deg_n = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.LATITUDE_NAME
        )
        these_longitudes_deg_e = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.LONGITUDE_NAME
        )

        latitudes_deg_n = numpy.concatenate((
            latitudes_deg_n, these_latitudes_deg_n
        ))
        longitudes_deg_e = numpy.concatenate((
            longitudes_deg_e, these_longitudes_deg_e
        ))

    coord_matrix = numpy.transpose(numpy.vstack((
        latitudes_deg_n, longitudes_deg_e
    )))
    coord_matrix = number_rounding.round_to_nearest(
        coord_matrix, LATLNG_TOLERANCE_DEG
    )
    coord_matrix = numpy.unique(coord_matrix, axis=0)

    latitudes_deg_n = coord_matrix[:, 0]
    longitudes_deg_e = coord_matrix[:, 1]

    figure_object, axes_object, basemap_object = (
        plotting_utils.create_equidist_cylindrical_map(
            min_latitude_deg=MIN_PLOT_LATITUDE_DEG_N,
            max_latitude_deg=MAX_PLOT_LATITUDE_DEG_N,
            min_longitude_deg=MIN_PLOT_LONGITUDE_DEG_E,
            max_longitude_deg=MAX_PLOT_LONGITUDE_DEG_E,
            resolution_string='l'
        )
    )

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR, line_width=BORDER_WIDTH
    )
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR, line_width=BORDER_WIDTH
    )
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        num_parallels=NUM_PARALLELS, line_colour=GRID_LINE_COLOUR,
        line_width=GRID_LINE_WIDTH, font_size=FONT_SIZE
    )
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS, line_colour=GRID_LINE_COLOUR,
        line_width=GRID_LINE_WIDTH, font_size=FONT_SIZE
    )

    arctic_indices = numpy.where(latitudes_deg_n >= 66.5)[0]
    print(len(arctic_indices))

    arctic_x_coords, arctic_y_coords = basemap_object(
        longitudes_deg_e[arctic_indices], latitudes_deg_n[arctic_indices]
    )
    axes_object.plot(
        arctic_x_coords, arctic_y_coords, linestyle='None',
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=ARCTIC_COLOUR, markeredgecolor=ARCTIC_COLOUR
    )

    mid_latitude_indices = numpy.where(numpy.logical_and(
        latitudes_deg_n >= 30., latitudes_deg_n < 66.5
    ))[0]
    print(len(mid_latitude_indices))

    mid_latitude_x_coords, mid_latitude_y_coords = basemap_object(
        longitudes_deg_e[mid_latitude_indices],
        latitudes_deg_n[mid_latitude_indices]
    )
    axes_object.plot(
        mid_latitude_x_coords, mid_latitude_y_coords, linestyle='None',
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=MID_LATITUDE_COLOUR, markeredgecolor=MID_LATITUDE_COLOUR
    )

    tropical_indices = numpy.where(latitudes_deg_n < 30.)[0]
    print(len(tropical_indices))

    tropical_x_coords, tropical_y_coords = basemap_object(
        longitudes_deg_e[tropical_indices], latitudes_deg_n[tropical_indices]
    )
    axes_object.plot(
        tropical_x_coords, tropical_y_coords, linestyle='None',
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=TROPICAL_COLOUR, markeredgecolor=TROPICAL_COLOUR
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        tropical_example_dir_name=getattr(
            INPUT_ARG_OBJECT, TROPICAL_DIR_ARG_NAME
        ),
        non_tropical_example_dir_name=getattr(
            INPUT_ARG_OBJECT, NON_TROPICAL_DIR_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
