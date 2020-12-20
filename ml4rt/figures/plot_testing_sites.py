"""Plots sites with testing data."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils

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

TROPICAL_SITE_TO_LATLNG = {
    'Bishop, Grenada': numpy.array([12.049037, 298.225193]),
    'Rohlsen USVI': numpy.array([17.658863, 295.181625]),
    'San Juan PR': numpy.array([18.428686, 294.029716]),
    'Hilo HI': numpy.array([19.711315, 204.901398]),
    'Guantanamo Bay': numpy.array([19.843439, 284.852074]),
    'Honolulu HI': numpy.array([21.326672, 202.113266]),
    'Perdido oil rig': numpy.array([26.133038, 265.124474])
}

TROPICAL_SITE_TO_ALIGNMENT = {
    'Bishop, Grenada': ['right', 'center'],
    'Rohlsen USVI': ['right', 'top'],
    'San Juan PR': ['center', 'bottom'],
    'Hilo HI': ['left', 'top'],
    'Guantanamo Bay': ['right', 'center'],
    'Honolulu HI': ['left', 'bottom'],
    'Perdido oil rig': ['center', 'top']
}

ASSORTED2_SITE_TO_LATLNG = {
    'Lamont, Oklahoma': numpy.array([36.5693, 262.5687]),
    # 'Azores': numpy.array([39.0827, 360. - 28.0923]),
    'Tiksi, Russia': numpy.array([71.6260, 129.7669]),
    'North pole': numpy.array([89.9983, 164.0000]),
    'Bishop, Grenada': numpy.array([12.049037, 298.225193]),
    'Perdido oil rig': numpy.array([26.133038, 265.124474])
}

OUTPUT_DIR_ARG_NAME = 'output_dir_name'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_assorted2_sites(output_file_name):
    """Plots sites in the "Assorted2" set.

    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    site_names = list(ASSORTED2_SITE_TO_LATLNG.keys())
    latitudes_deg_n = numpy.array([
        ASSORTED2_SITE_TO_LATLNG[n][0] for n in site_names
    ])
    longitudes_deg_e = numpy.array([
        ASSORTED2_SITE_TO_LATLNG[n][1] for n in site_names
    ])

    figure_object, axes_object, basemap_object = (
        plotting_utils.create_equidist_cylindrical_map(
            min_latitude_deg=10., max_latitude_deg=90.,
            min_longitude_deg=0., max_longitude_deg=359.9999,
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
    arctic_site_names = [site_names[i] for i in arctic_indices]
    arctic_x_coords, arctic_y_coords = basemap_object(
        longitudes_deg_e[arctic_indices], latitudes_deg_n[arctic_indices]
    )

    axes_object.plot(
        arctic_x_coords, arctic_y_coords, linestyle='None',
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=ARCTIC_COLOUR, markeredgecolor=ARCTIC_COLOUR
    )

    for i in range(len(arctic_site_names)):
        axes_object.text(
            arctic_x_coords[i], arctic_y_coords[i] - 2., arctic_site_names[i],
            fontsize=FONT_SIZE, color=ARCTIC_COLOUR,
            horizontalalignment='center', verticalalignment='top'
        )

    mid_latitude_indices = numpy.where(numpy.logical_and(
        latitudes_deg_n >= 30., latitudes_deg_n < 66.5
    ))[0]
    mid_latitude_site_names = [site_names[i] for i in mid_latitude_indices]
    mid_latitude_x_coords, mid_latitude_y_coords = basemap_object(
        longitudes_deg_e[mid_latitude_indices],
        latitudes_deg_n[mid_latitude_indices]
    )

    axes_object.plot(
        mid_latitude_x_coords, mid_latitude_y_coords, linestyle='None',
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=MID_LATITUDE_COLOUR, markeredgecolor=MID_LATITUDE_COLOUR
    )

    for i in range(len(mid_latitude_site_names)):
        axes_object.text(
            mid_latitude_x_coords[i], mid_latitude_y_coords[i] + 2.,
            mid_latitude_site_names[i],
            fontsize=FONT_SIZE, color=MID_LATITUDE_COLOUR,
            horizontalalignment='center', verticalalignment='bottom'
        )

    tropical_indices = numpy.where(latitudes_deg_n < 30.)[0]
    tropical_site_names = [site_names[i] for i in tropical_indices]
    tropical_x_coords, tropical_y_coords = basemap_object(
        longitudes_deg_e[tropical_indices], latitudes_deg_n[tropical_indices]
    )

    axes_object.plot(
        tropical_x_coords, tropical_y_coords, linestyle='None',
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=TROPICAL_COLOUR, markeredgecolor=TROPICAL_COLOUR
    )

    for i in range(len(tropical_site_names)):
        axes_object.text(
            tropical_x_coords[i], tropical_y_coords[i] + 2.,
            tropical_site_names[i],
            fontsize=FONT_SIZE, color=TROPICAL_COLOUR,
            horizontalalignment='center', verticalalignment='bottom'
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_tropical_sites(output_file_name):
    """Plots tropical sites.

    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    site_names = list(TROPICAL_SITE_TO_LATLNG.keys())
    latitudes_deg_n = numpy.array([
        TROPICAL_SITE_TO_LATLNG[n][0] for n in site_names
    ])
    longitudes_deg_e = numpy.array([
        TROPICAL_SITE_TO_LATLNG[n][1] for n in site_names
    ])

    min_latitude_deg_n = numpy.floor(numpy.min(latitudes_deg_n) - 1.)
    max_latitude_deg_n = numpy.ceil(numpy.max(latitudes_deg_n) + 1.)
    min_longitude_deg_e = numpy.floor(numpy.min(longitudes_deg_e) - 1.)
    max_longitude_deg_e = numpy.ceil(numpy.max(longitudes_deg_e) + 1.)

    min_latitude_deg_n = max([min_latitude_deg_n, 10.])
    max_latitude_deg_n = min([max_latitude_deg_n, 90.])
    min_longitude_deg_e = max([min_longitude_deg_e, 0.])
    max_longitude_deg_e = min([max_longitude_deg_e, 359.9999])

    figure_object, axes_object, basemap_object = (
        plotting_utils.create_equidist_cylindrical_map(
            min_latitude_deg=min_latitude_deg_n,
            max_latitude_deg=max_latitude_deg_n,
            min_longitude_deg=min_longitude_deg_e,
            max_longitude_deg=max_longitude_deg_e,
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

    x_coords, y_coords = basemap_object(longitudes_deg_e, latitudes_deg_n)

    axes_object.plot(
        x_coords, y_coords, linestyle='None',
        marker=MARKER_TYPE, markersize=MARKER_SIZE, markeredgewidth=0,
        markerfacecolor=TROPICAL_COLOUR, markeredgecolor=TROPICAL_COLOUR
    )

    for i in range(len(site_names)):
        horiz_align_string = TROPICAL_SITE_TO_ALIGNMENT[site_names[i]][0]
        vertical_align_string = TROPICAL_SITE_TO_ALIGNMENT[site_names[i]][1]

        if horiz_align_string == 'left':
            this_x_coord = x_coords[i] + 0.75
        elif horiz_align_string == 'right':
            this_x_coord = x_coords[i] - 0.75
        else:
            this_x_coord = x_coords[i] + 0.

        if vertical_align_string == 'bottom':
            this_y_coord = y_coords[i] + 0.75
        elif vertical_align_string == 'top':
            this_y_coord = y_coords[i] - 0.75
        else:
            this_y_coord = y_coords[i] + 0.

        axes_object.text(
            this_x_coord, this_y_coord, site_names[i],
            fontsize=FONT_SIZE, color=TROPICAL_COLOUR,
            horizontalalignment=horiz_align_string,
            verticalalignment=vertical_align_string
        )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(output_dir_name):
    """Plots sites with testing data.

    This is effectively the main method.

    :param output_dir_name: See documentation at top of file.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    tropical_file_name = '{0:s}/tropical_sites.jpg'.format(output_dir_name)
    _plot_tropical_sites(tropical_file_name)

    assorted2_file_name = '{0:s}/assorted2_sites.jpg'.format(output_dir_name)
    _plot_assorted2_sites(assorted2_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
