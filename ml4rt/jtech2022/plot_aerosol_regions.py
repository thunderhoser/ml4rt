"""Plots aerosol regions on world map."""

import copy
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from gewittergefahr.gg_utils import general_utils as gg_general_utils
from gewittergefahr.gg_utils import file_system_utils
from ml4tc.io import border_io
from ml4rt.utils import aerosols
from ml4tc.plotting import plotting_utils

OUTPUT_FILE_ARG_NAME = 'output_file_name'
OUTPUT_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)

REGION_NAME_TO_VERBOSE = {
    aerosols.FIRST_URBAN_REGION_NAME: 'Urban #1',
    aerosols.SECOND_URBAN_REGION_NAME: 'Urban #2',
    aerosols.FIRST_DESERT_DUST_REGION_NAME: 'Desert dust #1',
    aerosols.SECOND_DESERT_DUST_REGION_NAME: 'Desert dust #2',
    aerosols.BIOMASS_BURNING_REGION_NAME: 'Biomass-burning'
}

REGION_NAME_TO_TEXT_COLOUR = {
    aerosols.FIRST_URBAN_REGION_NAME: numpy.array([117, 112, 179]),
    aerosols.SECOND_URBAN_REGION_NAME: numpy.array([117, 112, 179]),
    aerosols.FIRST_DESERT_DUST_REGION_NAME: numpy.array([217, 95, 2]),
    aerosols.SECOND_DESERT_DUST_REGION_NAME: numpy.array([217, 95, 2]),
    aerosols.BIOMASS_BURNING_REGION_NAME: numpy.array([27, 158, 119])
}

REGION_NAME_TO_POLYGON_COLOUR = copy.deepcopy(REGION_NAME_TO_TEXT_COLOUR)

for this_key in REGION_NAME_TO_TEXT_COLOUR:
    REGION_NAME_TO_TEXT_COLOUR[this_key] = (
        REGION_NAME_TO_TEXT_COLOUR[this_key].astype(float) / 255
    )
    REGION_NAME_TO_POLYGON_COLOUR[this_key] = matplotlib.colors.to_rgba(
        c=REGION_NAME_TO_TEXT_COLOUR[this_key], alpha=0.5
    )

FIGURE_WIDTH_INCHES = 25
FIGURE_HEIGHT_INCHES = 5


def _run(output_file_name):
    """Plots aerosol regions on world map.

    This is effectively the main method.

    :param output_file_name: See documentation at top of file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    region_dict = aerosols._read_region_coords()
    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object, line_width=1, line_colour=numpy.full(3, 0.)
    )

    all_latitudes_deg_n = numpy.array([])

    for this_region_name in region_dict:
        all_latitudes_deg_n = numpy.concatenate((
            all_latitudes_deg_n, region_dict[this_region_name][0]
        ))

    min_plot_latitude_deg_n = numpy.floor(numpy.nanmin(all_latitudes_deg_n)) - 1
    max_plot_latitude_deg_n = numpy.ceil(numpy.nanmax(all_latitudes_deg_n)) + 1
    plot_latitudes_deg_n = numpy.linspace(
        min_plot_latitude_deg_n, max_plot_latitude_deg_n,
        num=int(numpy.round(
            max_plot_latitude_deg_n - min_plot_latitude_deg_n + 1
        ))
    )

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=plot_latitudes_deg_n,
        plot_longitudes_deg_e=numpy.linspace(0, 359, num=360, dtype=float),
        axes_object=axes_object,
        parallel_spacing_deg=10., meridian_spacing_deg=20.
    )

    for this_region_name in region_dict:
        latitude_arrays_deg_n = gg_general_utils.split_array_by_nan(
            region_dict[this_region_name][0]
        )
        longitude_arrays_deg_e = gg_general_utils.split_array_by_nan(
            region_dict[this_region_name][1]
        )

        for i in range(len(latitude_arrays_deg_n)):
            axes_object.plot(
                longitude_arrays_deg_e[i], latitude_arrays_deg_n[i],
                color=REGION_NAME_TO_POLYGON_COLOUR[this_region_name],
                linewidth=5
            )

            if (
                    this_region_name == aerosols.SECOND_DESERT_DUST_REGION_NAME
                    and i == 1
            ):
                continue

            axes_object.text(
                numpy.mean(longitude_arrays_deg_e[i]),
                numpy.mean(latitude_arrays_deg_n[i]),
                REGION_NAME_TO_VERBOSE[this_region_name],
                fontsize=30, fontweight='bold',
                color=REGION_NAME_TO_TEXT_COLOUR[this_region_name],
                horizontalalignment='center', verticalalignment='center'
            )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=300, pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
