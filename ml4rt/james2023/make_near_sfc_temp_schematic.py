"""Makes schematic to show how near-sfc temps are perturbed."""

import os
import argparse
import numpy
import xarray
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import temperature_conversions as temperature_conv
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4rt.plotting import profile_plotting
from ml4rt.scripts import perturb_gfs_for_rrtm as perturb_gfs

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))

HEIGHT_DIM = 'height'
TEMPERATURE_KEY = 'temp'
TROPICAL_STANDARD_ATMO_INDEX = 0

KM_TO_METRES = 1000.

HEIGHTS_M_AGL = numpy.array([
    21, 44, 68, 93, 120, 149, 179, 212, 246, 282, 321, 361, 405, 450, 499, 550,
    604, 661, 722, 785, 853, 924, 999, 1078, 1161, 1249, 1342, 1439, 1542, 1649,
    1762, 1881, 2005, 2136, 2272, 2415, 2564, 2720, 2882, 3051, 3228, 3411,
    3601, 3798, 4002, 4214, 4433, 4659, 4892, 5132, 5379, 5633, 5894, 6162,
    6436, 6716, 7003, 7296, 7594, 7899, 8208, 8523, 8842, 9166, 9494, 9827,
    10164, 10505, 10849, 11198, 11550, 11906, 12266, 12630, 12997, 13368, 13744,
    14123, 14506, 14895, 15287, 15686, 16090, 16501, 16920, 17350, 17791, 18246,
    18717, 19205, 19715, 20249, 20809, 21400, 22022, 22681, 23379, 24119, 24903,
    25736, 26619, 27558, 28556, 29616, 30743, 31940, 33211, 34566, 36012, 37560,
    39218, 40990, 42882, 44899, 47042, 49299, 51644, 54067, 56552, 59089, 61677,
    64314, 67001, 69747, 72521, 75256, 77803
], dtype=float)

WHOLE_PROFILE_TEMP_OFFSET_KELVINS = 30.
SURFACE_TEMP_INCREASE_KELVINS = 8.
WARM_LAYER_DEPTH_METRES = 3000.

PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

ORIG_TEMPERATURE_COLOUR = numpy.array([55, 126, 184], dtype=float) / 255
TEMPERATURE_INCREASE_COLOUR = numpy.full(3, 152. / 255)
NEW_TEMPERATURE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255

FONT_SIZE = 25
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

OUTPUT_DIR_ARG_NAME = 'output_dir_name'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(output_dir_name):
    """Makes schematic to show how near-sfc temps are perturbed.

    This is effectively the main method.

    :param output_dir_name: See documentation at top of file.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    parent_dir_name = '/'.join(THIS_DIRECTORY_NAME.split('/')[:-1])
    trace_gas_file_name = '{0:s}/utils/trace_gases.nc'.format(parent_dir_name)
    trace_gas_table_xarray = xarray.open_dataset(trace_gas_file_name)

    tgt = trace_gas_table_xarray
    orig_temperatures_kelvins = (
        tgt[TEMPERATURE_KEY].values[:, TROPICAL_STANDARD_ATMO_INDEX] +
        WHOLE_PROFILE_TEMP_OFFSET_KELVINS
    )
    orig_heights_m_agl = KM_TO_METRES * tgt.coords[HEIGHT_DIM].values

    interp_object = interp1d(
        x=orig_heights_m_agl, y=orig_temperatures_kelvins, kind='cubic',
        bounds_error=True, assume_sorted=True
    )
    temperatures_kelvins = interp_object(HEIGHTS_M_AGL)

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=temperature_conv.kelvins_to_celsius(temperatures_kelvins),
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, use_log_scale=True, line_colour=ORIG_TEMPERATURE_COLOUR
    )

    axes_object.set_xlabel(r'Temperature ($^{\circ}$C)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('Original temperature profile')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(a)')

    panel_file_names = [
        '{0:s}/near_surface_temperature_part1.jpg'.format(output_dir_name)
    ]

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    layer_height_indices = perturb_gfs._heights_to_grid_indices(
        min_height_m_agl=0., max_height_m_agl=WARM_LAYER_DEPTH_METRES,
        sorted_grid_heights_m_agl=HEIGHTS_M_AGL
    )

    _, tropopause_height_index = perturb_gfs._find_tropopause(
        temperatures_kelvins=temperatures_kelvins,
        sorted_heights_m_agl=HEIGHTS_M_AGL
    )
    if tropopause_height_index is None:
        tropopause_height_index = numpy.argmin(numpy.absolute(
            HEIGHTS_M_AGL - perturb_gfs.MIN_TROPOPAUSE_HEIGHT_EVER_M_AGL
        ))

    layer_height_indices = layer_height_indices[
        layer_height_indices < tropopause_height_index
    ]

    layer_heights_m_agl = HEIGHTS_M_AGL[layer_height_indices]
    height_diffs_relative = layer_heights_m_agl / numpy.max(layer_heights_m_agl)

    temp_increases_kelvins = numpy.full(temperatures_kelvins.shape, 0.)
    temp_increases_kelvins[layer_height_indices] = (
        (1. - height_diffs_relative) * SURFACE_TEMP_INCREASE_KELVINS
    )

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=temp_increases_kelvins, heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, line_style='dashed', use_log_scale=True,
        line_colour=TEMPERATURE_INCREASE_COLOUR
    )

    axes_object.set_xlabel(r'Perturbation ($^{\circ}$C)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('Temperature perturbations')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(b)')

    panel_file_names.append(
        '{0:s}/near_surface_temperature_part2.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    temperatures_kelvins += temp_increases_kelvins

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=temperature_conv.kelvins_to_celsius(temperatures_kelvins),
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, use_log_scale=True, line_colour=NEW_TEMPERATURE_COLOUR
    )

    axes_object.set_xlabel(r'Temperature ($^{\circ}$C)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('New temperature profile')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(c)')

    panel_file_names.append(
        '{0:s}/near_surface_temperature_part3.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    temperatures_kelvins = numpy.minimum(
        temperatures_kelvins, perturb_gfs.MAX_TEMPERATURE_EVER_KELVINS
    )

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=temperature_conv.kelvins_to_celsius(temperatures_kelvins),
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, use_log_scale=True, line_colour=NEW_TEMPERATURE_COLOUR
    )

    axes_object.set_xlabel(r'Temperature ($^{\circ}$C)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title(
        'New temperature profile\n' + r'(all values $\leq$ 60 $^{\circ}$C)'
    )
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(d)')

    panel_file_names.append(
        '{0:s}/near_surface_temperature_part4.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    for this_file_name in panel_file_names:
        imagemagick_utils.resize_image(
            input_file_name=this_file_name, output_file_name=this_file_name,
            output_size_pixels=PANEL_SIZE_PX
        )

    concat_figure_file_name = (
        '{0:s}/near_surface_temperature_schematic.jpg'
    ).format(output_dir_name)

    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=2, num_panel_columns=2
    )
    imagemagick_utils.trim_whitespace(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        border_width_pixels=10
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
