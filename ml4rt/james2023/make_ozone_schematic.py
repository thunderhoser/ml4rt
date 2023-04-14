"""Makes schematic to show how ozone layer is created."""

import os
import argparse
import numpy
import xarray
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
import matplotlib.colors
import matplotlib.patches
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
METRES_TO_KM = 0.001
KG_TO_MILLIGRAMS = 1e6

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

LAYER_THICKNESS_METRES = 20000.
ORIG_LAYER_CENTER_M_AGL = 25000.
MAX_MIXING_RATIO_KG_KG01 = 1e-5
MIXING_RATIO_NOISE_STDEV_KG_KG01 = 1e-6

PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

OZONE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
TEMPERATURE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

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
    """Makes schematic to show how ozone layer is created.

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
        tgt[TEMPERATURE_KEY].values[:, TROPICAL_STANDARD_ATMO_INDEX]
    )
    orig_heights_m_agl = KM_TO_METRES * tgt.coords[HEIGHT_DIM].values

    interp_object = interp1d(
        x=orig_heights_m_agl, y=orig_temperatures_kelvins, kind='cubic',
        bounds_error=True, assume_sorted=True
    )
    temperatures_kelvins = interp_object(HEIGHTS_M_AGL)

    layer_bottom_m_agl = ORIG_LAYER_CENTER_M_AGL - LAYER_THICKNESS_METRES / 2
    layer_top_m_agl = ORIG_LAYER_CENTER_M_AGL + LAYER_THICKNESS_METRES / 2
    layer_height_indices = perturb_gfs._heights_to_grid_indices(
        min_height_m_agl=layer_bottom_m_agl,
        max_height_m_agl=layer_top_m_agl,
        sorted_grid_heights_m_agl=HEIGHTS_M_AGL
    )

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=temperature_conv.kelvins_to_celsius(temperatures_kelvins),
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, line_style='solid', use_log_scale=True,
        line_colour=TEMPERATURE_COLOUR
    )

    min_layer_height_km_agl = (
        METRES_TO_KM * HEIGHTS_M_AGL[layer_height_indices[0]]
    )
    max_layer_height_km_agl = (
        METRES_TO_KM * HEIGHTS_M_AGL[layer_height_indices[-1]]
    )
    polygon_y_coords = numpy.array([
        min_layer_height_km_agl, max_layer_height_km_agl,
        max_layer_height_km_agl, min_layer_height_km_agl,
        min_layer_height_km_agl
    ])

    x_min = axes_object.get_xlim()[0]
    x_max = axes_object.get_xlim()[1]
    polygon_x_coords = numpy.array([x_min, x_min, x_max, x_max, x_min])
    polygon_coord_matrix = numpy.transpose(numpy.vstack((
        polygon_x_coords, polygon_y_coords
    )))

    polygon_colour = matplotlib.colors.to_rgba(
        numpy.full(3, 0.), 0.5
    )
    patch_object = matplotlib.patches.Polygon(
        polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
    )
    axes_object.add_patch(patch_object)

    axes_object.set_xlabel(r'Temperature ($^{\circ}$C)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('Temperature profile + extent of\nozone layer')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(a)')

    panel_file_names = ['{0:s}/ozone_layer_part1.jpg'.format(output_dir_name)]
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    _, tropopause_height_index = perturb_gfs._find_tropopause(
        temperatures_kelvins=temperatures_kelvins,
        sorted_heights_m_agl=HEIGHTS_M_AGL
    )
    if tropopause_height_index is None:
        tropopause_height_index = numpy.argmin(numpy.absolute(
            HEIGHTS_M_AGL - perturb_gfs.MAX_TROPOPAUSE_HEIGHT_EVER_M_AGL
        ))

    layer_height_indices = layer_height_indices[
        layer_height_indices > tropopause_height_index
    ]
    print(HEIGHTS_M_AGL[tropopause_height_index])

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=temperature_conv.kelvins_to_celsius(temperatures_kelvins),
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, line_style='solid', use_log_scale=True,
        line_colour=TEMPERATURE_COLOUR
    )

    min_layer_height_km_agl = (
        METRES_TO_KM * HEIGHTS_M_AGL[layer_height_indices[0]]
    )
    max_layer_height_km_agl = (
        METRES_TO_KM * HEIGHTS_M_AGL[layer_height_indices[-1]]
    )
    polygon_y_coords = numpy.array([
        min_layer_height_km_agl, max_layer_height_km_agl,
        max_layer_height_km_agl, min_layer_height_km_agl,
        min_layer_height_km_agl
    ])

    x_min = axes_object.get_xlim()[0]
    x_max = axes_object.get_xlim()[1]
    polygon_x_coords = numpy.array([x_min, x_min, x_max, x_max, x_min])
    polygon_coord_matrix = numpy.transpose(numpy.vstack((
        polygon_x_coords, polygon_y_coords
    )))

    polygon_colour = matplotlib.colors.to_rgba(
        numpy.full(3, 0.), 0.5
    )
    patch_object = matplotlib.patches.Polygon(
        polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
    )
    axes_object.add_patch(patch_object)

    axes_object.set_xlabel(r'Temperature ($^{\circ}$C)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title(
        'Temperature profile + extent of\nozone layer (no troposphere)'
    )
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(b)')

    panel_file_names.append(
        '{0:s}/ozone_layer_part2.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    layer_height_indices = numpy.concatenate((
        layer_height_indices[[0]] - 1,
        layer_height_indices,
        layer_height_indices[[-1]] + 1
    ))

    num_heights = len(HEIGHTS_M_AGL)
    layer_height_indices = layer_height_indices[
        layer_height_indices < num_heights
    ]
    layer_height_indices = layer_height_indices[layer_height_indices >= 0]

    layer_heights_m_agl = HEIGHTS_M_AGL[layer_height_indices]

    if (
            ORIG_LAYER_CENTER_M_AGL < layer_heights_m_agl[0] or
            ORIG_LAYER_CENTER_M_AGL > layer_heights_m_agl[-1]
    ):
        min_index = numpy.argmin(
            numpy.absolute(ORIG_LAYER_CENTER_M_AGL - layer_heights_m_agl)
        )
        layer_center_m_agl = layer_heights_m_agl[min_index]
    else:
        layer_center_m_agl = ORIG_LAYER_CENTER_M_AGL + 0.

    height_diffs_metres = numpy.absolute(
        layer_center_m_agl - layer_heights_m_agl
    )
    height_diffs_relative = height_diffs_metres / numpy.max(height_diffs_metres)
    height_diffs_relative[0] = 1.
    height_diffs_relative[-1] = 1.

    ozone_mixing_ratios_kg_kg01 = numpy.full(num_heights, 0.)
    ozone_mixing_ratios_kg_kg01[layer_height_indices] = (
        (1. - height_diffs_relative) * MAX_MIXING_RATIO_KG_KG01
    )

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=KG_TO_MILLIGRAMS * ozone_mixing_ratios_kg_kg01,
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, line_style='solid', use_log_scale=True,
        line_colour=OZONE_COLOUR
    )

    axes_object.set_xlabel(r'Mixing ratio (mg kg$^{-1}$)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('New ozone profile')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(c)')

    panel_file_names.append(
        '{0:s}/ozone_layer_part3.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    while True:
        noise_values_kg_kg01 = numpy.random.normal(
            loc=0., scale=MIXING_RATIO_NOISE_STDEV_KG_KG01,
            size=len(layer_height_indices)
        )
        new_ozone_mixing_ratios_kg_kg01 = (
            ozone_mixing_ratios_kg_kg01[layer_height_indices] +
            noise_values_kg_kg01
        )

        if not numpy.any(new_ozone_mixing_ratios_kg_kg01 < 0):
            continue
        if not numpy.any(
                new_ozone_mixing_ratios_kg_kg01 > MAX_MIXING_RATIO_KG_KG01
        ):
            continue

        ozone_mixing_ratios_kg_kg01[layer_height_indices] += (
            noise_values_kg_kg01
        )
        break

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=KG_TO_MILLIGRAMS * ozone_mixing_ratios_kg_kg01,
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, line_style='solid', use_log_scale=True,
        line_colour=OZONE_COLOUR
    )

    axes_object.set_xlabel(r'Mixing ratio (mg kg$^{-1}$)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('New ozone profile\n(including Gaussian noise)')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(d)')

    panel_file_names.append(
        '{0:s}/ozone_layer_part4.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    ozone_mixing_ratios_kg_kg01 = numpy.maximum(ozone_mixing_ratios_kg_kg01, 0.)
    ozone_mixing_ratios_kg_kg01 = numpy.minimum(
        ozone_mixing_ratios_kg_kg01, MAX_MIXING_RATIO_KG_KG01
    )

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=KG_TO_MILLIGRAMS * ozone_mixing_ratios_kg_kg01,
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, line_style='solid', use_log_scale=True,
        line_colour=OZONE_COLOUR
    )

    axes_object.set_xlabel(r'Mixing ratio (mg kg$^{-1}$)')
    axes_object.set_ylabel('Height (km AGL)')

    title_string = (
        'New ozone profile\n(bounded to 0...{0:.0f}'
    ).format(KG_TO_MILLIGRAMS * MAX_MIXING_RATIO_KG_KG01)

    title_string += r' mg kg$^{-1}$)'
    axes_object.set_title(title_string)
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(d)')

    panel_file_names.append(
        '{0:s}/ozone_layer_part5.jpg'.format(output_dir_name)
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

    concat_figure_file_name = '{0:s}/ozone_layer_schematic.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=3, num_panel_columns=2
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
