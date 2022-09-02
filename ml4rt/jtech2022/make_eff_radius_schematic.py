"""Makes schematic to show how synthetic effective-radius profiles are made."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4rt.utils import example_utils
from ml4rt.plotting import profile_plotting

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

TEMPERATURES_DEG_C = numpy.linspace(-75, -10, num=127, dtype=float)[::1]

PROFILE_NOISE_STDEV_FRACTIONAL = 0.05
INDIV_NOISE_STDEV_FRACTIONAL = 0.005

METRES_TO_MICRONS = 1e6
METRES_TO_KM = 1e-3

PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

ICE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
LAND_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
OCEAN_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255

FONT_SIZE = 30
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
    """Makes schematic to show how synthetic effective-radius profiles are made.

    This is effectively the main method.

    :param output_dir_name: See documentation at top of file.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    ice_effective_radii_metres = (
        example_utils.ICE_EFF_RADIUS_INTERCEPT_METRES +
        example_utils.ICE_EFF_RADIUS_SLOPE_METRES_CELSIUS01 *
        TEMPERATURES_DEG_C
    )

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=ice_effective_radii_metres * METRES_TO_MICRONS,
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, use_log_scale=True, line_colour=ICE_COLOUR
    )

    axes_object.set_xlabel(r'Ice effective radius ($\mu$m)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('Step 1: Apply equation')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(a)')

    panel_file_names = ['{0:s}/ice_step1.jpg'.format(output_dir_name)]

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    ice_effective_radii_metres = example_utils._add_noise_to_profiles(
        data_matrix=numpy.expand_dims(ice_effective_radii_metres, axis=0),
        profile_noise_stdev_fractional=PROFILE_NOISE_STDEV_FRACTIONAL,
        indiv_noise_stdev_fractional=INDIV_NOISE_STDEV_FRACTIONAL
    )[0, :]

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=ice_effective_radii_metres * METRES_TO_MICRONS,
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, use_log_scale=True, line_colour=ICE_COLOUR
    )

    axes_object.set_xlabel(r'Ice effective radius ($\mu$m)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('Step 2: Add noise')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(b)')

    panel_file_names.append(
        '{0:s}/ice_step2.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    ice_effective_radii_metres = numpy.maximum(
        ice_effective_radii_metres, example_utils.MIN_ICE_EFF_RADIUS_METRES
    )
    ice_effective_radii_metres = numpy.minimum(
        ice_effective_radii_metres, example_utils.MAX_ICE_EFF_RADIUS_METRES
    )

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=ice_effective_radii_metres * METRES_TO_MICRONS,
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, use_log_scale=True, line_colour=ICE_COLOUR
    )

    axes_object.set_xlabel(r'Ice effective radius ($\mu$m)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('Step 3: Enforce bounds')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(c)')

    panel_file_names.append(
        '{0:s}/ice_step3.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    land_effective_radii_metres = numpy.full(
        len(HEIGHTS_M_AGL), example_utils.LIQUID_EFF_RADIUS_LAND_MEAN_METRES
    )
    ocean_effective_radii_metres = numpy.full(
        len(HEIGHTS_M_AGL), example_utils.LIQUID_EFF_RADIUS_OCEAN_MEAN_METRES
    )

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=ocean_effective_radii_metres * METRES_TO_MICRONS,
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, use_log_scale=True, line_colour=OCEAN_COLOUR
    )

    land_handle = axes_object.plot(
        land_effective_radii_metres * METRES_TO_MICRONS,
        HEIGHTS_M_AGL * METRES_TO_KM,
        color=LAND_COLOUR, linestyle='solid', linewidth=4
    )[0]

    ocean_handle = axes_object.plot(
        ocean_effective_radii_metres * METRES_TO_MICRONS,
        HEIGHTS_M_AGL * METRES_TO_KM,
        color=OCEAN_COLOUR, linestyle='solid', linewidth=4
    )[0]

    legend_strings = ['Land', 'Ocean']
    legend_handles = [land_handle, ocean_handle]

    axes_object.legend(
        legend_handles, legend_strings, loc='upper left',
        bbox_to_anchor=(0, 0.35), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=1., ncol=1
    )

    axes_object.set_xlabel(r'Liquid effective radius ($\mu$m)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('Step 1: Set constant profiles')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(d)')

    panel_file_names.append(
        '{0:s}/liquid_step1.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    this_stdev = (
        example_utils.LIQUID_EFF_RADIUS_LAND_STDEV_METRES /
        example_utils.LIQUID_EFF_RADIUS_LAND_MEAN_METRES
    )
    land_effective_radii_metres = example_utils._add_noise_to_profiles(
        data_matrix=numpy.expand_dims(land_effective_radii_metres, axis=0),
        profile_noise_stdev_fractional=this_stdev,
        indiv_noise_stdev_fractional=0.1 * this_stdev
    )[0, :]

    this_stdev = (
        example_utils.LIQUID_EFF_RADIUS_OCEAN_STDEV_METRES /
        example_utils.LIQUID_EFF_RADIUS_OCEAN_MEAN_METRES
    )
    ocean_effective_radii_metres = example_utils._add_noise_to_profiles(
        data_matrix=numpy.expand_dims(ocean_effective_radii_metres, axis=0),
        profile_noise_stdev_fractional=this_stdev,
        indiv_noise_stdev_fractional=0.1 * this_stdev
    )[0, :]

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=ocean_effective_radii_metres * METRES_TO_MICRONS,
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, use_log_scale=True, line_colour=OCEAN_COLOUR
    )

    land_handle = axes_object.plot(
        land_effective_radii_metres * METRES_TO_MICRONS,
        HEIGHTS_M_AGL * METRES_TO_KM,
        color=LAND_COLOUR, linestyle='solid', linewidth=4
    )[0]

    ocean_handle = axes_object.plot(
        ocean_effective_radii_metres * METRES_TO_MICRONS,
        HEIGHTS_M_AGL * METRES_TO_KM,
        color=OCEAN_COLOUR, linestyle='solid', linewidth=4
    )[0]

    legend_strings = ['Land', 'Ocean']
    legend_handles = [land_handle, ocean_handle]

    axes_object.legend(
        legend_handles, legend_strings, loc='upper left',
        bbox_to_anchor=(0, 0.35), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=1., ncol=1
    )

    axes_object.set_xlabel(r'Liquid effective radius ($\mu$m)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('Step 2: Add noise')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(e)')

    panel_file_names.append(
        '{0:s}/liquid_step2.jpg'.format(output_dir_name)
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

    concat_figure_file_name = '{0:s}/eff_radius_schematic.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=2, num_panel_columns=3
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
