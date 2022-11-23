"""Makes schematic to show how synthetic aerosol-extinctn profiles are made."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.stats import gaussian_kde
from scipy.integrate import simps
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4rt.utils import aerosols
from ml4rt.plotting import profile_plotting

MAX_AEROSOL_OPTICAL_DEPTH = 1.5
REGION_NAME = aerosols.SECOND_URBAN_REGION_NAME

GRID_HEIGHTS_M_AGL = numpy.array([
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

PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

EXTINCTION_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
CONSERVATIVE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
AGGRESSIVE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255

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
    """Makes schematic to show how synthetic aerosol-extinctn profiles are made.

    This is effectively the main method.

    :param output_dir_name: See documentation at top of file.
    """

    # Plot baseline extinction profile.
    num_heights = len(GRID_HEIGHTS_M_AGL)
    grid_height_matrix_metres = numpy.repeat(
        numpy.expand_dims(GRID_HEIGHTS_M_AGL, axis=0),
        axis=0, repeats=1
    )

    scale_heights_metres = numpy.random.normal(
        loc=aerosols.REGION_TO_SCALE_HEIGHT_MEAN_METRES[REGION_NAME],
        scale=aerosols.REGION_TO_SCALE_HEIGHT_STDEV_METRES[REGION_NAME],
        size=1
    )

    scale_heights_metres = numpy.maximum(scale_heights_metres, 100.)
    print(scale_heights_metres)
    scale_height_matrix_metres = numpy.repeat(
        numpy.expand_dims(scale_heights_metres, axis=-1),
        axis=-1, repeats=num_heights
    )
    baseline_extinction_matrix_metres01 = 0.001 * numpy.exp(
        -grid_height_matrix_metres / scale_height_matrix_metres
    )

    baseline_optical_depths = simps(
        y=baseline_extinction_matrix_metres01, x=GRID_HEIGHTS_M_AGL,
        axis=-1, even='avg'
    )

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=1000 * baseline_extinction_matrix_metres01[0, :],
        heights_m_agl=GRID_HEIGHTS_M_AGL,
        line_width=4, use_log_scale=True, line_colour=EXTINCTION_COLOUR
    )

    axes_object.set_xlabel(r'Aerosol extinction (km$^{-1}$)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('Step 5a: Compute baseline\nextinction profile')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(a)')

    panel_file_names = ['{0:s}/aerosol_step1.jpg'.format(output_dir_name)]

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot conservative AOD distributiom.
    narrow_optical_depths = 0.1 * numpy.random.gamma(
        shape=
        30 * aerosols.REGION_TO_OPTICAL_DEPTH_SHAPE_PARAM[REGION_NAME],
        scale=aerosols.REGION_TO_OPTICAL_DEPTH_SCALE_PARAM[REGION_NAME],
        size=int(1e6)
    )

    narrow_optical_depths = narrow_optical_depths[narrow_optical_depths >= 0]
    narrow_optical_depths = narrow_optical_depths[
        narrow_optical_depths <= MAX_AEROSOL_OPTICAL_DEPTH
    ]
    print(len(narrow_optical_depths))

    kde_object = gaussian_kde(narrow_optical_depths)
    x_values = numpy.linspace(
        0, MAX_AEROSOL_OPTICAL_DEPTH, num=1001, dtype=float
    )
    y_values = kde_object(x_values)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.plot(
        x_values, y_values, color=CONSERVATIVE_COLOUR,
        linewidth=4, linestyle='solid'
    )

    axes_object.set_xlabel('AOD')
    axes_object.set_ylabel('Probability density')
    axes_object.set_title('Step 6a: Create narrow\nAOD distribution')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(b)')

    panel_file_names.append(
        '{0:s}/aerosol_step2.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot aggressive AOD distribution.
    wide_optical_depths = 0.1 * numpy.random.gamma(
        shape=
        120 * aerosols.REGION_TO_OPTICAL_DEPTH_SHAPE_PARAM[REGION_NAME],
        scale=
        3 * aerosols.REGION_TO_OPTICAL_DEPTH_SCALE_PARAM[REGION_NAME],
        size=int(2e6)
    )

    kde_object = gaussian_kde(wide_optical_depths)
    x_values = numpy.linspace(0, 10, num=1001, dtype=float)
    y_values = kde_object(x_values)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.plot(
        x_values, y_values, color=AGGRESSIVE_COLOUR, linewidth=4,
        linestyle='solid'
    )

    axes_object.set_xlabel('AOD')
    axes_object.set_ylabel('Probability density')
    axes_object.set_title('Step 6b: Create wide\nAOD distribution')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(c)')

    panel_file_names.append(
        '{0:s}/aerosol_step3.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot shifted aggressive AOD distribution.
    shifted_wide_optical_depths = wide_optical_depths - (
        numpy.mean(wide_optical_depths) - numpy.mean(narrow_optical_depths)
    )

    shifted_wide_optical_depths = shifted_wide_optical_depths[
        shifted_wide_optical_depths >= 0
    ]
    shifted_wide_optical_depths = shifted_wide_optical_depths[
        shifted_wide_optical_depths <= MAX_AEROSOL_OPTICAL_DEPTH
    ]

    print(len(wide_optical_depths))

    kde_object = gaussian_kde(shifted_wide_optical_depths)
    x_values = numpy.linspace(
        0, MAX_AEROSOL_OPTICAL_DEPTH, num=1001, dtype=float
    )
    y_values = kde_object(x_values)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.plot(
        x_values, y_values, color=AGGRESSIVE_COLOUR, linewidth=4,
        linestyle='solid'
    )

    axes_object.set_xlabel('AOD')
    axes_object.set_ylabel('Probability density')
    axes_object.set_title('Step 6c-d: Shift and censor wide\nAOD distribution')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(d)')

    panel_file_names.append(
        '{0:s}/aerosol_step4.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot actual extinction profile.
    num_examples = 1
    this_sample_size = max([
        5 * num_examples, int(1e6)
    ])

    dummy_optical_depths = 0.1 * numpy.random.gamma(
        shape=
        30 * aerosols.REGION_TO_OPTICAL_DEPTH_SHAPE_PARAM[REGION_NAME],
        scale=aerosols.REGION_TO_OPTICAL_DEPTH_SCALE_PARAM[REGION_NAME],
        size=num_examples
    )

    actual_optical_depths = numpy.array([])

    while len(actual_optical_depths) < num_examples:
        these_depths = 0.1 * numpy.random.gamma(
            shape=
            120 * aerosols.REGION_TO_OPTICAL_DEPTH_SHAPE_PARAM[REGION_NAME],
            scale=
            3 * aerosols.REGION_TO_OPTICAL_DEPTH_SCALE_PARAM[REGION_NAME],
            size=this_sample_size
        )

        these_depths -= (
            numpy.mean(these_depths) - numpy.mean(dummy_optical_depths)
        )
        these_depths = these_depths[these_depths >= 0]
        these_depths = these_depths[
            these_depths <= MAX_AEROSOL_OPTICAL_DEPTH
        ]
        actual_optical_depths = numpy.concatenate(
            (actual_optical_depths, these_depths[:num_examples]),
            axis=0
        )

    actual_optical_depths = numpy.maximum(actual_optical_depths, 0.)
    actual_optical_depths = numpy.minimum(
        actual_optical_depths, MAX_AEROSOL_OPTICAL_DEPTH
    )

    print(actual_optical_depths)

    scale_factors = actual_optical_depths / baseline_optical_depths
    print(scale_factors)

    scale_factor_matrix = numpy.repeat(
        numpy.expand_dims(scale_factors, axis=-1),
        axis=-1, repeats=num_heights
    )

    extinction_matrix_metres01 = (
        scale_factor_matrix * baseline_extinction_matrix_metres01
    )

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=1000 * extinction_matrix_metres01[0, :],
        heights_m_agl=GRID_HEIGHTS_M_AGL,
        line_width=4, use_log_scale=True, line_colour=EXTINCTION_COLOUR
    )

    axes_object.set_xlabel(r'Aerosol extinction (km$^{-1}$)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('Step 7: Compute actual\nextinction profile')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(e)')

    panel_file_names.append(
        '{0:s}/aerosol_step5.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    concat_figure_file_name = '{0:s}/aerosol_schematic.jpg'.format(
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
