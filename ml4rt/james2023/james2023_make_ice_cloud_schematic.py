"""Makes schematic to show how ice-cloud layers are created."""

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
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4rt.utils import example_utils
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
KG_TO_GRAMS = 1000.

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

ORIG_WATER_CONTENTS_KG_M03 = numpy.full(HEIGHTS_M_AGL.shape, 0.)
ORIG_WATER_CONTENTS_KG_M03[55:64] = 0.001 * numpy.array([
    0.003691, 0.035799, 0.111804, 0.251723, 0.460531, 0.614783, 0.761428,
    0.693646, 0.347054
])

WHOLE_PROFILE_TEMP_OFFSET_KELVINS = -10.
MAX_CLOUD_THICKNESS_METRES = 5000.
MAX_NEW_WATER_CONTENT_KG_M03 = 0.005
WATER_CONTENT_NOISE_KG_M03 = 0.001

PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

WATER_CONTENT_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
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
    """Makes schematic to show how ice-cloud layers are created.

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
        WHOLE_PROFILE_TEMP_OFFSET_KELVINS +
        tgt[TEMPERATURE_KEY].values[:, TROPICAL_STANDARD_ATMO_INDEX]
    )
    orig_heights_m_agl = KM_TO_METRES * tgt.coords[HEIGHT_DIM].values

    interp_object = interp1d(
        x=orig_heights_m_agl, y=orig_temperatures_kelvins, kind='cubic',
        bounds_error=True, assume_sorted=True
    )
    temperatures_kelvins = interp_object(HEIGHTS_M_AGL)
    water_contents_kg_m03 = ORIG_WATER_CONTENTS_KG_M03 + 0.

    predictor_matrix = numpy.transpose(numpy.vstack((
        water_contents_kg_m03, temperatures_kelvins
    )))
    predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)
    predictor_names = [
        example_utils.ICE_WATER_CONTENT_NAME,
        example_utils.TEMPERATURE_NAME
    ]
    example_dict = {
        example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL,
        example_utils.VECTOR_PREDICTOR_VALS_KEY: predictor_matrix,
        example_utils.VECTOR_PREDICTOR_NAMES_KEY: predictor_names
    }

    handle_dict = profile_plotting.plot_predictors(
        example_dict=example_dict, example_index=0,
        predictor_names=predictor_names,
        predictor_colours=[WATER_CONTENT_COLOUR, TEMPERATURE_COLOUR],
        predictor_line_widths=numpy.full(shape=2, fill_value=4.),
        predictor_line_styles=['solid'] * 2,
        use_log_scale=True, include_units=True, handle_dict=None
    )

    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
    axes_objects = handle_dict[profile_plotting.AXES_OBJECTS_KEY]

    axes_objects[0].set_title('Original profiles')
    gg_plotting_utils.label_axes(
        axes_object=axes_objects[0], label_string='(a)', font_size=30
    )

    panel_file_names = [
        '{0:s}/ice_cloud_part1.jpg'.format(output_dir_name)
    ]

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    tropopause_height_m_agl, _ = perturb_gfs._find_tropopause(
        temperatures_kelvins=temperatures_kelvins,
        sorted_heights_m_agl=HEIGHTS_M_AGL
    )
    if tropopause_height_m_agl is None:
        tropopause_height_m_agl = perturb_gfs.MIN_TROPOPAUSE_HEIGHT_EVER_M_AGL

    num_cloud_layers = 2
    height_indices_by_layer_no_constraints = [
        numpy.where(water_contents_kg_m03 > 0)[0],
        numpy.array([], dtype=int)
    ]
    height_indices_by_layer = [
        numpy.where(water_contents_kg_m03 > 0)[0],
        numpy.array([], dtype=int)
    ]

    for k in range(1, num_cloud_layers):
        this_top_height_m_agl = HEIGHTS_M_AGL[
            numpy.where(water_contents_kg_m03 > 0)[0][-1]
        ]
        this_bottom_height_m_agl = max([
            this_top_height_m_agl - MAX_CLOUD_THICKNESS_METRES,
            0.
        ])

        height_indices_by_layer_no_constraints[k] = (
            perturb_gfs._heights_to_grid_indices(
                min_height_m_agl=this_bottom_height_m_agl,
                max_height_m_agl=this_top_height_m_agl,
                sorted_grid_heights_m_agl=HEIGHTS_M_AGL
            )
        )

        good_temperature_flags = (
            temperatures_kelvins[height_indices_by_layer_no_constraints[k]]
            < 273.15
        )
        height_indices_by_layer[k] = (
            height_indices_by_layer_no_constraints[k][good_temperature_flags]
        )

        test_indices = numpy.array([], dtype=int)

        for m in range(k):
            if len(height_indices_by_layer[m]) == 0:
                continue

            test_indices = numpy.concatenate((
                test_indices,
                height_indices_by_layer[m][[0]] - 1,
                height_indices_by_layer[m],
                height_indices_by_layer[m][[-1]] + 1
            ))

        height_indices_by_layer[k] = height_indices_by_layer[k][
            numpy.invert(numpy.isin(height_indices_by_layer[k], test_indices))
        ]

        if len(height_indices_by_layer[k]) < 2:
            height_indices_by_layer[k] = numpy.array([], dtype=int)
            continue

    if len(height_indices_by_layer[1]) == 0:
        return

    handle_dict = profile_plotting.plot_predictors(
        example_dict=example_dict, example_index=0,
        predictor_names=predictor_names,
        predictor_colours=[WATER_CONTENT_COLOUR, TEMPERATURE_COLOUR],
        predictor_line_widths=numpy.full(shape=2, fill_value=4.),
        predictor_line_styles=['solid'] * 2,
        use_log_scale=True, include_units=True, handle_dict=None
    )

    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
    axes_objects = handle_dict[profile_plotting.AXES_OBJECTS_KEY]

    min_height_new_layer_km_agl = (
        METRES_TO_KM *
        HEIGHTS_M_AGL[height_indices_by_layer_no_constraints[1][0]]
    )
    max_height_new_layer_km_agl = (
        METRES_TO_KM *
        HEIGHTS_M_AGL[height_indices_by_layer_no_constraints[1][-1]]
    )
    polygon_y_coords = numpy.array([
        min_height_new_layer_km_agl, max_height_new_layer_km_agl,
        max_height_new_layer_km_agl, min_height_new_layer_km_agl,
        min_height_new_layer_km_agl
    ])

    x_min = axes_objects[0].get_xlim()[0]
    x_max = axes_objects[0].get_xlim()[1]
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
    axes_objects[0].add_patch(patch_object)

    axes_objects[0].set_title('Original profiles + extent of new cloud')
    gg_plotting_utils.label_axes(
        axes_object=axes_objects[0], label_string='(b)', font_size=30
    )

    panel_file_names.append(
        '{0:s}/ice_cloud_part2.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    handle_dict = profile_plotting.plot_predictors(
        example_dict=example_dict, example_index=0,
        predictor_names=predictor_names,
        predictor_colours=[WATER_CONTENT_COLOUR, TEMPERATURE_COLOUR],
        predictor_line_widths=numpy.full(shape=2, fill_value=4.),
        predictor_line_styles=['solid'] * 2,
        use_log_scale=True, include_units=True, handle_dict=None
    )

    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
    axes_objects = handle_dict[profile_plotting.AXES_OBJECTS_KEY]

    min_height_new_layer_km_agl = (
        METRES_TO_KM * HEIGHTS_M_AGL[height_indices_by_layer[1][0]]
    )
    max_height_new_layer_km_agl = (
        METRES_TO_KM * HEIGHTS_M_AGL[height_indices_by_layer[1][-1]]
    )
    polygon_y_coords = numpy.array([
        min_height_new_layer_km_agl, max_height_new_layer_km_agl,
        max_height_new_layer_km_agl, min_height_new_layer_km_agl,
        min_height_new_layer_km_agl
    ])

    x_min = axes_objects[0].get_xlim()[0]
    x_max = axes_objects[0].get_xlim()[1]
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
    axes_objects[0].add_patch(patch_object)

    axes_objects[0].set_title(
        'Original profiles + extent of new cloud\n' +
        r'(temperature $<$ 0 $^{\circ}$C and no overlap)'
    )
    gg_plotting_utils.label_axes(
        axes_object=axes_objects[0], label_string='(c)', font_size=30
    )

    panel_file_names.append(
        '{0:s}/ice_cloud_part3.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    for k in range(1, num_cloud_layers):
        enhanced_height_indices = numpy.concatenate((
            height_indices_by_layer[k][[0]] - 1,
            height_indices_by_layer[k],
            height_indices_by_layer[k][[-1]] + 1
        ))

        enhanced_layer_heights_m_agl = HEIGHTS_M_AGL[enhanced_height_indices]
        layer_heights_m_agl = HEIGHTS_M_AGL[height_indices_by_layer[k]]

        layer_center_m_agl = numpy.mean(layer_heights_m_agl)
        max_height_diff_metres = numpy.max(
            numpy.absolute(layer_center_m_agl - enhanced_layer_heights_m_agl)
        )

        layer_height_diffs_metres = numpy.absolute(
            layer_center_m_agl - layer_heights_m_agl
        )
        layer_height_diffs_relative = (
            layer_height_diffs_metres / max_height_diff_metres
        )

        layer_water_contents_kg_m03 = (
            (1. - layer_height_diffs_relative) * MAX_NEW_WATER_CONTENT_KG_M03
        )
        water_contents_kg_m03[height_indices_by_layer[k]] = (
            layer_water_contents_kg_m03
        )

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=KG_TO_GRAMS * water_contents_kg_m03,
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, line_style='solid', use_log_scale=True,
        line_colour=WATER_CONTENT_COLOUR
    )

    axes_object.set_xlabel(r'Ice-water content (g m$^{-3}$)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('IWC profile with new cloud')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(d)')

    panel_file_names.append(
        '{0:s}/ice_cloud_part4.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    for k in range(1, num_cloud_layers):
        while True:
            noise_values_kg_m03 = numpy.random.normal(
                loc=0., scale=WATER_CONTENT_NOISE_KG_M03,
                size=len(height_indices_by_layer[k])
            )
            new_water_contents_kg_m03 = (
                water_contents_kg_m03[height_indices_by_layer[k]] +
                noise_values_kg_m03
            )

            if not numpy.any(new_water_contents_kg_m03 < 0):
                continue
            if not numpy.any(
                    new_water_contents_kg_m03 > MAX_NEW_WATER_CONTENT_KG_M03
            ):
                continue

            water_contents_kg_m03[height_indices_by_layer[k]] += (
                noise_values_kg_m03
            )
            break

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=KG_TO_GRAMS * water_contents_kg_m03,
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, line_style='solid', use_log_scale=True,
        line_colour=WATER_CONTENT_COLOUR
    )

    axes_object.set_xlabel(r'Ice-water content (g m$^{-3}$)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title(
        'IWC profile with new cloud\n(including Gaussian noise)'
    )
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(e)')

    panel_file_names.append(
        '{0:s}/ice_cloud_part5.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    water_contents_kg_m03 = numpy.maximum(water_contents_kg_m03, 0.)
    water_contents_kg_m03 = numpy.minimum(
        water_contents_kg_m03, MAX_NEW_WATER_CONTENT_KG_M03
    )

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=KG_TO_GRAMS * water_contents_kg_m03,
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, line_style='solid', use_log_scale=True,
        line_colour=WATER_CONTENT_COLOUR
    )

    axes_object.set_xlabel(r'Ice-water content (g m$^{-3}$)')
    axes_object.set_ylabel('Height (km AGL)')

    title_string = (
        'IWC profile with new cloud\n(bounded to 0...{0:.0f}'
    ).format(KG_TO_GRAMS * MAX_NEW_WATER_CONTENT_KG_M03)

    title_string += r' g m$^{-3}$)'
    axes_object.set_title(title_string)
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(f)')

    panel_file_names.append(
        '{0:s}/ice_cloud_part6.jpg'.format(output_dir_name)
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

    concat_figure_file_name = '{0:s}/ice_cloud_schematic.jpg'.format(
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
