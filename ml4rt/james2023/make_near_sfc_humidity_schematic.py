"""Makes schematic to show how near-sfc humidity is perturbed."""

import os
import argparse
import numpy
import xarray
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import moisture_conversions as moisture_conv
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
PRESSURE_KEY = 'pres'
TROPICAL_STANDARD_ATMO_INDEX = 0

KM_TO_METRES = 1000.
MB_TO_PASCALS = 100.
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

WHOLE_PROFILE_TEMP_OFFSET_KELVINS = 11.
SURFACE_RELATIVE_HUMIDITY = 1.
MOIST_LAYER_DEPTH_METRES = 3000.
ORIG_MOIST_LAYER_DEPTH_PX = 5

PANEL_SIZE_PX = int(2.5e6)
CONCAT_FIGURE_SIZE_PX = int(1e7)

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

PERTURBATION_COLOUR = numpy.full(3, 152. / 255)
THREE_PREDICTOR_COLOURS = [
    numpy.array([117, 112, 179], dtype=float) / 255,
    numpy.array([27, 158, 119], dtype=float) / 255,
    numpy.array([217, 95, 2], dtype=float) / 255
]

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
    """Makes schematic to show how near-sfc humidity is perturbed.

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
    orig_pressures_pa = (
        MB_TO_PASCALS *
        tgt[PRESSURE_KEY].values[:, TROPICAL_STANDARD_ATMO_INDEX]
    )
    orig_heights_m_agl = KM_TO_METRES * tgt.coords[HEIGHT_DIM].values

    interp_object = interp1d(
        x=orig_heights_m_agl, y=orig_temperatures_kelvins, kind='cubic',
        bounds_error=True, assume_sorted=True
    )
    temperatures_kelvins = interp_object(HEIGHTS_M_AGL)

    interp_object = interp1d(
        x=orig_heights_m_agl, y=orig_pressures_pa, kind='cubic',
        bounds_error=True, assume_sorted=True
    )
    pressures_pa = interp_object(HEIGHTS_M_AGL)

    relative_humidities = numpy.full(pressures_pa.shape, 0.5)
    relative_humidities[:ORIG_MOIST_LAYER_DEPTH_PX] = numpy.linspace(
        0.65, 0.75, num=ORIG_MOIST_LAYER_DEPTH_PX
    )

    dewpoints_kelvins = moisture_conv.relative_humidity_to_dewpoint(
        relative_humidities=relative_humidities,
        total_pressures_pascals=pressures_pa,
        temperatures_kelvins=temperatures_kelvins
    )
    specific_humidities_kg_kg01 = moisture_conv.dewpoint_to_specific_humidity(
        dewpoints_kelvins=dewpoints_kelvins,
        total_pressures_pascals=pressures_pa,
        temperatures_kelvins=temperatures_kelvins
    )
    mixing_ratios_kg_kg01 = moisture_conv.specific_humidity_to_mixing_ratio(
        specific_humidities_kg_kg01
    )
    mixing_ratios_kg_kg01[-1] = 0.

    increasing_indices = numpy.where(numpy.diff(mixing_ratios_kg_kg01) > 0)[0]
    increasing_indices = increasing_indices[
        increasing_indices > ORIG_MOIST_LAYER_DEPTH_PX
    ]
    mixing_ratios_kg_kg01[increasing_indices[0]:] = 0.

    specific_humidities_kg_kg01 = (
        moisture_conv.mixing_ratio_to_specific_humidity(mixing_ratios_kg_kg01)
    )
    dewpoints_kelvins = moisture_conv.specific_humidity_to_dewpoint(
        specific_humidities_kg_kg01=specific_humidities_kg_kg01,
        temperatures_kelvins=temperatures_kelvins,
        total_pressures_pascals=pressures_pa
    )

    predictor_matrix = numpy.transpose(numpy.vstack((
        mixing_ratios_kg_kg01, dewpoints_kelvins, temperatures_kelvins
    )))
    predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)
    predictor_names = [
        example_utils.MIXING_RATIO_NAME,
        example_utils.DEWPOINT_NAME,
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
        predictor_colours=THREE_PREDICTOR_COLOURS,
        predictor_line_widths=numpy.full(shape=3, fill_value=4.),
        predictor_line_styles=['solid'] * 3,
        use_log_scale=True, include_units=True, handle_dict=None
    )

    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
    axes_objects = handle_dict[profile_plotting.AXES_OBJECTS_KEY]

    orig_dewpoint_limits_deg_c = axes_objects[1].get_xlim()
    orig_temp_limits_deg_c = axes_objects[2].get_xlim()

    new_limits_deg_c = numpy.array([
        orig_temp_limits_deg_c[0], orig_dewpoint_limits_deg_c[1]
    ])
    axes_objects[1].set_xlim(new_limits_deg_c)
    axes_objects[2].set_xlim(new_limits_deg_c)

    axes_objects[0].set_title('Original thermodynamic profiles')
    gg_plotting_utils.label_axes(
        axes_object=axes_objects[0], label_string='(a)', font_size=30
    )

    panel_file_names = [
        '{0:s}/near_surface_humidity_part1.jpg'.format(output_dir_name)
    ]

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    layer_height_indices = perturb_gfs._heights_to_grid_indices(
        min_height_m_agl=0., max_height_m_agl=MOIST_LAYER_DEPTH_METRES,
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

    (
        surface_dewpoint_kelvins_as_array
    ) = moisture_conv.relative_humidity_to_dewpoint(
        relative_humidities=numpy.array([SURFACE_RELATIVE_HUMIDITY]),
        temperatures_kelvins=temperatures_kelvins[[0]],
        total_pressures_pascals=pressures_pa[[0]]
    )

    (
        surface_specific_humidity_kg_kg01_as_array
    ) = moisture_conv.dewpoint_to_specific_humidity(
        dewpoints_kelvins=surface_dewpoint_kelvins_as_array,
        temperatures_kelvins=temperatures_kelvins[[0]],
        total_pressures_pascals=pressures_pa[[0]]
    )

    surface_mixing_ratio_kg_kg01 = (
        moisture_conv.specific_humidity_to_mixing_ratio(
            surface_specific_humidity_kg_kg01_as_array
        )[0]
    )
    surface_mixr_increase_kg_kg01 = (
        surface_mixing_ratio_kg_kg01 - mixing_ratios_kg_kg01[0]
    )

    if surface_mixr_increase_kg_kg01 <= 0:
        return

    layer_heights_m_agl = HEIGHTS_M_AGL[layer_height_indices]
    height_diffs_relative = layer_heights_m_agl / numpy.max(layer_heights_m_agl)

    mixing_ratio_increases_kg_kg01 = numpy.full(temperatures_kelvins.shape, 0.)
    mixing_ratio_increases_kg_kg01[layer_height_indices] = (
        (1. - height_diffs_relative) * surface_mixr_increase_kg_kg01
    )

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=KG_TO_GRAMS * mixing_ratio_increases_kg_kg01,
        heights_m_agl=HEIGHTS_M_AGL,
        line_width=4, line_style='dashed', use_log_scale=True,
        line_colour=PERTURBATION_COLOUR
    )

    axes_object.set_xlabel(r'Perturbation (g kg$^{-1}$)')
    axes_object.set_ylabel('Height (km AGL)')
    axes_object.set_title('Mixing-ratio perturbations')
    gg_plotting_utils.label_axes(axes_object=axes_object, label_string='(b)')

    panel_file_names.append(
        '{0:s}/near_surface_humidity_part2.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    mixing_ratios_kg_kg01 += mixing_ratio_increases_kg_kg01
    specific_humidities_kg_kg01 = (
        moisture_conv.mixing_ratio_to_specific_humidity(mixing_ratios_kg_kg01)
    )
    dewpoints_kelvins = moisture_conv.specific_humidity_to_dewpoint(
        specific_humidities_kg_kg01=specific_humidities_kg_kg01,
        temperatures_kelvins=temperatures_kelvins,
        total_pressures_pascals=pressures_pa
    )

    predictor_matrix = numpy.transpose(numpy.vstack((
        mixing_ratios_kg_kg01, dewpoints_kelvins, temperatures_kelvins
    )))
    predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)
    predictor_names = [
        example_utils.MIXING_RATIO_NAME,
        example_utils.DEWPOINT_NAME,
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
        predictor_colours=THREE_PREDICTOR_COLOURS,
        predictor_line_widths=numpy.full(shape=3, fill_value=4.),
        predictor_line_styles=['solid'] * 3,
        use_log_scale=True, include_units=True, handle_dict=None
    )

    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
    axes_objects = handle_dict[profile_plotting.AXES_OBJECTS_KEY]

    orig_dewpoint_limits_deg_c = axes_objects[1].get_xlim()
    orig_temp_limits_deg_c = axes_objects[2].get_xlim()

    new_limits_deg_c = numpy.array([
        orig_temp_limits_deg_c[0], orig_dewpoint_limits_deg_c[1]
    ])
    axes_objects[1].set_xlim(new_limits_deg_c)
    axes_objects[2].set_xlim(new_limits_deg_c)

    axes_objects[0].set_title('New thermodynamic profiles')
    gg_plotting_utils.label_axes(
        axes_object=axes_objects[0], label_string='(c)', font_size=30
    )

    panel_file_names.append(
        '{0:s}/near_surface_humidity_part3.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    dewpoints_kelvins = numpy.minimum(dewpoints_kelvins, temperatures_kelvins)
    specific_humidities_kg_kg01 = moisture_conv.dewpoint_to_specific_humidity(
        dewpoints_kelvins=dewpoints_kelvins,
        temperatures_kelvins=temperatures_kelvins,
        total_pressures_pascals=pressures_pa
    )
    mixing_ratios_kg_kg01 = moisture_conv.specific_humidity_to_mixing_ratio(
        specific_humidities_kg_kg01
    )

    predictor_matrix = numpy.transpose(numpy.vstack((
        mixing_ratios_kg_kg01, dewpoints_kelvins, temperatures_kelvins
    )))
    predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)
    example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY] = predictor_matrix

    handle_dict = profile_plotting.plot_predictors(
        example_dict=example_dict, example_index=0,
        predictor_names=predictor_names,
        predictor_colours=THREE_PREDICTOR_COLOURS,
        predictor_line_widths=numpy.full(shape=3, fill_value=4.),
        predictor_line_styles=['solid'] * 3,
        use_log_scale=True, include_units=True, handle_dict=None
    )

    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
    axes_objects = handle_dict[profile_plotting.AXES_OBJECTS_KEY]

    orig_dewpoint_limits_deg_c = axes_objects[1].get_xlim()
    orig_temp_limits_deg_c = axes_objects[2].get_xlim()

    new_limits_deg_c = numpy.array([
        orig_temp_limits_deg_c[0], orig_dewpoint_limits_deg_c[1]
    ])
    axes_objects[1].set_xlim(new_limits_deg_c)
    axes_objects[2].set_xlim(new_limits_deg_c)

    title_string = 'New thermodynamic profiles\n' + r'(temperature $\leq$ dewpoint)'
    axes_objects[0].set_title(title_string)
    gg_plotting_utils.label_axes(
        axes_object=axes_objects[0], label_string='(d)', font_size=30
    )

    panel_file_names.append(
        '{0:s}/near_surface_humidity_part4.jpg'.format(output_dir_name)
    )

    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))
    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    mixing_ratios_kg_kg01 = numpy.minimum(
        mixing_ratios_kg_kg01, perturb_gfs.MAX_MIXING_RATIO_EVER_KG_KG01
    )
    specific_humidities_kg_kg01 = (
        moisture_conv.mixing_ratio_to_specific_humidity(mixing_ratios_kg_kg01)
    )
    dewpoints_kelvins = moisture_conv.specific_humidity_to_dewpoint(
        specific_humidities_kg_kg01=specific_humidities_kg_kg01,
        temperatures_kelvins=temperatures_kelvins,
        total_pressures_pascals=pressures_pa
    )

    predictor_matrix = numpy.transpose(numpy.vstack((
        mixing_ratios_kg_kg01, dewpoints_kelvins, temperatures_kelvins
    )))
    predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)
    example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY] = predictor_matrix

    handle_dict = profile_plotting.plot_predictors(
        example_dict=example_dict, example_index=0,
        predictor_names=predictor_names,
        predictor_colours=THREE_PREDICTOR_COLOURS,
        predictor_line_widths=numpy.full(shape=3, fill_value=4.),
        predictor_line_styles=['solid'] * 3,
        use_log_scale=True, include_units=True, handle_dict=None
    )

    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
    axes_objects = handle_dict[profile_plotting.AXES_OBJECTS_KEY]

    orig_dewpoint_limits_deg_c = axes_objects[1].get_xlim()
    orig_temp_limits_deg_c = axes_objects[2].get_xlim()

    new_limits_deg_c = numpy.array([
        orig_temp_limits_deg_c[0], orig_dewpoint_limits_deg_c[1]
    ])
    axes_objects[1].set_xlim(new_limits_deg_c)
    axes_objects[2].set_xlim(new_limits_deg_c)

    title_string = (
        'New thermodynamic profiles\n' + r'(mixing ratio $\leq$ 40 g kg$^{-1}$)'
    )
    axes_objects[0].set_title(title_string)
    gg_plotting_utils.label_axes(
        axes_object=axes_objects[0], label_string='(e)', font_size=30
    )

    panel_file_names.append(
        '{0:s}/near_surface_humidity_part5.jpg'.format(output_dir_name)
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
        '{0:s}/near_surface_humidity_schematic.jpg'
    ).format(output_dir_name)

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
