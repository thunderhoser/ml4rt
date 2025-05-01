"""For each data example, plots data at different levels of perturbation."""

import os
import sys
import argparse
import numpy
import xarray
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import moisture_conversions as moisture_conv
import file_system_utils
import error_checking
import imagemagick_utils
import rrtm_io
import example_io
import example_utils
import profile_plotting
import perturb_gfs_for_rrtm
import prepare_gfs_for_rrtm_no_interp as prepare_gfs_for_rrtm

MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

GRID_LINE_COLOUR = numpy.full(3, 152. / 255)
GRID_LINE_WIDTH = 1.

CONVERT_EXE_NAME = 'convert'
TITLE_FONT_SIZE = 150
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

LWC_COLOUR = numpy.array([166, 206, 227], dtype=float) / 255
IWC_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
TEMPERATURE_COLOUR = numpy.full(3, 0.)
OZONE_COLOUR = numpy.array([51, 160, 44], dtype=float) / 255
HUMIDITY_COLOUR = numpy.array([251, 154, 153], dtype=float) / 255

PREDICTOR_NAMES = [
    example_utils.LIQUID_WATER_CONTENT_NAME,
    example_utils.ICE_WATER_CONTENT_NAME,
    example_utils.TEMPERATURE_NAME,
    example_utils.O3_MIXING_RATIO_NAME,
    example_utils.SPECIFIC_HUMIDITY_NAME
]

PREDICTOR_COLOURS = [
    LWC_COLOUR,
    IWC_COLOUR,
    TEMPERATURE_COLOUR,
    OZONE_COLOUR,
    HUMIDITY_COLOUR
]

LINE_WIDTH = 3
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_example_file_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing unperturbed data.  Will be read by '
    '`example_io.read_file`.'
)
NUM_EXAMPLES_HELP_STRING = (
    'Number of data examples to plot.  These will be randomly selected from '
    'the file.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _put_perturbed_values_in_example_dict(gfs_table_xarray, example_dict):
    """Puts perturbed values in example dictionary.

    :param gfs_table_xarray: xarray table containing only one example.
    :param example_dict: Dictionary in format returned by
        `example_io.read_file`, containing only one example.
    :return: example_dict: Same as input but with perturbed values.
    """

    gfs_tbl = gfs_table_xarray

    k = example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY].index(
        example_utils.TEMPERATURE_NAME
    )
    example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY][0, :, k] = (
        gfs_tbl[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[0, 0, :]
    )

    k = example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY].index(
        example_utils.SPECIFIC_HUMIDITY_NAME
    )
    example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY][0, :, k] = (
        moisture_conv.mixing_ratio_to_specific_humidity(
            gfs_tbl[prepare_gfs_for_rrtm.VAPOUR_MIXR_KEY_KG_KG01].values[0, 0, :]
        )
    )

    k = example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY].index(
        example_utils.O3_MIXING_RATIO_NAME
    )
    example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY][0, :, k] = (
        gfs_tbl[prepare_gfs_for_rrtm.OZONE_MIXR_KEY_KG_KG01].values[0, 0, :]
    )

    k = example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY].index(
        example_utils.LIQUID_WATER_CONTENT_NAME
    )
    new_lwc_matrix_kg_m03 = rrtm_io._layerwise_water_path_to_content(
        layerwise_path_matrix_kg_m02=
        gfs_tbl[prepare_gfs_for_rrtm.LIQUID_WATER_PATH_KEY_KG_M02].values[[0], 0, :],
        heights_m_agl=
        gfs_tbl[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[0, 0, :]
    )
    example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY][0, :, k] = (
        new_lwc_matrix_kg_m03[0, :]
    )

    k = example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY].index(
        example_utils.ICE_WATER_CONTENT_NAME
    )
    new_iwc_matrix_kg_m03 = rrtm_io._layerwise_water_path_to_content(
        layerwise_path_matrix_kg_m02=
        gfs_tbl[prepare_gfs_for_rrtm.ICE_WATER_PATH_KEY_KG_M02].values[[0], 0, :],
        heights_m_agl=
        gfs_tbl[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[0, 0, :]
    )
    example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY][0, :, k] = (
        new_iwc_matrix_kg_m03[0, :]
    )

    return example_dict


def _overlay_text(
        image_file_name, x_offset_from_left_px, y_offset_from_top_px,
        text_string):
    """Creates two figures showing overall evaluation of uncertainty quant (UQ).

    :param image_file_name: Path to image file.
    :param x_offset_from_left_px: Left-relative x-coordinate (pixels).
    :param y_offset_from_top_px: Top-relative y-coordinate (pixels).
    :param text_string: String to overlay.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    command_string = (
        '"{0:s}" "{1:s}" -pointsize {2:d} -font "{3:s}" '
        '-fill "rgb(0, 0, 0)" -annotate {4:+d}{5:+d} "{6:s}" "{1:s}"'
    ).format(
        CONVERT_EXE_NAME, image_file_name, TITLE_FONT_SIZE, TITLE_FONT_NAME,
        x_offset_from_left_px, y_offset_from_top_px, text_string
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _plot_one_example(example_dict, example_index, output_dir_name):
    """Plots one data example at different levels of perturbation.

    :param example_dict: Dictionary returned by `example_io.read_file`.
    :param example_index: Will plot the [k]th example, where k = example_index.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    """

    example_id_string = (
        example_dict[example_utils.EXAMPLE_IDS_KEY][example_index]
    )

    handle_dict = profile_plotting.plot_predictors(
        example_dict=example_dict,
        example_index=example_index,
        predictor_names=PREDICTOR_NAMES,
        predictor_colours=PREDICTOR_COLOURS,
        predictor_line_widths=numpy.full(len(PREDICTOR_NAMES), LINE_WIDTH),
        predictor_line_styles=['solid'] * len(PREDICTOR_NAMES),
        use_log_scale=True,
        all_axes_on_bottom=True
    )
    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
    axes_objects = handle_dict[profile_plotting.AXES_OBJECTS_KEY]

    axes_objects[0].grid(
        which='major', axis='y',
        color=GRID_LINE_COLOUR, linewidth=GRID_LINE_WIDTH, linestyle='dashed'
    )
    axes_objects[0].set_title('Clean')

    this_file_name = '{0:s}/{1:s}_clean.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )
    panel_file_names = [this_file_name]

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    num_heights = len(example_dict[example_utils.HEIGHTS_KEY])
    coord_dict = {
        prepare_gfs_for_rrtm.TIME_DIMENSION: numpy.array(
            [example_dict[example_utils.VALID_TIMES_KEY][example_index]],
            dtype=int
        ),
        prepare_gfs_for_rrtm.HEIGHT_DIMENSION: numpy.linspace(
            0, num_heights - 1, num=num_heights, dtype=int
        ),
        prepare_gfs_for_rrtm.SITE_DIMENSION: numpy.array([0], dtype=int)
    }

    these_dims_3d = (
        prepare_gfs_for_rrtm.TIME_DIMENSION,
        prepare_gfs_for_rrtm.SITE_DIMENSION,
        prepare_gfs_for_rrtm.HEIGHT_DIMENSION
    )
    these_dims_2d = (
        prepare_gfs_for_rrtm.TIME_DIMENSION,
        prepare_gfs_for_rrtm.SITE_DIMENSION
    )

    temp_matrix_kelvins = example_utils.get_field_from_dict(
        example_dict=example_dict, field_name=example_utils.TEMPERATURE_NAME
    )
    temp_matrix_kelvins = numpy.expand_dims(temp_matrix_kelvins[[0], :], axis=1)
    height_matrix_m_agl = example_utils.get_field_from_dict(
        example_dict=example_dict, field_name=example_utils.HEIGHT_NAME
    )
    height_matrix_m_agl = numpy.expand_dims(height_matrix_m_agl[[0], :], axis=1)
    pressure_matrix_pascals = example_utils.get_field_from_dict(
        example_dict=example_dict, field_name=example_utils.PRESSURE_NAME
    )
    pressure_matrix_pascals = numpy.expand_dims(
        pressure_matrix_pascals[[0], :], axis=1
    )
    mixing_ratio_matrix_kg_kg01 = moisture_conv.specific_humidity_to_mixing_ratio(
        example_utils.get_field_from_dict(
            example_dict=example_dict,
            field_name=example_utils.SPECIFIC_HUMIDITY_NAME
        )
    )
    mixing_ratio_matrix_kg_kg01 = numpy.expand_dims(
        mixing_ratio_matrix_kg_kg01[[0], :], axis=1
    )
    surface_temp_kelvins = example_utils.parse_example_ids(
        [example_id_string]
    )[example_utils.TEMPERATURES_10M_KEY][0]
    surface_temp_matrix_kelvins = numpy.full((1, 1), surface_temp_kelvins)
    ozone_mixing_ratio_matrix_kg_kg01 = example_utils.get_field_from_dict(
        example_dict=example_dict, field_name=example_utils.O3_MIXING_RATIO_NAME
    )
    ozone_mixing_ratio_matrix_kg_kg01 = numpy.expand_dims(
        ozone_mixing_ratio_matrix_kg_kg01[[0], :], axis=1
    )

    lwc_matrix_kg_m03 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_utils.LIQUID_WATER_CONTENT_NAME
    )[[0], :]
    layerwise_lwp_matrix_kg_m02 = rrtm_io._water_content_to_layerwise_path(
        water_content_matrix_kg_m03=lwc_matrix_kg_m03,
        heights_m_agl=height_matrix_m_agl[0, 0, :]
    )
    layerwise_lwp_matrix_kg_m02 = numpy.expand_dims(
        layerwise_lwp_matrix_kg_m02, axis=1
    )

    iwc_matrix_kg_m03 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_utils.ICE_WATER_CONTENT_NAME
    )[[0], :]
    layerwise_iwp_matrix_kg_m02 = rrtm_io._water_content_to_layerwise_path(
        water_content_matrix_kg_m03=iwc_matrix_kg_m03,
        heights_m_agl=height_matrix_m_agl[0, 0, :]
    )
    layerwise_iwp_matrix_kg_m02 = numpy.expand_dims(
        layerwise_iwp_matrix_kg_m02, axis=1
    )

    main_data_dict = {
        prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS: (
            these_dims_3d, temp_matrix_kelvins
        ),
        prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL: (
            these_dims_3d, height_matrix_m_agl
        ),
        prepare_gfs_for_rrtm.PRESSURE_KEY_PASCALS: (
            these_dims_3d, pressure_matrix_pascals
        ),
        prepare_gfs_for_rrtm.VAPOUR_MIXR_KEY_KG_KG01: (
            these_dims_3d, mixing_ratio_matrix_kg_kg01
        ),
        prepare_gfs_for_rrtm.OZONE_MIXR_KEY_KG_KG01: (
            these_dims_3d, ozone_mixing_ratio_matrix_kg_kg01
        ),
        prepare_gfs_for_rrtm.SURFACE_TEMPERATURE_KEY: (
            these_dims_2d, surface_temp_matrix_kelvins
        ),
        prepare_gfs_for_rrtm.LIQUID_WATER_PATH_KEY_KG_M02: (
            these_dims_3d, layerwise_lwp_matrix_kg_m02
        ),
        prepare_gfs_for_rrtm.ICE_WATER_PATH_KEY_KG_M02: (
            these_dims_3d, layerwise_iwp_matrix_kg_m02
        )
    }

    gfs_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=coord_dict
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_ozone_layer(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        thickness_limits_metres=numpy.array([40000, 60000.]),
        center_limits_metres=numpy.array([20000, 50000.]),
        max_mixing_ratio_kg_kg01=0.00002,
        mixing_ratio_noise_stdev_kg_kg01=0.00000025
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_surface_based_warm_layer(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        max_layer_thickness_metres=1250.,
        max_temp_increase_kelvins=2.
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_surface_based_moist_layer(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        max_layer_thickness_metres=1250.,
        surface_relative_humidity_limits=numpy.array([0.5, 0.625])
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_cloud(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        max_num_cloud_layers=2,
        max_layer_thickness_metres=5000.,
        max_water_content_kg_m03=0.002,
        water_content_noise_stdev_kg_m03=0.00025,
        liquid_flag=True
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_cloud(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        max_num_cloud_layers=2,
        max_layer_thickness_metres=5000.,
        max_water_content_kg_m03=0.002,
        water_content_noise_stdev_kg_m03=0.00025,
        liquid_flag=False
    )

    new_example_dict = example_utils.subset_by_index(
        example_dict=example_dict, desired_indices=numpy.array([example_index])
    )
    new_example_dict = _put_perturbed_values_in_example_dict(
        gfs_table_xarray=gfs_table_xarray,
        example_dict=new_example_dict
    )
    handle_dict = profile_plotting.plot_predictors(
        example_dict=new_example_dict,
        example_index=0,
        predictor_names=PREDICTOR_NAMES,
        predictor_colours=PREDICTOR_COLOURS,
        predictor_line_widths=numpy.full(len(PREDICTOR_NAMES), LINE_WIDTH),
        predictor_line_styles=['solid'] * len(PREDICTOR_NAMES),
        use_log_scale=True,
        all_axes_on_bottom=True
    )
    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
    axes_objects = handle_dict[profile_plotting.AXES_OBJECTS_KEY]

    axes_objects[0].grid(
        which='major', axis='y',
        color=GRID_LINE_COLOUR, linewidth=GRID_LINE_WIDTH, linestyle='dashed'
    )
    axes_objects[0].set_yticklabels([''] * len(axes_objects[0].get_yticks()))
    axes_objects[0].set_ylabel('')
    axes_objects[0].set_title('Lightly perturbed')

    this_file_name = '{0:s}/{1:s}_lightly_perturbed.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )
    panel_file_names.append(this_file_name)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    gfs_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=coord_dict
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_ozone_layer(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        thickness_limits_metres=numpy.array([25000, 60000.]),
        center_limits_metres=numpy.array([20000, 50000.]),
        max_mixing_ratio_kg_kg01=0.000025,
        mixing_ratio_noise_stdev_kg_kg01=0.0000005
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_surface_based_warm_layer(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        max_layer_thickness_metres=2500.,
        max_temp_increase_kelvins=4.
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_surface_based_moist_layer(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        max_layer_thickness_metres=2500.,
        surface_relative_humidity_limits=numpy.array([0.5, 0.75])
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_cloud(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        max_num_cloud_layers=3,
        max_layer_thickness_metres=5000.,
        max_water_content_kg_m03=0.0025,
        water_content_noise_stdev_kg_m03=0.0005,
        liquid_flag=True
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_cloud(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        max_num_cloud_layers=3,
        max_layer_thickness_metres=5000.,
        max_water_content_kg_m03=0.0025,
        water_content_noise_stdev_kg_m03=0.0005,
        liquid_flag=False
    )

    new_example_dict = example_utils.subset_by_index(
        example_dict=example_dict, desired_indices=numpy.array([example_index])
    )
    new_example_dict = _put_perturbed_values_in_example_dict(
        gfs_table_xarray=gfs_table_xarray,
        example_dict=new_example_dict
    )
    handle_dict = profile_plotting.plot_predictors(
        example_dict=new_example_dict,
        example_index=0,
        predictor_names=PREDICTOR_NAMES,
        predictor_colours=PREDICTOR_COLOURS,
        predictor_line_widths=numpy.full(len(PREDICTOR_NAMES), LINE_WIDTH),
        predictor_line_styles=['solid'] * len(PREDICTOR_NAMES),
        use_log_scale=True,
        all_axes_on_bottom=True
    )
    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
    axes_objects = handle_dict[profile_plotting.AXES_OBJECTS_KEY]

    axes_objects[0].grid(
        which='major', axis='y',
        color=GRID_LINE_COLOUR, linewidth=GRID_LINE_WIDTH, linestyle='dashed'
    )
    axes_objects[0].set_title('Moderately perturbed')

    this_file_name = '{0:s}/{1:s}_moderately_perturbed.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )
    panel_file_names.append(this_file_name)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    gfs_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=coord_dict
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_ozone_layer(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        thickness_limits_metres=numpy.array([100, 60000.]),
        center_limits_metres=numpy.array([20000, 50000.]),
        max_mixing_ratio_kg_kg01=0.00005,
        mixing_ratio_noise_stdev_kg_kg01=0.000001
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_surface_based_warm_layer(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        max_layer_thickness_metres=5000.,
        max_temp_increase_kelvins=8.
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_surface_based_moist_layer(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        max_layer_thickness_metres=5000.,
        surface_relative_humidity_limits=numpy.array([0.5, 1])
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_cloud(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        max_num_cloud_layers=5,
        max_layer_thickness_metres=5000.,
        max_water_content_kg_m03=0.005,
        water_content_noise_stdev_kg_m03=0.001,
        liquid_flag=True
    )
    gfs_table_xarray = perturb_gfs_for_rrtm._create_cloud(
        gfs_table_xarray=gfs_table_xarray,
        time_index=0,
        site_index=0,
        max_num_cloud_layers=5,
        max_layer_thickness_metres=5000.,
        max_water_content_kg_m03=0.005,
        water_content_noise_stdev_kg_m03=0.001,
        liquid_flag=False
    )

    new_example_dict = example_utils.subset_by_index(
        example_dict=example_dict, desired_indices=numpy.array([example_index])
    )
    new_example_dict = _put_perturbed_values_in_example_dict(
        gfs_table_xarray=gfs_table_xarray,
        example_dict=new_example_dict
    )
    handle_dict = profile_plotting.plot_predictors(
        example_dict=new_example_dict,
        example_index=0,
        predictor_names=PREDICTOR_NAMES,
        predictor_colours=PREDICTOR_COLOURS,
        predictor_line_widths=numpy.full(len(PREDICTOR_NAMES), LINE_WIDTH),
        predictor_line_styles=['solid'] * len(PREDICTOR_NAMES),
        use_log_scale=True,
        all_axes_on_bottom=True
    )
    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
    axes_objects = handle_dict[profile_plotting.AXES_OBJECTS_KEY]

    axes_objects[0].grid(
        which='major', axis='y',
        color=GRID_LINE_COLOUR, linewidth=GRID_LINE_WIDTH, linestyle='dashed'
    )
    axes_objects[0].set_yticklabels([''] * len(axes_objects[0].get_yticks()))
    axes_objects[0].set_ylabel('')
    axes_objects[0].set_title('Heavily perturbed')

    this_file_name = '{0:s}/{1:s}_heavily_perturbed.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )
    panel_file_names.append(this_file_name)

    print('Saving figure to: "{0:s}"...'.format(this_file_name))
    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    panel_letter = None

    for this_file_name in panel_file_names:
        if panel_letter is None:
            panel_letter = 'a'
        else:
            panel_letter = chr(ord(panel_letter) + 1)

        imagemagick_utils.trim_whitespace(
            input_file_name=this_file_name,
            output_file_name=this_file_name,
            border_width_pixels=TITLE_FONT_SIZE + 75
        )
        _overlay_text(
            image_file_name=this_file_name,
            x_offset_from_left_px=TITLE_FONT_SIZE + 50,
            y_offset_from_top_px=TITLE_FONT_SIZE + 100,
            text_string='({0:s})'.format(panel_letter)
        )
        imagemagick_utils.trim_whitespace(
            input_file_name=this_file_name,
            output_file_name=this_file_name
        )

    num_panels = len(panel_file_names)
    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_panels)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_panels) / num_panel_rows
    ))
    concat_file_name = '{0:s}/{1:s}.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )

    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=int(1e7)
    )

    for this_file_name in panel_file_names:
        os.remove(this_file_name)


def _run(example_file_name, num_examples, output_dir_name):
    """For each data example, plots data at different levels of perturbation.

    This is effectively the main method.

    :param example_file_names: See documentation at top of this script.
    :param num_examples: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_greater(num_examples, 0)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(example_file_name))
    example_dict = example_io.read_file(example_file_name)
    num_examples_total = len(example_dict[example_utils.EXAMPLE_IDS_KEY])

    if num_examples < num_examples_total:
        desired_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )
        desired_indices = numpy.random.choice(
            desired_indices, size=num_examples, replace=False
        )
        example_dict = example_utils.subset_by_index(
            example_dict=example_dict,
            desired_indices=desired_indices
        )

    for i in range(num_examples):
        _plot_one_example(
            example_dict=example_dict,
            example_index=i,
            output_dir_name=output_dir_name
        )
        print(MINOR_SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
