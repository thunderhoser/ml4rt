"""Plots error metrics as a function of lat-long location."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import grids
import number_rounding
import longitude_conversion as lng_conversion
import file_system_utils
import error_checking
import gg_plotting_utils
import radar_plotting
import border_io
import example_io
import prediction_io
import example_utils
import evaluation
import neural_net
import plotting_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TOLERANCE = 1e-6
DUMMY_FIELD_NAME = 'reflectivity_column_max_dbz'

SHORTWAVE_NET_FLUX_NAME = 'net_shortwave_flux_w_m02'
LONGWAVE_NET_FLUX_NAME = 'net_longwave_flux_w_m02'

TARGET_NAME_TO_VERBOSE = {
    example_utils.SHORTWAVE_HEATING_RATE_NAME: 'SW heating rate',
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: r'SW $F_{down}^{sfc}$',
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME: r'SW $F_{up}^{TOA}$',
    SHORTWAVE_NET_FLUX_NAME: 'SW net flux',
    example_utils.LONGWAVE_HEATING_RATE_NAME: 'LW heating rate',
    example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME: r'LW $F_{down}^{sfc}$',
    example_utils.LONGWAVE_TOA_UP_FLUX_NAME: r'LW $F_{up}^{TOA}$',
    LONGWAVE_NET_FLUX_NAME: 'LW net flux'
}
TARGET_NAME_TO_UNITS = {
    example_utils.SHORTWAVE_HEATING_RATE_NAME: r'K day$^{-1}$',
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME: r'W m$^{-2}$',
    SHORTWAVE_NET_FLUX_NAME: r'W m$^{-2}$',
    example_utils.LONGWAVE_HEATING_RATE_NAME: r'K day$^{-1}$',
    example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_utils.LONGWAVE_TOA_UP_FLUX_NAME: r'W m$^{-2}$',
    LONGWAVE_NET_FLUX_NAME: r'W m$^{-2}$'
}
TARGET_NAME_TO_SQUARED_UNITS = {
    example_utils.SHORTWAVE_HEATING_RATE_NAME: r'K$^{2}$ day$^{-2}$',
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    SHORTWAVE_NET_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    example_utils.LONGWAVE_HEATING_RATE_NAME: r'K$^{2}$ day$^{-2}$',
    example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    example_utils.LONGWAVE_TOA_UP_FLUX_NAME: r'W$^{2}$ m$^{-4}$',
    LONGWAVE_NET_FLUX_NAME: r'W$^{2}$ m$^{-4}$'
}

BIAS_NAME = 'Bias'
CORRELATION_NAME = 'Pearson correlation'
MAE_NAME = 'Mean absolute error'
MSE_NAME = 'Mean squared error'
KGE_NAME = 'Kling-Gupta efficiency'
MAE_SKILL_SCORE_NAME = 'Mean-absolute-error skill score'
MSE_SKILL_SCORE_NAME = 'Mean-squared-error skill score'
SKILL_SCORE_NAMES = [KGE_NAME, MAE_SKILL_SCORE_NAME, MSE_SKILL_SCORE_NAME]

SCORE_TO_MIN_COLOUR_VALUE = {
    CORRELATION_NAME: 0.5,
    MAE_NAME: 0.,
    MSE_NAME: 0.,
    KGE_NAME: 0.,
    MAE_SKILL_SCORE_NAME: 0.,
    MSE_SKILL_SCORE_NAME: 0.
}

SCORE_TO_CBAR_EXTEND_MIN_FLAG = {
    BIAS_NAME: True,
    CORRELATION_NAME: True,
    MAE_NAME: False,
    MSE_NAME: False,
    KGE_NAME: True,
    MAE_SKILL_SCORE_NAME: True,
    MSE_SKILL_SCORE_NAME: True
}

SCORE_TO_MAX_COLOUR_VALUE = {
    CORRELATION_NAME: 1.,
    KGE_NAME: 1.,
    MAE_SKILL_SCORE_NAME: 1.,
    MSE_SKILL_SCORE_NAME: 1.
}

SCORE_TO_CBAR_EXTEND_MAX_FLAG = {
    BIAS_NAME: True,
    CORRELATION_NAME: False,
    MAE_NAME: True,
    MSE_NAME: True,
    KGE_NAME: False,
    MAE_SKILL_SCORE_NAME: False,
    MSE_SKILL_SCORE_NAME: False
}

MAIN_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis', lut=20)
BIAS_COLOUR_MAP_OBJECT = pyplot.get_cmap(name='seismic', lut=20)
MAIN_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))
BIAS_COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))

SCORE_TO_COLOUR_MAP_OBJECT = {
    BIAS_NAME: BIAS_COLOUR_MAP_OBJECT,
    CORRELATION_NAME: MAIN_COLOUR_MAP_OBJECT,
    MAE_NAME: MAIN_COLOUR_MAP_OBJECT,
    MSE_NAME: MAIN_COLOUR_MAP_OBJECT,
    KGE_NAME: MAIN_COLOUR_MAP_OBJECT,
    MAE_SKILL_SCORE_NAME: MAIN_COLOUR_MAP_OBJECT,
    MSE_SKILL_SCORE_NAME: MAIN_COLOUR_MAP_OBJECT
}

SCORE_AND_TARGET_TO_MIN_COLOUR_VALUE = {
    (BIAS_NAME, example_utils.SHORTWAVE_HEATING_RATE_NAME): -1.,
    (BIAS_NAME, example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME): -100.,
    (BIAS_NAME, example_utils.SHORTWAVE_TOA_UP_FLUX_NAME): -100.,
    (BIAS_NAME, SHORTWAVE_NET_FLUX_NAME): -100.,
    (BIAS_NAME, example_utils.LONGWAVE_HEATING_RATE_NAME): -1.,
    (BIAS_NAME, example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME): -100.,
    (BIAS_NAME, example_utils.LONGWAVE_TOA_UP_FLUX_NAME): -100.,
    (BIAS_NAME, LONGWAVE_NET_FLUX_NAME): -100.,
}

SCORE_AND_TARGET_TO_MAX_COLOUR_VALUE = {
    (BIAS_NAME, example_utils.SHORTWAVE_HEATING_RATE_NAME): 1.,
    (BIAS_NAME, example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME): 100.,
    (BIAS_NAME, example_utils.SHORTWAVE_TOA_UP_FLUX_NAME): 100.,
    (BIAS_NAME, SHORTWAVE_NET_FLUX_NAME): 100.,
    (MAE_NAME, example_utils.SHORTWAVE_HEATING_RATE_NAME): 1.,
    (MAE_NAME, example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME): 100.,
    (MAE_NAME, example_utils.SHORTWAVE_TOA_UP_FLUX_NAME): 100.,
    (MAE_NAME, SHORTWAVE_NET_FLUX_NAME): 100.,
    (MSE_NAME, example_utils.SHORTWAVE_HEATING_RATE_NAME): 4.,
    (MSE_NAME, example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME): 1e4,
    (MSE_NAME, example_utils.SHORTWAVE_TOA_UP_FLUX_NAME): 1e4,
    (MSE_NAME, SHORTWAVE_NET_FLUX_NAME): 1e4,
    (BIAS_NAME, example_utils.LONGWAVE_HEATING_RATE_NAME): 1.,
    (BIAS_NAME, example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME): 100.,
    (BIAS_NAME, example_utils.LONGWAVE_TOA_UP_FLUX_NAME): 100.,
    (BIAS_NAME, LONGWAVE_NET_FLUX_NAME): 100.,
    (MAE_NAME, example_utils.LONGWAVE_HEATING_RATE_NAME): 1.,
    (MAE_NAME, example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME): 100.,
    (MAE_NAME, example_utils.LONGWAVE_TOA_UP_FLUX_NAME): 100.,
    (MAE_NAME, LONGWAVE_NET_FLUX_NAME): 100.,
    (MSE_NAME, example_utils.LONGWAVE_HEATING_RATE_NAME): 4.,
    (MSE_NAME, example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME): 1e4,
    (MSE_NAME, example_utils.LONGWAVE_TOA_UP_FLUX_NAME): 1e4,
    (MSE_NAME, LONGWAVE_NET_FLUX_NAME): 1e4
}

MIN_LATITUDE_DEG_N = -90.
MAX_LATITUDE_DEG_N = 90.
MIN_LONGITUDE_DEG_E = 0.
MAX_LONGITUDE_DEG_E = 360.

BORDER_LINE_WIDTH = 2.
BORDER_Z_ORDER = -1e8
BORDER_COLOUR = numpy.array([139, 69, 19], dtype=float) / 255

FIGURE_RESOLUTION_DPI = 300
FIGURE_WIDTH_INCHES = 20
FIGURE_HEIGHT_INCHES = 11.25

FONT_SIZE = 40
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
HEATING_RATE_HEIGHT_ARG_NAME = 'heating_rate_height_m_agl'
GRID_SPACING_ARG_NAME = 'grid_spacing_deg'
MIN_EXAMPLES_ARG_NAME = 'min_num_examples'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to file with predicted and actual target values.  Will be read by '
    '`prediction_io.read_file`.'
)
HEATING_RATE_HEIGHT_HELP_STRING = (
    'Will plot heating-rate errors at this height (metres above ground level).'
    '  If you want to plot flux errors instead, leave this alone.'
)
GRID_SPACING_HELP_STRING = 'Grid spacing (degrees).'
MIN_EXAMPLES_HELP_STRING = (
    'Minimum number of examples.  For any grid cell with fewer examples, error '
    'metrics will not be plotted.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HEATING_RATE_HEIGHT_ARG_NAME, type=int, required=False, default=-1,
    help=HEATING_RATE_HEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GRID_SPACING_ARG_NAME, type=float, required=True,
    help=GRID_SPACING_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_EXAMPLES_ARG_NAME, type=int, required=True,
    help=MIN_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _create_latlng_grid(
        min_latitude_deg_n, max_latitude_deg_n, latitude_spacing_deg,
        min_longitude_deg_e, max_longitude_deg_e, longitude_spacing_deg):
    """Creates lat-long grid.

    M = number of rows in grid
    N = number of columns in grid

    :param min_latitude_deg_n: Minimum latitude (deg N) in grid.
    :param max_latitude_deg_n: Max latitude (deg N) in grid.
    :param latitude_spacing_deg: Spacing (deg N) between grid points in adjacent
        rows.
    :param min_longitude_deg_e: Minimum longitude (deg E) in grid.
    :param max_longitude_deg_e: Max longitude (deg E) in grid.
    :param longitude_spacing_deg: Spacing (deg E) between grid points in
        adjacent columns.
    :return: grid_point_latitudes_deg: length-M numpy array with latitudes
        (deg N) of grid points.
    :return: grid_point_longitudes_deg: length-N numpy array with longitudes
        (deg E) of grid points.
    """

    # TODO(thunderhoser): Make this handle wrap-around issues.

    min_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        min_longitude_deg_e
    )
    max_longitude_deg_e = lng_conversion.convert_lng_positive_in_west(
        max_longitude_deg_e
    )

    min_latitude_deg_n = number_rounding.floor_to_nearest(
        min_latitude_deg_n, latitude_spacing_deg
    )
    max_latitude_deg_n = number_rounding.ceiling_to_nearest(
        max_latitude_deg_n, latitude_spacing_deg
    )
    min_longitude_deg_e = number_rounding.floor_to_nearest(
        min_longitude_deg_e, longitude_spacing_deg
    )
    max_longitude_deg_e = number_rounding.ceiling_to_nearest(
        max_longitude_deg_e, longitude_spacing_deg
    )

    num_grid_rows = 1 + int(numpy.round(
        (max_latitude_deg_n - min_latitude_deg_n) / latitude_spacing_deg
    ))
    num_grid_columns = 1 + int(numpy.round(
        (max_longitude_deg_e - min_longitude_deg_e) / longitude_spacing_deg
    ))

    return grids.get_latlng_grid_points(
        min_latitude_deg=min_latitude_deg_n,
        min_longitude_deg=min_longitude_deg_e,
        lat_spacing_deg=latitude_spacing_deg,
        lng_spacing_deg=longitude_spacing_deg,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )


def _plot_one_score(
        score_matrix, grid_latitudes_deg_n, grid_longitudes_deg_e,
        border_latitudes_deg_n, border_longitudes_deg_e, score_name,
        target_name, output_file_name, title_string):
    """Plots one score on 2-D georeferenced grid.

    M = number of rows in grid
    N = number of columns in grid
    P = number of points in border set

    :param score_matrix: M-by-N numpy array of scores.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param score_name: Name of score to be plotted.
    :param target_name: Name of target variable.
    :param output_file_name: Path to output file (figure will be saved here).
    :param title_string: Title (will be added above figure).  If you do not want
        a title, make this None.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )

    if score_name == '':
        colour_map_object = MAIN_COLOUR_MAP_OBJECT
        min_colour_value = 0
        max_colour_value = numpy.nanmax(score_matrix)
        cbar_extend_min_flag = False
        cbar_extend_max_flag = False
    else:
        colour_map_object = SCORE_TO_COLOUR_MAP_OBJECT[score_name]

        if score_name in SCORE_TO_MIN_COLOUR_VALUE:
            min_colour_value = SCORE_TO_MIN_COLOUR_VALUE[score_name]
        else:
            min_colour_value = SCORE_AND_TARGET_TO_MIN_COLOUR_VALUE[
                (score_name, target_name)
            ]

        if score_name in SCORE_TO_MAX_COLOUR_VALUE:
            max_colour_value = SCORE_TO_MAX_COLOUR_VALUE[score_name]
        else:
            max_colour_value = SCORE_AND_TARGET_TO_MAX_COLOUR_VALUE[
                (score_name, target_name)
            ]

        cbar_extend_min_flag = SCORE_TO_CBAR_EXTEND_MIN_FLAG[score_name]
        cbar_extend_max_flag = SCORE_TO_CBAR_EXTEND_MAX_FLAG[score_name]

    colour_norm_object = pyplot.Normalize(
        vmin=min_colour_value, vmax=max_colour_value
    )

    sort_indices = numpy.argsort(grid_longitudes_deg_e)
    sorted_grid_longitudes_deg_e = grid_longitudes_deg_e[sort_indices]
    sorted_score_matrix = score_matrix[:, sort_indices]

    radar_plotting.plot_latlng_grid(
        field_matrix=sorted_score_matrix, field_name=DUMMY_FIELD_NAME,
        axes_object=axes_object,
        min_grid_point_latitude_deg=numpy.min(grid_latitudes_deg_n),
        min_grid_point_longitude_deg=numpy.min(sorted_grid_longitudes_deg_e),
        latitude_spacing_deg=numpy.diff(grid_latitudes_deg_n[:2])[0],
        longitude_spacing_deg=numpy.diff(sorted_grid_longitudes_deg_e[:2])[0],
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object
    )

    colour_bar_object = gg_plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=score_matrix,
        colour_map_object=colour_map_object,
        min_value=min_colour_value, max_value=max_colour_value,
        orientation_string='vertical',
        extend_min=cbar_extend_min_flag, extend_max=cbar_extend_max_flag,
        padding=0.01, font_size=FONT_SIZE
    )

    colour_bar_object.set_label(score_name)
    tick_values = colour_bar_object.get_ticks()
    tick_strings = ['{0:.2g}'.format(v) for v in tick_values]
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    plotting_utils.plot_grid_lines(
        plot_latitudes_deg_n=grid_latitudes_deg_n,
        plot_longitudes_deg_e=sorted_grid_longitudes_deg_e,
        axes_object=axes_object,
        parallel_spacing_deg=10., meridian_spacing_deg=20., font_size=FONT_SIZE
    )

    latitude_spacing_deg = numpy.diff(grid_latitudes_deg_n[:2])[0]
    min_plot_latitude_deg_n = max([
        grid_latitudes_deg_n[0] - latitude_spacing_deg / 2,
        -90.
    ])
    max_plot_latitude_deg_n = min([
        grid_latitudes_deg_n[-1] + latitude_spacing_deg / 2,
        90.
    ])

    longitude_spacing_deg = numpy.diff(sorted_grid_longitudes_deg_e[:2])[0]
    min_plot_longitude_deg_e = (
        sorted_grid_longitudes_deg_e[0] - longitude_spacing_deg / 2
    )
    max_plot_longitude_deg_e = (
        sorted_grid_longitudes_deg_e[-1] + longitude_spacing_deg / 2
    )

    if min_plot_longitude_deg_e < -180 or max_plot_longitude_deg_e > 180:
        min_plot_longitude_deg_e = -180.
        max_plot_longitude_deg_e = 180.

    axes_object.set_xlim(min_plot_longitude_deg_e, max_plot_longitude_deg_e)
    axes_object.set_ylim(min_plot_latitude_deg_n, max_plot_latitude_deg_n)
    axes_object.set_title(title_string)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(prediction_file_name, heating_rate_height_m_agl, grid_spacing_deg,
         min_num_examples, output_dir_name):
    """Plots error metrics as a function of lat-long location.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param heating_rate_height_m_agl: Same.
    :param grid_spacing_deg: Same.
    :param min_num_examples: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_greater(grid_spacing_deg, 0.)
    error_checking.assert_is_leq(grid_spacing_deg, 10.)
    error_checking.assert_is_greater(min_num_examples, 0)
    if heating_rate_height_m_agl <= 0:
        heating_rate_height_m_agl = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    border_latitudes_deg_n, border_longitudes_deg_e = border_io.read_file()

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    normalization_file_name = (
        training_option_dict[neural_net.NORMALIZATION_FILE_KEY]
    )

    print((
        'Reading training examples (for climatology) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)

    if heating_rate_height_m_agl is None:
        target_names = [
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME,
            example_utils.SHORTWAVE_TOA_UP_FLUX_NAME,
            SHORTWAVE_NET_FLUX_NAME,
            example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME,
            example_utils.LONGWAVE_TOA_UP_FLUX_NAME,
            LONGWAVE_NET_FLUX_NAME
        ]
        target_names = [
            n for n in target_names
            if n in training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
        ]
    else:
        target_names = [
            example_utils.SHORTWAVE_HEATING_RATE_NAME,
            example_utils.LONGWAVE_HEATING_RATE_NAME
        ]
        target_names = [
            n for n in target_names
            if n in training_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
        ]

    num_target_vars = len(target_names)
    climo_value_by_target = numpy.full(num_target_vars, numpy.nan)

    for k in range(num_target_vars):
        if target_names[k] in [
                example_utils.SHORTWAVE_HEATING_RATE_NAME,
                example_utils.LONGWAVE_HEATING_RATE_NAME
        ]:
            these_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=target_names[k],
                height_m_agl=heating_rate_height_m_agl
            )
        elif target_names[k] == SHORTWAVE_NET_FLUX_NAME:
            down_flux_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
            )
            up_flux_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
            )
            these_values = down_flux_values - up_flux_values
        elif target_names[k] == LONGWAVE_NET_FLUX_NAME:
            down_flux_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
            )
            up_flux_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=example_utils.LONGWAVE_TOA_UP_FLUX_NAME
            )
            these_values = down_flux_values - up_flux_values
        else:
            these_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict, field_name=target_names[k]
            )

        climo_value_by_target[k] = numpy.mean(these_values)

    metadata_dict = example_utils.parse_example_ids(
        prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    )
    latitudes_deg_n = metadata_dict[example_utils.LATITUDES_KEY]
    longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        metadata_dict[example_utils.LONGITUDES_KEY]
    )

    # Create grid.
    grid_latitudes_deg_n, grid_longitudes_deg_e = _create_latlng_grid(
        min_latitude_deg_n=MIN_LATITUDE_DEG_N,
        max_latitude_deg_n=MAX_LATITUDE_DEG_N,
        latitude_spacing_deg=grid_spacing_deg,
        min_longitude_deg_e=MIN_LONGITUDE_DEG_E,
        max_longitude_deg_e=MAX_LONGITUDE_DEG_E - grid_spacing_deg,
        longitude_spacing_deg=grid_spacing_deg
    )
    grid_longitudes_deg_e += grid_spacing_deg / 2
    num_grid_rows = len(grid_latitudes_deg_n)
    num_grid_columns = len(grid_longitudes_deg_e)

    grid_edge_latitudes_deg, grid_edge_longitudes_deg = (
        grids.get_latlng_grid_cell_edges(
            min_latitude_deg=grid_latitudes_deg_n[0],
            min_longitude_deg=grid_longitudes_deg_e[0],
            lat_spacing_deg=numpy.diff(grid_latitudes_deg_n[:2])[0],
            lng_spacing_deg=numpy.diff(grid_longitudes_deg_e[:2])[0],
            num_rows=num_grid_rows, num_columns=num_grid_columns
        )
    )

    grid_edge_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        grid_edge_longitudes_deg
    )
    border_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        border_longitudes_deg_e
    )
    grid_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        grid_longitudes_deg_e
    )

    # Do actual stuff.
    for k in range(len(target_names)):
        print(SEPARATOR_STRING)

        dimensions = (num_grid_rows, num_grid_columns)
        bias_matrix = numpy.full(dimensions, numpy.nan)
        correlation_matrix = numpy.full(dimensions, numpy.nan)
        mae_matrix = numpy.full(dimensions, numpy.nan)
        mae_skill_score_matrix = numpy.full(dimensions, numpy.nan)
        mse_matrix = numpy.full(dimensions, numpy.nan)
        mse_skill_score_matrix = numpy.full(dimensions, numpy.nan)
        kge_matrix = numpy.full(dimensions, numpy.nan)
        num_examples_matrix = numpy.full(dimensions, -1, dtype=int)

        if target_names[k] in [
                example_utils.SHORTWAVE_HEATING_RATE_NAME,
                example_utils.LONGWAVE_HEATING_RATE_NAME
        ]:
            height_diffs_metres = numpy.absolute(
                prediction_dict[prediction_io.HEIGHTS_KEY] -
                heating_rate_height_m_agl
            )
            assert numpy.min(height_diffs_metres) <= TOLERANCE
            height_index = numpy.argmin(height_diffs_metres)

            channel_index = training_option_dict[
                neural_net.VECTOR_TARGET_NAMES_KEY
            ].index(target_names[k])

            actual_values = prediction_dict[
                prediction_io.VECTOR_TARGETS_KEY
            ][:, height_index, channel_index]

            predicted_values = prediction_dict[
                prediction_io.VECTOR_PREDICTIONS_KEY
            ][:, height_index, channel_index]

        elif target_names[k] in [
                SHORTWAVE_NET_FLUX_NAME, LONGWAVE_NET_FLUX_NAME
        ]:
            if target_names[k] == SHORTWAVE_NET_FLUX_NAME:
                down_flux_index = training_option_dict[
                    neural_net.SCALAR_TARGET_NAMES_KEY
                ].index(example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME)

                up_flux_index = training_option_dict[
                    neural_net.SCALAR_TARGET_NAMES_KEY
                ].index(example_utils.SHORTWAVE_TOA_UP_FLUX_NAME)
            else:
                down_flux_index = training_option_dict[
                    neural_net.SCALAR_TARGET_NAMES_KEY
                ].index(example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME)

                up_flux_index = training_option_dict[
                    neural_net.SCALAR_TARGET_NAMES_KEY
                ].index(example_utils.LONGWAVE_TOA_UP_FLUX_NAME)

            actual_values = (
                prediction_dict[prediction_io.SCALAR_TARGETS_KEY][
                    :, down_flux_index
                ] -
                prediction_dict[prediction_io.SCALAR_TARGETS_KEY][
                    :, up_flux_index
                ]
            )

            predicted_values = (
                prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][
                    :, down_flux_index
                ] -
                prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY][
                    :, up_flux_index
                ]
            )

        else:
            channel_index = training_option_dict[
                neural_net.SCALAR_TARGET_NAMES_KEY
            ].index(target_names[k])

            actual_values = prediction_dict[
                prediction_io.SCALAR_TARGETS_KEY
            ][:, channel_index]

            predicted_values = prediction_dict[
                prediction_io.SCALAR_PREDICTIONS_KEY
            ][:, channel_index]

        for i in range(num_grid_rows):
            print((
                'Have computed {0:s} errors for {1:d} of {2:d} grid rows...'
            ).format(
                target_names[k], i, num_grid_rows
            ))

            for j in range(num_grid_columns):
                these_indices = grids.find_events_in_grid_cell(
                    event_x_coords_metres=longitudes_deg_e,
                    event_y_coords_metres=latitudes_deg_n,
                    grid_edge_x_coords_metres=grid_edge_longitudes_deg,
                    grid_edge_y_coords_metres=grid_edge_latitudes_deg,
                    row_index=i, column_index=j, verbose=False
                )

                num_examples_matrix[i, j] = len(these_indices)
                if len(these_indices) < min_num_examples:
                    continue

                bias_matrix[i, j] = evaluation._get_bias_one_scalar(
                    target_values=actual_values[these_indices],
                    predicted_values=predicted_values[these_indices]
                )
                correlation_matrix[i, j] = (
                    evaluation._get_correlation_one_scalar(
                        target_values=actual_values[these_indices],
                        predicted_values=predicted_values[these_indices]
                    )
                )
                mae_matrix[i, j] = evaluation._get_mae_one_scalar(
                    target_values=actual_values[these_indices],
                    predicted_values=predicted_values[these_indices]
                )
                mse_matrix[i, j] = evaluation._get_mse_one_scalar(
                    target_values=actual_values[these_indices],
                    predicted_values=predicted_values[these_indices]
                )[0]
                kge_matrix[i, j] = evaluation._get_kge_one_scalar(
                    target_values=actual_values[these_indices],
                    predicted_values=predicted_values[these_indices]
                )

                mae_skill_score_matrix[i, j] = (
                    evaluation._get_mae_ss_one_scalar(
                        target_values=actual_values[these_indices],
                        predicted_values=predicted_values[these_indices],
                        mean_training_target_value=climo_value_by_target[k]
                    )
                )
                mse_skill_score_matrix[i, j] = (
                    evaluation._get_mse_ss_one_scalar(
                        target_values=actual_values[these_indices],
                        predicted_values=predicted_values[these_indices],
                        mean_training_target_value=climo_value_by_target[k]
                    )
                )

        print('Have computed {0:s} errors for all {1:d} grid rows!'.format(
            target_names[k], num_grid_rows
        ))
        print(SEPARATOR_STRING)

        title_string = (
            'Bias for {0:s}{1:s} ({2:s}) where sample size >= {3:d}'
        ).format(
            '' if heating_rate_height_m_agl is None
            else '{0:d}-m-AGL '.format(heating_rate_height_m_agl),
            TARGET_NAME_TO_VERBOSE[target_names[k]],
            TARGET_NAME_TO_UNITS[target_names[k]],
            min_num_examples
        )

        _plot_one_score(
            score_matrix=bias_matrix,
            grid_latitudes_deg_n=grid_latitudes_deg_n,
            grid_longitudes_deg_e=grid_longitudes_deg_e,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            score_name=BIAS_NAME, target_name=target_names[k],
            title_string=title_string,
            output_file_name='{0:s}/{1:s}_bias.jpg'.format(
                output_dir_name, target_names[k].replace('_', '-')
            )
        )

        title_string = (
            'Correlation for {0:s}{1:s} where sample size >= {2:d}'
        ).format(
            '' if heating_rate_height_m_agl is None
            else '{0:d}-m-AGL '.format(heating_rate_height_m_agl),
            TARGET_NAME_TO_VERBOSE[target_names[k]],
            min_num_examples
        )

        _plot_one_score(
            score_matrix=correlation_matrix,
            grid_latitudes_deg_n=grid_latitudes_deg_n,
            grid_longitudes_deg_e=grid_longitudes_deg_e,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            score_name=CORRELATION_NAME, target_name=target_names[k],
            title_string=title_string,
            output_file_name='{0:s}/{1:s}_correlation.jpg'.format(
                output_dir_name, target_names[k].replace('_', '-')
            )
        )

        title_string = (
            'MAE for {0:s}{1:s} ({2:s}) where sample size >= {3:d}'
        ).format(
            '' if heating_rate_height_m_agl is None
            else '{0:d}-m-AGL '.format(heating_rate_height_m_agl),
            TARGET_NAME_TO_VERBOSE[target_names[k]],
            TARGET_NAME_TO_UNITS[target_names[k]],
            min_num_examples
        )

        _plot_one_score(
            score_matrix=mae_matrix,
            grid_latitudes_deg_n=grid_latitudes_deg_n,
            grid_longitudes_deg_e=grid_longitudes_deg_e,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            score_name=MAE_NAME, target_name=target_names[k],
            title_string=title_string,
            output_file_name='{0:s}/{1:s}_mae.jpg'.format(
                output_dir_name, target_names[k].replace('_', '-')
            )
        )

        title_string = (
            'MSE for {0:s}{1:s} ({2:s}) where sample size >= {3:d}'
        ).format(
            '' if heating_rate_height_m_agl is None
            else '{0:d}-m-AGL '.format(heating_rate_height_m_agl),
            TARGET_NAME_TO_VERBOSE[target_names[k]],
            TARGET_NAME_TO_SQUARED_UNITS[target_names[k]],
            min_num_examples
        )

        _plot_one_score(
            score_matrix=mse_matrix,
            grid_latitudes_deg_n=grid_latitudes_deg_n,
            grid_longitudes_deg_e=grid_longitudes_deg_e,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            score_name=MSE_NAME, target_name=target_names[k],
            title_string=title_string,
            output_file_name='{0:s}/{1:s}_mse.jpg'.format(
                output_dir_name, target_names[k].replace('_', '-')
            )
        )

        title_string = (
            'KGE for {0:s}{1:s} where sample size >= {2:d}'
        ).format(
            '' if heating_rate_height_m_agl is None
            else '{0:d}-m-AGL '.format(heating_rate_height_m_agl),
            TARGET_NAME_TO_VERBOSE[target_names[k]],
            min_num_examples
        )

        _plot_one_score(
            score_matrix=kge_matrix,
            grid_latitudes_deg_n=grid_latitudes_deg_n,
            grid_longitudes_deg_e=grid_longitudes_deg_e,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            score_name=KGE_NAME, target_name=target_names[k],
            title_string=title_string,
            output_file_name='{0:s}/{1:s}_kge.jpg'.format(
                output_dir_name, target_names[k].replace('_', '-')
            )
        )

        title_string = (
            'MAESS for {0:s}{1:s} where sample size >= {2:d}'
        ).format(
            '' if heating_rate_height_m_agl is None
            else '{0:d}-m-AGL '.format(heating_rate_height_m_agl),
            TARGET_NAME_TO_VERBOSE[target_names[k]],
            min_num_examples
        )

        _plot_one_score(
            score_matrix=mae_skill_score_matrix,
            grid_latitudes_deg_n=grid_latitudes_deg_n,
            grid_longitudes_deg_e=grid_longitudes_deg_e,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            score_name=MAE_SKILL_SCORE_NAME, target_name=target_names[k],
            title_string=title_string,
            output_file_name='{0:s}/{1:s}_maess.jpg'.format(
                output_dir_name, target_names[k].replace('_', '-')
            )
        )

        title_string = (
            'MSESS for {0:s}{1:s} where sample size >= {2:d}'
        ).format(
            '' if heating_rate_height_m_agl is None
            else '{0:d}-m-AGL '.format(heating_rate_height_m_agl),
            TARGET_NAME_TO_VERBOSE[target_names[k]],
            min_num_examples
        )

        _plot_one_score(
            score_matrix=mse_skill_score_matrix,
            grid_latitudes_deg_n=grid_latitudes_deg_n,
            grid_longitudes_deg_e=grid_longitudes_deg_e,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            score_name=MSE_SKILL_SCORE_NAME, target_name=target_names[k],
            title_string=title_string,
            output_file_name='{0:s}/{1:s}_msess.jpg'.format(
                output_dir_name, target_names[k].replace('_', '-')
            )
        )

        _plot_one_score(
            score_matrix=num_examples_matrix.astype(float),
            grid_latitudes_deg_n=grid_latitudes_deg_n,
            grid_longitudes_deg_e=grid_longitudes_deg_e,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            score_name='', target_name=target_names[k],
            title_string='Number of examples',
            output_file_name='{0:s}/num_examples.jpg'.format(output_dir_name)
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        heating_rate_height_m_agl=getattr(
            INPUT_ARG_OBJECT, HEATING_RATE_HEIGHT_ARG_NAME
        ),
        grid_spacing_deg=getattr(INPUT_ARG_OBJECT, GRID_SPACING_ARG_NAME),
        min_num_examples=getattr(INPUT_ARG_OBJECT, MIN_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
