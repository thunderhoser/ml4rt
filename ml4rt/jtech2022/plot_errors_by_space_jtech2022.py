"""Plots error metrics vs. geographic location on a world map."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from gewittergefahr.plotting import radar_plotting
from gewittergefahr.plotting import imagemagick_utils
from ml4rt.io import border_io
from ml4rt.io import prediction_io
from ml4rt.utils import example_utils
from ml4rt.utils import evaluation
from ml4rt.machine_learning import neural_net
from ml4rt.plotting import plotting_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
DUMMY_FIELD_NAME = 'reflectivity_column_max_dbz'

SHORTWAVE_ALL_FLUX_NAME = 'all_shortwave_flux_w_m02'
SHORTWAVE_NET_FLUX_NAME = 'net_shortwave_flux_w_m02'
LONGWAVE_ALL_FLUX_NAME = 'all_longwave_flux_w_m02'
LONGWAVE_NET_FLUX_NAME = 'net_longwave_flux_w_m02'

STATISTIC_NAMES = [
    'shortwave_rmse', 'shortwave_near_sfc_rmse',
    'shortwave_bias', 'shortwave_near_sfc_bias',
    'shortwave_all_flux_rmse', 'shortwave_net_flux_rmse',
    'shortwave_net_flux_bias',
    'longwave_rmse', 'longwave_near_sfc_rmse',
    'longwave_bias', 'longwave_near_sfc_bias',
    'longwave_all_flux_rmse', 'longwave_net_flux_rmse',
    'longwave_net_flux_bias'
]
STATISTIC_NAMES_FANCY = [
    r'RMSE$_{hr}$ (K day$^{-1}$)',
    r'Near-surface RMSE$_{hr}$ (K day$^{-1}$)',
    r'Bias$_{hr}$ (K day$^{-1}$)',
    r'Near-surface bias$_{hr}$ (K day$^{-1}$)',
    r'RMSE$_{flux}$ (W m$^{-2}$)',
    r'RMSE for net flux only (W m$^{-2}$)',
    r'Bias for net flux only (W m$^{-2}$)',
    r'RMSE$_{hr}$ (K day$^{-1}$)',
    r'Near-surface RMSE$_{hr}$ (K day$^{-1}$)',
    r'Bias$_{hr}$ (K day$^{-1}$)',
    r'Near-surface bias$_{hr}$ (K day$^{-1}$)',
    r'RMSE$_{flux}$ (W m$^{-2}$)',
    r'RMSE for net flux only (W m$^{-2}$)',
    r'Bias for net flux only (W m$^{-2}$)'
]
TARGET_NAME_BY_STATISTIC = [
    example_utils.SHORTWAVE_HEATING_RATE_NAME,
    example_utils.SHORTWAVE_HEATING_RATE_NAME,
    example_utils.SHORTWAVE_HEATING_RATE_NAME,
    example_utils.SHORTWAVE_HEATING_RATE_NAME,
    SHORTWAVE_ALL_FLUX_NAME,
    SHORTWAVE_NET_FLUX_NAME,
    SHORTWAVE_NET_FLUX_NAME,
    example_utils.LONGWAVE_HEATING_RATE_NAME,
    example_utils.LONGWAVE_HEATING_RATE_NAME,
    example_utils.LONGWAVE_HEATING_RATE_NAME,
    example_utils.LONGWAVE_HEATING_RATE_NAME,
    LONGWAVE_ALL_FLUX_NAME,
    LONGWAVE_NET_FLUX_NAME,
    LONGWAVE_NET_FLUX_NAME
]
TARGET_HEIGHT_INDEX_BY_STATISTIC = numpy.array(
    [-1, 0, -1, 0, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1], dtype=int
)

COLOUR_MAP_OBJECT = pyplot.get_cmap(name='viridis', lut=20)
COLOUR_MAP_OBJECT.set_bad(numpy.full(3, 152. / 255))

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
GRID_SPACING_ARG_NAME = 'grid_spacing_deg'
MIN_EXAMPLES_ARG_NAME = 'min_num_examples'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to file with predicted and actual target values.  Will be read by '
    '`prediction_io.read_file`.'
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
        border_latitudes_deg_n, border_longitudes_deg_e, title_string,
        letter_label, output_file_name):
    """Plots one score on 2-D georeferenced grid.

    M = number of rows in grid
    N = number of columns in grid
    P = number of points in border set

    :param score_matrix: M-by-N numpy array of scores.
    :param grid_latitudes_deg_n: length-M numpy array of latitudes (deg N).
    :param grid_longitudes_deg_e: length-N numpy array of longitudes (deg E).
    :param border_latitudes_deg_n: length-P numpy array of latitudes (deg N).
    :param border_longitudes_deg_e: length-P numpy array of longitudes (deg E).
    :param title_string: Title (will be plotted above figure).
    :param letter_label: Letter label (will be plotted above and left of figure
        in parentheses).
    :param output_file_name: Path to output file (figure will be saved here).
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    plotting_utils.plot_borders(
        border_latitudes_deg_n=border_latitudes_deg_n,
        border_longitudes_deg_e=border_longitudes_deg_e,
        axes_object=axes_object
    )

    min_colour_value = numpy.nanpercentile(score_matrix, 0.)
    max_colour_value = numpy.nanpercentile(score_matrix, 99.)
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
        colour_map_object=COLOUR_MAP_OBJECT,
        colour_norm_object=colour_norm_object
    )

    colour_bar_object = gg_plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=score_matrix,
        colour_map_object=COLOUR_MAP_OBJECT,
        min_value=min_colour_value, max_value=max_colour_value,
        orientation_string='vertical', extend_min=True, extend_max=True,
        padding=0.01, font_size=FONT_SIZE
    )

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
    gg_plotting_utils.label_axes(
        axes_object=axes_object, label_string='({0:s})'.format(letter_label)
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(prediction_file_name, grid_spacing_deg, min_num_examples,
         output_dir_name):
    """Plots error metrics vs. geographic location on a world map.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param grid_spacing_deg: Same.
    :param min_num_examples: Same.
    :param output_dir_name: Same.
    """

    error_checking.assert_is_greater(grid_spacing_deg, 0.)
    error_checking.assert_is_leq(grid_spacing_deg, 10.)
    error_checking.assert_is_greater(min_num_examples, 0)

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

    available_target_names = [
        training_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY] +
        training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
    ]

    while isinstance(available_target_names[0], list):
        available_target_names = available_target_names[0]

    if (
            example_utils.SHORTWAVE_TOA_UP_FLUX_NAME in available_target_names
            and example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME in
            available_target_names
    ):
        available_target_names += [
            SHORTWAVE_ALL_FLUX_NAME, SHORTWAVE_NET_FLUX_NAME
        ]

    if (
            example_utils.LONGWAVE_TOA_UP_FLUX_NAME in available_target_names
            and example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME in
            available_target_names
    ):
        available_target_names += [
            LONGWAVE_ALL_FLUX_NAME, LONGWAVE_NET_FLUX_NAME
        ]

    found_data_flags = numpy.array(
        [t in available_target_names for t in TARGET_NAME_BY_STATISTIC],
        dtype=bool
    )
    plot_statistic_indices = numpy.where(found_data_flags)[0]
    num_statistics = len(plot_statistic_indices)

    pd = prediction_dict
    letter_label = None
    panel_file_names = [''] * num_statistics

    for m in range(num_statistics):
        print(SEPARATOR_STRING)

        k = plot_statistic_indices[m]
        metric_matrix = numpy.full((num_grid_rows, num_grid_columns), numpy.nan)

        if TARGET_NAME_BY_STATISTIC[k] in [
                example_utils.SHORTWAVE_HEATING_RATE_NAME,
                example_utils.LONGWAVE_HEATING_RATE_NAME
        ]:
            channel_index = training_option_dict[
                neural_net.VECTOR_TARGET_NAMES_KEY
            ].index(TARGET_NAME_BY_STATISTIC[k])

            actual_values = (
                pd[prediction_io.VECTOR_TARGETS_KEY][..., channel_index]
            )
            predicted_values = (
                pd[prediction_io.VECTOR_PREDICTIONS_KEY][..., channel_index]
            )

            if TARGET_HEIGHT_INDEX_BY_STATISTIC[k] > -1:
                actual_values = (
                    actual_values[:, TARGET_HEIGHT_INDEX_BY_STATISTIC[k]]
                )
                predicted_values = (
                    predicted_values[:, TARGET_HEIGHT_INDEX_BY_STATISTIC[k]]
                )

        elif TARGET_NAME_BY_STATISTIC[k] in [
                SHORTWAVE_NET_FLUX_NAME, SHORTWAVE_ALL_FLUX_NAME
        ]:
            down_flux_index = training_option_dict[
                neural_net.SCALAR_TARGET_NAMES_KEY
            ].index(example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME)

            up_flux_index = training_option_dict[
                neural_net.SCALAR_TARGET_NAMES_KEY
            ].index(example_utils.SHORTWAVE_TOA_UP_FLUX_NAME)

            actual_net_flux_values = (
                pd[prediction_io.SCALAR_TARGETS_KEY][:, down_flux_index] -
                pd[prediction_io.SCALAR_TARGETS_KEY][:, up_flux_index]
            )
            predicted_net_flux_values = (
                pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, down_flux_index] -
                pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, up_flux_index]
            )

            if TARGET_NAME_BY_STATISTIC[k] == SHORTWAVE_NET_FLUX_NAME:
                actual_values = actual_net_flux_values
                predicted_values = predicted_net_flux_values
            else:
                actual_values = numpy.concatenate((
                    pd[prediction_io.SCALAR_TARGETS_KEY][:, down_flux_index],
                    pd[prediction_io.SCALAR_TARGETS_KEY][:, up_flux_index],
                    actual_net_flux_values
                ))
                predicted_values = numpy.concatenate((
                    pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, down_flux_index],
                    pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, up_flux_index],
                    predicted_net_flux_values
                ))
        else:
            down_flux_index = training_option_dict[
                neural_net.SCALAR_TARGET_NAMES_KEY
            ].index(example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME)

            up_flux_index = training_option_dict[
                neural_net.SCALAR_TARGET_NAMES_KEY
            ].index(example_utils.LONGWAVE_TOA_UP_FLUX_NAME)

            actual_net_flux_values = (
                pd[prediction_io.SCALAR_TARGETS_KEY][:, down_flux_index] -
                pd[prediction_io.SCALAR_TARGETS_KEY][:, up_flux_index]
            )
            predicted_net_flux_values = (
                pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, down_flux_index] -
                pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, up_flux_index]
            )

            if TARGET_NAME_BY_STATISTIC[k] == LONGWAVE_NET_FLUX_NAME:
                actual_values = actual_net_flux_values
                predicted_values = predicted_net_flux_values
            else:
                actual_values = numpy.concatenate((
                    pd[prediction_io.SCALAR_TARGETS_KEY][:, down_flux_index],
                    pd[prediction_io.SCALAR_TARGETS_KEY][:, up_flux_index],
                    actual_net_flux_values
                ))
                predicted_values = numpy.concatenate((
                    pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, down_flux_index],
                    pd[prediction_io.SCALAR_PREDICTIONS_KEY][:, up_flux_index],
                    predicted_net_flux_values
                ))

        for i in range(num_grid_rows):
            print((
                'Have computed {0:s} at {1:d} of {2:d} grid rows...'
            ).format(
                STATISTIC_NAMES[k], i, num_grid_rows
            ))

            for j in range(num_grid_columns):
                these_indices = grids.find_events_in_grid_cell(
                    event_x_coords_metres=longitudes_deg_e,
                    event_y_coords_metres=latitudes_deg_n,
                    grid_edge_x_coords_metres=grid_edge_longitudes_deg,
                    grid_edge_y_coords_metres=grid_edge_latitudes_deg,
                    row_index=i, column_index=j, verbose=False
                )

                if len(these_indices) < min_num_examples:
                    continue

                if 'rmse' in STATISTIC_NAMES[k]:
                    metric_matrix[i, j] = evaluation._get_mse_one_scalar(
                        target_values=actual_values[these_indices],
                        predicted_values=predicted_values[these_indices]
                    )[0]

                    metric_matrix[i, j] = numpy.sqrt(metric_matrix[i, j])
                elif 'dwmse' in STATISTIC_NAMES[k]:
                    these_weights = numpy.maximum(
                        numpy.absolute(actual_values[these_indices]),
                        numpy.absolute(predicted_values[these_indices])
                    )
                    these_squared_errors = (
                        actual_values[these_indices] -
                        predicted_values[these_indices]
                    ) ** 2

                    metric_matrix[i, j] = numpy.mean(
                        these_weights * these_squared_errors
                    )
                else:
                    metric_matrix[i, j] = evaluation._get_bias_one_scalar(
                        target_values=actual_values[these_indices],
                        predicted_values=predicted_values[these_indices]
                    )

        print('Have computed {0:s} for all {1:d} grid rows!'.format(
            STATISTIC_NAMES[k], num_grid_rows
        ))
        print(SEPARATOR_STRING)

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        panel_file_names[m] = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, STATISTIC_NAMES[k]
        )

        _plot_one_score(
            score_matrix=metric_matrix,
            grid_latitudes_deg_n=grid_latitudes_deg_n,
            grid_longitudes_deg_e=grid_longitudes_deg_e,
            border_latitudes_deg_n=border_latitudes_deg_n,
            border_longitudes_deg_e=border_longitudes_deg_e,
            title_string=STATISTIC_NAMES_FANCY[k], letter_label=letter_label,
            output_file_name=panel_file_names[m]
        )

    num_panel_rows = int(numpy.floor(
        numpy.sqrt(num_statistics)
    ))
    num_panel_columns = int(numpy.ceil(
        float(num_statistics) / num_panel_rows
    ))

    concat_file_name = '{0:s}/errors_by_space.jpg'.format(output_dir_name)
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))
    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_file_name,
        num_panel_rows=num_panel_rows, num_panel_columns=num_panel_columns
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_file_name,
        output_file_name=concat_file_name,
        output_size_pixels=int(1e7)
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        grid_spacing_deg=getattr(INPUT_ARG_OBJECT, GRID_SPACING_ARG_NAME),
        min_num_examples=getattr(INPUT_ARG_OBJECT, MIN_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
