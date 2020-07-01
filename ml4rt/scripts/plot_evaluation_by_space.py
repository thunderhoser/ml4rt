"""Plots evaluation scores by spatial region."""

import os
import argparse
import numpy
import scipy.stats
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import imagemagick_utils
from ml4rt.io import example_io
from ml4rt.io import prediction_io
from ml4rt.utils import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SCALAR_SKEWNESS_KEY = 'scalar_skewness'
VECTOR_SKEWNESS_KEY = 'vector_skewness'
AUX_SKEWNESS_KEY = 'aux_skewness'
NUM_EXAMPLES_KEY = 'num_examples'

# TODO(thunderhoser): Put this somewhere general.
TARGET_NAME_TO_UNITS = {
    example_io.SHORTWAVE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_io.SHORTWAVE_UP_FLUX_NAME: r'W m$^{-2}$',
    example_io.SHORTWAVE_HEATING_RATE_NAME: r'K day$^{-1}$',
    example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_io.SHORTWAVE_TOA_UP_FLUX_NAME: r'W m$^{-2}$',
    evaluation.NET_FLUX_NAME: r'W m$^{-2}$',
    evaluation.HIGHEST_UP_FLUX_NAME: r'W m$^{-2}$',
    evaluation.LOWEST_DOWN_FLUX_NAME: r'W m$^{-2}$'
}

MIN_COLOUR_PERCENTILE = 1.
MAX_COLOUR_PERCENTILE = 99.

DEFAULT_COLOUR_MAP_OBJECT = pyplot.get_cmap('plasma')
COUNT_COLOUR_MAP_OBJECT = pyplot.get_cmap('viridis')
BIAS_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')
SKEWNESS_COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')

COLOUR_BAR_FONT_SIZE = 16
TITLE_FONT_SIZE = 24
PANEL_LETTER_FONT_SIZE = 36

NUM_PARALLELS = 8
NUM_MERIDIANS = 6
RESOLUTION_STRING = 'l'
BORDER_WIDTH = 0.25
BORDER_COLOUR = numpy.full(3, 152. / 255)

FIGURE_RESOLUTION_DPI = 300
CONCAT_FIGURE_SIZE_PX = int(1e7)

pyplot.rc('font', size=TITLE_FONT_SIZE)
pyplot.rc('axes', titlesize=TITLE_FONT_SIZE)
pyplot.rc('axes', labelsize=TITLE_FONT_SIZE)
pyplot.rc('xtick', labelsize=TITLE_FONT_SIZE)
pyplot.rc('ytick', labelsize=TITLE_FONT_SIZE)
pyplot.rc('legend', fontsize=TITLE_FONT_SIZE)
pyplot.rc('figure', titlesize=TITLE_FONT_SIZE)

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
GRID_METAFILE_ARG_NAME = 'input_grid_metafile_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of directory with evaluation files (one for each grid cell).  Files '
    'will be found by `evaluation.find_file` and read by '
    '`evaluation.read_file`.'
)
GRID_METAFILE_HELP_STRING = (
    'Path to file with grid metadata.  Will be read by '
    '`prediction_io.read_grid_metafile`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + GRID_METAFILE_ARG_NAME, type=str, required=True,
    help=GRID_METAFILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_score_one_field(
        latitude_matrix_deg, longitude_matrix_deg, score_matrix,
        colour_map_object, min_colour_value, max_colour_value,
        taper_cbar_top, taper_cbar_bottom, log_scale=False):
    """Plots one score for one field.

    M = number of rows in grid
    N = number of columns in grid

    :param latitude_matrix_deg: M-by-N numpy array of latitudes (deg N).
    :param longitude_matrix_deg: M-by-N numpy array of longitudes (deg E).
    :param score_matrix: M-by-N numpy array of score values.
    :param colour_map_object: Colour scheme (instance of `matplotlib.pyplot.cm`).
    :param min_colour_value: Minimum value in colour bar.
    :param max_colour_value: Max value in colour bar.
    :param taper_cbar_top: Boolean flag.  If True, will taper bottom of colour
        bar, implying that lower values are possible.
    :param taper_cbar_bottom: Same but for top of colour bar.
    :param log_scale: Boolean flag.  If True, will make colour bar logarithmic.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    (
        figure_object, axes_object, basemap_object
    ) = plotting_utils.create_equidist_cylindrical_map(
        min_latitude_deg=latitude_matrix_deg[0, 0],
        max_latitude_deg=latitude_matrix_deg[-1, -1],
        min_longitude_deg=longitude_matrix_deg[0, 0],
        max_longitude_deg=longitude_matrix_deg[-1, -1],
        resolution_string=RESOLUTION_STRING
    )

    latitude_spacing_deg = latitude_matrix_deg[1, 0] - latitude_matrix_deg[0, 0]
    longitude_spacing_deg = (
        longitude_matrix_deg[0, 1] - longitude_matrix_deg[0, 0]
    )

    print(numpy.sum(numpy.invert(numpy.isnan(score_matrix))))

    (
        score_matrix_at_edges, grid_edge_latitudes_deg, grid_edge_longitudes_deg
    ) = grids.latlng_field_grid_points_to_edges(
        field_matrix=score_matrix,
        min_latitude_deg=latitude_matrix_deg[0, 0],
        min_longitude_deg=longitude_matrix_deg[0, 0],
        lat_spacing_deg=latitude_spacing_deg,
        lng_spacing_deg=longitude_spacing_deg
    )

    score_matrix_at_edges = numpy.ma.masked_where(
        numpy.isnan(score_matrix_at_edges), score_matrix_at_edges
    )

    plotting_utils.plot_coastlines(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR, line_width=BORDER_WIDTH
    )
    plotting_utils.plot_countries(
        basemap_object=basemap_object, axes_object=axes_object,
        line_colour=BORDER_COLOUR, line_width=BORDER_WIDTH
    )
    # plotting_utils.plot_states_and_provinces(
    #     basemap_object=basemap_object, axes_object=axes_object,
    #     line_colour=BORDER_COLOUR
    # )
    plotting_utils.plot_parallels(
        basemap_object=basemap_object, axes_object=axes_object,
        num_parallels=NUM_PARALLELS, line_width=0
    )
    plotting_utils.plot_meridians(
        basemap_object=basemap_object, axes_object=axes_object,
        num_meridians=NUM_MERIDIANS, line_width=0
    )

    pyplot.pcolormesh(
        grid_edge_longitudes_deg, grid_edge_latitudes_deg,
        score_matrix_at_edges, cmap=colour_map_object,
        vmin=min_colour_value, vmax=max_colour_value, shading='flat',
        edgecolors='None', axes=axes_object, zorder=-1e12
    )

    colour_bar_object = plotting_utils.plot_linear_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=score_matrix,
        colour_map_object=colour_map_object, min_value=min_colour_value,
        max_value=max_colour_value, orientation_string='horizontal',
        extend_min=taper_cbar_bottom, extend_max=taper_cbar_top,
        padding=0.05, font_size=COLOUR_BAR_FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()

    if log_scale:
        tick_strings = [
            '{0:d}'.format(int(numpy.round(10 ** v))) for v in tick_values
        ]
    elif numpy.nanmax(numpy.absolute(score_matrix)) >= 6:
        tick_strings = [
            '{0:d}'.format(int(numpy.round(v))) for v in tick_values
        ]
    else:
        tick_strings = ['{0:.2f}'.format(v) for v in tick_values]

    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    return figure_object, axes_object


def _plot_all_scores_one_field(
        latitude_matrix_deg, longitude_matrix_deg, mae_matrix, rmse_matrix,
        bias_matrix, mae_skill_score_matrix, mse_skill_score_matrix,
        correlation_matrix, skewness_matrix, field_name, output_dir_name,
        height_m_agl=None):
    """Plots all evaluation scores for one field.

    M = number of rows in grid
    N = number of columns in grid

    :param latitude_matrix_deg: M-by-N numpy array of latitudes (deg N).
    :param longitude_matrix_deg: M-by-N numpy array of longitudes (deg E).
    :param mae_matrix: M-by-N numpy array of MAE (mean absolute error) values.
    :param rmse_matrix: Same but for RMSE (root mean squared error).
    :param bias_matrix: Same but for bias.
    :param mae_skill_score_matrix: Same but for MAE skill score.
    :param mse_skill_score_matrix: Same but for MSE skill score.
    :param correlation_matrix: Same but for correlation.
    :param skewness_matrix: Same but for skewness.
    :param field_name: Name of field for which scores are being plotted.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    :param height_m_agl: Height (metres above ground level).  If plotting for
        scalar field, leave this argument alone.
    """

    out_file_name_prefix = '{0:s}/{1:s}'.format(
        output_dir_name, field_name.replace('_', '-')
    )

    if height_m_agl is not None:
        out_file_name_prefix += '_{0:05d}metres'.format(
            int(numpy.round(height_m_agl))
        )

    panel_file_names = []

    # Plot MAE.
    this_min_value = numpy.nanpercentile(mae_matrix, MIN_COLOUR_PERCENTILE)
    this_max_value = numpy.nanpercentile(mae_matrix, MAX_COLOUR_PERCENTILE)

    figure_object, axes_object = _plot_score_one_field(
        latitude_matrix_deg=latitude_matrix_deg,
        longitude_matrix_deg=longitude_matrix_deg, score_matrix=mae_matrix,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        min_colour_value=this_min_value, max_colour_value=this_max_value,
        taper_cbar_top=True, taper_cbar_bottom=this_min_value != 0,
        log_scale=False
    )

    title_string = 'MAE ({0:s})'.format(TARGET_NAME_TO_UNITS[field_name])
    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)
    plotting_utils.label_axes(
        axes_object=axes_object, label_string='(a)',
        font_size=PANEL_LETTER_FONT_SIZE
    )

    panel_file_names.append('{0:s}_mae.jpg'.format(out_file_name_prefix))
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))

    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot RMSE.
    this_min_value = numpy.nanpercentile(rmse_matrix, MIN_COLOUR_PERCENTILE)
    this_max_value = numpy.nanpercentile(rmse_matrix, MAX_COLOUR_PERCENTILE)

    figure_object, axes_object = _plot_score_one_field(
        latitude_matrix_deg=latitude_matrix_deg,
        longitude_matrix_deg=longitude_matrix_deg, score_matrix=rmse_matrix,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        min_colour_value=this_min_value, max_colour_value=this_max_value,
        taper_cbar_top=True, taper_cbar_bottom=this_min_value != 0,
        log_scale=False
    )

    title_string = 'RMSE ({0:s})'.format(TARGET_NAME_TO_UNITS[field_name])
    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)
    plotting_utils.label_axes(
        axes_object=axes_object, label_string='(b)',
        font_size=PANEL_LETTER_FONT_SIZE
    )

    panel_file_names.append('{0:s}_rmse.jpg'.format(out_file_name_prefix))
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))

    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot bias.
    this_max_value = numpy.nanpercentile(
        numpy.absolute(bias_matrix), MAX_COLOUR_PERCENTILE
    )
    this_min_value = -1 * this_max_value

    figure_object, axes_object = _plot_score_one_field(
        latitude_matrix_deg=latitude_matrix_deg,
        longitude_matrix_deg=longitude_matrix_deg, score_matrix=bias_matrix,
        colour_map_object=BIAS_COLOUR_MAP_OBJECT,
        min_colour_value=this_min_value, max_colour_value=this_max_value,
        taper_cbar_top=True, taper_cbar_bottom=True, log_scale=False
    )

    title_string = 'Bias ({0:s})'.format(TARGET_NAME_TO_UNITS[field_name])
    axes_object.set_title(title_string, fontsize=TITLE_FONT_SIZE)
    plotting_utils.label_axes(
        axes_object=axes_object, label_string='(c)',
        font_size=PANEL_LETTER_FONT_SIZE
    )

    panel_file_names.append('{0:s}_bias.jpg'.format(out_file_name_prefix))
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))

    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot MAE skill score.
    this_min_value = numpy.nanpercentile(
        mae_skill_score_matrix, MIN_COLOUR_PERCENTILE
    )
    this_max_value = numpy.nanpercentile(
        mae_skill_score_matrix, MAX_COLOUR_PERCENTILE
    )

    figure_object, axes_object = _plot_score_one_field(
        latitude_matrix_deg=latitude_matrix_deg,
        longitude_matrix_deg=longitude_matrix_deg,
        score_matrix=mae_skill_score_matrix,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        min_colour_value=this_min_value, max_colour_value=this_max_value,
        taper_cbar_top=this_max_value != 1, taper_cbar_bottom=True,
        log_scale=False
    )

    axes_object.set_title('MAE skill score', fontsize=TITLE_FONT_SIZE)
    plotting_utils.label_axes(
        axes_object=axes_object, label_string='(d)',
        font_size=PANEL_LETTER_FONT_SIZE
    )

    panel_file_names.append('{0:s}_maess.jpg'.format(out_file_name_prefix))
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))

    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot MSE skill score.
    this_min_value = numpy.nanpercentile(
        mse_skill_score_matrix, MIN_COLOUR_PERCENTILE
    )
    this_max_value = numpy.nanpercentile(
        mse_skill_score_matrix, MAX_COLOUR_PERCENTILE
    )

    figure_object, axes_object = _plot_score_one_field(
        latitude_matrix_deg=latitude_matrix_deg,
        longitude_matrix_deg=longitude_matrix_deg,
        score_matrix=mse_skill_score_matrix,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        min_colour_value=this_min_value, max_colour_value=this_max_value,
        taper_cbar_top=this_max_value != 1, taper_cbar_bottom=True,
        log_scale=False
    )

    axes_object.set_title('MSE skill score', fontsize=TITLE_FONT_SIZE)
    plotting_utils.label_axes(
        axes_object=axes_object, label_string='(e)',
        font_size=PANEL_LETTER_FONT_SIZE
    )

    panel_file_names.append('{0:s}_msess.jpg'.format(out_file_name_prefix))
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))

    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot correlation.
    this_min_value = numpy.nanpercentile(
        correlation_matrix, MIN_COLOUR_PERCENTILE
    )
    this_max_value = numpy.nanpercentile(
        correlation_matrix, MAX_COLOUR_PERCENTILE
    )

    figure_object, axes_object = _plot_score_one_field(
        latitude_matrix_deg=latitude_matrix_deg,
        longitude_matrix_deg=longitude_matrix_deg,
        score_matrix=correlation_matrix,
        colour_map_object=DEFAULT_COLOUR_MAP_OBJECT,
        min_colour_value=this_min_value, max_colour_value=this_max_value,
        taper_cbar_top=this_max_value != 1,
        taper_cbar_bottom=this_min_value != 0,
        log_scale=False
    )

    axes_object.set_title('Correlation', fontsize=TITLE_FONT_SIZE)
    plotting_utils.label_axes(
        axes_object=axes_object, label_string='(f)',
        font_size=PANEL_LETTER_FONT_SIZE
    )

    panel_file_names.append('{0:s}_correlation.jpg'.format(
        out_file_name_prefix
    ))
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))

    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    # Plot skewness.
    this_max_value = numpy.nanpercentile(
        numpy.absolute(skewness_matrix), MAX_COLOUR_PERCENTILE
    )
    this_min_value = -1 * this_max_value

    figure_object, axes_object = _plot_score_one_field(
        latitude_matrix_deg=latitude_matrix_deg,
        longitude_matrix_deg=longitude_matrix_deg, score_matrix=skewness_matrix,
        colour_map_object=SKEWNESS_COLOUR_MAP_OBJECT,
        min_colour_value=this_min_value, max_colour_value=this_max_value,
        taper_cbar_top=True, taper_cbar_bottom=True, log_scale=False
    )

    axes_object.set_title('Skewness of actual values', fontsize=TITLE_FONT_SIZE)
    plotting_utils.label_axes(
        axes_object=axes_object, label_string='(g)',
        font_size=PANEL_LETTER_FONT_SIZE
    )

    panel_file_names.append('{0:s}_skewness.jpg'.format(out_file_name_prefix))
    print('Saving figure to: "{0:s}"...'.format(panel_file_names[-1]))

    figure_object.savefig(
        panel_file_names[-1], dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)

    concat_file_name = '{0:s}_all.jpg'.format(out_file_name_prefix)
    print('Concatenating panels to: "{0:s}"...'.format(concat_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names, output_file_name=concat_file_name,
        num_panel_rows=3, num_panel_columns=3
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_file_name, output_file_name=concat_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )

    for this_file_name in panel_file_names:
        os.remove(this_file_name)


def _augment_eval_table(result_table_xarray):
    """Augments evaluation table.

    Specifically, adds number of examples and skewness for each target variable.

    :param result_table_xarray: Table returned by `evaluation.read_file`.
    :return: result_table_xarray: Same but with number of examples and skewness
        for each target variable.
    """

    prediction_file_name = (
        result_table_xarray.attrs[evaluation.PREDICTION_FILE_KEY]
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    num_examples = len(prediction_dict[prediction_io.EXAMPLE_IDS_KEY])
    result_table_xarray.attrs[NUM_EXAMPLES_KEY] = num_examples

    scalar_target_matrix = prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    if scalar_target_matrix.size == 0:
        scalar_skewness_matrix = numpy.full(0, 0.)
    else:
        scalar_skewness_matrix = scipy.stats.skew(
            scalar_target_matrix, axis=0, bias=False, nan_policy='omit'
        )

    these_dim = (evaluation.SCALAR_FIELD_DIM,)
    result_table_xarray.update({
        SCALAR_SKEWNESS_KEY: (these_dim, scalar_skewness_matrix)
    })

    vector_skewness_matrix = scipy.stats.skew(
        prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
        axis=0, bias=False, nan_policy='omit'
    )

    these_dim = (evaluation.HEIGHT_DIM, evaluation.VECTOR_FIELD_DIM)
    result_table_xarray.update({
        VECTOR_SKEWNESS_KEY: (these_dim, vector_skewness_matrix)
    })

    try:
        _ = result_table_xarray.coords[evaluation.AUX_TARGET_FIELD_DIM].values
    except KeyError:
        return result_table_xarray

    example_dict = {
        example_io.SCALAR_TARGET_NAMES_KEY:
            result_table_xarray.coords[evaluation.SCALAR_FIELD_DIM].values,
        example_io.VECTOR_TARGET_NAMES_KEY:
            result_table_xarray.coords[evaluation.VECTOR_FIELD_DIM].values,
        example_io.HEIGHTS_KEY:
            numpy.round(
                result_table_xarray.coords[evaluation.HEIGHT_DIM].values
            ).astype(int)
    }

    aux_target_matrix = evaluation.get_aux_fields(
        prediction_dict=prediction_dict, example_dict=example_dict
    )[evaluation.AUX_TARGET_VALS_KEY]

    aux_skewness_matrix = scipy.stats.skew(
        aux_target_matrix, axis=0, bias=False, nan_policy='omit'
    )

    these_dim = (evaluation.AUX_TARGET_FIELD_DIM,)
    result_table_xarray.update({
        AUX_SKEWNESS_KEY: (these_dim, aux_skewness_matrix)
    })

    return result_table_xarray


def _run(evaluation_dir_name, grid_metafile_name, output_dir_name):
    """Plots evaluation scores by spatial region.

    This is effectively the main method.

    :param evaluation_dir_name: See documentation at top of file.
    :param grid_metafile_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Read metadata for grid.
    print('Reading grid metadata from: "{0:s}"...'.format(grid_metafile_name))
    grid_point_latitudes_deg, grid_point_longitudes_deg = (
        prediction_io.read_grid_metafile(grid_metafile_name)
    )

    num_grid_rows = len(grid_point_latitudes_deg)
    num_grid_columns = len(grid_point_longitudes_deg)

    latitude_matrix_deg, longitude_matrix_deg = (
        grids.latlng_vectors_to_matrices(
            unique_latitudes_deg=grid_point_latitudes_deg,
            unique_longitudes_deg=grid_point_longitudes_deg
        )
    )

    # Read evaluation files.
    eval_table_matrix_xarray = numpy.full(
        (num_grid_rows, num_grid_columns), None, dtype=object
    )

    scalar_field_names = None
    aux_field_names = None
    vector_field_names = None
    heights_m_agl = None

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            this_file_name = evaluation.find_file(
                directory_name=evaluation_dir_name, grid_row=i, grid_column=j,
                raise_error_if_missing=False
            )

            if not os.path.isfile(this_file_name):
                continue

            print('Reading data from: "{0:s}"...'.format(this_file_name))
            eval_table_matrix_xarray[i, j] = evaluation.read_file(
                this_file_name
            )
            eval_table_matrix_xarray[i, j] = _augment_eval_table(
                eval_table_matrix_xarray[i, j]
            )

            if scalar_field_names is None:
                t = eval_table_matrix_xarray[i, j]

                scalar_field_names = (
                    t.coords[evaluation.SCALAR_FIELD_DIM].values
                )
                vector_field_names = (
                    t.coords[evaluation.VECTOR_FIELD_DIM].values
                )
                heights_m_agl = numpy.round(
                    t.coords[evaluation.HEIGHT_DIM].values
                ).astype(int)

                try:
                    aux_field_names = (
                        t.coords[evaluation.AUX_TARGET_FIELD_DIM].values
                    )
                except KeyError:
                    aux_field_names = []

    print(SEPARATOR_STRING)

    evaluation_tables_xarray = numpy.reshape(
        eval_table_matrix_xarray, num_grid_rows * num_grid_columns
    )
    nan_array = numpy.full(len(scalar_field_names), numpy.nan)

    scalar_mae_matrix = numpy.vstack([
        nan_array if t is None else t[evaluation.SCALAR_MAE_KEY].values
        for t in evaluation_tables_xarray
    ])
    scalar_rmse_matrix = numpy.sqrt(numpy.vstack([
        nan_array if t is None else t[evaluation.SCALAR_MSE_KEY].values
        for t in evaluation_tables_xarray
    ]))
    scalar_bias_matrix = numpy.vstack([
        nan_array if t is None else t[evaluation.SCALAR_BIAS_KEY].values
        for t in evaluation_tables_xarray
    ])
    scalar_mae_skill_matrix = numpy.vstack([
        nan_array if t is None else t[evaluation.SCALAR_MAE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ])
    scalar_mse_skill_matrix = numpy.vstack([
        nan_array if t is None else t[evaluation.SCALAR_MSE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ])
    scalar_correlation_matrix = numpy.vstack([
        nan_array if t is None else t[evaluation.SCALAR_CORRELATION_KEY].values
        for t in evaluation_tables_xarray
    ])
    scalar_skewness_matrix = numpy.vstack([
        nan_array if t is None else t[SCALAR_SKEWNESS_KEY].values
        for t in evaluation_tables_xarray
    ])

    grid_dim_tuple = (num_grid_rows, num_grid_columns)

    for k in range(len(scalar_field_names)):
        _plot_all_scores_one_field(
            latitude_matrix_deg=latitude_matrix_deg,
            longitude_matrix_deg=longitude_matrix_deg,
            mae_matrix=numpy.reshape(scalar_mae_matrix[:, k], grid_dim_tuple),
            rmse_matrix=numpy.reshape(scalar_rmse_matrix[:, k], grid_dim_tuple),
            bias_matrix=numpy.reshape(scalar_bias_matrix[:, k], grid_dim_tuple),
            mae_skill_score_matrix=numpy.reshape(
                scalar_mae_skill_matrix[:, k], grid_dim_tuple
            ),
            mse_skill_score_matrix=numpy.reshape(
                scalar_mse_skill_matrix[:, k], grid_dim_tuple
            ),
            correlation_matrix=numpy.reshape(
                scalar_correlation_matrix[:, k], grid_dim_tuple
            ),
            skewness_matrix=numpy.reshape(
                scalar_skewness_matrix[:, k], grid_dim_tuple
            ),
            field_name=scalar_field_names[k], output_dir_name=output_dir_name
        )

        if k == len(scalar_field_names) - 1:
            print(SEPARATOR_STRING)
        else:
            print('\n')

    if len(aux_field_names) > 0:
        nan_array = numpy.full(len(aux_field_names), numpy.nan)

        aux_mae_matrix = numpy.vstack([
            nan_array if t is None else t[evaluation.AUX_MAE_KEY].values
            for t in evaluation_tables_xarray
        ])
        aux_rmse_matrix = numpy.sqrt(numpy.vstack([
            nan_array if t is None else t[evaluation.AUX_MSE_KEY].values
            for t in evaluation_tables_xarray
        ]))
        aux_bias_matrix = numpy.vstack([
            nan_array if t is None else t[evaluation.AUX_BIAS_KEY].values
            for t in evaluation_tables_xarray
        ])
        aux_mae_skill_matrix = numpy.vstack([
            nan_array if t is None else t[evaluation.AUX_MAE_SKILL_KEY].values
            for t in evaluation_tables_xarray
        ])
        aux_mse_skill_matrix = numpy.vstack([
            nan_array if t is None else t[evaluation.AUX_MSE_SKILL_KEY].values
            for t in evaluation_tables_xarray
        ])
        aux_correlation_matrix = numpy.vstack([
            nan_array if t is None else t[evaluation.AUX_CORRELATION_KEY].values
            for t in evaluation_tables_xarray
        ])
        aux_skewness_matrix = numpy.vstack([
            nan_array if t is None else t[AUX_SKEWNESS_KEY].values
            for t in evaluation_tables_xarray
        ])

    for k in range(len(aux_field_names)):
        _plot_all_scores_one_field(
            latitude_matrix_deg=latitude_matrix_deg,
            longitude_matrix_deg=longitude_matrix_deg,
            mae_matrix=numpy.reshape(aux_mae_matrix[:, k], grid_dim_tuple),
            rmse_matrix=numpy.reshape(aux_rmse_matrix[:, k], grid_dim_tuple),
            bias_matrix=numpy.reshape(aux_bias_matrix[:, k], grid_dim_tuple),
            mae_skill_score_matrix=numpy.reshape(
                aux_mae_skill_matrix[:, k], grid_dim_tuple
            ),
            mse_skill_score_matrix=numpy.reshape(
                aux_mse_skill_matrix[:, k], grid_dim_tuple
            ),
            correlation_matrix=numpy.reshape(
                aux_correlation_matrix[:, k], grid_dim_tuple
            ),
            skewness_matrix=numpy.reshape(
                aux_skewness_matrix[:, k], grid_dim_tuple
            ),
            field_name=aux_field_names[k], output_dir_name=output_dir_name
        )

        if k == len(aux_field_names) - 1:
            print(SEPARATOR_STRING)
        else:
            print('\n')

    nan_array = numpy.full(
        (len(heights_m_agl), len(vector_field_names)), numpy.nan
    )

    vector_mae_matrix = numpy.stack([
        nan_array if t is None else t[evaluation.VECTOR_MAE_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)
    vector_rmse_matrix = numpy.sqrt(numpy.stack([
        nan_array if t is None else t[evaluation.VECTOR_MSE_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0))
    vector_bias_matrix = numpy.stack([
        nan_array if t is None else t[evaluation.VECTOR_BIAS_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)
    vector_mae_skill_matrix = numpy.stack([
        nan_array if t is None else t[evaluation.VECTOR_MAE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)
    vector_mse_skill_matrix = numpy.stack([
        nan_array if t is None else t[evaluation.VECTOR_MSE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)
    vector_correlation_matrix = numpy.stack([
        nan_array if t is None else t[evaluation.VECTOR_CORRELATION_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)
    vector_skewness_matrix = numpy.stack([
        nan_array if t is None else t[VECTOR_SKEWNESS_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)

    for k in range(len(vector_field_names)):
        for j in range(len(heights_m_agl)):
            _plot_all_scores_one_field(
                latitude_matrix_deg=latitude_matrix_deg,
                longitude_matrix_deg=longitude_matrix_deg,
                mae_matrix=numpy.reshape(
                    vector_mae_matrix[:, j, k], grid_dim_tuple
                ),
                rmse_matrix=numpy.reshape(
                    vector_rmse_matrix[:, j, k], grid_dim_tuple
                ),
                bias_matrix=numpy.reshape(
                    vector_bias_matrix[:, j, k], grid_dim_tuple
                ),
                mae_skill_score_matrix=numpy.reshape(
                    vector_mae_skill_matrix[:, j, k], grid_dim_tuple
                ),
                mse_skill_score_matrix=numpy.reshape(
                    vector_mse_skill_matrix[:, j, k], grid_dim_tuple
                ),
                correlation_matrix=numpy.reshape(
                    vector_correlation_matrix[:, j, k], grid_dim_tuple
                ),
                skewness_matrix=numpy.reshape(
                    vector_skewness_matrix[:, j, k], grid_dim_tuple
                ),
                field_name=vector_field_names[k], height_m_agl=heights_m_agl[j],
                output_dir_name=output_dir_name
            )

        if k == len(vector_field_names) - 1:
            print(SEPARATOR_STRING)
        else:
            print('\n')

    num_examples_array = numpy.array([
        numpy.nan if t is None else t.attrs[NUM_EXAMPLES_KEY]
        for t in evaluation_tables_xarray
    ])
    num_examples_matrix = numpy.reshape(num_examples_array, grid_dim_tuple)
    max_colour_value = numpy.nanpercentile(
        num_examples_matrix, MAX_COLOUR_PERCENTILE
    )

    figure_object, axes_object = _plot_score_one_field(
        latitude_matrix_deg=latitude_matrix_deg,
        longitude_matrix_deg=longitude_matrix_deg,
        score_matrix=num_examples_matrix,
        colour_map_object=COUNT_COLOUR_MAP_OBJECT,
        min_colour_value=0., max_colour_value=max_colour_value,
        taper_cbar_top=True, taper_cbar_bottom=False, log_scale=False
    )

    axes_object.set_title('Number of examples', fontsize=TITLE_FONT_SIZE)
    figure_file_name = '{0:s}/num_examples.jpg'.format(output_dir_name)

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        grid_metafile_name=getattr(INPUT_ARG_OBJECT, GRID_METAFILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
