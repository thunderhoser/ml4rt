"""Plots evaluation scores by time of day and year."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from descartes import PolygonPatch
from gewittergefahr.gg_utils import polygons
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io
from ml4rt.utils import evaluation

# TODO(thunderhoser): Make confidence level input arg to script (once evaluation
# files deal with bootstrapping).

# TODO(thunderhoser): Include reference to prediction file in evaluation files.

MARKER_TYPE = 'o'
MARKER_SIZE = 16
LINE_WIDTH = 4

MAE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
RMSE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
BIAS_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255

MAE_SKILL_COLOUR = MAE_COLOUR
MSE_SKILL_COLOUR = RMSE_COLOUR
CORRELATION_COLOUR = BIAS_COLOUR
POLYGON_OPACITY = 0.5

MONTH_INDICES = numpy.linspace(1, 12, num=12, dtype=int)
MONTH_STRINGS = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
    'Dec'
]

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

FONT_SIZE = 30
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
NUM_ANGLE_BINS_ARG_NAME = 'num_zenith_angle_bins'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of directory with evaluation files (one for each zenith-angle bin, '
    'one for each month).  Files will be found by `evaluation.find_file` and '
    'read by `evaluation.read_file`.'
)
NUM_ANGLE_BINS_HELP_STRING = 'Number of bins for zenith angle.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ANGLE_BINS_ARG_NAME, type=int, required=False, default=9,
    help=NUM_ANGLE_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _read_files_one_split(evaluation_file_names):
    """Reads evaluation files for one time split.

    T = number of time chunks in this split

    :param evaluation_file_names: length-T list of paths to input files.
    :return: evaluation_tables_xarray: length-T list of xarray tables with
        results.
    """

    num_time_chunks = len(evaluation_file_names)
    evaluation_tables_xarray = []

    for i in range(num_time_chunks):
        print('Reading data from: "{0:s}"...'.format(evaluation_file_names[i]))
        evaluation_tables_xarray.append(evaluation.read_file(
            evaluation_file_names[i]
        ))

    return evaluation_tables_xarray


def _confidence_interval_to_polygon(x_values, y_value_matrix, confidence_level):
    """Turns confidence interval into polygon.

    P = number of points
    B = number of bootstrap replicates

    :param x_values: length-P numpy array of x-values.
    :param y_value_matrix: P-by-B numpy array of y-values.
    :param confidence_level: Confidence level (in range 0...1).
    :return: polygon_object: Instance of `shapely.geometry.Polygon`.
    """

    min_percentile = 50 * (1. - confidence_level)
    max_percentile = 50 * (1. + confidence_level)

    y_values_bottom = numpy.nanpercentile(
        y_value_matrix, min_percentile, axis=1, interpolation='linear'
    )
    y_values_top = numpy.nanpercentile(
        y_value_matrix, max_percentile, axis=1, interpolation='linear'
    )

    real_indices = numpy.where(
        numpy.invert(numpy.isnan(y_values_bottom))
    )[0]

    if len(real_indices) == 0:
        return None

    real_x_values = x_values[real_indices]
    real_y_values_bottom = y_values_bottom[real_indices]
    real_y_values_top = y_values_top[real_indices]

    these_x = numpy.concatenate((
        real_x_values, real_x_values[::-1], real_x_values[[0]]
    ))
    these_y = numpy.concatenate((
        real_y_values_top, real_y_values_bottom[::-1], real_y_values_top[[0]]
    ))

    return polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=these_x, exterior_y_coords=these_y
    )


def _plot_scores_with_units(mae_matrix, rmse_matrix, bias_matrix, plot_legend,
                            confidence_level=None):
    """Plots scores with physical units.

    B = number of bootstrap replicates
    T = number of time chunks

    :param mae_matrix: T-by-B numpy array of MAE (mean absolute error) values.
    :param rmse_matrix: T-by-B numpy array of RMSE (root mean squared error)
        values.
    :param bias_matrix: T-by-B numpy array of biases.
    :param plot_legend: Boolean flag.  If True, will plot legend above figure.
    :param confidence_level: [used only if B > 1]
        Level for confidence intervals (in range 0...1).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Housekeeping.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    num_time_chunks = mae_matrix.shape[0]
    num_bootstrap_reps = mae_matrix.shape[1]

    x_values = numpy.linspace(
        0, num_time_chunks - 1, num=num_time_chunks, dtype=float
    )

    # Plot mean MAE.
    this_handle = axes_object.plot(
        x_values, numpy.mean(mae_matrix, axis=1), color=MAE_COLOUR,
        linewidth=LINE_WIDTH, marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=MAE_COLOUR, markeredgecolor=MAE_COLOUR,
        markeredgewidth=0
    )[0]

    legend_handles = [this_handle]
    legend_strings = ['MAE']

    # Plot confidence interval for MAE.
    if num_bootstrap_reps > 1:
        polygon_object = _confidence_interval_to_polygon(
            x_values=x_values, y_value_matrix=mae_matrix,
            confidence_level=confidence_level
        )

        polygon_colour = matplotlib.colors.to_rgba(MAE_COLOUR, POLYGON_OPACITY)
        patch_object = PolygonPatch(
            polygon_object, lw=0, ec=polygon_object, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    # Plot mean RMSE.
    this_handle = axes_object.plot(
        x_values, numpy.mean(rmse_matrix, axis=1), color=RMSE_COLOUR,
        linewidth=LINE_WIDTH, marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=RMSE_COLOUR, markeredgecolor=RMSE_COLOUR,
        markeredgewidth=0
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('RMSE')

    # Plot confidence interval for RMSE.
    if num_bootstrap_reps > 1:
        polygon_object = _confidence_interval_to_polygon(
            x_values=x_values, y_value_matrix=rmse_matrix,
            confidence_level=confidence_level
        )

        polygon_colour = matplotlib.colors.to_rgba(RMSE_COLOUR, POLYGON_OPACITY)
        patch_object = PolygonPatch(
            polygon_object, lw=0, ec=polygon_object, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    # Plot mean bias.
    this_handle = axes_object.plot(
        x_values, numpy.mean(bias_matrix, axis=1), color=BIAS_COLOUR,
        linewidth=LINE_WIDTH, marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=BIAS_COLOUR, markeredgecolor=BIAS_COLOUR,
        markeredgewidth=0
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Bias')

    # Plot confidence interval for bias.
    if num_bootstrap_reps > 1:
        polygon_object = _confidence_interval_to_polygon(
            x_values=x_values, y_value_matrix=bias_matrix,
            confidence_level=confidence_level
        )

        polygon_colour = matplotlib.colors.to_rgba(BIAS_COLOUR, POLYGON_OPACITY)
        patch_object = PolygonPatch(
            polygon_object, lw=0, ec=polygon_object, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    axes_object.set_xlim(
        numpy.min(x_values) - 0.5, numpy.max(x_values) + 0.5
    )

    if plot_legend:
        axes_object.legend(
            legend_handles, legend_strings, loc='lower center',
            bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True,
            ncol=len(legend_handles)
        )

    return figure_object, axes_object


def _plot_unitless_scores(
        mae_skill_score_matrix, mse_skill_score_matrix, correlation_matrix,
        plot_legend, confidence_level=None):
    """Plots scores without physical units.

    B = number of bootstrap replicates
    T = number of time chunks

    :param mae_skill_score_matrix: T-by-B numpy array of MAE (mean absolute
        error) skill scores.
    :param mse_skill_score_matrix: T-by-B numpy array of MSE (mean squared
        error) skill scores.
    :param correlation_matrix: T-by-B numpy array of correlations.
    :param plot_legend: See doc for `_plot_scores_with_units`.
    :param confidence_level: Same.
    :return: figure_object: Same.
    :return: axes_object: Same.
    """

    # Housekeeping.
    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    num_time_chunks = mae_skill_score_matrix.shape[0]
    num_bootstrap_reps = mae_skill_score_matrix.shape[1]

    x_values = numpy.linspace(
        0, num_time_chunks - 1, num=num_time_chunks, dtype=float
    )

    # Plot mean MAE skill score.
    this_handle = axes_object.plot(
        x_values, numpy.mean(mae_skill_score_matrix, axis=1),
        color=MAE_SKILL_COLOUR, linewidth=LINE_WIDTH, marker=MARKER_TYPE,
        markersize=MARKER_SIZE, markerfacecolor=MAE_SKILL_COLOUR,
        markeredgecolor=MAE_SKILL_COLOUR, markeredgewidth=0
    )[0]

    legend_handles = [this_handle]
    legend_strings = ['MAE skill']

    # Plot confidence interval for MAE skill score.
    if num_bootstrap_reps > 1:
        polygon_object = _confidence_interval_to_polygon(
            x_values=x_values, y_value_matrix=mae_skill_score_matrix,
            confidence_level=confidence_level
        )

        polygon_colour = matplotlib.colors.to_rgba(
            MAE_SKILL_COLOUR, POLYGON_OPACITY
        )
        patch_object = PolygonPatch(
            polygon_object, lw=0, ec=polygon_object, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    # Plot mean MSE skill score.
    this_handle = axes_object.plot(
        x_values, numpy.mean(mse_skill_score_matrix, axis=1),
        color=MSE_SKILL_COLOUR, linewidth=LINE_WIDTH, marker=MARKER_TYPE,
        markersize=MARKER_SIZE, markerfacecolor=MSE_SKILL_COLOUR,
        markeredgecolor=MSE_SKILL_COLOUR, markeredgewidth=0
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('MSE skill')

    # Plot confidence interval for MSE skill score.
    if num_bootstrap_reps > 1:
        polygon_object = _confidence_interval_to_polygon(
            x_values=x_values, y_value_matrix=mse_skill_score_matrix,
            confidence_level=confidence_level
        )

        polygon_colour = matplotlib.colors.to_rgba(
            MSE_SKILL_COLOUR, POLYGON_OPACITY
        )
        patch_object = PolygonPatch(
            polygon_object, lw=0, ec=polygon_object, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    # Plot mean correlation.
    this_handle = axes_object.plot(
        x_values, numpy.mean(correlation_matrix, axis=1),
        color=CORRELATION_COLOUR, linewidth=LINE_WIDTH, marker=MARKER_TYPE,
        markersize=MARKER_SIZE, markerfacecolor=CORRELATION_COLOUR,
        markeredgecolor=CORRELATION_COLOUR, markeredgewidth=0
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Correlation')

    # Plot confidence interval for correlation.
    if num_bootstrap_reps > 1:
        polygon_object = _confidence_interval_to_polygon(
            x_values=x_values, y_value_matrix=correlation_matrix,
            confidence_level=confidence_level
        )

        polygon_colour = matplotlib.colors.to_rgba(
            CORRELATION_COLOUR, POLYGON_OPACITY
        )
        patch_object = PolygonPatch(
            polygon_object, lw=0, ec=polygon_object, fc=polygon_colour
        )
        axes_object.add_patch(patch_object)

    y_min, y_max = axes_object.get_ylim()
    y_min = numpy.maximum(y_min, -1.)
    y_max = numpy.minimum(y_max, 1.)
    axes_object.set_ylim(y_min, y_max)

    axes_object.set_xlim(
        numpy.min(x_values) - 0.5, numpy.max(x_values) + 0.5
    )

    if plot_legend:
        axes_object.legend(
            legend_handles, legend_strings, loc='lower center',
            bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True,
            ncol=len(legend_handles)
        )

    return figure_object, axes_object


def _get_score_keys_one_field(evaluation_table_xarray, field_name,
                              height_m_agl=None):
    """Returns keys to extract scores for one field from evaluation table.

    :param evaluation_table_xarray: Single xarray table with evaluation scores.
    :param field_name: Name of field (target variable) for which to read scores.
    :param height_m_agl: Height (metres above ground level).  If field is not a
        vector (vertical profile), leave this argument alone.
    :return: score_keys: Keys in table (`evaluation_table_xarray`) with scores
        for desired field.
    :return: field_index: Index of desired field in numpy arrays stored in
        table.
    :return: height_index: Index of desired height in numpy arrays stored in
        table.  If `height_m_agl is None`, this is also None.
    """

    scalar_field_names = evaluation_table_xarray.coords[
        evaluation.SCALAR_FIELD_DIM
    ].values.tolist()

    try:
        field_index = scalar_field_names.index(field_name)
        score_keys = [
            evaluation.SCALAR_MAE_KEY, evaluation.SCALAR_MSE_KEY,
            evaluation.SCALAR_BIAS_KEY, evaluation.SCALAR_MAE_SKILL_KEY,
            evaluation.SCALAR_MSE_SKILL_KEY, evaluation.SCALAR_CORRELATION_KEY
        ]

        return score_keys, field_index, None
    except ValueError:
        pass

    try:
        aux_field_names = evaluation_table_xarray.coords[
            evaluation.AUX_TARGET_FIELD_DIM
        ].values.tolist()

        field_index = aux_field_names.index(field_name)
        score_keys = [
            evaluation.AUX_MAE_KEY, evaluation.AUX_MSE_KEY,
            evaluation.AUX_BIAS_KEY, evaluation.AUX_MAE_SKILL_KEY,
            evaluation.AUX_MSE_SKILL_KEY, evaluation.AUX_CORRELATION_KEY
        ]

        return score_keys, field_index, None
    except:
        pass

    vector_field_names = evaluation_table_xarray.coords[
        evaluation.VECTOR_FIELD_DIM
    ].values.tolist()

    field_index = vector_field_names.index(field_name)

    height_index = example_io._match_heights(
        heights_m_agl=
        evaluation_table_xarray.coords[evaluation.HEIGHT_DIM].values,
        desired_height_m_agl=height_m_agl
    )

    score_keys = [
        evaluation.VECTOR_MAE_KEY, evaluation.VECTOR_MSE_KEY,
        evaluation.VECTOR_BIAS_KEY, evaluation.VECTOR_MAE_SKILL_KEY,
        evaluation.VECTOR_MSE_SKILL_KEY, evaluation.VECTOR_CORRELATION_KEY
    ]

    return score_keys, field_index, height_index


def _read_scores_one_split_one_var(
        evaluation_tables_xarray, field_name, height_m_agl=None):
    """Reads scores for one time split and one field.

    T = number of time chunks

    :param evaluation_tables_xarray: length-T list of xarray tables with
        results.
    :param field_name: Name of field (target variable) for which to read scores.
    :param height_m_agl: Height (metres above ground level).  If field is not a
        vector (vertical profile), leave this argument alone.
    :return: score_dict: Dictionary with the following keys.
    score_dict['mae_matrix']: T-by-1 numpy array of MAE (mean absolute error)
        values.
    score_dict['rmse_matrix']: T-by-1 numpy array of RMSE (root mean squared
        error) values.
    score_dict['bias_matrix']: T-by-1 numpy array of biases.
    score_dict['mae_skill_score_matrix']: T-by-1 numpy array of MAE skill
        scores.
    score_dict['mse_skill_score_matrix']: T-by-1 numpy array of MSE skill
        scores.
    score_dict['correlation_matrix']: T-by-1 numpy array of Pearson
        correlations.
    """

    # TODO(thunderhoser): This code is disgusting and should be cleaned up.

    score_keys, field_index, height_index = _get_score_keys_one_field(
        evaluation_table_xarray=evaluation_tables_xarray[0],
        field_name=field_name, height_m_agl=height_m_agl
    )

    num_time_chunks = len(evaluation_tables_xarray)

    mae_matrix = numpy.full((num_time_chunks, 1), numpy.nan)
    rmse_matrix = numpy.full((num_time_chunks, 1), numpy.nan)
    bias_matrix = numpy.full((num_time_chunks, 1), numpy.nan)
    mae_skill_score_matrix = numpy.full((num_time_chunks, 1), numpy.nan)
    mse_skill_score_matrix = numpy.full((num_time_chunks, 1), numpy.nan)
    correlation_matrix = numpy.full((num_time_chunks, 1), numpy.nan)

    for i in range(num_time_chunks):
        this_table = evaluation_tables_xarray[i]

        if height_index is None:
            mae_matrix[i, 0] = this_table[score_keys[0]].values[field_index]
            rmse_matrix[i, 0] = this_table[score_keys[1]].values[field_index]
            bias_matrix[i, 0] = this_table[score_keys[2]].values[field_index]
            mae_skill_score_matrix[i, 0] = (
                this_table[score_keys[3]].values[field_index]
            )
            mse_skill_score_matrix[i, 0] = (
                this_table[score_keys[4]].values[field_index]
            )
            correlation_matrix[i, 0] = (
                this_table[score_keys[5]].values[field_index]
            )
        else:
            mae_matrix[i, 0] = (
                this_table[score_keys[0]].values[height_index, field_index]
            )
            rmse_matrix[i, 0] = (
                this_table[score_keys[1]].values[height_index, field_index]
            )
            bias_matrix[i, 0] = (
                this_table[score_keys[2]].values[height_index, field_index]
            )
            mae_skill_score_matrix[i, 0] = (
                this_table[score_keys[3]].values[height_index, field_index]
            )
            mse_skill_score_matrix[i, 0] = (
                this_table[score_keys[4]].values[height_index, field_index]
            )
            correlation_matrix[i, 0] = (
                this_table[score_keys[5]].values[height_index, field_index]
            )

    rmse_matrix = numpy.sqrt(rmse_matrix)

    return {
        'mae_matrix': mae_matrix,
        'rmse_matrix': rmse_matrix,
        'bias_matrix': bias_matrix,
        'mae_skill_score_matrix': mae_skill_score_matrix,
        'mse_skill_score_matrix': mse_skill_score_matrix,
        'correlation_matrix': correlation_matrix
    }


def _plot_all_scores_by_month(evaluation_dir_name, confidence_level,
                              output_dir_name):
    """Plots all evaluation scores by month.

    :param evaluation_dir_name: See documentation at top of file.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    """

    months = numpy.linspace(1, 12, num=12, dtype=int)
    evaluation_file_names = [
        evaluation.find_file(directory_name=evaluation_dir_name, month=k)
        for k in months
    ]
    evaluation_tables_xarray = _read_files_one_split(evaluation_file_names)

    scalar_field_names = evaluation_tables_xarray[0].coords[
        evaluation.SCALAR_FIELD_DIM
    ].values.tolist()

    vector_field_names = evaluation_tables_xarray[0].coords[
        evaluation.VECTOR_FIELD_DIM
    ].values.tolist()

    try:
        aux_field_names = evaluation_tables_xarray[0].coords[
            evaluation.AUX_TARGET_FIELD_DIM
        ].values.tolist()
    except KeyError:
        aux_field_names = []

    all_field_names = scalar_field_names + vector_field_names + aux_field_names
    heights_m_agl = (
        evaluation_tables_xarray[0].coords[evaluation.HEIGHT_DIM].values
    )

    for this_field_name in all_field_names:
        if this_field_name in vector_field_names:
            these_heights_m_agl = heights_m_agl + 0
        else:
            these_heights_m_agl = None

        for this_height_m_agl in these_heights_m_agl:
            this_score_dict = _read_scores_one_split_one_var(
                evaluation_tables_xarray=evaluation_tables_xarray,
                field_name=this_field_name, height_m_agl=this_height_m_agl
            )

            this_figure_object, this_axes_object = _plot_scores_with_units(
                mae_matrix=this_score_dict['mae_matrix'],
                rmse_matrix=this_score_dict['rmse_matrix'],
                bias_matrix=this_score_dict['bias_matrix'],
                plot_legend=True
            )
            this_axes_object.set_xticks(MONTH_INDICES)
            this_axes_object.set_xticklabels(MONTH_STRINGS, rotation=90.)

            if this_height_m_agl is None:
                this_file_name = '{0:s}/{1:s}_scores_with_units.jpg'.format(
                    output_dir_name, this_field_name.replace('_', '-')
                )
            else:
                this_file_name = (
                    '{0:s}/{1:s}_{2:05d}metres_scores_with_units.jpg'
                ).format(
                    output_dir_name, this_field_name.replace('_', '-'),
                    int(numpy.round(this_height_m_agl))
                )

            print('Saving figure to: "{0:s}"...'.format(this_file_name))

            this_figure_object.savefig(
                this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(this_figure_object)

            this_figure_object, this_axes_object = _plot_unitless_scores(
                mae_skill_score_matrix=
                this_score_dict['mae_skill_score_matrix'],
                mse_skill_score_matrix=
                this_score_dict['mse_skill_score_matrix'],
                correlation_matrix=this_score_dict['correlation_matrix'],
                plot_legend=True
            )
            this_axes_object.set_xticks(MONTH_INDICES)
            this_axes_object.set_xticklabels(MONTH_STRINGS, rotation=90.)

            this_file_name = this_file_name.replace(
                'scores_with_units', 'scores_without_units'
            )
            print('Saving figure to: "{0:s}"...'.format(this_file_name))

            this_figure_object.savefig(
                this_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(this_figure_object)


def _run(evaluation_dir_name, num_zenith_angle_bins, top_output_dir_name):
    """Plots evaluation scores by time of day and year.

    This is effectively the main method.

    :param evaluation_dir_name: See documentation at top of file.
    :param num_zenith_angle_bins: Same.
    :param top_output_dir_name: Same.
    """

    error_checking.assert_is_geq(num_zenith_angle_bins, 2)

    # bin_indices = numpy.linspace(
    #     0, num_zenith_angle_bins - 1, num=num_zenith_angle_bins, dtype=int
    # )
    # file_name_by_angle_bin = [
    #     evaluation.find_file(
    #         directory_name=evaluation_dir_name, zenith_angle_bin=k
    #     ) for k in bin_indices
    # ]

    month_output_dir_name = '{0:s}/by_month'.format(top_output_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=month_output_dir_name
    )

    _plot_all_scores_by_month(
        evaluation_dir_name=evaluation_dir_name, confidence_level=0.95,
        output_dir_name=month_output_dir_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        num_zenith_angle_bins=getattr(
            INPUT_ARG_OBJECT, NUM_ANGLE_BINS_ARG_NAME
        ),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
