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
from ml4rt.utils import evaluation

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MIN_ZENITH_ANGLE_RAD = 0.
MAX_ZENITH_ANGLE_RAD = numpy.pi / 2
RADIANS_TO_DEGREES = 180. / numpy.pi

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
    """Plots scores with physical units, for one time split and one field.

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
    """Plots scores without physical units, for one time split and one field.

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


def _plot_all_scores_one_split(evaluation_dir_name, output_dir_name, by_month,
                               num_zenith_angle_bins=None):
    """Plots all scores for one time split.

    :param evaluation_dir_name: See documentation at top of file.
    :param output_dir_name: Same.
    :param by_month: Boolean flag.  If True (False), will plot scores by month
        (solar zenith angle).
    :param num_zenith_angle_bins: See documentation at top of file.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if by_month:
        months = numpy.linspace(1, 12, num=12, dtype=int)
        evaluation_file_names = [
            evaluation.find_file(directory_name=evaluation_dir_name, month=k)
            for k in months
        ]

        x_tick_label_strings = MONTH_STRINGS
        x_axis_label_string = ''
    else:
        bin_indices = numpy.linspace(
            0, num_zenith_angle_bins - 1, num=num_zenith_angle_bins, dtype=int
        )
        evaluation_file_names = [
            evaluation.find_file(
                directory_name=evaluation_dir_name, zenith_angle_bin=k
            ) for k in bin_indices
        ]

        bin_edge_angles_deg = RADIANS_TO_DEGREES * numpy.linspace(
            MIN_ZENITH_ANGLE_RAD, MAX_ZENITH_ANGLE_RAD,
            num=num_zenith_angle_bins + 1, dtype=float
        )

        x_tick_label_strings = ['foo'] * num_zenith_angle_bins

        for k in range(num_zenith_angle_bins):
            x_tick_label_strings[k] = '[{0:s}, {1:s}'.format(
                bin_edge_angles_deg[k], bin_edge_angles_deg[k + 1]
            )

            if k == num_zenith_angle_bins - 1:
                x_tick_label_strings[k] += ']'
            else:
                x_tick_label_strings[k] += ')'

            x_tick_label_strings[k] += r'$^{\circ}$'

        x_axis_label_string = 'Solar zenith angle'

    evaluation_tables_xarray = _read_files_one_split(evaluation_file_names)
    print(SEPARATOR_STRING)

    scalar_field_names = (
        evaluation_tables_xarray[0].coords[evaluation.SCALAR_FIELD_DIM].values
    )
    scalar_mae_matrix = numpy.vstack([
        t[evaluation.SCALAR_MAE_KEY].values for t in evaluation_tables_xarray
    ])
    scalar_rmse_matrix = numpy.sqrt(numpy.vstack([
        t[evaluation.SCALAR_MSE_KEY].values for t in evaluation_tables_xarray
    ]))
    scalar_bias_matrix = numpy.vstack([
        t[evaluation.SCALAR_BIAS_KEY].values for t in evaluation_tables_xarray
    ])
    scalar_mae_skill_matrix = numpy.vstack([
        t[evaluation.SCALAR_MAE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ])
    scalar_mse_skill_matrix = numpy.vstack([
        t[evaluation.SCALAR_MSE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ])
    scalar_correlation_matrix = numpy.vstack([
        t[evaluation.SCALAR_CORRELATION_KEY].values
        for t in evaluation_tables_xarray
    ])

    for k in range(len(scalar_field_names)):
        figure_object, axes_object = _plot_scores_with_units(
            mae_matrix=scalar_mae_matrix[:, k],
            rmse_matrix=scalar_rmse_matrix[:, k],
            bias_matrix=scalar_bias_matrix[:, k],
            plot_legend=True
        )
        axes_object.set_xticklabels(x_tick_label_strings, rotation=90.)
        axes_object.set_xlabel(x_axis_label_string)

        figure_file_name = '{0:s}/{1:s}_scores_with_units.jpg'.format(
            output_dir_name, scalar_field_names[k].replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)

        figure_object, axes_object = _plot_unitless_scores(
            mae_skill_score_matrix=scalar_mae_skill_matrix[:, k],
            mse_skill_score_matrix=scalar_mse_skill_matrix[:, k],
            correlation_matrix=scalar_correlation_matrix[:, k],
            plot_legend=True
        )
        axes_object.set_xticklabels(x_tick_label_strings, rotation=90.)
        axes_object.set_xlabel(x_axis_label_string)

        figure_file_name = '{0:s}/{1:s}_scores_without_units.jpg'.format(
            output_dir_name, scalar_field_names[k].replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)

    print(SEPARATOR_STRING)

    aux_field_names = (
        evaluation_tables_xarray[0].coords[
            evaluation.AUX_TARGET_FIELD_DIM
        ].values
    )
    aux_mae_matrix = numpy.vstack([
        t[evaluation.AUX_MAE_KEY].values for t in evaluation_tables_xarray
    ])
    aux_rmse_matrix = numpy.sqrt(numpy.vstack([
        t[evaluation.AUX_MSE_KEY].values for t in evaluation_tables_xarray
    ]))
    aux_bias_matrix = numpy.vstack([
        t[evaluation.AUX_BIAS_KEY].values for t in evaluation_tables_xarray
    ])
    aux_mae_skill_matrix = numpy.vstack([
        t[evaluation.AUX_MAE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ])
    aux_mse_skill_matrix = numpy.vstack([
        t[evaluation.AUX_MSE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ])
    aux_correlation_matrix = numpy.vstack([
        t[evaluation.AUX_CORRELATION_KEY].values
        for t in evaluation_tables_xarray
    ])

    for k in range(len(aux_field_names)):
        figure_object, axes_object = _plot_scores_with_units(
            mae_matrix=aux_mae_matrix[:, k],
            rmse_matrix=aux_rmse_matrix[:, k],
            bias_matrix=aux_bias_matrix[:, k],
            plot_legend=True
        )
        axes_object.set_xticklabels(x_tick_label_strings, rotation=90.)
        axes_object.set_xlabel(x_axis_label_string)

        figure_file_name = '{0:s}/{1:s}_scores_with_units.jpg'.format(
            output_dir_name, aux_field_names[k].replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)

        figure_object, axes_object = _plot_unitless_scores(
            mae_skill_score_matrix=aux_mae_skill_matrix[:, k],
            mse_skill_score_matrix=aux_mse_skill_matrix[:, k],
            correlation_matrix=aux_correlation_matrix[:, k],
            plot_legend=True
        )
        axes_object.set_xticklabels(x_tick_label_strings, rotation=90.)
        axes_object.set_xlabel(x_axis_label_string)

        figure_file_name = '{0:s}/{1:s}_scores_without_units.jpg'.format(
            output_dir_name, aux_field_names[k].replace('_', '-')
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)

    print(SEPARATOR_STRING)

    vector_field_names = (
        evaluation_tables_xarray[0].coords[evaluation.VECTOR_FIELD_DIM].values
    )
    heights_m_agl = numpy.round(
        evaluation_tables_xarray[0].coords[evaluation.HEIGHT_DIM].values
    ).astype(int)

    vector_mae_matrix = numpy.stack([
        t[evaluation.VECTOR_MAE_KEY].values for t in evaluation_tables_xarray
    ], axis=0)
    vector_rmse_matrix = numpy.sqrt(numpy.stack([
        t[evaluation.VECTOR_MSE_KEY].values for t in evaluation_tables_xarray
    ], axis=0))
    vector_bias_matrix = numpy.stack([
        t[evaluation.VECTOR_BIAS_KEY].values for t in evaluation_tables_xarray
    ], axis=0)
    vector_mae_skill_matrix = numpy.stack([
        t[evaluation.VECTOR_MAE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)
    vector_mse_skill_matrix = numpy.stack([
        t[evaluation.VECTOR_MSE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)
    vector_correlation_matrix = numpy.stack([
        t[evaluation.VECTOR_CORRELATION_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)

    for j in range(len(heights_m_agl)):
        for k in range(len(vector_field_names)):
            figure_object, axes_object = _plot_scores_with_units(
                mae_matrix=vector_mae_matrix[:, j, k],
                rmse_matrix=vector_rmse_matrix[:, j, k],
                bias_matrix=vector_bias_matrix[:, j, k],
                plot_legend=True
            )
            axes_object.set_xticklabels(x_tick_label_strings, rotation=90.)
            axes_object.set_xlabel(x_axis_label_string)

            figure_file_name = (
                '{0:s}/{1:s}_{2:05d}metres_scores_with_units.jpg'
            ).format(
                output_dir_name, vector_field_names[k].replace('_', '-'),
                heights_m_agl[j]
            )

            print('Saving figure to: "{0:s}"...'.format(figure_file_name))
            figure_object.savefig(
                figure_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(figure_object)

            figure_object, axes_object = _plot_unitless_scores(
                mae_skill_score_matrix=vector_mae_skill_matrix[:, j, k],
                mse_skill_score_matrix=vector_mse_skill_matrix[:, j, k],
                correlation_matrix=vector_correlation_matrix[:, j, k],
                plot_legend=True
            )
            axes_object.set_xticklabels(x_tick_label_strings, rotation=90.)
            axes_object.set_xlabel(x_axis_label_string)

            figure_file_name = (
                '{0:s}/{1:s}_{2:05d}metres_scores_without_units.jpg'
            ).format(
                output_dir_name, vector_field_names[k].replace('_', '-'),
                heights_m_agl[j]
            )

            print('Saving figure to: "{0:s}"...'.format(figure_file_name))
            figure_object.savefig(
                figure_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
                bbox_inches='tight'
            )
            pyplot.close(figure_object)


def _run(evaluation_dir_name, num_zenith_angle_bins, top_output_dir_name):
    """Plots evaluation scores by time of day and year.

    This is effectively the main method.

    :param evaluation_dir_name: See documentation at top of file.
    :param num_zenith_angle_bins: Same.
    :param top_output_dir_name: Same.
    """

    error_checking.assert_is_geq(num_zenith_angle_bins, 2)

    _plot_all_scores_one_split(
        evaluation_dir_name=evaluation_dir_name,
        output_dir_name='{0:s}/by_month'.format(top_output_dir_name),
        by_month=True
    )
    print(SEPARATOR_STRING)

    _plot_all_scores_one_split(
        evaluation_dir_name=evaluation_dir_name,
        output_dir_name='{0:s}/by_zenith_angle'.format(top_output_dir_name),
        by_month=False, num_zenith_angle_bins=num_zenith_angle_bins
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
