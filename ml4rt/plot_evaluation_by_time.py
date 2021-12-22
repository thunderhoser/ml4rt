"""Plots evaluation scores by time of day and year."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
import matplotlib.patches
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import prediction_io
import example_utils
import evaluation
import plot_evaluation
import split_predictions_by_time as split_predictions

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MONTH_STRING = 'month'
ZENITH_ANGLE_STRING = 'zenith_angle'
ALBEDO_STRING = 'albedo'
SURFACE_DOWN_FLUX_STRING = 'surface_down_flux'
AEROSOL_OPTICAL_DEPTH_STRING = 'aerosol_optical_depth'

VALID_SPLIT_TYPE_STRINGS = [
    MONTH_STRING, ZENITH_ANGLE_STRING, ALBEDO_STRING, SURFACE_DOWN_FLUX_STRING,
    AEROSOL_OPTICAL_DEPTH_STRING
]

MIN_ZENITH_ANGLE_RAD = 0.
MAX_ZENITH_ANGLE_RAD = numpy.pi / 2
RADIANS_TO_DEGREES = 180. / numpy.pi
MAX_SURFACE_DOWN_FLUX_W_M02 = split_predictions.MAX_SURFACE_DOWN_FLUX_W_M02
MAX_AEROSOL_OPTICAL_DEPTH = split_predictions.MAX_AEROSOL_OPTICAL_DEPTH

EXAMPLE_DIM = 'example'
SCALAR_TARGET_KEY = 'scalar_target_value'
VECTOR_TARGET_KEY = 'vector_target_value'
AUX_TARGET_KEY = 'aux_target_value'
NUM_EXAMPLES_KEY = 'num_examples'

TARGET_NAME_TO_VERBOSE = plot_evaluation.TARGET_NAME_TO_VERBOSE
TARGET_NAME_TO_UNITS = plot_evaluation.TARGET_NAME_TO_UNITS

MARKER_TYPE = 'o'
MARKER_SIZE = 16
LINE_WIDTH = 4
REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
REFERENCE_LINE_WIDTH = 3

MAE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
RMSE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
BIAS_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255

MAE_SKILL_COLOUR = MAE_COLOUR
MSE_SKILL_COLOUR = RMSE_COLOUR
CORRELATION_COLOUR = BIAS_COLOUR
KGE_COLOUR = numpy.full(3, 0.)
POLYGON_OPACITY = 0.5

HISTOGRAM_EDGE_WIDTH = 1.5
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)
HISTOGRAM_FACE_COLOUR = numpy.full(3, 152. / 255)

MONTH_STRINGS = [
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',
    'Dec'
]

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

FONT_SIZE = 36
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

INPUT_DIR_ARG_NAME = 'input_evaluation_dir_name'
NUM_ANGLE_BINS_ARG_NAME = 'num_zenith_angle_bins'
NUM_ALBEDO_BINS_ARG_NAME = 'num_albedo_bins'
NUM_DOWN_FLUX_BINS_ARG_NAME = 'num_surface_down_flux_bins'
NUM_AOD_BINS_ARG_NAME = 'num_aod_bins'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of directory with evaluation files (one for each zenith-angle bin, '
    'one for each month).  Files will be found by `evaluation.find_file` and '
    'read by `evaluation.read_file`.'
)
NUM_ANGLE_BINS_HELP_STRING = 'Number of bins for zenith angle.'
NUM_ALBEDO_BINS_HELP_STRING = 'Number of bins for albedo.'
NUM_DOWN_FLUX_BINS_HELP_STRING = (
    'Number of bins for surface downwelling flux.  If you do not want to split '
    'by surface downwelling flux, make this argument <= 0.'
)
NUM_AOD_BINS_HELP_STRING = (
    'Number of bins for aerosol optical depth (AOD).  If you do not want to '
    'split by AOD, make this argument <= 0.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level (from 0...1).  If you do not want to plot confidence '
    'intervals, leave this alone.'
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
    '--' + NUM_ANGLE_BINS_ARG_NAME, type=int, required=False, default=9,
    help=NUM_ANGLE_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ALBEDO_BINS_ARG_NAME, type=int, required=False, default=20,
    help=NUM_ALBEDO_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_DOWN_FLUX_BINS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_DOWN_FLUX_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_AOD_BINS_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_AOD_BINS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=-1,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _augment_eval_table(result_table_xarray):
    """Augments evaluation table.

    Specifically, adds number of examples and actual values for each target
    variable.

    :param result_table_xarray: Table returned by `evaluation.read_file`.
    :return: result_table_xarray: Same but with number of examples and actual
        values for each target variable.
    """

    prediction_file_name = (
        result_table_xarray.attrs[evaluation.PREDICTION_FILE_KEY]
    )

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    num_examples = len(prediction_dict[prediction_io.EXAMPLE_IDS_KEY])
    result_table_xarray.attrs[NUM_EXAMPLES_KEY] = num_examples

    scalar_target_matrix = prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    these_dim = (EXAMPLE_DIM, evaluation.SCALAR_FIELD_DIM)
    result_table_xarray.update({
        SCALAR_TARGET_KEY: (these_dim, scalar_target_matrix)
    })

    vector_target_matrix = prediction_dict[prediction_io.VECTOR_TARGETS_KEY]
    these_dim = (
        EXAMPLE_DIM, evaluation.HEIGHT_DIM, evaluation.VECTOR_FIELD_DIM
    )
    result_table_xarray.update({
        VECTOR_TARGET_KEY: (these_dim, vector_target_matrix)
    })

    try:
        _ = result_table_xarray.coords[evaluation.AUX_TARGET_FIELD_DIM].values
    except KeyError:
        return result_table_xarray

    example_dict = {
        example_utils.SCALAR_TARGET_NAMES_KEY:
            numpy.array(
                result_table_xarray.coords[evaluation.SCALAR_FIELD_DIM].values
            ).tolist(),
        example_utils.VECTOR_TARGET_NAMES_KEY:
            numpy.array(
                result_table_xarray.coords[evaluation.VECTOR_FIELD_DIM].values
            ).tolist(),
        example_utils.HEIGHTS_KEY:
            numpy.round(
                result_table_xarray.coords[evaluation.HEIGHT_DIM].values
            ).astype(int)
    }

    aux_target_matrix = evaluation.get_aux_fields(
        prediction_dict=prediction_dict, example_dict=example_dict
    )[evaluation.AUX_TARGET_VALS_KEY]

    these_dim = (EXAMPLE_DIM, evaluation.AUX_TARGET_FIELD_DIM)
    result_table_xarray.update({
        AUX_TARGET_KEY: (these_dim, aux_target_matrix)
    })

    return result_table_xarray


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

        this_table = _augment_eval_table(
            evaluation.read_file(evaluation_file_names[i])
        )
        evaluation_tables_xarray.append(this_table)

    return evaluation_tables_xarray


def _plot_scores_with_units(
        mae_matrix, rmse_matrix, bias_matrix, target_matrices, plot_legend,
        confidence_level):
    """Plots scores with physical units, for one time split and one field.

    B = number of bootstrap replicates
    T = number of time chunks

    :param mae_matrix: T-by-B numpy array of MAE (mean absolute error) values.
    :param rmse_matrix: T-by-B numpy array of RMSE (root mean squared error)
        values.
    :param bias_matrix: T-by-B numpy array of biases.
    :param target_matrices: length-T list of numpy arrays.  The [i]th array has
        length E_i, where E_i = number of examples for [i]th time chunk.
    :param plot_legend: Boolean flag.  If True, will plot legend above figure.
    :param confidence_level: See documentation at top of file.
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

    # Plot boxplots.
    boxplot_style_dict = {
        'color': 'k',
        'linewidth': 2
    }

    legend_handles = []
    legend_strings = []

    for i in range(num_time_chunks):
        this_dict = axes_object.boxplot(
            target_matrices[i], widths=1., notch=False, sym='', whis=(5, 95),
            medianprops=boxplot_style_dict, boxprops=boxplot_style_dict,
            whiskerprops=boxplot_style_dict, capprops=boxplot_style_dict,
            positions=x_values[[i]]
        )

        if i != 0:
            continue

        legend_handles.append(this_dict['boxes'][0])
        legend_strings.append('Boxplot of\nactual values')

    # Plot reference line.
    axes_object.plot(
        axes_object.get_xlim(), numpy.full(2, 0.), color=REFERENCE_LINE_COLOUR,
        linewidth=REFERENCE_LINE_WIDTH, linestyle='dashed'
    )

    # Plot mean MAE.
    this_handle = axes_object.plot(
        x_values, numpy.mean(mae_matrix, axis=1), color=MAE_COLOUR,
        linewidth=LINE_WIDTH, marker=MARKER_TYPE, markersize=MARKER_SIZE,
        markerfacecolor=MAE_COLOUR, markeredgecolor=MAE_COLOUR,
        markeredgewidth=0
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('MAE')

    # Plot confidence interval for MAE.
    if num_bootstrap_reps > 1 and confidence_level is not None:
        polygon_coord_matrix = evaluation.confidence_interval_to_polygon(
            x_value_matrix=numpy.expand_dims(x_values, axis=-1),
            y_value_matrix=mae_matrix,
            confidence_level=confidence_level, same_order=True
        )

        if polygon_coord_matrix is not None:
            polygon_colour = matplotlib.colors.to_rgba(
                MAE_COLOUR, POLYGON_OPACITY
            )
            patch_object = matplotlib.patches.Polygon(
                polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
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
    if num_bootstrap_reps > 1 and confidence_level is not None:
        polygon_coord_matrix = evaluation.confidence_interval_to_polygon(
            x_value_matrix=numpy.expand_dims(x_values, axis=-1),
            y_value_matrix=rmse_matrix,
            confidence_level=confidence_level, same_order=True
        )

        if polygon_coord_matrix is not None:
            polygon_colour = matplotlib.colors.to_rgba(
                RMSE_COLOUR, POLYGON_OPACITY
            )
            patch_object = matplotlib.patches.Polygon(
                polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
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
    if num_bootstrap_reps > 1 and confidence_level is not None:
        polygon_coord_matrix = evaluation.confidence_interval_to_polygon(
            x_value_matrix=numpy.expand_dims(x_values, axis=-1),
            y_value_matrix=bias_matrix,
            confidence_level=confidence_level, same_order=True
        )

        if polygon_coord_matrix is not None:
            polygon_colour = matplotlib.colors.to_rgba(
                BIAS_COLOUR, POLYGON_OPACITY
            )
            patch_object = matplotlib.patches.Polygon(
                polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
            )
            axes_object.add_patch(patch_object)

    axes_object.set_xticks(x_values)
    axes_object.set_xlim(
        numpy.min(x_values) - 0.5, numpy.max(x_values) + 0.5
    )

    if plot_legend:
        axes_object.legend(
            legend_handles, legend_strings, loc='upper right',
            bbox_to_anchor=(1, 1), fancybox=True, shadow=False,
            facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
        )

    return figure_object, axes_object


def _plot_unitless_scores(
        mae_skill_score_matrix, mse_skill_score_matrix, correlation_matrix,
        kge_matrix, num_examples_array, plot_legend, confidence_level):
    """Plots scores without physical units, for one time split and one field.

    B = number of bootstrap replicates
    T = number of time chunks

    :param mae_skill_score_matrix: T-by-B numpy array of MAE (mean absolute
        error) skill scores.
    :param mse_skill_score_matrix: T-by-B numpy array of MSE (mean squared
        error) skill scores.
    :param correlation_matrix: T-by-B numpy array of correlations.
    :param kge_matrix: T-by-B numpy array of KGE values.
    :param num_examples_array: length-T numpy array with number of examples for
        each time chunk.
    :param plot_legend: See doc for `_plot_scores_with_units`.
    :param confidence_level: Same.
    :return: figure_object: Same.
    :return: axes_object: Same.
    """

    # Housekeeping.
    figure_object, main_axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    histogram_axes_object = main_axes_object.twinx()
    main_axes_object.set_zorder(histogram_axes_object.get_zorder() + 1)
    main_axes_object.patch.set_visible(False)

    num_time_chunks = mae_skill_score_matrix.shape[0]
    num_bootstrap_reps = mae_skill_score_matrix.shape[1]

    x_values = numpy.linspace(
        0, num_time_chunks - 1, num=num_time_chunks, dtype=float
    )

    # Plot mean MAE skill score.
    this_handle = main_axes_object.plot(
        x_values, numpy.mean(mae_skill_score_matrix, axis=1),
        color=MAE_SKILL_COLOUR, linewidth=LINE_WIDTH, marker=MARKER_TYPE,
        markersize=MARKER_SIZE, markerfacecolor=MAE_SKILL_COLOUR,
        markeredgecolor=MAE_SKILL_COLOUR, markeredgewidth=0
    )[0]

    legend_handles = [this_handle]
    legend_strings = ['MAE skill score']

    # Plot confidence interval for MAE skill score.
    if num_bootstrap_reps > 1 and confidence_level is not None:
        polygon_coord_matrix = evaluation.confidence_interval_to_polygon(
            x_value_matrix=numpy.expand_dims(x_values, axis=-1),
            y_value_matrix=mae_skill_score_matrix,
            confidence_level=confidence_level, same_order=True
        )

        if polygon_coord_matrix is not None:
            polygon_colour = matplotlib.colors.to_rgba(
                MAE_SKILL_COLOUR, POLYGON_OPACITY
            )
            patch_object = matplotlib.patches.Polygon(
                polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
            )
            main_axes_object.add_patch(patch_object)

    # Plot mean MSE skill score.
    this_handle = main_axes_object.plot(
        x_values, numpy.mean(mse_skill_score_matrix, axis=1),
        color=MSE_SKILL_COLOUR, linewidth=LINE_WIDTH, marker=MARKER_TYPE,
        markersize=MARKER_SIZE, markerfacecolor=MSE_SKILL_COLOUR,
        markeredgecolor=MSE_SKILL_COLOUR, markeredgewidth=0
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('MSE skill score')

    # Plot confidence interval for MSE skill score.
    if num_bootstrap_reps > 1 and confidence_level is not None:
        polygon_coord_matrix = evaluation.confidence_interval_to_polygon(
            x_value_matrix=numpy.expand_dims(x_values, axis=-1),
            y_value_matrix=mse_skill_score_matrix,
            confidence_level=confidence_level, same_order=True
        )

        if polygon_coord_matrix is not None:
            polygon_colour = matplotlib.colors.to_rgba(
                MSE_SKILL_COLOUR, POLYGON_OPACITY
            )
            patch_object = matplotlib.patches.Polygon(
                polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
            )
            main_axes_object.add_patch(patch_object)

    # Plot mean correlation.
    this_handle = main_axes_object.plot(
        x_values, numpy.mean(correlation_matrix, axis=1),
        color=CORRELATION_COLOUR, linewidth=LINE_WIDTH, marker=MARKER_TYPE,
        markersize=MARKER_SIZE, markerfacecolor=CORRELATION_COLOUR,
        markeredgecolor=CORRELATION_COLOUR, markeredgewidth=0
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('Correlation')

    # Plot confidence interval for correlation.
    if num_bootstrap_reps > 1 and confidence_level is not None:
        polygon_coord_matrix = evaluation.confidence_interval_to_polygon(
            x_value_matrix=numpy.expand_dims(x_values, axis=-1),
            y_value_matrix=correlation_matrix,
            confidence_level=confidence_level, same_order=True
        )

        if polygon_coord_matrix is not None:
            polygon_colour = matplotlib.colors.to_rgba(
                CORRELATION_COLOUR, POLYGON_OPACITY
            )
            patch_object = matplotlib.patches.Polygon(
                polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
            )
            main_axes_object.add_patch(patch_object)

    # Plot mean KGE.
    this_handle = main_axes_object.plot(
        x_values, numpy.mean(kge_matrix, axis=1),
        color=KGE_COLOUR, linewidth=LINE_WIDTH, marker=MARKER_TYPE,
        markersize=MARKER_SIZE, markerfacecolor=KGE_COLOUR,
        markeredgecolor=KGE_COLOUR, markeredgewidth=0
    )[0]

    legend_handles.append(this_handle)
    legend_strings.append('KGE')

    # Plot confidence interval for KGE.
    if num_bootstrap_reps > 1 and confidence_level is not None:
        polygon_coord_matrix = evaluation.confidence_interval_to_polygon(
            x_value_matrix=numpy.expand_dims(x_values, axis=-1),
            y_value_matrix=kge_matrix,
            confidence_level=confidence_level, same_order=True
        )

        if polygon_coord_matrix is not None:
            polygon_colour = matplotlib.colors.to_rgba(
                KGE_COLOUR, POLYGON_OPACITY
            )
            patch_object = matplotlib.patches.Polygon(
                polygon_coord_matrix, lw=0, ec=polygon_colour, fc=polygon_colour
            )
            main_axes_object.add_patch(patch_object)

    y_min, y_max = main_axes_object.get_ylim()
    y_min = numpy.maximum(y_min, -1.)
    y_max = numpy.minimum(y_max, 1.)
    main_axes_object.set_ylim(y_min, y_max)

    main_axes_object.set_xticks(x_values)
    main_axes_object.set_xlim(
        numpy.min(x_values) - 0.5, numpy.max(x_values) + 0.5
    )

    # Plot histogram of example counts.
    y_values = numpy.maximum(numpy.log10(num_examples_array), 0.)

    histogram_axes_object.bar(
        x=x_values, height=y_values, width=1., color=HISTOGRAM_FACE_COLOUR,
        edgecolor=HISTOGRAM_EDGE_COLOUR, linewidth=HISTOGRAM_EDGE_WIDTH
    )
    histogram_axes_object.set_ylabel('Number of examples')

    tick_values = histogram_axes_object.get_yticks()
    tick_strings = [
        '{0:d}'.format(int(numpy.round(10 ** v))) for v in tick_values
    ]
    histogram_axes_object.set_yticklabels(tick_strings)

    print('Number of examples by chunk: {0:s}'.format(
        str(num_examples_array)
    ))

    if plot_legend:
        main_axes_object.legend(
            legend_handles, legend_strings, loc='lower right',
            bbox_to_anchor=(1, 0), fancybox=True, shadow=False,
            facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
        )

    return figure_object, main_axes_object


def _check_split_type(split_type_string):
    """Error-checks split type.

    :param split_type_string: Split type.
    :raises: ValueError: if `split_type_string not in VALID_SPLIT_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(split_type_string)
    if split_type_string in VALID_SPLIT_TYPE_STRINGS:
        return

    error_string = (
        '\nValid split types (listed below) do not include "{0:s}":\n{1:s}'
    ).format(split_type_string, str(VALID_SPLIT_TYPE_STRINGS))

    raise ValueError(error_string)


def _plot_all_scores_one_split(
        evaluation_dir_name, confidence_level, output_dir_name,
        split_type_string, num_bins=None):
    """Plots all scores for one time split.

    :param evaluation_dir_name: See documentation at top of file.
    :param confidence_level: Same.
    :param output_dir_name: Same.
    :param split_type_string: Method used to split evaluation files (must be
        accepted by `_check_split_type`).
    :param num_bins:
        [used only if split type is zenith angle, albedo, surface downwelling
        flux, or aerosol optical depth]
        Number of bins.
    """

    _check_split_type(split_type_string)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if split_type_string == MONTH_STRING:
        months = numpy.linspace(1, 12, num=12, dtype=int)
        evaluation_file_names = [
            evaluation.find_file(directory_name=evaluation_dir_name, month=k)
            for k in months
        ]

        x_tick_label_strings = MONTH_STRINGS
        x_axis_label_string = ''

    elif split_type_string == ZENITH_ANGLE_STRING:
        bin_indices = numpy.linspace(0, num_bins - 1, num=num_bins, dtype=int)
        evaluation_file_names = [
            evaluation.find_file(
                directory_name=evaluation_dir_name, zenith_angle_bin=k
            ) for k in bin_indices
        ]

        bin_edge_angles_deg = RADIANS_TO_DEGREES * numpy.linspace(
            MIN_ZENITH_ANGLE_RAD, MAX_ZENITH_ANGLE_RAD,
            num=num_bins + 1, dtype=float
        )

        x_tick_label_strings = ['foo'] * num_bins

        for k in range(num_bins):
            x_tick_label_strings[k] = '[{0:.1f}, {1:.1f}'.format(
                bin_edge_angles_deg[k], bin_edge_angles_deg[k + 1]
            )

            if k == num_bins - 1:
                x_tick_label_strings[k] += ']'
            else:
                x_tick_label_strings[k] += ')'

            x_tick_label_strings[k] += r'$^{\circ}$'

        x_axis_label_string = 'Solar zenith angle'

    elif split_type_string == ALBEDO_STRING:
        bin_indices = numpy.linspace(0, num_bins - 1, num=num_bins, dtype=int)
        evaluation_file_names = [
            evaluation.find_file(
                directory_name=evaluation_dir_name, albedo_bin=k
            ) for k in bin_indices
        ]

        edge_albedos = numpy.linspace(0, 1, num=num_bins + 1, dtype=float)
        x_tick_label_strings = ['foo'] * num_bins

        for k in range(num_bins):
            x_tick_label_strings[k] = '[{0:.3f}, {1:.3f}'.format(
                edge_albedos[k], edge_albedos[k + 1]
            )

            if k == num_bins - 1:
                x_tick_label_strings[k] += ']'
            else:
                x_tick_label_strings[k] += ')'

        x_axis_label_string = 'Albedo'

    elif split_type_string == SURFACE_DOWN_FLUX_STRING:
        bin_indices = numpy.linspace(0, num_bins - 1, num=num_bins, dtype=int)
        evaluation_file_names = [
            evaluation.find_file(
                directory_name=evaluation_dir_name, surface_down_flux_bin=k
            ) for k in bin_indices
        ]

        edge_surface_down_fluxes_w_m02 = numpy.linspace(
            0, MAX_SURFACE_DOWN_FLUX_W_M02, num=num_bins + 1, dtype=float
        )
        edge_surface_down_fluxes_w_m02[-1] = numpy.inf
        x_tick_label_strings = ['foo'] * num_bins

        for k in range(num_bins):
            x_tick_label_strings[k] = '[{0:.3f}, {1:.3f}]'.format(
                edge_surface_down_fluxes_w_m02[k],
                edge_surface_down_fluxes_w_m02[k + 1]
            )

        x_axis_label_string = r'Surface downwelling flxu (W m$^{-2}$)'

    else:
        bin_indices = numpy.linspace(0, num_bins - 1, num=num_bins, dtype=int)
        evaluation_file_names = [
            evaluation.find_file(
                directory_name=evaluation_dir_name, aerosol_optical_depth_bin=k
            ) for k in bin_indices
        ]

        edge_aerosol_optical_depths = numpy.linspace(
            0, MAX_AEROSOL_OPTICAL_DEPTH, num=num_bins + 1, dtype=float
        )
        edge_aerosol_optical_depths[-1] = numpy.inf
        x_tick_label_strings = ['foo'] * num_bins

        for k in range(num_bins):
            x_tick_label_strings[k] = '[{0:.3f}, {1:.3f}]'.format(
                edge_aerosol_optical_depths[k],
                edge_aerosol_optical_depths[k + 1]
            )

        x_axis_label_string = 'Aerosol optical depth'

    evaluation_tables_xarray = _read_files_one_split(evaluation_file_names)
    print(SEPARATOR_STRING)

    scalar_field_names = numpy.array(
        evaluation_tables_xarray[0].coords[evaluation.SCALAR_FIELD_DIM].values
    )
    scalar_mae_matrix = numpy.stack([
        t[evaluation.SCALAR_MAE_KEY].values for t in evaluation_tables_xarray
    ], axis=0)
    scalar_rmse_matrix = numpy.sqrt(numpy.stack([
        t[evaluation.SCALAR_MSE_KEY].values for t in evaluation_tables_xarray
    ], axis=0))
    scalar_bias_matrix = numpy.stack([
        t[evaluation.SCALAR_BIAS_KEY].values for t in evaluation_tables_xarray
    ], axis=0)
    scalar_mae_skill_matrix = numpy.stack([
        t[evaluation.SCALAR_MAE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)
    scalar_mse_skill_matrix = numpy.stack([
        t[evaluation.SCALAR_MSE_SKILL_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)
    scalar_correlation_matrix = numpy.stack([
        t[evaluation.SCALAR_CORRELATION_KEY].values
        for t in evaluation_tables_xarray
    ], axis=0)
    scalar_kge_matrix = numpy.stack([
        t[evaluation.SCALAR_KGE_KEY].values for t in evaluation_tables_xarray
    ], axis=0)
    scalar_target_matrices = [
        t[SCALAR_TARGET_KEY].values for t in evaluation_tables_xarray
    ]
    num_examples_array = numpy.array([
        t.attrs[NUM_EXAMPLES_KEY] for t in evaluation_tables_xarray
    ], dtype=int)

    for k in range(len(scalar_field_names)):
        figure_object, axes_object = _plot_scores_with_units(
            mae_matrix=scalar_mae_matrix[:, k, :],
            rmse_matrix=scalar_rmse_matrix[:, k, :],
            bias_matrix=scalar_bias_matrix[:, k, :],
            target_matrices=[a[..., k] for a in scalar_target_matrices],
            plot_legend=True, confidence_level=confidence_level
        )
        axes_object.set_title('Scores for {0:s} ({1:s})'.format(
            TARGET_NAME_TO_VERBOSE[scalar_field_names[k]],
            TARGET_NAME_TO_UNITS[scalar_field_names[k]]
        ))
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
            mae_skill_score_matrix=scalar_mae_skill_matrix[:, k, :],
            mse_skill_score_matrix=scalar_mse_skill_matrix[:, k, :],
            correlation_matrix=scalar_correlation_matrix[:, k, :],
            kge_matrix=scalar_kge_matrix[:, k, :],
            num_examples_array=num_examples_array, plot_legend=True,
            confidence_level=confidence_level
        )
        axes_object.set_title('Scores for {0:s}'.format(
            TARGET_NAME_TO_VERBOSE[scalar_field_names[k]]
        ))
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

    try:
        aux_field_names = numpy.array(
            evaluation_tables_xarray[0].coords[
                evaluation.AUX_TARGET_FIELD_DIM
            ].values
        ).tolist()
        aux_mae_matrix = numpy.stack([
            t[evaluation.AUX_MAE_KEY].values for t in evaluation_tables_xarray
        ], axis=0)
        aux_rmse_matrix = numpy.sqrt(numpy.stack([
            t[evaluation.AUX_MSE_KEY].values for t in evaluation_tables_xarray
        ], axis=0))
        aux_bias_matrix = numpy.stack([
            t[evaluation.AUX_BIAS_KEY].values for t in evaluation_tables_xarray
        ], axis=0)
        aux_mae_skill_matrix = numpy.stack([
            t[evaluation.AUX_MAE_SKILL_KEY].values
            for t in evaluation_tables_xarray
        ], axis=0)
        aux_mse_skill_matrix = numpy.stack([
            t[evaluation.AUX_MSE_SKILL_KEY].values
            for t in evaluation_tables_xarray
        ], axis=0)
        aux_correlation_matrix = numpy.stack([
            t[evaluation.AUX_CORRELATION_KEY].values
            for t in evaluation_tables_xarray
        ], axis=0)
        aux_kge_matrix = numpy.stack([
            t[evaluation.AUX_KGE_KEY].values for t in evaluation_tables_xarray
        ], axis=0)
        aux_target_matrices = [
            t[AUX_TARGET_KEY].values for t in evaluation_tables_xarray
        ]
    except KeyError:
        aux_field_names = []

    for k in range(len(aux_field_names)):
        figure_object, axes_object = _plot_scores_with_units(
            mae_matrix=aux_mae_matrix[:, k, :],
            rmse_matrix=aux_rmse_matrix[:, k, :],
            bias_matrix=aux_bias_matrix[:, k, :],
            target_matrices=[a[..., k] for a in aux_target_matrices],
            plot_legend=True, confidence_level=confidence_level
        )
        axes_object.set_title('Scores for {0:s} ({1:s})'.format(
            TARGET_NAME_TO_VERBOSE[aux_field_names[k]],
            TARGET_NAME_TO_UNITS[aux_field_names[k]]
        ))
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
            mae_skill_score_matrix=aux_mae_skill_matrix[:, k, :],
            mse_skill_score_matrix=aux_mse_skill_matrix[:, k, :],
            correlation_matrix=aux_correlation_matrix[:, k, :],
            kge_matrix=aux_kge_matrix[:, k, :],
            num_examples_array=num_examples_array, plot_legend=True,
            confidence_level=confidence_level
        )
        axes_object.set_title('Scores for {0:s}'.format(
            TARGET_NAME_TO_VERBOSE[aux_field_names[k]]
        ))
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

    vector_field_names = numpy.array(
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
    vector_kge_matrix = numpy.stack([
        t[evaluation.VECTOR_KGE_KEY].values for t in evaluation_tables_xarray
    ], axis=0)
    vector_target_matrices = [
        t[VECTOR_TARGET_KEY].values for t in evaluation_tables_xarray
    ]

    for j in range(len(heights_m_agl)):
        for k in range(len(vector_field_names)):
            figure_object, axes_object = _plot_scores_with_units(
                mae_matrix=vector_mae_matrix[:, j, k, :],
                rmse_matrix=vector_rmse_matrix[:, j, k, :],
                bias_matrix=vector_bias_matrix[:, j, k, :],
                target_matrices=[a[..., j, k] for a in vector_target_matrices],
                plot_legend=True, confidence_level=confidence_level
            )

            title_string = 'Scores for {0:s} ({1:s}) at {2:d} m AGL'.format(
                TARGET_NAME_TO_VERBOSE[vector_field_names[k]],
                TARGET_NAME_TO_UNITS[vector_field_names[k]],
                heights_m_agl[j]
            )

            axes_object.set_title(title_string)
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
                mae_skill_score_matrix=vector_mae_skill_matrix[:, j, k, :],
                mse_skill_score_matrix=vector_mse_skill_matrix[:, j, k, :],
                correlation_matrix=vector_correlation_matrix[:, j, k, :],
                kge_matrix=vector_kge_matrix[:, j, k, :],
                num_examples_array=num_examples_array, plot_legend=True,
                confidence_level=confidence_level
            )
            axes_object.set_title('Scores for {0:s} at {1:d} m AGL'.format(
                TARGET_NAME_TO_VERBOSE[vector_field_names[k]], heights_m_agl[j]
            ))
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


def _run(evaluation_dir_name, num_zenith_angle_bins, num_albedo_bins,
         num_surface_down_flux_bins, num_aod_bins, confidence_level,
         top_output_dir_name):
    """Plots evaluation scores by time of day and year.

    This is effectively the main method.

    :param evaluation_dir_name: See documentation at top of file.
    :param num_zenith_angle_bins: Same.
    :param num_albedo_bins: Same.
    :param num_surface_down_flux_bins: Same.
    :param num_aod_bins: Same.
    :param confidence_level: Same.
    :param top_output_dir_name: Same.
    """

    error_checking.assert_is_geq(num_zenith_angle_bins, 2)

    _plot_all_scores_one_split(
        evaluation_dir_name=evaluation_dir_name,
        output_dir_name='{0:s}/by_month'.format(top_output_dir_name),
        split_type_string=MONTH_STRING, confidence_level=confidence_level
    )
    print(SEPARATOR_STRING)

    _plot_all_scores_one_split(
        evaluation_dir_name=evaluation_dir_name,
        output_dir_name='{0:s}/by_zenith_angle'.format(top_output_dir_name),
        split_type_string=ZENITH_ANGLE_STRING, num_bins=num_zenith_angle_bins,
        confidence_level=confidence_level
    )
    print(SEPARATOR_STRING)

    _plot_all_scores_one_split(
        evaluation_dir_name=evaluation_dir_name,
        output_dir_name='{0:s}/by_albedo'.format(top_output_dir_name),
        split_type_string=ALBEDO_STRING, num_bins=num_albedo_bins,
        confidence_level=confidence_level
    )

    if num_surface_down_flux_bins > 0:
        print(SEPARATOR_STRING)
        _plot_all_scores_one_split(
            evaluation_dir_name=evaluation_dir_name,
            output_dir_name='{0:s}/by_surface_down_flux'.format(
                top_output_dir_name
            ),
            split_type_string=SURFACE_DOWN_FLUX_STRING,
            num_bins=num_surface_down_flux_bins,
            confidence_level=confidence_level
        )

    if num_aod_bins > 0:
        print(SEPARATOR_STRING)
        _plot_all_scores_one_split(
            evaluation_dir_name=evaluation_dir_name,
            output_dir_name='{0:s}/by_aerosol_optical_depth'.format(
                top_output_dir_name
            ),
            split_type_string=AEROSOL_OPTICAL_DEPTH_STRING,
            num_bins=num_aod_bins, confidence_level=confidence_level
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        evaluation_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        num_zenith_angle_bins=getattr(
            INPUT_ARG_OBJECT, NUM_ANGLE_BINS_ARG_NAME
        ),
        num_albedo_bins=getattr(INPUT_ARG_OBJECT, NUM_ALBEDO_BINS_ARG_NAME),
        num_surface_down_flux_bins=getattr(
            INPUT_ARG_OBJECT, NUM_DOWN_FLUX_BINS_ARG_NAME
        ),
        num_aod_bins=getattr(INPUT_ARG_OBJECT, NUM_AOD_BINS_ARG_NAME),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        top_output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
