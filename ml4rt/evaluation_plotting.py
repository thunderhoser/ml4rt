"""Plotting methods for model evaluation."""

import os
import sys
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

import error_checking
import plotting_utils
import taylor_diagram
import profile_plotting

# TODO(thunderhoser): Incorporate confidence intervals into this module, instead
# of scripts.

METRES_TO_KM = 0.001

MSE_NAME = 'mean_squared_error'
MSE_SKILL_SCORE_NAME = 'mse_skill_score'
MAE_NAME = 'mean_absolute_error'
MAE_SKILL_SCORE_NAME = 'mae_skill_score'
BIAS_NAME = 'bias'
CORRELATION_NAME = 'correlation'
KGE_NAME = 'kge'
VALID_SCORE_NAMES = [
    MSE_NAME, MSE_SKILL_SCORE_NAME, MAE_NAME, MAE_SKILL_SCORE_NAME,
    BIAS_NAME, CORRELATION_NAME, KGE_NAME
]

RELIABILITY_LINE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
RELIABILITY_LINE_WIDTH = 3.

REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
REFERENCE_LINE_WIDTH = 2.

CLIMO_LINE_COLOUR = numpy.full(3, 152. / 255)
CLIMO_LINE_WIDTH = 2.

ZERO_SKILL_LINE_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
ZERO_SKILL_LINE_WIDTH = 2.
POSITIVE_SKILL_AREA_OPACITY = 0.2

HISTOGRAM_FACE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)
HISTOGRAM_EDGE_WIDTH = 2.
HISTOGRAM_FONT_SIZE = 24

TAYLOR_TARGET_MARKER_TYPE = '*'
TAYLOR_TARGET_MARKER_SIZE = 24
TAYLOR_PREDICTION_MARKER_TYPE = 'o'
TAYLOR_PREDICTION_MARKER_SIZE = 20

DEFAULT_HEIGHT_CMAP_OBJECT = pyplot.get_cmap('viridis')

FONT_SIZE = 36
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _check_score_name(score_name):
    """Error-checks name of score.

    :param score_name: Name of score.
    :raises: ValueError: if `score_name not in VALID_SCORE_NAMES`.
    """

    error_checking.assert_is_string(score_name)
    if score_name in VALID_SCORE_NAMES:
        return

    error_string = (
        '\nField "{0:s}" is not a valid score.  Valid options listed below:'
        '\n{1:s}'
    ).format(score_name, str(VALID_SCORE_NAMES))

    raise ValueError(error_string)


def _plot_reliability_curve(
        axes_object, mean_predictions, mean_observations, min_value_to_plot,
        max_value_to_plot, line_colour=RELIABILITY_LINE_COLOUR,
        line_style='solid', line_width=RELIABILITY_LINE_WIDTH):
    """Plots reliability curve.

    B = number of bins

    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param mean_predictions: length-B numpy array of mean predicted values.
    :param mean_observations: length-B numpy array of mean observed values.
    :param min_value_to_plot: See doc for `plot_attributes_diagram`.
    :param max_value_to_plot: Same.
    :param line_colour: Line colour (in any format accepted by matplotlib).
    :param line_style: Line style (in any format accepted by matplotlib).
    :param line_width: Line width (in any format accepted by matplotlib).
    :return: main_line_handle: Handle for main line (reliability curve).
    """

    perfect_x_coords = numpy.array([min_value_to_plot, max_value_to_plot])
    perfect_y_coords = numpy.array([min_value_to_plot, max_value_to_plot])

    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=REFERENCE_LINE_COLOUR,
        linestyle='dashed', linewidth=REFERENCE_LINE_WIDTH
    )

    nan_flags = numpy.logical_or(
        numpy.isnan(mean_predictions), numpy.isnan(mean_observations)
    )

    if numpy.all(nan_flags):
        main_line_handle = None
    else:
        real_indices = numpy.where(numpy.invert(nan_flags))[0]

        main_line_handle = axes_object.plot(
            mean_predictions[real_indices], mean_observations[real_indices],
            color=line_colour, linestyle=line_style, linewidth=line_width
        )[0]

    axes_object.set_xlabel('Prediction')
    axes_object.set_ylabel('Conditional mean observation')
    axes_object.set_xlim(min_value_to_plot, max_value_to_plot)
    axes_object.set_ylim(min_value_to_plot, max_value_to_plot)

    return main_line_handle


def _get_positive_skill_area(mean_value_in_training, min_value_in_plot,
                             max_value_in_plot):
    """Returns positive-skill area (where BSS > 0) for attributes diagram.

    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_in_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_in_plot: Max value in plot (for both x- and y-axes).
    :return: x_coords_left: length-5 numpy array of x-coordinates for left part
        of positive-skill area.
    :return: y_coords_left: Same but for y-coordinates.
    :return: x_coords_right: length-5 numpy array of x-coordinates for right
        part of positive-skill area.
    :return: y_coords_right: Same but for y-coordinates.
    """

    x_coords_left = numpy.array([
        min_value_in_plot, mean_value_in_training, mean_value_in_training,
        min_value_in_plot, min_value_in_plot
    ])
    y_coords_left = numpy.array([
        min_value_in_plot, min_value_in_plot, mean_value_in_training,
        (min_value_in_plot + mean_value_in_training) / 2, min_value_in_plot
    ])

    x_coords_right = numpy.array([
        mean_value_in_training, max_value_in_plot, max_value_in_plot,
        mean_value_in_training, mean_value_in_training
    ])
    y_coords_right = numpy.array([
        mean_value_in_training,
        (max_value_in_plot + mean_value_in_training) / 2,
        max_value_in_plot, max_value_in_plot, mean_value_in_training
    ])

    return x_coords_left, y_coords_left, x_coords_right, y_coords_right


def _get_zero_skill_line(mean_value_in_training, min_value_in_plot,
                         max_value_in_plot):
    """Returns zero-skill line (where BSS = 0) for attributes diagram.

    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_in_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_in_plot: Max value in plot (for both x- and y-axes).
    :return: x_coords: length-2 numpy array of x-coordinates.
    :return: y_coords: Same but for y-coordinates.
    """

    x_coords = numpy.array([min_value_in_plot, max_value_in_plot], dtype=float)
    y_coords = 0.5 * (mean_value_in_training + x_coords)

    return x_coords, y_coords


def _plot_attr_diagram_background(
        axes_object, mean_value_in_training, min_value_in_plot,
        max_value_in_plot):
    """Plots background (reference lines and polygons) of attributes diagram.

    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_in_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_in_plot: Max value in plot (for both x- and y-axes).
    """

    x_coords_left, y_coords_left, x_coords_right, y_coords_right = (
        _get_positive_skill_area(
            mean_value_in_training=mean_value_in_training,
            min_value_in_plot=min_value_in_plot,
            max_value_in_plot=max_value_in_plot
        )
    )

    skill_area_colour = matplotlib.colors.to_rgba(
        ZERO_SKILL_LINE_COLOUR, POSITIVE_SKILL_AREA_OPACITY
    )

    left_polygon_coord_matrix = numpy.transpose(numpy.vstack((
        x_coords_left, y_coords_left
    )))
    left_patch_object = matplotlib.patches.Polygon(
        left_polygon_coord_matrix, lw=0,
        ec=skill_area_colour, fc=skill_area_colour
    )
    axes_object.add_patch(left_patch_object)

    right_polygon_coord_matrix = numpy.transpose(numpy.vstack((
        x_coords_right, y_coords_right
    )))
    right_patch_object = matplotlib.patches.Polygon(
        right_polygon_coord_matrix, lw=0,
        ec=skill_area_colour, fc=skill_area_colour
    )
    axes_object.add_patch(right_patch_object)

    no_skill_x_coords, no_skill_y_coords = _get_zero_skill_line(
        mean_value_in_training=mean_value_in_training,
        min_value_in_plot=min_value_in_plot,
        max_value_in_plot=max_value_in_plot
    )

    axes_object.plot(
        no_skill_x_coords, no_skill_y_coords, color=ZERO_SKILL_LINE_COLOUR,
        linestyle='solid', linewidth=ZERO_SKILL_LINE_WIDTH
    )

    climo_x_coords = numpy.full(2, mean_value_in_training)
    climo_y_coords = numpy.array([min_value_in_plot, max_value_in_plot])
    axes_object.plot(
        climo_x_coords, climo_y_coords, color=CLIMO_LINE_COLOUR,
        linestyle='dashed', linewidth=CLIMO_LINE_WIDTH
    )

    axes_object.plot(
        climo_y_coords, climo_x_coords, color=CLIMO_LINE_COLOUR,
        linestyle='dashed', linewidth=CLIMO_LINE_WIDTH
    )


def plot_inset_histogram(
        figure_object, bin_centers, bin_counts, has_predictions,
        bar_colour=HISTOGRAM_FACE_COLOUR):
    """Plots histogram as inset in attributes diagram.

    B = number of bins

    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :param bin_centers: length-B numpy array with value at center of each bin.
        These values will be plotted on the x-axis.
    :param bin_counts: length-B numpy array with number of examples in each bin.
        These values will be plotted on the y-axis.
    :param has_predictions: Boolean flag.  If True, histogram will contain
        prediction frequencies.  If False, will contain observation frequencies.
    :param bar_colour: Bar colour (in any format accepted by matplotlib).
    """

    error_checking.assert_is_numpy_array(bin_centers, num_dimensions=1)
    error_checking.assert_is_boolean(has_predictions)

    num_bins = len(bin_centers)
    expected_dim = numpy.array([num_bins], dtype=int)

    error_checking.assert_is_integer_numpy_array(bin_counts)
    error_checking.assert_is_geq_numpy_array(bin_counts, 0)
    error_checking.assert_is_numpy_array(
        bin_counts, exact_dimensions=expected_dim
    )

    bin_frequencies = bin_counts.astype(float) / numpy.sum(bin_counts)

    if has_predictions:
        inset_axes_object = figure_object.add_axes([0.675, 0.2, 0.2, 0.2])
    else:
        inset_axes_object = figure_object.add_axes([0.2, 0.65, 0.2, 0.2])

    fake_bin_centers = (
        0.5 + numpy.linspace(0, num_bins - 1, num=num_bins, dtype=float)
    )

    real_indices = numpy.where(numpy.invert(numpy.isnan(bin_centers)))[0]

    inset_axes_object.bar(
        fake_bin_centers[real_indices], bin_frequencies[real_indices], 1.,
        color=bar_colour, edgecolor=HISTOGRAM_EDGE_COLOUR,
        linewidth=HISTOGRAM_EDGE_WIDTH
    )
    inset_axes_object.set_ylim(bottom=0.)

    tick_indices = []

    for i in real_indices:
        if numpy.mod(i, 2) == 0:
            tick_indices.append(i)
            continue

        if i - 1 in real_indices or i + 1 in real_indices:
            continue

        tick_indices.append(i)

    x_tick_values = fake_bin_centers[tick_indices]
    x_tick_labels = ['{0:.1f}'.format(b) for b in bin_centers[tick_indices]]
    inset_axes_object.set_xticks(x_tick_values)
    inset_axes_object.set_xticklabels(x_tick_labels)

    for this_tick_object in inset_axes_object.xaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(HISTOGRAM_FONT_SIZE)
        this_tick_object.label.set_rotation('vertical')

    for this_tick_object in inset_axes_object.yaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(HISTOGRAM_FONT_SIZE)

    inset_axes_object.set_title(
        'Prediction frequency' if has_predictions else 'Observation frequency',
        fontsize=HISTOGRAM_FONT_SIZE
    )


def plot_attributes_diagram(
        figure_object, axes_object, mean_predictions, mean_observations,
        mean_value_in_training, min_value_to_plot, max_value_to_plot,
        line_colour=RELIABILITY_LINE_COLOUR, line_style='solid',
        line_width=RELIABILITY_LINE_WIDTH, example_counts=None,
        inv_mean_observations=None, inv_example_counts=None):
    """Plots attributes diagram.

    If `example_counts is None`, will not plot histogram of predicted values.

    If `inv_mean_observations is None` and `inv_example_counts is None`, will
    not plot histogram of observed values.

    B = number of bins

    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param mean_predictions: length-B numpy array of mean predicted values.
    :param mean_observations: length-B numpy array of mean observed values.
    :param mean_value_in_training: Mean of target variable in training data.
    :param min_value_to_plot: Minimum value in plot (for both x- and y-axes).
    :param max_value_to_plot: Max value in plot (for both x- and y-axes).
        If None, will be determined automatically.
    :param line_colour: See doc for `_plot_reliability_curve`.
    :param line_width: Same.
    :param line_style: Same.
    :param example_counts: length-B numpy array with number of examples in each
        bin.
    :param inv_mean_observations: length-B numpy array of mean observed values
        for inverted reliability curve.
    :param inv_example_counts: length-B numpy array of example counts for
        inverted reliability curve.
    :return: main_line_handle: See doc for `_plot_reliability_curve`.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(mean_predictions, num_dimensions=1)

    num_bins = len(mean_predictions)
    expected_dim = numpy.array([num_bins], dtype=int)
    error_checking.assert_is_numpy_array(
        mean_observations, exact_dimensions=expected_dim
    )

    plot_prediction_histogram = example_counts is not None

    if plot_prediction_histogram:
        error_checking.assert_is_integer_numpy_array(example_counts)
        error_checking.assert_is_geq_numpy_array(example_counts, 0)
        error_checking.assert_is_numpy_array(
            example_counts, exact_dimensions=expected_dim
        )

    error_checking.assert_is_not_nan(mean_value_in_training)
    error_checking.assert_is_geq(max_value_to_plot, min_value_to_plot)
    if max_value_to_plot == min_value_to_plot:
        max_value_to_plot = min_value_to_plot + 1.

    plot_obs_histogram = not(
        inv_mean_observations is None and inv_example_counts is None
    )

    if plot_obs_histogram:
        error_checking.assert_is_numpy_array(
            inv_mean_observations, exact_dimensions=expected_dim
        )

        error_checking.assert_is_integer_numpy_array(inv_example_counts)
        error_checking.assert_is_geq_numpy_array(inv_example_counts, 0)
        error_checking.assert_is_numpy_array(
            inv_example_counts, exact_dimensions=expected_dim
        )

    _plot_attr_diagram_background(
        axes_object=axes_object, mean_value_in_training=mean_value_in_training,
        min_value_in_plot=min_value_to_plot, max_value_in_plot=max_value_to_plot
    )

    if plot_prediction_histogram:
        plot_inset_histogram(
            figure_object=figure_object, bin_centers=mean_predictions,
            bin_counts=example_counts, has_predictions=True,
            bar_colour=line_colour
        )

    if plot_obs_histogram:
        plot_inset_histogram(
            figure_object=figure_object, bin_centers=inv_mean_observations,
            bin_counts=inv_example_counts, has_predictions=False,
            bar_colour=line_colour
        )

    return _plot_reliability_curve(
        axes_object=axes_object, mean_predictions=mean_predictions,
        mean_observations=mean_observations,
        min_value_to_plot=min_value_to_plot,
        max_value_to_plot=max_value_to_plot,
        line_colour=line_colour, line_style=line_style, line_width=line_width
    )


def plot_taylor_diagram(target_stdev, prediction_stdev, correlation,
                        marker_colour, figure_object):
    """Plots Taylor diagram.

    :param target_stdev: Standard deviation of target (actual) values.
    :param prediction_stdev: Standard deviation of predicted values.
    :param correlation: Correlation between actual and predicted values.
    :param marker_colour: Colour for markers (in any format accepted by
        matplotlib).
    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :return: taylor_diagram_object: Handle for Taylor diagram (instance of
        `taylor_diagram.TaylorDiagram`).
    """

    error_checking.assert_is_geq(target_stdev, 0.)
    error_checking.assert_is_geq(prediction_stdev, 0.)
    error_checking.assert_is_geq(correlation, -1., allow_nan=True)
    error_checking.assert_is_leq(correlation, 1., allow_nan=True)

    taylor_diagram_object = taylor_diagram.TaylorDiagram(
        refstd=target_stdev, fig=figure_object, srange=(0, 2), extend=False
    )

    target_marker_object = taylor_diagram_object.samplePoints[0]
    target_marker_object.set_marker(TAYLOR_TARGET_MARKER_TYPE)
    target_marker_object.set_markersize(TAYLOR_TARGET_MARKER_SIZE)
    target_marker_object.set_markerfacecolor(marker_colour)
    target_marker_object.set_markeredgewidth(0)

    if not numpy.isnan(correlation):
        taylor_diagram_object.add_sample(
            stddev=prediction_stdev, corrcoef=correlation
        )

        prediction_marker_object = taylor_diagram_object.samplePoints[-1]
        prediction_marker_object.set_marker(TAYLOR_PREDICTION_MARKER_TYPE)
        prediction_marker_object.set_markersize(TAYLOR_PREDICTION_MARKER_SIZE)
        prediction_marker_object.set_markerfacecolor(marker_colour)
        prediction_marker_object.set_markeredgewidth(0)

    crmse_contour_object = taylor_diagram_object.add_contours(
        levels=5, colors='0.5'
    )
    pyplot.clabel(crmse_contour_object, inline=1, fmt='%.0f')

    taylor_diagram_object.add_grid()
    taylor_diagram_object._ax.axis[:].major_ticks.set_tick_out(True)

    return taylor_diagram_object


def plot_score_profile(
        heights_m_agl, score_values, score_name, line_colour, line_width,
        use_log_scale, axes_object, are_axes_new, line_style='solid'):
    """Plots vertical profile of one score.

    H = number of heights

    :param heights_m_agl: length-H numpy array of heights (metres above ground
        level).
    :param score_values: length-H numpy array with values of score.
    :param score_name: Name of score (must be accepted by `_check_score_name`).
    :param line_colour: Line colour (in any format accepted by matplotlib).
    :param line_width: Line width (in any format accepted by matplotlib).
    :param use_log_scale: Boolean flag.  If True, will plot height (y-axis) in
        logarithmic scale.  If False, will plot height in linear scale.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param are_axes_new: Boolean flag.  If True, axes are new (do not already
        contain plot).
    :param line_style: Line style (in any format accepted by matplotlib).
    :return: main_line_handle: Handle for main line (score profile).
    """

    error_checking.assert_is_numpy_array(heights_m_agl, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(heights_m_agl, 0.)

    num_heights = len(heights_m_agl)
    error_checking.assert_is_numpy_array(
        score_values, exact_dimensions=numpy.array([num_heights], dtype=int)
    )

    _check_score_name(score_name)
    error_checking.assert_is_boolean(use_log_scale)
    error_checking.assert_is_boolean(are_axes_new)

    if are_axes_new:
        orig_x_limits = []
        orig_y_limits = []
    else:
        orig_x_limits = axes_object.get_xlim()
        orig_y_limits = axes_object.get_ylim()

    if use_log_scale:
        axes_object.set_yscale('log')

    heights_km_agl = heights_m_agl * METRES_TO_KM
    min_height_km_agl = numpy.min(heights_km_agl)
    max_height_km_agl = numpy.max(heights_km_agl)

    if not are_axes_new:
        min_height_km_agl = numpy.minimum(min_height_km_agl, orig_y_limits[0])
        max_height_km_agl = numpy.maximum(max_height_km_agl, orig_y_limits[1])

    skill_score_names = [MAE_SKILL_SCORE_NAME, MSE_SKILL_SCORE_NAME]
    possibly_negative_score_names = (
        skill_score_names + [BIAS_NAME, CORRELATION_NAME, KGE_NAME]
    )

    if score_name in possibly_negative_score_names:
        reference_x_coords = numpy.full(2, 0.)
        reference_y_coords = numpy.array([0, max_height_km_agl], dtype=float)

        axes_object.plot(
            reference_x_coords, reference_y_coords, color=REFERENCE_LINE_COLOUR,
            linestyle='dashed', linewidth=REFERENCE_LINE_WIDTH
        )

    finite_indices = numpy.where(numpy.isfinite(score_values))[0]

    main_line_handle = axes_object.plot(
        score_values[finite_indices], heights_km_agl[finite_indices],
        color=line_colour, linestyle=line_style, linewidth=line_width
    )[0]

    x_max = numpy.maximum(
        numpy.max(score_values[finite_indices]), 0.
    )

    if not are_axes_new:
        x_max = numpy.maximum(x_max, orig_x_limits[1])

    if score_name in possibly_negative_score_names:
        x_min = numpy.minimum(
            numpy.min(score_values[finite_indices]), 0.
        )

        if not are_axes_new:
            x_min = numpy.minimum(x_min, orig_x_limits[0])

        if score_name in skill_score_names:
            x_min = numpy.maximum(x_min, -1.)
    else:
        x_min = 0.

    axes_object.set_xlim(x_min, x_max)

    if use_log_scale:
        axes_object.set_ylim(min_height_km_agl, max_height_km_agl)
    else:
        axes_object.set_ylim(0, max_height_km_agl)

    y_tick_strings = profile_plotting.create_height_labels(
        tick_values_km_agl=axes_object.get_yticks(),
        use_log_scale=use_log_scale
    )
    axes_object.set_yticklabels(y_tick_strings)
    axes_object.set_ylabel('Height (km AGL)')

    return main_line_handle


def plot_error_distribution(
        error_values, min_error_to_plot, max_error_to_plot, axes_object):
    """Plots one error distribution.

    :param error_values: 1-D numpy array of signed errors.
    :param min_error_to_plot: See doc for `plot_error_dist_many_heights`.
    :param max_error_to_plot: Same.
    :param axes_object: Same.
    """

    error_checking.assert_is_numpy_array_without_nan(error_values)
    error_checking.assert_is_numpy_array(error_values, num_dimensions=1)
    error_checking.assert_is_greater(max_error_to_plot, min_error_to_plot)

    boxplot_style_dict = {
        'color': 'k',
        'linewidth': 2
    }

    axes_object.boxplot(
        error_values, vert=True, notch=False, sym='o', whis=(5, 95),
        medianprops=boxplot_style_dict, boxprops=boxplot_style_dict,
        whiskerprops=boxplot_style_dict, capprops=boxplot_style_dict
    )

    axes_object.set_ylim(min_error_to_plot, max_error_to_plot)
    axes_object.set_xticks([], [])


def plot_error_dist_many_heights(
        error_matrix, heights_m_agl, min_error_to_plot, max_error_to_plot,
        axes_object):
    """Plots error distribution at each height on the same axes.

    E = number of examples
    H = number of heights

    :param error_matrix: E-by-H numpy array of signed errors.
    :param heights_m_agl: length-H numpy array of heights (metres above ground
        level).
    :param min_error_to_plot: Lower limit for x-axis.
    :param max_error_to_plot: Upper limit for x-axis.
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    error_checking.assert_is_numpy_array_without_nan(error_matrix)
    error_checking.assert_is_numpy_array(error_matrix, num_dimensions=2)

    num_heights = error_matrix.shape[1]

    error_checking.assert_is_geq_numpy_array(heights_m_agl, 0.)
    error_checking.assert_is_numpy_array(
        heights_m_agl, exact_dimensions=numpy.array([num_heights], dtype=int)
    )

    error_checking.assert_is_greater(max_error_to_plot, min_error_to_plot)

    boxplot_style_dict = {
        'color': 'k',
        'linewidth': 2
    }

    y_values = numpy.linspace(
        0, num_heights - 1, num=num_heights, dtype=float
    )

    for j in range(num_heights):
        axes_object.boxplot(
            error_matrix[:, j], widths=0.9, vert=False, notch=False, sym='o',
            whis=(5, 95),
            medianprops=boxplot_style_dict, boxprops=boxplot_style_dict,
            whiskerprops=boxplot_style_dict, capprops=boxplot_style_dict,
            positions=y_values[[j]]
        )

    axes_object.set_xlim(min_error_to_plot, max_error_to_plot)

    y_tick_strings = profile_plotting.create_height_labels(
        tick_values_km_agl=METRES_TO_KM * heights_m_agl,
        use_log_scale=False
    )

    for j in range(len(y_tick_strings)):
        if numpy.mod(j, 5) == 0:
            continue

        y_tick_strings[j] = ' '

    axes_object.set_yticklabels(y_tick_strings)
    axes_object.set_ylabel('Height (km AGL)')


def plot_rel_curve_many_heights(
        mean_target_matrix, mean_prediction_matrix, heights_m_agl,
        min_value_to_plot, max_value_to_plot, axes_object,
        colour_map_object=DEFAULT_HEIGHT_CMAP_OBJECT):
    """Plots reliability curves for many heights on the same axes.

    Reliability curves should be for the same variable, just at different
    heights.

    B = number of forecast bins
    H = number of heights

    :param mean_target_matrix: H-by-B numpy array of mean target (actual)
        values.
    :param mean_prediction_matrix: H-by-B numpy array of mean predicted values.
    :param heights_m_agl: length-H numpy array of heights (metres above ground
        level).
    :param min_value_to_plot: Minimum value to plot (for both x- and y-axes).
    :param max_value_to_plot: Max value to plot (for both x- and y-axes).
    :param axes_object: Will plot on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm` or
        similar).  Will be used to colour reliability curves by height.
    """

    error_checking.assert_is_numpy_array(mean_target_matrix, num_dimensions=2)
    error_checking.assert_is_numpy_array(
        mean_prediction_matrix,
        exact_dimensions=numpy.array(mean_target_matrix.shape, dtype=int)
    )

    num_heights = mean_target_matrix.shape[0]

    error_checking.assert_is_geq_numpy_array(heights_m_agl, 0.)
    error_checking.assert_is_numpy_array(
        heights_m_agl, exact_dimensions=numpy.array([num_heights], dtype=int)
    )

    error_checking.assert_is_greater(max_value_to_plot, min_value_to_plot)

    heights_km_agl = heights_m_agl * METRES_TO_KM
    colour_norm_object = matplotlib.colors.LogNorm(
        vmin=numpy.min(heights_km_agl), vmax=numpy.max(heights_km_agl)
    )

    for j in range(num_heights):
        this_colour = colour_map_object(colour_norm_object(
            heights_km_agl[j]
        ))

        _plot_reliability_curve(
            axes_object=axes_object,
            mean_predictions=mean_prediction_matrix[j, :],
            mean_observations=mean_target_matrix[j, :],
            line_colour=this_colour, min_value_to_plot=min_value_to_plot,
            max_value_to_plot=max_value_to_plot
        )

    axes_object.set_xlim(min_value_to_plot, max_value_to_plot)
    axes_object.set_ylim(min_value_to_plot, max_value_to_plot)

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=axes_object, data_matrix=heights_km_agl,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        font_size=FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = profile_plotting.create_height_labels(
        tick_values_km_agl=tick_values, use_log_scale=True
    )
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    colour_bar_object.set_label('Height (km AGL)', fontsize=FONT_SIZE)


def plot_taylor_diagram_many_heights(
        target_stdevs, prediction_stdevs, correlations, heights_m_agl,
        figure_object, colour_map_object=DEFAULT_HEIGHT_CMAP_OBJECT):
    """Plots Taylor diagram for many heights on the same axes.

    Each point should be for the same variable, just at different
    heights.

    H = number of heights

    :param target_stdevs: length-H numpy array with standard deviations of
        target (actual) values.
    :param prediction_stdevs: length-H numpy array with standard deviations of
        predicted values.
    :param correlations: length-H numpy array of correlations.
    :param heights_m_agl: length-H numpy array of heights (metres above ground
        level).
    :param figure_object: Will plot on this figure (instance of
        `matplotlib.figure.Figure`).
    :param colour_map_object: Colour map (instance of `matplotlib.pyplot.cm` or
        similar).  Will be used to colour points in Taylor diagram by height.
    :return: taylor_diagram_object: Handle for Taylor diagram (instance of
        `taylor_diagram.TaylorDiagram`).
    """

    error_checking.assert_is_geq_numpy_array(target_stdevs, 0.)
    error_checking.assert_is_numpy_array(target_stdevs, num_dimensions=1)

    num_heights = len(target_stdevs)
    expected_dim = numpy.array([num_heights], dtype=int)

    error_checking.assert_is_geq_numpy_array(prediction_stdevs, 0.)
    error_checking.assert_is_numpy_array(
        prediction_stdevs, exact_dimensions=expected_dim
    )

    error_checking.assert_is_geq_numpy_array(correlations, -1., allow_nan=True)
    error_checking.assert_is_leq_numpy_array(correlations, 1., allow_nan=True)
    error_checking.assert_is_numpy_array(
        correlations, exact_dimensions=expected_dim
    )

    error_checking.assert_is_geq_numpy_array(heights_m_agl, 0.)
    error_checking.assert_is_numpy_array(
        heights_m_agl, exact_dimensions=expected_dim
    )

    heights_km_agl = heights_m_agl * METRES_TO_KM
    colour_norm_object = matplotlib.colors.LogNorm(
        vmin=numpy.min(heights_km_agl), vmax=numpy.max(heights_km_agl)
    )

    mean_target_stdev = numpy.mean(target_stdevs)
    this_ratio = numpy.maximum(
        numpy.max(target_stdevs), numpy.max(prediction_stdevs)
    ) / mean_target_stdev

    taylor_diagram_object = taylor_diagram.TaylorDiagram(
        refstd=mean_target_stdev, fig=figure_object, srange=(0, this_ratio),
        extend=False, plot_reference_line=False
    )

    this_marker_object = taylor_diagram_object.samplePoints[0]
    this_marker_object.set_visible(False)

    for j in range(num_heights):
        if numpy.isnan(correlations[j]):
            continue

        this_colour = colour_map_object(colour_norm_object(
            heights_km_agl[j]
        ))

        taylor_diagram_object.add_sample(stddev=target_stdevs[j], corrcoef=1.)

        this_marker_object = taylor_diagram_object.samplePoints[-1]
        this_marker_object.set_marker(TAYLOR_TARGET_MARKER_TYPE)
        this_marker_object.set_markersize(TAYLOR_TARGET_MARKER_SIZE)
        this_marker_object.set_markerfacecolor(this_colour)
        this_marker_object.set_markeredgewidth(0)

        taylor_diagram_object.add_sample(
            stddev=prediction_stdevs[j], corrcoef=correlations[j]
        )

        this_marker_object = taylor_diagram_object.samplePoints[-1]
        this_marker_object.set_marker(TAYLOR_PREDICTION_MARKER_TYPE)
        this_marker_object.set_markersize(TAYLOR_PREDICTION_MARKER_SIZE)
        this_marker_object.set_markerfacecolor(this_colour)
        this_marker_object.set_markeredgewidth(0)

    crmse_contour_object = taylor_diagram_object.add_contours(
        levels=5, colors='0.5'
    )
    pyplot.clabel(crmse_contour_object, inline=1, fmt='%.0f')

    taylor_diagram_object.add_grid()
    taylor_diagram_object._ax.axis[:].major_ticks.set_tick_out(True)

    colour_bar_object = plotting_utils.plot_colour_bar(
        axes_object_or_matrix=figure_object.axes[0], data_matrix=heights_km_agl,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        orientation_string='vertical', extend_min=False, extend_max=False,
        fraction_of_axis_length=0.85, font_size=FONT_SIZE
    )

    tick_values = colour_bar_object.get_ticks()
    tick_strings = profile_plotting.create_height_labels(
        tick_values_km_agl=tick_values, use_log_scale=True
    )
    colour_bar_object.set_ticks(tick_values)
    colour_bar_object.set_ticklabels(tick_strings)

    colour_bar_object.set_label('Height (km AGL)', fontsize=FONT_SIZE)

    return taylor_diagram_object
