"""Plotting methods for evaluation of uncertainty quantification (UQ)."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from ml4rt.utils import example_utils
from ml4rt.utils import uq_evaluation
from ml4rt.utils import pit_utils
from ml4rt.utils import spread_skill_utils as ss_utils
from ml4rt.utils import discard_test_utils as dt_utils

TOLERANCE = 1e-6

TARGET_NAME_ABBREV_TO_FANCY = {
    example_utils.SHORTWAVE_HEATING_RATE_NAME: 'SW heating rate',
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: r'SW $F_{down}^{sfc}$',
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME: r'SW $F_{up}^{TOA}$',
    uq_evaluation.SHORTWAVE_NET_FLUX_NAME: r'SW $F_{net}$',
    example_utils.LONGWAVE_HEATING_RATE_NAME: 'LW heating rate',
    example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME: r'LW $F_{down}^{sfc}$',
    example_utils.LONGWAVE_TOA_UP_FLUX_NAME: r'LW $F_{up}^{TOA}$',
    uq_evaluation.LONGWAVE_NET_FLUX_NAME: r'LW $F_{net}$'
}

TARGET_NAME_TO_UNITS = {
    example_utils.SHORTWAVE_HEATING_RATE_NAME: r'K day$^{-1}$',
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME: r'W m$^{-2}$',
    uq_evaluation.SHORTWAVE_NET_FLUX_NAME: r'W m$^{-2}$',
    example_utils.LONGWAVE_HEATING_RATE_NAME: r'K day$^{-1}$',
    example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_utils.LONGWAVE_TOA_UP_FLUX_NAME: r'W m$^{-2}$',
    uq_evaluation.LONGWAVE_NET_FLUX_NAME: r'W m$^{-2}$'
}

REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
REFERENCE_LINE_WIDTH = 2.

DEFAULT_LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
DEFAULT_LINE_WIDTH = 3.

INSET_HISTO_FACE_COLOUR = numpy.full(3, 152. / 255)
INSET_HISTO_EDGE_COLOUR = numpy.full(3, 0.)
INSET_HISTO_EDGE_WIDTH = 1.

DEFAULT_HISTOGRAM_FACE_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
DEFAULT_HISTOGRAM_EDGE_COLOUR = numpy.full(3, 0.)
DEFAULT_HISTOGRAM_EDGE_WIDTH = 2.

MEAN_PREDICTION_LINE_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
MEAN_PREDICTION_COLOUR_STRING = 'purple'
MEAN_TARGET_LINE_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
MEAN_TARGET_COLOUR_STRING = 'green'

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15

FONT_SIZE = 40
INSET_FONT_SIZE = 20

pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _plot_means_as_inset(
        figure_object, bin_centers, bin_mean_predictions,
        bin_mean_target_values, plotting_corner_string, for_spread_skill_plot):
    """Plots means (mean prediction and target by bin) as inset in another fig.

    B = number of bins

    :param figure_object: Will plot as inset in this figure (instance of
        `matplotlib.figure.Figure`).
    :param bin_centers: length-B numpy array with value at center of each bin.
        These values will be plotted on the x-axis.
    :param bin_mean_predictions: length-B numpy array with mean prediction in
        each bin.  These values will be plotted on the y-axis.
    :param bin_mean_target_values: length-B numpy array with mean target value
        (event frequency) in each bin.  These values will be plotted on the
        y-axis.
    :param plotting_corner_string: String in
        ['top_right', 'top_left', 'bottom_right', 'bottom_left'].
    :param for_spread_skill_plot: Boolean flag.
    :return: inset_axes_object: Axes handle for histogram (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    if plotting_corner_string == 'top_right':
        inset_axes_object = figure_object.add_axes([0.625, 0.55, 0.25, 0.25])
    elif plotting_corner_string == 'bottom_right':
        inset_axes_object = figure_object.add_axes([0.625, 0.3, 0.25, 0.25])
    elif plotting_corner_string == 'bottom_left':
        inset_axes_object = figure_object.add_axes([0.2, 0.3, 0.25, 0.25])
    elif plotting_corner_string == 'top_left':
        inset_axes_object = figure_object.add_axes([0.2, 0.55, 0.25, 0.25])

    nan_flags = numpy.logical_or(
        numpy.isnan(bin_mean_target_values),
        numpy.isnan(bin_mean_predictions)
    )
    assert not numpy.all(nan_flags)
    real_indices = numpy.where(numpy.invert(nan_flags))[0]

    target_handle = inset_axes_object.plot(
        bin_centers[real_indices], bin_mean_target_values[real_indices],
        color=MEAN_TARGET_LINE_COLOUR, linestyle='solid', linewidth=2
    )[0]

    prediction_handle = inset_axes_object.plot(
        bin_centers[real_indices], bin_mean_predictions[real_indices],
        color=MEAN_PREDICTION_LINE_COLOUR, linestyle='dashed', linewidth=2
    )[0]

    y_max = max([
        numpy.nanmax(bin_mean_predictions),
        numpy.nanmax(bin_mean_target_values)
    ])
    y_min = min([
        numpy.nanmin(bin_mean_predictions),
        numpy.nanmin(bin_mean_target_values)
    ])
    inset_axes_object.set_ylim(y_min, y_max)
    inset_axes_object.set_xlim(left=0.)

    for this_tick_object in inset_axes_object.xaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(INSET_FONT_SIZE)
        this_tick_object.label.set_rotation('vertical')

    for this_tick_object in inset_axes_object.yaxis.get_major_ticks():
        this_tick_object.label.set_fontsize(INSET_FONT_SIZE)

    if for_spread_skill_plot:
        anchor_arg = (0.5, -0.25)
    else:
        anchor_arg = (0.5, -0.2)

    inset_axes_object.legend(
        [target_handle, prediction_handle],
        ['Mean target', 'Mean prediction'],
        loc='upper center', bbox_to_anchor=anchor_arg,
        fancybox=True, shadow=True, ncol=1, fontsize=INSET_FONT_SIZE
    )

    return inset_axes_object


def _plot_histogram(axes_object, bin_edges, bin_frequencies):
    """Plots histogram on existing axes.

    B = number of bins

    :param axes_object: Will plot histogram on these axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param bin_edges: length-(B + 1) numpy array with values at edges of each
        bin. These values will be plotted on the x-axis.
    :param bin_frequencies: length-B numpy array with fraction of examples in
        each bin. These values will be plotted on the y-axis.
    :return: histogram_axes_object: Axes handle for histogram only (also
        instance of `matplotlib.axes._subplots.AxesSubplot`).
    """

    histogram_axes_object = axes_object.twinx()
    axes_object.set_zorder(histogram_axes_object.get_zorder() + 1)
    axes_object.patch.set_visible(False)

    histogram_axes_object.bar(
        x=bin_edges[:-1], height=bin_frequencies, width=numpy.diff(bin_edges),
        color=INSET_HISTO_FACE_COLOUR, edgecolor=INSET_HISTO_EDGE_COLOUR,
        linewidth=INSET_HISTO_EDGE_WIDTH, align='edge'
    )

    return histogram_axes_object


def plot_spread_vs_skill(
        result_table_xarray, target_var_name, target_height_m_agl=None,
        line_colour=DEFAULT_LINE_COLOUR, line_style='solid',
        line_width=DEFAULT_LINE_WIDTH):
    """Creates spread-skill plot for one target variable.

    :param result_table_xarray: xarray table in format returned by
        `spread_skill_utils.get_results_all_vars`.
    :param target_var_name: Will create spread-skill plot for this target
        variable.
    :param target_height_m_agl: Will create spread-skill plot for given target
        variable at this height (metres above ground).  If `target_var_name`
        does not correspond to a vector target variable, make this None.
    :param line_colour: Line colour (in any format accepted by matplotlib).
    :param line_style: Line style (in any format accepted by matplotlib).
    :param line_width: Line width (in any format accepted by matplotlib).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Find values for the chosen target variable.
    t = result_table_xarray

    try:
        k = t.coords[ss_utils.SCALAR_FIELD_DIM].values.tolist().index(
            target_var_name
        )

        spread_skill_reliability = t[ss_utils.SCALAR_SSREL_KEY].values[k]
        spread_skill_ratio = t[ss_utils.SCALAR_SSRAT_KEY].values[k]
        mean_prediction_stdevs = t[ss_utils.SCALAR_MEAN_STDEV_KEY].values[k, :]
        rmse_values = t[ss_utils.SCALAR_RMSE_KEY].values[k, :]
        bin_edges = t[ss_utils.SCALAR_BIN_EDGE_KEY].values[k, :]
        example_counts = t[ss_utils.SCALAR_EXAMPLE_COUNT_KEY].values[k, :]
        mean_mean_predictions = (
            t[ss_utils.SCALAR_MEAN_MEAN_PREDICTION_KEY].values[k, :]
        )
        mean_target_values = t[ss_utils.SCALAR_MEAN_TARGET_KEY].values[k, :]
    except ValueError:
        try:
            k = t.coords[ss_utils.AUX_TARGET_FIELD_DIM].values.tolist().index(
                target_var_name
            )

            spread_skill_reliability = t[ss_utils.AUX_SSREL_KEY].values[k]
            spread_skill_ratio = t[ss_utils.AUX_SSRAT_KEY].values[k]
            mean_prediction_stdevs = t[ss_utils.AUX_MEAN_STDEV_KEY].values[k, :]
            rmse_values = t[ss_utils.AUX_RMSE_KEY].values[k, :]
            bin_edges = t[ss_utils.AUX_BIN_EDGE_KEY].values[k, :]
            example_counts = t[ss_utils.AUX_EXAMPLE_COUNT_KEY].values[k, :]
            mean_mean_predictions = (
                t[ss_utils.AUX_MEAN_MEAN_PREDICTION_KEY].values[k, :]
            )
            mean_target_values = t[ss_utils.AUX_MEAN_TARGET_KEY].values[k, :]
        except ValueError:
            k = t.coords[ss_utils.VECTOR_FIELD_DIM].values.tolist().index(
                target_var_name
            )

            if target_height_m_agl is None:
                spread_skill_reliability = (
                    t[ss_utils.VECTOR_FLAT_SSREL_KEY].values[k]
                )
                spread_skill_ratio = (
                    t[ss_utils.VECTOR_FLAT_SSRAT_KEY].values[k]
                )
                mean_prediction_stdevs = (
                    t[ss_utils.VECTOR_FLAT_MEAN_STDEV_KEY].values[k, :]
                )
                rmse_values = t[ss_utils.VECTOR_FLAT_RMSE_KEY].values[k, :]
                bin_edges = t[ss_utils.VECTOR_FLAT_BIN_EDGE_KEY].values[k, :]
                example_counts = (
                    t[ss_utils.VECTOR_FLAT_EXAMPLE_COUNT_KEY].values[k, :]
                )
                mean_target_values = (
                    t[ss_utils.VECTOR_FLAT_MEAN_TARGET_KEY].values[k, :]
                )
                mean_mean_predictions = t[
                    ss_utils.VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY
                ].values[k, :]
            else:
                height_diffs_metres = numpy.absolute(
                    t.coords[ss_utils.HEIGHT_DIM].values - target_height_m_agl
                )
                j = numpy.where(height_diffs_metres <= TOLERANCE)[0][0]

                spread_skill_reliability = (
                    t[ss_utils.VECTOR_SSREL_KEY].values[k, j]
                )
                spread_skill_ratio = t[ss_utils.VECTOR_SSRAT_KEY].values[k, j]
                mean_prediction_stdevs = (
                    t[ss_utils.VECTOR_MEAN_STDEV_KEY].values[k, j, :]
                )
                rmse_values = t[ss_utils.VECTOR_RMSE_KEY].values[k, j, :]
                bin_edges = t[ss_utils.VECTOR_BIN_EDGE_KEY].values[k, j, :]
                example_counts = (
                    t[ss_utils.VECTOR_EXAMPLE_COUNT_KEY].values[k, j, :]
                )
                mean_target_values = (
                    t[ss_utils.VECTOR_MEAN_TARGET_KEY].values[k, j, :]
                )
                mean_mean_predictions = t[
                    ss_utils.VECTOR_MEAN_MEAN_PREDICTION_KEY
                ].values[k, j, :]

    # Do actual stuff.
    nan_flags = numpy.logical_or(
        numpy.isnan(mean_prediction_stdevs),
        numpy.isnan(rmse_values)
    )
    assert not numpy.all(nan_flags)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    max_value_to_plot = 1.01 * max([
        numpy.nanmax(mean_prediction_stdevs),
        numpy.nanmax(rmse_values)
    ])
    perfect_x_coords = numpy.array([0, max_value_to_plot])
    perfect_y_coords = numpy.array([0, max_value_to_plot])
    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=REFERENCE_LINE_COLOUR,
        linestyle='dashed', linewidth=REFERENCE_LINE_WIDTH
    )

    real_indices = numpy.where(numpy.invert(nan_flags))[0]
    axes_object.plot(
        mean_prediction_stdevs[real_indices],
        rmse_values[real_indices],
        color=line_colour, linestyle=line_style, linewidth=line_width,
        marker='o', markersize=12, markeredgewidth=0,
        markerfacecolor=line_colour, markeredgecolor=line_colour
    )

    unit_string = TARGET_NAME_TO_UNITS[target_var_name]
    axes_object.set_xlabel(
        'Spread (stdev of predictive distribution; {0:s})'.format(unit_string)
    )
    axes_object.set_ylabel(
        'Skill (RMSE of mean prediction; {0:s})'.format(unit_string)
    )

    bin_frequencies = example_counts.astype(float) / numpy.sum(example_counts)

    if numpy.isnan(mean_prediction_stdevs[-1]):
        bin_edges[-1] = bin_edges[-2] + (bin_edges[-2] - bin_edges[-3])
    else:
        bin_edges[-1] = (
            bin_edges[-2] + 2 * (mean_prediction_stdevs[-1] - bin_edges[-2])
        )

    histogram_axes_object = _plot_histogram(
        axes_object=axes_object, bin_edges=bin_edges,
        bin_frequencies=bin_frequencies * 100
    )
    histogram_axes_object.set_ylabel('% examples in each bin')

    # axes_object.set_xlim(min([bin_edges[0], 0]), bin_edges[-1])
    # axes_object.set_ylim(0, 1.01 * numpy.nanmax(rmse_values))

    axes_object.set_xlim(min([bin_edges[0], 0]), max_value_to_plot)
    axes_object.set_ylim(0, max_value_to_plot)

    inset_axes_object = _plot_means_as_inset(
        figure_object=figure_object, bin_centers=mean_prediction_stdevs,
        bin_mean_predictions=mean_mean_predictions,
        bin_mean_target_values=mean_target_values,
        plotting_corner_string='bottom_right',
        for_spread_skill_plot=True
    )
    inset_axes_object.set_zorder(axes_object.get_zorder() + 1)

    inset_axes_object.set_xticks(axes_object.get_xticks())
    inset_axes_object.set_xlim(axes_object.get_xlim())
    inset_axes_object.set_xlabel(
        'Spread ({0:s})'.format(unit_string),
        fontsize=INSET_FONT_SIZE
    )
    inset_axes_object.set_ylabel(
        'Avg target or pred ({0:s})'.format(unit_string),
        fontsize=INSET_FONT_SIZE
    )
    inset_axes_object.set_title(
        'Avgs by model spread', fontsize=INSET_FONT_SIZE
    )

    title_string = (
        'Spread vs. skill for {0:s}{1:s}\n'
        'SSREL = {2:.3f} {3:s}; SSRAT = {4:.3f}'
    ).format(
        TARGET_NAME_ABBREV_TO_FANCY[target_var_name],
        '' if target_height_m_agl is None
        else ' at {0:d} m AGL'.format(int(numpy.round(target_height_m_agl))),
        spread_skill_reliability,
        TARGET_NAME_TO_UNITS[target_var_name],
        spread_skill_ratio
    )

    axes_object.set_title(title_string)

    return figure_object, axes_object


def plot_discard_test(
        result_table_xarray, target_var_name, target_height_m_agl=None,
        line_colour=DEFAULT_LINE_COLOUR, line_style='solid',
        line_width=DEFAULT_LINE_WIDTH):
    """Plots results of discard test.

    :param result_table_xarray: xarray table in format returned by
        `discard_test_utils.run_discard_test`.
    :param target_var_name: Will plot discard test for this target variable.
    :param target_height_m_agl: Will plot discard test for given target variable
        at this height (metres above ground).  If `target_var_name` does not
        correspond to a vector target variable, make this None.
    :param line_colour: See doc for `plot_spread_vs_skill`.
    :param line_style: Same.
    :param line_width: Same.
    :return: figure_object: Same.
    :return: axes_object: Same.
    """

    # Find values for the chosen target variable.
    t = result_table_xarray

    try:
        k = t.coords[dt_utils.SCALAR_FIELD_DIM].values.tolist().index(
            target_var_name
        )

        error_values = t[dt_utils.SCALAR_POST_DISCARD_ERROR_KEY].values[k, :]
        mean_mean_predictions = (
            t[dt_utils.SCALAR_MEAN_MEAN_PREDICTION_KEY].values[k, :]
        )
        mean_target_values = t[dt_utils.SCALAR_MEAN_TARGET_KEY].values[k, :]
        mono_fraction = t[dt_utils.SCALAR_MONO_FRACTION_KEY].values[k]
        mean_di = t[dt_utils.SCALAR_MEAN_DI_KEY].values[k]

        y_label_string = r'MSE for mean prediction (W$^2$ m$^{-4}$)'
    except ValueError:
        try:
            k = t.coords[dt_utils.AUX_TARGET_FIELD_DIM].values.tolist().index(
                target_var_name
            )

            error_values = t[dt_utils.AUX_POST_DISCARD_ERROR_KEY].values[k, :]
            mean_mean_predictions = (
                t[dt_utils.AUX_MEAN_MEAN_PREDICTION_KEY].values[k, :]
            )
            mean_target_values = t[dt_utils.AUX_MEAN_TARGET_KEY].values[k, :]
            mono_fraction = t[dt_utils.AUX_MONO_FRACTION_KEY].values[k]
            mean_di = t[dt_utils.AUX_MEAN_DI_KEY].values[k]

            y_label_string = r'MSE for mean prediction (W$^2$ m$^{-4}$)'
        except ValueError:
            k = t.coords[dt_utils.VECTOR_FIELD_DIM].values.tolist().index(
                target_var_name
            )

            if target_height_m_agl is None:
                error_values = (
                    t[dt_utils.VECTOR_FLAT_POST_DISCARD_ERROR_KEY].values[k, :]
                )
                mean_mean_predictions = t[
                    dt_utils.VECTOR_FLAT_MEAN_MEAN_PREDICTION_KEY
                ].values[k, :]
                mean_target_values = (
                    t[dt_utils.VECTOR_FLAT_MEAN_TARGET_KEY].values[k, :]
                )
                mono_fraction = (
                    t[dt_utils.VECTOR_FLAT_MONO_FRACTION_KEY].values[k]
                )
                mean_di = t[dt_utils.VECTOR_FLAT_MEAN_DI_KEY].values[k]
            else:
                height_diffs_metres = numpy.absolute(
                    t.coords[dt_utils.HEIGHT_DIM].values - target_height_m_agl
                )
                j = numpy.where(height_diffs_metres <= TOLERANCE)[0][0]

                error_values = (
                    t[dt_utils.VECTOR_POST_DISCARD_ERROR_KEY].values[k, j, :]
                )
                mean_mean_predictions = t[
                    dt_utils.VECTOR_MEAN_MEAN_PREDICTION_KEY
                ].values[k, j, :]
                mean_target_values = (
                    t[dt_utils.VECTOR_MEAN_TARGET_KEY].values[k, j, :]
                )
                mono_fraction = (
                    t[dt_utils.VECTOR_MONO_FRACTION_KEY].values[k, j]
                )
                mean_di = t[dt_utils.VECTOR_MEAN_DI_KEY].values[k, j]

            y_label_string = r'DWMSE for mean prediction (K$^3$ day$^{-3}$)'

    discard_fractions = (
        1. - result_table_xarray[dt_utils.EXAMPLE_FRACTION_KEY].values
    )
    # error_values = (
    #     result_table_xarray[dt_utils.POST_DISCARD_ERROR_KEY].values
    # )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.plot(
        discard_fractions, error_values,
        color=line_colour, linestyle=line_style, linewidth=line_width,
        marker='o', markersize=12, markeredgewidth=0,
        markerfacecolor=line_colour, markeredgecolor=line_colour
    )

    axes_object.set_xlabel('Discard fraction')
    axes_object.set_ylabel(y_label_string)
    axes_object.set_xlim(left=0.)

    inset_axes_object = _plot_means_as_inset(
        figure_object=figure_object, bin_centers=discard_fractions,
        bin_mean_predictions=mean_mean_predictions,
        bin_mean_target_values=mean_target_values,
        plotting_corner_string='top_right',
        for_spread_skill_plot=False
    )
    inset_axes_object.set_zorder(axes_object.get_zorder() + 1)

    inset_axes_object.set_xticks(axes_object.get_xticks())
    inset_axes_object.set_xlim(axes_object.get_xlim())
    inset_axes_object.set_xlabel(
        'Discard fraction',
        fontsize=INSET_FONT_SIZE
    )
    unit_string = TARGET_NAME_TO_UNITS[target_var_name]
    inset_axes_object.set_ylabel(
        'Avg target or pred ({0:s})'.format(unit_string),
        fontsize=INSET_FONT_SIZE
    )
    inset_axes_object.set_title(
        'Avgs by discard fraction', fontsize=INSET_FONT_SIZE
    )

    title_string = (
        'Discard test for {0:s}{1:s}\n'
        'MF = {2:.1f}%'
    ).format(
        TARGET_NAME_ABBREV_TO_FANCY[target_var_name],
        '' if target_height_m_agl is None
        else ' at {0:d} m AGL'.format(int(numpy.round(target_height_m_agl))),
        100 * mono_fraction
    )

    axes_object.set_title(title_string)

    return figure_object, axes_object


def plot_pit_histogram(
        result_table_xarray, target_var_name, target_height_m_agl=None,
        face_colour=DEFAULT_HISTOGRAM_FACE_COLOUR,
        edge_colour=DEFAULT_HISTOGRAM_EDGE_COLOUR,
        edge_width=DEFAULT_HISTOGRAM_EDGE_WIDTH):
    """Plots PIT (prob integral transform) histogram for one target variable.

    :param result_table_xarray: xarray table in format returned by
        `pit_utils.get_histogram_all_vars`.
    :param target_var_name: Will plot PIT histogram for this target variable.
    :param target_height_m_agl: Will plot PIT histogram for given target
        variable at this height (metres above ground).  If `target_var_name`
        does not correspond to a vector target variable, make this None.
    :param face_colour: Face colour (in any format accepted by matplotlib).
    :param edge_colour: Edge colour (in any format accepted by matplotlib).
    :param edge_width: Edge width (in any format accepted by matplotlib).
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Find values for the chosen target variable.
    t = result_table_xarray

    try:
        k = t.coords[pit_utils.SCALAR_FIELD_DIM].values.tolist().index(
            target_var_name
        )

        bin_counts = t[pit_utils.SCALAR_BIN_COUNT_KEY].values[k, :]
        pitd_value = t[pit_utils.SCALAR_PITD_KEY].values[k]
    except ValueError:
        try:
            k = t.coords[pit_utils.AUX_TARGET_FIELD_DIM].values.tolist().index(
                target_var_name
            )

            bin_counts = t[pit_utils.AUX_BIN_COUNT_KEY].values[k, :]
            pitd_value = t[pit_utils.AUX_PITD_KEY].values[k]
        except ValueError:
            k = t.coords[pit_utils.VECTOR_FIELD_DIM].values.tolist().index(
                target_var_name
            )

            if target_height_m_agl is None:
                bin_counts = t[pit_utils.VECTOR_FLAT_BIN_COUNT_KEY].values[k, :]
                pitd_value = t[pit_utils.VECTOR_FLAT_PITD_KEY].values[k]
            else:
                height_diffs_metres = numpy.absolute(
                    t.coords[pit_utils.HEIGHT_DIM].values - target_height_m_agl
                )
                j = numpy.where(height_diffs_metres <= TOLERANCE)[0][0]

                bin_counts = t[pit_utils.VECTOR_BIN_COUNT_KEY].values[k, j, :]
                pitd_value = t[pit_utils.VECTOR_PITD_KEY].values[k, j]

    bin_edges = t.coords[pit_utils.BIN_EDGE_DIM].values
    bin_frequencies = bin_counts.astype(float) / numpy.sum(bin_counts)

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    axes_object.bar(
        x=bin_edges[:-1], height=bin_frequencies, width=numpy.diff(bin_edges),
        color=face_colour, edgecolor=edge_colour, linewidth=edge_width,
        align='edge'
    )

    num_bins = len(bin_edges) - 1
    perfect_x_coords = numpy.array([0, 1], dtype=float)
    perfect_y_coords = numpy.array([1. / num_bins, 1. / num_bins])
    axes_object.plot(
        perfect_x_coords, perfect_y_coords, color=REFERENCE_LINE_COLOUR,
        linestyle='dashed', linewidth=REFERENCE_LINE_WIDTH
    )

    axes_object.set_xlabel('PIT value')
    axes_object.set_ylabel('Frequency')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(bottom=0.)

    title_string = (
        'PIT histogram for {0:s}{1:s}\n'
        'PITD = {2:.4f}'
    ).format(
        TARGET_NAME_ABBREV_TO_FANCY[target_var_name],
        '' if target_height_m_agl is None
        else ' at {0:d} m AGL'.format(int(numpy.round(target_height_m_agl))),
        pitd_value
    )

    axes_object.set_title(title_string)

    return figure_object, axes_object
