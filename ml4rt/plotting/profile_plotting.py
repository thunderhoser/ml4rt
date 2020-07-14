"""Methods for plotting vertical profiles."""

import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import temperature_conversions as temperature_conv
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io

METRES_TO_KM = 0.001
KG_TO_GRAMS = 1000.
KG_TO_MILLIGRAMS = 1e6

FIGURE_HANDLE_KEY = 'figure_object'
TEMPERATURE_HANDLE_KEY = 'temperature_axes_object'
HUMIDITY_HANDLE_KEY = 'humidity_axes_object'
WATER_CONTENT_HANDLE_KEY = 'water_content_axes_object'
HEATING_RATE_HANDLE_KEY = 'heating_rate_axes_object'
DOWN_FLUX_HANDLE_KEY = 'down_flux_axes_object'
UP_FLUX_HANDLE_KEY = 'up_flux_axes_object'

DEFAULT_LINE_WIDTH = 2

FANCY_FONT_SIZE = 12
FANCY_FIGURE_WIDTH_INCHES = 8
FANCY_FIGURE_HEIGHT_INCHES = 40

SIMPLE_FONT_SIZE = 30
SIMPLE_FIGURE_WIDTH_INCHES = 15
SIMPLE_FIGURE_HEIGHT_INCHES = 15

HEATING_RATE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
DOWNWELLING_FLUX_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
UPWELLING_FLUX_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255

TEMPERATURE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
HUMIDITY_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
WATER_CONTENT_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255


def _set_font_size(font_size):
    """Sets font size for a bunch of figure elements.

    :param font_size: Font size.
    """

    pyplot.rc('font', size=font_size)
    pyplot.rc('axes', titlesize=font_size)
    pyplot.rc('axes', labelsize=font_size)
    pyplot.rc('xtick', labelsize=font_size)
    pyplot.rc('ytick', labelsize=font_size)
    pyplot.rc('legend', fontsize=font_size)
    pyplot.rc('figure', titlesize=font_size)


def _make_spines_invisible(axes_object):
    """Makes spines along axis invisible.

    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    axes_object.set_frame_on(True)
    axes_object.patch.set_visible(False)

    for this_spine_object in axes_object.spines.values():
        this_spine_object.set_visible(False)


def create_height_labels(tick_values_km_agl, use_log_scale):
    """Creates labels for height axis.

    H = number of tick values

    :param tick_values_km_agl: length-H numpy array of tick values (km above
        ground level).
    :param use_log_scale: Boolean flag.  If True, will assume that height axis is
        logarithmic.
    :return: tick_strings: length-H list of text labels.
    """

    error_checking.assert_is_geq_numpy_array(tick_values_km_agl, 0.)
    error_checking.assert_is_numpy_array(tick_values_km_agl, num_dimensions=1)
    error_checking.assert_is_boolean(use_log_scale)

    num_ticks = len(tick_values_km_agl)
    tick_strings = ['foo'] * num_ticks

    for i in range(num_ticks):
        try:
            this_order_of_magnitude = int(numpy.floor(
                numpy.log10(tick_values_km_agl[i])
            ))
        except OverflowError:
            this_order_of_magnitude = -1

        if this_order_of_magnitude >= 0:
            this_num_decimal_places = 1
        else:
            this_num_decimal_places = numpy.absolute(this_order_of_magnitude)

            if not use_log_scale:
                this_num_decimal_places += 1

        this_num_decimal_places = numpy.minimum(this_num_decimal_places, 2)

        this_format_string = (
            '{0:.' + '{0:d}'.format(this_num_decimal_places) + 'f}'
        )
        tick_strings[i] = this_format_string.format(tick_values_km_agl[i])

    return tick_strings


def plot_predictors(
        example_dict, example_index, plot_ice, use_log_scale,
        line_width=DEFAULT_LINE_WIDTH, line_style='solid', handle_dict=None):
    """Plots predictors (temperature, spec humidity, liquid/ice-water content).

    :param example_dict: See doc for `example_io.read_file`.
    :param example_index: Will plot the [i]th example, where
        i = `example_index`.
    :param plot_ice: Boolean flag.  If True, will plot ice-water content.  If
        False, will plot liquid-water content.
    :param use_log_scale: Boolean flag.  If True, will plot height (y-axis) in
        logarithmic scale.  If False, will plot height in linear scale.
    :param line_width: Line width.
    :param line_style: Line style (in any format accepted by matplotlib).
    :param handle_dict: See output doc.  If None, will create new figure on the
        fly.
    :return: handle_dict: Dictionary with the following keys.
    handle_dict['figure_object']: Figure handle (instance of
        `matplotlib.figure.Figure`).
    handle_dict['temperature_axes_object']: Handle for temperature axes
        (instance of `matplotlib.axes._subplots.AxesSubplot`).
    handle_dict['humidity_axes_object']: Handle for humidity axes (same object
        type).
    handle_dict['water_content_axes_object']: Handle for water-content axes
        (same type).
    """

    _set_font_size(FANCY_FONT_SIZE)

    error_checking.assert_is_integer(example_index)
    error_checking.assert_is_geq(example_index, 0)
    error_checking.assert_is_boolean(plot_ice)
    error_checking.assert_is_boolean(use_log_scale)

    if handle_dict is None:
        figure_object, temperature_axes_object = pyplot.subplots(
            1, 1, figsize=(FANCY_FIGURE_WIDTH_INCHES, FANCY_FIGURE_HEIGHT_INCHES)
        )

        if use_log_scale:
            pyplot.yscale('log')

        figure_object.subplots_adjust(bottom=0.75)
        humidity_axes_object = temperature_axes_object.twiny()
        water_content_axes_object = temperature_axes_object.twiny()

        water_content_axes_object.spines['top'].set_position(('axes', 1.125))
        _make_spines_invisible(water_content_axes_object)
        water_content_axes_object.spines['top'].set_visible(True)
    else:
        figure_object = handle_dict[FIGURE_HANDLE_KEY]
        temperature_axes_object = handle_dict[TEMPERATURE_HANDLE_KEY]
        humidity_axes_object = handle_dict[HUMIDITY_HANDLE_KEY]
        water_content_axes_object = handle_dict[WATER_CONTENT_HANDLE_KEY]

    heights_km_agl = METRES_TO_KM * example_dict[example_io.HEIGHTS_KEY]

    temperatures_kelvins = example_io.get_field_from_dict(
        example_dict=example_dict, field_name=example_io.TEMPERATURE_NAME
    )[example_index, ...]

    temperatures_deg_c = temperature_conv.kelvins_to_celsius(
        temperatures_kelvins
    )
    temperature_axes_object.plot(
        temperatures_deg_c, heights_km_agl, color=TEMPERATURE_COLOUR,
        linewidth=line_width, linestyle=line_style
    )

    specific_humidities_kg_kg01 = example_io.get_field_from_dict(
        example_dict=example_dict, field_name=example_io.SPECIFIC_HUMIDITY_NAME
    )[example_index, ...]

    humidity_axes_object.plot(
        KG_TO_GRAMS * specific_humidities_kg_kg01, heights_km_agl,
        color=HUMIDITY_COLOUR, linewidth=line_width, linestyle=line_style
    )

    if plot_ice:
        iwc_values_kg_m03 = example_io.get_field_from_dict(
            example_dict=example_dict,
            field_name=example_io.ICE_WATER_CONTENT_NAME
        )[example_index, ...]

        print(numpy.max(iwc_values_kg_m03))

        water_content_axes_object.plot(
            KG_TO_MILLIGRAMS * iwc_values_kg_m03, heights_km_agl,
            color=WATER_CONTENT_COLOUR, linewidth=line_width,
            linestyle=line_style
        )

        if handle_dict is None:
            water_content_axes_object.set_xlabel(
                r'Ice-water content (mg m$^{-3}$)'
            )
    else:
        lwc_values_kg_m03 = example_io.get_field_from_dict(
            example_dict=example_dict,
            field_name=example_io.LIQUID_WATER_CONTENT_NAME
        )[example_index, ...]

        print(lwc_values_kg_m03)

        water_content_axes_object.plot(
            KG_TO_GRAMS * lwc_values_kg_m03, heights_km_agl,
            color=WATER_CONTENT_COLOUR, linewidth=line_width,
            linestyle=line_style
        )

        if handle_dict is None:
            water_content_axes_object.set_xlabel(
                r'Liquid-water content (g m$^{-3}$)'
            )

    if handle_dict is None:
        temperature_axes_object.set_ylabel('Height (km AGL)')
        temperature_axes_object.set_xlabel(r'Temperature ($^{\circ}$C)')
        humidity_axes_object.set_xlabel(r'Specific humidity (g kg$^{-1}$)')

        temperature_axes_object.set_ylim([
            numpy.min(heights_km_agl), numpy.max(heights_km_agl)
        ])

        height_strings = create_height_labels(
            tick_values_km_agl=temperature_axes_object.get_yticks(),
            use_log_scale=use_log_scale
        )
        temperature_axes_object.set_yticklabels(height_strings)

        temperature_axes_object.xaxis.label.set_color(TEMPERATURE_COLOUR)
        humidity_axes_object.xaxis.label.set_color(HUMIDITY_COLOUR)
        water_content_axes_object.xaxis.label.set_color(WATER_CONTENT_COLOUR)

        tick_mark_dict = dict(size=4, width=1.5)
        temperature_axes_object.tick_params(axis='y', **tick_mark_dict)

        temperature_axes_object.tick_params(
            axis='x', colors=TEMPERATURE_COLOUR, **tick_mark_dict
        )
        humidity_axes_object.tick_params(
            axis='x', colors=HUMIDITY_COLOUR, **tick_mark_dict
        )
        water_content_axes_object.tick_params(
            axis='x', colors=WATER_CONTENT_COLOUR, **tick_mark_dict
        )

    return {
        FIGURE_HANDLE_KEY: figure_object,
        TEMPERATURE_HANDLE_KEY: temperature_axes_object,
        HUMIDITY_HANDLE_KEY: humidity_axes_object,
        WATER_CONTENT_HANDLE_KEY: water_content_axes_object
    }


def plot_targets(
        example_dict, example_index, use_log_scale,
        line_width=DEFAULT_LINE_WIDTH, line_style='solid', handle_dict=None):
    """Plots targets (shortwave upwelling flux, down flux, heating rate).

    :param example_dict: See doc for `plot_predictors`.
    :param example_index: Same.
    :param use_log_scale: Same.
    :param line_width: Same.
    :param line_style: Same.
    :param handle_dict: Same.
    :return: handle_dict: Dictionary with the following keys.
    handle_dict['figure_object']: Figure handle (instance of
        `matplotlib.figure.Figure`).
    handle_dict['heating_rate_axes_object']: Handle for heating-rate axes
        (instance of `matplotlib.axes._subplots.AxesSubplot`).
    handle_dict['down_flux_axes_object']: Handle for downwelling-flux axes
        (same object type).
    handle_dict['up_flux_axes_object']: Handle for upwelling-flux axes
        (same type).
    """

    _set_font_size(FANCY_FONT_SIZE)

    error_checking.assert_is_integer(example_index)
    error_checking.assert_is_geq(example_index, 0)
    error_checking.assert_is_boolean(use_log_scale)

    if handle_dict is None:
        figure_object, heating_rate_axes_object = pyplot.subplots(
            1, 1, figsize=(FANCY_FIGURE_WIDTH_INCHES, FANCY_FIGURE_HEIGHT_INCHES)
        )

        if use_log_scale:
            pyplot.yscale('log')

        figure_object.subplots_adjust(bottom=0.75)
        down_flux_axes_object = heating_rate_axes_object.twiny()
        up_flux_axes_object = heating_rate_axes_object.twiny()

        up_flux_axes_object.spines['top'].set_position(('axes', 1.125))
        _make_spines_invisible(up_flux_axes_object)
        up_flux_axes_object.spines['top'].set_visible(True)
    else:
        figure_object = handle_dict[FIGURE_HANDLE_KEY]
        heating_rate_axes_object = handle_dict[HEATING_RATE_HANDLE_KEY]
        down_flux_axes_object = handle_dict[DOWN_FLUX_HANDLE_KEY]
        up_flux_axes_object = handle_dict[UP_FLUX_HANDLE_KEY]

    heights_km_agl = METRES_TO_KM * example_dict[example_io.HEIGHTS_KEY]

    heating_rates_kelvins_day01 = example_io.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_io.SHORTWAVE_HEATING_RATE_NAME
    )[example_index, ...]

    heating_rate_axes_object.plot(
        heating_rates_kelvins_day01, heights_km_agl,
        color=HEATING_RATE_COLOUR, linewidth=line_width, linestyle=line_style
    )

    downwelling_fluxes_w_m02 = example_io.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_io.SHORTWAVE_DOWN_FLUX_NAME
    )[example_index, ...]

    down_flux_axes_object.plot(
        downwelling_fluxes_w_m02, heights_km_agl,
        color=DOWNWELLING_FLUX_COLOUR, linewidth=line_width,
        linestyle=line_style
    )

    upwelling_fluxes_w_m02 = example_io.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_io.SHORTWAVE_UP_FLUX_NAME
    )[example_index, ...]

    up_flux_axes_object.plot(
        upwelling_fluxes_w_m02, heights_km_agl,
        color=UPWELLING_FLUX_COLOUR, linewidth=line_width, linestyle=line_style
    )

    if handle_dict is None:
        heating_rate_axes_object.set_ylabel('Height (km AGL)')
        heating_rate_axes_object.set_xlabel(
            r'Shortwave heating rate (K day$^{-1}$)'
        )
        down_flux_axes_object.set_xlabel(
            r'Downwelling shortwave flux (W m$^{-2}$)'
        )
        up_flux_axes_object.set_xlabel(
            r'Upwelling shortwave flux (W m$^{-2}$)'
        )

        heating_rate_axes_object.set_ylim([
            numpy.min(heights_km_agl), numpy.max(heights_km_agl)
        ])

        height_strings = create_height_labels(
            tick_values_km_agl=heating_rate_axes_object.get_yticks(),
            use_log_scale=use_log_scale
        )
        heating_rate_axes_object.set_yticklabels(height_strings)

        heating_rate_axes_object.xaxis.label.set_color(HEATING_RATE_COLOUR)
        down_flux_axes_object.xaxis.label.set_color(DOWNWELLING_FLUX_COLOUR)
        up_flux_axes_object.xaxis.label.set_color(UPWELLING_FLUX_COLOUR)

        tick_mark_dict = dict(size=4, width=1.5)
        heating_rate_axes_object.tick_params(axis='y', **tick_mark_dict)

        heating_rate_axes_object.tick_params(
            axis='x', colors=HEATING_RATE_COLOUR, **tick_mark_dict
        )
        down_flux_axes_object.tick_params(
            axis='x', colors=DOWNWELLING_FLUX_COLOUR, **tick_mark_dict
        )
        up_flux_axes_object.tick_params(
            axis='x', colors=UPWELLING_FLUX_COLOUR, **tick_mark_dict
        )

    these_fluxes_w_m02 = numpy.array([
        down_flux_axes_object.get_xlim()[1],
        up_flux_axes_object.get_xlim()[1],
        numpy.max(downwelling_fluxes_w_m02), numpy.max(upwelling_fluxes_w_m02)
    ])

    max_flux_w_m02 = 1.025 * numpy.max(these_fluxes_w_m02)

    down_flux_axes_object.set_xlim(0, max_flux_w_m02)
    up_flux_axes_object.set_xlim(0, max_flux_w_m02)
    heating_rate_axes_object.set_xlim(left=0)

    return {
        FIGURE_HANDLE_KEY: figure_object,
        HEATING_RATE_HANDLE_KEY: heating_rate_axes_object,
        DOWN_FLUX_HANDLE_KEY: down_flux_axes_object,
        UP_FLUX_HANDLE_KEY: up_flux_axes_object
    }


def plot_one_variable(
        values, heights_m_agl, use_log_scale, line_colour,
        line_width=DEFAULT_LINE_WIDTH, line_style='solid', figure_object=None):
    """Plots vertical profile of one variable.

    H = number of heights

    :param values: length-H numpy array with values of given variable.
    :param heights_m_agl: length-H numpy array of heights (metres above ground
        level).
    :param use_log_scale: See doc for `plot_predictors`.
    :param line_colour: Line colour (in any format accepted by matplotlib).
    :param line_width: See doc for `plot_predictors`.
    :param line_style: Same.
    :param figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).  If None, will create new figure on the
        fly.
    :return: figure_object: See input doc.
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    _set_font_size(SIMPLE_FONT_SIZE)

    error_checking.assert_is_numpy_array(values, num_dimensions=1)
    error_checking.assert_is_boolean(use_log_scale)

    num_heights = len(values)
    expected_dim = numpy.array([num_heights], dtype=int)

    error_checking.assert_is_numpy_array_without_nan(heights_m_agl)
    error_checking.assert_is_numpy_array(
        heights_m_agl, exact_dimensions=expected_dim
    )

    was_figure_object_input = figure_object is not None

    if was_figure_object_input:
        axes_object = figure_object.axes[0]
    else:
        figure_object, axes_object = pyplot.subplots(
            1, 1,
            figsize=(SIMPLE_FIGURE_WIDTH_INCHES, SIMPLE_FIGURE_HEIGHT_INCHES)
        )

        if use_log_scale:
            pyplot.yscale('log')

    heights_km_agl = heights_m_agl * METRES_TO_KM
    min_height_km_agl = numpy.min(heights_km_agl)
    max_height_km_agl = numpy.max(heights_km_agl)

    axes_object.plot(
        values, heights_km_agl, color=line_colour, linestyle=line_style,
        linewidth=line_width
    )

    x_max = numpy.maximum(
        axes_object.get_xlim()[1], numpy.max(values)
    )
    axes_object.set_xlim(0, x_max)

    if was_figure_object_input:
        return figure_object, axes_object

    if use_log_scale:
        axes_object.set_ylim(min_height_km_agl, max_height_km_agl)
    else:
        axes_object.set_ylim(0, max_height_km_agl)

    height_strings = create_height_labels(
        tick_values_km_agl=axes_object.get_yticks(),
        use_log_scale=use_log_scale
    )
    axes_object.set_yticklabels(height_strings)
    axes_object.set_ylabel('Height (km AGL)')

    return figure_object, axes_object
