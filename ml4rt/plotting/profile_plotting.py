"""Methods for plotting vertical profiles."""

import numpy
import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import temperature_conversions as temperature_conv
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io

METRES_TO_KM = 0.001
KG_TO_GRAMS = 1000.
KG_TO_MILLIGRAMS = 1e6
DAYS_TO_SECONDS = 86400.

DEFAULT_LINE_WIDTH = 2
FIGURE_WIDTH_INCHES = 8
FIGURE_HEIGHT_INCHES = 40

HEATING_RATE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
# DOWN_FLUX_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
# UP_FLUX_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
FLUX_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255

TEMPERATURE_COLOUR = numpy.array([51, 160, 44], dtype=float) / 255
HUMIDITY_COLOUR = numpy.array([178, 223, 138], dtype=float) / 255
LWC_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
IWC_COLOUR = numpy.array([166, 206, 227], dtype=float) / 255

FONT_SIZE = 12
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _make_spines_invisible(axes_object):
    """Makes spines along axis invisible.

    :param axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    axes_object.set_frame_on(True)
    axes_object.patch.set_visible(False)

    for this_spine_object in axes_object.spines.values():
        this_spine_object.set_visible(False)


def plot_predictors(example_dict, example_index, line_width=DEFAULT_LINE_WIDTH):
    """Plots predictors (temperature, spec humidity, liquid/ice water contents).

    :param example_dict: See doc for `example_io.read_file`.
    :param example_index: Will plot the [i]th example, where
        i = `example_index`.
    :param line_width: Line width.
    :return: figure_object: Figure handle (instance of
        `matplotlib.figure.Figure`).
    :return: axes_object: Axes handle (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    """

    error_checking.assert_is_integer(example_index)
    error_checking.assert_is_geq(example_index, 0)

    figure_object, temperature_axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    figure_object.subplots_adjust(bottom=0.75)
    humidity_axes_object = temperature_axes_object.twiny()
    lwc_axes_object = temperature_axes_object.twiny()
    iwc_axes_object = temperature_axes_object.twiny()

    humidity_axes_object.xaxis.set_ticks_position('bottom')
    humidity_axes_object.xaxis.set_label_position('bottom')
    humidity_axes_object.spines['bottom'].set_position(('axes', -0.125))
    _make_spines_invisible(humidity_axes_object)
    humidity_axes_object.spines['bottom'].set_visible(True)

    iwc_axes_object.spines['top'].set_position(('axes', 1.125))
    _make_spines_invisible(iwc_axes_object)
    iwc_axes_object.spines['top'].set_visible(True)

    heights_km_agl = METRES_TO_KM * example_dict[example_io.HEIGHTS_KEY]

    temperatures_kelvins = example_io.get_field_from_dict(
        example_dict=example_dict, field_name=example_io.TEMPERATURE_NAME
    )[example_index, ...]

    temperatures_deg_c = temperature_conv.kelvins_to_celsius(
        temperatures_kelvins
    )
    temperature_axes_object.plot(
        temperatures_deg_c, heights_km_agl, color=TEMPERATURE_COLOUR,
        linewidth=line_width
    )

    specific_humidities_kg_kg01 = example_io.get_field_from_dict(
        example_dict=example_dict, field_name=example_io.SPECIFIC_HUMIDITY_NAME
    )[example_index, ...]

    humidity_axes_object.plot(
        KG_TO_GRAMS * specific_humidities_kg_kg01, heights_km_agl,
        color=HUMIDITY_COLOUR, linewidth=line_width
    )

    lwc_values_kg_m02 = example_io.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_io.LIQUID_WATER_CONTENT_NAME
    )[example_index, ...]

    print(lwc_values_kg_m02)

    lwc_axes_object.plot(
        KG_TO_GRAMS * lwc_values_kg_m02, heights_km_agl,
        color=LWC_COLOUR, linewidth=line_width
    )

    iwc_values_kg_m02 = example_io.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_io.ICE_WATER_CONTENT_NAME
    )[example_index, ...]

    print(numpy.max(iwc_values_kg_m02))

    iwc_axes_object.plot(
        KG_TO_MILLIGRAMS * iwc_values_kg_m02, heights_km_agl,
        color=IWC_COLOUR, linewidth=line_width
    )

    temperature_axes_object.set_ylabel('Height (km AGL)')
    temperature_axes_object.set_xlabel(r'Temperature ($^{\circ}$C)')
    humidity_axes_object.set_xlabel(r'Specific humidity (g kg$^{-1}$)')
    lwc_axes_object.set_xlabel(r'Liquid-water content (g m$^{-2}$)')
    iwc_axes_object.set_xlabel(r'Ice-water content (mg m$^{-2}$)')

    temperature_axes_object.set_ylim([
        numpy.min(heights_km_agl), numpy.max(heights_km_agl)
    ])

    temperature_axes_object.xaxis.label.set_color(TEMPERATURE_COLOUR)
    humidity_axes_object.xaxis.label.set_color(HUMIDITY_COLOUR)
    lwc_axes_object.xaxis.label.set_color(LWC_COLOUR)
    iwc_axes_object.xaxis.label.set_color(IWC_COLOUR)

    tick_mark_dict = dict(size=4, width=1.5)
    temperature_axes_object.tick_params(axis='y', **tick_mark_dict)

    temperature_axes_object.tick_params(
        axis='x', colors=TEMPERATURE_COLOUR, **tick_mark_dict
    )
    humidity_axes_object.tick_params(
        axis='x', colors=HUMIDITY_COLOUR, **tick_mark_dict
    )
    lwc_axes_object.tick_params(
        axis='x', colors=LWC_COLOUR, **tick_mark_dict
    )
    iwc_axes_object.tick_params(
        axis='x', colors=IWC_COLOUR, **tick_mark_dict
    )

    return figure_object, temperature_axes_object


def plot_targets(example_dict, example_index, line_width=DEFAULT_LINE_WIDTH):
    """Plots targets (shortwave upwelling flux, down flux, heating rate).

    :param example_dict: See doc for `plot_predictors`.
    :param example_index: Same.
    :param line_width: Same.
    :return: figure_object: Same.
    :return: axes_object: Same.
    """

    error_checking.assert_is_integer(example_index)
    error_checking.assert_is_geq(example_index, 0)

    figure_object, heating_rate_axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    figure_object.subplots_adjust(bottom=0.75)
    flux_axes_object = heating_rate_axes_object.twiny()

    # flux_axes_object.spines['top'].set_position(('axes', 1.125))
    # _make_spines_invisible(flux_axes_object)
    # flux_axes_object.spines['top'].set_visible(True)

    heights_km_agl = METRES_TO_KM * example_dict[example_io.HEIGHTS_KEY]

    heating_rates_kelvins_s01 = example_io.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_io.SHORTWAVE_HEATING_RATE_NAME
    )[example_index, ...]

    heating_rate_axes_object.plot(
        DAYS_TO_SECONDS * heating_rates_kelvins_s01, heights_km_agl,
        color=HEATING_RATE_COLOUR, linewidth=line_width, linestyle='solid'
    )

    downwelling_fluxes_w_m02 = example_io.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_io.SHORTWAVE_DOWN_FLUX_NAME
    )[example_index, ...]

    flux_axes_object.plot(
        downwelling_fluxes_w_m02, heights_km_agl,
        color=FLUX_COLOUR, linewidth=line_width, linestyle='solid'
    )

    upwelling_fluxes_w_m02 = example_io.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_io.SHORTWAVE_UP_FLUX_NAME
    )[example_index, ...]

    flux_axes_object.plot(
        upwelling_fluxes_w_m02, heights_km_agl,
        color=FLUX_COLOUR, linewidth=line_width, linestyle='dashed'
    )

    heating_rate_axes_object.set_ylabel('Height (km AGL)')
    heating_rate_axes_object.set_xlabel(
        r'Shortwave heating rate ($^{\circ}$C day$^{-1}$)'
    )
    flux_axes_object.set_xlabel(
        r'Shortwave flux (solid = down; dashed = up; W m$^{-2}$)'
    )

    heating_rate_axes_object.set_ylim([
        numpy.min(heights_km_agl), numpy.max(heights_km_agl)
    ])

    heating_rate_axes_object.xaxis.label.set_color(HEATING_RATE_COLOUR)
    flux_axes_object.xaxis.label.set_color(FLUX_COLOUR)

    tick_mark_dict = dict(size=4, width=1.5)
    heating_rate_axes_object.tick_params(axis='y', **tick_mark_dict)

    heating_rate_axes_object.tick_params(
        axis='x', colors=HEATING_RATE_COLOUR, **tick_mark_dict
    )
    flux_axes_object.tick_params(
        axis='x', colors=FLUX_COLOUR, **tick_mark_dict
    )

    return figure_object, heating_rate_axes_object
