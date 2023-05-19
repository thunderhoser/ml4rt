"""Methods for plotting vertical profiles."""

import os
import sys
import copy
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

import temperature_conversions as temperature_conv
import error_checking
import example_utils

METRES_TO_KM = 0.001
KG_TO_GRAMS = 1000.
PASCALS_TO_MB = 0.01
KG_TO_MILLIGRAMS = 1e6
RADIANS_TO_DEGREES = 180. / numpy.pi
METRES_TO_MICRONS = 1e6

OPACITY_FOR_UNCERTAINTY = 0.6

FIGURE_HANDLE_KEY = 'figure_object'
AXES_OBJECTS_KEY = 'axes_objects'
HEATING_RATE_HANDLE_KEY = 'heating_rate_axes_object'
DOWN_FLUX_HANDLE_KEY = 'down_flux_axes_object'
UP_FLUX_HANDLE_KEY = 'up_flux_axes_object'

PREDICTOR_NAME_TO_VERBOSE = {
    example_utils.RELATIVE_HUMIDITY_NAME: r'Relative humidity',
    example_utils.SPECIFIC_HUMIDITY_NAME: r'Specific humidity (g kg$^{-1}$)',
    example_utils.MIXING_RATIO_NAME: r'Mixing ratio (g kg$^{-1}$)',
    example_utils.DEWPOINT_NAME: r'Dewpoint ($^{\circ}$C)',
    example_utils.WATER_VAPOUR_PATH_NAME:
        r'Downward water-vapour path (kg m$^{-2}$)',
    example_utils.UPWARD_WATER_VAPOUR_PATH_NAME:
        r'Upward water-vapour path (kg m$^{-2}$)',
    example_utils.LIQUID_WATER_CONTENT_NAME:
        r'Liquid-water content (g m$^{-3}$)',
    example_utils.LIQUID_WATER_PATH_NAME:
        r'Downward liquid-water path (g m$^{-2}$)',
    example_utils.UPWARD_LIQUID_WATER_PATH_NAME:
        r'Upward liquid-water path (g m$^{-2}$)',
    example_utils.LIQUID_EFF_RADIUS_NAME: r'Liquid effective radius ($\mu$m)',
    example_utils.ICE_WATER_CONTENT_NAME: r'Ice-water content (g m$^{-3}$)',
    example_utils.ICE_WATER_PATH_NAME: r'Downward ice-water path (mg m$^{-2}$)',
    example_utils.UPWARD_ICE_WATER_PATH_NAME:
        r'Upward ice-water path (mg m$^{-2}$)',
    example_utils.ICE_EFF_RADIUS_NAME: r'Ice effective radius ($\mu$m)',
    example_utils.TEMPERATURE_NAME: r'Temperature ($^{\circ}$C)',
    example_utils.PRESSURE_NAME: 'Pressure (mb)',
    example_utils.HEIGHT_THICKNESS_NAME: 'Height thickness (m)',
    example_utils.PRESSURE_THICKNESS_NAME: 'Pressure thickness (mb)',
    example_utils.O3_MIXING_RATIO_NAME: r'O$_3$ mixing ratio (mg kg$^{-1}$)',
    example_utils.CO2_CONCENTRATION_NAME: r'CO$_2$ concentration (ppmv)',
    example_utils.N2O_CONCENTRATION_NAME: r'N$_2$O concentration (ppmv)',
    example_utils.CH4_CONCENTRATION_NAME: r'CH$_4$ concentration (ppmv)',
    example_utils.ZENITH_ANGLE_NAME: r'Zenith angle ($^{\circ}$)',
    example_utils.SURFACE_TEMPERATURE_NAME: r'Surface temp ($^{\circ}$C)',
    example_utils.SURFACE_EMISSIVITY_NAME: 'Surface emissivity',
    example_utils.ALBEDO_NAME: 'Albedo',
    example_utils.LATITUDE_NAME: r'Latitude ($^{\circ}$N)',
    example_utils.LONGITUDE_NAME: r'Longitude ($^{\circ}$E)',
    example_utils.COLUMN_LIQUID_WATER_PATH_NAME:
        r'Column liquid-water path (g m$^{-2}$)',
    example_utils.COLUMN_ICE_WATER_PATH_NAME:
        r'Column ice-water path (mg m$^{-2}$)',
    example_utils.AEROSOL_EXTINCTION_NAME: r'Aerosol extinction (km$^{-1}$)',
    example_utils.AEROSOL_ALBEDO_NAME: 'Aerosol single-scattering albedo',
    example_utils.AEROSOL_ASYMMETRY_PARAM_NAME: 'Aerosol asymmetry param'
}

PREDICTOR_NAME_TO_CONV_FACTOR = {
    example_utils.RELATIVE_HUMIDITY_NAME: 1.,
    example_utils.SPECIFIC_HUMIDITY_NAME: KG_TO_GRAMS,
    example_utils.MIXING_RATIO_NAME: KG_TO_GRAMS,
    example_utils.WATER_VAPOUR_PATH_NAME: 1.,
    example_utils.UPWARD_WATER_VAPOUR_PATH_NAME: 1.,
    example_utils.LIQUID_WATER_CONTENT_NAME: KG_TO_GRAMS,
    example_utils.LIQUID_WATER_PATH_NAME: KG_TO_GRAMS,
    example_utils.UPWARD_LIQUID_WATER_PATH_NAME: KG_TO_GRAMS,
    example_utils.LIQUID_EFF_RADIUS_NAME: METRES_TO_MICRONS,
    example_utils.ICE_WATER_CONTENT_NAME: KG_TO_GRAMS,
    example_utils.ICE_WATER_PATH_NAME: KG_TO_MILLIGRAMS,
    example_utils.UPWARD_ICE_WATER_PATH_NAME: KG_TO_MILLIGRAMS,
    example_utils.ICE_EFF_RADIUS_NAME: METRES_TO_MICRONS,
    example_utils.PRESSURE_NAME: PASCALS_TO_MB,
    example_utils.HEIGHT_THICKNESS_NAME: 1.,
    example_utils.PRESSURE_THICKNESS_NAME: PASCALS_TO_MB,
    example_utils.O3_MIXING_RATIO_NAME: KG_TO_MILLIGRAMS,
    example_utils.CO2_CONCENTRATION_NAME: 1.,
    example_utils.N2O_CONCENTRATION_NAME: 1.,
    example_utils.CH4_CONCENTRATION_NAME: 1.,
    example_utils.ZENITH_ANGLE_NAME: RADIANS_TO_DEGREES,
    example_utils.SURFACE_EMISSIVITY_NAME: 1.,
    example_utils.ALBEDO_NAME: 1.,
    example_utils.LATITUDE_NAME: 1.,
    example_utils.LONGITUDE_NAME: 1.,
    example_utils.COLUMN_LIQUID_WATER_PATH_NAME: KG_TO_GRAMS,
    example_utils.COLUMN_ICE_WATER_PATH_NAME: KG_TO_MILLIGRAMS,
    example_utils.AEROSOL_EXTINCTION_NAME: 1000.,
    example_utils.AEROSOL_ALBEDO_NAME: 1.,
    example_utils.AEROSOL_ASYMMETRY_PARAM_NAME: 1.
}

DEFAULT_LINE_WIDTH = 2

FANCY_FONT_SIZE = 20
FANCY_FIGURE_WIDTH_INCHES = 8
FANCY_FIGURE_HEIGHT_INCHES = 40

SIMPLE_FONT_SIZE = 50
SIMPLE_FIGURE_WIDTH_INCHES = 15
SIMPLE_FIGURE_HEIGHT_INCHES = 15

BLACK_COLOUR = numpy.full(3, 0.)
HEATING_RATE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
DOWNWELLING_FLUX_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
UPWELLING_FLUX_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255


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
    :param use_log_scale: Boolean flag.  If True, will assume that height axis
        is logarithmic.
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
        example_dict, example_index, predictor_names, predictor_colours,
        predictor_line_widths, predictor_line_styles, use_log_scale,
        include_units=True, handle_dict=None):
    """Plots several predictors on the same set of axes.

    P = number of predictors to plot (must all be profiles)

    :param example_dict: See doc for `example_io.read_file`.
    :param example_index: Will plot the [i]th example, where
        i = `example_index`.
    :param predictor_names: length-P list with names of predictors to plot.
    :param predictor_colours: length-P list of colours (each colour in any
        format accepted by matplotlib).
    :param predictor_line_widths: length-P numpy array of line widths.
    :param predictor_line_styles: length-P list of line styles (each style in
        any format accepted by matplotlib).
    :param use_log_scale: Boolean flag.  If True, will plot height (y-axis) in
        logarithmic scale.  If False, will plot height in linear scale.
    :param include_units: Boolean flag.  If True, axis titles will include units
        and values will be converted from default to plotting units.  If False,
        axis titles will *not* include units and this method will *not* convert
        units.
    :param handle_dict: See output doc.  If None, will create new figure on the
        fly.
    :return: handle_dict: Dictionary with the following keys.
    handle_dict['figure_object']: Figure handle (instance of
        `matplotlib.figure.Figure`).
    handle_dict['axes_objects']: length-P list of axes handles (each an instance
        of `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Check input args.
    error_checking.assert_is_integer(example_index)
    error_checking.assert_is_geq(example_index, 0)
    error_checking.assert_is_boolean(use_log_scale)
    error_checking.assert_is_boolean(include_units)

    error_checking.assert_is_string_list(predictor_names)
    num_predictors = len(predictor_names)
    error_checking.assert_is_leq(num_predictors, 4)

    for k in range(num_predictors):
        pass
        # assert predictor_names[k] in example_utils.ALL_PREDICTOR_NAMES
        # assert predictor_names[k] in example_utils.ALL_VECTOR_PREDICTOR_NAMES

    assert len(predictor_colours) == num_predictors
    assert len(predictor_line_widths) == num_predictors
    assert len(predictor_line_styles) == num_predictors

    # Housekeeping.
    _set_font_size(FANCY_FONT_SIZE)

    if handle_dict is None:
        figure_object, first_axes_object = pyplot.subplots(
            1, 1,
            figsize=(FANCY_FIGURE_WIDTH_INCHES, FANCY_FIGURE_HEIGHT_INCHES)
        )

        axes_objects = [first_axes_object]
        figure_object.subplots_adjust(bottom=0.75)

        if use_log_scale:
            pyplot.yscale('log')

        for k in range(1, num_predictors):
            axes_objects.append(axes_objects[0].twiny())

            if k == 2:
                axes_objects[k].spines['top'].set_position(('axes', 1.15))
                _make_spines_invisible(axes_objects[k])
                axes_objects[k].spines['top'].set_visible(True)

            if k == 3:
                axes_objects[k].xaxis.set_ticks_position('bottom')
                axes_objects[k].xaxis.set_label_position('bottom')
                axes_objects[k].spines['bottom'].set_position(('axes', -0.15))
                _make_spines_invisible(axes_objects[k])
                axes_objects[k].spines['bottom'].set_visible(True)
    else:
        figure_object = handle_dict[FIGURE_HANDLE_KEY]
        axes_objects = handle_dict[AXES_OBJECTS_KEY]

    try:
        heights_km_agl = METRES_TO_KM * example_utils.get_field_from_dict(
            example_dict=example_dict, field_name=example_utils.HEIGHT_NAME
        )[example_index, :]
    except:
        heights_km_agl = METRES_TO_KM * example_dict[example_utils.HEIGHTS_KEY]

    tick_mark_dict = dict(size=4, width=1.5)

    for k in range(num_predictors):
        if predictor_names[k] in example_utils.ALL_SCALAR_PREDICTOR_NAMES:

            # TODO(thunderhoser): This is a HACK to deal with saliency maps.
            j = example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY].index(
                predictor_names[k]
            )

            these_predictor_values = numpy.full(
                len(heights_km_agl),
                example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY][
                    example_index, j
                ]
            )
        else:
            these_predictor_values = example_utils.get_field_from_dict(
                example_dict=example_dict, field_name=predictor_names[k]
            )[example_index, ...]

        if include_units:
            if predictor_names[k] in [
                    example_utils.TEMPERATURE_NAME,
                    example_utils.SURFACE_TEMPERATURE_NAME,
                    example_utils.DEWPOINT_NAME
            ]:
                these_predictor_values = temperature_conv.kelvins_to_celsius(
                    these_predictor_values
                )
            else:
                these_predictor_values = (
                    PREDICTOR_NAME_TO_CONV_FACTOR[predictor_names[k]] *
                    these_predictor_values
                )

        axes_objects[k].plot(
            these_predictor_values, heights_km_agl, color=predictor_colours[k],
            linewidth=predictor_line_widths[k],
            linestyle=predictor_line_styles[k]
        )

        x_label_string = copy.deepcopy(
            PREDICTOR_NAME_TO_VERBOSE[predictor_names[k]]
        )
        if not include_units:
            x_label_string = x_label_string.split(' (')[0]

        axes_objects[k].set_xlabel(x_label_string)
        axes_objects[k].xaxis.label.set_color(predictor_colours[k])
        axes_objects[k].tick_params(
            axis='x', colors=predictor_colours[k], **tick_mark_dict
        )

    axes_objects[0].set_ylabel('Height (km AGL)')
    axes_objects[0].set_ylim([
        numpy.min(heights_km_agl), numpy.max(heights_km_agl)
    ])

    height_strings = create_height_labels(
        tick_values_km_agl=axes_objects[0].get_yticks(),
        use_log_scale=use_log_scale
    )
    axes_objects[0].set_yticklabels(height_strings)
    axes_objects[0].tick_params(axis='y', **tick_mark_dict)

    return {
        FIGURE_HANDLE_KEY: figure_object,
        AXES_OBJECTS_KEY: axes_objects
    }


def plot_targets(
        example_dict, example_index, use_log_scale, for_shortwave=True,
        line_width=DEFAULT_LINE_WIDTH, line_style='solid',
        handle_dict=None):
    """Plots targets (upwelling flux, downwelling flux, heating rate).

    :param example_dict: See doc for `plot_predictors`.
    :param example_index: Same.
    :param use_log_scale: Same.
    :param for_shortwave: Boolean flag.  If True (False), will plot targets for
        shortwave (longwave) flux.
    :param line_width: See doc for `plot_predictors`.
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
    error_checking.assert_is_boolean(for_shortwave)

    if handle_dict is None:
        figure_object, heating_rate_axes_object = pyplot.subplots(
            1, 1,
            figsize=(FANCY_FIGURE_WIDTH_INCHES, FANCY_FIGURE_HEIGHT_INCHES)
        )

        if use_log_scale:
            pyplot.yscale('log')

        figure_object.subplots_adjust(bottom=0.75)
        down_flux_axes_object = heating_rate_axes_object.twiny()
        up_flux_axes_object = heating_rate_axes_object.twiny()

        up_flux_axes_object.spines['top'].set_position(('axes', 1.15))
        _make_spines_invisible(up_flux_axes_object)
        up_flux_axes_object.spines['top'].set_visible(True)
    else:
        figure_object = handle_dict[FIGURE_HANDLE_KEY]
        heating_rate_axes_object = handle_dict[HEATING_RATE_HANDLE_KEY]
        down_flux_axes_object = handle_dict[DOWN_FLUX_HANDLE_KEY]
        up_flux_axes_object = handle_dict[UP_FLUX_HANDLE_KEY]

    try:
        heights_km_agl = METRES_TO_KM * example_utils.get_field_from_dict(
            example_dict=example_dict, field_name=example_utils.HEIGHT_NAME
        )[example_index, :]
    except:
        heights_km_agl = METRES_TO_KM * example_dict[example_utils.HEIGHTS_KEY]

    heating_rates_kelvins_day01 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=(
            example_utils.SHORTWAVE_HEATING_RATE_NAME if for_shortwave
            else example_utils.LONGWAVE_HEATING_RATE_NAME
        )
    )[example_index, ...]

    heating_rate_axes_object.plot(
        heating_rates_kelvins_day01, heights_km_agl,
        color=HEATING_RATE_COLOUR, linewidth=line_width, linestyle=line_style
    )

    downwelling_fluxes_w_m02 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=(
            example_utils.SHORTWAVE_DOWN_FLUX_NAME if for_shortwave
            else example_utils.LONGWAVE_DOWN_FLUX_NAME
        )
    )[example_index, ...]

    down_flux_axes_object.plot(
        downwelling_fluxes_w_m02, heights_km_agl,
        color=DOWNWELLING_FLUX_COLOUR, linewidth=line_width,
        linestyle=line_style
    )

    upwelling_fluxes_w_m02 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=(
            example_utils.SHORTWAVE_UP_FLUX_NAME if for_shortwave
            else example_utils.LONGWAVE_UP_FLUX_NAME
        )
    )[example_index, ...]

    up_flux_axes_object.plot(
        upwelling_fluxes_w_m02, heights_km_agl,
        color=UPWELLING_FLUX_COLOUR, linewidth=line_width, linestyle=line_style
    )

    if handle_dict is None:
        if for_shortwave:
            band_string = 'Shortwave'
        else:
            band_string = 'Longwave'

        heating_rate_axes_object.set_ylabel('Height (km AGL)')
        heating_rate_axes_object.set_xlabel(
            band_string + r' heating rate (K day$^{-1}$)'
        )
        down_flux_axes_object.set_xlabel(
            r'Downwelling ' + band_string.lower() + ' flux (W m$^{-2}$)'
        )
        up_flux_axes_object.set_xlabel(
            r'Upwelling ' + band_string.lower() + ' flux (W m$^{-2}$)'
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
        numpy.max(downwelling_fluxes_w_m02),
        numpy.max(upwelling_fluxes_w_m02)
    ])

    max_flux_w_m02 = 1.025 * numpy.max(these_fluxes_w_m02)

    these_fluxes_w_m02 = numpy.array([
        down_flux_axes_object.get_xlim()[0],
        up_flux_axes_object.get_xlim()[0],
        numpy.min(downwelling_fluxes_w_m02),
        numpy.min(upwelling_fluxes_w_m02)
    ])

    min_flux_w_m02 = 1.025 * numpy.min(these_fluxes_w_m02)
    min_flux_w_m02 = numpy.minimum(min_flux_w_m02, 0.)

    down_flux_axes_object.set_xlim(min_flux_w_m02, max_flux_w_m02)
    up_flux_axes_object.set_xlim(min_flux_w_m02, max_flux_w_m02)

    min_heating_rate_k_day01 = numpy.minimum(
        numpy.min(heating_rates_kelvins_day01),
        heating_rate_axes_object.get_xlim()[0]
    )
    min_heating_rate_k_day01 = numpy.minimum(min_heating_rate_k_day01, 0.)
    max_heating_rate_k_day01 = numpy.maximum(
        numpy.max(heating_rates_kelvins_day01),
        heating_rate_axes_object.get_xlim()[1]
    )

    heating_rate_axes_object.set_xlim(
        min_heating_rate_k_day01, max_heating_rate_k_day01
    )

    return {
        FIGURE_HANDLE_KEY: figure_object,
        HEATING_RATE_HANDLE_KEY: heating_rate_axes_object,
        DOWN_FLUX_HANDLE_KEY: down_flux_axes_object,
        UP_FLUX_HANDLE_KEY: up_flux_axes_object
    }


def plot_actual_and_predicted(
        actual_values, prediction_matrix, heights_m_agl, fancy_target_name,
        line_colours, line_widths, line_styles, use_log_scale,
        add_two_dummy_axes=False,
        plot_uncertainty_with_violin=False,
        plot_uncertainty_with_shading=False,
        plot_uncertainty_with_error_bars=False, confidence_level=None):
    """Plots actual and predicted values of one target variable.

    H = number of heights
    S = number of ensemble members

    :param actual_values: length-H numpy array of actual values.
    :param prediction_matrix: H-by-S numpy array of predicted values.
    :param heights_m_agl: length-H numpy array of heights (metres above ground
        level).
    :param fancy_target_name: Fancy name of target variable.
    :param line_colours: length-2 list of colours -- for actual and then
        predictions -- each colour in any format accepted by matplotlib.
    :param line_widths: length-2 numpy array of line widths.
    :param line_styles: length-2 list of line styles (each style in any format
        accepted by matplotlib).
    :param use_log_scale: Boolean flag.  If True, will plot height (y-axis) in
        logarithmic scale.  If False, will plot height in linear scale.
    :param add_two_dummy_axes: Boolean flag.  If True, will add two dummy x-axes
        that correspond to nothing.  The only reason for doing this is to make
        the vertical scale of the figure match another figure with 4 variables
        plotted.
    :param plot_uncertainty_with_violin: Boolean flag.  If True, will plot
        uncertainty in predictions with violin plot at each height.
    :param plot_uncertainty_with_shading: Boolean flag.  If True, will plot
        uncertainty in predictions with shaded envelope.
    :param plot_uncertainty_with_error_bars: Boolean flag.  If True, will plot
        uncertainty in predictions with error bars at each height.
    :param confidence_level:
        [used only if plot_uncertainty_with_shading == True or
        plot_uncertainty_with_error_bars == True]
        Confidence level to display, ranging from 0...1.

    :return: handle_dict: Dictionary with the following keys.
    handle_dict['figure_object']: Figure handle (instance of
        `matplotlib.figure.Figure`).
    handle_dict['axes_objects']: length-P list of axes handles (each an instance
        of `matplotlib.axes._subplots.AxesSubplot`).
    """

    # Check input args.
    error_checking.assert_is_numpy_array_without_nan(actual_values)
    error_checking.assert_is_numpy_array(actual_values, num_dimensions=1)

    error_checking.assert_is_numpy_array_without_nan(prediction_matrix)
    error_checking.assert_is_numpy_array(prediction_matrix, num_dimensions=2)

    num_heights = len(actual_values)
    ensemble_size = prediction_matrix.shape[1]
    error_checking.assert_is_numpy_array(
        prediction_matrix,
        exact_dimensions=numpy.array([num_heights, ensemble_size], dtype=int)
    )

    error_checking.assert_is_greater_numpy_array(heights_m_agl, 0.)
    error_checking.assert_is_greater_numpy_array(numpy.diff(heights_m_agl), 0.)
    error_checking.assert_is_numpy_array(
        heights_m_agl, exact_dimensions=numpy.array([num_heights], dtype=int)
    )

    error_checking.assert_is_string(fancy_target_name)
    error_checking.assert_is_boolean(use_log_scale)

    assert len(line_colours) == 2
    assert len(line_widths) == 2
    assert len(line_styles) == 2

    error_checking.assert_is_boolean(plot_uncertainty_with_violin)
    error_checking.assert_is_boolean(plot_uncertainty_with_shading)
    error_checking.assert_is_boolean(plot_uncertainty_with_error_bars)

    if plot_uncertainty_with_violin:
        plot_uncertainty_with_shading = False
        plot_uncertainty_with_error_bars = False
    if plot_uncertainty_with_shading:
        plot_uncertainty_with_error_bars = False
    if plot_uncertainty_with_shading or plot_uncertainty_with_error_bars:
        error_checking.assert_is_leq(confidence_level, 1.)
        error_checking.assert_is_geq(confidence_level, 0.9)

    # Housekeeping.
    _set_font_size(FANCY_FONT_SIZE)

    figure_object, first_axes_object = pyplot.subplots(
        1, 1,
        figsize=(FANCY_FIGURE_WIDTH_INCHES, FANCY_FIGURE_HEIGHT_INCHES)
    )

    axes_objects = [first_axes_object]
    figure_object.subplots_adjust(bottom=0.75)

    if use_log_scale:
        pyplot.yscale('log')

    axes_objects.append(axes_objects[0].twiny())

    if add_two_dummy_axes:
        for k in range(2, 4):
            axes_objects.append(axes_objects[0].twiny())

            if k == 2:
                axes_objects[k].spines['top'].set_position(('axes', 1.15))
                _make_spines_invisible(axes_objects[k])
                axes_objects[k].spines['top'].set_visible(True)

            if k == 3:
                axes_objects[k].xaxis.set_ticks_position('bottom')
                axes_objects[k].xaxis.set_label_position('bottom')
                axes_objects[k].spines['bottom'].set_position(('axes', -0.15))
                _make_spines_invisible(axes_objects[k])
                axes_objects[k].spines['bottom'].set_visible(True)

    heights_km_agl = METRES_TO_KM * heights_m_agl
    tick_mark_dict = dict(size=4, width=1.5)

    for k in range(2):
        axes_objects[k].plot(
            actual_values if k == 0 else numpy.mean(prediction_matrix, axis=1),
            heights_km_agl, color=line_colours[k],
            linewidth=line_widths[k], linestyle=line_styles[k],
            zorder=1e12
        )

        axes_objects[k].set_xlabel('{0:s} {1:s}'.format(
            'Actual' if k == 0 else 'Predicted', fancy_target_name
        ))
        axes_objects[k].xaxis.label.set_color(line_colours[k])
        axes_objects[k].tick_params(
            axis='x', colors=line_colours[k], **tick_mark_dict
        )

    if plot_uncertainty_with_violin:
        pixel_widths_km = METRES_TO_KM * example_utils.get_grid_cell_widths(
            example_utils.get_grid_cell_edges(heights_m_agl)
        )

        violin_handles = axes_objects[1].violinplot(
            numpy.transpose(prediction_matrix),
            positions=heights_km_agl,
            vert=False, widths=pixel_widths_km, showmeans=False,
            showmedians=False, showextrema=True
        )

        for part_name in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
            try:
                this_handle = violin_handles[part_name]
            except:
                continue

            this_handle.set_edgecolor(line_colours[1])
            this_handle.set_linewidth(0)

        for this_handle in violin_handles['bodies']:
            this_handle.set_facecolor(
                matplotlib.colors.to_rgba(
                    c=line_colours[1], alpha=OPACITY_FOR_UNCERTAINTY
                )
            )
            this_handle.set_edgecolor(
                matplotlib.colors.to_rgba(
                    c=line_colours[1], alpha=OPACITY_FOR_UNCERTAINTY
                )
            )
            this_handle.set_linewidth(0)
            this_handle.set_alpha(0.5)

    if plot_uncertainty_with_error_bars:
        min_prediction_by_height = numpy.percentile(
            prediction_matrix, 50 * (1. - confidence_level), axis=1
        )
        max_prediction_by_height = numpy.percentile(
            prediction_matrix, 50 * (1. + confidence_level), axis=1
        )
        mean_prediction_by_height = numpy.mean(prediction_matrix, axis=1)
        error_matrix = numpy.vstack((
            mean_prediction_by_height - min_prediction_by_height,
            max_prediction_by_height - mean_prediction_by_height
        ))

        axes_objects[1].errorbar(
            x=mean_prediction_by_height, y=heights_km_agl, xerr=error_matrix,
            linewidth=0, ecolor=line_colours[1], elinewidth=line_widths[1],
            capsize=6, capthick=3
        )

    if plot_uncertainty_with_shading:
        min_prediction_by_height = numpy.percentile(
            prediction_matrix, 50 * (1. - confidence_level), axis=1
        )
        max_prediction_by_height = numpy.percentile(
            prediction_matrix, 50 * (1. + confidence_level), axis=1
        )

        polygon_colour = 1. + OPACITY_FOR_UNCERTAINTY * (line_colours[1] - 1)

        axes_objects[1].fill_betweenx(
            y=heights_km_agl,
            x1=min_prediction_by_height, x2=max_prediction_by_height,
            facecolor=polygon_colour, alpha=1.,
            linewidth=5, edgecolor=polygon_colour, zorder=-1e12
        )

    if add_two_dummy_axes:
        for k in range(2, 4):
            axes_objects[k].set_xlabel(fancy_target_name)
            axes_objects[k].xaxis.label.set_color(BLACK_COLOUR)
            axes_objects[k].tick_params(
                axis='x', colors=BLACK_COLOUR, **tick_mark_dict
            )

    axes_objects[1].plot(
        actual_values, heights_km_agl, color=line_colours[0],
        linewidth=line_widths[0], linestyle=line_styles[0],
        zorder=1e12
    )
    axes_objects[1].plot(
        numpy.mean(prediction_matrix, axis=1),
        heights_km_agl, color=line_colours[1],
        linewidth=line_widths[1], linestyle=line_styles[1],
        zorder=1e12
    )

    x_min = min([
        axes_objects[0].get_xlim()[0],
        axes_objects[1].get_xlim()[0]
    ])
    x_max = max([
        axes_objects[0].get_xlim()[1],
        axes_objects[1].get_xlim()[1]
    ])
    axes_objects[0].set_xlim([x_min, x_max])
    axes_objects[1].set_xlim([x_min, x_max])

    axes_objects[0].set_ylabel('Height (km AGL)')
    axes_objects[0].set_ylim([
        numpy.min(heights_km_agl), numpy.max(heights_km_agl)
    ])

    height_strings = create_height_labels(
        tick_values_km_agl=axes_objects[0].get_yticks(),
        use_log_scale=use_log_scale
    )
    axes_objects[0].set_yticklabels(height_strings)
    axes_objects[0].tick_params(axis='y', **tick_mark_dict)

    return {
        FIGURE_HANDLE_KEY: figure_object,
        AXES_OBJECTS_KEY: axes_objects
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

    x_min = numpy.minimum(
        axes_object.get_xlim()[0], numpy.min(values)
    )
    x_min = numpy.minimum(x_min, 0.)
    x_max = numpy.maximum(
        axes_object.get_xlim()[1], numpy.max(values)
    )
    axes_object.set_xlim(x_min, x_max)

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
