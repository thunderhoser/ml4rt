"""Perturbs GFS data before input to the RRTM.

The goal is to create out-of-regime data samples, i.e., samples that are very
dissimilar from the training data.
"""

import os
import sys
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import moisture_conversions as moisture_conv
import file_system_utils
import error_checking
import rrtm_io
import prepare_gfs_for_rrtm_no_interp as prepare_gfs_for_rrtm

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MAX_TROPOPAUSE_LAPSE_RATE_KELVINS_M01 = 0.002
TROPOPAUSE_LOOK_ABOVE_HEIGHT_METRES = 2000.
MIN_TROPOPAUSE_HEIGHT_EVER_M_AGL = 5000.
MAX_TROPOPAUSE_HEIGHT_EVER_M_AGL = 20000.
MAX_PLAUSIBLE_TROPOPAUSE_HEIGHT_M_AGL = 18000.

MIN_TEMP_FOR_LIQUID_CLOUD_KELVINS = 233.15

MAX_TEMPERATURE_EVER_KELVINS = 333.15
MAX_MIXING_RATIO_EVER_KG_KG01 = 0.04

INPUT_FILE_ARG_NAME = 'input_file_name'
MAX_TEMP_INCREASE_ARG_NAME = 'max_temp_increase_kelvins'
MAX_WARM_LAYER_THICKNESS_ARG_NAME = 'max_warm_layer_thickness_metres'
SURFACE_RH_LIMITS_ARG_NAME = 'surface_relative_humidity_limits'
MAX_MOIST_LAYER_THICKNESS_ARG_NAME = 'max_moist_layer_thickness_metres'
OZONE_LAYER_THICKNESS_LIMITS_ARG_NAME = 'ozone_layer_thickness_limits_metres'
OZONE_LAYER_CENTER_LIMITS_ARG_NAME = 'ozone_layer_center_limits_m_agl'
MAX_OZONE_MIXING_RATIO_ARG_NAME = 'max_ozone_mixing_ratio_kg_kg01'
OZONE_MIXING_RATIO_NOISE_ARG_NAME = 'ozone_mixing_ratio_noise_stdev_kg_kg01'
MAX_NUM_LIQUID_CLOUDS_ARG_NAME = 'max_num_liquid_cloud_layers'
MAX_LIQUID_CLOUD_THICKNESS_ARG_NAME = 'max_liquid_cloud_layer_thickness_metres'
MAX_LIQUID_WATER_CONTENT_ARG_NAME = 'max_liquid_water_content_kg_m03'
LIQUID_WATER_CONTENT_NOISE_ARG_NAME = 'liquid_water_content_noise_stdev_kg_m03'
MAX_NUM_ICE_CLOUDS_ARG_NAME = 'max_num_ice_cloud_layers'
MAX_ICE_CLOUD_THICKNESS_ARG_NAME = 'max_ice_cloud_layer_thickness_metres'
MAX_ICE_WATER_CONTENT_ARG_NAME = 'max_ice_water_content_kg_m03'
ICE_WATER_CONTENT_NOISE_ARG_NAME = 'ice_water_content_noise_stdev_kg_m03'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (created by prepare_gfs_for_rrtm_no_interp.py), '
    'containing GFS data for one init time.'
)
MAX_TEMP_INCREASE_HELP_STRING = (
    'Max temperature increase.  The max increase will always occur at the '
    'surface.'
)
MAX_WARM_LAYER_THICKNESS_HELP_STRING = (
    'Max thickness of warm layer.  The bottom of the warm layer will always be '
    'at the surface.'
)
SURFACE_RH_LIMITS_HELP_STRING = (
    'List with [min, max] of uniform distribution for surface relative '
    'humidity.'
)
MAX_MOIST_LAYER_THICKNESS_HELP_STRING = (
    'Max thickness of moist layer.  The bottom of the moist layer will always '
    'be at the surface.'
)
OZONE_LAYER_THICKNESS_LIMITS_HELP_STRING = (
    'List with [min, max] of uniform distribution for ozone-layer thickness.'
)
OZONE_LAYER_CENTER_LIMITS_HELP_STRING = (
    'List with [min, max] of uniform distribution for ozone-layer center '
    '(height of max mixing ratio, in metres above ground).'
)
MAX_OZONE_MIXING_RATIO_HELP_STRING = (
    'Max of uniform distribution for layer-max ozone mixing ratio.  The minimum'
    ' of the uniform distribution is always 0 kg/kg.'
)
OZONE_MIXING_RATIO_NOISE_HELP_STRING = (
    'Standard deviation of Gaussian noise for ozone mixing ratio.'
)
MAX_NUM_LIQUID_CLOUDS_HELP_STRING = (
    'Max of uniform distribution for number of liquid-cloud layers.  The '
    'minimum of the distribution is always 0.'
)
MAX_LIQUID_CLOUD_THICKNESS_HELP_STRING = (
    'Max of uniform distribution for thickness of liquid-cloud layer.  The '
    'minimum of the distribution is always 0.'
)
MAX_LIQUID_WATER_CONTENT_HELP_STRING = (
    'Max of uniform distribution for max LWC in a given cloud layer.  The '
    'minimum of the distribution is always 0.'
)
LIQUID_WATER_CONTENT_NOISE_HELP_STRING = (
    'Standard deviation of Gaussian noise for LWC, applied to each level with '
    'LWC > 0.'
)
MAX_NUM_ICE_CLOUDS_HELP_STRING = 'Same as `{0:s}` but for ice.'.format(
    MAX_NUM_LIQUID_CLOUDS_ARG_NAME
)
MAX_ICE_CLOUD_THICKNESS_HELP_STRING = 'Same as `{0:s}` but for ice.'.format(
    MAX_LIQUID_CLOUD_THICKNESS_ARG_NAME
)
MAX_ICE_WATER_CONTENT_HELP_STRING = 'Same as `{0:s}` but for ice.'.format(
    MAX_LIQUID_WATER_CONTENT_ARG_NAME
)
ICE_WATER_CONTENT_NOISE_HELP_STRING = 'Same as `{0:s}` but for ice.'.format(
    LIQUID_WATER_CONTENT_NOISE_ARG_NAME
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output (NetCDF) file.  Format will be same as input file, but '
    'output file will contain perturbed profiles, not real ones.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_TEMP_INCREASE_ARG_NAME, type=float, required=True,
    help=MAX_TEMP_INCREASE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_WARM_LAYER_THICKNESS_ARG_NAME, type=float, required=True,
    help=MAX_WARM_LAYER_THICKNESS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SURFACE_RH_LIMITS_ARG_NAME, type=float, nargs=2, required=True,
    help=SURFACE_RH_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_MOIST_LAYER_THICKNESS_ARG_NAME, type=float, required=True,
    help=MAX_MOIST_LAYER_THICKNESS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OZONE_LAYER_THICKNESS_LIMITS_ARG_NAME, type=float, nargs=2,
    required=True, help=OZONE_LAYER_THICKNESS_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OZONE_LAYER_CENTER_LIMITS_ARG_NAME, type=float, nargs=2,
    required=True, help=OZONE_LAYER_CENTER_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_OZONE_MIXING_RATIO_ARG_NAME, type=float, required=True,
    help=MAX_OZONE_MIXING_RATIO_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OZONE_MIXING_RATIO_NOISE_ARG_NAME, type=float, required=True,
    help=OZONE_MIXING_RATIO_NOISE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_NUM_LIQUID_CLOUDS_ARG_NAME, type=int, required=True,
    help=MAX_NUM_LIQUID_CLOUDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LIQUID_CLOUD_THICKNESS_ARG_NAME, type=float, required=True,
    help=MAX_LIQUID_CLOUD_THICKNESS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LIQUID_WATER_CONTENT_ARG_NAME, type=float, required=True,
    help=MAX_LIQUID_WATER_CONTENT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LIQUID_WATER_CONTENT_NOISE_ARG_NAME, type=float, required=True,
    help=LIQUID_WATER_CONTENT_NOISE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_NUM_ICE_CLOUDS_ARG_NAME, type=int, required=True,
    help=MAX_NUM_ICE_CLOUDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_ICE_CLOUD_THICKNESS_ARG_NAME, type=float, required=True,
    help=MAX_ICE_CLOUD_THICKNESS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_ICE_WATER_CONTENT_ARG_NAME, type=float, required=True,
    help=MAX_ICE_WATER_CONTENT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ICE_WATER_CONTENT_NOISE_ARG_NAME, type=float, required=True,
    help=ICE_WATER_CONTENT_NOISE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _heights_to_grid_indices(
        min_height_m_agl, max_height_m_agl, sorted_grid_heights_m_agl):
    """Converts heights within layer to grid-point indices.

    :param min_height_m_agl: Minimum height in layer (metres above ground).
    :param max_height_m_agl: Max height in layer (metres above ground).
    :param sorted_grid_heights_m_agl: 1-D numpy array of grid-point heights
        (metres above ground).  This method assumes that the array is sorted in
        ascending order.
    :return: grid_point_indices: 1-D numpy array of indices in layer.
    """

    min_index = numpy.argmin(
        numpy.absolute(min_height_m_agl - sorted_grid_heights_m_agl)
    )
    max_index = numpy.argmin(
        numpy.absolute(max_height_m_agl - sorted_grid_heights_m_agl)
    )
    return numpy.linspace(
        min_index, max_index, num=max_index - min_index + 1, dtype=int
    )


def _find_tropopause(temperatures_kelvins, sorted_heights_m_agl):
    """Finds the lowest tropopause.

    H = number of heights

    :param temperatures_kelvins: length-H numpy array of temperatures.
    :param sorted_heights_m_agl: length-H numpy array of heights (metres above
        ground).  This method assumes that the array is sorted in ascending
        order.
    :return: tropopause_height_m_agl: Tropopause height (metres above ground).
    :return: tropopause_height_index: Grid-point index at tropopause.
    """

    lapse_rates_kelvins_m01 = (
        -1 * numpy.diff(temperatures_kelvins) / numpy.diff(sorted_heights_m_agl)
    )

    these_flags = numpy.logical_and(
        sorted_heights_m_agl[:-1] >= MIN_TROPOPAUSE_HEIGHT_EVER_M_AGL,
        sorted_heights_m_agl[:-1] <= MAX_TROPOPAUSE_HEIGHT_EVER_M_AGL
    )
    these_indices = numpy.where(numpy.logical_and(
        lapse_rates_kelvins_m01 < MAX_TROPOPAUSE_LAPSE_RATE_KELVINS_M01,
        these_flags
    ))[0]

    if len(these_indices) == 0:
        return None, None

    for lower_index in these_indices:
        lower_height_m_agl = sorted_heights_m_agl[lower_index]
        upper_height_m_agl = (
            lower_height_m_agl + TROPOPAUSE_LOOK_ABOVE_HEIGHT_METRES
        )

        upper_index = numpy.argmin(numpy.absolute(
            sorted_heights_m_agl - upper_height_m_agl
        ))

        if (
                sorted_heights_m_agl[upper_index] < upper_height_m_agl and
                upper_index < len(sorted_heights_m_agl) - 1
        ):
            upper_index += 1

        numerator = (
            temperatures_kelvins[lower_index] -
            temperatures_kelvins[upper_index]
        )
        denominator = (
            sorted_heights_m_agl[upper_index] -
            sorted_heights_m_agl[lower_index]
        )
        this_lapse_rate_kelvins_m01 = numerator / denominator

        if this_lapse_rate_kelvins_m01 >= MAX_TROPOPAUSE_LAPSE_RATE_KELVINS_M01:
            continue

        return sorted_heights_m_agl[lower_index], lower_index

    return None, None


def _create_cloud(
        gfs_table_xarray, time_index, site_index, max_num_cloud_layers,
        max_layer_thickness_metres, max_water_content_kg_m03,
        water_content_noise_stdev_kg_m03, liquid_flag):
    """Creates fictitious liquid or ice cloud.

    Allowing cloud up to 2 km above tropopause is motivated by:
    https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JD034808

    :param gfs_table_xarray: xarray table with GFS data.
    :param time_index: Will create fictitious warm layer for this time.
    :param site_index: Will create fictitious warm layer for this site.
    :param max_num_cloud_layers: See documentation at top of file.
    :param max_layer_thickness_metres: Same.
    :param max_water_content_kg_m03: Same.
    :param water_content_noise_stdev_kg_m03: Same.
    :param liquid_flag: Boolean flag.  If True (False), will create liquid (ice)
        cloud.
    :return: gfs_table_xarray: Same as input but with fictitious liquid cloud
        for the given time step and site.
    """

    i = time_index
    j = site_index
    t = gfs_table_xarray

    num_cloud_layers = numpy.random.random_integers(
        low=0, high=max_num_cloud_layers, size=1
    )[0]

    if num_cloud_layers == 0:
        return gfs_table_xarray

    tropopause_height_m_agl, _ = _find_tropopause(
        temperatures_kelvins=
        t[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[i, j, :],
        sorted_heights_m_agl=
        t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[i, j, :]
    )
    if tropopause_height_m_agl is None:
        tropopause_height_m_agl = MIN_TROPOPAUSE_HEIGHT_EVER_M_AGL

    height_indices_by_layer = [numpy.array([], dtype=int)] * num_cloud_layers

    # if liquid_flag:
    #     water_path_key = prepare_gfs_for_rrtm.LIQUID_WATER_PATH_KEY_KG_M02
    # else:
    #     water_path_key = prepare_gfs_for_rrtm.ICE_WATER_PATH_KEY_KG_M02
    #
    # changed_water_paths = False

    num_heights = t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values.shape[2]
    new_water_contents_kg_m03 = numpy.full(num_heights, 0.)

    for k in range(num_cloud_layers):
        this_top_height_m_agl = numpy.random.uniform(
            low=0., high=tropopause_height_m_agl + 2000, size=1
        )[0]
        this_thickness_metres = numpy.random.uniform(
            low=0., high=max_layer_thickness_metres, size=1
        )[0]
        this_bottom_height_m_agl = max([
            this_top_height_m_agl - this_thickness_metres,
            0.
        ])

        height_indices_by_layer[k] = _heights_to_grid_indices(
            min_height_m_agl=this_bottom_height_m_agl,
            max_height_m_agl=this_top_height_m_agl,
            sorted_grid_heights_m_agl=
            t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[i, j, :]
        )

        if liquid_flag:
            good_temperature_flags = (
                t[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[
                    i, j, height_indices_by_layer[k]
                ] >= MIN_TEMP_FOR_LIQUID_CLOUD_KELVINS
            )
        else:
            good_temperature_flags = (
                t[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[
                    i, j, height_indices_by_layer[k]
                ] < 273.15
            )

        height_indices_by_layer[k] = (
            height_indices_by_layer[k][good_temperature_flags]
        )

        if k == 0:
            continue

        test_indices = numpy.array([], dtype=int)

        for m in range(k):
            if len(height_indices_by_layer[m]) == 0:
                continue

            test_indices = numpy.concatenate((
                test_indices,
                height_indices_by_layer[m][[0]] - 1,
                height_indices_by_layer[m],
                height_indices_by_layer[m][[-1]] + 1
            ))

        height_indices_by_layer[k] = height_indices_by_layer[k][
            numpy.invert(numpy.isin(height_indices_by_layer[k], test_indices))
        ]

        if len(height_indices_by_layer[k]) < 2:
            height_indices_by_layer[k] = numpy.array([], dtype=int)
            continue

        # if not changed_water_paths:
        #     t[water_path_key].values[i, j, :] = 0.
        #     changed_water_paths = True

        enhanced_height_indices = numpy.concatenate((
            height_indices_by_layer[k][[0]] - 1,
            height_indices_by_layer[k],
            height_indices_by_layer[k][[-1]] + 1
        ))

        enhanced_layer_heights_m_agl = (
            t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[
                i, j, enhanced_height_indices
            ]
        )

        layer_heights_m_agl = (
            t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[
                i, j, height_indices_by_layer[k]
            ]
        )

        layer_center_m_agl = numpy.mean(layer_heights_m_agl)
        max_height_diff_metres = numpy.max(
            numpy.absolute(layer_center_m_agl - enhanced_layer_heights_m_agl)
        )

        layer_height_diffs_metres = numpy.absolute(
            layer_center_m_agl - layer_heights_m_agl
        )
        layer_height_diffs_relative = (
            layer_height_diffs_metres / max_height_diff_metres
        )

        layer_max_water_content_kg_m03 = numpy.random.uniform(
            low=0., high=max_water_content_kg_m03, size=1
        )[0]
        layer_water_contents_kg_m03 = (
            (1. - layer_height_diffs_relative) * layer_max_water_content_kg_m03
        )
        layer_water_contents_kg_m03 += numpy.random.normal(
            loc=0., scale=water_content_noise_stdev_kg_m03,
            size=len(layer_water_contents_kg_m03)
        )
        layer_water_contents_kg_m03 = numpy.maximum(
            layer_water_contents_kg_m03, 0.
        )
        layer_water_contents_kg_m03 = numpy.minimum(
            layer_water_contents_kg_m03, layer_max_water_content_kg_m03
        )

        new_water_contents_kg_m03[height_indices_by_layer[k]] = (
            layer_water_contents_kg_m03
        )

    new_cloud_water_paths_kg_m02 = rrtm_io._water_content_to_layerwise_path(
        water_content_matrix_kg_m03=numpy.expand_dims(
            new_water_contents_kg_m03, axis=0
        ),
        heights_m_agl=t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[i, j, :]
    )[0, :]

    if liquid_flag:
        t[prepare_gfs_for_rrtm.LIQUID_WATER_PATH_KEY_KG_M02].values[
            i, j, :
        ] = new_cloud_water_paths_kg_m02
    else:
        t[prepare_gfs_for_rrtm.ICE_WATER_PATH_KEY_KG_M02].values[
            i, j, :
        ] = new_cloud_water_paths_kg_m02

    gfs_table_xarray = t
    return gfs_table_xarray


def _create_surface_based_moist_layer(
        gfs_table_xarray, time_index, site_index, max_layer_thickness_metres,
        surface_relative_humidity_limits):
    """Creates fictitious surface-based moist layer.

    :param gfs_table_xarray: xarray table with GFS data.
    :param time_index: Will create fictitious warm layer for this time.
    :param site_index: Will create fictitious warm layer for this site.
    :param max_layer_thickness_metres: See documentation at top of file.
    :param surface_relative_humidity_limits: Same.
    :return: gfs_table_xarray: Same as input but with fictitious moist layer for
        the given time step and site.
    """

    i = time_index
    j = site_index
    t = gfs_table_xarray

    surface_relative_humidity = numpy.random.uniform(
        low=surface_relative_humidity_limits[0],
        high=surface_relative_humidity_limits[1], size=1
    )[0]
    surface_relative_humidity = max([surface_relative_humidity, 0.])
    surface_relative_humidity = min([surface_relative_humidity, 1.])

    layer_thickness_metres = numpy.random.uniform(
        low=0., high=max_layer_thickness_metres, size=1
    )[0]
    layer_height_indices = _heights_to_grid_indices(
        min_height_m_agl=0.,
        max_height_m_agl=layer_thickness_metres,
        sorted_grid_heights_m_agl=
        t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[i, j, :]
    )

    _, tropopause_height_index = _find_tropopause(
        temperatures_kelvins=
        t[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[i, j, :],
        sorted_heights_m_agl=
        t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[i, j, :]
    )
    if tropopause_height_index is None:
        tropopause_height_index = numpy.argmin(numpy.absolute(
            t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[i, j, :] -
            MIN_TROPOPAUSE_HEIGHT_EVER_M_AGL
        ))

    layer_height_indices = layer_height_indices[
        layer_height_indices < tropopause_height_index
    ]

    if len(layer_height_indices) == 0:
        return gfs_table_xarray

    (
        surface_dewpoint_kelvins_as_array
    ) = moisture_conv.relative_humidity_to_dewpoint(
        relative_humidities=numpy.array([surface_relative_humidity]),
        temperatures_kelvins=
        t[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[i, j, [0]],
        total_pressures_pascals=
        t[prepare_gfs_for_rrtm.PRESSURE_KEY_PASCALS].values[i, j, [0]]
    )

    (
        surface_specific_humidity_kg_kg01_as_array
    ) = moisture_conv.dewpoint_to_specific_humidity(
        dewpoints_kelvins=surface_dewpoint_kelvins_as_array,
        temperatures_kelvins=
        t[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[i, j, [0]],
        total_pressures_pascals=
        t[prepare_gfs_for_rrtm.PRESSURE_KEY_PASCALS].values[i, j, [0]]
    )

    surface_mixing_ratio_kg_kg01 = (
        moisture_conv.specific_humidity_to_mixing_ratio(
            surface_specific_humidity_kg_kg01_as_array
        )[0]
    )
    surface_mixing_ratio_kg_kg01 = min([
        surface_mixing_ratio_kg_kg01, MAX_MIXING_RATIO_EVER_KG_KG01
    ])
    surface_mixr_increase_kg_kg01 = (
        surface_mixing_ratio_kg_kg01 -
        t[prepare_gfs_for_rrtm.VAPOUR_MIXR_KEY_KG_KG01].values[i, j, 0]
    )

    if surface_mixr_increase_kg_kg01 <= 0:
        return gfs_table_xarray

    layer_heights_m_agl = t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[
        i, j, layer_height_indices
    ]
    height_diffs_relative = layer_heights_m_agl / numpy.max(layer_heights_m_agl)

    layer_mixing_ratios_kg_kg01 = (
        t[prepare_gfs_for_rrtm.VAPOUR_MIXR_KEY_KG_KG01].values[
            i, j, layer_height_indices
        ]
        + (1. - height_diffs_relative) * surface_mixr_increase_kg_kg01
    )

    layer_specific_humidities_kg_kg01 = (
        moisture_conv.mixing_ratio_to_specific_humidity(
            layer_mixing_ratios_kg_kg01
        )
    )

    layer_dewpoints_kelvins = moisture_conv.specific_humidity_to_dewpoint(
        specific_humidities_kg_kg01=layer_specific_humidities_kg_kg01,
        temperatures_kelvins=
        t[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[
            i, j, layer_height_indices
        ],
        total_pressures_pascals=
        t[prepare_gfs_for_rrtm.PRESSURE_KEY_PASCALS].values[
            i, j, layer_height_indices
        ]
    )

    layer_dewpoints_kelvins = numpy.minimum(
        layer_dewpoints_kelvins,
        t[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[
            i, j, layer_height_indices
        ]
    )

    (
        layer_specific_humidities_kg_kg01
    ) = moisture_conv.dewpoint_to_specific_humidity(
        dewpoints_kelvins=layer_dewpoints_kelvins,
        temperatures_kelvins=
        t[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[
            i, j, layer_height_indices
        ],
        total_pressures_pascals=
        t[prepare_gfs_for_rrtm.PRESSURE_KEY_PASCALS].values[
            i, j, layer_height_indices
        ]
    )

    t[prepare_gfs_for_rrtm.VAPOUR_MIXR_KEY_KG_KG01].values[
        i, j, layer_height_indices
    ] = moisture_conv.specific_humidity_to_mixing_ratio(
        layer_specific_humidities_kg_kg01
    )

    gfs_table_xarray = t
    return gfs_table_xarray


def _create_surface_based_warm_layer(
        gfs_table_xarray, time_index, site_index, max_layer_thickness_metres,
        max_temp_increase_kelvins):
    """Creates fictitious surface-based warm layer.

    :param gfs_table_xarray: xarray table with GFS data.
    :param time_index: Will create fictitious warm layer for this time.
    :param site_index: Will create fictitious warm layer for this site.
    :param max_layer_thickness_metres: See documentation at top of file.
    :param max_temp_increase_kelvins: Same.
    :return: gfs_table_xarray: Same as input but with fictitious warm layer for
        the given time step and site.
    """

    i = time_index
    j = site_index
    t = gfs_table_xarray

    layer_thickness_metres = numpy.random.uniform(
        low=0., high=max_layer_thickness_metres, size=1
    )[0]
    layer_height_indices = _heights_to_grid_indices(
        min_height_m_agl=0.,
        max_height_m_agl=layer_thickness_metres,
        sorted_grid_heights_m_agl=
        t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[i, j, :]
    )

    _, tropopause_height_index = _find_tropopause(
        temperatures_kelvins=
        t[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[i, j, :],
        sorted_heights_m_agl=
        t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[i, j, :]
    )
    if tropopause_height_index is None:
        tropopause_height_index = numpy.argmin(numpy.absolute(
            t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[i, j, :] -
            MIN_TROPOPAUSE_HEIGHT_EVER_M_AGL
        ))

    layer_height_indices = layer_height_indices[
        layer_height_indices < tropopause_height_index
    ]

    if len(layer_height_indices) == 0:
        return gfs_table_xarray

    layer_heights_m_agl = t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[
        i, j, layer_height_indices
    ]
    height_diffs_relative = layer_heights_m_agl / numpy.max(layer_heights_m_agl)

    surface_temp_increase_kelvins = numpy.random.uniform(
        low=0., high=max_temp_increase_kelvins, size=1
    )[0]
    t[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[
        i, j, layer_height_indices
    ] += (1. - height_diffs_relative) * surface_temp_increase_kelvins

    t[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[
        i, j, layer_height_indices
    ] = numpy.minimum(
        t[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[
            i, j, layer_height_indices
        ],
        MAX_TEMPERATURE_EVER_KELVINS
    )

    t[prepare_gfs_for_rrtm.SURFACE_TEMPERATURE_KEY].values[i, j] += (
        surface_temp_increase_kelvins
    )
    t[prepare_gfs_for_rrtm.SURFACE_TEMPERATURE_KEY].values[i, j] = (
        numpy.minimum(
            t[prepare_gfs_for_rrtm.SURFACE_TEMPERATURE_KEY].values[i, j],
            MAX_TEMPERATURE_EVER_KELVINS
        )
    )

    gfs_table_xarray = t
    return gfs_table_xarray


def _create_ozone_layer(
        gfs_table_xarray, time_index, site_index, thickness_limits_metres,
        center_limits_metres, max_mixing_ratio_kg_kg01,
        mixing_ratio_noise_stdev_kg_kg01):
    """Creates fictitious ozone layer.

    :param gfs_table_xarray: xarray table with GFS data.
    :param time_index: Will create fictitious ozone layer for this time.
    :param site_index: Will create fictitious ozone layer for this site.
    :param thickness_limits_metres: See documentation at top of file.
    :param center_limits_metres: Same.
    :param max_mixing_ratio_kg_kg01: Same.
    :param mixing_ratio_noise_stdev_kg_kg01: Same.
    :return: gfs_table_xarray: Same as input but with fictitious ozone layer for
        the given time step and site.
    """

    i = time_index
    j = site_index
    t = gfs_table_xarray

    layer_thickness_metres = numpy.random.uniform(
        low=thickness_limits_metres[0], high=thickness_limits_metres[1], size=1
    )[0]
    layer_center_m_agl = numpy.random.uniform(
        low=center_limits_metres[0], high=center_limits_metres[1], size=1
    )[0]
    layer_bottom_m_agl = layer_center_m_agl - layer_thickness_metres / 2
    layer_top_m_agl = layer_center_m_agl + layer_thickness_metres / 2

    layer_height_indices = _heights_to_grid_indices(
        min_height_m_agl=layer_bottom_m_agl,
        max_height_m_agl=layer_top_m_agl,
        sorted_grid_heights_m_agl=
        t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[i, j, :]
    )

    _, tropopause_height_index = _find_tropopause(
        temperatures_kelvins=
        t[prepare_gfs_for_rrtm.TEMPERATURE_KEY_KELVINS].values[i, j, :],
        sorted_heights_m_agl=
        t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[i, j, :]
    )
    if tropopause_height_index is None:
        tropopause_height_index = numpy.argmin(numpy.absolute(
            t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[i, j, :] -
            MAX_TROPOPAUSE_HEIGHT_EVER_M_AGL
        ))

    layer_height_indices = layer_height_indices[
        layer_height_indices > tropopause_height_index
    ]

    t[prepare_gfs_for_rrtm.OZONE_MIXR_KEY_KG_KG01].values[i, j, :] = 0.
    if len(layer_height_indices) == 0:
        gfs_table_xarray = t
        return gfs_table_xarray

    layer_height_indices = numpy.concatenate((
        layer_height_indices[[0]] - 1,
        layer_height_indices,
        layer_height_indices[[-1]] + 1
    ))

    num_heights = len(t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[i, j, :])
    layer_height_indices = layer_height_indices[
        layer_height_indices < num_heights
    ]
    layer_height_indices = layer_height_indices[layer_height_indices >= 0]

    layer_heights_m_agl = t[prepare_gfs_for_rrtm.HEIGHT_KEY_M_AGL].values[
        i, j, layer_height_indices
    ]

    if (
            layer_center_m_agl < layer_heights_m_agl[0] or
            layer_center_m_agl > layer_heights_m_agl[-1]
    ):
        min_index = numpy.argmin(
            numpy.absolute(layer_center_m_agl - layer_heights_m_agl)
        )
        layer_center_m_agl = layer_heights_m_agl[min_index]

    height_diffs_metres = numpy.absolute(
        layer_center_m_agl - layer_heights_m_agl
    )
    height_diffs_relative = height_diffs_metres / numpy.max(height_diffs_metres)
    height_diffs_relative[0] = 1.
    height_diffs_relative[-1] = 1.

    this_max_mixing_ratio_kg_kg01 = numpy.random.uniform(
        low=0., high=max_mixing_ratio_kg_kg01, size=1
    )[0]
    t[prepare_gfs_for_rrtm.OZONE_MIXR_KEY_KG_KG01].values[
        i, j, layer_height_indices
    ] = (1. - height_diffs_relative) * this_max_mixing_ratio_kg_kg01

    t[prepare_gfs_for_rrtm.OZONE_MIXR_KEY_KG_KG01].values[
        i, j, layer_height_indices
    ] += numpy.random.normal(
        loc=0., scale=mixing_ratio_noise_stdev_kg_kg01,
        size=len(layer_height_indices)
    )

    t[prepare_gfs_for_rrtm.OZONE_MIXR_KEY_KG_KG01].values[
        i, j, layer_height_indices
    ] = numpy.maximum(
        t[prepare_gfs_for_rrtm.OZONE_MIXR_KEY_KG_KG01].values[
            i, j, layer_height_indices
        ],
        0.
    )

    t[prepare_gfs_for_rrtm.OZONE_MIXR_KEY_KG_KG01].values[
        i, j, layer_height_indices
    ] = numpy.minimum(
        t[prepare_gfs_for_rrtm.OZONE_MIXR_KEY_KG_KG01].values[
            i, j, layer_height_indices
        ],
        this_max_mixing_ratio_kg_kg01
    )

    gfs_table_xarray = t
    return gfs_table_xarray


def _run(input_file_name, max_temp_increase_kelvins,
         max_warm_layer_thickness_metres,
         surface_relative_humidity_limits, max_moist_layer_thickness_metres,
         ozone_layer_thickness_limits_metres, ozone_layer_center_limits_metres,
         max_ozone_mixing_ratio_kg_kg01, ozone_mixing_ratio_noise_stdev_kg_kg01,
         max_num_liquid_cloud_layers, max_liquid_cloud_layer_thickness_metres,
         max_liquid_water_content_kg_m03,
         liquid_water_content_noise_stdev_kg_m03,
         max_num_ice_cloud_layers, max_ice_cloud_layer_thickness_metres,
         max_ice_water_content_kg_m03, ice_water_content_noise_stdev_kg_m03,
         output_file_name):
    """Perturbs GFS data before input to the RRTM.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param max_temp_increase_kelvins: Same.
    :param max_warm_layer_thickness_metres: Same.
    :param surface_relative_humidity_limits: Same.
    :param max_moist_layer_thickness_metres: Same.
    :param ozone_layer_thickness_limits_metres: Same.
    :param ozone_layer_center_limits_metres: Same.
    :param max_ozone_mixing_ratio_kg_kg01: Same.
    :param ozone_mixing_ratio_noise_stdev_kg_kg01: Same.
    :param max_num_liquid_cloud_layers: Same.
    :param max_liquid_cloud_layer_thickness_metres: Same.
    :param max_liquid_water_content_kg_m03: Same.
    :param liquid_water_content_noise_stdev_kg_m03: Same.
    :param max_num_ice_cloud_layers: Same.
    :param max_ice_cloud_layer_thickness_metres: Same.
    :param max_ice_water_content_kg_m03: Same.
    :param ice_water_content_noise_stdev_kg_m03: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    error_checking.assert_is_greater(max_temp_increase_kelvins, 0.)
    error_checking.assert_is_greater(max_warm_layer_thickness_metres, 0.)

    surface_relative_humidity_limits = numpy.sort(
        surface_relative_humidity_limits
    )
    error_checking.assert_is_geq_numpy_array(
        surface_relative_humidity_limits, 0.
    )
    error_checking.assert_is_leq_numpy_array(
        surface_relative_humidity_limits, 1.
    )

    error_checking.assert_is_greater(max_moist_layer_thickness_metres, 0.)

    ozone_layer_thickness_limits_metres = numpy.sort(
        ozone_layer_thickness_limits_metres
    )
    error_checking.assert_is_greater_numpy_array(
        ozone_layer_thickness_limits_metres, 0.
    )

    ozone_layer_center_limits_metres = numpy.sort(
        ozone_layer_center_limits_metres
    )
    error_checking.assert_is_greater_numpy_array(
        ozone_layer_center_limits_metres, 0.
    )

    error_checking.assert_is_greater(max_ozone_mixing_ratio_kg_kg01, 0.)
    error_checking.assert_is_greater(ozone_mixing_ratio_noise_stdev_kg_kg01, 0.)
    error_checking.assert_is_greater(max_num_liquid_cloud_layers, 0)
    error_checking.assert_is_leq(max_num_liquid_cloud_layers, 10)
    error_checking.assert_is_greater(
        max_liquid_cloud_layer_thickness_metres, 0.
    )
    error_checking.assert_is_greater(max_liquid_water_content_kg_m03, 0.)
    error_checking.assert_is_greater(
        liquid_water_content_noise_stdev_kg_m03, 0.
    )
    error_checking.assert_is_greater(max_num_ice_cloud_layers, 0)
    error_checking.assert_is_leq(max_num_ice_cloud_layers, 10)
    error_checking.assert_is_greater(max_ice_cloud_layer_thickness_metres, 0.)
    error_checking.assert_is_greater(max_ice_water_content_kg_m03, 0.)
    error_checking.assert_is_greater(ice_water_content_noise_stdev_kg_m03, 0.)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    # Do actual stuff.
    print('Reading data from: "{0:s}"...'.format(input_file_name))
    gfs_table_xarray = xarray.open_dataset(input_file_name)

    num_times = len(
        gfs_table_xarray.coords[prepare_gfs_for_rrtm.TIME_DIMENSION]
    )
    num_sites = len(
        gfs_table_xarray.coords[prepare_gfs_for_rrtm.SITE_DIMENSION]
    )

    # num_profiles = num_times * num_sites
    # if num_profiles_to_perturb < num_profiles:
    #     num_sites_to_perturb = int(numpy.round(
    #         float(num_profiles_to_perturb) / num_times
    #     ))
    #
    #     site_indices = numpy.linspace(
    #         0, num_sites - 1, num=num_sites, dtype=int
    #     )
    #     site_indices = numpy.random.choice(
    #         site_indices, size=num_sites_to_perturb, replace=False
    #     )
    #     gfs_table_xarray = gfs_table_xarray.isel(
    #         indexers={prepare_gfs_for_rrtm.SITE_DIMENSION: site_indices}
    #     )
    #
    # num_sites = len(
    #     gfs_table_xarray.coords[prepare_gfs_for_rrtm.SITE_DIMENSION]
    # )

    print(SEPARATOR_STRING)

    for i in range(num_times):
        for j in range(num_sites):
            if numpy.mod(j, 10) == 0:
                print((
                    'Have perturbed profiles for {0:d} of {1:d} sites at '
                    '{2:d}th of {3:d} time steps...'
                ).format(
                    j, num_sites, i + 1, num_times
                ))

            perturbation_flags = numpy.random.uniform(low=0., high=1., size=5)
            perturbation_flags = numpy.round(perturbation_flags).astype(bool)
            
            if not numpy.any(perturbation_flags):
                this_index = numpy.random.random_integers(low=0, high=4, size=1)
                perturbation_flags[this_index] = True

            # Create fictitious ozone layer.
            if perturbation_flags[0]:
                gfs_table_xarray = _create_ozone_layer(
                    gfs_table_xarray=gfs_table_xarray,
                    time_index=i, site_index=j,
                    thickness_limits_metres=ozone_layer_thickness_limits_metres,
                    center_limits_metres=ozone_layer_center_limits_metres,
                    max_mixing_ratio_kg_kg01=max_ozone_mixing_ratio_kg_kg01,
                    mixing_ratio_noise_stdev_kg_kg01=
                    ozone_mixing_ratio_noise_stdev_kg_kg01
                )

            # Create fictitious surface-based warm layer.
            if perturbation_flags[1]:
                gfs_table_xarray = _create_surface_based_warm_layer(
                    gfs_table_xarray=gfs_table_xarray,
                    time_index=i, site_index=j,
                    max_layer_thickness_metres=max_warm_layer_thickness_metres,
                    max_temp_increase_kelvins=max_temp_increase_kelvins
                )

            # Create fictitious surface-based moist layer.
            if perturbation_flags[2]:
                gfs_table_xarray = _create_surface_based_moist_layer(
                    gfs_table_xarray=gfs_table_xarray,
                    time_index=i, site_index=j,
                    max_layer_thickness_metres=max_moist_layer_thickness_metres,
                    surface_relative_humidity_limits=
                    surface_relative_humidity_limits
                )

            # Create fictitious liquid cloud.
            if perturbation_flags[3]:
                gfs_table_xarray = _create_cloud(
                    gfs_table_xarray=gfs_table_xarray,
                    time_index=i, site_index=j,
                    max_num_cloud_layers=max_num_liquid_cloud_layers,
                    max_layer_thickness_metres=
                    max_liquid_cloud_layer_thickness_metres,
                    max_water_content_kg_m03=max_liquid_water_content_kg_m03,
                    water_content_noise_stdev_kg_m03=
                    liquid_water_content_noise_stdev_kg_m03,
                    liquid_flag=True
                )

            # Create fictitious ice cloud.
            if perturbation_flags[4]:
                gfs_table_xarray = _create_cloud(
                    gfs_table_xarray=gfs_table_xarray,
                    time_index=i, site_index=j,
                    max_num_cloud_layers=max_num_ice_cloud_layers,
                    max_layer_thickness_metres=
                    max_ice_cloud_layer_thickness_metres,
                    max_water_content_kg_m03=max_ice_water_content_kg_m03,
                    water_content_noise_stdev_kg_m03=
                    ice_water_content_noise_stdev_kg_m03,
                    liquid_flag=False
                )

        print((
            'Have perturbed profiles for all {0:d} sites at '
            '{1:d}th of {2:d} time steps!'
        ).format(
            num_sites, i + 1, num_times
        ))

    print(SEPARATOR_STRING)
    print('Writing perturbed profiles to: "{0:s}"...'.format(output_file_name))
    gfs_table_xarray.to_netcdf(
        path=output_file_name, mode='w', format='NETCDF3_64BIT'
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        max_temp_increase_kelvins=getattr(
            INPUT_ARG_OBJECT, MAX_TEMP_INCREASE_ARG_NAME
        ),
        max_warm_layer_thickness_metres=getattr(
            INPUT_ARG_OBJECT, MAX_WARM_LAYER_THICKNESS_ARG_NAME
        ),
        surface_relative_humidity_limits=numpy.array(
            getattr(INPUT_ARG_OBJECT, SURFACE_RH_LIMITS_ARG_NAME), dtype=float
        ),
        max_moist_layer_thickness_metres=getattr(
            INPUT_ARG_OBJECT, MAX_MOIST_LAYER_THICKNESS_ARG_NAME
        ),
        ozone_layer_thickness_limits_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, OZONE_LAYER_THICKNESS_LIMITS_ARG_NAME),
            dtype=float
        ),
        ozone_layer_center_limits_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, OZONE_LAYER_CENTER_LIMITS_ARG_NAME),
            dtype=float
        ),
        max_ozone_mixing_ratio_kg_kg01=getattr(
            INPUT_ARG_OBJECT, MAX_OZONE_MIXING_RATIO_ARG_NAME
        ),
        ozone_mixing_ratio_noise_stdev_kg_kg01=getattr(
            INPUT_ARG_OBJECT, OZONE_MIXING_RATIO_NOISE_ARG_NAME
        ),
        max_num_liquid_cloud_layers=getattr(
            INPUT_ARG_OBJECT, MAX_NUM_LIQUID_CLOUDS_ARG_NAME
        ),
        max_liquid_cloud_layer_thickness_metres=getattr(
            INPUT_ARG_OBJECT, MAX_LIQUID_CLOUD_THICKNESS_ARG_NAME
        ),
        max_liquid_water_content_kg_m03=getattr(
            INPUT_ARG_OBJECT, MAX_LIQUID_WATER_CONTENT_ARG_NAME
        ),
        liquid_water_content_noise_stdev_kg_m03=getattr(
            INPUT_ARG_OBJECT, LIQUID_WATER_CONTENT_NOISE_ARG_NAME
        ),
        max_num_ice_cloud_layers=getattr(
            INPUT_ARG_OBJECT, MAX_NUM_ICE_CLOUDS_ARG_NAME
        ),
        max_ice_cloud_layer_thickness_metres=getattr(
            INPUT_ARG_OBJECT, MAX_ICE_CLOUD_THICKNESS_ARG_NAME
        ),
        max_ice_water_content_kg_m03=getattr(
            INPUT_ARG_OBJECT, MAX_ICE_WATER_CONTENT_ARG_NAME
        ),
        ice_water_content_noise_stdev_kg_m03=getattr(
            INPUT_ARG_OBJECT, ICE_WATER_CONTENT_NOISE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
