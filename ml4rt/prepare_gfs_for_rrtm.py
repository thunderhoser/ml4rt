"""Prepares GFS data for input to the RRTM."""

import os
import sys
import argparse
import numpy
import xarray
from scipy.interpolate import interp1d

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import moisture_conversions as moisture_conv
import temperature_conversions as temperature_conv
import longitude_conversion as longitude_conv
import file_system_utils
import error_checking
import rrtm_io
import example_utils
import interp_rap_profiles

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

NOISE_STDEV_FRACTIONAL = 0.05

SECONDS_TO_HOURS = 1. / 3600
PASCALS_TO_MB = 0.01
KG_TO_GRAMS = 1e3
UNITLESS_TO_PERCENT = 100.
PERCENT_TO_UNITLESS = 0.01
DUMMY_DOWN_SURFACE_FLUX_W_M02 = 1000.
DRY_AIR_GAS_CONSTANT_J_KG01_K01 = 287.04

SITE_DIMENSION_ORIG = 'site_index'
TIME_DIMENSION_ORIG = 'valid_time_unix_sec'
HEIGHT_DIMENSION_ORIG = 'pfull'

SITE_DIMENSION = 'sites'
TIME_DIMENSION = 'valid_time_unix_sec'
HEIGHT_DIMENSION = 'height_bin'
HEIGHT_AT_EDGE_DIMENSION = 'height_bin_edge'

LATITUDE_KEY_ORIG_DEG_N = 'lat'
LONGITUDE_KEY_ORIG_DEG_E = 'lon'
DELTA_HEIGHT_KEY_ORIG_METRES = 'delz'
DELTA_PRESSURE_KEY_ORIG_PASCALS = 'dpres'
SURFACE_PRESSURE_KEY_ORIG_PASCALS = 'pressfc'

CLOUD_FRACTION_KEY_ORIG = 'cld_amt'
CLOUD_WATER_MIXR_KEY_ORIG_KG_KG01 = 'clwmr'
RAIN_MIXR_KEY_ORIG_KG_KG01 = 'rwmr'
CLOUD_ICE_MIXR_KEY_ORIG_KG_KG01 = 'icmr'
GRAUPEL_MIXR_KEY_ORIG_KG_KG01 = 'grle'
SNOW_MIXR_KEY_ORIG_KG_KG01 = 'snmr'
OZONE_MIXR_KEY_ORIG_KG_KG01 = 'o3mr'
TEMPERATURE_KEY_ORIG_KELVINS = 'tmp'
VAPOUR_MIXR_KEY_ORIG_KG_KG01 = 'r0'
SPECIFIC_HUMIDITY_KEY_ORIG_KG_KG01 = 'spfh'
ALBEDO_KEY_ORIG_PERCENT = 'albdo_ave'
PRESSURE_KEY_ORIG_PASCALS = 'pressure_pa'
PRESSURE_AT_EDGE_KEY_ORIG_PASCALS = 'pressure_at_edge_pa'

O2_MIXR_KEY_ORIG_KG_KG01 = 'o2_mixing_ratio_kg_kg01'
CO2_MIXR_KEY_ORIG_KG_KG01 = 'co2_mixing_ratio_kg_kg01'
CH4_MIXR_KEY_ORIG_KG_KG01 = 'ch4_mixing_ratio_kg_kg01'
N2O_MIXR_KEY_ORIG_KG_KG01 = 'n2o_mixing_ratio_kg_kg01'
LIQUID_EFF_RADIUS_KEY_ORIG_METRES = 'liquid_eff_radius_metres'
ICE_EFF_RADIUS_KEY_ORIG_METRES = 'ice_eff_radius_metres'
AEROSOL_OPTICAL_DEPTH_KEY_ORIG_METRES = 'aerosol_optical_depth_metres'
AEROSOL_ALBEDO_KEY_ORIG = 'aerosol_albedo'
AEROSOL_ASYMMETRY_PARAM_KEY_ORIG = 'aerosol_asymmetry_param'

MAIN_KEYS_ORIG = [
    CLOUD_FRACTION_KEY_ORIG, CLOUD_WATER_MIXR_KEY_ORIG_KG_KG01,
    RAIN_MIXR_KEY_ORIG_KG_KG01, CLOUD_ICE_MIXR_KEY_ORIG_KG_KG01,
    GRAUPEL_MIXR_KEY_ORIG_KG_KG01, SNOW_MIXR_KEY_ORIG_KG_KG01,
    OZONE_MIXR_KEY_ORIG_KG_KG01, TEMPERATURE_KEY_ORIG_KELVINS,
    VAPOUR_MIXR_KEY_ORIG_KG_KG01
]
CONSERVE_JUMP_KEYS_ORIG = [
    CLOUD_WATER_MIXR_KEY_ORIG_KG_KG01, RAIN_MIXR_KEY_ORIG_KG_KG01,
    CLOUD_ICE_MIXR_KEY_ORIG_KG_KG01, GRAUPEL_MIXR_KEY_ORIG_KG_KG01,
    SNOW_MIXR_KEY_ORIG_KG_KG01
]

SITE_NAME_KEY = 'dsite'
LATITUDE_KEY_DEG_N = 'mlat'
LONGITUDE_KEY_DEG_E = 'mlon'
FORECAST_HOUR_KEY = 'forecast'
HEIGHT_KEY_M_AGL = 'height'
TEMPERATURE_KEY_CELSIUS = 't0'
VAPOUR_MIXR_KEY_G_KG01 = 'r0'
LIQUID_WATER_CONTENT_KEY_KG_KG01 = 'lwc0'
ICE_WATER_PATH_KEY_KG_M02 = 'iwp0'
DOWN_SURFACE_FLUX_KEY_W_M02 = 'dswsfc0'
UP_SURFACE_FLUX_KEY_W_M02 = 'uswsfc0'
CLOUD_FRACTION_KEY_PERCENT = 'totcc0'

ORIG_TO_NEW_KEY_DICT = {
    CLOUD_FRACTION_KEY_ORIG: CLOUD_FRACTION_KEY_PERCENT,
    CLOUD_WATER_MIXR_KEY_ORIG_KG_KG01: LIQUID_WATER_CONTENT_KEY_KG_KG01,
    RAIN_MIXR_KEY_ORIG_KG_KG01: 'rain_mixing_ratio_kg_kg01',
    CLOUD_ICE_MIXR_KEY_ORIG_KG_KG01: ICE_WATER_PATH_KEY_KG_M02,
    GRAUPEL_MIXR_KEY_ORIG_KG_KG01: 'graupel_mixing_ratio_kg_kg01',
    SNOW_MIXR_KEY_ORIG_KG_KG01: 'snow_mixing_ratio_kg_kg01',
    OZONE_MIXR_KEY_ORIG_KG_KG01: 'ozone_mixing_ratio_kg_kg01',
    TEMPERATURE_KEY_ORIG_KELVINS: TEMPERATURE_KEY_CELSIUS,
    VAPOUR_MIXR_KEY_ORIG_KG_KG01: VAPOUR_MIXR_KEY_G_KG01,
    PRESSURE_KEY_ORIG_PASCALS: 'p0',
    PRESSURE_AT_EDGE_KEY_ORIG_PASCALS: 'p0_edge',
    O2_MIXR_KEY_ORIG_KG_KG01: O2_MIXR_KEY_ORIG_KG_KG01,
    CO2_MIXR_KEY_ORIG_KG_KG01: CO2_MIXR_KEY_ORIG_KG_KG01,
    CH4_MIXR_KEY_ORIG_KG_KG01: CH4_MIXR_KEY_ORIG_KG_KG01,
    N2O_MIXR_KEY_ORIG_KG_KG01: N2O_MIXR_KEY_ORIG_KG_KG01,
    LIQUID_EFF_RADIUS_KEY_ORIG_METRES: LIQUID_EFF_RADIUS_KEY_ORIG_METRES,
    ICE_EFF_RADIUS_KEY_ORIG_METRES: ICE_EFF_RADIUS_KEY_ORIG_METRES,
    AEROSOL_OPTICAL_DEPTH_KEY_ORIG_METRES:
        AEROSOL_OPTICAL_DEPTH_KEY_ORIG_METRES,
    AEROSOL_ALBEDO_KEY_ORIG: AEROSOL_ALBEDO_KEY_ORIG,
    AEROSOL_ASYMMETRY_PARAM_KEY_ORIG: AEROSOL_ASYMMETRY_PARAM_KEY_ORIG
}

INPUT_FILE_ARG_NAME = 'input_file_name'
NEW_HEIGHTS_ARG_NAME = 'new_heights_m_agl'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (created by subset_gfs_from_jebb.py), containing GFS '
    'data for one init time.'
)
NEW_HEIGHTS_HELP_STRING = 'Heights (metres above ground level) in new grid.'
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Processed data (in the format required by the RRTM) '
    'will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEW_HEIGHTS_ARG_NAME, type=int, nargs='+', required=True,
    help=NEW_HEIGHTS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _interp_data_one_profile(
        orig_gfs_table_xarray, time_index, site_index, new_heights_m_agl,
        new_heights_at_edges_m_agl, interp_data_dict):
    """Interpolates data from one profile to new heights.

    H = number of heights in new grid

    :param orig_gfs_table_xarray: xarray table with GFS data in original (Jebb)
        format.
    :param time_index: Will interpolate data for the [i]th time in the table,
        where i is this index.
    :param site_index: Will interpolate data for the [j]th site in the table,
        where j is this index.
    :param new_heights_m_agl: length-H numpy array of heights (metres above
        ground level).
    :param new_heights_at_edges_m_agl: length-(H + 1) numpy array of heights
        (metres above ground level).
    :param interp_data_dict: Dictionary, where each key is a variable name and
        the corresponding value is a 3-D numpy array (time x site x new_height).
    :return: interp_data_dict: Same as input but with new values for the given
        time and site.
    """

    i = time_index
    j = site_index

    orig_pressure_diffs_pa = numpy.cumsum(numpy.flip(
        orig_gfs_table_xarray[DELTA_PRESSURE_KEY_ORIG_PASCALS].values[i, :, j]
    ))
    orig_pressures_pa = (
        orig_gfs_table_xarray[SURFACE_PRESSURE_KEY_ORIG_PASCALS].values[i, j]
        - orig_pressure_diffs_pa
    )
    orig_heights_m_agl = numpy.cumsum(numpy.flip(
        -1 * orig_gfs_table_xarray[DELTA_HEIGHT_KEY_ORIG_METRES].values[i, :, j]
    ))

    log_offset = 1. + -1 * numpy.min(orig_pressures_pa)
    assert not numpy.isnan(log_offset)

    interp_object = interp1d(
        x=orig_heights_m_agl, y=numpy.log(orig_pressures_pa),
        kind='linear', bounds_error=False, assume_sorted=True,
        fill_value='extrapolate'
    )
    interp_data_dict[PRESSURE_KEY_ORIG_PASCALS][i, j, :] = (
        numpy.exp(interp_object(new_heights_m_agl)) - log_offset
    )

    interp_data_dict[PRESSURE_AT_EDGE_KEY_ORIG_PASCALS][i, j, :] = (
        numpy.exp(interp_object(new_heights_at_edges_m_agl))
    )

    for this_key in MAIN_KEYS_ORIG:
        if this_key in CONSERVE_JUMP_KEYS_ORIG:
            orig_values = numpy.flip(
                orig_gfs_table_xarray[this_key].values[i, :, j]
            )
            orig_data_matrix = numpy.expand_dims(orig_values, axis=0)

            interp_data_dict[this_key][i, j, :] = (
                interp_rap_profiles._interp_and_conserve_jumps(
                    orig_data_matrix=orig_data_matrix,
                    orig_heights_metres=orig_heights_m_agl,
                    new_heights_metres=new_heights_m_agl,
                    extrapolate=False
                )
            )

            continue

        orig_values = numpy.flip(
            orig_gfs_table_xarray[this_key].values[i, :, j]
        )
        log_offset = 1. + -1 * numpy.min(orig_values)
        assert not numpy.isnan(log_offset)

        interp_object = interp1d(
            x=orig_heights_m_agl, y=numpy.log(log_offset + orig_values),
            kind='linear', bounds_error=False, assume_sorted=True,
            fill_value=(orig_values[0], orig_values[-1])
        )
        interp_data_dict[this_key][i, j, :] = (
            numpy.exp(interp_object(new_heights_m_agl)) - log_offset
        )

    return interp_data_dict


def _add_trace_gases(orig_gfs_table_xarray, new_heights_m_agl,
                     interp_data_dict, test_mode=False):
    """Adds profiles of trace-gas mixing ratios.

    :param orig_gfs_table_xarray: xarray table with GFS data in original (Jebb)
        format.
    :param new_heights_m_agl: 1-D numpy array of heights (metres above ground
        level) in new grid.
    :param interp_data_dict: Dictionary, where each key is a variable name and
        the corresponding value is a 3-D numpy array
        (time x site x new_height).
    :param test_mode: Leave this alone.
    :return: interp_data_dict: Same as input but with additional keys for trace
        gases.
    :return: dummy_example_dict: Dictionary with fake learning examples, in
        format specified by `example_io.read_file`.
    """

    valid_times_unix_sec = (
        orig_gfs_table_xarray.coords[TIME_DIMENSION_ORIG].values
    )
    num_times = len(valid_times_unix_sec)
    num_sites = len(orig_gfs_table_xarray.coords[SITE_DIMENSION_ORIG].values)
    num_heights_new = len(new_heights_m_agl)

    latitudes_deg_n = orig_gfs_table_xarray[LATITUDE_KEY_ORIG_DEG_N].values
    longitudes_deg_e = orig_gfs_table_xarray[LONGITUDE_KEY_ORIG_DEG_E].values
    longitudes_deg_e = longitude_conv.convert_lng_positive_in_west(
        longitudes_deg_e, allow_nan=False
    )

    dummy_id_string_matrix = numpy.full(
        (num_times, num_sites), '', dtype=object
    )

    for i in range(num_times):
        for j in range(num_sites):
            dummy_id_string_matrix[i, j] = (
                'lat={0:09.6f}_long={1:010.6f}_zenith-angle-rad=01.000000_'
                'time={2:010d}_atmo=1_albedo=0.123456_'
                'temp-10m-kelvins=273.150000'
            ).format(
                latitudes_deg_n[j], longitudes_deg_e[j], valid_times_unix_sec[i]
            )

    dummy_id_strings = numpy.ravel(dummy_id_string_matrix).tolist()
    dummy_predictor_matrix = numpy.reshape(
        interp_data_dict[TEMPERATURE_KEY_ORIG_KELVINS],
        (num_times * num_sites, num_heights_new)
    )
    dummy_predictor_matrix = numpy.expand_dims(dummy_predictor_matrix, axis=-1)

    dummy_example_dict = {
        example_utils.EXAMPLE_IDS_KEY: dummy_id_strings,
        example_utils.HEIGHTS_KEY: new_heights_m_agl,
        example_utils.VECTOR_PREDICTOR_NAMES_KEY:
            [example_utils.TEMPERATURE_NAME],
        example_utils.VECTOR_PREDICTOR_VALS_KEY: dummy_predictor_matrix
    }
    dummy_example_dict = example_utils.add_trace_gases(
        example_dict=dummy_example_dict,
        noise_stdev_fractional=0. if test_mode else NOISE_STDEV_FRACTIONAL
    )

    o2_mixing_ratio_matrix_kg_kg01 = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.O2_MIXING_RATIO_NAME
    )
    interp_data_dict[O2_MIXR_KEY_ORIG_KG_KG01] = numpy.reshape(
        o2_mixing_ratio_matrix_kg_kg01,
        (num_times, num_sites, num_heights_new)
    )

    co2_mixing_ratio_matrix_kg_kg01 = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.CO2_MIXING_RATIO_NAME
    )
    interp_data_dict[CO2_MIXR_KEY_ORIG_KG_KG01] = numpy.reshape(
        co2_mixing_ratio_matrix_kg_kg01,
        (num_times, num_sites, num_heights_new)
    )

    ch4_mixing_ratio_matrix_kg_kg01 = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.CH4_MIXING_RATIO_NAME
    )
    interp_data_dict[CH4_MIXR_KEY_ORIG_KG_KG01] = numpy.reshape(
        ch4_mixing_ratio_matrix_kg_kg01,
        (num_times, num_sites, num_heights_new)
    )

    n2o_mixing_ratio_matrix_kg_kg01 = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.N2O_MIXING_RATIO_NAME
    )
    interp_data_dict[N2O_MIXR_KEY_ORIG_KG_KG01] = numpy.reshape(
        n2o_mixing_ratio_matrix_kg_kg01,
        (num_times, num_sites, num_heights_new)
    )

    return interp_data_dict, dummy_example_dict


def _convert_ice_mixr_to_path(orig_gfs_table_xarray, new_heights_m_agl,
                              interp_data_dict):
    """Converts ice-water mixing ratios to ice-water paths.

    :param orig_gfs_table_xarray: xarray table with GFS data in original (Jebb)
        format.
    :param new_heights_m_agl: 1-D numpy array of heights (metres above ground
        level) in new grid.
    :param interp_data_dict: Dictionary, where each key is a variable name and
        the corresponding value is a 3-D numpy array
        (time x site x new_height).
    :return: interp_data_dict: Same as input but with ice-water paths instead of
        mixing ratios.
    """

    num_times = len(orig_gfs_table_xarray.coords[TIME_DIMENSION_ORIG].values)
    num_sites = len(orig_gfs_table_xarray.coords[SITE_DIMENSION_ORIG].values)
    num_heights_new = len(new_heights_m_agl)

    vapour_pressure_matrix_pa = moisture_conv.mixing_ratio_to_vapour_pressure(
        mixing_ratios_kg_kg01=interp_data_dict[VAPOUR_MIXR_KEY_ORIG_KG_KG01],
        total_pressures_pascals=interp_data_dict[PRESSURE_KEY_ORIG_PASCALS]
    )
    virtual_temp_matrix_kelvins = (
        moisture_conv.temperature_to_virtual_temperature(
            temperatures_kelvins=interp_data_dict[TEMPERATURE_KEY_ORIG_KELVINS],
            total_pressures_pascals=interp_data_dict[PRESSURE_KEY_ORIG_PASCALS],
            vapour_pressures_pascals=vapour_pressure_matrix_pa
        )
    )
    air_density_matrix_kg_m03 = (
        interp_data_dict[PRESSURE_KEY_ORIG_PASCALS] /
        virtual_temp_matrix_kelvins
    )
    air_density_matrix_kg_m03 = (
        air_density_matrix_kg_m03 / DRY_AIR_GAS_CONSTANT_J_KG01_K01
    )
    ice_mixing_ratio_matrix_kg_m03 = (
        interp_data_dict[CLOUD_ICE_MIXR_KEY_ORIG_KG_KG01] *
        air_density_matrix_kg_m03
    )

    ice_mixing_ratio_matrix_kg_m03 = numpy.reshape(
        ice_mixing_ratio_matrix_kg_m03,
        (num_times * num_sites, num_heights_new)
    )
    ice_water_path_matrix_kg_m02 = rrtm_io._water_content_to_layerwise_path(
        water_content_matrix_kg_m03=ice_mixing_ratio_matrix_kg_m03,
        heights_m_agl=new_heights_m_agl
    )
    ice_water_path_matrix_kg_m02 = numpy.fliplr(numpy.cumsum(
        numpy.fliplr(ice_water_path_matrix_kg_m02), axis=1
    ))
    ice_water_path_matrix_kg_m02 = numpy.reshape(
        ice_water_path_matrix_kg_m02,
        (num_times, num_sites, num_heights_new)
    )
    interp_data_dict[CLOUD_ICE_MIXR_KEY_ORIG_KG_KG01] = (
        ice_water_path_matrix_kg_m02
    )

    return interp_data_dict


def _run(input_file_name, new_heights_m_agl, output_file_name):
    """Prepares GFS data for input to the RRTM.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param new_heights_m_agl: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    error_checking.assert_is_geq_numpy_array(new_heights_m_agl, 0)
    error_checking.assert_is_geq_numpy_array(numpy.diff(new_heights_m_agl), 0)
    new_heights_m_agl = new_heights_m_agl.astype(float)

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    # Read input file and convert specific humidity to vapour mixing ratio.
    print('Reading data from: "{0:s}"...'.format(input_file_name))
    orig_gfs_table_xarray = xarray.open_dataset(input_file_name)

    this_matrix = moisture_conv.specific_humidity_to_mixing_ratio(
        orig_gfs_table_xarray[SPECIFIC_HUMIDITY_KEY_ORIG_KG_KG01].values
    )
    orig_gfs_table_xarray = orig_gfs_table_xarray.assign({
        VAPOUR_MIXR_KEY_ORIG_KG_KG01: (
            orig_gfs_table_xarray[SPECIFIC_HUMIDITY_KEY_ORIG_KG_KG01].dims,
            this_matrix
        )
    })

    # Create metadata dict for output file.
    valid_times_unix_sec = (
        orig_gfs_table_xarray.coords[TIME_DIMENSION_ORIG].values
    )
    num_times = len(valid_times_unix_sec)
    num_sites = len(orig_gfs_table_xarray.coords[SITE_DIMENSION_ORIG].values)
    num_heights_new = len(new_heights_m_agl)

    new_metadata_dict = {
        TIME_DIMENSION: valid_times_unix_sec,
        HEIGHT_DIMENSION: numpy.linspace(
            0, num_heights_new - 1, num=num_heights_new, dtype=int
        ),
        HEIGHT_AT_EDGE_DIMENSION: numpy.linspace(
            0, num_heights_new, num=num_heights_new + 1, dtype=int
        ),
        SITE_DIMENSION: numpy.linspace(
            0, num_sites - 1, num=num_sites, dtype=int
        )
    }

    dummy_forecast_hours = 6 + numpy.round(
        SECONDS_TO_HOURS * (valid_times_unix_sec - valid_times_unix_sec[0])
    ).astype(int)

    # Start main data dict for output file.
    latitudes_deg_n = orig_gfs_table_xarray[LATITUDE_KEY_ORIG_DEG_N].values
    longitudes_deg_e = orig_gfs_table_xarray[LONGITUDE_KEY_ORIG_DEG_E].values
    longitudes_deg_e = longitude_conv.convert_lng_positive_in_west(
        longitudes_deg_e, allow_nan=False
    )
    site_names = [
        'latitude-deg-n={0:.6f}_longitude-deg-e={1:.6f}'.format(y, x)
        for y, x in zip(latitudes_deg_n, longitudes_deg_e)
    ]

    new_data_dict = {
        SITE_NAME_KEY: ((SITE_DIMENSION,), site_names),
        LATITUDE_KEY_DEG_N: ((SITE_DIMENSION,), latitudes_deg_n),
        LONGITUDE_KEY_DEG_E: ((SITE_DIMENSION,), longitudes_deg_e),
        FORECAST_HOUR_KEY: ((TIME_DIMENSION,), dummy_forecast_hours),
        HEIGHT_KEY_M_AGL: ((HEIGHT_DIMENSION,), new_heights_m_agl)
    }

    # Interpolate main variables to new heights.
    interp_data_dict = dict()
    for this_key in MAIN_KEYS_ORIG + [PRESSURE_KEY_ORIG_PASCALS]:
        interp_data_dict[this_key] = numpy.full(
            (num_times, num_sites, num_heights_new), numpy.nan
        )

    interp_data_dict[PRESSURE_AT_EDGE_KEY_ORIG_PASCALS] = numpy.full(
        (num_times, num_sites, num_heights_new + 1), numpy.nan
    )
    new_heights_at_edges_m_agl = example_utils.get_grid_cell_edges(
        new_heights_m_agl
    )

    for i in range(num_times):
        for j in range(num_sites):
            if numpy.mod(j, 10) == 0:
                print((
                    'Interpolating data to new heights for forecast hour {0:d},'
                    ' site {1:d} of {2:d}...'
                ).format(
                    dummy_forecast_hours[i], j + 1, num_sites
                ))

            interp_data_dict = _interp_data_one_profile(
                orig_gfs_table_xarray=orig_gfs_table_xarray,
                time_index=i, site_index=j,
                new_heights_m_agl=new_heights_m_agl,
                new_heights_at_edges_m_agl=new_heights_at_edges_m_agl,
                interp_data_dict=interp_data_dict
            )

        print((
            'Have interpolated data to new heights for all {0:d} sites at '
            'forecast hour {1:d}!'
        ).format(
            num_sites, dummy_forecast_hours[i]
        ))
        print(SEPARATOR_STRING)

    # Add other variables.
    print('Adding trace gases...')
    interp_data_dict, dummy_example_dict = _add_trace_gases(
        orig_gfs_table_xarray=orig_gfs_table_xarray,
        new_heights_m_agl=new_heights_m_agl,
        interp_data_dict=interp_data_dict
    )

    print('Adding effective radii of liquid and ice particles...')
    dummy_example_dict = example_utils.add_effective_radii(
        example_dict=dummy_example_dict,
        ice_noise_stdev_fractional=NOISE_STDEV_FRACTIONAL
    )
    liquid_eff_radius_matrix_metres = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.LIQUID_EFF_RADIUS_NAME
    )
    interp_data_dict[LIQUID_EFF_RADIUS_KEY_ORIG_METRES] = numpy.reshape(
        liquid_eff_radius_matrix_metres,
        (num_times, num_sites, num_heights_new)
    )

    ice_eff_radius_matrix_metres = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.ICE_EFF_RADIUS_NAME
    )
    interp_data_dict[ICE_EFF_RADIUS_KEY_ORIG_METRES] = numpy.reshape(
        ice_eff_radius_matrix_metres,
        (num_times, num_sites, num_heights_new)
    )

    print('Adding aerosols...')
    dummy_example_dict = example_utils.add_aerosols(dummy_example_dict)
    aerosol_optical_depth_matrix_metres = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.AEROSOL_OPTICAL_DEPTH_NAME
    )
    interp_data_dict[AEROSOL_OPTICAL_DEPTH_KEY_ORIG_METRES] = numpy.reshape(
        aerosol_optical_depth_matrix_metres,
        (num_times, num_sites, num_heights_new)
    )

    aerosol_albedo_matrix = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.AEROSOL_ALBEDO_NAME
    )
    interp_data_dict[AEROSOL_ALBEDO_KEY_ORIG] = numpy.reshape(
        aerosol_albedo_matrix,
        (num_times, num_sites, num_heights_new)
    )

    aerosol_asymmetry_param_matrix = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.AEROSOL_ASYMMETRY_PARAM_NAME
    )
    interp_data_dict[AEROSOL_ASYMMETRY_PARAM_KEY_ORIG] = numpy.reshape(
        aerosol_asymmetry_param_matrix,
        (num_times, num_sites, num_heights_new)
    )

    # TODO(thunderhoser): Will likely need to convert other vars in same way.
    print('Converting ice-water mixing ratios (kg/kg) to paths (kg/m^2)...')
    interp_data_dict = _convert_ice_mixr_to_path(
        orig_gfs_table_xarray=orig_gfs_table_xarray,
        new_heights_m_agl=new_heights_m_agl,
        interp_data_dict=interp_data_dict
    )

    print('Converting pressures from Pa to mb...')
    interp_data_dict[PRESSURE_KEY_ORIG_PASCALS] *= PASCALS_TO_MB
    interp_data_dict[PRESSURE_AT_EDGE_KEY_ORIG_PASCALS] *= PASCALS_TO_MB

    print('Converting temperatures from Kelvins to deg C...')
    interp_data_dict[TEMPERATURE_KEY_ORIG_KELVINS] = (
        temperature_conv.kelvins_to_celsius(
            interp_data_dict[TEMPERATURE_KEY_ORIG_KELVINS]
        )
    )

    print('Converting vapour mixing ratios from kg/kg to g/kg...')
    interp_data_dict[VAPOUR_MIXR_KEY_ORIG_KG_KG01] *= KG_TO_GRAMS

    print('Converting cloud fractions from unitless to percent...')
    interp_data_dict[CLOUD_FRACTION_KEY_ORIG] *= UNITLESS_TO_PERCENT

    print(
        'Converting surface albedo to dummy upwelling and downwelling fluxes...'
    )
    albedo_matrix = (
        PERCENT_TO_UNITLESS *
        orig_gfs_table_xarray[ALBEDO_KEY_ORIG_PERCENT].values
    )
    up_flux_matrix_w_m02 = albedo_matrix * DUMMY_DOWN_SURFACE_FLUX_W_M02
    down_flux_matrix_w_m02 = numpy.full(
        up_flux_matrix_w_m02.shape, DUMMY_DOWN_SURFACE_FLUX_W_M02
    )

    # Create xarray table with interpolated data.
    for this_key_orig in interp_data_dict:
        if this_key_orig == PRESSURE_AT_EDGE_KEY_ORIG_PASCALS:
            these_dim = (
                TIME_DIMENSION, SITE_DIMENSION, HEIGHT_AT_EDGE_DIMENSION
            )
        else:
            these_dim = (TIME_DIMENSION, SITE_DIMENSION, HEIGHT_DIMENSION)

        this_key = ORIG_TO_NEW_KEY_DICT[this_key_orig]
        new_data_dict[this_key] = (these_dim, interp_data_dict[this_key_orig])

    these_dim = (TIME_DIMENSION, SITE_DIMENSION)
    new_data_dict.update({
        UP_SURFACE_FLUX_KEY_W_M02: (these_dim, up_flux_matrix_w_m02),
        DOWN_SURFACE_FLUX_KEY_W_M02: (these_dim, down_flux_matrix_w_m02)
    })

    new_gfs_table_xarray = xarray.Dataset(
        data_vars=new_data_dict, coords=new_metadata_dict
    )

    for this_key in new_gfs_table_xarray.variables:
        if this_key == SITE_NAME_KEY:
            continue

        print(this_key)
        print(numpy.any(numpy.isnan(new_gfs_table_xarray[this_key].values)))

    for this_key in new_gfs_table_xarray.variables:
        if this_key == SITE_NAME_KEY:
            continue

        error_checking.assert_is_numpy_array_without_nan(
            new_gfs_table_xarray[this_key].values
        )

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    new_gfs_table_xarray.to_netcdf(
        path=output_file_name, mode='w', format='NETCDF3_64BIT'
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        new_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEW_HEIGHTS_ARG_NAME), dtype=int
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
