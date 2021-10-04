"""Prepares GFS data for input to the RRTM."""

import argparse
import numpy
import xarray
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import moisture_conversions as moisture_conv
from gewittergefahr.gg_utils import longitude_conversion as longitude_conv
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import rrtm_io
from ml4rt.utils import example_utils
from ml4rt.scripts import interp_rap_profiles

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

PROFILE_NOISE_STDEV_FRACTIONAL = 0.05
INDIV_NOISE_STDEV_FRACTIONAL = 0.005

SECONDS_TO_HOURS = 1. / 3600
UNITLESS_TO_PERCENT = 100.
PERCENT_TO_UNITLESS = 0.01
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

O2_CONCENTRATION_KEY_ORIG_PPMV = 'o2_concentration_ppmv'
CO2_CONCENTRATION_KEY_ORIG_PPMV = 'co2_concentration_ppmv'
CH4_CONCENTRATION_KEY_ORIG_PPMV = 'ch4_concentration_ppmv'
N2O_CONCENTRATION_KEY_ORIG_PPMV = 'n2o_concentration_ppmv'
LIQUID_EFF_RADIUS_KEY_ORIG_METRES = 'liquid_eff_radius_metres'
ICE_EFF_RADIUS_KEY_ORIG_METRES = 'ice_eff_radius_metres'
AEROSOL_EXTINCTION_KEY_ORIG_METRES01 = 'aerosol_extinction_metres01'
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
SCALAR_KEYS_ORIG = [AEROSOL_ALBEDO_KEY_ORIG, AEROSOL_ASYMMETRY_PARAM_KEY_ORIG]

SITE_NAME_KEY = 'site_name'
LATITUDE_KEY_DEG_N = 'site_latitude_deg_n'
LONGITUDE_KEY_DEG_E = 'site_longitude_deg_e'
FORECAST_HOUR_KEY = 'forecast_hour'
HEIGHT_KEY_M_AGL = 'height_m_agl'
PRESSURE_KEY_PASCALS = 'pressure_pascals'
PRESSURE_AT_EDGE_KEY_PASCALS = 'pressure_at_edge_pascals'
TEMPERATURE_KEY_KELVINS = 'temperature_kelvins'
VAPOUR_MIXR_KEY_KG_KG01 = 'vapour_mixing_ratio_kg_kg01'

# TODO(thunderhoser): Should I be adding paths together?
LIQUID_WATER_PATH_KEY_KG_M02 = 'layerwise_liquid_water_path_kg_m02'
RAIN_WATER_PATH_KEY_KG_M02 = 'layerwise_rain_water_path_kg_m02'
ICE_WATER_PATH_KEY_KG_M02 = 'layerwise_ice_water_path_kg_m02'
GRAUPEL_PATH_KEY_KG_M02 = 'layerwise_graupel_path_kg_m02'
SNOW_PATH_KEY_KG_M02 = 'layerwise_snow_path_kg_m02'
OZONE_MIXR_KEY_KG_KG01 = 'ozone_mixing_ratio_kg_kg01'
CLOUD_FRACTION_KEY_PERCENT = 'cloud_fraction_percent'
ALBEDO_KEY = 'surface_albedo'

ORIG_TO_NEW_KEY_DICT = {
    CLOUD_FRACTION_KEY_ORIG: CLOUD_FRACTION_KEY_PERCENT,
    CLOUD_WATER_MIXR_KEY_ORIG_KG_KG01: LIQUID_WATER_PATH_KEY_KG_M02,
    RAIN_MIXR_KEY_ORIG_KG_KG01: RAIN_WATER_PATH_KEY_KG_M02,
    CLOUD_ICE_MIXR_KEY_ORIG_KG_KG01: ICE_WATER_PATH_KEY_KG_M02,
    GRAUPEL_MIXR_KEY_ORIG_KG_KG01: GRAUPEL_PATH_KEY_KG_M02,
    SNOW_MIXR_KEY_ORIG_KG_KG01: SNOW_PATH_KEY_KG_M02,
    OZONE_MIXR_KEY_ORIG_KG_KG01: OZONE_MIXR_KEY_KG_KG01,
    TEMPERATURE_KEY_ORIG_KELVINS: TEMPERATURE_KEY_KELVINS,
    VAPOUR_MIXR_KEY_ORIG_KG_KG01: VAPOUR_MIXR_KEY_KG_KG01,
    PRESSURE_KEY_ORIG_PASCALS: PRESSURE_KEY_PASCALS,
    PRESSURE_AT_EDGE_KEY_ORIG_PASCALS: PRESSURE_AT_EDGE_KEY_PASCALS,
    O2_CONCENTRATION_KEY_ORIG_PPMV: O2_CONCENTRATION_KEY_ORIG_PPMV,
    CO2_CONCENTRATION_KEY_ORIG_PPMV: CO2_CONCENTRATION_KEY_ORIG_PPMV,
    CH4_CONCENTRATION_KEY_ORIG_PPMV: CH4_CONCENTRATION_KEY_ORIG_PPMV,
    N2O_CONCENTRATION_KEY_ORIG_PPMV: N2O_CONCENTRATION_KEY_ORIG_PPMV,
    LIQUID_EFF_RADIUS_KEY_ORIG_METRES: LIQUID_EFF_RADIUS_KEY_ORIG_METRES,
    ICE_EFF_RADIUS_KEY_ORIG_METRES: ICE_EFF_RADIUS_KEY_ORIG_METRES,
    AEROSOL_EXTINCTION_KEY_ORIG_METRES01: AEROSOL_EXTINCTION_KEY_ORIG_METRES01,
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

    orig_pressures_pa = numpy.flip(numpy.cumsum(
        orig_gfs_table_xarray[DELTA_PRESSURE_KEY_ORIG_PASCALS].values[i, :, j]
    ))
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

        bottom_value = numpy.log(log_offset + orig_values[0])
        top_value = numpy.log(log_offset + orig_values[1])

        interp_object = interp1d(
            x=orig_heights_m_agl, y=numpy.log(log_offset + orig_values),
            kind='linear', bounds_error=False, assume_sorted=True,
            fill_value=(bottom_value, top_value)
        )
        interp_data_dict[this_key][i, j, :] = (
            numpy.exp(interp_object(new_heights_m_agl)) - log_offset
        )

    return interp_data_dict


def _add_trace_gases(orig_gfs_table_xarray, new_heights_m_agl,
                     interp_data_dict, test_mode=False):
    """Adds profiles of trace-gas concentrations.

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
    dummy_vector_predictor_matrix = numpy.reshape(
        interp_data_dict[TEMPERATURE_KEY_ORIG_KELVINS],
        (num_times * num_sites, num_heights_new)
    )
    dummy_vector_predictor_matrix = numpy.expand_dims(
        dummy_vector_predictor_matrix, axis=-1
    )

    dummy_example_dict = {
        example_utils.EXAMPLE_IDS_KEY: dummy_id_strings,
        example_utils.HEIGHTS_KEY: new_heights_m_agl,
        example_utils.VECTOR_PREDICTOR_NAMES_KEY:
            [example_utils.TEMPERATURE_NAME],
        example_utils.VECTOR_PREDICTOR_VALS_KEY: dummy_vector_predictor_matrix,
        example_utils.SCALAR_PREDICTOR_NAMES_KEY:
            [example_utils.ZENITH_ANGLE_NAME],
        example_utils.SCALAR_PREDICTOR_VALS_KEY:
            dummy_vector_predictor_matrix[:, 0, :]
    }
    dummy_example_dict = example_utils.add_trace_gases(
        example_dict=dummy_example_dict,
        profile_noise_stdev_fractional=
        0. if test_mode else PROFILE_NOISE_STDEV_FRACTIONAL,
        indiv_noise_stdev_fractional=
        0. if test_mode else INDIV_NOISE_STDEV_FRACTIONAL
    )

    o2_concentration_matrix_ppmv = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.O2_CONCENTRATION_NAME
    )
    interp_data_dict[O2_CONCENTRATION_KEY_ORIG_PPMV] = numpy.reshape(
        o2_concentration_matrix_ppmv,
        (num_times, num_sites, num_heights_new)
    )

    co2_concentration_matrix_ppmv = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.CO2_CONCENTRATION_NAME
    )
    interp_data_dict[CO2_CONCENTRATION_KEY_ORIG_PPMV] = numpy.reshape(
        co2_concentration_matrix_ppmv,
        (num_times, num_sites, num_heights_new)
    )

    ch4_concentration_matrix_ppmv = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.CH4_CONCENTRATION_NAME
    )
    interp_data_dict[CH4_CONCENTRATION_KEY_ORIG_PPMV] = numpy.reshape(
        ch4_concentration_matrix_ppmv,
        (num_times, num_sites, num_heights_new)
    )

    n2o_concentration_matrix_ppmv = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.N2O_CONCENTRATION_NAME
    )
    interp_data_dict[N2O_CONCENTRATION_KEY_ORIG_PPMV] = numpy.reshape(
        n2o_concentration_matrix_ppmv,
        (num_times, num_sites, num_heights_new)
    )

    return interp_data_dict, dummy_example_dict


def _mixing_ratio_to_layerwise_path(orig_gfs_table_xarray, new_heights_m_agl,
                                    interp_data_dict, variable_name):
    """Converts mixing ratios (kg kg^-1) to layerwise paths (kg m^-2).

    :param orig_gfs_table_xarray: xarray table with GFS data in original (Jebb)
        format.
    :param new_heights_m_agl: 1-D numpy array of heights (metres above ground
        level) in new grid.
    :param interp_data_dict: Dictionary, where each key is a variable name and
        the corresponding value is a 3-D numpy array
        (time x site x new_height).
    :param variable_name: Name of variable to convert.  This must be a key into
        the dictionary `interp_data_dict`.
    :return: interp_data_dict: Same as input but with layerwise paths instead of
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

    mixing_ratio_matrix_kg_m03 = (
        interp_data_dict[variable_name] * air_density_matrix_kg_m03
    )
    mixing_ratio_matrix_kg_m03 = numpy.reshape(
        mixing_ratio_matrix_kg_m03,
        (num_times * num_sites, num_heights_new)
    )
    layerwise_path_matrix_kg_m02 = rrtm_io._water_content_to_layerwise_path(
        water_content_matrix_kg_m03=mixing_ratio_matrix_kg_m03,
        heights_m_agl=new_heights_m_agl
    )
    # path_matrix_kg_m02 = numpy.fliplr(numpy.cumsum(
    #     numpy.fliplr(layerwise_path_matrix_kg_m02), axis=1
    # ))
    layerwise_path_matrix_kg_m02 = numpy.reshape(
        layerwise_path_matrix_kg_m02,
        (num_times, num_sites, num_heights_new)
    )
    interp_data_dict[variable_name] = layerwise_path_matrix_kg_m02

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
    # orig_gfs_table_xarray = orig_gfs_table_xarray.isel(
    #     indexers={
    #         SITE_DIMENSION_ORIG: numpy.linspace(0, 4800, num=17, dtype=int)
    #     },
    #     drop=False
    # )

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
        ice_profile_noise_stdev_fractional=PROFILE_NOISE_STDEV_FRACTIONAL,
        ice_indiv_noise_stdev_fractional=INDIV_NOISE_STDEV_FRACTIONAL
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
    aerosol_extinction_matrix_metres01 = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.AEROSOL_EXTINCTION_NAME
    )
    interp_data_dict[AEROSOL_EXTINCTION_KEY_ORIG_METRES01] = numpy.reshape(
        aerosol_extinction_matrix_metres01,
        (num_times, num_sites, num_heights_new)
    )

    aerosol_albedo_matrix = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.AEROSOL_ALBEDO_NAME
    )
    interp_data_dict[AEROSOL_ALBEDO_KEY_ORIG] = numpy.reshape(
        aerosol_albedo_matrix, (num_times, num_sites, 1)
    )
    interp_data_dict[AEROSOL_ALBEDO_KEY_ORIG] = (
        interp_data_dict[AEROSOL_ALBEDO_KEY_ORIG][..., 0]
    )

    aerosol_asymmetry_param_matrix = example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.AEROSOL_ASYMMETRY_PARAM_NAME
    )
    interp_data_dict[AEROSOL_ASYMMETRY_PARAM_KEY_ORIG] = numpy.reshape(
        aerosol_asymmetry_param_matrix, (num_times, num_sites, 1)
    )
    interp_data_dict[AEROSOL_ASYMMETRY_PARAM_KEY_ORIG] = (
        interp_data_dict[AEROSOL_ASYMMETRY_PARAM_KEY_ORIG][..., 0]
    )

    print(
        'Converting ice-water mixing ratios (kg/kg) to layerwise paths '
        '(kg/m^2)...'
    )
    interp_data_dict = _mixing_ratio_to_layerwise_path(
        orig_gfs_table_xarray=orig_gfs_table_xarray,
        new_heights_m_agl=new_heights_m_agl,
        interp_data_dict=interp_data_dict,
        variable_name=CLOUD_ICE_MIXR_KEY_ORIG_KG_KG01
    )

    print('Converting graupel mixing ratios to layerwise paths...')
    interp_data_dict = _mixing_ratio_to_layerwise_path(
        orig_gfs_table_xarray=orig_gfs_table_xarray,
        new_heights_m_agl=new_heights_m_agl,
        interp_data_dict=interp_data_dict,
        variable_name=GRAUPEL_MIXR_KEY_ORIG_KG_KG01
    )

    print('Converting snow mixing ratios to layerwise paths...')
    interp_data_dict = _mixing_ratio_to_layerwise_path(
        orig_gfs_table_xarray=orig_gfs_table_xarray,
        new_heights_m_agl=new_heights_m_agl,
        interp_data_dict=interp_data_dict,
        variable_name=SNOW_MIXR_KEY_ORIG_KG_KG01
    )

    print('Converting cloud-water mixing ratios to layerwise paths...')
    interp_data_dict = _mixing_ratio_to_layerwise_path(
        orig_gfs_table_xarray=orig_gfs_table_xarray,
        new_heights_m_agl=new_heights_m_agl,
        interp_data_dict=interp_data_dict,
        variable_name=CLOUD_WATER_MIXR_KEY_ORIG_KG_KG01
    )

    print('Converting rain mixing ratios to layerwise paths...')
    interp_data_dict = _mixing_ratio_to_layerwise_path(
        orig_gfs_table_xarray=orig_gfs_table_xarray,
        new_heights_m_agl=new_heights_m_agl,
        interp_data_dict=interp_data_dict,
        variable_name=RAIN_MIXR_KEY_ORIG_KG_KG01
    )

    print('Converting cloud fractions from unitless to percent...')
    interp_data_dict[CLOUD_FRACTION_KEY_ORIG] *= UNITLESS_TO_PERCENT

    print('Converting surface albedos from percent to unitless...')
    albedo_matrix = (
        PERCENT_TO_UNITLESS *
        orig_gfs_table_xarray[ALBEDO_KEY_ORIG_PERCENT].values
    )

    # Create xarray table with interpolated data.
    for this_key_orig in interp_data_dict:
        if this_key_orig == PRESSURE_AT_EDGE_KEY_ORIG_PASCALS:
            these_dim = (
                TIME_DIMENSION, SITE_DIMENSION, HEIGHT_AT_EDGE_DIMENSION
            )
        elif this_key_orig in SCALAR_KEYS_ORIG:
            these_dim = (TIME_DIMENSION, SITE_DIMENSION)
        else:
            these_dim = (TIME_DIMENSION, SITE_DIMENSION, HEIGHT_DIMENSION)

        this_key = ORIG_TO_NEW_KEY_DICT[this_key_orig]
        new_data_dict[this_key] = (these_dim, interp_data_dict[this_key_orig])

    these_dim = (TIME_DIMENSION, SITE_DIMENSION)
    new_data_dict.update({
        ALBEDO_KEY: (these_dim, albedo_matrix),
    })

    new_gfs_table_xarray = xarray.Dataset(
        data_vars=new_data_dict, coords=new_metadata_dict
    )

    # for this_key in new_gfs_table_xarray.variables:
    #     if this_key == SITE_NAME_KEY:
    #         continue
    #
    #     if not numpy.any(numpy.isnan(new_gfs_table_xarray[this_key].values)):
    #         continue
    #
    #     print(this_key)
    #     print(numpy.any(numpy.isnan(new_gfs_table_xarray[this_key].values)))

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
