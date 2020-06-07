"""Input/output methods for learning examples."""

import numpy
import netCDF4
from gewittergefahr.gg_utils import longitude_conversion as longitude_conv

KM_TO_METRES = 1000.
DEG_TO_RADIANS = numpy.pi / 180

SCALAR_PREDICTOR_VALS_KEY = 'scalar_predictor_matrix'
SCALAR_PREDICTOR_NAMES_KEY = 'scalar_predictor_names'
VECTOR_PREDICTOR_VALS_KEY = 'vector_predictor_matrix'
VECTOR_PREDICTOR_NAMES_KEY = 'vector_predictor_names'
SCALAR_TARGET_VALS_KEY = 'scalar_target_matrix'
SCALAR_TARGET_NAMES_KEY = 'scalar_target_names'
VECTOR_TARGET_VALS_KEY = 'vector_target_matrix'
VECTOR_TARGET_NAMES_KEY = 'vector_target_names'
VALID_TIMES_KEY = 'valid_times_unix_sec'
HEIGHTS_KEY = 'heights_m_agl'
STANDARD_ATMO_FLAGS_KEY = 'standard_atmo_flags'

VALID_TIMES_KEY_ORIG = 'time'
HEIGHTS_KEY_ORIG = 'height'
STANDARD_ATMO_FLAGS_KEY_ORIG = 'stdatmos'

TROPICS_ENUM = 1
MIDLATITUDE_SUMMER_ENUM = 2
MIDLATITUDE_WINTER_ENUM = 3
SUBARCTIC_SUMMER_ENUM = 4
SUBARCTIC_WINTER_ENUM = 5
US_STANDARD_ATMO_ENUM = 6
STANDARD_ATMO_ENUMS = [
    TROPICS_ENUM, MIDLATITUDE_SUMMER_ENUM, MIDLATITUDE_WINTER_ENUM,
    SUBARCTIC_SUMMER_ENUM, SUBARCTIC_WINTER_ENUM, US_STANDARD_ATMO_ENUM
]

ZENITH_ANGLE_NAME = 'zenith_angle_radians'
LATITUDE_NAME = 'latitude_deg_n'
LONGITUDE_NAME = 'longitude_deg_e'
ALBEDO_NAME = 'albedo'
LIQUID_WATER_PATH_NAME = 'liquid_water_path_kg_m02'
ICE_WATER_PATH_NAME = 'ice_water_path_kg_m02'
PRESSURE_NAME = 'pressure_pascals'
TEMPERATURE_NAME = 'temperature_kelvins'
SPECIFIC_HUMIDITY_NAME = 'specific_humidity_kg_kg01'
LIQUID_WATER_CONTENT_NAME = 'liquid_water_content_kg_m02'
ICE_WATER_CONTENT_NAME = 'ice_water_content_kg_m02'

SCALAR_PREDICTOR_NAMES = [
    ZENITH_ANGLE_NAME, LATITUDE_NAME, LONGITUDE_NAME, ALBEDO_NAME,
    LIQUID_WATER_PATH_NAME, ICE_WATER_PATH_NAME
]
VECTOR_PREDICTOR_NAMES = [
    PRESSURE_NAME, TEMPERATURE_NAME, SPECIFIC_HUMIDITY_NAME,
    LIQUID_WATER_CONTENT_NAME, ICE_WATER_CONTENT_NAME
]
PREDICTOR_NAMES = SCALAR_PREDICTOR_NAMES + VECTOR_PREDICTOR_NAMES

PREDICTOR_NAME_TO_ORIG = {
    ZENITH_ANGLE_NAME: 'sza',
    LATITUDE_NAME: 'lat',
    LONGITUDE_NAME: 'lon',
    ALBEDO_NAME: 'albedo',
    LIQUID_WATER_PATH_NAME: 'lwp',
    ICE_WATER_PATH_NAME: 'iwp',
    PRESSURE_NAME: 'p',
    TEMPERATURE_NAME: 't',
    SPECIFIC_HUMIDITY_NAME: 'q',
    LIQUID_WATER_CONTENT_NAME: 'lwc',
    ICE_WATER_CONTENT_NAME: 'iwc'
}

PREDICTOR_NAME_TO_CONV_FACTOR = {
    ZENITH_ANGLE_NAME: DEG_TO_RADIANS,
    LATITUDE_NAME: 1.,
    LONGITUDE_NAME: 1.,
    ALBEDO_NAME: 1.,
    LIQUID_WATER_PATH_NAME: 0.001,
    ICE_WATER_PATH_NAME: 0.001,
    PRESSURE_NAME: 100.,
    TEMPERATURE_NAME: 1.,
    SPECIFIC_HUMIDITY_NAME: 0.001,
    LIQUID_WATER_CONTENT_NAME: 0.001,
    ICE_WATER_CONTENT_NAME: 0.001
}

SHORTWAVE_HEATING_RATE_NAME = 'shortwave_heating_rate_K_s01'
SHORTWAVE_DOWN_FLUX_NAME = 'shortwave_down_flux_W_m02'
SHORTWAVE_UP_FLUX_NAME = 'shortwave_up_flux_W_m02'
SHORTWAVE_SURFACE_DOWN_FLUX_NAME = 'shortwave_surface_down_flux_W_m02'
SHORTWAVE_TOA_UP_FLUX_NAME = 'shortwave_toa_up_flux_W_m02'

SCALAR_TARGET_NAMES = [
    SHORTWAVE_SURFACE_DOWN_FLUX_NAME, SHORTWAVE_TOA_UP_FLUX_NAME
]
VECTOR_TARGET_NAMES = [
    SHORTWAVE_DOWN_FLUX_NAME, SHORTWAVE_UP_FLUX_NAME,
    SHORTWAVE_HEATING_RATE_NAME
]
TARGET_NAMES = SCALAR_TARGET_NAMES + VECTOR_TARGET_NAMES

TARGET_NAME_TO_ORIG = {
    SHORTWAVE_SURFACE_DOWN_FLUX_NAME: 'sfcflux',
    SHORTWAVE_TOA_UP_FLUX_NAME: 'toaflux',
    SHORTWAVE_DOWN_FLUX_NAME: 'fluxd',
    SHORTWAVE_UP_FLUX_NAME: 'fluxu',
    SHORTWAVE_HEATING_RATE_NAME: 'hr'
}

TARGET_NAME_TO_CONV_FACTOR = {
    SHORTWAVE_SURFACE_DOWN_FLUX_NAME: 1.,
    SHORTWAVE_TOA_UP_FLUX_NAME: 1.,
    SHORTWAVE_DOWN_FLUX_NAME: 1.,
    SHORTWAVE_UP_FLUX_NAME: 1.,
    SHORTWAVE_HEATING_RATE_NAME: 1. / 86400
}


def read_file(example_file_name):
    """Reads NetCDF file with learning examples.

    T = number of times
    H = number of heights
    P_s = number of scalar predictors
    P_v = number of vector predictors
    T_s = number of scalar targets
    T_v = number of vector targets

    :param example_file_name: Path to NetCDF file with learning examples.
    :return: example_dict: Dictionary with the following keys.
    example_dict['scalar_predictor_matrix']: numpy array (T x P_s) with values
        of scalar predictors.
    example_dict['scalar_predictor_names']: list (length P_s) with names of
        scalar predictors.
    example_dict['vector_predictor_matrix']: numpy array (T x H x P_v) with
        values of vector predictors.
    example_dict['vector_predictor_names']: list (length P_v) with names of
        vector predictors.
    example_dict['scalar_target_matrix']: numpy array (T x T_s) with values of
        scalar targets.
    example_dict['scalar_predictor_names']: list (length T_s) with names of
        scalar targets.
    example_dict['vector_target_matrix']: numpy array (T x H x T_v) with values
        of vector targets.
    example_dict['vector_predictor_names']: list (length T_v) with names of
        vector targets.
    example_dict['valid_times_unix_sec']: length-T numpy array of valid times
        (Unix seconds).
    example_dict['heights_m_agl']: length-H numpy array of heights (metres above
        ground level).
    example_dict['standard_atmo_flags']: length-T numpy array of flags (each in
        the list `STANDARD_ATMO_ENUMS`).
    """

    dataset_object = netCDF4.Dataset(example_file_name)

    example_dict = {
        SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
        VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
        SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
        VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
        VALID_TIMES_KEY: numpy.array(
            dataset_object.variables[VALID_TIMES_KEY_ORIG][:], dtype=int
        ),
        HEIGHTS_KEY: KM_TO_METRES * numpy.array(
            dataset_object.variables[HEIGHTS_KEY_ORIG][:], dtype=float
        ),
        STANDARD_ATMO_FLAGS_KEY: numpy.array(
            numpy.round(
                dataset_object.variables[STANDARD_ATMO_FLAGS_KEY_ORIG][:]
            ), dtype=int
        )
    }

    num_times = len(example_dict[VALID_TIMES_KEY])
    num_heights = len(example_dict[HEIGHTS_KEY])
    num_scalar_predictors = len(SCALAR_PREDICTOR_NAMES)
    num_vector_predictors = len(VECTOR_PREDICTOR_NAMES)
    num_scalar_targets = len(SCALAR_TARGET_NAMES)
    num_vector_targets = len(VECTOR_TARGET_NAMES)

    scalar_predictor_matrix = numpy.full(
        (num_times, num_scalar_predictors), numpy.nan
    )
    vector_predictor_matrix = numpy.full(
        (num_times, num_heights, num_vector_predictors), numpy.nan
    )
    scalar_target_matrix = numpy.full(
        (num_times, num_scalar_targets), numpy.nan
    )
    vector_target_matrix = numpy.full(
        (num_times, num_heights, num_vector_targets), numpy.nan
    )

    for k in range(num_scalar_predictors):
        this_predictor_name_orig = (
            PREDICTOR_NAME_TO_ORIG[SCALAR_PREDICTOR_NAMES[k]]
        )
        this_conversion_factor = (
            PREDICTOR_NAME_TO_CONV_FACTOR[SCALAR_PREDICTOR_NAMES[k]]
        )
        scalar_predictor_matrix[:, k] = this_conversion_factor * numpy.array(
            dataset_object.variables[this_predictor_name_orig][:], dtype=float
        )

    for k in range(num_vector_predictors):
        this_predictor_name_orig = (
            PREDICTOR_NAME_TO_ORIG[VECTOR_PREDICTOR_NAMES[k]]
        )
        this_conversion_factor = (
            PREDICTOR_NAME_TO_CONV_FACTOR[VECTOR_PREDICTOR_NAMES[k]]
        )
        vector_predictor_matrix[..., k] = this_conversion_factor * numpy.array(
            dataset_object.variables[this_predictor_name_orig][:], dtype=float
        )

    for k in range(num_scalar_targets):
        this_target_name_orig = TARGET_NAME_TO_ORIG[SCALAR_TARGET_NAMES[k]]
        this_conversion_factor = (
            TARGET_NAME_TO_CONV_FACTOR[SCALAR_TARGET_NAMES[k]]
        )
        scalar_target_matrix[:, k] = this_conversion_factor * numpy.array(
            dataset_object.variables[this_target_name_orig][:], dtype=float
        )

    for k in range(num_vector_targets):
        this_target_name_orig = TARGET_NAME_TO_ORIG[VECTOR_TARGET_NAMES[k]]
        this_conversion_factor = (
            TARGET_NAME_TO_CONV_FACTOR[VECTOR_TARGET_NAMES[k]]
        )
        vector_target_matrix[..., k] = this_conversion_factor * numpy.array(
            dataset_object.variables[this_target_name_orig][:], dtype=float
        )

    longitude_index = SCALAR_PREDICTOR_NAMES.index(LONGITUDE_NAME)
    scalar_predictor_matrix[:, longitude_index] = (
        longitude_conv.convert_lng_positive_in_west(
            longitudes_deg=scalar_predictor_matrix[:, longitude_index],
            allow_nan=False
        )
    )

    example_dict.update({
        SCALAR_PREDICTOR_VALS_KEY: scalar_predictor_matrix,
        VECTOR_PREDICTOR_VALS_KEY: vector_predictor_matrix,
        SCALAR_TARGET_VALS_KEY: scalar_target_matrix,
        VECTOR_TARGET_VALS_KEY: vector_target_matrix
    })

    dataset_object.close()
    return example_dict
