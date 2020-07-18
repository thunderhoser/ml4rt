"""Input/output methods for learning examples."""

import copy
import os.path
import numpy
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import prob_matched_means as pmm
from gewittergefahr.gg_utils import longitude_conversion as longitude_conv
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6

KM_TO_METRES = 1000.
DEG_TO_RADIANS = numpy.pi / 180

DAYS_TO_SECONDS = 86400.
GRAVITY_CONSTANT_M_S02 = 9.80665
DRY_AIR_SPECIFIC_HEAT_J_KG01_K01 = 287.04 * 3.5

DEFAULT_MAX_PMM_PERCENTILE_LEVEL = 99.

DEFAULT_HEIGHTS_M_AGL = numpy.array([
    10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350,
    400, 450, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
    1700, 1800, 1900, 2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600,
    3800, 4000, 4200, 4400, 4600, 4800, 5000, 5500, 6000, 6500, 7000, 8000,
    9000, 10000, 11000, 12000, 13000, 14000, 15000, 18000, 20000, 22000, 24000,
    27000, 30000, 33000, 36000, 39000, 42000, 46000, 50000
], dtype=float)

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

LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'
ZENITH_ANGLES_KEY = 'zenith_angles_rad'

DICTIONARY_KEYS = [
    SCALAR_PREDICTOR_VALS_KEY, SCALAR_PREDICTOR_NAMES_KEY,
    VECTOR_PREDICTOR_VALS_KEY, VECTOR_PREDICTOR_NAMES_KEY,
    SCALAR_TARGET_VALS_KEY, SCALAR_TARGET_NAMES_KEY,
    VECTOR_TARGET_VALS_KEY, VECTOR_TARGET_NAMES_KEY,
    VALID_TIMES_KEY, HEIGHTS_KEY, STANDARD_ATMO_FLAGS_KEY
]
ONE_PER_EXAMPLE_KEYS = [
    SCALAR_PREDICTOR_VALS_KEY, VECTOR_PREDICTOR_VALS_KEY,
    SCALAR_TARGET_VALS_KEY, VECTOR_TARGET_VALS_KEY,
    VALID_TIMES_KEY, STANDARD_ATMO_FLAGS_KEY
]
ONE_PER_FIELD_KEYS = [
    SCALAR_PREDICTOR_VALS_KEY, SCALAR_PREDICTOR_NAMES_KEY,
    VECTOR_PREDICTOR_VALS_KEY, VECTOR_PREDICTOR_NAMES_KEY,
    SCALAR_TARGET_VALS_KEY, SCALAR_TARGET_NAMES_KEY,
    VECTOR_TARGET_VALS_KEY, VECTOR_TARGET_NAMES_KEY
]

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
COLUMN_LIQUID_WATER_PATH_NAME = 'column_liquid_water_path_kg_m02'
COLUMN_ICE_WATER_PATH_NAME = 'column_ice_water_path_kg_m02'
PRESSURE_NAME = 'pressure_pascals'
TEMPERATURE_NAME = 'temperature_kelvins'
SPECIFIC_HUMIDITY_NAME = 'specific_humidity_kg_kg01'
LIQUID_WATER_CONTENT_NAME = 'liquid_water_content_kg_m03'
ICE_WATER_CONTENT_NAME = 'ice_water_content_kg_m03'
LIQUID_WATER_PATH_NAME = 'liquid_water_path_kg_m02'
ICE_WATER_PATH_NAME = 'ice_water_path_kg_m02'
UPWARD_LIQUID_WATER_PATH_NAME = 'upward_liquid_water_path_kg_m02'
UPWARD_ICE_WATER_PATH_NAME = 'upward_ice_water_path_kg_m02'

DEFAULT_SCALAR_PREDICTOR_NAMES = [
    ZENITH_ANGLE_NAME, LATITUDE_NAME, LONGITUDE_NAME, ALBEDO_NAME,
    COLUMN_LIQUID_WATER_PATH_NAME, COLUMN_ICE_WATER_PATH_NAME
]
ALL_SCALAR_PREDICTOR_NAMES = DEFAULT_SCALAR_PREDICTOR_NAMES

DEFAULT_VECTOR_PREDICTOR_NAMES = [
    PRESSURE_NAME, TEMPERATURE_NAME, SPECIFIC_HUMIDITY_NAME,
    LIQUID_WATER_CONTENT_NAME, ICE_WATER_CONTENT_NAME
]
ALL_VECTOR_PREDICTOR_NAMES = DEFAULT_VECTOR_PREDICTOR_NAMES + [
    LIQUID_WATER_PATH_NAME, ICE_WATER_PATH_NAME, UPWARD_LIQUID_WATER_PATH_NAME,
    UPWARD_ICE_WATER_PATH_NAME
]

ALL_PREDICTOR_NAMES = ALL_SCALAR_PREDICTOR_NAMES + ALL_VECTOR_PREDICTOR_NAMES

PREDICTOR_NAME_TO_ORIG = {
    ZENITH_ANGLE_NAME: 'sza',
    LATITUDE_NAME: 'lat',
    LONGITUDE_NAME: 'lon',
    ALBEDO_NAME: 'albedo',
    COLUMN_LIQUID_WATER_PATH_NAME: 'lwp',
    COLUMN_ICE_WATER_PATH_NAME: 'iwp',
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
    COLUMN_LIQUID_WATER_PATH_NAME: 0.001,
    COLUMN_ICE_WATER_PATH_NAME: 0.001,
    PRESSURE_NAME: 100.,
    TEMPERATURE_NAME: 1.,
    SPECIFIC_HUMIDITY_NAME: 0.001,
    LIQUID_WATER_CONTENT_NAME: 0.001,
    ICE_WATER_CONTENT_NAME: 0.001
}

SHORTWAVE_HEATING_RATE_NAME = 'shortwave_heating_rate_k_day01'
SHORTWAVE_DOWN_FLUX_NAME = 'shortwave_down_flux_w_m02'
SHORTWAVE_UP_FLUX_NAME = 'shortwave_up_flux_w_m02'
SHORTWAVE_DOWN_FLUX_INC_NAME = 'shortwave_down_flux_increment_w_m02'
SHORTWAVE_UP_FLUX_INC_NAME = 'shortwave_up_flux_increment_w_m02'
SHORTWAVE_SURFACE_DOWN_FLUX_NAME = 'shortwave_surface_down_flux_w_m02'
SHORTWAVE_TOA_UP_FLUX_NAME = 'shortwave_toa_up_flux_w_m02'

DEFAULT_SCALAR_TARGET_NAMES = [
    SHORTWAVE_SURFACE_DOWN_FLUX_NAME, SHORTWAVE_TOA_UP_FLUX_NAME
]
ALL_SCALAR_TARGET_NAMES = DEFAULT_SCALAR_TARGET_NAMES

DEFAULT_VECTOR_TARGET_NAMES = [
    SHORTWAVE_DOWN_FLUX_NAME, SHORTWAVE_UP_FLUX_NAME,
    SHORTWAVE_HEATING_RATE_NAME
]
ALL_VECTOR_TARGET_NAMES = DEFAULT_VECTOR_TARGET_NAMES + [
    SHORTWAVE_DOWN_FLUX_INC_NAME, SHORTWAVE_UP_FLUX_INC_NAME
]

ALL_TARGET_NAMES = ALL_SCALAR_TARGET_NAMES + ALL_VECTOR_TARGET_NAMES

TARGET_NAME_TO_ORIG = {
    SHORTWAVE_SURFACE_DOWN_FLUX_NAME: 'sfcflux',
    SHORTWAVE_TOA_UP_FLUX_NAME: 'toaflux',
    SHORTWAVE_DOWN_FLUX_NAME: 'fluxd',
    SHORTWAVE_UP_FLUX_NAME: 'fluxu',
    SHORTWAVE_HEATING_RATE_NAME: 'hr'
}


def _get_water_content_profiles(layerwise_path_matrix_kg_m02, heights_m_agl):
    """Computes profiles of liquid- or ice-water content (LWC or IWC).

    E = number of examples
    H = number of heights

    :param layerwise_path_matrix_kg_m02: E-by-H numpy array of layerwise water
        paths (kg m^-2).  "Layerwise" means that the value in each grid cell is
        only the water path through that grid cell, not integrated over multiple
        grid cells.
    :param heights_m_agl: length-H numpy array with heights of grid-cell centers
        (metres above ground level).
    :return: water_content_matrix_kg_m03: E-by-H numpy array of water contents
        (kg m^-3).
    """

    edge_heights_m_agl = get_grid_cell_edges(heights_m_agl)
    grid_cell_widths_metres = get_grid_cell_widths(edge_heights_m_agl)

    num_examples = layerwise_path_matrix_kg_m02.shape[0]
    num_heights = layerwise_path_matrix_kg_m02.shape[1]

    grid_cell_width_matrix_metres = numpy.reshape(
        grid_cell_widths_metres, (1, num_heights)
    )
    grid_cell_width_matrix_metres = numpy.repeat(
        grid_cell_width_matrix_metres, repeats=num_examples, axis=0
    )

    return layerwise_path_matrix_kg_m02 / grid_cell_width_matrix_metres


def _get_water_path_profiles(example_dict, get_lwp=True, get_iwp=True,
                             integrate_upward=False):
    """Computes profiles of liquid-water path (LWP) and/or ice-water path (IWP).

    If `integrate_upward == False`, then at height z, the LWP (IWP) is the
    integral of LWC (IWC) from the top of atmosphere to z.

    If `integrate_upward == True`, then at height z, the LWP (IWP) is the
    integral of LWC (IWC) from the surface to z.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :param get_lwp: Boolean flag.  If True, will compute LWP profile for each
        example.
    :param get_iwp: Boolean flag.  If True, will compute IWP profile for each
        example.
    :param integrate_upward: Boolean flag.  If True, will integrate from the
        surface up.  If False, will integrate from the top of atmosphere down.
    :return: example_dict: Same as input but with extra predictor variables.
    """

    vector_predictor_names = example_dict[VECTOR_PREDICTOR_NAMES_KEY]

    if integrate_upward:
        this_liquid_path_name = UPWARD_LIQUID_WATER_PATH_NAME
        this_ice_path_name = UPWARD_ICE_WATER_PATH_NAME
    else:
        this_liquid_path_name = LIQUID_WATER_PATH_NAME
        this_ice_path_name = ICE_WATER_PATH_NAME

    get_lwp = get_lwp and this_liquid_path_name not in vector_predictor_names
    get_iwp = get_iwp and this_ice_path_name not in vector_predictor_names

    if not (get_lwp or get_iwp):
        return example_dict

    edge_heights_m_agl = get_grid_cell_edges(example_dict[HEIGHTS_KEY])
    grid_cell_widths_metres = get_grid_cell_widths(edge_heights_m_agl)

    num_examples = len(example_dict[VALID_TIMES_KEY])
    num_heights = len(example_dict[HEIGHTS_KEY])

    grid_cell_width_matrix_metres = numpy.reshape(
        grid_cell_widths_metres, (1, num_heights)
    )
    grid_cell_width_matrix_metres = numpy.repeat(
        grid_cell_width_matrix_metres, repeats=num_examples, axis=0
    )

    if get_lwp:
        lwc_matrix_kg_m03 = get_field_from_dict(
            example_dict=example_dict, field_name=LIQUID_WATER_CONTENT_NAME
        )

        if integrate_upward:
            lwp_matrix_kg_m02 = numpy.cumsum(
                lwc_matrix_kg_m03 * grid_cell_width_matrix_metres, axis=1
            )
        else:
            lwp_matrix_kg_m02 = numpy.fliplr(numpy.cumsum(
                numpy.fliplr(lwc_matrix_kg_m03 * grid_cell_width_matrix_metres),
                axis=1
            ))

        example_dict[VECTOR_PREDICTOR_NAMES_KEY].append(this_liquid_path_name)
        example_dict[VECTOR_PREDICTOR_VALS_KEY] = numpy.concatenate((
            example_dict[VECTOR_PREDICTOR_VALS_KEY],
            numpy.expand_dims(lwp_matrix_kg_m02, axis=-1)
        ), axis=-1)

    if get_iwp:
        iwc_matrix_kg_m03 = get_field_from_dict(
            example_dict=example_dict, field_name=ICE_WATER_CONTENT_NAME
        )

        if integrate_upward:
            iwp_matrix_kg_m02 = numpy.cumsum(
                iwc_matrix_kg_m03 * grid_cell_width_matrix_metres, axis=1
            )
        else:
            iwp_matrix_kg_m02 = numpy.fliplr(numpy.cumsum(
                numpy.fliplr(iwc_matrix_kg_m03 * grid_cell_width_matrix_metres),
                axis=1
            ))

        example_dict[VECTOR_PREDICTOR_NAMES_KEY].append(this_ice_path_name)
        example_dict[VECTOR_PREDICTOR_VALS_KEY] = numpy.concatenate((
            example_dict[VECTOR_PREDICTOR_VALS_KEY],
            numpy.expand_dims(iwp_matrix_kg_m02, axis=-1)
        ), axis=-1)

    return example_dict


def get_grid_cell_edges(heights_m_agl):
    """Computes heights at edges (rather than centers) of grid cells.

    H = number of grid cells

    :param heights_m_agl: length-H numpy array of heights (metres above ground
        level) at centers of grid cells.
    :return: edge_heights_m_agl: length-(H + 1) numpy array of heights (metres
        above ground level) at edges of grid cells.
    """

    error_checking.assert_is_geq_numpy_array(heights_m_agl, 0.)
    error_checking.assert_is_numpy_array(heights_m_agl, num_dimensions=1)

    height_diffs_metres = numpy.diff(heights_m_agl)
    edge_heights_m_agl = heights_m_agl[:-1] + height_diffs_metres / 2
    bottom_edge_m_agl = heights_m_agl[0] - height_diffs_metres[0] / 2
    top_edge_m_agl = heights_m_agl[-1] + height_diffs_metres[-1] / 2

    return numpy.concatenate((
        numpy.array([bottom_edge_m_agl]),
        edge_heights_m_agl,
        numpy.array([top_edge_m_agl])
    ))


def get_grid_cell_widths(edge_heights_m_agl):
    """Computes width of each grid cell.

    H = number of grid cells

    :param edge_heights_m_agl: length-(H + 1) numpy array of heights (metres
        above ground level) at edges of grid cells.
    :return: widths_metres: length-H numpy array of grid-cell widths.
    """

    error_checking.assert_is_geq_numpy_array(edge_heights_m_agl, 0.)
    error_checking.assert_is_numpy_array(edge_heights_m_agl, num_dimensions=1)

    return numpy.diff(edge_heights_m_agl)


def fluxes_to_heating_rate(example_dict):
    """For each example at each height, converts up/down fluxes to heating rate.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :return: example_dict: Same but with heating-rate profiles.
    """

    # TODO(thunderhoser): Figure out why this method gives different answers
    # than the RRTM files from Dave Turner.

    # if SHORTWAVE_HEATING_RATE_NAME in example_dict[VECTOR_TARGET_NAMES_KEY]:
    #     raise ValueError('Dictionary already contains heating-rate profiles.')

    down_flux_matrix_w_m02 = get_field_from_dict(
        example_dict=example_dict, field_name=SHORTWAVE_DOWN_FLUX_NAME
    )
    up_flux_matrix_w_m02 = get_field_from_dict(
        example_dict=example_dict, field_name=SHORTWAVE_UP_FLUX_NAME
    )
    pressure_matrix_pascals = get_field_from_dict(
        example_dict=example_dict, field_name=PRESSURE_NAME
    )

    net_flux_matrix_w_m02 = down_flux_matrix_w_m02 - up_flux_matrix_w_m02
    coefficient = GRAVITY_CONSTANT_M_S02 / DRY_AIR_SPECIFIC_HEAT_J_KG01_K01

    heating_rate_matrix_k_day01 = DAYS_TO_SECONDS * coefficient * (
        numpy.gradient(net_flux_matrix_w_m02, axis=1) /
        numpy.absolute(numpy.gradient(pressure_matrix_pascals, axis=1))
    )

    vector_target_names = example_dict[VECTOR_TARGET_NAMES_KEY]
    found_heating_rate = SHORTWAVE_HEATING_RATE_NAME in vector_target_names
    if not found_heating_rate:
        vector_target_names.append(SHORTWAVE_HEATING_RATE_NAME)

    heating_rate_index = vector_target_names.index(SHORTWAVE_HEATING_RATE_NAME)
    example_dict[VECTOR_TARGET_NAMES_KEY] = vector_target_names

    if found_heating_rate:
        example_dict[VECTOR_TARGET_VALS_KEY][..., heating_rate_index] = (
            heating_rate_matrix_k_day01
        )
    else:
        example_dict[VECTOR_TARGET_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_TARGET_VALS_KEY],
            obj=heating_rate_index, values=heating_rate_matrix_k_day01, axis=-1
        )

    return example_dict


def fluxes_actual_to_increments(example_dict):
    """For each example, converts flux profiles to flux-increment profiles.

    In a "flux profile," the values at each height are the total upwelling and
    downwelling fluxes.

    In a "flux-increment profile," the values at the [j]th height are the
    upwelling- and downwelling-flux increments added by the [j]th layer.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :return: example_dict: Same but with flux-increment profiles.
    """

    down_flux_matrix_w_m02 = get_field_from_dict(
        example_dict=example_dict, field_name=SHORTWAVE_DOWN_FLUX_NAME
    )
    up_flux_matrix_w_m02 = get_field_from_dict(
        example_dict=example_dict, field_name=SHORTWAVE_UP_FLUX_NAME
    )

    down_flux_increment_matrix_w_m02 = numpy.diff(
        down_flux_matrix_w_m02, axis=1, prepend=0.
    )
    up_flux_increment_matrix_w_m02 = numpy.diff(
        up_flux_matrix_w_m02, axis=1, prepend=0.
    )

    vector_target_names = example_dict[VECTOR_TARGET_NAMES_KEY]
    found_down_increment = SHORTWAVE_DOWN_FLUX_INC_NAME in vector_target_names
    found_up_increment = SHORTWAVE_UP_FLUX_INC_NAME in vector_target_names

    if not found_down_increment:
        vector_target_names.append(SHORTWAVE_DOWN_FLUX_INC_NAME)
    if not found_up_increment:
        vector_target_names.append(SHORTWAVE_UP_FLUX_INC_NAME)

    down_increment_index = vector_target_names.index(
        SHORTWAVE_DOWN_FLUX_INC_NAME
    )
    up_increment_index = vector_target_names.index(SHORTWAVE_UP_FLUX_INC_NAME)
    example_dict[VECTOR_TARGET_NAMES_KEY] = vector_target_names

    if found_down_increment:
        example_dict[VECTOR_TARGET_VALS_KEY][..., down_increment_index] = (
            down_flux_increment_matrix_w_m02
        )
    else:
        example_dict[VECTOR_TARGET_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_TARGET_VALS_KEY], obj=down_increment_index,
            values=down_flux_increment_matrix_w_m02, axis=-1
        )

    if found_up_increment:
        example_dict[VECTOR_TARGET_VALS_KEY][..., up_increment_index] = (
            up_flux_increment_matrix_w_m02
        )
    else:
        example_dict[VECTOR_TARGET_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_TARGET_VALS_KEY], obj=up_increment_index,
            values=up_flux_increment_matrix_w_m02, axis=-1
        )

    return example_dict


def fluxes_increments_to_actual(example_dict):
    """For each example, converts flux-increment profiles to flux profiles.

    This method is the inverse of `fluxes_actual_to_increments`.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :return: example_dict: Same but with actual flux profiles.
    """

    down_flux_increment_matrix_w_m02 = get_field_from_dict(
        example_dict=example_dict, field_name=SHORTWAVE_DOWN_FLUX_INC_NAME
    )
    up_flux_increment_matrix_w_m02 = get_field_from_dict(
        example_dict=example_dict, field_name=SHORTWAVE_UP_FLUX_INC_NAME
    )

    down_flux_matrix_w_m02 = numpy.cumsum(
        down_flux_increment_matrix_w_m02, axis=1
    )
    up_flux_matrix_w_m02 = numpy.cumsum(up_flux_increment_matrix_w_m02, axis=1)

    vector_target_names = example_dict[VECTOR_TARGET_NAMES_KEY]
    found_down_flux = SHORTWAVE_DOWN_FLUX_NAME in vector_target_names
    found_up_flux = SHORTWAVE_UP_FLUX_NAME in vector_target_names

    if not found_down_flux:
        vector_target_names.append(SHORTWAVE_DOWN_FLUX_NAME)
    if not found_up_flux:
        vector_target_names.append(SHORTWAVE_UP_FLUX_NAME)

    down_flux_index = vector_target_names.index(SHORTWAVE_DOWN_FLUX_NAME)
    up_flux_index = vector_target_names.index(SHORTWAVE_UP_FLUX_NAME)
    example_dict[VECTOR_TARGET_NAMES_KEY] = vector_target_names

    if found_down_flux:
        example_dict[VECTOR_TARGET_VALS_KEY][..., down_flux_index] = (
            down_flux_matrix_w_m02
        )
    else:
        example_dict[VECTOR_TARGET_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_TARGET_VALS_KEY],
            obj=down_flux_index, values=down_flux_matrix_w_m02, axis=-1
        )

    if found_up_flux:
        example_dict[VECTOR_TARGET_VALS_KEY][..., up_flux_index] = (
            up_flux_matrix_w_m02
        )
    else:
        example_dict[VECTOR_TARGET_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_TARGET_VALS_KEY],
            obj=up_flux_index, values=up_flux_matrix_w_m02, axis=-1
        )

    return example_dict


def match_heights(heights_m_agl, desired_height_m_agl):
    """Finds nearest available height to desired height.

    :param heights_m_agl: 1-D numpy array of available heights (metres above
        ground level).
    :param desired_height_m_agl: Desired height (metres above ground level).
    :return: matching_index: Index of desired height in array.  If
        `matching_index == k`, then `heights_m_agl[k]` is the desired height.
    :raises: ValueError: if there is no available height within 0.5 metres of
        the desired height.
    """

    error_checking.assert_is_geq_numpy_array(heights_m_agl, 0.)
    error_checking.assert_is_numpy_array(heights_m_agl, num_dimensions=1)
    error_checking.assert_is_geq(desired_height_m_agl, 0.)

    height_diffs_metres = numpy.absolute(heights_m_agl - desired_height_m_agl)
    matching_index = numpy.argmin(height_diffs_metres)

    if height_diffs_metres[matching_index] <= 0.5:
        return matching_index

    error_string = (
        'Cannot find available height within 0.5 metres of desired height '
        '({0:.1f} m AGL).  Nearest available height is {1:.1f} m AGL.'
    ).format(desired_height_m_agl, heights_m_agl[matching_index])

    raise ValueError(error_string)


def check_field_name(field_name):
    """Ensures that field name is valid (either predictor or target variable).

    :param field_name: Field name.
    :raises: ValueError: if
        `field_name not in ALL_PREDICTOR_NAMES + ALL_TARGET_NAMES`.
    """

    error_checking.assert_is_string(field_name)
    if field_name in ALL_PREDICTOR_NAMES + ALL_TARGET_NAMES:
        return

    error_string = (
        '\nField "{0:s}" is not valid predictor or target variable.  Valid '
        'options listed below:\n{1:s}'
    ).format(field_name, str(ALL_PREDICTOR_NAMES + ALL_TARGET_NAMES))

    raise ValueError(error_string)


def check_standard_atmo_type(standard_atmo_enum):
    """Ensures that standard-atmosphere type is valid.

    :param standard_atmo_enum: Integer (must be in `STANDARD_ATMO_ENUMS`).
    :raises: ValueError: if `standard_atmo_enum not in STANDARD_ATMO_ENUMS`.
    """

    error_checking.assert_is_integer(standard_atmo_enum)

    if standard_atmo_enum not in STANDARD_ATMO_ENUMS:
        error_string = (
            'Standard-atmosphere type {0:d} is invalid.  Must be in the '
            'following list:\n{1:s}'
        ).format(standard_atmo_enum, str(STANDARD_ATMO_ENUMS))

        raise ValueError(error_string)


def find_file(example_dir_name, year, raise_error_if_missing=True):
    """Finds NetCDF file with learning examples.

    :param example_dir_name: Name of directory where file is expected.
    :param year: Year (integer).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: example_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(example_dir_name)
    error_checking.assert_is_integer(year)
    error_checking.assert_is_boolean(raise_error_if_missing)

    example_file_name = '{0:s}/radiative_transfer_examples_{1:04d}.nc'.format(
        example_dir_name, year
    )

    if raise_error_if_missing and not os.path.isfile(example_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            example_file_name
        )
        raise ValueError(error_string)

    return example_file_name


def file_name_to_year(example_file_name):
    """Parses year from file name.

    :param example_file_name: Path to example file (readable by `read_file`).
    :return: year: Year (integer).
    """

    error_checking.assert_is_string(example_file_name)
    pathless_file_name = os.path.split(example_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]

    return int(extensionless_file_name.split('_')[-1])


def find_many_files(
        example_dir_name, first_time_unix_sec, last_time_unix_sec,
        raise_error_if_any_missing=True, raise_error_if_all_missing=True,
        test_mode=False):
    """Finds many NetCDF files with learning examples.

    :param example_dir_name: Name of directory where files are expected.
    :param first_time_unix_sec: First time at which examples are desired.
    :param last_time_unix_sec: Last time at which examples are desired.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :param test_mode: Leave this alone.
    :return: example_file_names: 1-D list of paths to example files.  This list
        does *not* contain expected paths to non-existent files.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    error_checking.assert_is_boolean(test_mode)

    start_year = int(
        time_conversion.unix_sec_to_string(first_time_unix_sec, '%Y')
    )
    end_year = int(
        time_conversion.unix_sec_to_string(last_time_unix_sec, '%Y')
    )
    years = numpy.linspace(
        start_year, end_year, num=end_year - start_year + 1, dtype=int
    )

    example_file_names = []

    for this_year in years:
        this_file_name = find_file(
            example_dir_name=example_dir_name, year=this_year,
            raise_error_if_missing=raise_error_if_any_missing
        )

        if test_mode or os.path.isfile(this_file_name):
            example_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(example_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from years {1:d}-{2:d}.'
        ).format(
            example_dir_name, start_year, end_year
        )
        raise ValueError(error_string)

    return example_file_names


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
    example_dict['scalar_target_names']: list (length T_s) with names of scalar
        targets.
    example_dict['vector_target_matrix']: numpy array (T x H x T_v) with values
        of vector targets.
    example_dict['vector_target_names']: list (length T_v) with names of vector
        targets.
    example_dict['valid_times_unix_sec']: length-T numpy array of valid times
        (Unix seconds).
    example_dict['heights_m_agl']: length-H numpy array of heights (metres above
        ground level).
    example_dict['standard_atmo_flags']: length-T numpy array of flags (each in
        the list `STANDARD_ATMO_ENUMS`).
    """

    if not os.path.isfile(example_file_name):
        example_file_name = example_file_name.replace(
            '/home/ryan.lagerquist', '/home/ralager'
        )

    dataset_object = netCDF4.Dataset(example_file_name)

    example_dict = {
        SCALAR_PREDICTOR_NAMES_KEY:
            copy.deepcopy(DEFAULT_SCALAR_PREDICTOR_NAMES),
        VECTOR_PREDICTOR_NAMES_KEY:
            copy.deepcopy(DEFAULT_VECTOR_PREDICTOR_NAMES),
        SCALAR_TARGET_NAMES_KEY: copy.deepcopy(DEFAULT_SCALAR_TARGET_NAMES),
        VECTOR_TARGET_NAMES_KEY: copy.deepcopy(DEFAULT_VECTOR_TARGET_NAMES),
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
    num_scalar_predictors = len(DEFAULT_SCALAR_PREDICTOR_NAMES)
    num_vector_predictors = len(DEFAULT_VECTOR_PREDICTOR_NAMES)
    num_scalar_targets = len(DEFAULT_SCALAR_TARGET_NAMES)
    num_vector_targets = len(DEFAULT_VECTOR_TARGET_NAMES)

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
            PREDICTOR_NAME_TO_ORIG[DEFAULT_SCALAR_PREDICTOR_NAMES[k]]
        )
        this_conversion_factor = (
            PREDICTOR_NAME_TO_CONV_FACTOR[DEFAULT_SCALAR_PREDICTOR_NAMES[k]]
        )
        scalar_predictor_matrix[:, k] = this_conversion_factor * numpy.array(
            dataset_object.variables[this_predictor_name_orig][:], dtype=float
        )

    for k in range(num_vector_predictors):
        this_predictor_name_orig = (
            PREDICTOR_NAME_TO_ORIG[DEFAULT_VECTOR_PREDICTOR_NAMES[k]]
        )
        this_conversion_factor = (
            PREDICTOR_NAME_TO_CONV_FACTOR[DEFAULT_VECTOR_PREDICTOR_NAMES[k]]
        )
        vector_predictor_matrix[..., k] = this_conversion_factor * numpy.array(
            dataset_object.variables[this_predictor_name_orig][:], dtype=float
        )

        if DEFAULT_VECTOR_PREDICTOR_NAMES[k] in [
                LIQUID_WATER_CONTENT_NAME, ICE_WATER_CONTENT_NAME
        ]:
            vector_predictor_matrix[..., k] = _get_water_content_profiles(
                layerwise_path_matrix_kg_m02=vector_predictor_matrix[..., k],
                heights_m_agl=example_dict[HEIGHTS_KEY]
            )

    for k in range(num_scalar_targets):
        this_target_name_orig = (
            TARGET_NAME_TO_ORIG[DEFAULT_SCALAR_TARGET_NAMES[k]]
        )
        scalar_target_matrix[:, k] = numpy.array(
            dataset_object.variables[this_target_name_orig][:], dtype=float
        )

    for k in range(num_vector_targets):
        this_target_name_orig = (
            TARGET_NAME_TO_ORIG[DEFAULT_VECTOR_TARGET_NAMES[k]]
        )
        vector_target_matrix[..., k] = numpy.array(
            dataset_object.variables[this_target_name_orig][:], dtype=float
        )

    longitude_index = DEFAULT_SCALAR_PREDICTOR_NAMES.index(LONGITUDE_NAME)
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

    example_dict = _get_water_path_profiles(
        example_dict=example_dict, get_lwp=True, get_iwp=True,
        integrate_upward=False
    )

    return _get_water_path_profiles(
        example_dict=example_dict, get_lwp=True, get_iwp=True,
        integrate_upward=True
    )


def concat_examples(example_dicts):
    """Concatenates many dictionaries with examples into one.

    :param example_dicts: List of dictionaries, each in the format returned by
        `read_file`.
    :return: example_dict: Single dictionary, also in the format returned by
        `read_file`.
    :raises: ValueError: if any two dictionaries have different predictor
        variables, target variables, or height coordinates.
    """

    example_dict = copy.deepcopy(example_dicts[0])

    keys_to_match = [
        SCALAR_PREDICTOR_NAMES_KEY, VECTOR_PREDICTOR_NAMES_KEY,
        SCALAR_TARGET_NAMES_KEY, VECTOR_TARGET_NAMES_KEY, HEIGHTS_KEY
    ]

    for i in range(1, len(example_dicts)):
        if not numpy.allclose(
                example_dict[HEIGHTS_KEY], example_dicts[i][HEIGHTS_KEY],
                atol=TOLERANCE
        ):
            error_string = (
                '1st and {0:d}th dictionaries have different height coords '
                '(units are m AGL).  1st dictionary:\n{1:s}\n\n'
                '{0:d}th dictionary:\n{2:s}'
            ).format(
                i + 1, str(example_dict[HEIGHTS_KEY]),
                str(example_dicts[i][HEIGHTS_KEY])
            )

            raise ValueError(error_string)

        for this_key in keys_to_match:
            if this_key == HEIGHTS_KEY:
                continue

            if example_dict[this_key] == example_dicts[i][this_key]:
                continue

            error_string = (
                '1st and {0:d}th dictionaries have different values for '
                '"{1:s}".  1st dictionary:\n{2:s}\n\n'
                '{0:d}th dictionary:\n{3:s}'
            ).format(
                i + 1, this_key, str(example_dict[this_key]),
                str(example_dicts[i][this_key])
            )

            raise ValueError(error_string)

        for this_key in DICTIONARY_KEYS:
            if this_key in keys_to_match:
                continue

            example_dict[this_key] = numpy.concatenate((
                example_dict[this_key], example_dicts[i][this_key]
            ), axis=0)

    return example_dict


def create_example_ids(example_dict):
    """Creates example IDs (unique identifiers).

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :return: example_id_strings: 1-D list of example IDs.
    """

    latitudes_deg_n = get_field_from_dict(
        example_dict=example_dict, field_name=LATITUDE_NAME
    )
    longitudes_deg_e = get_field_from_dict(
        example_dict=example_dict, field_name=LONGITUDE_NAME
    )
    zenith_angles_rad = get_field_from_dict(
        example_dict=example_dict, field_name=ZENITH_ANGLE_NAME
    )
    valid_times_unix_sec = example_dict[VALID_TIMES_KEY]
    standard_atmo_flags = example_dict[STANDARD_ATMO_FLAGS_KEY]

    return [
        'lat={0:09.6f}_long={1:010.6f}_zenith-angle-rad={2:08.6f}_' \
        'time={3:010d}_atmo={4:1d}'.format(
            lat, long, theta, t, f
        )
        for lat, long, theta, t, f in
        zip(
            latitudes_deg_n, longitudes_deg_e, zenith_angles_rad,
            valid_times_unix_sec, standard_atmo_flags
        )
    ]


def parse_example_ids(example_id_strings):
    """Parses example IDs.

    E = number of examples

    :param example_id_strings: length-E list of example IDs.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['latitudes_deg_n']: length-E numpy array of latitudes (deg N).
    metadata_dict['longitudes_deg_e']: length-E numpy array of longitudes
        (deg E).
    metadata_dict['zenith_angles_rad']: length-E numpy array of solar zenith
        angles (radians).
    metadata_dict['valid_times_unix_sec']: length-E numpy array of valid times.
    metadata_dict['standard_atmo_flags']: length-E numpy array of standard-
        atmosphere flags (integers).
    """

    error_checking.assert_is_numpy_array(
        numpy.array(example_id_strings), num_dimensions=1
    )

    num_examples = len(example_id_strings)
    latitudes_deg_n = numpy.full(num_examples, numpy.nan)
    longitudes_deg_e = numpy.full(num_examples, numpy.nan)
    zenith_angles_rad = numpy.full(num_examples, numpy.nan)
    valid_times_unix_sec = numpy.full(num_examples, -1, dtype=int)
    standard_atmo_flags = numpy.full(num_examples, -1, dtype=int)

    for i in range(num_examples):
        these_words = example_id_strings[i].split('_')

        assert these_words[0].startswith('lat=')
        latitudes_deg_n[i] = float(these_words[0].replace('lat=', ''))

        assert these_words[1].startswith('long=')
        longitudes_deg_e[i] = float(these_words[1].replace('long=', ''))

        assert these_words[2].startswith('zenith-angle-rad=')
        zenith_angles_rad[i] = float(
            these_words[2].replace('zenith-angle-rad=', '')
        )

        assert these_words[3].startswith('time=')
        valid_times_unix_sec[i] = int(these_words[3].replace('time=', ''))

        assert these_words[4].startswith('atmo=')
        standard_atmo_flags[i] = int(these_words[4].replace('atmo=', ''))

    return {
        LATITUDES_KEY: latitudes_deg_n,
        LONGITUDES_KEY: longitudes_deg_e,
        ZENITH_ANGLES_KEY: zenith_angles_rad,
        VALID_TIMES_KEY: valid_times_unix_sec,
        STANDARD_ATMO_FLAGS_KEY: standard_atmo_flags
    }


def get_field_from_dict(example_dict, field_name, height_m_agl=None):
    """Returns field from dictionary of examples.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :param field_name: Name of field (may be predictor or target variable).
    :param height_m_agl: Height (metres above ground level).  For scalar field,
        `height_m_agl` will not be used.  For vector field, `height_m_agl` will
        be used only if `height_m_agl is not None`.
    :return: data_matrix: numpy array with data values for given field.
    """

    check_field_name(field_name)

    if field_name in ALL_SCALAR_PREDICTOR_NAMES:
        height_m_agl = None
        field_index = example_dict[SCALAR_PREDICTOR_NAMES_KEY].index(field_name)
        data_matrix = example_dict[SCALAR_PREDICTOR_VALS_KEY][..., field_index]
    elif field_name in ALL_SCALAR_TARGET_NAMES:
        height_m_agl = None
        field_index = example_dict[SCALAR_TARGET_NAMES_KEY].index(field_name)
        data_matrix = example_dict[SCALAR_TARGET_VALS_KEY][..., field_index]
    elif field_name in ALL_VECTOR_PREDICTOR_NAMES:
        field_index = example_dict[VECTOR_PREDICTOR_NAMES_KEY].index(field_name)
        data_matrix = example_dict[VECTOR_PREDICTOR_VALS_KEY][..., field_index]
    else:
        field_index = example_dict[VECTOR_TARGET_NAMES_KEY].index(field_name)
        data_matrix = example_dict[VECTOR_TARGET_VALS_KEY][..., field_index]

    if height_m_agl is None:
        return data_matrix

    height_index = match_heights(
        heights_m_agl=example_dict[HEIGHTS_KEY],
        desired_height_m_agl=height_m_agl
    )

    return data_matrix[..., height_index]


def reduce_sample_size(example_dict, num_examples_to_keep,
                       first_example_to_keep=None):
    """Reduces sample size by randomly removing examples.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :param num_examples_to_keep: Number of examples to keep.
    :param first_example_to_keep: Index of first example to keep.  If None, will
        subset examples randomly.
    :return: example_dict: Same as input but with fewer examples.
    :return: example_indices: 1-D numpy array with indices of examples kept.
    """

    error_checking.assert_is_integer(num_examples_to_keep)
    error_checking.assert_is_greater(num_examples_to_keep, 0)

    num_examples_total = len(example_dict[VALID_TIMES_KEY])
    all_indices = numpy.linspace(
        0, num_examples_total - 1, num=num_examples_total, dtype=int
    )

    if first_example_to_keep is not None:
        error_checking.assert_is_integer(first_example_to_keep)
        error_checking.assert_is_geq(first_example_to_keep, 0)
        all_indices = all_indices[all_indices >= first_example_to_keep]

    if len(all_indices) <= num_examples_to_keep:
        for this_key in ONE_PER_EXAMPLE_KEYS:
            example_dict[this_key] = example_dict[this_key][all_indices, ...]

        return example_dict, all_indices

    if first_example_to_keep is None:
        indices_to_keep = numpy.random.choice(
            all_indices, size=num_examples_to_keep, replace=False
        )
    else:
        if first_example_to_keep >= num_examples_total:
            indices_to_keep = numpy.array([], dtype=int)
        else:
            last_example_to_keep = min([
                first_example_to_keep + num_examples_to_keep - 1,
                num_examples_total - 1
            ])
            indices_to_keep = numpy.linspace(
                first_example_to_keep, last_example_to_keep,
                num=last_example_to_keep - first_example_to_keep + 1, dtype=int
            )

    for this_key in ONE_PER_EXAMPLE_KEYS:
        example_dict[this_key] = example_dict[this_key][indices_to_keep, ...]

    return example_dict, indices_to_keep


def subset_by_time(example_dict, first_time_unix_sec, last_time_unix_sec):
    """Subsets examples by time.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :param first_time_unix_sec: Earliest time to keep.
    :param last_time_unix_sec: Latest time to keep.
    :return: example_dict: Same as input but with fewer examples.
    :return: example_indices: 1-D numpy array with indices of examples kept.
    """

    error_checking.assert_is_integer(first_time_unix_sec)
    error_checking.assert_is_integer(last_time_unix_sec)
    error_checking.assert_is_geq(last_time_unix_sec, first_time_unix_sec)

    good_indices = numpy.where(numpy.logical_and(
        example_dict[VALID_TIMES_KEY] >= first_time_unix_sec,
        example_dict[VALID_TIMES_KEY] <= last_time_unix_sec
    ))[0]

    for this_key in ONE_PER_EXAMPLE_KEYS:
        example_dict[this_key] = example_dict[this_key][good_indices, ...]

    return example_dict, good_indices


def subset_by_standard_atmo(example_dict, standard_atmo_enum):
    """Subsets examples by standard-atmosphere type.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :param standard_atmo_enum: See doc for `check_standard_atmo_type`.
    :return: example_dict: Same as input but with fewer examples.
    :return: example_indices: 1-D numpy array with indices of examples kept.
    """

    check_standard_atmo_type(standard_atmo_enum)

    good_indices = numpy.where(
        example_dict[STANDARD_ATMO_FLAGS_KEY] == standard_atmo_enum,
    )[0]

    for this_key in ONE_PER_EXAMPLE_KEYS:
        example_dict[this_key] = example_dict[this_key][good_indices, ...]

    return example_dict, good_indices


def subset_by_field(example_dict, field_names):
    """Subsets examples by field.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :param field_names: 1-D list of field names to keep (each must be accepted
        by `check_field_name`).
    :return: example_dict: Same as input but with fewer fields.
    """

    error_checking.assert_is_numpy_array(
        numpy.array(field_names), num_dimensions=1
    )

    scalar_predictor_indices = []
    scalar_target_indices = []
    vector_predictor_indices = []
    vector_target_indices = []

    for this_field_name in field_names:
        check_field_name(this_field_name)

        if this_field_name in ALL_SCALAR_PREDICTOR_NAMES:
            scalar_predictor_indices.append(
                example_dict[SCALAR_PREDICTOR_NAMES_KEY].index(this_field_name)
            )
        elif this_field_name in ALL_SCALAR_TARGET_NAMES:
            scalar_target_indices.append(
                example_dict[SCALAR_TARGET_NAMES_KEY].index(this_field_name)
            )
        elif this_field_name in ALL_VECTOR_PREDICTOR_NAMES:
            vector_predictor_indices.append(
                example_dict[VECTOR_PREDICTOR_NAMES_KEY].index(this_field_name)
            )
        else:
            vector_target_indices.append(
                example_dict[VECTOR_TARGET_NAMES_KEY].index(this_field_name)
            )

    scalar_predictor_indices = numpy.array(scalar_predictor_indices, dtype=int)
    scalar_target_indices = numpy.array(scalar_target_indices, dtype=int)
    vector_predictor_indices = numpy.array(vector_predictor_indices, dtype=int)
    vector_target_indices = numpy.array(vector_target_indices, dtype=int)

    example_dict[SCALAR_PREDICTOR_NAMES_KEY] = [
        example_dict[SCALAR_PREDICTOR_NAMES_KEY][k]
        for k in scalar_predictor_indices
    ]
    example_dict[SCALAR_TARGET_NAMES_KEY] = [
        example_dict[SCALAR_TARGET_NAMES_KEY][k] for k in scalar_target_indices
    ]
    example_dict[VECTOR_PREDICTOR_NAMES_KEY] = [
        example_dict[VECTOR_PREDICTOR_NAMES_KEY][k]
        for k in vector_predictor_indices
    ]
    example_dict[VECTOR_TARGET_NAMES_KEY] = [
        example_dict[VECTOR_TARGET_NAMES_KEY][k] for k in vector_target_indices
    ]

    example_dict[SCALAR_PREDICTOR_VALS_KEY] = (
        example_dict[SCALAR_PREDICTOR_VALS_KEY][..., scalar_predictor_indices]
    )
    example_dict[SCALAR_TARGET_VALS_KEY] = (
        example_dict[SCALAR_TARGET_VALS_KEY][..., scalar_target_indices]
    )
    example_dict[VECTOR_PREDICTOR_VALS_KEY] = (
        example_dict[VECTOR_PREDICTOR_VALS_KEY][..., vector_predictor_indices]
    )
    example_dict[VECTOR_TARGET_VALS_KEY] = (
        example_dict[VECTOR_TARGET_VALS_KEY][..., vector_target_indices]
    )

    return example_dict


def subset_by_height(example_dict, heights_m_agl):
    """Subsets examples by height.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :param heights_m_agl: 1-D numpy array of heights to keep (metres above
        ground level).
    :return: example_dict: Same as input but with fewer heights.
    """

    error_checking.assert_is_numpy_array_without_nan(heights_m_agl)
    error_checking.assert_is_numpy_array(
        numpy.array(heights_m_agl), num_dimensions=1
    )

    indices_to_keep = [
        match_heights(
            heights_m_agl=example_dict[HEIGHTS_KEY], desired_height_m_agl=h
        ) for h in heights_m_agl
    ]
    indices_to_keep = numpy.array(indices_to_keep, dtype=int)

    example_dict[VECTOR_PREDICTOR_VALS_KEY] = (
        example_dict[VECTOR_PREDICTOR_VALS_KEY][:, indices_to_keep, :]
    )
    example_dict[VECTOR_TARGET_VALS_KEY] = (
        example_dict[VECTOR_TARGET_VALS_KEY][:, indices_to_keep, :]
    )
    example_dict[HEIGHTS_KEY] = example_dict[HEIGHTS_KEY][indices_to_keep]

    return example_dict


def subset_by_column_lwp(example_dict, min_lwp_kg_m02, max_lwp_kg_m02):
    """Subsets examples by full-column liquid-water path (LWP).

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :param min_lwp_kg_m02: Minimum LWP desired (kg m^-2).
    :param max_lwp_kg_m02: Maximum LWP desired (kg m^-2).
    :return: example_dict: Same as input but with fewer examples.
    :return: example_indices: 1-D numpy array with indices of examples kept.
    """

    error_checking.assert_is_geq(min_lwp_kg_m02, 0.)
    error_checking.assert_is_greater(max_lwp_kg_m02, min_lwp_kg_m02)

    column_lwp_values_kg_m02 = get_field_from_dict(
        example_dict=example_dict, field_name=COLUMN_LIQUID_WATER_PATH_NAME
    )

    good_indices = numpy.where(numpy.logical_and(
        column_lwp_values_kg_m02 >= min_lwp_kg_m02,
        column_lwp_values_kg_m02 <= max_lwp_kg_m02
    ))[0]

    for this_key in ONE_PER_EXAMPLE_KEYS:
        example_dict[this_key] = example_dict[this_key][good_indices, ...]

    return example_dict, good_indices


def find_examples(all_id_strings, desired_id_strings, allow_missing=False):
    """Finds examples with desired IDs.

    E = number of desired examples

    :param all_id_strings: 1-D list with all example IDs.
    :param desired_id_strings: length-E list of desired IDs.
    :param allow_missing: Boolean flag.  If True, will allow some desired IDs to
        be missing.  If False, will raise error if any desired ID is missing.
    :return: desired_indices: length-E numpy array with indices of desired
        examples.  Missing IDs are denoted by an index of -1.
    :raises: ValueError: if either list of IDs has non-unique entries.
    :raises: ValueError: if `allow_missing == False` and any desired ID is
        missing.
    """

    error_checking.assert_is_string_list(all_id_strings)
    error_checking.assert_is_string_list(desired_id_strings)
    error_checking.assert_is_boolean(allow_missing)

    all_id_strings_numpy = numpy.array(all_id_strings)
    desired_id_strings_numpy = numpy.array(desired_id_strings)

    these_unique_strings, these_counts = numpy.unique(
        all_id_strings_numpy, return_counts=True
    )
    if numpy.any(these_counts > 1):
        these_indices = numpy.where(these_counts > 1)[0]

        error_string = (
            '\nall_id_strings contains {0:d} repeated entries, listed below:'
            '\n{1:s}'
        ).format(
            len(these_indices), str(these_unique_strings[these_indices])
        )
        raise ValueError(error_string)

    these_unique_strings, these_counts = numpy.unique(
        desired_id_strings_numpy, return_counts=True
    )
    if numpy.any(these_counts > 1):
        these_indices = numpy.where(these_counts > 1)[0]

        error_string = (
            '\ndesired_id_strings contains {0:d} repeated entries, listed '
            'below:\n{1:s}'
        ).format(
            len(these_indices), str(these_unique_strings[these_indices])
        )
        raise ValueError(error_string)

    sort_indices = numpy.argsort(all_id_strings_numpy)
    desired_indices = numpy.searchsorted(
        all_id_strings_numpy[sort_indices], desired_id_strings_numpy,
        side='left'
    ).astype(int)

    desired_indices = sort_indices[desired_indices]
    desired_indices = numpy.maximum(desired_indices, 0)
    desired_indices = numpy.minimum(desired_indices, len(all_id_strings) - 1)

    if allow_missing:
        bad_indices = numpy.where(
            all_id_strings_numpy[desired_indices] != desired_id_strings_numpy
        )[0]
        desired_indices[bad_indices] = -1
        return desired_indices

    if numpy.array_equal(
            all_id_strings_numpy[desired_indices], desired_id_strings_numpy
    ):
        return desired_indices

    missing_flags = (
        all_id_strings_numpy[desired_indices] != desired_id_strings_numpy
    )

    error_string = (
        '{0:d} of {1:d} desired IDs (listed below) are missing:\n{2:s}'
    ).format(
        numpy.sum(missing_flags), len(desired_id_strings),
        str(desired_id_strings_numpy[missing_flags])
    )

    raise ValueError(error_string)


def subset_by_index(example_dict, desired_indices):
    """Subsets examples by index.

    :param example_dict: See doc for `read_file`.
    :param desired_indices: 1-D numpy array of desired indices.
    :return: example_dict: Same as input but with fewer examples.
    """

    error_checking.assert_is_numpy_array(desired_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_indices)
    error_checking.assert_is_geq_numpy_array(desired_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_indices, len(example_dict[VALID_TIMES_KEY])
    )

    for this_key in ONE_PER_EXAMPLE_KEYS:
        example_dict[this_key] = example_dict[this_key][desired_indices, ...]

    return example_dict


def average_examples(
        example_dict, use_pmm,
        max_pmm_percentile_level=DEFAULT_MAX_PMM_PERCENTILE_LEVEL):
    """Averages predictor and target fields over many examples.

    H = number of heights
    P_s = number of scalar predictors
    P_v = number of vector predictors
    T_s = number of scalar targets
    T_v = number of vector targets

    :param example_dict: See doc for `read_file`.
    :param use_pmm: Boolean flag.  If True, will use probability-matched means
        for vector fields (vertical profiles).  If False, will use arithmetic
        means for vector fields.
    :param max_pmm_percentile_level: [used only if `use_pmm == True`]
        Max percentile level for probability-matched means.
    :return: mean_example_dict: Dictionary with the following keys.
    mean_example_dict['scalar_predictor_matrix']: numpy array (1 x P_s) with
        values of scalar predictors.
    mean_example_dict['scalar_predictor_names']: Same as input.
    mean_example_dict['vector_predictor_matrix']: numpy array (1 x H x P_v) with
        values of vector predictors.
    mean_example_dict['vector_predictor_names']: Same as input.
    mean_example_dict['scalar_target_matrix']: numpy array (1 x T_s) with values
        of scalar targets.
    mean_example_dict['scalar_predictor_names']: Same as input.
    mean_example_dict['vector_target_matrix']: numpy array (1 x H x T_v) with
        values of vector targets.
    mean_example_dict['vector_predictor_names']: Same as input.
    mean_example_dict['heights_m_agl']: length-H numpy array of heights (metres
        above ground level).
    """

    error_checking.assert_is_boolean(use_pmm)
    error_checking.assert_is_geq(max_pmm_percentile_level, 90.)
    error_checking.assert_is_leq(max_pmm_percentile_level, 100.)

    mean_scalar_predictor_matrix = numpy.mean(
        example_dict[SCALAR_PREDICTOR_VALS_KEY], axis=0
    )
    mean_scalar_predictor_matrix = numpy.expand_dims(
        mean_scalar_predictor_matrix, axis=0
    )

    mean_scalar_target_matrix = numpy.mean(
        example_dict[SCALAR_TARGET_VALS_KEY], axis=0
    )
    mean_scalar_target_matrix = numpy.expand_dims(
        mean_scalar_target_matrix, axis=0
    )

    if use_pmm:
        mean_vector_predictor_matrix = pmm.run_pmm_many_variables(
            input_matrix=example_dict[VECTOR_PREDICTOR_VALS_KEY],
            max_percentile_level=max_pmm_percentile_level
        )
    else:
        mean_vector_predictor_matrix = numpy.mean(
            example_dict[VECTOR_PREDICTOR_VALS_KEY], axis=0
        )

    mean_vector_predictor_matrix = numpy.expand_dims(
        mean_vector_predictor_matrix, axis=0
    )

    if use_pmm:
        mean_vector_target_matrix = pmm.run_pmm_many_variables(
            input_matrix=example_dict[VECTOR_TARGET_VALS_KEY],
            max_percentile_level=max_pmm_percentile_level
        )
    else:
        mean_vector_target_matrix = numpy.mean(
            example_dict[VECTOR_TARGET_VALS_KEY], axis=0
        )

    mean_vector_target_matrix = numpy.expand_dims(
        mean_vector_target_matrix, axis=0
    )

    return {
        SCALAR_PREDICTOR_NAMES_KEY: example_dict[SCALAR_PREDICTOR_NAMES_KEY],
        SCALAR_PREDICTOR_VALS_KEY: mean_scalar_predictor_matrix,
        SCALAR_TARGET_NAMES_KEY: example_dict[SCALAR_TARGET_NAMES_KEY],
        SCALAR_TARGET_VALS_KEY: mean_scalar_target_matrix,
        VECTOR_PREDICTOR_NAMES_KEY: example_dict[VECTOR_PREDICTOR_NAMES_KEY],
        VECTOR_PREDICTOR_VALS_KEY: mean_vector_predictor_matrix,
        VECTOR_TARGET_NAMES_KEY: example_dict[VECTOR_TARGET_NAMES_KEY],
        VECTOR_TARGET_VALS_KEY: mean_vector_target_matrix,
        HEIGHTS_KEY: example_dict[HEIGHTS_KEY]
    }
