"""Helper methods for learning examples."""

import copy
import warnings
import numpy
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import prob_matched_means as pmm
from gewittergefahr.gg_utils import temperature_conversions as temp_conversions
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from ml4rt.utils import trace_gases
from ml4rt.utils import land_ocean_mask

TOLERANCE = 1e-6

DAYS_TO_SECONDS = 86400.
GRAVITY_CONSTANT_M_S02 = 9.8066
DRY_AIR_SPECIFIC_HEAT_J_KG01_K01 = 1004.
# GRAVITY_CONSTANT_M_S02 = 9.80665
# DRY_AIR_SPECIFIC_HEAT_J_KG01_K01 = 287.04 * 3.5

LIQUID_EFF_RADIUS_LAND_MEAN_METRES = 6e-6
LIQUID_EFF_RADIUS_LAND_STDEV_METRES = 1e-6
LIQUID_EFF_RADIUS_OCEAN_MEAN_METRES = 9.5e-6
LIQUID_EFF_RADIUS_OCEAN_STDEV_METRES = 1.2e-6
ICE_EFF_RADIUS_INTERCEPT_METRES = 86.73e-6
ICE_EFF_RADIUS_SLOPE_METRES_CELSIUS01 = 1.07e-6
MIN_EFFECTIVE_RADIUS_METRES = 1e-6

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
EXAMPLE_IDS_KEY = 'example_id_strings'

LATITUDES_KEY = 'latitudes_deg_n'
LONGITUDES_KEY = 'longitudes_deg_e'
ZENITH_ANGLES_KEY = 'zenith_angles_rad'
ALBEDOS_KEY = 'albedos'
TEMPERATURES_10M_KEY = 'temperatures_10m_kelvins'

DICTIONARY_KEYS = [
    SCALAR_PREDICTOR_VALS_KEY, SCALAR_PREDICTOR_NAMES_KEY,
    VECTOR_PREDICTOR_VALS_KEY, VECTOR_PREDICTOR_NAMES_KEY,
    SCALAR_TARGET_VALS_KEY, SCALAR_TARGET_NAMES_KEY,
    VECTOR_TARGET_VALS_KEY, VECTOR_TARGET_NAMES_KEY,
    VALID_TIMES_KEY, HEIGHTS_KEY, STANDARD_ATMO_FLAGS_KEY, EXAMPLE_IDS_KEY
]
ONE_PER_EXAMPLE_KEYS = [
    SCALAR_PREDICTOR_VALS_KEY, VECTOR_PREDICTOR_VALS_KEY,
    SCALAR_TARGET_VALS_KEY, VECTOR_TARGET_VALS_KEY,
    VALID_TIMES_KEY, STANDARD_ATMO_FLAGS_KEY, EXAMPLE_IDS_KEY
]
ONE_PER_FIELD_KEYS = [
    SCALAR_PREDICTOR_VALS_KEY, SCALAR_PREDICTOR_NAMES_KEY,
    VECTOR_PREDICTOR_VALS_KEY, VECTOR_PREDICTOR_NAMES_KEY,
    SCALAR_TARGET_VALS_KEY, SCALAR_TARGET_NAMES_KEY,
    VECTOR_TARGET_VALS_KEY, VECTOR_TARGET_NAMES_KEY
]

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
RELATIVE_HUMIDITY_NAME = 'relative_humidity_unitless'
LIQUID_WATER_CONTENT_NAME = 'liquid_water_content_kg_m03'
ICE_WATER_CONTENT_NAME = 'ice_water_content_kg_m03'
LIQUID_WATER_PATH_NAME = 'liquid_water_path_kg_m02'
ICE_WATER_PATH_NAME = 'ice_water_path_kg_m02'
WATER_VAPOUR_PATH_NAME = 'vapour_path_kg_m02'
UPWARD_LIQUID_WATER_PATH_NAME = 'upward_liquid_water_path_kg_m02'
UPWARD_ICE_WATER_PATH_NAME = 'upward_ice_water_path_kg_m02'
UPWARD_WATER_VAPOUR_PATH_NAME = 'upward_vapour_path_kg_m02'

LIQUID_EFF_RADIUS_NAME = 'liquid_effective_radius_metres'
ICE_EFF_RADIUS_NAME = 'ice_effective_radius_metres'
O2_MIXING_RATIO_NAME = 'o2_mixing_ratio_kg_kg01'
CO2_MIXING_RATIO_NAME = 'co2_mixing_ratio_kg_kg01'
CH4_MIXING_RATIO_NAME = 'ch4_mixing_ratio_kg_kg01'
N2O_MIXING_RATIO_NAME = 'n2o_mixing_ratio_kg_kg01'
AEROSOL_OPTICAL_DEPTH_NAME = 'aerosol_optical_depth_metres'
AEROSOL_ALBEDO_NAME = 'aerosol_single_scattering_albedo'
AEROSOL_ASYMMETRY_PARAM_NAME = 'aerosol_asymmetry_param'

TRACE_GAS_NAMES = [
    O2_MIXING_RATIO_NAME, CO2_MIXING_RATIO_NAME,
    CH4_MIXING_RATIO_NAME, N2O_MIXING_RATIO_NAME
]
EFFECTIVE_RADIUS_NAMES = [LIQUID_EFF_RADIUS_NAME, ICE_EFF_RADIUS_NAME]
AEROSOL_NAMES = [
    AEROSOL_OPTICAL_DEPTH_NAME, AEROSOL_ALBEDO_NAME,
    AEROSOL_ASYMMETRY_PARAM_NAME
]

ALL_SCALAR_PREDICTOR_NAMES = [
    ZENITH_ANGLE_NAME, LATITUDE_NAME, LONGITUDE_NAME, ALBEDO_NAME,
    COLUMN_LIQUID_WATER_PATH_NAME, COLUMN_ICE_WATER_PATH_NAME
]

BASIC_VECTOR_PREDICTOR_NAMES = [
    PRESSURE_NAME, TEMPERATURE_NAME, SPECIFIC_HUMIDITY_NAME,
    LIQUID_WATER_CONTENT_NAME, ICE_WATER_CONTENT_NAME,
    RELATIVE_HUMIDITY_NAME, LIQUID_WATER_PATH_NAME, ICE_WATER_PATH_NAME,
    WATER_VAPOUR_PATH_NAME, UPWARD_LIQUID_WATER_PATH_NAME,
    UPWARD_ICE_WATER_PATH_NAME, UPWARD_WATER_VAPOUR_PATH_NAME
]

ALL_VECTOR_PREDICTOR_NAMES = BASIC_VECTOR_PREDICTOR_NAMES + [
    LIQUID_EFF_RADIUS_NAME, ICE_EFF_RADIUS_NAME,
    O2_MIXING_RATIO_NAME, CO2_MIXING_RATIO_NAME, CH4_MIXING_RATIO_NAME,
    N2O_MIXING_RATIO_NAME, AEROSOL_OPTICAL_DEPTH_NAME, AEROSOL_ALBEDO_NAME,
    AEROSOL_ASYMMETRY_PARAM_NAME
]

ALL_PREDICTOR_NAMES = ALL_SCALAR_PREDICTOR_NAMES + ALL_VECTOR_PREDICTOR_NAMES

SHORTWAVE_HEATING_RATE_NAME = 'shortwave_heating_rate_k_day01'
SHORTWAVE_DOWN_FLUX_NAME = 'shortwave_down_flux_w_m02'
SHORTWAVE_UP_FLUX_NAME = 'shortwave_up_flux_w_m02'
SHORTWAVE_DOWN_FLUX_INC_NAME = 'shortwave_down_flux_increment_w_m03'
SHORTWAVE_UP_FLUX_INC_NAME = 'shortwave_up_flux_increment_w_m03'
SHORTWAVE_SURFACE_DOWN_FLUX_NAME = 'shortwave_surface_down_flux_w_m02'
SHORTWAVE_TOA_UP_FLUX_NAME = 'shortwave_toa_up_flux_w_m02'

ALL_SCALAR_TARGET_NAMES = [
    SHORTWAVE_SURFACE_DOWN_FLUX_NAME, SHORTWAVE_TOA_UP_FLUX_NAME
]

ALL_VECTOR_TARGET_NAMES = [
    SHORTWAVE_DOWN_FLUX_NAME, SHORTWAVE_UP_FLUX_NAME,
    SHORTWAVE_HEATING_RATE_NAME,
    SHORTWAVE_DOWN_FLUX_INC_NAME, SHORTWAVE_UP_FLUX_INC_NAME
]

ALL_TARGET_NAMES = ALL_SCALAR_TARGET_NAMES + ALL_VECTOR_TARGET_NAMES


def _find_nonzero_runs(values):
    """Finds runs of non-zero values in numpy array.

    N = number of non-zero runs

    :param values: 1-D numpy array of real values.
    :return: start_indices: length-N numpy array with array index at start of
        each non-zero run.
    :return: end_indices: length-N numpy array with array index at end of each
        non-zero run.
    """

    error_checking.assert_is_numpy_array_without_nan(values)
    error_checking.assert_is_numpy_array(values, num_dimensions=1)

    zero_flags = numpy.concatenate((
        [True], numpy.equal(values, 0), [True]
    ))

    nonzero_flags = numpy.invert(zero_flags)
    differences = numpy.abs(numpy.diff(nonzero_flags))
    index_matrix = numpy.where(differences == 1)[0].reshape(-1, 2)

    return index_matrix[:, 0], index_matrix[:, 1] - 1


def _add_height_padding(example_dict, desired_heights_m_agl):
    """Adds height-padding to profiles.

    :param example_dict: See doc for `example_io.read_file`.
    :param desired_heights_m_agl: 1-D numpy array with all desired heights (real
        and fake), in metres above ground level.
    :return: example_dict: Same as input but with extra heights.
    :raises: ValueError: if `desired_heights_m_agl` contains anything other than
        heights currently in the example dict, followed by heights not in the
        example dict.
    """

    error_checking.assert_is_numpy_array(
        desired_heights_m_agl, num_dimensions=1
    )
    error_checking.assert_is_geq_numpy_array(desired_heights_m_agl, 0.)

    current_heights_m_agl = example_dict[HEIGHTS_KEY]
    desired_heights_m_agl = numpy.sort(desired_heights_m_agl)

    num_desired_heights = len(desired_heights_m_agl)
    first_new_height_index = None

    for j in range(num_desired_heights):
        found_this_height = False

        try:
            match_heights(
                heights_m_agl=current_heights_m_agl,
                desired_height_m_agl=desired_heights_m_agl[j]
            )
            found_this_height = True
        except ValueError:
            if desired_heights_m_agl[j] <= numpy.max(current_heights_m_agl):
                raise

        if found_this_height:
            if first_new_height_index is None:
                continue

            error_string = (
                'desired_heights_m_agl should contain heights present in the '
                'example dict, followed by heights not present in the example.'
                '  However, desired_heights_m_agl contains {0:d} m AGL (not '
                'present), followed by {1:d} m AGL (present).'
            ).format(
                int(numpy.round(desired_heights_m_agl[j])),
                int(numpy.round(desired_heights_m_agl[first_new_height_index]))
            )

            raise ValueError(error_string)

        if first_new_height_index is not None:
            continue

        first_new_height_index = j

    if first_new_height_index is None:
        return example_dict

    new_heights_m_agl = desired_heights_m_agl[first_new_height_index:]
    example_dict[HEIGHTS_KEY] = numpy.concatenate(
        (example_dict[HEIGHTS_KEY], new_heights_m_agl), axis=0
    )

    num_new_heights = len(new_heights_m_agl)
    pad_width_input_arg = (
        (0, 0), (0, num_new_heights), (0, 0)
    )

    example_dict[VECTOR_PREDICTOR_VALS_KEY] = numpy.pad(
        example_dict[VECTOR_PREDICTOR_VALS_KEY],
        pad_width=pad_width_input_arg, mode='edge'
    )
    example_dict[VECTOR_TARGET_VALS_KEY] = numpy.pad(
        example_dict[VECTOR_TARGET_VALS_KEY],
        pad_width=pad_width_input_arg, mode='constant', constant_values=0.
    )

    return example_dict


def _interp_mixing_ratios(orig_mixing_ratios_kg_kg01, orig_heights_m_asl,
                          new_heights_m_agl):
    """Interpolates mixing ratios to new heights.

    h = number of heights in original grid
    H = number of heights in new grid

    Note that this method does *not* try to match heights above ground and above
    sea level.  It assumes that ground level = sea level everywhere.

    :param orig_mixing_ratios_kg_kg01: length-h numpy array of mixing ratios.
    :param orig_heights_m_asl: length-h numpy array of heights (metres above sea
        level).
    :param new_heights_m_agl: length-H numpy array of heights (metres above
        ground level).
    :return: new_mixing_ratios_kg_kg01: length-H numpy array of mixing ratios.
    """

    interp_object = interp1d(
        x=orig_heights_m_asl, y=orig_mixing_ratios_kg_kg01,
        kind='linear', bounds_error=True, assume_sorted=True
    )

    return interp_object(new_heights_m_agl)


def _example_ids_to_standard_atmos(example_id_strings):
    """Converts each example ID to standard atmosphere.

    E = number of examples

    :param example_id_strings: length-E list of example IDs, in format required
        by `parse_example_ids`.
    :return: standard_atmo_enums: length-E numpy array of standard atmospheres,
        in format required by `check_standard_atmo_type`.
    """

    metadata_dict = parse_example_ids(example_id_strings)
    latitudes_deg_n = metadata_dict[LATITUDES_KEY]

    months = numpy.array([
        int(time_conversion.unix_sec_to_string(t, '%m'))
        for t in metadata_dict[VALID_TIMES_KEY]
    ], dtype=int)

    num_examples = len(example_id_strings)
    standard_atmo_enums = numpy.full(num_examples, -1, dtype=int)

    nh_midlatitude_indices = numpy.where(
        numpy.logical_and(latitudes_deg_n > 20, latitudes_deg_n < 65)
    )[0]
    nh_midlatitude_summer_flags = numpy.logical_and(
        months[nh_midlatitude_indices] >= 5,
        months[nh_midlatitude_indices] <= 10
    )
    standard_atmo_enums[
        nh_midlatitude_indices[nh_midlatitude_summer_flags]
    ] = MIDLATITUDE_SUMMER_ENUM
    standard_atmo_enums[
        nh_midlatitude_indices[nh_midlatitude_summer_flags == False]
    ] = MIDLATITUDE_WINTER_ENUM

    sh_midlatitude_indices = numpy.where(
        numpy.logical_and(latitudes_deg_n > -65, latitudes_deg_n < -20)
    )[0]
    sh_midlatitude_summer_flags = numpy.invert(numpy.logical_and(
        months[sh_midlatitude_indices] >= 5,
        months[sh_midlatitude_indices] <= 10
    ))
    standard_atmo_enums[
        sh_midlatitude_indices[sh_midlatitude_summer_flags]
    ] = MIDLATITUDE_SUMMER_ENUM
    standard_atmo_enums[
        sh_midlatitude_indices[sh_midlatitude_summer_flags == False]
    ] = MIDLATITUDE_WINTER_ENUM

    nh_arctic_indices = numpy.where(latitudes_deg_n >= 65)[0]
    nh_arctic_summer_flags = numpy.logical_and(
        months[nh_arctic_indices] >= 5, months[nh_arctic_indices] <= 10
    )
    standard_atmo_enums[
        nh_arctic_indices[nh_arctic_summer_flags]
    ] = SUBARCTIC_SUMMER_ENUM
    standard_atmo_enums[
        nh_arctic_indices[nh_arctic_summer_flags == False]
    ] = SUBARCTIC_WINTER_ENUM

    sh_arctic_indices = numpy.where(latitudes_deg_n <= -65)[0]
    sh_arctic_summer_flags = numpy.invert(numpy.logical_and(
        months[sh_arctic_indices] >= 5, months[sh_arctic_indices] <= 10
    ))
    standard_atmo_enums[
        sh_arctic_indices[sh_arctic_summer_flags]
    ] = SUBARCTIC_SUMMER_ENUM
    standard_atmo_enums[
        sh_arctic_indices[sh_arctic_summer_flags == False]
    ] = SUBARCTIC_WINTER_ENUM

    unfilled_indices = numpy.where(standard_atmo_enums == -1)[0]
    tropical_indices = numpy.where(numpy.absolute(latitudes_deg_n) <= 20)[0]
    assert numpy.array_equal(unfilled_indices, tropical_indices)

    standard_atmo_enums[tropical_indices] = TROPICS_ENUM
    return standard_atmo_enums


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
        `example_io.read_file`).
    :return: example_dict: Same but with heating-rate profiles.
    """

    down_flux_matrix_w_m02 = get_field_from_dict(
        example_dict=example_dict, field_name=SHORTWAVE_DOWN_FLUX_NAME
    )
    up_flux_matrix_w_m02 = get_field_from_dict(
        example_dict=example_dict, field_name=SHORTWAVE_UP_FLUX_NAME
    )
    pressure_matrix_pascals = get_field_from_dict(
        example_dict=example_dict, field_name=PRESSURE_NAME
    ) + 0.

    dummy_pressure_matrix_pascals = (
        pressure_matrix_pascals[:, [-1]] +
        (pressure_matrix_pascals[:, [-1]] - pressure_matrix_pascals[:, [-2]])
    )
    pressure_matrix_pascals = numpy.concatenate(
        (pressure_matrix_pascals, dummy_pressure_matrix_pascals), axis=1
    )

    net_flux_matrix_w_m02 = down_flux_matrix_w_m02 - up_flux_matrix_w_m02
    dummy_net_flux_matrix_w_m02 = (
        net_flux_matrix_w_m02[:, [-1]] +
        (net_flux_matrix_w_m02[:, [-1]] - net_flux_matrix_w_m02[:, [-2]])
    )
    net_flux_matrix_w_m02 = numpy.concatenate(
        (net_flux_matrix_w_m02, dummy_net_flux_matrix_w_m02), axis=1
    )

    coefficient = GRAVITY_CONSTANT_M_S02 / DRY_AIR_SPECIFIC_HEAT_J_KG01_K01

    # heating_rate_matrix_k_day01 = DAYS_TO_SECONDS * coefficient * (
    #     numpy.gradient(net_flux_matrix_w_m02, axis=1) /
    #     numpy.absolute(numpy.gradient(pressure_matrix_pascals, axis=1))
    # )

    heating_rate_matrix_k_day01 = DAYS_TO_SECONDS * coefficient * (
        numpy.diff(net_flux_matrix_w_m02, axis=1) /
        numpy.absolute(numpy.diff(pressure_matrix_pascals, axis=1))
    )

    error_checking.assert_is_numpy_array_without_nan(net_flux_matrix_w_m02)
    error_checking.assert_is_numpy_array_without_nan(pressure_matrix_pascals)
    heating_rate_matrix_k_day01[numpy.isnan(heating_rate_matrix_k_day01)] = 0.

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
    upwelling- and downwelling-flux increments added by the [j]th layer,
    divided by the pressure difference over the [j]th layer.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :return: example_dict: Same but with flux-increment profiles.
    """

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

    down_flux_matrix_w_m02 = get_field_from_dict(
        example_dict=example_dict, field_name=SHORTWAVE_DOWN_FLUX_NAME
    )
    up_flux_matrix_w_m02 = get_field_from_dict(
        example_dict=example_dict, field_name=SHORTWAVE_UP_FLUX_NAME
    )

    down_flux_increment_matrix_w_m03 = numpy.diff(
        down_flux_matrix_w_m02, axis=1, prepend=0.
    ) / grid_cell_width_matrix_metres

    up_flux_increment_matrix_w_m03 = numpy.diff(
        up_flux_matrix_w_m02, axis=1, prepend=0.
    ) / grid_cell_width_matrix_metres

    # down_flux_increment_matrix_w_m03 = numpy.maximum(
    #     down_flux_increment_matrix_w_m03, 0.
    # )
    # up_flux_increment_matrix_w_m03 = numpy.maximum(
    #     up_flux_increment_matrix_w_m03, 0.
    # )

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
            down_flux_increment_matrix_w_m03
        )
    else:
        example_dict[VECTOR_TARGET_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_TARGET_VALS_KEY], obj=down_increment_index,
            values=down_flux_increment_matrix_w_m03, axis=-1
        )

    if found_up_increment:
        example_dict[VECTOR_TARGET_VALS_KEY][..., up_increment_index] = (
            up_flux_increment_matrix_w_m03
        )
    else:
        example_dict[VECTOR_TARGET_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_TARGET_VALS_KEY], obj=up_increment_index,
            values=up_flux_increment_matrix_w_m03, axis=-1
        )

    return example_dict


def fluxes_increments_to_actual(example_dict):
    """For each example, converts flux-increment profiles to flux profiles.

    This method is the inverse of `fluxes_actual_to_increments`.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :return: example_dict: Same but with actual flux profiles.
    """

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

    down_flux_increment_matrix_w_m03 = get_field_from_dict(
        example_dict=example_dict, field_name=SHORTWAVE_DOWN_FLUX_INC_NAME
    )
    up_flux_increment_matrix_w_m03 = get_field_from_dict(
        example_dict=example_dict, field_name=SHORTWAVE_UP_FLUX_INC_NAME
    )

    down_flux_matrix_w_m02 = numpy.cumsum(
        down_flux_increment_matrix_w_m03 * grid_cell_width_matrix_metres,
        axis=1
    )
    up_flux_matrix_w_m02 = numpy.cumsum(
        up_flux_increment_matrix_w_m03 * grid_cell_width_matrix_metres,
        axis=1
    )

    down_flux_matrix_w_m02 = numpy.maximum(down_flux_matrix_w_m02, 0.)
    up_flux_matrix_w_m02 = numpy.maximum(up_flux_matrix_w_m02, 0.)

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


def add_trace_gases(example_dict, noise_stdev_fractional):
    """Adds trace-gas profiles to dictionary.

    This method handles four trace gases: O2, CO2, N2O, CH4.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).  Must contain temperature profiles.
    :param noise_stdev_fractional: Standard deviation of Gaussian noise.  If you
        do not want to add Gaussian noise to profiles, make this non-positive.
    :return: example_dict: Same but with trace-gas profiles.
    """

    error_checking.assert_is_less_than(noise_stdev_fractional, 1.)
    if noise_stdev_fractional <= 0:
        noise_stdev_fractional = None

    mixing_ratio_dict = trace_gases.read_profiles()
    standard_atmo_enum_by_example = _example_ids_to_standard_atmos(
        example_dict[EXAMPLE_IDS_KEY]
    )

    num_examples = len(example_dict[EXAMPLE_IDS_KEY])
    num_heights = len(example_dict[HEIGHTS_KEY])
    these_dim = (num_examples, num_heights)

    o2_mixing_ratio_matrix_kg_kg01 = numpy.full(these_dim, numpy.nan)
    co2_mixing_ratio_matrix_kg_kg01 = numpy.full(these_dim, numpy.nan)
    ch4_mixing_ratio_matrix_kg_kg01 = numpy.full(these_dim, numpy.nan)
    n2o_mixing_ratio_matrix_kg_kg01 = numpy.full(these_dim, numpy.nan)

    for i in STANDARD_ATMO_ENUMS:
        example_indices = numpy.where(standard_atmo_enum_by_example == i)[0]
        if len(example_indices) == 0:
            continue

        new_mixing_ratios_kg_kg01 = _interp_mixing_ratios(
            orig_mixing_ratios_kg_kg01=
            mixing_ratio_dict[trace_gases.O2_MIXING_RATIOS_KEY][i - 1, :],
            orig_heights_m_asl=mixing_ratio_dict[trace_gases.HEIGHTS_KEY],
            new_heights_m_agl=example_dict[HEIGHTS_KEY]
        )
        for k in example_indices:
            o2_mixing_ratio_matrix_kg_kg01[k, :] = new_mixing_ratios_kg_kg01

        if noise_stdev_fractional is not None:
            noise_matrix = numpy.random.normal(
                loc=0., scale=noise_stdev_fractional,
                size=o2_mixing_ratio_matrix_kg_kg01[example_indices, :].shape
            )
            o2_mixing_ratio_matrix_kg_kg01[example_indices, :] += (
                o2_mixing_ratio_matrix_kg_kg01[example_indices, :] *
                noise_matrix
            )

        new_mixing_ratios_kg_kg01 = _interp_mixing_ratios(
            orig_mixing_ratios_kg_kg01=
            mixing_ratio_dict[trace_gases.CO2_MIXING_RATIOS_KEY][i - 1, :],
            orig_heights_m_asl=mixing_ratio_dict[trace_gases.HEIGHTS_KEY],
            new_heights_m_agl=example_dict[HEIGHTS_KEY]
        )
        for k in example_indices:
            co2_mixing_ratio_matrix_kg_kg01[k, :] = new_mixing_ratios_kg_kg01

        if noise_stdev_fractional is not None:
            noise_matrix = numpy.random.normal(
                loc=0., scale=noise_stdev_fractional,
                size=co2_mixing_ratio_matrix_kg_kg01[example_indices, :].shape
            )
            co2_mixing_ratio_matrix_kg_kg01[example_indices, :] += (
                co2_mixing_ratio_matrix_kg_kg01[example_indices, :] *
                noise_matrix
            )

        new_mixing_ratios_kg_kg01 = _interp_mixing_ratios(
            orig_mixing_ratios_kg_kg01=
            mixing_ratio_dict[trace_gases.CH4_MIXING_RATIOS_KEY][i - 1, :],
            orig_heights_m_asl=mixing_ratio_dict[trace_gases.HEIGHTS_KEY],
            new_heights_m_agl=example_dict[HEIGHTS_KEY]
        )
        for k in example_indices:
            ch4_mixing_ratio_matrix_kg_kg01[k, :] = new_mixing_ratios_kg_kg01

        if noise_stdev_fractional is not None:
            noise_matrix = numpy.random.normal(
                loc=0., scale=noise_stdev_fractional,
                size=ch4_mixing_ratio_matrix_kg_kg01[example_indices, :].shape
            )
            ch4_mixing_ratio_matrix_kg_kg01[example_indices, :] += (
                ch4_mixing_ratio_matrix_kg_kg01[example_indices, :] *
                noise_matrix
            )

        new_mixing_ratios_kg_kg01 = _interp_mixing_ratios(
            orig_mixing_ratios_kg_kg01=
            mixing_ratio_dict[trace_gases.N2O_MIXING_RATIOS_KEY][i - 1, :],
            orig_heights_m_asl=mixing_ratio_dict[trace_gases.HEIGHTS_KEY],
            new_heights_m_agl=example_dict[HEIGHTS_KEY]
        )
        for k in example_indices:
            n2o_mixing_ratio_matrix_kg_kg01[k, :] = new_mixing_ratios_kg_kg01

        if noise_stdev_fractional is not None:
            noise_matrix = numpy.random.normal(
                loc=0., scale=noise_stdev_fractional,
                size=n2o_mixing_ratio_matrix_kg_kg01[example_indices, :].shape
            )
            n2o_mixing_ratio_matrix_kg_kg01[example_indices, :] += (
                n2o_mixing_ratio_matrix_kg_kg01[example_indices, :] *
                noise_matrix
            )

    vector_predictor_names = example_dict[VECTOR_PREDICTOR_NAMES_KEY]
    found_o2 = O2_MIXING_RATIO_NAME in vector_predictor_names
    found_co2 = CO2_MIXING_RATIO_NAME in vector_predictor_names
    found_ch4 = CH4_MIXING_RATIO_NAME in vector_predictor_names
    found_n2o = N2O_MIXING_RATIO_NAME in vector_predictor_names

    if not found_o2:
        vector_predictor_names.append(O2_MIXING_RATIO_NAME)
    if not found_co2:
        vector_predictor_names.append(CO2_MIXING_RATIO_NAME)
    if not found_ch4:
        vector_predictor_names.append(CH4_MIXING_RATIO_NAME)
    if not found_n2o:
        vector_predictor_names.append(N2O_MIXING_RATIO_NAME)

    o2_index = vector_predictor_names.index(O2_MIXING_RATIO_NAME)
    co2_index = vector_predictor_names.index(CO2_MIXING_RATIO_NAME)
    ch4_index = vector_predictor_names.index(CH4_MIXING_RATIO_NAME)
    n2o_index = vector_predictor_names.index(N2O_MIXING_RATIO_NAME)
    example_dict[VECTOR_PREDICTOR_NAMES_KEY] = vector_predictor_names

    if found_o2:
        example_dict[VECTOR_PREDICTOR_VALS_KEY][..., o2_index] = (
            o2_mixing_ratio_matrix_kg_kg01
        )
    else:
        example_dict[VECTOR_PREDICTOR_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_PREDICTOR_VALS_KEY],
            obj=o2_index, values=o2_mixing_ratio_matrix_kg_kg01, axis=-1
        )

    if found_co2:
        example_dict[VECTOR_PREDICTOR_VALS_KEY][..., co2_index] = (
            co2_mixing_ratio_matrix_kg_kg01
        )
    else:
        example_dict[VECTOR_PREDICTOR_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_PREDICTOR_VALS_KEY],
            obj=co2_index, values=co2_mixing_ratio_matrix_kg_kg01, axis=-1
        )

    if found_ch4:
        example_dict[VECTOR_PREDICTOR_VALS_KEY][..., ch4_index] = (
            ch4_mixing_ratio_matrix_kg_kg01
        )
    else:
        example_dict[VECTOR_PREDICTOR_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_PREDICTOR_VALS_KEY],
            obj=ch4_index, values=ch4_mixing_ratio_matrix_kg_kg01, axis=-1
        )

    if found_n2o:
        example_dict[VECTOR_PREDICTOR_VALS_KEY][..., n2o_index] = (
            n2o_mixing_ratio_matrix_kg_kg01
        )
    else:
        example_dict[VECTOR_PREDICTOR_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_PREDICTOR_VALS_KEY],
            obj=n2o_index, values=n2o_mixing_ratio_matrix_kg_kg01, axis=-1
        )

    return example_dict


def add_effective_radii(example_dict, ice_noise_stdev_fractional,
                        test_mode=False):
    """Adds effective-radius profiles (for both ice and liquid) to dict.

    For effective radius of liquid water, using approximation of:
    https://doi.org/10.1175/1520-0469(2000)057%3C0295:CDSDIL%3E2.0.CO;2

    For effective radius of ice water, using:
    https://doi.org/10.1002/2013JD020602

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).  Must contain temperature profiles.
    :param ice_noise_stdev_fractional: Standard deviation of Gaussian noise for
        effective radius of ice water.  If you do not want to add Gaussian noise
        to profiles, make this non-positive.
    :param test_mode: Leave this alone.
    :return: example_dict: Same but with effective-radius profiles.
    """

    error_checking.assert_is_boolean(test_mode)
    if test_mode:
        ice_noise_stdev_fractional = -1.

    error_checking.assert_is_less_than(ice_noise_stdev_fractional, 1.)
    if ice_noise_stdev_fractional <= 0:
        ice_noise_stdev_fractional = None

    temperature_matrix_kelvins = get_field_from_dict(
        example_dict=example_dict, field_name=TEMPERATURE_NAME
    )
    temperature_matrix_celsius = temp_conversions.kelvins_to_celsius(
        temperature_matrix_kelvins
    )
    ice_eff_radius_matrix_metres = (
        ICE_EFF_RADIUS_INTERCEPT_METRES +
        ICE_EFF_RADIUS_SLOPE_METRES_CELSIUS01 * temperature_matrix_celsius
    )

    if ice_noise_stdev_fractional is not None:
        noise_matrix = numpy.random.normal(
            loc=0., scale=ice_noise_stdev_fractional,
            size=ice_eff_radius_matrix_metres.shape
        )
        ice_eff_radius_matrix_metres += (
            ice_eff_radius_matrix_metres * noise_matrix
        )

    ice_eff_radius_matrix_metres = numpy.maximum(
        ice_eff_radius_matrix_metres, MIN_EFFECTIVE_RADIUS_METRES
    )

    metadata_dict = parse_example_ids(example_dict[EXAMPLE_IDS_KEY])
    latitudes_deg_n = metadata_dict[LATITUDES_KEY]
    longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        metadata_dict[LONGITUDES_KEY]
    )

    land_flags = numpy.array([
        land_ocean_mask.is_land(lat=y, lon=x)
        for y, x in zip(latitudes_deg_n, longitudes_deg_e)
    ], dtype=bool)

    land_indices = numpy.where(land_flags)[0]
    ocean_indices = numpy.where(numpy.invert(land_flags))[0]
    liquid_eff_radius_matrix_metres = numpy.full(
        temperature_matrix_kelvins.shape, numpy.nan
    )

    if len(land_indices) > 0:
        liquid_eff_radius_matrix_metres[land_indices, :] = numpy.random.normal(
            loc=LIQUID_EFF_RADIUS_LAND_MEAN_METRES,
            scale=0. if test_mode else LIQUID_EFF_RADIUS_LAND_STDEV_METRES,
            size=liquid_eff_radius_matrix_metres[land_indices, :].shape
        )

    if len(ocean_indices) > 0:
        liquid_eff_radius_matrix_metres[ocean_indices, :] = numpy.random.normal(
            loc=LIQUID_EFF_RADIUS_OCEAN_MEAN_METRES,
            scale=0. if test_mode else LIQUID_EFF_RADIUS_OCEAN_STDEV_METRES,
            size=liquid_eff_radius_matrix_metres[ocean_indices, :].shape
        )

    liquid_eff_radius_matrix_metres = numpy.maximum(
        liquid_eff_radius_matrix_metres, MIN_EFFECTIVE_RADIUS_METRES
    )

    vector_predictor_names = example_dict[VECTOR_PREDICTOR_NAMES_KEY]
    found_liquid_eff_radius = LIQUID_EFF_RADIUS_NAME in vector_predictor_names
    found_ice_eff_radius = ICE_EFF_RADIUS_NAME in vector_predictor_names

    if not found_liquid_eff_radius:
        vector_predictor_names.append(LIQUID_EFF_RADIUS_NAME)
    if not found_ice_eff_radius:
        vector_predictor_names.append(ICE_EFF_RADIUS_NAME)

    liquid_eff_radius_index = vector_predictor_names.index(
        LIQUID_EFF_RADIUS_NAME
    )
    ice_eff_radius_index = vector_predictor_names.index(ICE_EFF_RADIUS_NAME)
    example_dict[VECTOR_PREDICTOR_NAMES_KEY] = vector_predictor_names

    if found_liquid_eff_radius:
        example_dict[VECTOR_PREDICTOR_VALS_KEY][
            ..., liquid_eff_radius_index
        ] = liquid_eff_radius_matrix_metres
    else:
        example_dict[VECTOR_PREDICTOR_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_PREDICTOR_VALS_KEY],
            obj=liquid_eff_radius_index, values=liquid_eff_radius_matrix_metres,
            axis=-1
        )

    if found_ice_eff_radius:
        example_dict[VECTOR_PREDICTOR_VALS_KEY][..., ice_eff_radius_index] = (
            ice_eff_radius_matrix_metres
        )
    else:
        example_dict[VECTOR_PREDICTOR_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_PREDICTOR_VALS_KEY],
            obj=ice_eff_radius_index, values=ice_eff_radius_matrix_metres,
            axis=-1
        )

    return example_dict


def add_aerosols(example_dict):
    """Adds aerosol profiles to dictionary.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :return: example_dict: Same but with aerosol profiles.
    """

    num_examples = len(example_dict[EXAMPLE_IDS_KEY])
    num_heights = len(example_dict[HEIGHTS_KEY])

    these_dim = (num_examples, num_heights)
    optical_depth_matrix_metres = numpy.full(these_dim, 0.)
    albedo_matrix = numpy.full(these_dim, 1.)
    asymmetry_param_matrix = numpy.full(these_dim, 2.)

    vector_predictor_names = example_dict[VECTOR_PREDICTOR_NAMES_KEY]
    found_optical_depth = AEROSOL_OPTICAL_DEPTH_NAME in vector_predictor_names
    found_albedo = AEROSOL_ALBEDO_NAME in vector_predictor_names
    found_asymmetry_param = (
        AEROSOL_ASYMMETRY_PARAM_NAME in vector_predictor_names
    )

    if not found_optical_depth:
        vector_predictor_names.append(AEROSOL_OPTICAL_DEPTH_NAME)
    if not found_albedo:
        vector_predictor_names.append(AEROSOL_ALBEDO_NAME)
    if not found_asymmetry_param:
        vector_predictor_names.append(AEROSOL_ASYMMETRY_PARAM_NAME)

    optical_depth_index = vector_predictor_names.index(
        AEROSOL_OPTICAL_DEPTH_NAME
    )
    albedo_index = vector_predictor_names.index(AEROSOL_ALBEDO_NAME)
    asymmetry_param_index = vector_predictor_names.index(
        AEROSOL_ASYMMETRY_PARAM_NAME
    )
    example_dict[VECTOR_PREDICTOR_NAMES_KEY] = vector_predictor_names

    if found_optical_depth:
        example_dict[VECTOR_PREDICTOR_VALS_KEY][
            ..., optical_depth_index
        ] = optical_depth_matrix_metres
    else:
        example_dict[VECTOR_PREDICTOR_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_PREDICTOR_VALS_KEY],
            obj=optical_depth_index, values=optical_depth_matrix_metres,
            axis=-1
        )

    if found_albedo:
        example_dict[VECTOR_PREDICTOR_VALS_KEY][
            ..., albedo_index
        ] = albedo_matrix
    else:
        example_dict[VECTOR_PREDICTOR_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_PREDICTOR_VALS_KEY],
            obj=albedo_index, values=albedo_matrix, axis=-1
        )

    if found_asymmetry_param:
        example_dict[VECTOR_PREDICTOR_VALS_KEY][
            ..., asymmetry_param_index
        ] = asymmetry_param_matrix
    else:
        example_dict[VECTOR_PREDICTOR_VALS_KEY] = numpy.insert(
            example_dict[VECTOR_PREDICTOR_VALS_KEY],
            obj=asymmetry_param_index, values=asymmetry_param_matrix, axis=-1
        )

    return example_dict


def find_cloud_layers(example_dict, min_path_kg_m02, for_ice=False):
    """Finds liquid- or ice-cloud layers in each profile.

    E = number of examples
    H = number of heights

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :param min_path_kg_m02: Minimum path in each cloud layer (kg m^-2).
    :param for_ice: Boolean flag.  If True, will find ice clouds.  If False,
        will find liquid clouds.
    :return: cloud_mask_matrix: E-by-H numpy array of Boolean flags, indicating
        where clouds exist.
    :return: cloud_layer_counts: length-E numpy array with number of cloud
        layers for each example.
    """

    error_checking.assert_is_greater(min_path_kg_m02, 0.)
    error_checking.assert_is_boolean(for_ice)

    if for_ice:
        path_matrix_kg_m02 = get_field_from_dict(
            example_dict=example_dict, field_name=UPWARD_ICE_WATER_PATH_NAME
        )
    else:
        path_matrix_kg_m02 = get_field_from_dict(
            example_dict=example_dict, field_name=UPWARD_LIQUID_WATER_PATH_NAME
        )

    path_diff_matrix_kg_m02 = numpy.diff(path_matrix_kg_m02, axis=1, prepend=0.)

    num_examples = path_matrix_kg_m02.shape[0]
    cloud_mask_matrix = numpy.full(path_matrix_kg_m02.shape, False, dtype=bool)
    cloud_layer_counts = numpy.full(num_examples, 0, dtype=int)

    for i in range(num_examples):
        these_diffs = path_diff_matrix_kg_m02[i, :] + 0.

        if for_ice:
            these_diffs[these_diffs <= 1e9] = 0
        else:
            these_diffs[these_diffs <= 1e6] = 0

        these_start_indices, these_end_indices = _find_nonzero_runs(
            path_diff_matrix_kg_m02[i, :]
        )

        this_num_layers = len(these_start_indices)

        for j in range(this_num_layers):
            this_path_kg_m02 = numpy.sum(
                path_diff_matrix_kg_m02[
                    i, these_start_indices[j]:(these_end_indices[j] + 1)
                ]
            )

            if this_path_kg_m02 < min_path_kg_m02:
                continue

            cloud_layer_counts[i] += 1
            cloud_mask_matrix[
                i, these_start_indices[j]:(these_end_indices[j] + 1)
            ] = True

    return cloud_mask_matrix, cloud_layer_counts


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


def create_fake_heights(real_heights_m_agl, num_padding_heights):
    """Creates fake heights for padding at top of profile.

    :param real_heights_m_agl: 1-D numpy array of real heights (metres above
        ground level).
    :param num_padding_heights: Number of heights to pad at top.
    :return: heights_m_agl: 1-D numpy array with all heights (real followed by
        fake).
    """

    error_checking.assert_is_numpy_array(real_heights_m_agl, num_dimensions=1)
    error_checking.assert_is_geq_numpy_array(real_heights_m_agl, 0.)
    assert numpy.allclose(
        real_heights_m_agl, numpy.sort(real_heights_m_agl), atol=TOLERANCE
    )

    error_checking.assert_is_integer(num_padding_heights)
    error_checking.assert_is_geq(num_padding_heights, 0)

    if num_padding_heights == 0:
        return real_heights_m_agl

    fake_heights_m_agl = numpy.linspace(
        1, num_padding_heights, num=num_padding_heights, dtype=float
    )
    fake_heights_m_agl = real_heights_m_agl[-1] + 1e6 * fake_heights_m_agl

    return numpy.concatenate(
        (real_heights_m_agl, fake_heights_m_agl), axis=0
    )


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


def concat_examples(example_dicts):
    """Concatenates many dictionaries with examples into one.

    :param example_dicts: List of dictionaries, each in the format returned by
        `example_io.read_file`.
    :return: example_dict: Single dictionary, also in the format returned by
        `example_io.read_file`.
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

            if isinstance(example_dict[this_key], list):
                example_dict[this_key] += example_dicts[i][this_key]
            else:
                example_dict[this_key] = numpy.concatenate((
                    example_dict[this_key], example_dicts[i][this_key]
                ), axis=0)

    return example_dict


def create_example_ids(example_dict):
    """Creates example IDs (unique identifiers).

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
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
    albedos = get_field_from_dict(
        example_dict=example_dict, field_name=ALBEDO_NAME
    )
    valid_times_unix_sec = example_dict[VALID_TIMES_KEY]
    standard_atmo_flags = example_dict[STANDARD_ATMO_FLAGS_KEY]

    temperatures_10m_kelvins = get_field_from_dict(
        example_dict=example_dict, field_name=TEMPERATURE_NAME
    )[:, 0]

    return [
        'lat={0:09.6f}_long={1:010.6f}_zenith-angle-rad={2:08.6f}_' \
        'time={3:010d}_atmo={4:1d}_albedo={5:.6f}_' \
        'temp-10m-kelvins={6:010.6f}'.format(
            lat, long, theta, t, f, alpha, t10
        )
        for lat, long, theta, t, f, alpha, t10 in
        zip(
            latitudes_deg_n, longitudes_deg_e, zenith_angles_rad,
            valid_times_unix_sec, standard_atmo_flags, albedos,
            temperatures_10m_kelvins
        )
    ]


def get_dummy_example_id():
    """Creates dummy example ID.

    :return: dummy_id_string: Dummy example ID.
    """

    return (
        'lat={0:09.6f}_long={1:010.6f}_zenith-angle-rad={2:08.6f}_'
        'time={3:010d}_atmo={4:1d}_albedo={5:.6f}_temp-10m-kelvins={6:010.6f}'
    ).format(
        0, 0, 0, 0, 0, 0, 0
    )


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
    metadata_dict['albedos']: length-E numpy array of albedos (unitless).
    metadata_dict['temperatures_10m_kelvins']: length-E numpy array of
        temperatures at 10 m above ground level.
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
    albedos = numpy.full(num_examples, numpy.nan)
    temperatures_10m_kelvins = numpy.full(num_examples, numpy.nan)

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

        assert these_words[5].startswith('albedo=')
        albedos[i] = float(these_words[5].replace('albedo=', ''))

        assert these_words[6].startswith('temp-10m-kelvins=')
        temperatures_10m_kelvins[i] = float(
            these_words[6].replace('temp-10m-kelvins=', '')
        )

    return {
        LATITUDES_KEY: latitudes_deg_n,
        LONGITUDES_KEY: longitudes_deg_e,
        ZENITH_ANGLES_KEY: zenith_angles_rad,
        VALID_TIMES_KEY: valid_times_unix_sec,
        STANDARD_ATMO_FLAGS_KEY: standard_atmo_flags,
        ALBEDOS_KEY: albedos,
        TEMPERATURES_10M_KEY: temperatures_10m_kelvins
    }


def get_field_from_dict(example_dict, field_name, height_m_agl=None):
    """Returns field from dictionary of examples.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
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


def subset_by_time(example_dict, first_time_unix_sec, last_time_unix_sec):
    """Subsets examples by time.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
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
        if isinstance(example_dict[this_key], list):
            example_dict[this_key] = [
                example_dict[this_key][k] for k in good_indices
            ]
        else:
            example_dict[this_key] = (
                example_dict[this_key][good_indices, ...]
            )

    return example_dict, good_indices


def subset_by_standard_atmo(example_dict, standard_atmo_enum):
    """Subsets examples by standard-atmosphere type.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
    :param standard_atmo_enum: See doc for `check_standard_atmo_type`.
    :return: example_dict: Same as input but with fewer examples.
    :return: example_indices: 1-D numpy array with indices of examples kept.
    """

    check_standard_atmo_type(standard_atmo_enum)

    good_indices = numpy.where(
        example_dict[STANDARD_ATMO_FLAGS_KEY] == standard_atmo_enum,
    )[0]

    for this_key in ONE_PER_EXAMPLE_KEYS:
        if isinstance(example_dict[this_key], list):
            example_dict[this_key] = [
                example_dict[this_key][k] for k in good_indices
            ]
        else:
            example_dict[this_key] = (
                example_dict[this_key][good_indices, ...]
            )

    return example_dict, good_indices


def subset_by_field(example_dict, field_names):
    """Subsets examples by field.

    :param example_dict: Dictionary of examples (in the format returned by
        `example_io.read_file`).
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
        `example_io.read_file`).
    :param heights_m_agl: 1-D numpy array of heights to keep (metres above
        ground level).
    :return: example_dict: Same as input but with fewer heights.
    """

    example_dict = _add_height_padding(
        example_dict=example_dict, desired_heights_m_agl=heights_m_agl
    )

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
        `example_io.read_file`).
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
        if isinstance(example_dict[this_key], list):
            example_dict[this_key] = [
                example_dict[this_key][k] for k in good_indices
            ]
        else:
            example_dict[this_key] = (
                example_dict[this_key][good_indices, ...]
            )

    return example_dict, good_indices


def find_examples_with_time_tolerance(
        all_id_strings, desired_id_strings, time_tolerance_sec,
        allow_missing=False, verbose=True, allow_non_unique_matches=False):
    """Finds examples with desired IDs, but allowing for slight diffs in time.

    E = number of desired examples

    :param all_id_strings: See doc for `find_examples`.
    :param desired_id_strings: Same.
    :param time_tolerance_sec: Tolerance (seconds).
    :param allow_missing: See doc for `find_examples`.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :param allow_non_unique_matches: Boolean flag.  If True, will allow
        non-unique matches.
    :return: desired_indices: See doc for `find_examples`.
    :raises: ValueError: if either list of IDs has non-unique entries.
    :raises: ValueError: if `allow_missing == False` and any desired ID is
        missing.
    :raises: ValueError: if `allow_non_unique_matches == False` and there is a
        non-unique match.
   """

    # Check input args.
    error_checking.assert_is_string_list(all_id_strings)
    error_checking.assert_is_string_list(desired_id_strings)
    error_checking.assert_is_integer(time_tolerance_sec)
    error_checking.assert_is_greater(time_tolerance_sec, 0)
    error_checking.assert_is_boolean(allow_missing)
    error_checking.assert_is_boolean(verbose)
    error_checking.assert_is_boolean(allow_non_unique_matches)

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

    # Do actual stuff.
    all_metadata_dict = parse_example_ids(all_id_strings)
    all_times_unix_sec = all_metadata_dict[VALID_TIMES_KEY]
    all_id_strings_no_time = [
        'lat={0:09.6f}_long={1:010.6f}_zenith-angle-rad=1.000000_' \
        'time=0000000000_atmo={2:1d}_albedo={3:.6f}_' \
        'temp-10m-kelvins={4:010.6f}'.format(
            lat, long, f, alpha, t10
        )
        for lat, long, f, alpha, t10 in
        zip(
            all_metadata_dict[LATITUDES_KEY],
            all_metadata_dict[LONGITUDES_KEY],
            all_metadata_dict[STANDARD_ATMO_FLAGS_KEY],
            all_metadata_dict[ALBEDOS_KEY],
            all_metadata_dict[TEMPERATURES_10M_KEY]
        )
    ]
    all_id_strings_no_time_numpy = numpy.array(all_id_strings_no_time)

    desired_metadata_dict = parse_example_ids(desired_id_strings)
    desired_times_unix_sec = desired_metadata_dict[VALID_TIMES_KEY]
    desired_id_strings_no_time = [
        'lat={0:09.6f}_long={1:010.6f}_zenith-angle-rad=1.000000_' \
        'time=0000000000_atmo={2:1d}_albedo={3:.6f}_' \
        'temp-10m-kelvins={4:010.6f}'.format(
            lat, long, f, alpha, t10
        )
        for lat, long, f, alpha, t10 in
        zip(
            desired_metadata_dict[LATITUDES_KEY],
            desired_metadata_dict[LONGITUDES_KEY],
            desired_metadata_dict[STANDARD_ATMO_FLAGS_KEY],
            desired_metadata_dict[ALBEDOS_KEY],
            desired_metadata_dict[TEMPERATURES_10M_KEY]
        )
    ]
    desired_id_strings_no_time_numpy = numpy.array(desired_id_strings_no_time)

    num_desired_examples = len(desired_id_strings)
    desired_indices = numpy.full(num_desired_examples, -1, dtype=int)

    for i in range(num_desired_examples):
        if verbose and numpy.mod(i, 100) == 0:
            print((
                'Have looked for {0:d} of {1:d} examples with time tolerance '
                '= {2:d} s...'
            ).format(
                i, num_desired_examples, time_tolerance_sec
            ))

        these_time_diffs_sec = numpy.absolute(
            all_times_unix_sec - desired_times_unix_sec[i]
        )
        these_indices = numpy.where(
            these_time_diffs_sec <= time_tolerance_sec
        )[0]

        if len(these_indices) == 0:
            continue

        these_subindices = numpy.where(
            all_id_strings_no_time_numpy[these_indices] ==
            desired_id_strings_no_time[i]
        )[0]

        if len(these_subindices) == 0:
            continue

        these_indices = these_indices[these_subindices]

        if len(these_subindices) == 1:
            desired_indices[i] = these_indices[0]
        else:
            these_id_strings = [
                all_id_strings[k] for k in these_indices[these_subindices]
            ]
            error_string = (
                'Found multiple matches for desired ID "{0:s}":\n{1:s}'
            ).format(desired_id_strings[i], str(these_id_strings))

            if allow_non_unique_matches:
                warning_string = 'POTENTIAL ERROR: {0:s}'.format(error_string)
                warnings.warn(warning_string)
            else:
                raise ValueError(error_string)

            sort_indices = numpy.argsort(these_time_diffs_sec[these_indices])
            desired_indices[i] = these_indices[sort_indices[0]]

    if verbose:
        print((
            'Have looked for all {0:d} examples with time tolerance = {1:d} s!'
        ).format(
            num_desired_examples, time_tolerance_sec
        ))

    if allow_missing:
        return desired_indices

    if numpy.array_equal(
            all_id_strings_no_time_numpy[desired_indices],
            desired_id_strings_no_time_numpy
    ):
        return desired_indices

    missing_flags = (
        all_id_strings_no_time_numpy[desired_indices] !=
        desired_id_strings_no_time_numpy
    )

    error_string = (
        '{0:d} of {1:d} desired IDs (listed below) are missing:\n{2:s}'
    ).format(
        numpy.sum(missing_flags), len(desired_id_strings),
        str(desired_id_strings_numpy[missing_flags])
    )

    raise ValueError(error_string)


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

    desired_indices = numpy.maximum(desired_indices, 0)
    desired_indices = numpy.minimum(desired_indices, len(all_id_strings) - 1)
    desired_indices = sort_indices[desired_indices]

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

    :param example_dict: See doc for `example_io.read_file`.
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
        if isinstance(example_dict[this_key], list):
            example_dict[this_key] = [
                example_dict[this_key][k] for k in desired_indices
            ]
        else:
            example_dict[this_key] = (
                example_dict[this_key][desired_indices, ...]
            )

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

    :param example_dict: See doc for `example_io.read_file`.
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
