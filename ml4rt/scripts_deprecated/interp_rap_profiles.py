"""Interpolates RAP profiles (RRTM input) to a different height grid."""

import os
import argparse
import numpy
import xarray
from scipy.integrate import simps
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import moisture_conversions
from gewittergefahr.gg_utils import temperature_conversions as temp_conversions
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

SMALL_NUMBER = 1e-14
SENTINEL_VALUE = -999.

KG_TO_GRAMS = 1000.
GRAMS_TO_KG = 0.001
MB_TO_PASCALS = 100.
DRY_AIR_GAS_CONSTANT_J_KG01_K01 = 287.04

HEIGHT_DIMENSION = 'height_bin'

HEIGHT_KEY = 'height'
NN_PRESSURE_KEY = 'p0'
BILINTERP_PRESSURE_KEY = 'p1'
SPECIAL_KEYS = [HEIGHT_KEY, NN_PRESSURE_KEY, BILINTERP_PRESSURE_KEY]

NN_ICE_WATER_PATH_KEY = 'iwp0'
BILINTERP_ICE_WATER_PATH_KEY = 'iwp1'
NN_ICE_WATER_CONTENT_KEY = 'iwc0'
BILINTERP_ICE_WATER_CONTENT_KEY = 'iwc1'
NN_MIXING_LENGTH_KEY = 'mxlen0'
BILINTERP_MIXING_LENGTH_KEY = 'mxlen1'
NAN_ALLOWED_KEYS = [
    NN_ICE_WATER_PATH_KEY, BILINTERP_ICE_WATER_PATH_KEY,
    NN_ICE_WATER_CONTENT_KEY, BILINTERP_ICE_WATER_CONTENT_KEY,
    NN_MIXING_LENGTH_KEY, BILINTERP_MIXING_LENGTH_KEY
]

NN_SPECIFIC_HUMIDITY_KEY = 'qv0'
BILINTERP_SPECIFIC_HUMIDITY_KEY = 'qv1'
NN_VIRTUAL_TEMP_KEY = 'tv0'
BILINTERP_VIRTUAL_TEMP_KEY = 'tv1'
NN_WIND_SPEED_KEY = 'wspd0'
BILINTERP_WIND_SPEED_KEY = 'wspd1'
DEPENDENT_KEYS = [
    NN_SPECIFIC_HUMIDITY_KEY, BILINTERP_SPECIFIC_HUMIDITY_KEY,
    NN_VIRTUAL_TEMP_KEY, BILINTERP_VIRTUAL_TEMP_KEY,
    NN_WIND_SPEED_KEY, BILINTERP_WIND_SPEED_KEY
]

NN_MIXING_RATIO_KEY = 'r0'
BILINTERP_MIXING_RATIO_KEY = 'r1'
NN_U_WIND_KEY = 'u0'
BILINTERP_U_WIND_KEY = 'u1'
NN_V_WIND_KEY = 'v0'
BILINTERP_V_WIND_KEY = 'v1'
NN_TEMPERATURE_KEY = 't0'
BILINTERP_TEMPERATURE_KEY = 't1'
NN_LIQUID_WATER_CONTENT_KEY = 'lwc0'
BILINTERP_LIQUID_WATER_CONTENT_KEY = 'lwc1'

INPUT_FILES_ARG_NAME = 'input_rap_file_names'
NEW_HEIGHTS_ARG_NAME = 'new_heights_m_agl'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to input files, each containing RAP profiles on original '
    'height grid.  Each of these NetCDF files will be read by '
    '`xarray.open_dataset`.'
)
NEW_HEIGHTS_HELP_STRING = 'Heights (metres above ground level) in new grid.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  New files, each containing RAP profiles on new '
    'height grid, will be written here by the `to_netcdf` method for an xarray '
    'table.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEW_HEIGHTS_ARG_NAME, type=int, nargs='+', required=True,
    help=NEW_HEIGHTS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _conserve_mass_one_variable(
        orig_conc_matrix_kg_kg01, orig_air_dens_matrix_kg_m03,
        orig_heights_metres, new_conc_matrix_kg_kg01,
        new_air_dens_matrix_kg_m03, new_heights_metres, test_mode=False):
    """Conserves the mass of one chemical species between two grids.

    h = number of heights in original grid
    H = number of heights in new grid

    :param orig_conc_matrix_kg_kg01: numpy array with concentrations of chemical
        species.  Last axis must have length h.
    :param orig_air_dens_matrix_kg_m03: numpy array of air densities.  Must have
        same shape as `orig_conc_matrix_kg_kg01`.
    :param orig_heights_metres: length-h numpy array of heights.
    :param new_conc_matrix_kg_kg01: numpy array with concentrations of chemical
        species.  Last axis must have length H.
    :param new_air_dens_matrix_kg_m03: numpy array of air densities.  Must have
        same shape as `new_conc_matrix_kg_kg01`.
    :param new_heights_metres: length-H numpy array of heights.
    :param test_mode: Leave this alone.
    :return: new_conc_matrix_kg_kg01: Same as input but conserving original mass
        of chemical species over the entire column.
    """

    orig_conc_matrix_kg_m03 = (
        orig_conc_matrix_kg_kg01 * orig_air_dens_matrix_kg_m03
    )

    if test_mode:
        num_axes = len(orig_conc_matrix_kg_m03.shape)
        orig_height_matrix_metres = orig_heights_metres + 0.

        for k in reversed(range(num_axes - 1)):
            orig_height_matrix_metres = numpy.expand_dims(
                orig_height_matrix_metres, axis=0
            )
            orig_height_matrix_metres = numpy.repeat(
                orig_height_matrix_metres, axis=0,
                repeats=orig_conc_matrix_kg_m03.shape[k]
            )

        orig_mass_matrix_kg_m02 = numpy.sum(
            orig_conc_matrix_kg_m03 * orig_height_matrix_metres, axis=-1
        )
    else:
        orig_mass_matrix_kg_m02 = simps(
            y=orig_conc_matrix_kg_m03, x=orig_heights_metres,
            axis=-1, even='avg'
        )

    new_conc_matrix_kg_m03 = (
        new_conc_matrix_kg_kg01 * new_air_dens_matrix_kg_m03
    )

    if test_mode:
        num_axes = len(new_conc_matrix_kg_m03.shape)
        new_height_matrix_metres = new_heights_metres + 0.

        for k in reversed(range(num_axes - 1)):
            new_height_matrix_metres = numpy.expand_dims(
                new_height_matrix_metres, axis=0
            )
            new_height_matrix_metres = numpy.repeat(
                new_height_matrix_metres, axis=0,
                repeats=new_conc_matrix_kg_m03.shape[k]
            )

        new_mass_matrix_kg_m02 = numpy.sum(
            new_conc_matrix_kg_m03 * new_height_matrix_metres, axis=-1
        )
    else:
        new_mass_matrix_kg_m02 = simps(
            y=new_conc_matrix_kg_m03, x=new_heights_metres, axis=-1, even='avg'
        )

    new_mass_matrix_kg_m02[new_mass_matrix_kg_m02 < SMALL_NUMBER] = numpy.nan
    mass_ratio_matrix = orig_mass_matrix_kg_m02 / new_mass_matrix_kg_m02
    mass_ratio_matrix[numpy.isnan(mass_ratio_matrix)] = 0.

    mass_ratio_matrix = numpy.expand_dims(mass_ratio_matrix, axis=-1)
    mass_ratio_matrix = numpy.repeat(
        mass_ratio_matrix, axis=-1, repeats=new_conc_matrix_kg_m03.shape[-1]
    )
    new_conc_matrix_kg_m03 = new_conc_matrix_kg_m03 * mass_ratio_matrix

    return new_conc_matrix_kg_m03 / new_air_dens_matrix_kg_m03


def _conserve_masses(orig_data_dict, new_data_dict):
    """Conserves masses of various chemical species between the two grids.

    :param orig_data_dict: Dictionary with data on original grid.
    :param new_data_dict: Dictionary with data on new grid.
    :return: new_data_dict: Same but with masses conserved.
    """

    # Compute nearest-neigh-interpolated air density on original grid.
    orig_nn_air_dens_matrix_kg_m03 = (
        MB_TO_PASCALS *
        numpy.array(orig_data_dict[NN_PRESSURE_KEY]['data']) /
        numpy.array(orig_data_dict[NN_VIRTUAL_TEMP_KEY]['data'])
    )
    orig_nn_air_dens_matrix_kg_m03 = (
        orig_nn_air_dens_matrix_kg_m03 / DRY_AIR_GAS_CONSTANT_J_KG01_K01
    )

    # Compute nearest-neigh-interpolated air density on new grid.
    new_nn_air_dens_matrix_kg_m03 = (
        MB_TO_PASCALS *
        new_data_dict[NN_PRESSURE_KEY][1] /
        new_data_dict[NN_VIRTUAL_TEMP_KEY][1]
    )
    new_nn_air_dens_matrix_kg_m03 = (
        new_nn_air_dens_matrix_kg_m03 / DRY_AIR_GAS_CONSTANT_J_KG01_K01
    )

    # Compute bilinearly interpolated air density on original grid.
    orig_bilinterp_air_dens_matrix_kg_m03 = (
        MB_TO_PASCALS *
        numpy.array(orig_data_dict[BILINTERP_PRESSURE_KEY]['data']) /
        numpy.array(orig_data_dict[BILINTERP_VIRTUAL_TEMP_KEY]['data'])
    )
    orig_bilinterp_air_dens_matrix_kg_m03 = (
        orig_bilinterp_air_dens_matrix_kg_m03 / DRY_AIR_GAS_CONSTANT_J_KG01_K01
    )

    # Compute bilinearly interpolated air density on new grid.
    new_bilinterp_air_dens_matrix_kg_m03 = (
        MB_TO_PASCALS *
        new_data_dict[BILINTERP_PRESSURE_KEY][1] /
        new_data_dict[BILINTERP_VIRTUAL_TEMP_KEY][1]
    )
    new_bilinterp_air_dens_matrix_kg_m03 = (
        new_bilinterp_air_dens_matrix_kg_m03 / DRY_AIR_GAS_CONSTANT_J_KG01_K01
    )

    # Converse masses.
    orig_heights_m_agl = numpy.array(orig_data_dict[HEIGHT_KEY]['data'])
    new_heights_m_agl = numpy.array(new_data_dict[HEIGHT_KEY][1])

    print('Conserving column integral of {0:s}...'.format(NN_MIXING_RATIO_KEY))
    orig_nn_mixing_ratio_matrix_kg_kg01 = GRAMS_TO_KG * numpy.array(
        orig_data_dict[NN_MIXING_RATIO_KEY]['data']
    )
    new_nn_mixing_ratio_matrix_kg_kg01 = (
        GRAMS_TO_KG * new_data_dict[NN_MIXING_RATIO_KEY][1]
    )

    new_nn_mixing_ratio_matrix_kg_kg01 = _conserve_mass_one_variable(
        orig_conc_matrix_kg_kg01=orig_nn_mixing_ratio_matrix_kg_kg01,
        orig_air_dens_matrix_kg_m03=orig_nn_air_dens_matrix_kg_m03,
        orig_heights_metres=orig_heights_m_agl,
        new_conc_matrix_kg_kg01=new_nn_mixing_ratio_matrix_kg_kg01,
        new_air_dens_matrix_kg_m03=new_nn_air_dens_matrix_kg_m03,
        new_heights_metres=new_heights_m_agl
    )
    new_data_dict[NN_MIXING_RATIO_KEY] = (
        orig_data_dict[NN_MIXING_RATIO_KEY]['dims'],
        KG_TO_GRAMS * new_nn_mixing_ratio_matrix_kg_kg01
    )

    print('Conserving column integral of {0:s}...'.format(
        BILINTERP_MIXING_RATIO_KEY
    ))
    orig_bilinterp_mixing_ratio_matrix_kg_kg01 = GRAMS_TO_KG * numpy.array(
        orig_data_dict[BILINTERP_MIXING_RATIO_KEY]['data']
    )
    new_bilinterp_mixing_ratio_matrix_kg_kg01 = (
        GRAMS_TO_KG * new_data_dict[BILINTERP_MIXING_RATIO_KEY][1]
    )

    new_bilinterp_mixing_ratio_matrix_kg_kg01 = _conserve_mass_one_variable(
        orig_conc_matrix_kg_kg01=orig_bilinterp_mixing_ratio_matrix_kg_kg01,
        orig_air_dens_matrix_kg_m03=orig_bilinterp_air_dens_matrix_kg_m03,
        orig_heights_metres=orig_heights_m_agl,
        new_conc_matrix_kg_kg01=new_bilinterp_mixing_ratio_matrix_kg_kg01,
        new_air_dens_matrix_kg_m03=new_bilinterp_air_dens_matrix_kg_m03,
        new_heights_metres=new_heights_m_agl
    )
    new_data_dict[BILINTERP_MIXING_RATIO_KEY] = (
        orig_data_dict[BILINTERP_MIXING_RATIO_KEY]['dims'],
        KG_TO_GRAMS * new_bilinterp_mixing_ratio_matrix_kg_kg01
    )

    print('Conserving column integral of {0:s}...'.format(
        NN_LIQUID_WATER_CONTENT_KEY
    ))
    orig_nn_liquid_water_matrix_kg_kg01 = numpy.array(
        orig_data_dict[NN_LIQUID_WATER_CONTENT_KEY]['data']
    )
    new_nn_liquid_water_matrix_kg_kg01 = new_data_dict[NN_LIQUID_WATER_CONTENT_KEY][1]

    new_nn_liquid_water_matrix_kg_kg01 = _conserve_mass_one_variable(
        orig_conc_matrix_kg_kg01=orig_nn_liquid_water_matrix_kg_kg01,
        orig_air_dens_matrix_kg_m03=orig_nn_air_dens_matrix_kg_m03,
        orig_heights_metres=orig_heights_m_agl,
        new_conc_matrix_kg_kg01=new_nn_liquid_water_matrix_kg_kg01,
        new_air_dens_matrix_kg_m03=new_nn_air_dens_matrix_kg_m03,
        new_heights_metres=new_heights_m_agl
    )
    new_data_dict[NN_LIQUID_WATER_CONTENT_KEY] = (
        orig_data_dict[NN_LIQUID_WATER_CONTENT_KEY]['dims'],
        new_nn_liquid_water_matrix_kg_kg01
    )

    print('Conserving column integral of {0:s}...'.format(
        BILINTERP_LIQUID_WATER_CONTENT_KEY
    ))
    orig_bilinterp_liquid_water_matrix_kg_kg01 = numpy.array(
        orig_data_dict[BILINTERP_LIQUID_WATER_CONTENT_KEY]['data']
    )
    new_bilinterp_liquid_water_matrix_kg_kg01 = (
        new_data_dict[BILINTERP_LIQUID_WATER_CONTENT_KEY][1]
    )

    new_bilinterp_liquid_water_matrix_kg_kg01 = _conserve_mass_one_variable(
        orig_conc_matrix_kg_kg01=orig_bilinterp_liquid_water_matrix_kg_kg01,
        orig_air_dens_matrix_kg_m03=orig_bilinterp_air_dens_matrix_kg_m03,
        orig_heights_metres=orig_heights_m_agl,
        new_conc_matrix_kg_kg01=new_bilinterp_liquid_water_matrix_kg_kg01,
        new_air_dens_matrix_kg_m03=new_bilinterp_air_dens_matrix_kg_m03,
        new_heights_metres=new_heights_m_agl
    )
    new_data_dict[BILINTERP_LIQUID_WATER_CONTENT_KEY] = (
        orig_data_dict[BILINTERP_LIQUID_WATER_CONTENT_KEY]['dims'],
        new_bilinterp_liquid_water_matrix_kg_kg01
    )

    return new_data_dict


def _compute_dependent_vars(orig_data_dict, new_data_dict, verbose):
    """Computes dependent variables.

    :param orig_data_dict: Dictionary with data on original grid.
    :param new_data_dict: Dictionary with data on new grid.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: new_data_dict: Same but including dependent variables.
    """

    if verbose:
        print('Converting {0:s} to {1:s}...'.format(
            NN_MIXING_RATIO_KEY, NN_SPECIFIC_HUMIDITY_KEY
        ))

    new_data_dict[NN_MIXING_RATIO_KEY] = (
        orig_data_dict[NN_MIXING_RATIO_KEY]['dims'],
        numpy.maximum(new_data_dict[NN_MIXING_RATIO_KEY][1], 0.)
    )
    new_data_dict[BILINTERP_MIXING_RATIO_KEY] = (
        orig_data_dict[BILINTERP_MIXING_RATIO_KEY]['dims'],
        numpy.maximum(new_data_dict[BILINTERP_MIXING_RATIO_KEY][1], 0.)
    )

    new_nn_spec_humid_matrix_kg_kg01 = (
        moisture_conversions.mixing_ratio_to_specific_humidity(
            new_data_dict[NN_MIXING_RATIO_KEY][1] * GRAMS_TO_KG
        )
    )
    new_data_dict[NN_SPECIFIC_HUMIDITY_KEY] = (
        orig_data_dict[NN_SPECIFIC_HUMIDITY_KEY]['dims'],
        KG_TO_GRAMS * new_nn_spec_humid_matrix_kg_kg01
    )

    if verbose:
        print('Converting {0:s} to {1:s}...'.format(
            BILINTERP_MIXING_RATIO_KEY, BILINTERP_SPECIFIC_HUMIDITY_KEY
        ))

    new_bilinterp_spec_humid_matrix_kg_kg01 = (
        moisture_conversions.mixing_ratio_to_specific_humidity(
            new_data_dict[BILINTERP_MIXING_RATIO_KEY][1] * GRAMS_TO_KG
        )
    )
    new_data_dict[BILINTERP_SPECIFIC_HUMIDITY_KEY] = (
        orig_data_dict[BILINTERP_SPECIFIC_HUMIDITY_KEY]['dims'],
        KG_TO_GRAMS * new_bilinterp_spec_humid_matrix_kg_kg01
    )

    if verbose:
        print('Converting {0:s} and {1:s} to {2:s}...'.format(
            NN_U_WIND_KEY, NN_V_WIND_KEY, NN_WIND_SPEED_KEY
        ))

    new_nn_wind_speed_matrix_m_s01 = numpy.sqrt(
        new_data_dict[NN_U_WIND_KEY][1] ** 2 +
        new_data_dict[NN_V_WIND_KEY][1] ** 2
    )
    new_data_dict[NN_WIND_SPEED_KEY] = (
        orig_data_dict[NN_WIND_SPEED_KEY]['dims'],
        new_nn_wind_speed_matrix_m_s01
    )

    if verbose:
        print('Converting {0:s} and {1:s} to {2:s}...'.format(
            BILINTERP_U_WIND_KEY, BILINTERP_V_WIND_KEY, BILINTERP_WIND_SPEED_KEY
        ))

    new_bilinterp_wind_speed_matrix_m_s01 = numpy.sqrt(
        new_data_dict[BILINTERP_U_WIND_KEY][1] ** 2 +
        new_data_dict[BILINTERP_V_WIND_KEY][1] ** 2
    )
    new_data_dict[BILINTERP_WIND_SPEED_KEY] = (
        orig_data_dict[BILINTERP_WIND_SPEED_KEY]['dims'],
        new_bilinterp_wind_speed_matrix_m_s01
    )

    if verbose:
        print('Converting {0:s}, {1:s}, and {2:s} to {3:s}...'.format(
            NN_TEMPERATURE_KEY, NN_MIXING_RATIO_KEY, NN_PRESSURE_KEY,
            NN_VIRTUAL_TEMP_KEY
        ))

    new_nn_vapour_pressure_matrix_pa = (
        moisture_conversions.mixing_ratio_to_vapour_pressure(
            mixing_ratios_kg_kg01=
            new_data_dict[NN_MIXING_RATIO_KEY][1] * GRAMS_TO_KG,
            total_pressures_pascals=
            new_data_dict[NN_PRESSURE_KEY][1] * MB_TO_PASCALS
        )
    )
    new_nn_virtual_temp_matrix_kelvins = (
        moisture_conversions.temperature_to_virtual_temperature(
            temperatures_kelvins=temp_conversions.celsius_to_kelvins(
                new_data_dict[NN_TEMPERATURE_KEY][1]
            ),
            total_pressures_pascals=
            new_data_dict[NN_PRESSURE_KEY][1] * MB_TO_PASCALS,
            vapour_pressures_pascals=new_nn_vapour_pressure_matrix_pa
        )
    )
    new_data_dict[NN_VIRTUAL_TEMP_KEY] = (
        orig_data_dict[NN_VIRTUAL_TEMP_KEY]['dims'],
        new_nn_virtual_temp_matrix_kelvins
    )

    if verbose:
        print('Converting {0:s}, {1:s}, and {2:s} to {3:s}...'.format(
            BILINTERP_TEMPERATURE_KEY, BILINTERP_MIXING_RATIO_KEY,
            BILINTERP_PRESSURE_KEY,
            BILINTERP_VIRTUAL_TEMP_KEY
        ))

    new_bilinterp_vapour_pressure_matrix_pa = (
        moisture_conversions.mixing_ratio_to_vapour_pressure(
            mixing_ratios_kg_kg01=
            new_data_dict[BILINTERP_MIXING_RATIO_KEY][1] * GRAMS_TO_KG,
            total_pressures_pascals=
            new_data_dict[BILINTERP_PRESSURE_KEY][1] * MB_TO_PASCALS
        )
    )
    new_bilinterp_virtual_temp_matrix_kelvins = (
        moisture_conversions.temperature_to_virtual_temperature(
            temperatures_kelvins=temp_conversions.celsius_to_kelvins(
                new_data_dict[BILINTERP_TEMPERATURE_KEY][1]
            ),
            total_pressures_pascals=
            new_data_dict[BILINTERP_PRESSURE_KEY][1] * MB_TO_PASCALS,
            vapour_pressures_pascals=new_bilinterp_vapour_pressure_matrix_pa
        )
    )
    new_data_dict[BILINTERP_VIRTUAL_TEMP_KEY] = (
        orig_data_dict[BILINTERP_VIRTUAL_TEMP_KEY]['dims'],
        new_bilinterp_virtual_temp_matrix_kelvins
    )

    return new_data_dict


def _find_jumps(orig_data_matrix, orig_heights_metres, new_heights_metres):
    """Finds jumps in each profile for one variable.

    A "jump" is any two adjacent heights with a zero value and non-zero value.

    h = number of heights in original grid
    H = number of heights in new grid

    :param orig_data_matrix: numpy array with original data.  Last axis must
        have length h.
    :param orig_heights_metres: length-h numpy array of heights.
    :param new_heights_metres: length-H numpy array of heights.
    :return: jump_flag_matrix: numpy array of Boolean flags with same shape as
        `orig_data_matrix`, except that last axis has length H.
    """

    zero_flag_matrix = (orig_data_matrix < SMALL_NUMBER).astype(int)
    jump_flag_matrix = numpy.full(orig_data_matrix.shape, 0, dtype=int)

    forward_diff_matrix = numpy.absolute(numpy.diff(zero_flag_matrix, axis=-1))
    jump_flag_matrix[..., :-1] = (
        jump_flag_matrix[..., :-1] + forward_diff_matrix
    )

    backwards_diff_matrix = numpy.absolute(numpy.diff(
        numpy.flip(zero_flag_matrix, axis=-1), axis=-1
    ))
    backwards_diff_matrix = numpy.flip(backwards_diff_matrix, axis=-1)
    jump_flag_matrix[..., 1:] = (
        jump_flag_matrix[..., 1:] + backwards_diff_matrix
    )

    jump_flag_matrix = jump_flag_matrix > 0

    new_to_orig_height_indices = numpy.array([
        numpy.argmin(numpy.absolute(h - orig_heights_metres))
        for h in new_heights_metres
    ], dtype=int)

    return jump_flag_matrix[..., new_to_orig_height_indices]


def _interp_and_conserve_jumps(orig_data_matrix, orig_heights_metres,
                               new_heights_metres, extrapolate=True):
    """Applies jump-conserving interpolation to one variable.

    A "jump" is any two adjacent heights with a zero value and non-zero value.

    :param orig_data_matrix: See doc for `_find_jumps`.
    :param orig_heights_metres: Same.
    :param new_heights_metres: Same.
    :param extrapolate: Boolean flag.  If True, will extrapolate beyond original
        heights.
    """

    jump_flag_matrix = _find_jumps(
        orig_data_matrix=orig_data_matrix,
        orig_heights_metres=orig_heights_metres,
        new_heights_metres=new_heights_metres
    )

    # log_offset = 1. + -1 * numpy.min(orig_data_matrix)
    # assert not numpy.isnan(log_offset)
    #
    # if extrapolate:
    #     interp_object = interp1d(
    #         x=orig_heights_metres, y=numpy.log(log_offset + orig_data_matrix),
    #         axis=-1, kind='linear', bounds_error=False, assume_sorted=True,
    #         fill_value='extrapolate'
    #     )
    # else:
    #     bottom_value_matrix = numpy.log(log_offset + orig_data_matrix[..., 0])
    #     top_value_matrix = numpy.log(log_offset + orig_data_matrix[..., -1])
    #
    #     interp_object = interp1d(
    #         x=orig_heights_metres, y=numpy.log(log_offset + orig_data_matrix),
    #         axis=-1, kind='linear', bounds_error=False, assume_sorted=True,
    #         fill_value=(bottom_value_matrix, top_value_matrix)
    #     )
    #
    # new_data_matrix = numpy.exp(interp_object(new_heights_metres)) - log_offset

    if extrapolate:
        interp_object = interp1d(
            x=orig_heights_metres, y=orig_data_matrix,
            axis=-1, kind='linear', bounds_error=False, assume_sorted=True,
            fill_value='extrapolate'
        )
    else:
        interp_object = interp1d(
            x=orig_heights_metres, y=orig_data_matrix,
            axis=-1, kind='linear', bounds_error=False, assume_sorted=True,
            fill_value=(orig_data_matrix[..., 0], orig_data_matrix[..., -1])
        )

    new_data_matrix = interp_object(new_heights_metres)

    # if extrapolate:
    #     interp_object = interp1d(
    #         x=orig_heights_metres, y=numpy.log(log_offset + orig_data_matrix),
    #         axis=-1, kind='nearest', bounds_error=False, assume_sorted=True,
    #         fill_value='extrapolate'
    #     )
    # else:
    #     bottom_value_matrix = numpy.log(log_offset + orig_data_matrix[..., 0])
    #     top_value_matrix = numpy.log(log_offset + orig_data_matrix[..., -1])
    #
    #     interp_object = interp1d(
    #         x=orig_heights_metres, y=numpy.log(log_offset + orig_data_matrix),
    #         axis=-1, kind='nearest', bounds_error=False, assume_sorted=True,
    #         fill_value=(bottom_value_matrix, top_value_matrix)
    #     )
    #
    # new_data_matrix_nn = (
    #     numpy.exp(interp_object(new_heights_metres)) - log_offset
    # )

    if extrapolate:
        interp_object = interp1d(
            x=orig_heights_metres, y=orig_data_matrix,
            axis=-1, kind='nearest', bounds_error=False, assume_sorted=True,
            fill_value='extrapolate'
        )
    else:
        interp_object = interp1d(
            x=orig_heights_metres, y=orig_data_matrix,
            axis=-1, kind='nearest', bounds_error=False, assume_sorted=True,
            fill_value=(orig_data_matrix[..., 0], orig_data_matrix[..., -1])
        )

    new_data_matrix_nn = interp_object(new_heights_metres)

    new_data_matrix[jump_flag_matrix] = new_data_matrix_nn[jump_flag_matrix]
    return new_data_matrix


def _interp_rap_profiles_one_day(
        input_file_name, new_heights_m_agl, output_file_name):
    """Interpolates RAP profiles for one day to a different height grid.

    :param input_file_name: Path to input file.  This NetCDF file will be read
        by `xarray.open_dataset`.
    :param new_heights_m_agl: See documentation at top of file.
    :param output_file_name: Path to output file.  New file will be written here
        by the `to_netcdf` method for an xarray table.
    """

    num_new_heights = len(new_heights_m_agl)

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    orig_table_xarray = xarray.open_dataset(input_file_name)

    # Create metadata for output file (on new height grid).
    orig_metadata_dict = orig_table_xarray.to_dict()['dims']
    new_metadata_dict = dict()

    for this_key in orig_metadata_dict:
        if this_key == HEIGHT_DIMENSION:
            new_metadata_dict[this_key] = numpy.linspace(
                0, num_new_heights - 1, num=num_new_heights, dtype=int
            )
            continue

        this_axis_length = orig_metadata_dict[this_key]
        new_metadata_dict[this_key] = numpy.linspace(
            0, this_axis_length - 1, num=this_axis_length, dtype=int
        )

    # Create actual data for output file (on new height grid).
    orig_data_dict = orig_table_xarray.to_dict()['data_vars']
    new_data_dict = dict()

    for this_key in orig_table_xarray.variables:
        if HEIGHT_DIMENSION not in orig_table_xarray[this_key].dims:
            new_data_dict[this_key] = (
                orig_data_dict[this_key]['dims'],
                orig_data_dict[this_key]['data']
            )
            continue

        if this_key not in SPECIAL_KEYS:
            continue

        if this_key == HEIGHT_KEY:
            new_data_matrix = new_heights_m_agl + 0.
        else:
            print('Interpolating {0:s} to new heights...'.format(this_key))

            # interp_object = interp1d(
            #     x=numpy.array(orig_data_dict[HEIGHT_KEY]['data']),
            #     y=numpy.log(numpy.array(orig_data_dict[this_key]['data'])),
            #     axis=-1, kind='linear', bounds_error=True, assume_sorted=True
            # )
            # new_data_matrix = numpy.exp(interp_object(new_heights_m_agl))

            interp_object = interp1d(
                x=numpy.array(orig_data_dict[HEIGHT_KEY]['data']),
                y=numpy.array(orig_data_dict[this_key]['data']),
                axis=-1, kind='linear', bounds_error=True, assume_sorted=True
            )
            new_data_matrix = interp_object(new_heights_m_agl)

        new_data_dict[this_key] = (
            orig_data_dict[this_key]['dims'],
            new_data_matrix
        )

    for this_key in orig_table_xarray.variables:
        if HEIGHT_DIMENSION not in orig_table_xarray[this_key].dims:
            continue
        if this_key in SPECIAL_KEYS:
            continue
        if this_key in DEPENDENT_KEYS:
            continue

        orig_data_matrix = numpy.array(orig_data_dict[this_key]['data'])

        # TODO(thunderhoser): HACK!
        if this_key in NAN_ALLOWED_KEYS:
            orig_data_matrix[:] = SENTINEL_VALUE

        all_nan_flag = numpy.all(orig_data_matrix < (SENTINEL_VALUE + 1))
        all_valid_flag = numpy.all(orig_data_matrix >= (SENTINEL_VALUE + 1))

        if this_key in NAN_ALLOWED_KEYS:
            assert all_nan_flag
        else:
            assert all_valid_flag

        if this_key in [
                NN_LIQUID_WATER_CONTENT_KEY, BILINTERP_LIQUID_WATER_CONTENT_KEY
        ]:
            print((
                'Interpolating {0:s} to new heights with jump conservation...'
            ).format(
                this_key
            ))

            new_data_matrix = _interp_and_conserve_jumps(
                orig_data_matrix=orig_data_matrix,
                orig_heights_metres=
                numpy.array(orig_data_dict[HEIGHT_KEY]['data']),
                new_heights_metres=new_heights_m_agl
            )
        else:
            print('Interpolating {0:s} to new heights...'.format(this_key))

            # log_offset = 1. + -1 * numpy.min(orig_data_matrix)
            # assert not numpy.isnan(log_offset)
            #
            # interp_object = interp1d(
            #     x=numpy.array(orig_data_dict[HEIGHT_KEY]['data']),
            #     y=numpy.log(log_offset + orig_data_matrix),
            #     axis=-1, kind='linear', bounds_error=True, assume_sorted=True
            # )
            # new_data_matrix = (
            #     numpy.exp(interp_object(new_heights_m_agl)) - log_offset
            # )

            interp_object = interp1d(
                x=numpy.array(orig_data_dict[HEIGHT_KEY]['data']),
                y=orig_data_matrix,
                axis=-1, kind='linear', bounds_error=True, assume_sorted=True
            )
            new_data_matrix = interp_object(new_heights_m_agl)

        new_data_dict[this_key] = (
            orig_data_dict[this_key]['dims'],
            new_data_matrix
        )

    new_data_dict = _compute_dependent_vars(
        orig_data_dict=orig_data_dict, new_data_dict=new_data_dict,
        verbose=False
    )
    new_data_dict = _conserve_masses(
        orig_data_dict=orig_data_dict, new_data_dict=new_data_dict
    )
    new_data_dict = _compute_dependent_vars(
        orig_data_dict=orig_data_dict, new_data_dict=new_data_dict,
        verbose=True
    )
    new_table_xarray = xarray.Dataset(
        data_vars=new_data_dict, coords=new_metadata_dict,
        attrs=orig_table_xarray.attrs
    )

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    new_table_xarray.to_netcdf(
        path=output_file_name, mode='w', format='NETCDF3_64BIT'
    )


def _run(input_file_names, new_heights_m_agl, output_dir_name):
    """Interpolates RAP profiles (RRTM input) to a different height grid.

    This is effectively the main method.

    :param input_file_names: See documentation at top of file.
    :param new_heights_m_agl: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    # Check input args.
    error_checking.assert_is_geq_numpy_array(new_heights_m_agl, 0)
    error_checking.assert_is_geq_numpy_array(numpy.diff(new_heights_m_agl), 0)
    new_heights_m_agl = new_heights_m_agl.astype(float)

    for this_input_file_name in input_file_names:
        this_output_file_name = '{0:s}/{1:s}'.format(
            output_dir_name, os.path.split(this_input_file_name)[1]
        )

        _interp_rap_profiles_one_day(
            input_file_name=this_input_file_name,
            new_heights_m_agl=new_heights_m_agl,
            output_file_name=this_output_file_name
        )
        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        new_heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEW_HEIGHTS_ARG_NAME), dtype=int
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
