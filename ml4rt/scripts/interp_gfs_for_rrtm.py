"""Interpolates GFS data, meant for input to RRTM, to new heights."""

import argparse
import numpy
import xarray
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import moisture_conversions as moisture_conv
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import rrtm_io
from ml4rt.utils import example_utils
from ml4rt.scripts import interp_rap_profiles

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

DRY_AIR_GAS_CONSTANT_J_KG01_K01 = 287.04

MOLAR_MASS_DRY_AIR_GRAMS_MOL01 = 28.97
MOLAR_MASS_O2_GRAMS_MOL01 = 31.9988
MOLAR_MASS_CO2_GRAMS_MOL01 = 44.01
MOLAR_MASS_CH4_GRAMS_MOL01 = 16.04
MOLAR_MASS_N2O_GRAMS_MOL01 = 44.013

SITE_DIMENSION = 'sites'
TIME_DIMENSION = 'valid_time_unix_sec'
HEIGHT_DIMENSION = 'height_bin'
HEIGHT_AT_EDGE_DIMENSION = 'height_bin_edge'

SITE_NAME_KEY = 'site_name'
LATITUDE_KEY = 'site_latitude_deg_n'
LONGITUDE_KEY = 'site_longitude_deg_e'
FORECAST_HOUR_KEY = 'forecast_hour'
HEIGHT_KEY = 'height_m_agl'
PRESSURE_KEY = 'pressure_pascals'
PRESSURE_AT_EDGE_KEY = 'pressure_at_edge_pascals'
TEMPERATURE_KEY = 'temperature_kelvins'
VAPOUR_MIXING_RATIO_KEY = 'vapour_mixing_ratio_kg_kg01'
LIQUID_WATER_PATH_KEY = 'layerwise_liquid_water_path_kg_m02'
RAIN_WATER_PATH_KEY = 'layerwise_rain_water_path_kg_m02'
ICE_WATER_PATH_KEY = 'layerwise_ice_water_path_kg_m02'
GRAUPEL_PATH_KEY = 'layerwise_graupel_path_kg_m02'
SNOW_PATH_KEY = 'layerwise_snow_path_kg_m02'
OZONE_MIXING_RATIO_KEY = 'ozone_mixing_ratio_kg_kg01'
O2_CONCENTRATION_KEY = 'o2_concentration_ppmv'
CO2_CONCENTRATION_KEY = 'co2_concentration_ppmv'
CH4_CONCENTRATION_KEY = 'ch4_concentration_ppmv'
N2O_CONCENTRATION_KEY = 'n2o_concentration_ppmv'
LIQUID_EFF_RADIUS_KEY = 'liquid_eff_radius_metres'
ICE_EFF_RADIUS_KEY = 'ice_eff_radius_metres'
CLOUD_FRACTION_KEY = 'cloud_fraction_percent'
ALBEDO_KEY = 'surface_albedo'
AEROSOL_EXTINCTION_KEY = 'aerosol_extinction_metres01'
AEROSOL_ALBEDO_KEY = 'aerosol_albedo'
AEROSOL_ASYMMETRY_PARAM_KEY = 'aerosol_asymmetry_param'

LIQUID_WATER_MIXING_RATIO_KEY = 'liquid_water_mixing_ratio_kg_kg01'
RAIN_WATER_MIXING_RATIO_KEY = 'rain_water_mixing_ratio_kg_kg01'
ICE_WATER_MIXING_RATIO_KEY = 'ice_water_mixing_ratio_kg_kg01'
GRAUPEL_MIXING_RATIO_KEY = 'graupel_mixing_ratio_kg_kg01'
SNOW_MIXING_RATIO_KEY = 'snow_mixing_ratio_kg_kg01'
O2_MIXING_RATIO_KEY = 'o2_mixing_ratio_kg_kg01'
CO2_MIXING_RATIO_KEY = 'co2_mixing_ratio_kg_kg01'
CH4_MIXING_RATIO_KEY = 'ch4_mixing_ratio_kg_kg01'
N2O_MIXING_RATIO_KEY = 'n2o_mixing_ratio_kg_kg01'

INTERP_KEYS = [
    PRESSURE_KEY, PRESSURE_AT_EDGE_KEY, TEMPERATURE_KEY,
    VAPOUR_MIXING_RATIO_KEY, LIQUID_WATER_MIXING_RATIO_KEY,
    RAIN_WATER_MIXING_RATIO_KEY, ICE_WATER_MIXING_RATIO_KEY,
    GRAUPEL_MIXING_RATIO_KEY, SNOW_MIXING_RATIO_KEY, OZONE_MIXING_RATIO_KEY,
    O2_MIXING_RATIO_KEY, CO2_MIXING_RATIO_KEY, CH4_MIXING_RATIO_KEY,
    N2O_MIXING_RATIO_KEY, LIQUID_EFF_RADIUS_KEY, ICE_EFF_RADIUS_KEY,
    CLOUD_FRACTION_KEY, AEROSOL_EXTINCTION_KEY
]
CONSERVE_JUMP_KEYS = [
    LIQUID_WATER_MIXING_RATIO_KEY, RAIN_WATER_MIXING_RATIO_KEY,
    ICE_WATER_MIXING_RATIO_KEY, GRAUPEL_MIXING_RATIO_KEY, SNOW_MIXING_RATIO_KEY
]
CONSERVE_MASS_KEYS = [
    VAPOUR_MIXING_RATIO_KEY, LIQUID_WATER_MIXING_RATIO_KEY,
    RAIN_WATER_MIXING_RATIO_KEY, ICE_WATER_MIXING_RATIO_KEY,
    GRAUPEL_MIXING_RATIO_KEY, SNOW_MIXING_RATIO_KEY,
    OZONE_MIXING_RATIO_KEY, O2_MIXING_RATIO_KEY, CO2_MIXING_RATIO_KEY,
    CH4_MIXING_RATIO_KEY, N2O_MIXING_RATIO_KEY
]
SCALAR_KEYS = [
    SITE_NAME_KEY, LATITUDE_KEY, LONGITUDE_KEY, FORECAST_HOUR_KEY,
    ALBEDO_KEY, AEROSOL_ALBEDO_KEY, AEROSOL_ASYMMETRY_PARAM_KEY
]

INPUT_FILE_ARG_NAME = 'input_file_name'
NEW_HEIGHTS_ARG_NAME = 'new_heights_m_agl'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (created by prepare_gfs_for_rrtm_no_interp.py), '
    'containing GFS data for one init time.'
)
NEW_HEIGHTS_HELP_STRING = 'Heights (metres above ground level) in new grid.'
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Interpolated data (in the same format as the input) '
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

    :param orig_gfs_table_xarray: xarray table with original GFS data (before
        interp).
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

    orig_pressures_pa = orig_gfs_table_xarray[PRESSURE_KEY].values[i, j, :]
    orig_heights_m_agl = orig_gfs_table_xarray[HEIGHT_KEY].values[i, j, :]

    interp_object = interp1d(
        x=orig_heights_m_agl, y=numpy.log(orig_pressures_pa),
        kind='linear', bounds_error=False, assume_sorted=True,
        fill_value='extrapolate'
    )

    interp_data_dict[PRESSURE_KEY][i, j, :] = (
        numpy.exp(interp_object(new_heights_m_agl))
    )
    interp_data_dict[PRESSURE_AT_EDGE_KEY][i, j, :] = (
        numpy.exp(interp_object(new_heights_at_edges_m_agl))
    )

    assert numpy.all(interp_data_dict[PRESSURE_KEY][i, j, :] > 0)
    assert numpy.all(interp_data_dict[PRESSURE_AT_EDGE_KEY][i, j, :] > 0)

    for this_key in INTERP_KEYS:
        if this_key in [PRESSURE_KEY, PRESSURE_AT_EDGE_KEY]:
            continue

        if this_key in CONSERVE_JUMP_KEYS:
            orig_data_matrix = numpy.expand_dims(
                orig_gfs_table_xarray[this_key].values[i, j, :], axis=0
            )
            interp_data_dict[this_key][i, j, :] = (
                interp_rap_profiles._interp_and_conserve_jumps(
                    orig_data_matrix=orig_data_matrix,
                    orig_heights_metres=orig_heights_m_agl,
                    new_heights_metres=new_heights_m_agl,
                    extrapolate=False
                )
            )

            continue

        # orig_values = orig_gfs_table_xarray[this_key].values[i, j, :]
        # log_offset = 1. + -1 * numpy.min(orig_values)
        # assert not numpy.isnan(log_offset)
        #
        # bottom_value = numpy.log(log_offset + orig_values[0])
        # top_value = numpy.log(log_offset + orig_values[1])
        #
        # interp_object = interp1d(
        #     x=orig_heights_m_agl, y=numpy.log(log_offset + orig_values),
        #     kind='linear', bounds_error=False, assume_sorted=True,
        #     fill_value=(bottom_value, top_value)
        # )
        # interp_data_dict[this_key][i, j, :] = (
        #     numpy.exp(interp_object(new_heights_m_agl)) - log_offset
        # )

        orig_values = orig_gfs_table_xarray[this_key].values[i, j, :]
        bottom_value = orig_values[0]
        top_value = orig_values[1]

        interp_object = interp1d(
            x=orig_heights_m_agl, y=orig_values,
            kind='linear', bounds_error=False, assume_sorted=True,
            fill_value=(bottom_value, top_value)
        )
        interp_data_dict[this_key][i, j, :] = interp_object(new_heights_m_agl)

    return interp_data_dict


def _concentrations_to_mixing_ratios(orig_gfs_table_xarray):
    """Converts trace-gas concentrations (ppmv) to mixing ratios (kg kg^-1).

    :param orig_gfs_table_xarray: xarray table with original GFS data (before
        interp).
    :return: orig_gfs_table_xarray: Same as input but with additional keys for
        mixing ratios.
    """

    o2_mixing_ratio_matrix_kg_kg01 = (
        1e-6 * orig_gfs_table_xarray[O2_CONCENTRATION_KEY].values *
        MOLAR_MASS_O2_GRAMS_MOL01 / MOLAR_MASS_DRY_AIR_GRAMS_MOL01
    )
    co2_mixing_ratio_matrix_kg_kg01 = (
        1e-6 * orig_gfs_table_xarray[CO2_CONCENTRATION_KEY].values *
        MOLAR_MASS_CO2_GRAMS_MOL01 / MOLAR_MASS_DRY_AIR_GRAMS_MOL01
    )
    ch4_mixing_ratio_matrix_kg_kg01 = (
        1e-6 * orig_gfs_table_xarray[CH4_CONCENTRATION_KEY].values *
        MOLAR_MASS_CH4_GRAMS_MOL01 / MOLAR_MASS_DRY_AIR_GRAMS_MOL01
    )
    n2o_mixing_ratio_matrix_kg_kg01 = (
        1e-6 * orig_gfs_table_xarray[N2O_CONCENTRATION_KEY].values *
        MOLAR_MASS_N2O_GRAMS_MOL01 / MOLAR_MASS_DRY_AIR_GRAMS_MOL01
    )

    these_dim = orig_gfs_table_xarray[O2_CONCENTRATION_KEY].dims
    this_dict = {
        O2_MIXING_RATIO_KEY: (these_dim, o2_mixing_ratio_matrix_kg_kg01),
        CO2_MIXING_RATIO_KEY: (these_dim, co2_mixing_ratio_matrix_kg_kg01),
        CH4_MIXING_RATIO_KEY: (these_dim, ch4_mixing_ratio_matrix_kg_kg01),
        N2O_MIXING_RATIO_KEY: (these_dim, n2o_mixing_ratio_matrix_kg_kg01)
    }

    return orig_gfs_table_xarray.assign(this_dict)


def _mixing_ratios_to_concentrations(interp_data_dict):
    """Converts trace-gas mixing ratios (kg kg^-1) to concentrations (ppmv).

    :param interp_data_dict: Dictionary, where each key is a variable name and
        the corresponding value is a 3-D numpy array (time x site x new_height).
    :return: interp_data_dict: Same as input but with additional keys for
        concentrations.
    """

    interp_data_dict[O2_CONCENTRATION_KEY] = (
        1e6 * interp_data_dict[O2_MIXING_RATIO_KEY] *
        MOLAR_MASS_DRY_AIR_GRAMS_MOL01 / MOLAR_MASS_O2_GRAMS_MOL01
    )
    interp_data_dict[CO2_CONCENTRATION_KEY] = (
        1e6 * interp_data_dict[CO2_MIXING_RATIO_KEY] *
        MOLAR_MASS_DRY_AIR_GRAMS_MOL01 / MOLAR_MASS_CO2_GRAMS_MOL01
    )
    interp_data_dict[CH4_CONCENTRATION_KEY] = (
        1e6 * interp_data_dict[CH4_MIXING_RATIO_KEY] *
        MOLAR_MASS_DRY_AIR_GRAMS_MOL01 / MOLAR_MASS_CH4_GRAMS_MOL01
    )
    interp_data_dict[N2O_CONCENTRATION_KEY] = (
        1e6 * interp_data_dict[N2O_MIXING_RATIO_KEY] *
        MOLAR_MASS_DRY_AIR_GRAMS_MOL01 / MOLAR_MASS_N2O_GRAMS_MOL01
    )

    interp_data_dict.pop(O2_MIXING_RATIO_KEY)
    interp_data_dict.pop(CO2_MIXING_RATIO_KEY)
    interp_data_dict.pop(CH4_MIXING_RATIO_KEY)
    interp_data_dict.pop(N2O_MIXING_RATIO_KEY)

    return interp_data_dict


def _layerwise_path_to_mixing_ratio(orig_gfs_table_xarray, orig_variable_name,
                                    new_variable_name):
    """Converts layerwise paths (kg m^-2) to mixing ratios (kg kg^-1).

    :param orig_gfs_table_xarray: xarray table with original GFS data (before
        interp).
    :param orig_variable_name: Variable name for layerwise path.  This must be a
        key into `orig_gfs_table_xarray`.
    :param new_variable_name: Variable name for mixing ratio.  This will become
        a key in `orig_gfs_table_xarray`.
    :return: orig_gfs_table_xarray: Same as input but with an additional key for
        mixing ratios.
    """

    num_times = len(orig_gfs_table_xarray.coords[TIME_DIMENSION].values)
    num_sites = len(orig_gfs_table_xarray.coords[SITE_DIMENSION].values)
    num_heights = len(orig_gfs_table_xarray.coords[HEIGHT_DIMENSION].values)

    layerwise_path_matrix_kg_m02 = numpy.reshape(
        orig_gfs_table_xarray[orig_variable_name].values,
        (num_times * num_sites, num_heights)
    )
    height_matrix_m_agl = numpy.reshape(
        orig_gfs_table_xarray[HEIGHT_KEY].values,
        (num_times * num_sites, num_heights)
    )
    mixing_ratio_matrix_kg_m03 = numpy.full(
        layerwise_path_matrix_kg_m02.shape, numpy.nan
    )

    for i in range(num_times * num_sites):
        mixing_ratio_matrix_kg_m03[i, :] = (
            rrtm_io._layerwise_water_path_to_content(
                layerwise_path_matrix_kg_m02=
                layerwise_path_matrix_kg_m02[[i], :],
                heights_m_agl=height_matrix_m_agl[i, :]
            )[0, :]
        )

    vapour_pressure_matrix_pa = moisture_conv.mixing_ratio_to_vapour_pressure(
        mixing_ratios_kg_kg01=
        orig_gfs_table_xarray[VAPOUR_MIXING_RATIO_KEY].values,
        total_pressures_pascals=orig_gfs_table_xarray[PRESSURE_KEY].values
    )
    virtual_temp_matrix_kelvins = (
        moisture_conv.temperature_to_virtual_temperature(
            temperatures_kelvins=orig_gfs_table_xarray[TEMPERATURE_KEY].values,
            total_pressures_pascals=orig_gfs_table_xarray[PRESSURE_KEY].values,
            vapour_pressures_pascals=vapour_pressure_matrix_pa
        )
    )
    air_density_matrix_kg_m03 = (
        orig_gfs_table_xarray[PRESSURE_KEY].values / virtual_temp_matrix_kelvins
    )
    air_density_matrix_kg_m03 = (
        air_density_matrix_kg_m03 / DRY_AIR_GAS_CONSTANT_J_KG01_K01
    )

    mixing_ratio_matrix_kg_m03 = numpy.reshape(
        mixing_ratio_matrix_kg_m03,
        (num_times, num_sites, num_heights)
    )
    mixing_ratio_matrix_kg_kg01 = (
        mixing_ratio_matrix_kg_m03 / air_density_matrix_kg_m03
    )

    this_dict = {
        new_variable_name: (
            orig_gfs_table_xarray[orig_variable_name].dims,
            mixing_ratio_matrix_kg_kg01
        )
    }

    return orig_gfs_table_xarray.assign(this_dict)


def _mixing_ratio_to_layerwise_path(
        interp_data_dict, new_heights_m_agl, orig_variable_name,
        new_variable_name):
    """Converts mixing ratios (kg kg^-1) to layerwise paths (kg m^-2).

    :param interp_data_dict: Dictionary, where each key is a variable name and
        the corresponding value is a 3-D numpy array (time x site x new_height).
    :param new_heights_m_agl: 1-D numpy array of heights (metres above ground
        level) in new grid.
    :param orig_variable_name: Variable name for mixing ratio.  This must be a
        key into `interp_data_dict`.
    :param new_variable_name: Variable name for layerwise path.  This will
        become a key in `interp_data_dict`.
    :return: interp_data_dict: Same as input but with an additional key for
        layerwise paths.
    """

    vapour_pressure_matrix_pa = moisture_conv.mixing_ratio_to_vapour_pressure(
        mixing_ratios_kg_kg01=interp_data_dict[VAPOUR_MIXING_RATIO_KEY],
        total_pressures_pascals=interp_data_dict[PRESSURE_KEY]
    )
    virtual_temp_matrix_kelvins = (
        moisture_conv.temperature_to_virtual_temperature(
            temperatures_kelvins=interp_data_dict[TEMPERATURE_KEY],
            total_pressures_pascals=interp_data_dict[PRESSURE_KEY],
            vapour_pressures_pascals=vapour_pressure_matrix_pa
        )
    )
    air_density_matrix_kg_m03 = (
        interp_data_dict[PRESSURE_KEY] / virtual_temp_matrix_kelvins
    )
    air_density_matrix_kg_m03 = (
        air_density_matrix_kg_m03 / DRY_AIR_GAS_CONSTANT_J_KG01_K01
    )

    num_times = air_density_matrix_kg_m03.shape[0]
    num_sites = air_density_matrix_kg_m03.shape[1]
    num_heights = air_density_matrix_kg_m03.shape[2]

    mixing_ratio_matrix_kg_m03 = (
        interp_data_dict[orig_variable_name] * air_density_matrix_kg_m03
    )
    mixing_ratio_matrix_kg_m03 = numpy.reshape(
        mixing_ratio_matrix_kg_m03,
        (num_times * num_sites, num_heights)
    )
    layerwise_path_matrix_kg_m02 = rrtm_io._water_content_to_layerwise_path(
        water_content_matrix_kg_m03=mixing_ratio_matrix_kg_m03,
        heights_m_agl=new_heights_m_agl
    )
    layerwise_path_matrix_kg_m02 = numpy.reshape(
        layerwise_path_matrix_kg_m02,
        (num_times, num_sites, num_heights)
    )

    interp_data_dict[new_variable_name] = layerwise_path_matrix_kg_m02
    interp_data_dict.pop(orig_variable_name)

    return interp_data_dict


def _conserve_masses(orig_gfs_table_xarray, interp_data_dict,
                     new_heights_m_agl):
    """Conserves masses of various chemical species between the two grids.

    :param orig_gfs_table_xarray: xarray table with original GFS data (before
        interp).
    :param interp_data_dict: Dictionary, where each key is a variable name and
        the corresponding value is a 3-D numpy array (time x site x new_height).
    :param new_heights_m_agl: See documentation at top of file.
    :return: interp_data_dict: Same but with masses conserved.
    """

    vapour_pressure_matrix_pa = moisture_conv.mixing_ratio_to_vapour_pressure(
        mixing_ratios_kg_kg01=
        orig_gfs_table_xarray[VAPOUR_MIXING_RATIO_KEY].values,
        total_pressures_pascals=orig_gfs_table_xarray[PRESSURE_KEY].values
    )
    virtual_temp_matrix_kelvins = (
        moisture_conv.temperature_to_virtual_temperature(
            temperatures_kelvins=orig_gfs_table_xarray[TEMPERATURE_KEY].values,
            total_pressures_pascals=orig_gfs_table_xarray[PRESSURE_KEY].values,
            vapour_pressures_pascals=vapour_pressure_matrix_pa
        )
    )
    orig_air_dens_matrix_kg_m03 = (
        orig_gfs_table_xarray[PRESSURE_KEY].values / virtual_temp_matrix_kelvins
    )
    orig_air_dens_matrix_kg_m03 = (
        orig_air_dens_matrix_kg_m03 / DRY_AIR_GAS_CONSTANT_J_KG01_K01
    )

    vapour_pressure_matrix_pa = moisture_conv.mixing_ratio_to_vapour_pressure(
        mixing_ratios_kg_kg01=interp_data_dict[VAPOUR_MIXING_RATIO_KEY],
        total_pressures_pascals=interp_data_dict[PRESSURE_KEY]
    )
    virtual_temp_matrix_kelvins = (
        moisture_conv.temperature_to_virtual_temperature(
            temperatures_kelvins=interp_data_dict[TEMPERATURE_KEY],
            total_pressures_pascals=interp_data_dict[PRESSURE_KEY],
            vapour_pressures_pascals=vapour_pressure_matrix_pa
        )
    )
    new_air_dens_matrix_kg_m03 = (
        interp_data_dict[PRESSURE_KEY] / virtual_temp_matrix_kelvins
    )
    new_air_dens_matrix_kg_m03 = (
        new_air_dens_matrix_kg_m03 / DRY_AIR_GAS_CONSTANT_J_KG01_K01
    )

    num_times = len(orig_gfs_table_xarray.coords[TIME_DIMENSION].values)
    num_sites = len(orig_gfs_table_xarray.coords[SITE_DIMENSION].values)
    t = orig_gfs_table_xarray
    d = interp_data_dict

    for this_key in CONSERVE_MASS_KEYS:
        print('Conserving mass in each profile for {0:s}...'.format(this_key))

        for i in range(num_times):
            for j in range(num_sites):
                d[this_key][i, j, :] = (
                    interp_rap_profiles._conserve_mass_one_variable(
                        orig_conc_matrix_kg_kg01=t[this_key].values[[i], j, :],
                        orig_air_dens_matrix_kg_m03=
                        orig_air_dens_matrix_kg_m03[[i], j, :],
                        orig_heights_metres=t[HEIGHT_KEY].values[i, j, :],
                        new_conc_matrix_kg_kg01=d[this_key][[i], j, :],
                        new_air_dens_matrix_kg_m03=
                        new_air_dens_matrix_kg_m03[[i], j, :],
                        new_heights_metres=new_heights_m_agl
                    )[0, :]
                )

    interp_data_dict = d
    return interp_data_dict


def _run(input_file_name, new_heights_m_agl, output_file_name):
    """Interpolates GFS data, meant for input to RRTM, to new heights.

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

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    orig_gfs_table_xarray = xarray.open_dataset(input_file_name)
    forecast_hours = orig_gfs_table_xarray[FORECAST_HOUR_KEY].values

    # orig_gfs_table_xarray = orig_gfs_table_xarray.isel(
    #     indexers={
    #         SITE_DIMENSION: numpy.linspace(0, 4800, num=17, dtype=int)
    #     },
    #     drop=False
    # )

    # Create metadata dict for output file.
    valid_times_unix_sec = orig_gfs_table_xarray.coords[TIME_DIMENSION].values
    num_times = len(valid_times_unix_sec)
    num_sites = len(orig_gfs_table_xarray.coords[SITE_DIMENSION].values)
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

    # Create main data dict for output file.
    print('Converting trace-gas concentrations to mixing ratios...')
    orig_gfs_table_xarray = _concentrations_to_mixing_ratios(
        orig_gfs_table_xarray
    )

    print(
        'Converting layerwise ice-water paths (kg m^-2) to mixing ratios '
        '(kg kg^-1)...'
    )
    orig_gfs_table_xarray = _layerwise_path_to_mixing_ratio(
        orig_gfs_table_xarray=orig_gfs_table_xarray,
        orig_variable_name=ICE_WATER_PATH_KEY,
        new_variable_name=ICE_WATER_MIXING_RATIO_KEY
    )

    print('Converting layerwise graupel paths to mixing ratios...')
    orig_gfs_table_xarray = _layerwise_path_to_mixing_ratio(
        orig_gfs_table_xarray=orig_gfs_table_xarray,
        orig_variable_name=GRAUPEL_PATH_KEY,
        new_variable_name=GRAUPEL_MIXING_RATIO_KEY
    )

    print('Converting layerwise snow paths to mixing ratios...')
    orig_gfs_table_xarray = _layerwise_path_to_mixing_ratio(
        orig_gfs_table_xarray=orig_gfs_table_xarray,
        orig_variable_name=SNOW_PATH_KEY,
        new_variable_name=SNOW_MIXING_RATIO_KEY
    )

    print('Converting layerwise liquid-water paths to mixing ratios...')
    orig_gfs_table_xarray = _layerwise_path_to_mixing_ratio(
        orig_gfs_table_xarray=orig_gfs_table_xarray,
        orig_variable_name=LIQUID_WATER_PATH_KEY,
        new_variable_name=LIQUID_WATER_MIXING_RATIO_KEY
    )

    print('Converting layerwise rain paths to mixing ratios...')
    orig_gfs_table_xarray = _layerwise_path_to_mixing_ratio(
        orig_gfs_table_xarray=orig_gfs_table_xarray,
        orig_variable_name=RAIN_WATER_PATH_KEY,
        new_variable_name=RAIN_WATER_MIXING_RATIO_KEY
    )

    new_data_dict = dict()
    for this_key in SCALAR_KEYS:
        new_data_dict[this_key] = (
            orig_gfs_table_xarray[this_key].dims,
            orig_gfs_table_xarray[this_key].values
        )

    new_data_dict[HEIGHT_KEY] = (
        (HEIGHT_DIMENSION,), new_heights_m_agl
    )

    interp_data_dict = dict()
    for this_key in INTERP_KEYS:
        if this_key == PRESSURE_AT_EDGE_KEY:
            interp_data_dict[this_key] = numpy.full(
                (num_times, num_sites, num_heights_new + 1), numpy.nan
            )
        else:
            interp_data_dict[this_key] = numpy.full(
                (num_times, num_sites, num_heights_new), numpy.nan
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
                    forecast_hours[i], j + 1, num_sites
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
            num_sites, forecast_hours[i]
        ))
        print(SEPARATOR_STRING)

    interp_data_dict = _conserve_masses(
        orig_gfs_table_xarray=orig_gfs_table_xarray,
        interp_data_dict=interp_data_dict,
        new_heights_m_agl=new_heights_m_agl
    )

    print('Converting trace-gas mixing ratios to concentrations...')
    interp_data_dict = _mixing_ratios_to_concentrations(interp_data_dict)

    print(
        'Converting ice-water mixing ratios (kg kg^-1) to layerwise paths '
        '(kg m^-2)...'
    )
    interp_data_dict = _mixing_ratio_to_layerwise_path(
        interp_data_dict=interp_data_dict,
        new_heights_m_agl=new_heights_m_agl,
        orig_variable_name=ICE_WATER_MIXING_RATIO_KEY,
        new_variable_name=ICE_WATER_PATH_KEY
    )

    print('Converting graupel mixing ratios to layerwise paths...')
    interp_data_dict = _mixing_ratio_to_layerwise_path(
        interp_data_dict=interp_data_dict,
        new_heights_m_agl=new_heights_m_agl,
        orig_variable_name=GRAUPEL_MIXING_RATIO_KEY,
        new_variable_name=GRAUPEL_PATH_KEY
    )

    print('Converting snow mixing ratios to layerwise paths...')
    interp_data_dict = _mixing_ratio_to_layerwise_path(
        interp_data_dict=interp_data_dict,
        new_heights_m_agl=new_heights_m_agl,
        orig_variable_name=SNOW_MIXING_RATIO_KEY,
        new_variable_name=SNOW_PATH_KEY
    )

    print('Converting cloud-water mixing ratios to layerwise paths...')
    interp_data_dict = _mixing_ratio_to_layerwise_path(
        interp_data_dict=interp_data_dict,
        new_heights_m_agl=new_heights_m_agl,
        orig_variable_name=LIQUID_WATER_MIXING_RATIO_KEY,
        new_variable_name=LIQUID_WATER_PATH_KEY
    )

    print('Converting rain mixing ratios to layerwise paths...')
    interp_data_dict = _mixing_ratio_to_layerwise_path(
        interp_data_dict=interp_data_dict,
        new_heights_m_agl=new_heights_m_agl,
        orig_variable_name=RAIN_WATER_MIXING_RATIO_KEY,
        new_variable_name=RAIN_WATER_PATH_KEY
    )

    # Create xarray table with interpolated data.
    for this_key in interp_data_dict:
        if this_key == PRESSURE_AT_EDGE_KEY:
            these_dim = (
                TIME_DIMENSION, SITE_DIMENSION, HEIGHT_AT_EDGE_DIMENSION
            )
        else:
            these_dim = (TIME_DIMENSION, SITE_DIMENSION, HEIGHT_DIMENSION)

        new_data_dict[this_key] = (these_dim, interp_data_dict[this_key])

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
