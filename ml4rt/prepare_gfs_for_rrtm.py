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

PASCALS_TO_MB = 0.01
KG_TO_GRAMS = 1e3
UNITLESS_TO_PERCENT = 100.
PERCENT_TO_UNITLESS = 0.01
DUMMY_DOWN_SURFACE_FLUX_W_M02 = 1000.
DRY_AIR_GAS_CONSTANT_J_KG01_K01 = 287.04

SITE_DIMENSION_ORIG = 'site_index'
FORECAST_HOUR_DIMENSION_ORIG = 'forecast_hour'
HEIGHT_DIMENSION_ORIG = 'pfull'

SITE_DIMENSION = 'sites'
FORECAST_HOUR_DIMENSION = 'forecast_hour'
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
    PRESSURE_AT_EDGE_KEY_ORIG_PASCALS: 'p0_edge'
}

INPUT_FILE_ARG_NAME = 'input_file_name'
NEW_HEIGHTS_ARG_NAME = 'new_heights_m_agl'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file (created by subset_gfs_from_jebb.py), containing GFS '
    'data for one init time and one forecast hour.'
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
    orig_table_xarray = xarray.open_dataset(input_file_name)

    this_matrix = moisture_conv.specific_humidity_to_mixing_ratio(
        orig_table_xarray[SPECIFIC_HUMIDITY_KEY_ORIG_KG_KG01].values
    )
    orig_table_xarray = orig_table_xarray.assign({
        VAPOUR_MIXR_KEY_ORIG_KG_KG01: (
            orig_table_xarray[SPECIFIC_HUMIDITY_KEY_ORIG_KG_KG01].dims,
            this_matrix
        )
    })

    # Create metadata dict for output file.
    forecast_hours = (
        orig_table_xarray.coords[FORECAST_HOUR_DIMENSION_ORIG].values
    )
    num_forecast_hours = len(forecast_hours)
    num_sites = len(orig_table_xarray.coords[SITE_DIMENSION_ORIG].values)
    num_heights_new = len(new_heights_m_agl)

    new_metadata_dict = {
        FORECAST_HOUR_DIMENSION: forecast_hours,
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

    # Start main data dict for output file.
    latitudes_deg_n = orig_table_xarray[LATITUDE_KEY_ORIG_DEG_N].values
    longitudes_deg_e = orig_table_xarray[LONGITUDE_KEY_ORIG_DEG_E].values
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
        FORECAST_HOUR_KEY: ((FORECAST_HOUR_DIMENSION,), forecast_hours),
        HEIGHT_KEY_M_AGL: ((HEIGHT_DIMENSION,), new_heights_m_agl)
    }

    # Interpolate data to new heights.
    interp_data_dict = dict()
    for this_key in MAIN_KEYS_ORIG + [PRESSURE_KEY_ORIG_PASCALS]:
        interp_data_dict[this_key] = numpy.full(
            (num_forecast_hours, num_sites, num_heights_new), numpy.nan
        )

    interp_data_dict[PRESSURE_AT_EDGE_KEY_ORIG_PASCALS] = numpy.full(
        (num_forecast_hours, num_sites, num_heights_new + 1), numpy.nan
    )
    new_heights_at_edges_m_agl = example_utils.get_grid_cell_edges(
        new_heights_m_agl
    )

    for i in range(num_forecast_hours):
        for j in range(num_sites):
            verbose = numpy.mod(j, 10) == 0

            if verbose:
                print((
                    'Interpolating {0:s} to new heights for forecast hour '
                    '{1:d}, site {2:d} of {3:d}...'
                ).format(
                    PRESSURE_KEY_ORIG_PASCALS, forecast_hours[i],
                    j + 1, num_sites
                ))

            orig_pressure_diffs_pa = numpy.cumsum(numpy.flip(
                orig_table_xarray[
                    DELTA_PRESSURE_KEY_ORIG_PASCALS
                ].values[i, :, j]
            ))
            orig_pressures_pa = (
                orig_table_xarray[
                    SURFACE_PRESSURE_KEY_ORIG_PASCALS
                ].values[i, j]
                - orig_pressure_diffs_pa
            )
            orig_heights_m_agl = numpy.cumsum(numpy.flip(
                orig_table_xarray[DELTA_HEIGHT_KEY_ORIG_METRES].values[i, :, j]
                * -1
            ))

            interp_object = interp1d(
                x=orig_heights_m_agl, y=numpy.log(orig_pressures_pa),
                kind='linear', bounds_error=False, assume_sorted=True,
                fill_value='extrapolate'
            )
            interp_data_dict[PRESSURE_KEY_ORIG_PASCALS][i, j, :] = numpy.exp(
                interp_object(new_heights_m_agl)
            )

            if verbose:
                print((
                    'Interpolating {0:s} to new heights for forecast hour '
                    '{1:d}, site {2:d} of {3:d}...'
                ).format(
                    PRESSURE_AT_EDGE_KEY_ORIG_PASCALS, forecast_hours[i],
                    j + 1, num_sites
                ))

            interp_data_dict[PRESSURE_AT_EDGE_KEY_ORIG_PASCALS][i, j, :] = (
                numpy.exp(interp_object(new_heights_at_edges_m_agl))
            )

            for this_key in MAIN_KEYS_ORIG:
                if verbose:
                    print((
                        'Interpolating {0:s} to new heights for forecast hour '
                        '{1:d}, site {2:d} of {3:d}...'
                    ).format(
                        this_key, forecast_hours[i], j + 1, num_sites
                    ))

                if this_key in CONSERVE_JUMP_KEYS_ORIG:
                    orig_values = numpy.flip(
                        orig_table_xarray[this_key].values[i, :, j]
                    )
                    orig_data_matrix = numpy.expand_dims(orig_values, axis=0)

                    interp_data_dict[this_key][i, j, :] = (
                        interp_rap_profiles._interp_and_conserve_jumps(
                            orig_data_matrix=orig_data_matrix,
                            orig_heights_metres=orig_heights_m_agl,
                            new_heights_metres=new_heights_m_agl
                        )
                    )

                    continue

                orig_values = numpy.flip(
                    orig_table_xarray[this_key].values[i, :, j]
                )
                log_offset = 1. + -1 * numpy.min(orig_values)
                assert not numpy.isnan(log_offset)

                interp_object = interp1d(
                    x=orig_heights_m_agl, y=numpy.log(log_offset + orig_values),
                    kind='linear', bounds_error=False, assume_sorted=True,
                    fill_value='extrapolate'
                )
                interp_data_dict[this_key][i, j, :] = (
                    numpy.exp(interp_object(new_heights_m_agl)) - log_offset
                )

        print((
            'Have interpolated data to new heights for all {0:d} sites at '
            'forecast hour {1:d}!'
        ).format(
            num_sites, forecast_hours[i]
        ))
        print(SEPARATOR_STRING)

    # TODO(thunderhoser): Will likely need to convert other vars in same way.
    print('Converting ice-water mixing ratios (kg/kg) to paths (kg/m^2)...')
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
        (num_forecast_hours * num_sites, num_heights_new)
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
        (num_forecast_hours, num_sites, num_heights_new)
    )
    interp_data_dict[CLOUD_ICE_MIXR_KEY_ORIG_KG_KG01] = (
        ice_water_path_matrix_kg_m02
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
        PERCENT_TO_UNITLESS * orig_table_xarray[ALBEDO_KEY_ORIG_PERCENT].values
    )
    up_flux_matrix_w_m02 = albedo_matrix * DUMMY_DOWN_SURFACE_FLUX_W_M02
    down_flux_matrix_w_m02 = numpy.full(
        up_flux_matrix_w_m02.shape, DUMMY_DOWN_SURFACE_FLUX_W_M02
    )

    for this_key_orig in interp_data_dict:
        if this_key_orig == PRESSURE_AT_EDGE_KEY_ORIG_PASCALS:
            these_dim = (
                FORECAST_HOUR_DIMENSION, SITE_DIMENSION,
                HEIGHT_AT_EDGE_DIMENSION
            )
        else:
            these_dim = (
                FORECAST_HOUR_DIMENSION, SITE_DIMENSION, HEIGHT_DIMENSION
            )

        this_key = ORIG_TO_NEW_KEY_DICT[this_key_orig]
        new_data_dict[this_key] = (these_dim, interp_data_dict[this_key_orig])

    these_dim = (FORECAST_HOUR_DIMENSION, SITE_DIMENSION)
    new_data_dict.update({
        UP_SURFACE_FLUX_KEY_W_M02: (these_dim, up_flux_matrix_w_m02),
        DOWN_SURFACE_FLUX_KEY_W_M02: (these_dim, down_flux_matrix_w_m02)
    })

    new_table_xarray = xarray.Dataset(
        data_vars=new_data_dict, coords=new_metadata_dict
    )

    for this_key in new_table_xarray.variables:
        if this_key == SITE_NAME_KEY:
            continue

        error_checking.assert_is_numpy_array_without_nan(
            new_table_xarray[this_key].values
        )

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    new_table_xarray.to_netcdf(
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
