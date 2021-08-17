"""Handles concentration profiles for trace gases."""

import os
import numpy
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))

ORIG_HEIGHT_KEY = 'height'

HEIGHTS_KEY = 'heights_m_asl'
STANDARD_ATMOSPHERES_KEY = 'standard_atmo_enums'
O2_MIXING_RATIOS_KEY = 'o2_mixing_ratio_matrix_kg_kg01'
CO2_MIXING_RATIOS_KEY = 'co2_mixing_ratio_matrix_kg_kg01'
CH4_MIXING_RATIOS_KEY = 'ch4_mixing_ratio_matrix_kg_kg01'
N2O_MIXING_RATIOS_KEY = 'n2o_mixing_ratio_matrix_kg_kg01'

KM_TO_METRES = 1e3
MOLAR_MASS_DRY_AIR_GRAMS_MOL01 = 28.97
MOLAR_MASS_O2_GRAMS_MOL01 = 31.9988
MOLAR_MASS_CO2_GRAMS_MOL01 = 44.01
MOLAR_MASS_CH4_GRAMS_MOL01 = 16.04
MOLAR_MASS_N2O_GRAMS_MOL01 = 44.013


def read_profiles(netcdf_file_name=None):
    """Reads mixing-ratio profiles for trace gases.

    H = number of heights
    A = number of standard atmospheres

    :param netcdf_file_name: Path to input file.  If None, will use default
        location.
    :return: mixing_ratio_dict: Dictionary with the following keys.
    mixing_ratio_dict['heights_m_asl']: length-H numpy array of heights
        (metres above sea level).
    mixing_ratio_dict['standard_atmo_enums']: length-A numpy array of
        standard-atmosphere IDs, each accepted by
        `example_utils.check_standard_atmo_type`.
    mixing_ratio_dict['o2_mixing_ratio_matrix_kg_kg01']: A-by-H numpy array
        with mixing ratios of O2.
    mixing_ratio_dict['co2_mixing_ratio_matrix_kg_kg01']: A-by-H numpy array
        with mixing ratios of CO2.
    mixing_ratio_dict['ch4_mixing_ratio_matrix_kg_kg01']: A-by-H numpy array
        with mixing ratios of CH4.
    mixing_ratio_dict['n2o_mixing_ratio_matrix_kg_kg01']: A-by-H numpy array
        with mixing ratios of N2O.
    """

    if netcdf_file_name is None:
        netcdf_file_name = '{0:s}/trace_gases.nc'.format(THIS_DIRECTORY_NAME)

    print('Reading trace-gas profiles from: "{0:s}"...'.format(
        netcdf_file_name
    ))
    dataset_object = netCDF4.Dataset(netcdf_file_name)

    o2_mixing_ratio_matrix_kg_kg01 = numpy.transpose(
        dataset_object.variables['o2'][:] *
        MOLAR_MASS_O2_GRAMS_MOL01 / MOLAR_MASS_DRY_AIR_GRAMS_MOL01
    )
    co2_mixing_ratio_matrix_kg_kg01 = numpy.transpose(
        dataset_object.variables['co2'][:] *
        MOLAR_MASS_CO2_GRAMS_MOL01 / MOLAR_MASS_DRY_AIR_GRAMS_MOL01
    )
    ch4_mixing_ratio_matrix_kg_kg01 = numpy.transpose(
        dataset_object.variables['ch4'][:] *
        MOLAR_MASS_CH4_GRAMS_MOL01 / MOLAR_MASS_DRY_AIR_GRAMS_MOL01
    )
    n2o_mixing_ratio_matrix_kg_kg01 = numpy.transpose(
        dataset_object.variables['n2o'][:] *
        MOLAR_MASS_N2O_GRAMS_MOL01 / MOLAR_MASS_DRY_AIR_GRAMS_MOL01
    )

    heights_m_asl = KM_TO_METRES * dataset_object.variables[ORIG_HEIGHT_KEY][:]
    num_standard_atmospheres = n2o_mixing_ratio_matrix_kg_kg01.shape[1]

    return {
        HEIGHTS_KEY: heights_m_asl.filled(0.),
        STANDARD_ATMOSPHERES_KEY: numpy.linspace(
            1, num_standard_atmospheres, num=num_standard_atmospheres, dtype=int
        ),
        O2_MIXING_RATIOS_KEY: o2_mixing_ratio_matrix_kg_kg01.filled(0.),
        CO2_MIXING_RATIOS_KEY: co2_mixing_ratio_matrix_kg_kg01.filled(0.),
        CH4_MIXING_RATIOS_KEY: ch4_mixing_ratio_matrix_kg_kg01.filled(0.),
        N2O_MIXING_RATIOS_KEY: n2o_mixing_ratio_matrix_kg_kg01.filled(0.)
    }
