"""Unit tests for rrtm_io.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import moisture_conversions as moisture_conv
from ml4rt.io import rrtm_io
from ml4rt.utils import example_utils
from ml4rt.utils import example_utils_test

TOLERANCE = 1e-6

# The following constants are used to test _specific_to_relative_humidity.
HUMIDITY_MATRIX_KG_KG01 = 0.001 * numpy.array([
    [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7],
    [2, 4, 6, 8, 10, 12, 14, 2, 4, 6, 8, 10, 12, 14],
    [3, 6, 9, 12, 15, 18, 21, 3, 6, 9, 12, 15, 18, 21]
], dtype=float)

TEMPERATURE_MATRIX_KELVINS = 273.15 + numpy.array([
    [10, 11, 12, 13, 14, 15, 16, 10, 11, 12, 13, 14, 15, 16],
    [20, 21, 22, 23, 24, 25, 26, 20, 21, 22, 23, 24, 25, 26],
    [30, 31, 32, 33, 34, 35, 36, 30, 31, 32, 33, 34, 35, 36]
], dtype=float)

PRESSURE_MATRIX_PASCALS = numpy.full(TEMPERATURE_MATRIX_KELVINS.shape, 1e5)

CENTER_HEIGHTS_M_AGL = numpy.array([
    10, 20, 40, 60, 80, 100, 30000, 33000, 36000, 39000, 42000, 46000, 50000
], dtype=float)

DEWPOINT_MATRIX_KELVINS = moisture_conv.specific_humidity_to_dewpoint(
    specific_humidities_kg_kg01=HUMIDITY_MATRIX_KG_KG01,
    temperatures_kelvins=TEMPERATURE_MATRIX_KELVINS,
    total_pressures_pascals=PRESSURE_MATRIX_PASCALS
)

RELATIVE_HUMIDITY_MATRIX = moisture_conv.dewpoint_to_relative_humidity(
    dewpoints_kelvins=DEWPOINT_MATRIX_KELVINS,
    temperatures_kelvins=TEMPERATURE_MATRIX_KELVINS,
    total_pressures_pascals=PRESSURE_MATRIX_PASCALS
)

THESE_VECTOR_PREDICTOR_NAMES = [
    example_utils.SPECIFIC_HUMIDITY_NAME, example_utils.TEMPERATURE_NAME,
    example_utils.PRESSURE_NAME
]
THIS_VECTOR_PREDICTOR_MATRIX = numpy.stack((
    HUMIDITY_MATRIX_KG_KG01, TEMPERATURE_MATRIX_KELVINS, PRESSURE_MATRIX_PASCALS
), axis=-1)

EXAMPLE_DICT_SANS_RH = {
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: THESE_VECTOR_PREDICTOR_NAMES,
    example_utils.HEIGHTS_KEY: CENTER_HEIGHTS_M_AGL
}

THESE_VECTOR_PREDICTOR_NAMES = [
    example_utils.SPECIFIC_HUMIDITY_NAME, example_utils.TEMPERATURE_NAME,
    example_utils.PRESSURE_NAME, example_utils.RELATIVE_HUMIDITY_NAME
]
THIS_VECTOR_PREDICTOR_MATRIX = numpy.stack((
    HUMIDITY_MATRIX_KG_KG01, TEMPERATURE_MATRIX_KELVINS,
    PRESSURE_MATRIX_PASCALS, RELATIVE_HUMIDITY_MATRIX
), axis=-1)

EXAMPLE_DICT_WITH_RH = {
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: THESE_VECTOR_PREDICTOR_NAMES,
    example_utils.HEIGHTS_KEY: CENTER_HEIGHTS_M_AGL
}

# The following constants are used to test _layerwise_water_path_to_content,
# _water_content_to_layerwise_path, and _get_water_path_profiles.
EDGE_HEIGHTS_M_AGL = numpy.array([
    5, 15, 30, 50, 70, 90, 15050, 31500, 34500, 37500, 40500, 44000, 48000,
    52000
], dtype=float)

GRID_CELL_WIDTHS_METRES = numpy.array([
    10, 15, 20, 20, 20, 14960, 16450, 3000, 3000, 3000, 3500, 4000, 4000
], dtype=float)

LAYERWISE_PATH_MATRIX_KG_M02 = numpy.array([
    [1, 1, 1, 1, 1, 1000, 1000, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 1000, 1000, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 1000, 1000, 3, 3, 3, 3, 3, 3]
], dtype=float)

WATER_CONTENT_MATRIX_KG_M03 = numpy.array([
    [0.1, 1. / 15, 0.05, 0.05, 0.05, 1. / 14.96, 1. / 16.45,
     1. / 3000, 1. / 3000, 1. / 3000, 1. / 3500, 1. / 4000, 1. / 4000],
    [0.2, 2. / 15, 0.1, 0.1, 0.1, 1. / 14.96, 1. / 16.45,
     2. / 3000, 2. / 3000, 2. / 3000, 2. / 3500, 2. / 4000, 2. / 4000],
    [0.3, 3. / 15, 0.15, 0.15, 0.15, 1. / 14.96, 1. / 16.45,
     3. / 3000, 3. / 3000, 3. / 3000, 3. / 3500, 3. / 4000, 3. / 4000]
])

# VAPOUR_CONTENT_MATRIX_KG_M03 = numpy.array([
#     [1.22963760, 2.44913268, 3.65859468, 4.85813144, 6.04784935, 7.22785326,
#      8.39824657, 1.22963760, 2.44913268, 3.65859468, 4.85813144, 6.04784935,
#      7.227853260, 8.39824657],
#     [2.37394186, 4.72600480, 7.05643206, 9.36546336, 11.65333510, 13.92028056,
#      16.16652968, 2.37394186, 4.72600480, 7.05643206, 9.36546336, 11.65333510,
#      13.92028056, 16.16652968],
#     [3.44135997, 6.84762756, 10.21921533, 13.55652996, 16.85997240, 20.12993802,
#      23.36681634, 3.44135997, 6.84762756, 10.21921533, 13.55652996, 16.85997240,
#      20.12993802, 23.36681634]
# ])

DOWNWARD_PATH_MATRIX_KG_M02 = numpy.array([
    [2011, 2010, 2009, 2008, 2007, 2006, 1006, 6, 5, 4, 3, 2, 1],
    [2022, 2020, 2018, 2016, 2014, 2012, 1012, 12, 10, 8, 6, 4, 2],
    [2033, 2030, 2027, 2024, 2021, 2018, 1018, 18, 15, 12, 9, 6, 3]
], dtype=float)

UPWARD_PATH_MATRIX_KG_M02 = numpy.array([
    [1, 2, 3, 4, 5, 1005, 2005, 2006, 2007, 2008, 2009, 2010, 2011],
    [2, 4, 6, 8, 10, 1010, 2010, 2012, 2014, 2016, 2018, 2020, 2022],
    [3, 6, 9, 12, 15, 1015, 2015, 2018, 2021, 2024, 2027, 2030, 2033]
], dtype=float)

ORIG_VECTOR_PREDICTOR_NAMES = [
    example_utils.LIQUID_WATER_CONTENT_NAME,
    example_utils.ICE_WATER_CONTENT_NAME
]
ORIG_VECTOR_PREDICTOR_MATRIX = numpy.stack(
    (WATER_CONTENT_MATRIX_KG_M03, WATER_CONTENT_MATRIX_KG_M03 / 1000), axis=-1
)
VALID_TIMES_UNIX_SEC = numpy.array([300, 600, 900], dtype=int)

EXAMPLE_DICT_WITHOUT_PATHS = {
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: ORIG_VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: ORIG_VECTOR_PREDICTOR_MATRIX,
    example_utils.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC,
    example_utils.HEIGHTS_KEY: CENTER_HEIGHTS_M_AGL
}

THESE_VECTOR_PREDICTOR_NAMES = ORIG_VECTOR_PREDICTOR_NAMES + [
    example_utils.LIQUID_WATER_PATH_NAME, example_utils.ICE_WATER_PATH_NAME
]
NEW_PREDICTOR_MATRIX = numpy.stack(
    (DOWNWARD_PATH_MATRIX_KG_M02, DOWNWARD_PATH_MATRIX_KG_M02 / 1000), axis=-1
)
THIS_VECTOR_PREDICTOR_MATRIX = numpy.concatenate(
    (ORIG_VECTOR_PREDICTOR_MATRIX, NEW_PREDICTOR_MATRIX), axis=-1
)

EXAMPLE_DICT_WITH_DOWNWARD_PATHS = {
    example_utils.VECTOR_PREDICTOR_NAMES_KEY:
        copy.deepcopy(THESE_VECTOR_PREDICTOR_NAMES),
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX + 0.,
    example_utils.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC,
    example_utils.HEIGHTS_KEY: CENTER_HEIGHTS_M_AGL
}

THESE_VECTOR_PREDICTOR_NAMES = ORIG_VECTOR_PREDICTOR_NAMES + [
    example_utils.UPWARD_LIQUID_WATER_PATH_NAME,
    example_utils.UPWARD_ICE_WATER_PATH_NAME
]
NEW_PREDICTOR_MATRIX = numpy.stack(
    (UPWARD_PATH_MATRIX_KG_M02, UPWARD_PATH_MATRIX_KG_M02 / 1000), axis=-1
)
THIS_VECTOR_PREDICTOR_MATRIX = numpy.concatenate(
    (ORIG_VECTOR_PREDICTOR_MATRIX, NEW_PREDICTOR_MATRIX), axis=-1
)

EXAMPLE_DICT_WITH_UPWARD_PATHS = {
    example_utils.VECTOR_PREDICTOR_NAMES_KEY:
        copy.deepcopy(THESE_VECTOR_PREDICTOR_NAMES),
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX + 0.,
    example_utils.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC,
    example_utils.HEIGHTS_KEY: CENTER_HEIGHTS_M_AGL
}

# The following constants are used to test find_file and file_name_to_year.
RRTM_DIRECTORY_NAME = 'foo'
YEAR = 2018
RRTM_FILE_NAME = 'foo/rrtm_output_2018.nc'

# The following constants are used to test find_many_files.
FIRST_FILE_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '1999-12-31-235959', '%Y-%m-%d-%H%M%S'
)
LAST_FILE_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2005-01-01-000000', '%Y-%m-%d-%H%M%S'
)
RRTM_FILE_NAMES = [
    'foo/rrtm_output_1999.nc', 'foo/rrtm_output_2000.nc',
    'foo/rrtm_output_2001.nc', 'foo/rrtm_output_2002.nc',
    'foo/rrtm_output_2003.nc', 'foo/rrtm_output_2004.nc',
    'foo/rrtm_output_2005.nc'
]


class RrtmIoTests(unittest.TestCase):
    """Each method is a unit test for rrtm_io.py."""

    def test_specific_to_relative_humidity(self):
        """Ensures correct output from _specific_to_relative_humidity."""

        this_example_dict = rrtm_io._specific_to_relative_humidity(
            copy.deepcopy(EXAMPLE_DICT_SANS_RH)
        )

        self.assertTrue(example_utils_test._compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_WITH_RH
        ))

    def test_layerwise_water_path_to_content(self):
        """Ensures correct output from _layerwise_water_path_to_content."""

        this_content_matrix_kg_m03 = (
            rrtm_io._layerwise_water_path_to_content(
                layerwise_path_matrix_kg_m02=LAYERWISE_PATH_MATRIX_KG_M02,
                heights_m_agl=CENTER_HEIGHTS_M_AGL
            )
        )

        self.assertTrue(numpy.allclose(
            this_content_matrix_kg_m03, WATER_CONTENT_MATRIX_KG_M03,
            atol=TOLERANCE
        ))

    def test_water_content_to_layerwise_path(self):
        """Ensures correct output from _water_content_to_layerwise_path."""

        this_path_matrix_kg_m02 = rrtm_io._water_content_to_layerwise_path(
            water_content_matrix_kg_m03=WATER_CONTENT_MATRIX_KG_M03,
            heights_m_agl=CENTER_HEIGHTS_M_AGL
        )

        self.assertTrue(numpy.allclose(
            this_path_matrix_kg_m02, LAYERWISE_PATH_MATRIX_KG_M02,
            atol=TOLERANCE
        ))

    def test_get_water_path_profiles_downward(self):
        """Ensures correct output from _get_water_path_profiles.

        In this case, paths are integrated downward from top of atmosphere.
        """

        this_example_dict = rrtm_io._get_water_path_profiles(
            example_dict=copy.deepcopy(EXAMPLE_DICT_WITHOUT_PATHS),
            get_lwp=True, get_iwp=True, get_wvp=False, integrate_upward=False
        )

        self.assertTrue(example_utils_test._compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_WITH_DOWNWARD_PATHS
        ))

    def test_get_water_path_profiles_upward(self):
        """Ensures correct output from _get_water_path_profiles.

        In this case, paths are integrated upward from surface.
        """

        this_example_dict = rrtm_io._get_water_path_profiles(
            example_dict=copy.deepcopy(EXAMPLE_DICT_WITHOUT_PATHS),
            get_lwp=True, get_iwp=True, get_wvp=False, integrate_upward=True
        )

        self.assertTrue(example_utils_test._compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_WITH_UPWARD_PATHS
        ))

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = rrtm_io.find_file(
            directory_name=RRTM_DIRECTORY_NAME, year=YEAR,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == RRTM_FILE_NAME)

    def test_file_name_to_year(self):
        """Ensures correct output from file_name_to_year."""

        this_year = rrtm_io.file_name_to_year(RRTM_FILE_NAME)
        self.assertTrue(this_year == YEAR)

    def test_find_many_files(self):
        """Ensures correct output from find_many_files."""

        these_file_names = rrtm_io.find_many_files(
            directory_name=RRTM_DIRECTORY_NAME,
            first_time_unix_sec=FIRST_FILE_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_FILE_TIME_UNIX_SEC,
            raise_error_if_any_missing=False,
            raise_error_if_all_missing=False, test_mode=True
        )

        self.assertTrue(these_file_names == RRTM_FILE_NAMES)


if __name__ == '__main__':
    unittest.main()
